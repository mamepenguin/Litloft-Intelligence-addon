"""AI summaries endpoints.

Provides CRUD-like endpoints for LLM-generated file summaries:

- GET /files/{file_id}/summary             — fetch current summary
- POST /files/{file_id}/summary/regenerate — delete + re-enqueue
- POST /files/{file_id}/summary/edit       — user overwrite (keeps AI snapshot)
- POST /files/{file_id}/summary/revert     — restore AI snapshot
- POST /batch/summaries                    — queue a batch of files

Hide is handled entirely client-side (session-scoped) — the server
does not persist a hidden state so users always have a way to bring
a summary back by reloading the page.

Access control is enforced by the Generic Addon Proxy in the host
via pre_check rules; this router assumes the caller has permission
for the file_ids it receives.
"""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_search_db
from app.dependencies import get_llm_client, get_summaries_worker
from app.drive_context import assert_file_in_drive, require_drive
from app.models import IndexedFile
from app.schemas import (
    BatchSummariesRequest,
    BatchSummariesResponse,
    MessageResponse,
    SummaryEditRequest,
    SummaryResponse,
)
from app.workers.summaries import classify_missing_reason


def _require_file_in_drive(file_id: str, drive: str) -> None:
    """Raise 404 unless ``file_id`` belongs to ``drive``."""
    with get_search_db() as session:
        row = (
            session.query(IndexedFile.drive)
            .filter(
                IndexedFile.file_id == file_id,
                IndexedFile.active.is_(True),
            )
            .first()
        )
    if row is None:
        raise HTTPException(status_code=404, detail="File not indexed")
    assert_file_in_drive(row.drive, drive)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["summaries"])


def _require_llm_enabled() -> None:
    """Raise 400 if summaries feature or LLM client is not available.

    This is the ONLY gate between a router request and the summaries
    worker queue. main.py only starts the worker's run() loop when
    `features.summaries != "false" AND llm_client.enabled`; if a future
    code path bypasses this guard, enqueued items would never drain.
    Keep this check aligned with the worker-start condition in main.py.
    """
    if settings.features.summaries == "false":
        raise HTTPException(
            status_code=400, detail="Summaries feature is disabled"
        )
    if not get_llm_client().enabled:
        raise HTTPException(status_code=400, detail="LLM is not enabled")


@router.get("/files/{file_id}/summary", response_model=SummaryResponse)
async def get_summary(
    file_id: str,
    drive: str = Depends(require_drive),
) -> SummaryResponse:
    """Fetch the stored summary for a file.

    Returns available=False when:
    - the summaries feature is disabled
    - no summary has been generated yet
    - the underlying file has been soft-deleted (active=False)

    The soft-delete check matches the host's drive_access filter so a
    bypass of the proxy cannot resurrect summaries of deleted files.
    """
    if settings.features.summaries == "false":
        return SummaryResponse(available=False)

    _require_file_in_drive(file_id, drive)

    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT s.file_id, s.short_summary, s.long_summary, s.model, "
                "s.context_type, s.was_truncated, s.status, s.created_at, "
                "s.edited_at, s.short_original, s.long_original "
                "FROM file_summaries s "
                "INNER JOIN indexed_files f ON s.file_id = f.file_id "
                "WHERE s.file_id = :fid AND f.active = 1"
            ),
            {"fid": file_id},
        ).fetchone()

    if row is None:
        # Classify *why* there's no summary so the frontend can render
        # a useful state (button / "insufficient content" / "generate")
        # instead of always offering a generate button that would
        # silently skip when the file lacks usable content.
        return SummaryResponse(
            available=False,
            reason=classify_missing_reason(file_id),
        )

    return SummaryResponse(
        available=True,
        file_id=row[0],
        short_summary=row[1],
        long_summary=row[2],
        model=row[3],
        context_type=row[4],
        was_truncated=bool(row[5]),
        status=row[6],
        created_at=row[7],
        edited_at=row[8],
        has_original=row[9] is not None and row[10] is not None,
    )


@router.post(
    "/files/{file_id}/summary/regenerate", response_model=MessageResponse
)
async def regenerate_summary(
    file_id: str,
    drive: str = Depends(require_drive),
) -> MessageResponse:
    """Delete the existing summary and re-queue generation.

    Rejects the request upfront (400) when the file cannot be summarized
    — either the type is unsupported or the context is below threshold.
    This avoids a 40-second frontend polling timeout when the request
    would have been silently skipped by the worker anyway.
    """
    _require_llm_enabled()
    _require_file_in_drive(file_id, drive)

    # Pre-flight: surface skip reasons as 400 so the frontend can show
    # them immediately instead of waiting for a worker that will no-op.
    reason = classify_missing_reason(file_id)
    if reason in ("unsupported_type", "insufficient_content", "file_not_found"):
        raise HTTPException(status_code=400, detail=reason)

    summaries_worker = get_summaries_worker()

    with get_search_db() as session:
        session.execute(
            sql_text("DELETE FROM file_summaries WHERE file_id = :fid"),
            {"fid": file_id},
        )

    await summaries_worker.enqueue(file_id)
    return MessageResponse(
        status="accepted", message="Regeneration queued"
    )


def _row_to_response(row: object) -> SummaryResponse:
    """Build a ``SummaryResponse`` from a file_summaries row.

    Expected column order:
    ``(file_id, short_summary, long_summary, model, context_type,
      was_truncated, status, created_at, edited_at,
      short_original, long_original)``.

    Shared between edit / revert so their responses stay consistent
    with GET without re-reading through the feature-flag-gated GET
    path (edits are allowed while ``features.summaries = "false"``,
    but GET hides summaries in that state).
    """
    return SummaryResponse(
        available=True,
        file_id=row[0],
        short_summary=row[1],
        long_summary=row[2],
        model=row[3],
        context_type=row[4],
        was_truncated=bool(row[5]),
        status=row[6],
        created_at=row[7],
        edited_at=row[8],
        has_original=row[9] is not None and row[10] is not None,
    )


def _fetch_summary_row(session: object, file_id: str) -> object | None:
    """Load the file_summaries row in the column order expected by ``_row_to_response``."""
    return session.execute(
        sql_text(
            "SELECT file_id, short_summary, long_summary, model, context_type, "
            "was_truncated, status, created_at, edited_at, "
            "short_original, long_original "
            "FROM file_summaries WHERE file_id = :fid"
        ),
        {"fid": file_id},
    ).fetchone()


@router.post("/files/{file_id}/summary/edit", response_model=SummaryResponse)
async def edit_summary(
    file_id: str,
    body: SummaryEditRequest,
    drive: str = Depends(require_drive),
) -> SummaryResponse:
    """Overwrite the stored summary with user-edited text.

    The first edit snapshots the current AI output into ``short_original`` /
    ``long_original`` so the user can revert. Subsequent edits do not
    overwrite the snapshot — original always refers to the last AI version.

    ``features.summaries = "false"`` still allows edits: disabling the
    feature only gates new generation, not curation of existing records.
    ``status = "hidden"`` is preserved (editing does not un-hide).
    """
    _require_file_in_drive(file_id, drive)

    now = datetime.now(UTC).isoformat()

    with get_search_db() as session:
        existing = session.execute(
            sql_text(
                "SELECT short_summary, long_summary, short_original, long_original "
                "FROM file_summaries WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()

        if existing is None:
            raise HTTPException(status_code=404, detail="No summary to edit")

        # Snapshot the AI output only on the first edit. On re-edit, keep
        # the snapshot pinned to the original AI version so revert always
        # reaches "the generation this row started with" rather than a
        # previous user edit.
        short_original = existing[2] if existing[2] is not None else existing[0]
        long_original = existing[3] if existing[3] is not None else existing[1]

        session.execute(
            sql_text(
                "UPDATE file_summaries SET "
                "short_summary = :short, long_summary = :long, "
                "short_original = :short_original, "
                "long_original = :long_original, "
                "edited_at = :edited_at "
                "WHERE file_id = :fid"
            ),
            {
                "fid": file_id,
                "short": body.short_summary,
                "long": body.long_summary,
                "short_original": short_original,
                "long_original": long_original,
                "edited_at": now,
            },
        )

        row = _fetch_summary_row(session, file_id)

    # ``row`` is never None here — we just updated it inside the same
    # session. Build the response directly so edit works regardless of
    # ``features.summaries`` (GET's feature gate would hide it otherwise).
    assert row is not None
    return _row_to_response(row)


@router.post("/files/{file_id}/summary/revert", response_model=SummaryResponse)
async def revert_summary(
    file_id: str,
    drive: str = Depends(require_drive),
) -> SummaryResponse:
    """Restore the AI snapshot for a user-edited summary.

    Fails with 400 when no snapshot exists (never edited, or the row was
    regenerated after an edit — regenerate starts fresh with NULL originals).
    """
    _require_file_in_drive(file_id, drive)

    with get_search_db() as session:
        existing = session.execute(
            sql_text(
                "SELECT short_original, long_original "
                "FROM file_summaries WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()

        if existing is None:
            raise HTTPException(status_code=404, detail="No summary found")

        if existing[0] is None or existing[1] is None:
            raise HTTPException(
                status_code=400, detail="No AI version to revert to"
            )

        session.execute(
            sql_text(
                "UPDATE file_summaries SET "
                "short_summary = :short, long_summary = :long, "
                "short_original = NULL, long_original = NULL, "
                "edited_at = NULL "
                "WHERE file_id = :fid"
            ),
            {
                "fid": file_id,
                "short": existing[0],
                "long": existing[1],
            },
        )

        row = _fetch_summary_row(session, file_id)

    assert row is not None
    return _row_to_response(row)


@router.post("/batch/summaries", response_model=BatchSummariesResponse)
async def batch_summaries(
    body: BatchSummariesRequest,
    drive: str = Depends(require_drive),
) -> BatchSummariesResponse:
    """Queue summary generation for a batch of files in the current drive.

    Files outside the request drive are silently dropped (counted as
    skipped) so other drives' file_ids cannot be probed and an attacker
    in one drive cannot incur LLM cost for files in another.
    """
    _require_llm_enabled()

    summaries_worker = get_summaries_worker()

    in_drive: set[str] = set()
    existing: set[str] = set()
    if body.file_ids:
        with get_search_db() as session:
            in_drive = {
                row.file_id
                for row in session.query(IndexedFile.file_id)
                .filter(
                    IndexedFile.file_id.in_(body.file_ids),
                    IndexedFile.drive == drive,
                    IndexedFile.active.is_(True),
                )
                .all()
            }
            placeholders = ",".join(f":id{i}" for i in range(len(body.file_ids)))
            params = {f"id{i}": fid for i, fid in enumerate(body.file_ids)}
            for row in session.execute(
                sql_text(
                    f"SELECT file_id FROM file_summaries WHERE file_id IN ({placeholders})"
                ),
                params,
            ).fetchall():
                existing.add(row[0])

    queued = 0
    skipped = 0
    for file_id in body.file_ids:
        if file_id not in in_drive or file_id in existing:
            skipped += 1
        else:
            await summaries_worker.enqueue(file_id)
            queued += 1

    return BatchSummariesResponse(queued=queued, skipped=skipped)
