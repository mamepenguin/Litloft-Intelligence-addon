"""AI summaries endpoints.

Provides CRUD-like endpoints for LLM-generated file summaries:

- GET /files/{file_id}/summary             — fetch current summary
- POST /files/{file_id}/summary/regenerate — delete + re-enqueue
- POST /files/{file_id}/summary/edit       — user overwrite (keeps AI snapshot)
- POST /files/{file_id}/summary/revert     — restore AI snapshot
- POST /batch/summaries                    — queue a batch of files
- GET /files/{file_id}/summary/detailed    — fetch long-form Markdown summary
- POST /files/{file_id}/summary/detailed   — start generation (BackgroundTasks)
- DELETE /files/{file_id}/summary/detailed — clear detailed summary
- GET /files/{file_id}/summary/detailed.md — download Markdown file

Hide is handled entirely client-side (session-scoped) — the server
does not persist a hidden state so users always have a way to bring
a summary back by reloading the page.

Access control is enforced by the Generic Addon Proxy in the host
via pre_check rules; this router assumes the caller has permission
for the file_ids it receives.
"""

import json
import logging
import os
import re
from datetime import UTC, datetime
from urllib.parse import quote as urlquote

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_search_db
from app.dependencies import get_llm_client, get_summaries_worker
from app.drive_context import assert_file_in_drive, require_drive
from app.models import IndexedFile, generate_insight_id
from app.schemas import (
    BatchSummariesRequest,
    BatchSummariesResponse,
    DetailedSummaryCitationItem,
    DetailedSummaryCitationsResponse,
    DetailedSummaryEditRequest,
    DetailedSummaryRegenerateRequest,
    DetailedSummaryResponse,
    DetailedSummaryStartResponse,
    MessageResponse,
    SummaryEditRequest,
    SummaryResponse,
)
from app.workers.summaries import (
    DETAILED_STATUS_GENERATED,
    DETAILED_STATUS_GENERATING,
    _delete_detailed_summary,
    _emit_ws_event,
    _get_detailed_summary,
    _has_detailed_summary,
    _recalculate_citations,
    classify_detailed_missing_reason,
    classify_missing_reason,
    generate_detailed_summary,
)


async def _clear_core_active_summary(file_id: str) -> None:
    """Best-effort DELETE of the core ``file_active_summaries`` row.

    Called on detailed-summary regenerate so the file detail page flips
    back to the intelligence summary view. The 404 branch is expected
    whenever the user never promoted the summary to knowledge, and the
    network branch is swallowed because the host's active-summary
    pointer is a UI convenience — a stale pointer is harmless (the
    ``active-summary-view`` slot falls back to the AI summary once the
    `.md` is also gone, and will recover on the next page load if the
    network recovers).
    """
    base = os.environ.get(
        "HOMEVAULT_INTERNAL_API_URL", "http://backend:8000/api/internal"
    )
    url = f"{base}/file_active_summary/{urlquote(file_id, safe='')}"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.delete(url)
            if resp.status_code not in (204, 404):
                logger.warning(
                    "clear_core_active_summary unexpected %s for %s: %s",
                    resp.status_code, file_id, resp.text,
                )
    except httpx.HTTPError as exc:  # noqa: BLE001
        logger.warning("clear_core_active_summary network error: %s", exc)


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


def _require_detailed_feature_enabled() -> None:
    """Raise 400 if the detailed_summaries feature flag is disabled.

    Unlike ``_require_detailed_enabled``, this does NOT require the
    LLM to be enabled — it's for operations that work on stored text
    only (section edit, revert) and don't need to call the LLM.
    """
    if settings.features.detailed_summaries == "false":
        raise HTTPException(
            status_code=400,
            detail="Detailed summaries feature is disabled",
        )


def _require_detailed_enabled() -> None:
    """Raise 400 if detailed_summaries feature or LLM is unavailable.

    Independent of ``features.summaries`` so operators can enable the
    long-form variant without the short/long pair (or vice versa).
    """
    _require_detailed_feature_enabled()
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


# ---------------------------------------------------------------------------
# Detailed (long-form Markdown) summary endpoints
# ---------------------------------------------------------------------------


def _get_indexed_file_basics(file_id: str) -> tuple[str, str] | None:
    """Return ``(drive, filename)`` for an active file, or None if absent."""
    with get_search_db() as session:
        row = (
            session.query(IndexedFile.drive, IndexedFile.filename)
            .filter(
                IndexedFile.file_id == file_id,
                IndexedFile.active.is_(True),
            )
            .first()
        )
    if row is None:
        return None
    return (row[0], row[1])


def _detailed_row_to_response(
    file_id: str, data: dict
) -> DetailedSummaryResponse:
    """Shape ``_get_detailed_summary`` output into the API response."""
    status = data["detailed_status"]
    available = status == DETAILED_STATUS_GENERATED
    return DetailedSummaryResponse(
        available=available,
        file_id=file_id,
        detailed_summary=data["detailed_summary"] if available else None,
        status=status,
        model=data["detailed_model"],
        generated_at=data["detailed_generated_at"],
        context_chars=data["detailed_context_chars"],
        was_truncated=data["detailed_was_truncated"],
        error=data["detailed_error"],
        edited_at=data.get("detailed_edited_at"),
        has_original=data.get("detailed_original") is not None,
    )


@router.get(
    "/files/{file_id}/summary/detailed",
    response_model=DetailedSummaryResponse,
)
async def get_detailed_summary_route(
    file_id: str,
    drive: str = Depends(require_drive),
) -> DetailedSummaryResponse:
    """Fetch the long-form Markdown summary and its generation status.

    Returns ``available=False`` with a ``reason`` when the feature is
    disabled or no work has been started yet. Intermediate states
    (``generating`` / ``failed``) are returned with ``available=False``
    and the ``status`` field set so the frontend can poll or surface
    the error.
    """
    if settings.features.detailed_summaries == "false":
        return DetailedSummaryResponse(available=False)

    _require_file_in_drive(file_id, drive)

    data = _get_detailed_summary(file_id)
    if data is None:
        return DetailedSummaryResponse(
            available=False,
            reason=classify_detailed_missing_reason(file_id),
        )

    return _detailed_row_to_response(file_id, data)


@router.post(
    "/files/{file_id}/summary/detailed",
    response_model=DetailedSummaryStartResponse,
)
async def start_detailed_summary(
    file_id: str,
    background_tasks: BackgroundTasks,
    drive: str = Depends(require_drive),
) -> DetailedSummaryStartResponse:
    """Kick off detailed-summary generation in the background.

    Pre-flight validation matches the short/long regenerate path:
    unsupported type / insufficient content / missing file are 400
    so the frontend can render an immediate error rather than polling
    a ``generating`` row that would never complete.

    A second request while one is in flight (``status = 'generating'``)
    returns 409. Completed / failed rows must be cleared via DELETE
    before re-generation — this mirrors the short/long regenerate
    contract (delete + re-enqueue) and avoids accidental overwrite.
    """
    _require_detailed_enabled()
    _require_file_in_drive(file_id, drive)

    reason = classify_detailed_missing_reason(file_id)
    if reason in (
        "unsupported_type", "insufficient_content", "file_not_found",
    ):
        raise HTTPException(status_code=400, detail=reason)

    if _has_detailed_summary(file_id):
        raise HTTPException(
            status_code=409,
            detail=(
                "Detailed summary already exists — "
                "DELETE first to regenerate"
            ),
        )

    background_tasks.add_task(
        generate_detailed_summary, file_id, get_llm_client()
    )
    return DetailedSummaryStartResponse(
        status="accepted", message="Detailed summary generation started"
    )


@router.delete(
    "/files/{file_id}/summary/detailed",
    response_model=MessageResponse,
)
async def delete_detailed_summary_route(
    file_id: str,
    drive: str = Depends(require_drive),
) -> MessageResponse:
    """Clear any detailed-summary state for the file.

    Used by the "regenerate" flow in the UI — the client deletes first,
    then POSTs a new generation request. Returns 404 when no row was
    touched so clients don't silently get a success for stray IDs.
    """
    _require_file_in_drive(file_id, drive)

    deleted = _delete_detailed_summary(file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No summary to delete")

    return MessageResponse(
        status="ok", message="Detailed summary cleared"
    )


# Characters considered safe in a filename when sanitising the download
# disposition. Everything else is replaced with an underscore so callers
# can't inject headers or break the Content-Disposition parser.
_FILENAME_SANITIZER = re.compile(r"[^A-Za-z0-9._\- ]")


def _sanitize_ascii_filename(name: str) -> str:
    """Strip non-ASCII chars so legacy clients get a usable filename."""
    cleaned = _FILENAME_SANITIZER.sub("_", name)
    # Collapse runs of underscores so replaced Unicode runs don't leave
    # "________" noise in the fallback filename.
    cleaned = re.sub(r"_+", "_", cleaned).strip("._ ")
    return cleaned or "summary"


@router.get("/files/{file_id}/summary/detailed.md")
async def download_detailed_summary(
    file_id: str,
    drive: str = Depends(require_drive),
) -> Response:
    """Return the stored detailed summary as ``text/markdown`` for download.

    404 when:

    * the feature is disabled
    * no summary row exists
    * ``status != 'generated'`` (still working, or failed)

    The download filename is ``{stem}_summary.md``; non-ASCII filenames
    are exposed via RFC 5987 ``filename*`` alongside an ASCII fallback.
    """
    if settings.features.detailed_summaries == "false":
        raise HTTPException(status_code=404, detail="Feature disabled")

    basics = _get_indexed_file_basics(file_id)
    if basics is None:
        raise HTTPException(status_code=404, detail="File not indexed")
    file_drive, filename = basics
    assert_file_in_drive(file_drive, drive)

    data = _get_detailed_summary(file_id)
    if (
        data is None
        or data["detailed_status"] != DETAILED_STATUS_GENERATED
        or not data["detailed_summary"]
    ):
        raise HTTPException(status_code=404, detail="No summary available")

    # Strip the original extension so we produce ``lecture_summary.md``
    # rather than ``lecture.mp4_summary.md``. Works for any filename by
    # splitting on the last dot.
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    download_name = f"{stem}_summary.md"
    ascii_name = _sanitize_ascii_filename(download_name)
    # RFC 5987 ``filename*=UTF-8''…`` carries the Unicode name for
    # modern browsers; the ASCII ``filename=`` fallback keeps legacy
    # clients on a readable name.
    disposition = (
        f'attachment; filename="{ascii_name}"; '
        f"filename*=UTF-8''{urlquote(download_name)}"
    )

    return Response(
        content=data["detailed_summary"],
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": disposition},
    )


# ---------------------------------------------------------------------------
# Citations (Phase 1)
# ---------------------------------------------------------------------------


@router.get(
    "/files/{file_id}/summary/detailed/citations",
    response_model=DetailedSummaryCitationsResponse,
)
async def get_detailed_summary_citations(
    file_id: str,
    drive: str = Depends(require_drive),
) -> DetailedSummaryCitationsResponse:
    """Return per-segment citations for a detailed summary.

    Each citation row carries the segment's source ``section_path``
    (matching the UI-side parser output), the top-1 cosine similarity
    ``top_score``, and a ``has_citation`` flag driven by the
    ``summaries.citation_threshold`` setting. The list is empty when
    citations haven't been computed yet (e.g. detailed summary still
    generating, or an older summary that predates this feature — run
    the backfill script to populate).
    """
    _require_file_in_drive(file_id, drive)

    # Import lazily so test suites that stub out app.citations still
    # work — the module imports embedder which pulls sentence-transformers
    # at import time in some paths.
    from app.citations import get_citations

    rows = get_citations(file_id)
    return DetailedSummaryCitationsResponse(
        file_id=file_id,
        citations=[DetailedSummaryCitationItem(**row) for row in rows],
    )


# ---------------------------------------------------------------------------
# Detailed-summary edit / revert (Phase 2)
# ---------------------------------------------------------------------------


def _fetch_detailed_edit_state(
    session: object, file_id: str
) -> tuple[str | None, str | None, str | None] | None:
    """Return ``(detailed_summary, detailed_original, detailed_edited_at)``.

    Used by edit / revert / regenerate-check. Reads from
    ``file_insights`` (Step 2a) — the active row is the current body
    and, when it's a manual edit, the latest superseded intelligence
    row supplies ``original`` for revert.

    Returns ``None`` when there is no active insight row. Callers
    distinguish "no detailed work yet" (None) from "generated but no
    user edit" (``original`` / ``edited_at`` are None).
    """
    active = session.execute(
        sql_text(
            "SELECT content, metadata_json, created_by "
            "FROM file_insights "
            "WHERE file_id = :fid AND kind = 'detailed_summary' "
            "AND status = 'active'"
        ),
        {"fid": file_id},
    ).fetchone()
    if active is None:
        return None

    content = active[0]
    meta = json.loads(active[1]) if active[1] else {}
    created_by = active[2]

    if created_by != "manual":
        return (content, None, None)

    # Manual active: pull pre-edit AI body from the latest superseded
    # intelligence row so revert can splice it back.
    ai_row = session.execute(
        sql_text(
            "SELECT content FROM file_insights "
            "WHERE file_id = :fid AND kind = 'detailed_summary' "
            "AND created_by = 'intelligence' "
            "ORDER BY created_at DESC LIMIT 1"
        ),
        {"fid": file_id},
    ).fetchone()
    original = ai_row[0] if ai_row is not None else None
    edited_at = meta.get("edited_at")
    return (content, original, edited_at)


def _append_detailed_summary_insight(
    session: object,
    *,
    file_id: str,
    content: str,
    created_by: str,
    metadata: dict,
    created_at: str,
) -> None:
    """Append a ``kind='detailed_summary'`` event to ``file_insights``.

    Marks the existing active row superseded (if any) and inserts a
    new active row. Mirrors ``app.workers.summaries._supersede_and_insert_insight``
    but lives here to avoid pulling the worker module into the router
    (the worker imports heavy dependencies that the router does not
    need during cold start).

    The caller is responsible for committing the session.
    """
    session.execute(
        sql_text(
            "UPDATE file_insights SET status = 'superseded' "
            "WHERE file_id = :fid AND kind = 'detailed_summary' "
            "AND status = 'active'"
        ),
        {"fid": file_id},
    )
    session.execute(
        sql_text(
            "INSERT INTO file_insights "
            "(id, file_id, kind, content, metadata_json, "
            " status, created_by, created_at) "
            "VALUES (:id, :fid, 'detailed_summary', :c, :m, "
            " 'active', :cb, :ca)"
        ),
        {
            "id": generate_insight_id(),
            "fid": file_id,
            "c": content,
            "m": json.dumps(metadata) if metadata else None,
            "cb": created_by,
            "ca": created_at,
        },
    )


@router.put(
    "/files/{file_id}/summary/detailed/section",
    response_model=DetailedSummaryResponse,
)
async def edit_detailed_summary_section(
    file_id: str,
    body: DetailedSummaryEditRequest,
    background_tasks: BackgroundTasks,
    drive: str = Depends(require_drive),
) -> DetailedSummaryResponse:
    """Splice one heading-anchored range of the detailed summary.

    Behaviour:

    1. 404 if no detailed summary exists (nothing to edit).
    2. First edit snapshots ``detailed_summary`` into
       ``detailed_original`` so revert can restore it. Subsequent
       edits leave the snapshot pinned to the original AI output.
    3. ``detailed_summary`` is rewritten by splicing ``new_content``
       over the range anchored by ``section_heading`` (H2) and
       optionally ``subsection_heading`` (H3). The heading line is
       part of the replaced range so the user can rename it.
    4. ``detailed_edited_at`` is set to the current UTC timestamp.
    5. Citations are recalculated via FastAPI ``BackgroundTasks`` so
       the HTTP response returns immediately without waiting for
       embedding (which runs via ``run_in_executor`` and can take
       seconds). The frontend picks up citations via WebSocket
       ``citations_ready``, emitted from inside
       ``_recalculate_citations`` once it completes.
    6. ``intelligence.detailed_summary.updated`` is emitted synchronously
       (it's just a cheap notification) with ``citations_ready: false``.

    409 when the requested heading(s) no longer exist in the stored
    summary — treated as optimistic-lock failure ("the document
    changed, reload"), not client-side validation. The fragment
    itself is never validated; adding, removing, or restructuring
    ``##`` / ``###`` inside ``new_content`` is all accepted and
    reflected on reload.
    """
    _require_detailed_feature_enabled()
    _require_file_in_drive(file_id, drive)

    now = datetime.now(UTC).isoformat()

    with get_search_db() as session:
        state = _fetch_detailed_edit_state(session, file_id)
        if state is None or state[0] is None:
            raise HTTPException(
                status_code=404, detail="No detailed summary to edit"
            )
        current = state[0]

        # Lazy-import so the parser is only paid for on the edit path.
        from app.summary_parser import splice_section

        try:
            new_summary = splice_section(
                current,
                body.section_heading,
                body.subsection_heading,
                body.new_content,
            )
        except ValueError as e:
            # 409 (not 400): anchor missing = stale client state, not
            # a malformed request. Frontend prompts a reload.
            raise HTTPException(status_code=409, detail=str(e)) from e

        # Step 2b: the body + original + edited_at no longer live on
        # ``file_summaries``. Appending the manual FileInsight row is
        # the sole write — the previous active insight is marked
        # superseded (so revert can surface the pre-edit AI body) and
        # the new manual row carries ``metadata.edited_at`` for audit.
        _append_detailed_summary_insight(
            session,
            file_id=file_id,
            content=new_summary,
            created_by="manual",
            metadata={"edited_at": now},
            created_at=now,
        )

    # Cheap WS notification — emit synchronously so the frontend knows
    # the summary body changed before the (slow) citation recompute
    # finishes.
    await _emit_ws_event(
        "intelligence.detailed_summary.updated",
        {
            "file_id": file_id,
            "edited_at": now,
            "citations_ready": False,
        },
    )
    # Defer the expensive embedding work to a BackgroundTask so the
    # HTTP response goes out immediately. ``_recalculate_citations``
    # emits ``citations_ready`` itself when the work completes.
    background_tasks.add_task(_recalculate_citations, file_id, new_summary)

    data = _get_detailed_summary(file_id)
    if data is None:
        # Defensive: we just wrote the row, so this should be impossible.
        raise HTTPException(
            status_code=500, detail="Detailed summary vanished post-edit"
        )
    return _detailed_row_to_response(file_id, data)


@router.post(
    "/files/{file_id}/summary/detailed/revert",
    response_model=DetailedSummaryResponse,
)
async def revert_detailed_summary(
    file_id: str,
    background_tasks: BackgroundTasks,
    drive: str = Depends(require_drive),
) -> DetailedSummaryResponse:
    """Restore the AI-generated detailed summary from the snapshot.

    400 when no snapshot exists — either the summary was never edited,
    or a previous regenerate cleared the snapshot. The frontend hides
    the revert button in that state, but the 400 is a defensive check.

    Citation recompute runs in a ``BackgroundTask`` so the response
    does not block on embedding. See ``edit_detailed_summary_section``
    for the matching pattern.
    """
    _require_detailed_feature_enabled()
    _require_file_in_drive(file_id, drive)

    with get_search_db() as session:
        state = _fetch_detailed_edit_state(session, file_id)
        if state is None:
            raise HTTPException(
                status_code=404, detail="No detailed summary found"
            )
        _current, original, _edited_at = state
        if original is None:
            raise HTTPException(
                status_code=400, detail="No AI version to revert to"
            )

        # Step 2b: body + snapshot + edited_at are no longer stored
        # on ``file_summaries``; the FileInsight append alone restores
        # the AI version. ``reverted_from_manual`` metadata marks the
        # new active row as a user-initiated restore.
        _append_detailed_summary_insight(
            session,
            file_id=file_id,
            content=original,
            created_by="intelligence",
            metadata={
                "reverted_from_manual": True,
                "reverted_at": datetime.now(UTC).isoformat(),
            },
            created_at=datetime.now(UTC).isoformat(),
        )

    await _emit_ws_event(
        "intelligence.detailed_summary.updated",
        {
            "file_id": file_id,
            "edited_at": None,
            "citations_ready": False,
        },
    )
    # Defer citation recompute — see edit_detailed_summary_section.
    background_tasks.add_task(_recalculate_citations, file_id, original)

    data = _get_detailed_summary(file_id)
    if data is None:
        raise HTTPException(
            status_code=500, detail="Detailed summary vanished post-revert"
        )
    return _detailed_row_to_response(file_id, data)


@router.post(
    "/files/{file_id}/summary/detailed/regenerate",
    response_model=DetailedSummaryStartResponse,
)
async def regenerate_detailed_summary(
    file_id: str,
    background_tasks: BackgroundTasks,
    body: DetailedSummaryRegenerateRequest | None = None,
    drive: str = Depends(require_drive),
) -> DetailedSummaryStartResponse:
    """Re-run detailed-summary generation for a file.

    Convenience endpoint that combines DELETE + POST and adds the
    edit-conflict check:

    * 409 when ``detailed_edited_at IS NOT NULL`` and the request
      did not set ``force: true``. Frontend surfaces a confirmation
      dialog ("your edits will be lost — continue?") and resubmits
      with ``force: true`` on confirm.
    * 400 propagates the same pre-flight checks as ``start_detailed_summary``
      (unsupported type / insufficient content / missing file).
    """
    _require_detailed_enabled()
    _require_file_in_drive(file_id, drive)

    force = bool(body.force) if body is not None else False

    reason = classify_detailed_missing_reason(file_id)
    if reason in (
        "unsupported_type", "insufficient_content", "file_not_found",
    ):
        raise HTTPException(status_code=400, detail=reason)

    # Atomic conflict-check + delete: hold a single session across the
    # check and the clear so a concurrent edit cannot slip in between
    # ``_fetch_detailed_edit_state`` and the row wipe. Without this
    # guard, an edit landing mid-window would silently be overwritten
    # instead of producing 409. The delete logic is inlined (rather
    # than calling ``_delete_detailed_summary`` which opens its own
    # session) so the write lock is held continuously.
    with get_search_db() as session:
        state = _fetch_detailed_edit_state(session, file_id)
        edited_at = state[2] if state is not None else None
        if edited_at is not None and not force:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Detailed summary has user edits — "
                    "pass force=true to overwrite"
                ),
            )

        # Clear any existing row (AI or edited) so the generation path
        # starts from a pristine state — mirrors the DELETE + POST flow
        # the frontend previously did in two steps. Kept in the same
        # session as the edit-flag check above to close the TOCTOU
        # window.
        row = session.execute(
            sql_text(
                "SELECT short_summary, long_summary FROM file_summaries "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()
        if row is not None:
            short_val = row[0] or ""
            long_val = row[1] or ""
            if short_val or long_val:
                # Keep the file_summaries row (short/long are live
                # data) but clear the remaining two detailed_* workflow
                # columns. Status is re-seeded below so the frontend's
                # slot never flashes "hidden" during the
                # regenerate→worker handoff.
                session.execute(
                    sql_text(
                        "UPDATE file_summaries SET "
                        "detailed_status = NULL, "
                        "detailed_error = NULL "
                        "WHERE file_id = :fid"
                    ),
                    {"fid": file_id},
                )
            else:
                # Placeholder row with no short/long content — drop it
                # so repeat generation starts from the pristine state.
                # _set_detailed_status below will INSERT OR IGNORE a
                # fresh placeholder carrying the 'generating' marker.
                session.execute(
                    sql_text(
                        "DELETE FROM file_summaries WHERE file_id = :fid"
                    ),
                    {"fid": file_id},
                )

        # Citations + FileInsight history cleanup runs unconditionally
        # — even if file_summaries has no row, stray rows in these
        # tables (from a partial earlier state) must be purged to
        # honour the regenerate "clean slate" contract.
        session.execute(
            sql_text(
                "DELETE FROM detailed_summary_citations "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        )
        session.execute(
            sql_text(
                "DELETE FROM file_insights "
                "WHERE file_id = :fid AND kind = 'detailed_summary'"
            ),
            {"fid": file_id},
        )

    # Mark the file as ``generating`` synchronously before yielding
    # control to the background worker. Without this, the reader
    # would see "no active insight + no workflow status" in the
    # window between this endpoint returning and the worker's first
    # ``_set_detailed_status('generating')`` call, causing the
    # frontend to hide the slot entirely. With the marker in place,
    # subsequent polls render the "generating" spinner instead.
    from app.workers.summaries import (
        DETAILED_STATUS_GENERATING, _set_detailed_status,
    )
    _set_detailed_status(
        file_id, DETAILED_STATUS_GENERATING, model=settings.llm.model,
    )

    # Promotion to knowledge (Phase 3) records the current note as the
    # file's active summary in core. Regenerating the AI draft
    # semantically invalidates that pointer: the file detail page must
    # flip back to showing the freshly-generated AI summary. The knowledge
    # `.md` itself is preserved — only the pointer is cleared.
    await _clear_core_active_summary(file_id)

    background_tasks.add_task(
        generate_detailed_summary, file_id, get_llm_client()
    )
    return DetailedSummaryStartResponse(
        status="accepted",
        message="Detailed summary regeneration started",
    )
