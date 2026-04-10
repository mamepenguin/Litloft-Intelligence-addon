"""AI summaries endpoints.

Provides CRUD-like endpoints for LLM-generated file summaries:

- GET /files/{file_id}/summary            — fetch current summary
- POST /files/{file_id}/summary/regenerate — delete + re-enqueue
- POST /files/{file_id}/summary/hide       — mark as hidden
- POST /batch/summaries                    — queue a batch of files

Access control is enforced by the Generic Addon Proxy in the host
via pre_check rules; this router assumes the caller has permission
for the file_ids it receives.
"""

import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_search_db
from app.dependencies import get_llm_client, get_summaries_worker
from app.schemas import (
    BatchSummariesRequest,
    BatchSummariesResponse,
    MessageResponse,
    SummaryResponse,
)
from app.workers.summaries import classify_missing_reason

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
async def get_summary(file_id: str) -> SummaryResponse:
    """Fetch the stored summary for a file.

    Returns available=False when:
    - the summaries feature is disabled
    - no summary has been generated yet
    - the summary has been hidden by the user
    - the underlying file has been soft-deleted (active=False)

    The soft-delete check matches the host's drive_access filter so a
    bypass of the proxy cannot resurrect summaries of deleted files.
    """
    if settings.features.summaries == "false":
        return SummaryResponse(available=False)

    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT s.file_id, s.short_summary, s.long_summary, s.model, "
                "s.context_type, s.was_truncated, s.status, s.created_at "
                "FROM file_summaries s "
                "INNER JOIN indexed_files f ON s.file_id = f.file_id "
                "WHERE s.file_id = :fid AND f.active = 1"
            ),
            {"fid": file_id},
        ).fetchone()

    if row is None:
        # Classify *why* there's no summary so the frontend can render
        # a useful state (button / "insufficient content" / hidden)
        # instead of always offering a generate button that would
        # silently skip when the file lacks usable content.
        return SummaryResponse(
            available=False,
            reason=classify_missing_reason(file_id),
        )

    status_value = row[6]
    if status_value == "hidden":
        return SummaryResponse(available=False)

    return SummaryResponse(
        available=True,
        file_id=row[0],
        short_summary=row[1],
        long_summary=row[2],
        model=row[3],
        context_type=row[4],
        was_truncated=bool(row[5]),
        status=status_value,
        created_at=row[7],
    )


@router.post(
    "/files/{file_id}/summary/regenerate", response_model=MessageResponse
)
async def regenerate_summary(file_id: str) -> MessageResponse:
    """Delete the existing summary and re-queue generation.

    Rejects the request upfront (400) when the file cannot be summarized
    — either the type is unsupported or the context is below threshold.
    This avoids a 40-second frontend polling timeout when the request
    would have been silently skipped by the worker anyway.
    """
    _require_llm_enabled()

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


@router.post("/files/{file_id}/summary/hide", response_model=MessageResponse)
async def hide_summary(file_id: str) -> MessageResponse:
    """Mark a summary as hidden so it stops being displayed.

    The row is preserved (not deleted) so the data is still available
    for audit / debugging / potential un-hide UI.
    """
    with get_search_db() as session:
        result = session.execute(
            sql_text(
                "UPDATE file_summaries SET status = 'hidden' "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        )

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="No summary found")

    return MessageResponse(status="ok", message="Summary hidden")


@router.post("/batch/summaries", response_model=BatchSummariesResponse)
async def batch_summaries(
    body: BatchSummariesRequest,
) -> BatchSummariesResponse:
    """Queue summary generation for a batch of files.

    Files that already have a summary (any status) are skipped.

    SECURITY NOTE: the Generic Addon Proxy currently only enforces
    file_access pre_checks against path parameters, NOT against body
    payloads. The same limitation applies to /batch/suggested-tags.
    Until the proxy learns to filter body arrays, callers can enqueue
    LLM work for files they cannot read (the GET endpoint still blocks
    reading the result, but the operator incurs LLM cost). Tracked as
    follow-up work; acceptable for single-user home-LAN deployments.
    """
    _require_llm_enabled()

    summaries_worker = get_summaries_worker()

    # Find which files already have summaries to avoid duplicate work.
    existing: set[str] = set()
    if body.file_ids:
        with get_search_db() as session:
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
        if file_id in existing:
            skipped += 1
        else:
            await summaries_worker.enqueue(file_id)
            queued += 1

    return BatchSummariesResponse(queued=queued, skipped=skipped)
