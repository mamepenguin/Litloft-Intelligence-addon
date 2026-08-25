"""Transcript AI refine endpoints.

* ``POST /refine/files/{file_id}`` — enqueue refine for one file
* ``POST /refine/folders`` — refine every transcript-bearing file
  in the supplied ``file_ids`` list

Revert is deliberately not exposed: the refine pipeline now re-chunks
the transcript on LLM-inserted punctuation boundaries, which makes
per-chunk originals meaningless. Users who want to undo a refine
should re-run whisper indexing from scratch (`whisper_indexed=False`).

Access control (drive-scope) is enforced by the host's Generic Addon
Proxy. The router adds feature-flag gating, per-drive policy gating,
and existence checks so an attacker inside an allowed drive can't
silently burn LLM credits on non-existent file_ids.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from app.config import settings
from app.database import get_search_db
from app.dependencies import get_llm_client
from app.drive_context import require_drive
from app.models import IndexedFile, TranscriptChunk
from app.workers.refine import (
    WINDOW_SIZE,
    filter_transcript_file_ids,
    is_feature_enabled,
    realign_words_for_chunk,
    recompute_chunk_embeddings,
    start_refine_job,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["refine"])

# Cap per-request folder job fan-out. The host already bounds request
# size at the proxy layer, but the router is the authoritative gate on
# how much LLM traffic a single caller can kick off. Tuned conservatively
# so a single drive can't starve other drives of the shared LLM slot.
MAX_FOLDER_FILES = 100
FOLDER_CONCURRENCY = 2


def _require_feature_on() -> None:
    """Global feature-flag gate; 400 when off.

    Per-drive policy is applied in a separate pass so we can surface
    the distinct 403 expected by tests when a single drive opts out
    despite the global flag being "manual".
    """
    if settings.features.transcript_refine == "false":
        raise HTTPException(
            status_code=400,
            detail="Transcript refine feature is disabled",
        )
    llm = get_llm_client()
    if not llm.enabled:
        raise HTTPException(status_code=400, detail="LLM is not enabled")


async def _require_drive_policy(drive: str) -> None:
    """Per-drive policy gate; 403 when this drive opts out.

    ``is_feature_enabled`` is async because it dispatches to the host
    Internal API via the policy client — awaiting it from inside the
    running loop is the only way to honour the real decision.
    """
    enabled = await is_feature_enabled(drive)
    if not enabled:
        raise HTTPException(
            status_code=403,
            detail="Transcript refine disabled for this drive",
        )


def _fetch_indexed_file(session: Any, file_id: str, drive: str) -> Any | None:
    """Look up an indexed file restricted to a single drive.

    Filtering on ``drive`` here — not just ``file_id`` — prevents a
    caller authorised for drive A from refining a file that happens
    to exist in drive B by submitting B's file_id with A's X-Lit-Drive.
    The host proxy's ``file_access`` pre_check already blocks the
    obvious case, but we defence-in-depth at the addon layer too.
    """
    return (
        session.query(IndexedFile)
        .filter(
            IndexedFile.file_id == file_id,
            IndexedFile.drive == drive,
            IndexedFile.active.is_(True),
        )
        .first()
    )


@router.post("/refine/files/{file_id}")
async def refine_file(
    file_id: str,
    drive: str = Depends(require_drive),
) -> dict:
    """Start a refine job for a single file."""
    _require_feature_on()
    await _require_drive_policy(drive)

    with get_search_db() as session:
        indexed = _fetch_indexed_file(session, file_id, drive)
        if indexed is None:
            raise HTTPException(status_code=404, detail="File not found")

        chunk_count = (
            session.query(TranscriptChunk)
            .filter(TranscriptChunk.file_id == file_id)
            .count()
        )
        job_id = start_refine_job(session, file_id)

    return {"job_id": job_id, "chunk_count": chunk_count}


@router.post("/refine/folders")
async def refine_folder(
    body: dict = Body(...),
    drive: str = Depends(require_drive),
) -> dict:
    """Enqueue refine jobs for every transcript-bearing file in ``file_ids``.

    Same shape as ``/batch/summaries``: the frontend sends the file_ids
    it just rendered. We re-filter to (drive, active, has-transcript)
    here so a caller can't reach across drives or feed nonexistent ids
    to burn LLM credits.

    The body ``drive`` (if present) must match the ``X-Lit-Drive``
    header; mismatches signal a confused client or a deliberate
    cross-drive probe, so we 400 instead of silently using the header.
    """
    body_drive = body.get("drive")
    if body_drive is not None and body_drive != drive:
        raise HTTPException(
            status_code=400,
            detail="body.drive must match X-Lit-Drive header",
        )
    raw_ids = body.get("file_ids")
    if not isinstance(raw_ids, list) or not all(
        isinstance(x, str) for x in raw_ids
    ):
        raise HTTPException(
            status_code=400, detail="file_ids must be a list of strings"
        )

    _require_feature_on()
    await _require_drive_policy(drive)

    with get_search_db() as session:
        file_ids = filter_transcript_file_ids(session, drive, raw_ids)

    if len(file_ids) > MAX_FOLDER_FILES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Folder contains {len(file_ids)} files, "
                f"exceeds per-request cap of {MAX_FOLDER_FILES}"
            ),
        )

    # Bounded concurrency so one /folders call can't starve individual
    # /files/{id} calls of the shared LLM slot. A semaphore + gather is
    # simpler than a worker queue and sufficient at MAX_FOLDER_FILES.
    sem = asyncio.Semaphore(FOLDER_CONCURRENCY)
    queued_ids: list[str] = []

    async def _enqueue(fid: str) -> None:
        async with sem:
            with get_search_db() as session:
                start_refine_job(session, fid)
            queued_ids.append(fid)

    if file_ids:
        await asyncio.gather(*(_enqueue(fid) for fid in file_ids))

    return {"queued": len(queued_ids), "file_ids": queued_ids}


# Re-export for monkeypatch targets used in tests.
__all__ = [
    "FOLDER_CONCURRENCY",
    "MAX_FOLDER_FILES",
    "WINDOW_SIZE",
    "filter_transcript_file_ids",
    "get_llm_client",
    "get_search_db",
    "is_feature_enabled",
    "realign_words_for_chunk",
    "recompute_chunk_embeddings",
    "refine_file",
    "refine_folder",
    "settings",
    "start_refine_job",
]
