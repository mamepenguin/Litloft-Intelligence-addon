"""Transcript AI refine endpoints.

* ``POST /refine/files/{file_id}`` — enqueue refine for one file
* ``POST /refine/files/{file_id}/revert`` — restore text_original
* ``POST /refine/folders`` — refine every transcript-bearing file
  under a given drive+path

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
    find_transcript_files_in_folder,
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
    to exist in drive B by submitting B's file_id with A's X-HV-Drive.
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


def _validate_folder_path(path: str) -> str:
    """Reject absolute / traversal / empty paths; return the stripped form.

    The path is joined into a SQL LIKE pattern against ``file_path``
    prefixes. Anything starting with ``/`` escapes the drive root,
    ``..`` segments would let a caller match unrelated prefixes, and
    an empty string is meaningless (tests + UI always pass a folder).
    """
    if not isinstance(path, str) or path.strip() == "":
        raise HTTPException(status_code=400, detail="path is required")
    if path.startswith("/"):
        raise HTTPException(status_code=400, detail="path must be relative")
    # Segment-wise check so "foo..bar" (legitimate substring) doesn't trip.
    parts = path.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        raise HTTPException(status_code=400, detail="path must not contain '..'")
    return path


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
        job_id = await start_refine_job(session, file_id)

    return {"job_id": job_id, "chunk_count": chunk_count}


@router.post("/refine/files/{file_id}/revert")
async def revert_file(
    file_id: str,
    drive: str = Depends(require_drive),
) -> dict:
    """Revert a file's AI-refined chunks back to the stored originals."""
    _require_feature_on()
    await _require_drive_policy(drive)

    with get_search_db() as session:
        indexed = _fetch_indexed_file(session, file_id, drive)
        if indexed is None:
            raise HTTPException(status_code=404, detail="File not found")

        chunks = (
            session.query(TranscriptChunk)
            .filter(TranscriptChunk.file_id == file_id)
            .all()
        )

        reverted_ids: list[int] = []
        for chunk in chunks:
            if chunk.text_original is None:
                continue
            original = chunk.text_original
            chunk.text = original
            chunk.text_original = None
            chunk.text_refined_at = None
            reverted_ids.append(int(chunk.id))
            realign_words_for_chunk(
                session,
                chunk.file_id,
                chunk.timestamp_start,
                chunk.timestamp_end,
                original,
            )

        if reverted_ids:
            await recompute_chunk_embeddings(session, reverted_ids)

    return {"reverted_count": len(reverted_ids)}


@router.post("/refine/folders")
async def refine_folder(
    body: dict = Body(...),
    drive: str = Depends(require_drive),
) -> dict:
    """Enqueue refine jobs for every transcript-bearing file in a folder.

    The body ``drive`` (if present) must match the ``X-HV-Drive``
    header; mismatches are a signal of either a confused client or a
    deliberate cross-drive probe, so we 400 instead of silently using
    the header. The path is validated to reject traversal / absolute
    forms before it reaches the SQL LIKE match.
    """
    body_drive = body.get("drive")
    if body_drive is not None and body_drive != drive:
        raise HTTPException(
            status_code=400,
            detail="body.drive must match X-HV-Drive header",
        )
    path = body.get("path", "")
    path = _validate_folder_path(path)

    _require_feature_on()
    await _require_drive_policy(drive)

    with get_search_db() as session:
        file_ids = find_transcript_files_in_folder(session, drive, path)

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
                await start_refine_job(session, fid)
            queued_ids.append(fid)

    if file_ids:
        await asyncio.gather(*(_enqueue(fid) for fid in file_ids))

    return {"queued": len(queued_ids), "file_ids": queued_ids}


# Re-export for monkeypatch targets used in tests.
__all__ = [
    "FOLDER_CONCURRENCY",
    "MAX_FOLDER_FILES",
    "WINDOW_SIZE",
    "find_transcript_files_in_folder",
    "get_llm_client",
    "get_search_db",
    "is_feature_enabled",
    "realign_words_for_chunk",
    "recompute_chunk_embeddings",
    "refine_file",
    "refine_folder",
    "revert_file",
    "settings",
    "start_refine_job",
]
