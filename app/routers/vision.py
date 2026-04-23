"""Vision-LLM image description endpoints.

Spec: docs/superpowers/specs/2026-04-23-intelligence-vision-describe.md

Four routes:

* ``POST   /files/{file_id}/visual_description/generate`` — manual trigger
* ``GET    /files/{file_id}/visual_description``           — read state
* ``DELETE /files/{file_id}/visual_description``           — clear
* ``POST   /folders/visual_description/generate``          — bulk enqueue

Access gates (applied in order):

1. ``features.vision_describe == "false"`` → 404 on every route
2. ``llm.vision_model`` empty → 404 on generate routes; GET returns
   ``status="unsupported"`` so the UI can render the helpful notice.
3. Per-drive policy OFF → 404 on every route
4. Cross-drive access → 404 (consistent with host ``drive_access``)
5. Non-image mime → 404 on generate (other routes don't care: GET
   of a non-image row just returns ``available=False``).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from sqlalchemy import text as sql_text
from sqlalchemy.exc import OperationalError

from app.config import is_vision_describe_available, settings
from app.database import get_search_db
from app.dependencies import get_llm_client
from app.drive_context import require_drive
from app.models import Embedding, IndexedFile
from app.policy_client import is_feature_enabled as _policy_is_feature_enabled

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision_describe"])


# Hard cap on how many files the folder-fanout endpoint will enqueue in
# one request. Shields the worker queue from an operator who points the
# bulk button at a 50 000-image drive and kicks off an open-ended spend
# with an external vision API. 500 matches the "reasonable interactive
# batch" heuristic other bulk paths in the project use.
MAX_BULK_ENQUEUE = 500


# Tests monkeypatch this symbol directly; keep the indirection so
# per-drive policy can be stubbed without reaching into ``policy_client``.
async def is_feature_enabled(drive: str, feature: str = "vision_describe") -> bool:
    """Policy gate wrapper — async so tests can AsyncMock it directly."""
    try:
        return await _policy_is_feature_enabled(drive, feature)
    except Exception:
        # Match policy_client's fail-open posture so transient host
        # outages don't take the router down.
        return True


# ---------------------------------------------------------------------------
# Internal helpers — exposed for tests so the same indirection monkeypatch
# pattern (used by summaries / refine routers) keeps working.
# ---------------------------------------------------------------------------


def _fetch_indexed_file(file_id: str) -> Any | None:
    """Return the active IndexedFile row for ``file_id`` (or None).

    Missing / trash files are filtered out here so all vision routes
    skip them uniformly — matches the host convention that
    ``active=False`` rows are invisible to feature endpoints.
    """
    with get_search_db() as session:
        return (
            session.query(IndexedFile)
            .filter(
                IndexedFile.file_id == file_id,
                IndexedFile.active.is_(True),
            )
            .first()
        )


def _fetch_visual_description(file_id: str) -> dict | None:
    """Load the stored vision description + metadata for ``file_id``.

    Returns a dict of the four ``visual_description*`` columns, or
    ``None`` when the row doesn't exist. Tests override this to return
    canned shapes without hitting SQLite.
    """
    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT visual_description, visual_description_status, "
                "visual_description_model, visual_description_generated_at "
                "FROM file_summaries WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()
    if row is None:
        return None
    return {
        "visual_description": row[0],
        "visual_description_status": row[1],
        "visual_description_model": row[2],
        "visual_description_generated_at": row[3],
    }


def _clear_visual_description(file_id: str) -> bool:
    """Drop vision columns + vision_description embeddings for ``file_id``.

    Returns True when at least one vision artefact was actually cleared;
    False when there was nothing to delete. Callers shape 404 from that.
    """
    cleared_any = False
    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT visual_description, visual_description_status "
                "FROM file_summaries WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()
        if row is not None and (row[0] is not None or row[1] is not None):
            session.execute(
                sql_text(
                    "UPDATE file_summaries SET "
                    "visual_description = NULL, "
                    "visual_description_status = NULL, "
                    "visual_description_model = NULL, "
                    "visual_description_generated_at = NULL "
                    "WHERE file_id = :fid"
                ),
                {"fid": file_id},
            )
            cleared_any = True

        embeddings = (
            session.query(Embedding)
            .filter(
                Embedding.file_id == file_id,
                Embedding.embedding_type == "vision_description",
            )
            .all()
        )
        for emb in embeddings:
            table = emb.vector_table or ""
            if table.startswith("vec_"):
                try:
                    session.execute(
                        sql_text(
                            f"DELETE FROM {table} WHERE embedding_id = :id"
                        ),
                        {"id": emb.id},
                    )
                except OperationalError as e:
                    # Missing vec table in narrow test harnesses — log
                    # and carry on so the metadata row still drops.
                    logger.warning(
                        "vision: vec delete failed for %s (%s)",
                        table, type(e).__name__,
                    )
            session.delete(emb)
            cleared_any = True

    return cleared_any


async def enqueue_visual_description(file_id: str) -> bool:
    """Hand ``file_id`` to the vision worker.

    Prefers the dependency-injected worker singleton so the running
    loop sees the same queue as the background task. Tests override
    this helper wholesale.
    """
    try:
        from app.dependencies import _vision_worker  # type: ignore

        worker = _vision_worker
    except Exception:
        worker = None
    if worker is None:
        # Lazy-construct so manual trigger works even if the startup
        # hook didn't wire a singleton (e.g. in partial test harnesses).
        from app.workers.vision import VisionDescribeWorker

        worker = VisionDescribeWorker()
    return await worker.enqueue(file_id)


def find_image_files_in_folder(drive: str, path: str) -> list[str]:
    """Return image file_ids under ``drive/path`` (prefix match).

    Mirrors :func:`app.workers.refine.find_transcript_files_in_folder`
    but narrows to ``image/*`` mimes since vision_describe is only
    defined for images in Phase 1.
    """
    prefix = path.rstrip("/")
    with get_search_db() as session:
        rows = session.execute(
            sql_text(
                "SELECT f.file_id FROM indexed_files f "
                "WHERE f.drive = :drive AND f.active = 1 "
                "AND f.mime_type LIKE 'image/%' "
                "AND (f.file_path = :path OR f.file_path LIKE :like)"
            ),
            {
                "drive": drive,
                "path": prefix,
                "like": f"{prefix}/%",
            },
        ).fetchall()
    return [row[0] for row in rows]


# ---------------------------------------------------------------------------
# Access gates
# ---------------------------------------------------------------------------


def _require_feature_available() -> None:
    """Shared gate — 404 when the whole feature is unreachable."""
    if settings.features.vision_describe == "false":
        raise HTTPException(status_code=404, detail="Feature disabled")
    if not is_vision_describe_available(settings):
        # This collapses "features on but vision_model unset" into the
        # same 404 generate routes return. GET has its own override
        # below so users see a useful "unsupported" notice.
        raise HTTPException(status_code=404, detail="Vision model not configured")


async def _require_drive_policy(drive: str) -> None:
    """Per-drive policy gate — 404 when the drive opts out."""
    enabled = await is_feature_enabled(drive, "vision_describe")
    if not enabled:
        raise HTTPException(status_code=404, detail="Feature disabled for drive")


def _validate_folder_path(path: str) -> str:
    """Reject absolute / traversal / empty paths — same shape as refine."""
    if not isinstance(path, str) or path.strip() == "":
        raise HTTPException(status_code=400, detail="path is required")
    if path.startswith("/"):
        raise HTTPException(status_code=400, detail="path must be relative")
    parts = path.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        raise HTTPException(status_code=400, detail="path must not contain '..'")
    return path


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/files/{file_id}/visual_description/generate")
async def generate_visual_description(
    file_id: str,
    background_tasks: BackgroundTasks,
    drive: str = Depends(require_drive),
) -> dict:
    """Kick off a vision description for one file."""
    _require_feature_available()
    await _require_drive_policy(drive)

    file_row = _fetch_indexed_file(file_id)
    if file_row is None:
        raise HTTPException(status_code=404, detail="File not found")
    if getattr(file_row, "drive", None) != drive:
        # Cross-drive probe — answer 404 so the requester can't infer
        # which drive a file_id actually belongs to.
        raise HTTPException(status_code=404, detail="File not found")
    mime = getattr(file_row, "mime_type", None)
    if not mime or not str(mime).lower().startswith("image/"):
        raise HTTPException(status_code=404, detail="Not an image file")

    # Schedule the actual work off the request path so a slow LLM call
    # never blocks the browser. Tests can also await the helper directly
    # by monkeypatching ``enqueue_visual_description``.
    background_tasks.add_task(enqueue_visual_description, file_id)
    return {"status": "accepted", "file_id": file_id}


@router.get("/files/{file_id}/visual_description")
async def get_visual_description(
    file_id: str,
    drive: str = Depends(require_drive),
) -> dict:
    """Read current vision description state for ``file_id``.

    404 when the feature is disabled globally or for the drive. When
    ``vision_model`` is unset but features.vision_describe is truthy
    we surface ``status="unsupported"`` instead of 404 so the UI can
    render the guidance message without polling.
    """
    if settings.features.vision_describe == "false":
        raise HTTPException(status_code=404, detail="Feature disabled")
    await _require_drive_policy(drive)

    file_row = _fetch_indexed_file(file_id)
    if file_row is None:
        raise HTTPException(status_code=404, detail="File not found")
    if getattr(file_row, "drive", None) != drive:
        raise HTTPException(status_code=404, detail="File not found")

    if not is_vision_describe_available(settings):
        return {
            "file_id": file_id,
            "visual_description": None,
            "status": "unsupported",
            "model": None,
            "generated_at": None,
        }

    data = _fetch_visual_description(file_id)
    if data is None:
        return {
            "file_id": file_id,
            "visual_description": None,
            "status": None,
            "model": None,
            "generated_at": None,
        }

    return {
        "file_id": file_id,
        "visual_description": data.get("visual_description"),
        "status": data.get("visual_description_status"),
        "model": data.get("visual_description_model"),
        "generated_at": data.get("visual_description_generated_at"),
    }


@router.delete("/files/{file_id}/visual_description")
async def delete_visual_description(
    file_id: str,
    drive: str = Depends(require_drive),
) -> dict:
    """Clear the description + embeddings so the next generate restarts from scratch."""
    _require_feature_available()
    await _require_drive_policy(drive)

    file_row = _fetch_indexed_file(file_id)
    if file_row is None:
        raise HTTPException(status_code=404, detail="File not found")
    if getattr(file_row, "drive", None) != drive:
        raise HTTPException(status_code=404, detail="File not found")

    cleared = _clear_visual_description(file_id)
    if not cleared:
        raise HTTPException(status_code=404, detail="No description to clear")
    return {"status": "ok", "file_id": file_id}


@router.post("/folders/visual_description/generate")
async def generate_folder_visual_description(
    body: dict = Body(...),
    drive: str = Depends(require_drive),
) -> dict:
    """Fan-out: enqueue every image file under a folder prefix."""
    body_drive = body.get("drive") if isinstance(body, dict) else None
    if body_drive is not None and body_drive != drive:
        raise HTTPException(
            status_code=400,
            detail="body.drive must match X-Lit-Drive header",
        )
    path_value = body.get("path", "") if isinstance(body, dict) else ""
    path = _validate_folder_path(path_value)

    _require_feature_available()
    await _require_drive_policy(drive)

    file_ids = find_image_files_in_folder(drive, path)

    # Cap counts files that pass every acceptance gate (policy, mime,
    # stickiness) — not the raw folder scan. A folder with 10 000
    # images where 9 500 are already "success" still enqueues the
    # remaining 500 cleanly. Only genuine new work above the cap trips
    # 413. We stop calling ``enqueue`` the moment the cap would be
    # exceeded so we never put a 501st file onto the worker queue.
    queued = 0
    queued_ids: list[str] = []
    for fid in file_ids:
        if queued >= MAX_BULK_ENQUEUE:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "too_many_files",
                    "max": MAX_BULK_ENQUEUE,
                    "requested": len(file_ids),
                },
            )
        accepted = await enqueue_visual_description(fid)
        if accepted:
            queued += 1
            queued_ids.append(fid)

    return {"queued": queued, "file_ids": queued_ids}


__all__ = [
    "delete_visual_description",
    "enqueue_visual_description",
    "find_image_files_in_folder",
    "generate_folder_visual_description",
    "generate_visual_description",
    "get_llm_client",
    "get_search_db",
    "get_visual_description",
    "is_feature_enabled",
    "router",
    "settings",
    "_clear_visual_description",
    "_fetch_visual_description",
]
