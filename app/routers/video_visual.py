"""Video Visual Index endpoints.

Three routes (design doc "Video Visual Index" §11):

* ``GET  /files/{file_id}/visual-index``          — read state
* ``POST /files/{file_id}/visual-index/generate``  — manual trigger
* ``POST /files/{file_id}/visual-index/retry``     — retry failed scenes

Access gates (applied in order), mirroring ``app.routers.vision``:

1. ``features.video_visual_index == "false"`` → 404 on every route
2. ``llm.vision_model`` empty → 404 on generate/retry; GET still
   returns state with ``available=False`` so the UI can render guidance.
3. Per-drive policy OFF → 404 on every route
4. Cross-drive access / not a native video → 404
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.config import is_video_visual_index_available, settings
from app.database import get_search_db_read
from app.drive_context import require_drive
from app.models import IndexedFile, VideoVisualRun, VideoVisualScene
from app.policy_client import is_feature_enabled as _policy_is_feature_enabled
from app.schemas import (
    VideoVisualIndexResponse,
    VideoVisualRunSummary,
    VideoVisualSceneItem,
)
from app.workers.clip import VIDEO_TYPES
from app.workers.video_visual_selection import compute_candidate_fingerprint

logger = logging.getLogger(__name__)

router = APIRouter(tags=["video_visual"])


# Tests monkeypatch this symbol directly; keep the indirection so
# per-drive policy can be stubbed without reaching into ``policy_client``.
async def is_feature_enabled(drive: str, feature: str = "video_visual_index") -> bool:
    try:
        return await _policy_is_feature_enabled(
            drive,
            feature,
            default_on_failure=False,
        )
    except Exception:
        return False


def _fetch_indexed_file(file_id: str) -> Any | None:
    with get_search_db_read() as session:
        return (
            session.query(IndexedFile)
            .filter(IndexedFile.file_id == file_id, IndexedFile.active.is_(True))
            .first()
        )


async def get_video_visual_worker():
    """Resolve the worker singleton (lazy import breaks a startup cycle)."""
    from app.dependencies import get_video_visual_worker as _get

    return _get()


def _require_feature_available() -> None:
    if settings.features.video_visual_index == "false":
        raise HTTPException(status_code=404, detail="Feature disabled")
    if not is_video_visual_index_available(settings):
        raise HTTPException(status_code=404, detail="Vision model not configured")


async def _require_drive_policy(drive: str) -> None:
    try:
        enabled = await is_feature_enabled(drive, "video_visual_index")
    except Exception:
        enabled = False
    if not enabled:
        raise HTTPException(status_code=404, detail="Feature disabled for drive")


def _run_to_summary(run: VideoVisualRun) -> VideoVisualRunSummary:
    return VideoVisualRunSummary(
        run_id=run.id,
        status=run.status,
        selected_count=run.selected_count,
        completed_count=run.completed_count,
        succeeded_count=run.succeeded_count,
        failed_count=run.failed_count,
        created_at=run.created_at.isoformat() if run.created_at else None,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
    )


def _scene_to_item(scene: VideoVisualScene) -> VideoVisualSceneItem:
    return VideoVisualSceneItem(
        ordering=scene.ordering,
        start_time=scene.start_time,
        end_time=scene.end_time,
        status=scene.status,
        scene_type=scene.scene_type,
        scene_label=scene.scene_label,
        visible_text=scene.visible_text,
        transcript_excerpt=scene.transcript_excerpt,
    )


@router.get("/files/{file_id}/visual-index", response_model=VideoVisualIndexResponse)
async def get_visual_index(
    file_id: str,
    drive: str = Depends(require_drive),
) -> VideoVisualIndexResponse:
    """Return active scenes plus staged-run progress in one response.

    Keeps the active result visible while a replacement run is being
    staged (design doc §11) — the UI reads ``active_run``/``scenes`` for
    the persistent result and ``staged_run`` for in-flight/failed-update
    state layered on top.
    """
    if settings.features.video_visual_index == "false":
        raise HTTPException(status_code=404, detail="Feature disabled")
    await _require_drive_policy(drive)

    file_row = _fetch_indexed_file(file_id)
    if file_row is None or getattr(file_row, "drive", None) != drive:
        raise HTTPException(status_code=404, detail="File not found")

    if getattr(file_row, "mime_type", None) not in VIDEO_TYPES:
        return VideoVisualIndexResponse(eligible=False, file_id=file_id)

    with get_search_db_read() as session:
        active = (
            session.query(VideoVisualRun)
            .filter(VideoVisualRun.file_id == file_id, VideoVisualRun.is_active.is_(True))
            .first()
        )
        staged_query = session.query(VideoVisualRun).filter(
            VideoVisualRun.file_id == file_id,
            VideoVisualRun.status.in_(("queued", "running", "partial", "failed")),
        )
        if active is not None:
            staged_query = staged_query.filter(
                VideoVisualRun.id != active.id,
                VideoVisualRun.created_at > active.created_at,
            )
        staged = staged_query.order_by(VideoVisualRun.created_at.desc()).first()

        scenes: list[VideoVisualSceneItem] = []
        if active is not None:
            scene_rows = (
                session.query(VideoVisualScene)
                .filter(VideoVisualScene.run_id == active.id)
                .order_by(VideoVisualScene.ordering)
                .all()
            )
            scenes = [_scene_to_item(s) for s in scene_rows]

    stale = False
    if active is not None:
        from app.workers.video_visual import PIPELINE_VERSION, _load_candidates

        candidates, _duration = _load_candidates(file_id)
        stale = (
            active.pipeline_version != PIPELINE_VERSION
            or compute_candidate_fingerprint(candidates) != active.candidate_fingerprint
        )

    return VideoVisualIndexResponse(
        eligible=True,
        available=is_video_visual_index_available(settings),
        file_id=file_id,
        active_run=_run_to_summary(active) if active is not None else None,
        scenes=scenes,
        staged_run=_run_to_summary(staged) if staged is not None else None,
        stale=stale,
    )


# Reasons that mean "not now" rather than "not here": the file is
# eligible, something else has to happen first. Everything else is a 404,
# which also keeps a file in another drive indistinguishable from one
# that does not exist.
#
# ``unsupported_sticky`` is deliberately absent. This route is always a
# manual request, and manual skips the stickiness gate, so the worker
# cannot answer it here — listing it would be a branch that never runs
# and a message no one would ever read. A model that cannot see now
# fails the run it is given, visibly, instead of refusing to start.
_NOT_YET_REASONS = frozenset({"waiting_clip"})


@router.post("/files/{file_id}/visual-index/generate")
async def generate_visual_index(
    file_id: str,
    drive: str = Depends(require_drive),
) -> dict:
    """Stage a new run for one file (manual priority)."""
    _require_feature_available()
    await _require_drive_policy(drive)

    file_row = _fetch_indexed_file(file_id)
    if file_row is None or getattr(file_row, "drive", None) != drive:
        raise HTTPException(status_code=404, detail="File not found")
    if getattr(file_row, "mime_type", None) not in VIDEO_TYPES:
        raise HTTPException(status_code=404, detail="Not a native video file")

    worker = await get_video_visual_worker()
    result = await worker.enqueue(file_id, requested_by="manual")
    if not result.get("accepted"):
        reason = result.get("reason", "unavailable")
        if reason == "already_queued":
            # Not a refusal: what the caller asked for is on its way.
            return {"status": "already_queued", "file_id": file_id}
        if reason in _NOT_YET_REASONS:
            # A condition that clears on its own or with one operator
            # action. The reason travels as a field rather than as
            # prose, because the frontend has to tell them apart to say
            # anything useful — conflated, they all read as "wait for
            # scene indexing", which is wrong for every one but that.
            raise HTTPException(
                status_code=409,
                detail={"error": "not_queued", "reason": reason},
            )
        raise HTTPException(status_code=404, detail="Not available")

    return {"status": "accepted", "file_id": file_id, "run_id": result.get("run_id")}


@router.post("/files/{file_id}/visual-index/retry")
async def retry_visual_index(
    file_id: str,
    drive: str = Depends(require_drive),
) -> dict:
    """Re-queue only the failed scenes of the most recent retryable run."""
    _require_feature_available()
    await _require_drive_policy(drive)

    file_row = _fetch_indexed_file(file_id)
    if file_row is None or getattr(file_row, "drive", None) != drive:
        raise HTTPException(status_code=404, detail="File not found")

    worker = await get_video_visual_worker()
    result = await worker.retry(file_id)
    if not result.get("accepted"):
        raise HTTPException(status_code=404, detail="No failed scenes to retry")

    return {
        "status": "accepted",
        "file_id": file_id,
        "run_id": result.get("run_id"),
        "reset_count": result.get("reset_count"),
    }


__all__ = [
    "generate_visual_index",
    "get_video_visual_worker",
    "get_visual_index",
    "is_feature_enabled",
    "retry_visual_index",
    "router",
]
