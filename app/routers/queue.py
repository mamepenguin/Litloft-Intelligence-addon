"""Queue control endpoints: pause, resume, prioritize.

Per spec ``2026-05-24-intelligence-reindex-controls.md`` §1 the
``POST /queue/reindex`` global-reset handler is permanently removed —
its blast radius was unbounded (every active file across every drive
flipped to ``*_indexed=False`` on one click; hako WmAMUDZSsMHlutJFKsyAe
records the production incident). The per-file × per-task replacement
lives in ``app.routers.files.reindex_file``.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_index_manager
from app.schemas import MessageResponse, QueuePrioritize

router = APIRouter(tags=["queue"])


@router.post("/queue/prioritize", response_model=MessageResponse)
async def queue_prioritize(
    body: QueuePrioritize,
) -> MessageResponse:
    """Prioritize a specific file for immediate indexing."""
    manager = get_index_manager()
    success = await manager.prioritize(body.file_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"File {body.file_id} not found in index",
        )

    return MessageResponse(
        status="accepted",
        message=f"File {body.file_id} prioritized",
    )


def _pause_video_visual_worker() -> None:
    """Best-effort: also pause the video-visual worker (design doc §9).

    Not yet initialized (feature disabled, or very early startup) is a
    normal state, not an error — silently no-op in that case.
    """
    try:
        from app.dependencies import get_video_visual_worker

        get_video_visual_worker().pause()
    except Exception:
        pass


def _resume_video_visual_worker() -> None:
    try:
        from app.dependencies import get_video_visual_worker

        get_video_visual_worker().resume()
    except Exception:
        pass


@router.post("/queue/pause", response_model=MessageResponse)
async def queue_pause(
) -> MessageResponse:
    """Pause queue processing (index manager + video-visual worker)."""
    manager = get_index_manager()
    manager.pause()
    _pause_video_visual_worker()
    return MessageResponse(status="accepted", message="Queue paused")


@router.post("/queue/resume", response_model=MessageResponse)
async def queue_resume(
) -> MessageResponse:
    """Resume queue processing (index manager + video-visual worker)."""
    manager = get_index_manager()
    manager.resume()
    _resume_video_visual_worker()
    return MessageResponse(status="accepted", message="Queue resumed")
