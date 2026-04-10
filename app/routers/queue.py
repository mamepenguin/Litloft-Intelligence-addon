"""Queue control endpoints: pause, resume, reindex, prioritize."""

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_index_manager, verify_webhook_secret
from app.schemas import MessageResponse, QueuePrioritize

router = APIRouter(tags=["queue"])


@router.post("/queue/prioritize", response_model=MessageResponse)
async def queue_prioritize(
    body: QueuePrioritize,
    _: None = Depends(verify_webhook_secret),
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


@router.post("/queue/pause", response_model=MessageResponse)
async def queue_pause(
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Pause queue processing."""
    manager = get_index_manager()
    manager.pause()
    return MessageResponse(status="accepted", message="Queue paused")


@router.post("/queue/resume", response_model=MessageResponse)
async def queue_resume(
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Resume queue processing."""
    manager = get_index_manager()
    manager.resume()
    return MessageResponse(status="accepted", message="Queue resumed")


@router.post("/queue/reindex", response_model=MessageResponse)
async def queue_reindex(
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Trigger a full reindex of all files."""
    manager = get_index_manager()
    await manager.reindex_all()
    return MessageResponse(
        status="accepted", message="Full reindex initiated"
    )
