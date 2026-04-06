"""FastAPI entry point for the semantic search service.

Provides search, status, webhook, and queue control endpoints.
Initializes databases and starts the background indexing pipeline
on application startup.
"""

import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.config import settings
from app.database import init_homevault_db, init_search_db
from app.indexer import IndexManager
from app.search import search as execute_search
from app.webhook import (
    FilesDeletedPayload,
    FilesPurgedPayload,
    FilesRestoredPayload,
    PrioritizePayload,
    ScanCompletePayload,
    handle_files_deleted,
    handle_files_purged,
    handle_files_restored,
    handle_scan_complete,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Module-level index manager (initialized during lifespan)
_index_manager: IndexManager | None = None


def _get_index_manager() -> IndexManager:
    """Get the index manager instance.

    Raises:
        RuntimeError: If the manager is not initialized.
    """
    if _index_manager is None:
        raise RuntimeError("Index manager not initialized")
    return _index_manager


_WEBHOOK_SECRET = os.environ.get("SEARCH_WEBHOOK_SECRET", "")


async def verify_webhook_secret(
    x_webhook_secret: str = Header(default=""),
) -> None:
    """Verify webhook secret if configured."""
    if _WEBHOOK_SECRET and x_webhook_secret != _WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize databases and start indexer."""
    global _index_manager

    logger.info("Semantic search service starting (v%s)", settings.service_version)

    # Initialize databases
    init_search_db()
    logger.info("Search database initialized")

    try:
        init_homevault_db()
        logger.info("HomeVault database connected (read-only)")
    except FileNotFoundError:
        logger.warning(
            "HomeVault database not found at %s. "
            "Service will start but indexing will be unavailable until DB is accessible.",
            settings.homevault_db_path,
        )

    # Start index manager
    _index_manager = IndexManager()
    try:
        await _index_manager.start()
    except Exception as e:
        logger.error("Failed to start index manager: %s", e)

    logger.info("Semantic search service ready on port %d", settings.port)
    yield

    # Shutdown
    if _index_manager is not None:
        await _index_manager.stop()
    logger.info("Semantic search service stopped")


app = FastAPI(
    title="HomeVault Semantic Search",
    version=settings.service_version,
    lifespan=lifespan,
)


# --- Pydantic models for request/response ---


class SearchResultSegmentMatch(BaseModel):
    type: str
    text: str
    score: float


class SearchResultSegment(BaseModel):
    time_range: list[float] | None = None
    matches: list[SearchResultSegmentMatch]


class SearchResultItem(BaseModel):
    file_id: str
    drive: str
    filename: str
    file_type: str
    score: float
    match_types: list[str]
    segments: list[SearchResultSegment]


class SearchResponseModel(BaseModel):
    results: list[SearchResultItem]
    total: int
    indexed_files: int
    service_version: str


class StatusResponse(BaseModel):
    status: str
    indexed: dict[str, int]
    pending: dict[str, int]
    queue: dict[str, Any]
    models: dict[str, str]


class WebhookScanComplete(BaseModel):
    drive: str
    added: int = 0
    removed: int = 0


class WebhookFilesDeleted(BaseModel):
    file_ids: list[str] = Field(..., max_length=10000)
    type: str = "soft_delete"


class WebhookFilesRestored(BaseModel):
    file_ids: list[str] = Field(..., max_length=10000)


class WebhookFilesPurged(BaseModel):
    file_ids: list[str] = Field(..., max_length=10000)


class QueuePrioritize(BaseModel):
    file_id: str


class MessageResponse(BaseModel):
    status: str
    message: str


# --- Search endpoint ---


@app.get("/search", response_model=SearchResponseModel)
async def search_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
    type: str | None = Query(default=None, description="File type filter"),
    drive: str | None = Query(default=None, description="Drive name filter"),
) -> SearchResponseModel:
    """Execute a semantic search query.

    Combines vector similarity search with keyword matching
    to find relevant files across all indexed content.
    """
    try:
        result = execute_search(
            query=q, limit=limit, file_type=type, drive=drive
        )
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise HTTPException(status_code=500, detail="Search failed") from e

    return SearchResponseModel(
        results=[
            SearchResultItem(
                file_id=r.file_id,
                drive=r.drive,
                filename=r.filename,
                file_type=r.file_type,
                score=round(r.score, 4),
                match_types=list(r.match_types),
                segments=[
                    SearchResultSegment(
                        time_range=(
                            list(s.time_range) if s.time_range else None
                        ),
                        matches=[
                            SearchResultSegmentMatch(
                                type=m.match_type,
                                text=m.text,
                                score=round(m.score, 4),
                            )
                            for m in s.matches
                        ],
                    )
                    for s in r.segments
                ],
            )
            for r in result.results
        ],
        total=result.total,
        indexed_files=result.indexed_files,
        service_version=result.service_version,
    )


# --- Status endpoint ---


@app.get("/status", response_model=StatusResponse)
async def status_endpoint() -> StatusResponse:
    """Get current service and indexing status."""
    manager = _get_index_manager()
    index_status = manager.get_index_status()
    queue_status = manager.get_queue_status()

    return StatusResponse(
        status="running",
        indexed={
            "total": index_status.total_indexed,
            "metadata": index_status.metadata_indexed,
            "clip": index_status.clip_indexed,
            "whisper": index_status.whisper_indexed,
        },
        pending={
            "total": (
                index_status.pending_clip
                + index_status.pending_whisper
                + index_status.pending_text
            ),
            "clip": index_status.pending_clip,
            "whisper": index_status.pending_whisper,
        },
        queue={
            "processing": queue_status.processing_count,
            "waiting": queue_status.waiting_count,
            "paused": queue_status.state == "paused",
        },
        models={
            "whisper": settings.models.whisper,
            "clip": settings.models.clip,
            "text_embedding": settings.models.text_embedding,
        },
    )


# --- Webhook endpoints ---


@app.post("/webhook/scan-complete", response_model=MessageResponse)
async def webhook_scan_complete(
    body: WebhookScanComplete,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle scan-complete webhook from HomeVault."""
    manager = _get_index_manager()
    payload = ScanCompletePayload(
        drive=body.drive, added=body.added, removed=body.removed
    )
    result = await handle_scan_complete(payload, manager)
    return MessageResponse(**result)


@app.post("/webhook/files-deleted", response_model=MessageResponse)
async def webhook_files_deleted(
    body: WebhookFilesDeleted,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle files-deleted webhook from HomeVault."""
    manager = _get_index_manager()
    payload = FilesDeletedPayload(
        file_ids=tuple(body.file_ids), type=body.type
    )
    result = await handle_files_deleted(payload, manager)
    return MessageResponse(**result)


@app.post("/webhook/files-restored", response_model=MessageResponse)
async def webhook_files_restored(
    body: WebhookFilesRestored,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle files-restored webhook from HomeVault."""
    manager = _get_index_manager()
    payload = FilesRestoredPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_restored(payload, manager)
    return MessageResponse(**result)


@app.post("/webhook/files-purged", response_model=MessageResponse)
async def webhook_files_purged(
    body: WebhookFilesPurged,
    _: None = Depends(verify_webhook_secret),
) -> MessageResponse:
    """Handle files-purged webhook from HomeVault."""
    manager = _get_index_manager()
    payload = FilesPurgedPayload(file_ids=tuple(body.file_ids))
    result = await handle_files_purged(payload, manager)
    return MessageResponse(**result)


# --- Queue control endpoints ---


@app.post("/queue/prioritize", response_model=MessageResponse)
async def queue_prioritize(body: QueuePrioritize, _: None = Depends(verify_webhook_secret)) -> MessageResponse:
    """Prioritize a specific file for immediate indexing."""
    manager = _get_index_manager()
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


@app.post("/queue/pause", response_model=MessageResponse)
async def queue_pause(_: None = Depends(verify_webhook_secret)) -> MessageResponse:
    """Pause queue processing."""
    manager = _get_index_manager()
    manager.pause()
    return MessageResponse(status="accepted", message="Queue paused")


@app.post("/queue/resume", response_model=MessageResponse)
async def queue_resume(_: None = Depends(verify_webhook_secret)) -> MessageResponse:
    """Resume queue processing."""
    manager = _get_index_manager()
    manager.resume()
    return MessageResponse(status="accepted", message="Queue resumed")


@app.post("/queue/reindex", response_model=MessageResponse)
async def queue_reindex(_: None = Depends(verify_webhook_secret)) -> MessageResponse:
    """Trigger a full reindex of all files."""
    manager = _get_index_manager()
    await manager.reindex_all()
    return MessageResponse(
        status="accepted", message="Full reindex initiated"
    )


# --- Health check ---


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "ok", "version": settings.service_version}
