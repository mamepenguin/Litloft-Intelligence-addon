"""FastAPI entry point for the semantic search service.

Provides search, status, webhook, and queue control endpoints.
Initializes databases and starts the background indexing pipeline
on application startup.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

import subprocess

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.config import settings
from app.database import init_homevault_db, init_search_db
from app.indexer import IndexManager
from app.llm import LLMClient
from app.workers.auto_tags import AutoTagsWorker
from app.search import execute_search_compare, find_similar, search as execute_search
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

# Module-level index manager and auto-tags worker (initialized during lifespan)
_index_manager: IndexManager | None = None
_auto_tags_worker: AutoTagsWorker | None = None
_llm_client: LLMClient | None = None


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

    # Clean up orphaned data from potential crash during previous run
    from app.indexer import cleanup_orphaned_embeddings, reset_falsely_completed_clip
    cleaned = cleanup_orphaned_embeddings()
    if cleaned > 0:
        logger.info("Cleaned up %d orphaned embeddings from previous run", cleaned)

    # Reset files marked as clip_indexed but missing actual vectors
    reset = reset_falsely_completed_clip()
    if reset > 0:
        logger.info("Reset %d falsely completed CLIP files for re-indexing", reset)

    # Initialize LLM client and auto-tags worker
    global _llm_client, _auto_tags_worker

    _llm_client = LLMClient(settings.llm)
    if _llm_client.enabled:
        logger.info(
            "LLM client enabled: provider=%s, model=%s",
            settings.llm.provider, settings.llm.model,
        )
    else:
        logger.info("LLM client disabled")

    _auto_tags_worker = AutoTagsWorker(_llm_client)
    auto_tags_task: asyncio.Task | None = None

    if settings.features.auto_tags != "false" and _llm_client.enabled:
        auto_tags_task = asyncio.create_task(
            _auto_tags_worker.run(), name="auto_tags_worker"
        )
        logger.info("Auto-tags worker started (mode=%s)", settings.features.auto_tags)

        # on_index: queue already-indexed files that don't have suggested tags yet
        if settings.features.auto_tags == "on_index":
            pending = await _auto_tags_worker.enqueue_unprocessed()
            if pending > 0:
                logger.info("Auto-tags: queued %d previously indexed files", pending)

    # Start index manager (pass auto_tags_worker for post-metadata hook)
    _index_manager = IndexManager(auto_tags_worker=_auto_tags_worker)
    try:
        await _index_manager.start()
    except Exception as e:
        logger.error("Failed to start index manager: %s", e)

    logger.info("Semantic search service ready on port %d", settings.port)
    yield

    # Shutdown
    if auto_tags_task is not None:
        auto_tags_task.cancel()
        try:
            await auto_tags_task
        except asyncio.CancelledError:
            pass
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
    page: int | None = None


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


class FeaturesStatus(BaseModel):
    indexing: bool
    search: bool
    auto_tags: str


class LLMStatus(BaseModel):
    provider: str
    model: str
    enabled: bool


class StatusResponse(BaseModel):
    status: str
    indexed: dict[str, int]
    pending: dict[str, int]
    queue: dict[str, Any]
    models: dict[str, str]
    features: FeaturesStatus
    llm: LLMStatus


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


# --- Debug search endpoint ---


@app.get("/debug/search")
async def debug_search_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
) -> dict:
    """Debug search: returns raw scores from each search system."""
    from app.debug import debug_search
    result = debug_search(q)
    return result.model_dump()


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
                                page=m.page,
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


# --- Compare endpoint (temporary: side-by-side RRF vs cosine) ---


class SourceCountsModel(BaseModel):
    text_vector: int
    clip_vector: int
    keyword: int
    transcript_keyword: int
    text_content_keyword: int = 0


class CompareResponseModel(BaseModel):
    rrf: SearchResponseModel
    cosine: SearchResponseModel
    rrf_no_cutoff: SearchResponseModel
    cosine_no_cutoff: SearchResponseModel
    source_counts: SourceCountsModel


def _to_response_model(result: Any) -> SearchResponseModel:
    from app.search import SearchResponse as _SR
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
                                page=m.page,
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


@app.get("/search/compare", response_model=CompareResponseModel)
async def search_compare_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
    type: str | None = Query(default=None, description="File type filter"),
    drive: str | None = Query(default=None, description="Drive name filter"),
) -> CompareResponseModel:
    """Compare RRF vs cosine-similarity scoring side by side."""
    try:
        compare = execute_search_compare(
            query=q, limit=limit, file_type=type, drive=drive,
        )
    except Exception as e:
        logger.error("Compare search failed: %s", e)
        raise HTTPException(status_code=500, detail="Search failed") from e

    return CompareResponseModel(
        rrf=_to_response_model(compare.rrf),
        cosine=_to_response_model(compare.cosine),
        rrf_no_cutoff=_to_response_model(compare.rrf_no_cutoff),
        cosine_no_cutoff=_to_response_model(compare.cosine_no_cutoff),
        source_counts=SourceCountsModel(
            text_vector=compare.source_counts.text_vector,
            clip_vector=compare.source_counts.clip_vector,
            keyword=compare.source_counts.keyword,
            transcript_keyword=compare.source_counts.transcript_keyword,
            text_content_keyword=compare.source_counts.text_content_keyword,
        ),
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
                index_status.pending_metadata
                + index_status.pending_clip
                + index_status.pending_whisper
                + index_status.pending_text
            ),
            "metadata": index_status.pending_metadata,
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
        features=FeaturesStatus(
            indexing=settings.features.indexing,
            search=settings.features.search,
            auto_tags=settings.features.auto_tags,
        ),
        llm=LLMStatus(
            provider=settings.llm.provider,
            model=settings.llm.model,
            enabled=_llm_client.enabled if _llm_client else False,
        ),
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


# --- Similar files endpoint ---


class KeywordScore(BaseModel):
    word: str
    score: float | None = None
    source_tfidf: float | None = None
    target_tfidf: float | None = None
    relevance: float | None = None


class SimilarFileItem(BaseModel):
    file_id: str
    drive: str
    filename: str
    file_type: str
    mime_type: str
    score: float
    match_type: str
    primary_score: float | None = None
    secondary_score: float | None = None
    shared_keywords: list[KeywordScore] = []


class SimilarFilesResponse(BaseModel):
    results: list[SimilarFileItem]
    source_keywords: list[KeywordScore] = []


@app.get("/similar/{file_id}", response_model=SimilarFilesResponse)
async def similar_files_endpoint(
    file_id: str,
    limit: int = Query(default=6, ge=1, le=20),
    drive: str | None = Query(default=None),
) -> SimilarFilesResponse:
    """Find files similar to the given file using existing embeddings."""
    try:
        search_result = find_similar(file_id=file_id, limit=limit, drive=drive)
    except Exception as e:
        logger.error("Similar files search failed: %s", e)
        raise HTTPException(status_code=500, detail="Similar search failed") from e

    return SimilarFilesResponse(
        results=[
            SimilarFileItem(
                file_id=r.file_id,
                drive=r.drive,
                filename=r.filename,
                file_type=r.file_type,
                mime_type=r.mime_type,
                score=round(r.score, 4),
                match_type=r.match_type,
                primary_score=round(r.primary_score, 4) if r.primary_score is not None else None,
                secondary_score=round(r.secondary_score, 4) if r.secondary_score is not None else None,
                shared_keywords=[
                    KeywordScore(**kw) for kw in r.shared_keywords
                ],
            )
            for r in search_result.results
        ],
        source_keywords=[
            KeywordScore(**kw) for kw in search_result.source_keywords
        ],
    )


@app.get("/debug/similar/{file_id}")
async def debug_similar_endpoint(
    file_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    drive: str | None = Query(default=None),
) -> dict:
    """Debug similar files: returns raw scores from each embedding type."""
    from app.search import (
        _find_similar_by_embedding,
        _select_embedding_types,
    )
    from app.database import get_search_db
    from app.models import IndexedFile

    with get_search_db() as session:
        source = (
            session.query(IndexedFile)
            .filter(
                IndexedFile.file_id == file_id,
                IndexedFile.active.is_(True),
            )
            .first()
        )
        if not source:
            raise HTTPException(status_code=404, detail="File not indexed")

        source_info = {
            "file_id": source.file_id,
            "filename": source.filename,
            "file_type": source.file_type,
            "drive": source.drive,
        }

    primary_type, fallback_type = _select_embedding_types(source_info["file_type"])

    all_types = ["clip", "metadata", "whisper", "text_content"]
    results_by_type: dict[str, list[dict]] = {}

    for etype in all_types:
        raw = _find_similar_by_embedding(file_id, etype, limit, drive)
        results_by_type[etype] = [
            {
                "file_id": r["file_id"],
                "filename": r["filename"],
                "file_type": r["file_type"],
                "score": round(r["score"], 6),
            }
            for r in raw
        ]

    return {
        "source": source_info,
        "primary_type": primary_type,
        "fallback_type": fallback_type,
        "results_by_type": results_by_type,
    }


# --- File inspection endpoints ---


class TranscriptChunkResponse(BaseModel):
    index: int
    text: str
    start: float
    end: float


class TranscriptResponse(BaseModel):
    file_id: str
    drive: str
    language: str
    chunks: list[TranscriptChunkResponse]


class IndexDetailEmbeddingItem(BaseModel):
    content_preview: str
    start: float | None = None
    end: float | None = None


class IndexDetailType(BaseModel):
    count: int
    items: list[IndexDetailEmbeddingItem]


class IndexDetailsResponse(BaseModel):
    file_id: str
    drive: str
    filename: str
    status: dict[str, bool]
    indexed_at: str
    embeddings: dict[str, IndexDetailType]


class ClipTimestampItem(BaseModel):
    start: float
    content_preview: str


class ClipTimestampsResponse(BaseModel):
    file_id: str
    drive: str
    timestamps: list[ClipTimestampItem]


def _get_indexed_file_or_404(file_id: str) -> Any:
    """Get an indexed file by ID or raise 404."""
    from app.database import get_search_db
    from app.models import IndexedFile

    with get_search_db() as db:
        indexed = db.query(IndexedFile).filter(
            IndexedFile.file_id == file_id,
            IndexedFile.active.is_(True),
        ).first()
        if not indexed:
            raise HTTPException(status_code=404, detail="File not indexed")
        # Detach from session by copying attributes
        return {
            "file_id": indexed.file_id,
            "drive": indexed.drive,
            "filename": indexed.filename,
            "file_path": indexed.file_path,
            "file_type": indexed.file_type,
            "metadata_indexed": indexed.metadata_indexed,
            "clip_indexed": indexed.clip_indexed,
            "whisper_indexed": indexed.whisper_indexed,
            "text_indexed": indexed.text_indexed,
            "indexed_at": indexed.indexed_at.isoformat() if indexed.indexed_at else "",
        }


@app.get("/files/{file_id}/transcript", response_model=TranscriptResponse)
async def get_transcript(file_id: str) -> TranscriptResponse:
    """Get Whisper transcript chunks for a file."""
    from app.database import get_search_db
    from app.models import TranscriptChunk

    indexed = _get_indexed_file_or_404(file_id)

    with get_search_db() as db:
        chunks = (
            db.query(TranscriptChunk)
            .filter(TranscriptChunk.file_id == file_id)
            .order_by(TranscriptChunk.chunk_index)
            .all()
        )

    if not chunks:
        raise HTTPException(status_code=404, detail="No transcript available")

    language = chunks[0].language if chunks else ""

    return TranscriptResponse(
        file_id=file_id,
        drive=indexed["drive"],
        language=language,
        chunks=[
            TranscriptChunkResponse(
                index=c.chunk_index,
                text=c.text,
                start=c.timestamp_start,
                end=c.timestamp_end,
            )
            for c in chunks
        ],
    )


_EMBEDDING_TYPES = ("metadata", "clip", "whisper", "text_content", "blip_caption")
_ITEMS_PER_TYPE = 50


@app.get("/files/{file_id}/index-details", response_model=IndexDetailsResponse)
async def get_index_details(file_id: str) -> IndexDetailsResponse:
    """Get detailed indexing status and embedding info for a file."""
    from sqlalchemy import func
    from app.database import get_search_db
    from app.models import Embedding

    indexed = _get_indexed_file_or_404(file_id)

    embeddings_by_type: dict[str, IndexDetailType] = {}

    with get_search_db() as db:
        for etype in _EMBEDDING_TYPES:
            count = (
                db.query(func.count(Embedding.id))
                .filter(
                    Embedding.file_id == file_id,
                    Embedding.embedding_type == etype,
                )
                .scalar()
            ) or 0

            items_query = (
                db.query(Embedding)
                .filter(
                    Embedding.file_id == file_id,
                    Embedding.embedding_type == etype,
                )
                .order_by(Embedding.timestamp_start.asc().nullsfirst())
                .limit(_ITEMS_PER_TYPE)
                .all()
            )

            embeddings_by_type[etype] = IndexDetailType(
                count=count,
                items=[
                    IndexDetailEmbeddingItem(
                        content_preview=e.content_preview,
                        start=e.timestamp_start,
                        end=e.timestamp_end,
                    )
                    for e in items_query
                ],
            )

    return IndexDetailsResponse(
        file_id=file_id,
        drive=indexed["drive"],
        filename=indexed["filename"],
        status={
            "metadata": indexed["metadata_indexed"],
            "clip": indexed["clip_indexed"],
            "whisper": indexed["whisper_indexed"],
            "text": indexed["text_indexed"],
        },
        indexed_at=indexed["indexed_at"],
        embeddings=embeddings_by_type,
    )


@app.get("/files/{file_id}/clip-timestamps", response_model=ClipTimestampsResponse)
async def get_clip_timestamps(file_id: str) -> ClipTimestampsResponse:
    """Get CLIP frame extraction timestamps for a file."""
    from app.database import get_search_db
    from app.models import Embedding

    indexed = _get_indexed_file_or_404(file_id)

    with get_search_db() as db:
        clips = (
            db.query(Embedding)
            .filter(
                Embedding.file_id == file_id,
                Embedding.embedding_type == "clip",
            )
            .order_by(Embedding.timestamp_start.asc())
            .all()
        )

    return ClipTimestampsResponse(
        file_id=file_id,
        drive=indexed["drive"],
        timestamps=[
            ClipTimestampItem(
                start=c.timestamp_start or 0.0,
                content_preview=c.content_preview,
            )
            for c in clips
        ],
    )


@app.get("/files/{file_id}/frame")
async def get_frame(
    file_id: str,
    t: float = Query(..., ge=0, description="Timestamp in seconds"),
) -> Response:
    """Extract a single video frame at the given timestamp using ffmpeg."""
    from app.config import resolve_file_path, validate_file_path

    indexed = _get_indexed_file_or_404(file_id)

    if indexed["file_type"] != "video":
        raise HTTPException(status_code=400, detail="Not a video file")

    abs_path = resolve_file_path(indexed["drive"], indexed["file_path"])
    if not abs_path or not validate_file_path(abs_path):
        raise HTTPException(status_code=404, detail="Video file not accessible")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-ss", str(t),
                "-i", abs_path,
                "-frames:v", "1",
                "-vf", "scale=480:-1",
                "-f", "image2pipe",
                "-vcodec", "mjpeg",
                "-q:v", "3",
                "-",
            ],
            capture_output=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Frame extraction timed out")

    if result.returncode != 0 or not result.stdout:
        logger.warning("Frame extraction failed for %s at %.1fs", file_id, t)
        raise HTTPException(status_code=500, detail="Frame extraction failed")

    return Response(
        content=result.stdout,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# --- Suggested tags endpoints ---


class SuggestedTagsResponse(BaseModel):
    available: bool
    file_id: str | None = None
    tags: list[str] | None = None
    model: str | None = None
    status: str | None = None
    created_at: str | None = None


@app.get("/files/{file_id}/suggested-tags", response_model=SuggestedTagsResponse)
async def get_suggested_tags(file_id: str) -> SuggestedTagsResponse:
    """Get suggested tags for a file."""
    import json as json_mod
    from app.database import get_search_db
    from sqlalchemy import text as sql_text

    if settings.features.auto_tags == "false":
        return SuggestedTagsResponse(available=False)

    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT file_id, tags, model, status, created_at "
                "FROM suggested_tags WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()

    if row is None:
        return SuggestedTagsResponse(available=False)

    try:
        tags = json_mod.loads(row[1])
    except (json_mod.JSONDecodeError, TypeError):
        tags = []

    return SuggestedTagsResponse(
        available=True,
        file_id=row[0],
        tags=tags,
        model=row[2],
        status=row[3],
        created_at=row[4],
    )


@app.post("/files/{file_id}/suggested-tags/dismiss", response_model=MessageResponse)
async def dismiss_suggested_tags(file_id: str) -> MessageResponse:
    """Dismiss suggested tags for a file."""
    from app.database import get_search_db
    from sqlalchemy import text as sql_text

    with get_search_db() as session:
        result = session.execute(
            sql_text(
                "UPDATE suggested_tags SET status = 'dismissed' "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        )

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="No suggested tags found")

    return MessageResponse(status="ok", message="Suggested tags dismissed")


@app.post("/files/{file_id}/suggested-tags/regenerate", response_model=MessageResponse)
async def regenerate_suggested_tags(file_id: str) -> MessageResponse:
    """Delete existing suggested tags and re-queue for auto-tagging."""
    from app.database import get_search_db
    from sqlalchemy import text as sql_text

    if settings.features.auto_tags == "false":
        raise HTTPException(status_code=400, detail="Auto-tags feature is disabled")

    if _auto_tags_worker is None or _llm_client is None or not _llm_client.enabled:
        raise HTTPException(status_code=400, detail="LLM is not enabled")

    # Delete existing entry
    with get_search_db() as session:
        session.execute(
            sql_text("DELETE FROM suggested_tags WHERE file_id = :fid"),
            {"fid": file_id},
        )

    # Re-queue
    await _auto_tags_worker.enqueue(file_id)

    return MessageResponse(
        status="accepted", message="Regeneration queued"
    )


class BatchSuggestedTagsRequest(BaseModel):
    file_ids: list[str] = Field(..., max_length=500)


class BatchSuggestedTagsResponse(BaseModel):
    queued: int
    skipped: int


@app.post("/batch/suggested-tags", response_model=BatchSuggestedTagsResponse)
async def batch_suggested_tags(body: BatchSuggestedTagsRequest) -> BatchSuggestedTagsResponse:
    """Queue auto-tagging for a batch of files. Skips files that already have suggestions."""
    from app.database import get_search_db
    from sqlalchemy import text as sql_text

    if settings.features.auto_tags == "false":
        raise HTTPException(status_code=400, detail="Auto-tags feature is disabled")

    if _auto_tags_worker is None or _llm_client is None or not _llm_client.enabled:
        raise HTTPException(status_code=400, detail="LLM is not enabled")

    # Find which files already have suggested tags
    with get_search_db() as session:
        existing: set[str] = set()
        if body.file_ids:
            placeholders = ",".join(f":id{i}" for i in range(len(body.file_ids)))
            params = {f"id{i}": fid for i, fid in enumerate(body.file_ids)}
            for row in session.execute(
                sql_text(f"SELECT file_id FROM suggested_tags WHERE file_id IN ({placeholders})"),
                params,
            ).fetchall():
                existing.add(row[0])

    queued = 0
    skipped = 0
    for file_id in body.file_ids:
        if file_id in existing:
            skipped += 1
        else:
            await _auto_tags_worker.enqueue(file_id)
            queued += 1

    return BatchSuggestedTagsResponse(queued=queued, skipped=skipped)


# --- Health check ---


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "ok", "version": settings.service_version}
