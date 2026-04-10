"""FastAPI entry point for the semantic search service.

Provides search, status, webhook, and queue control endpoints.
Initializes databases and starts the background indexing pipeline
on application startup.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from app import dependencies
from app.config import settings
from app.database import init_homevault_db, init_search_db
from app.indexer import IndexManager
from app.llm import LLMClient
from app.routers import files, queue, rag, search, similar, summaries, webhooks
from app.schemas import FeaturesStatus, LLMStatus, StatusResponse
from app.workers.auto_tags import AutoTagsWorker
from app.workers.summaries import SummariesWorker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize databases and start indexer."""
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
    llm_client = LLMClient(settings.llm)
    dependencies._llm_client = llm_client

    if llm_client.enabled:
        logger.info(
            "LLM client enabled: provider=%s, model=%s",
            settings.llm.provider, settings.llm.model,
        )
    else:
        logger.info("LLM client disabled")

    auto_tags_worker = AutoTagsWorker(llm_client)
    dependencies._auto_tags_worker = auto_tags_worker
    auto_tags_task: asyncio.Task | None = None

    if settings.features.auto_tags != "false" and llm_client.enabled:
        auto_tags_task = asyncio.create_task(
            auto_tags_worker.run(), name="auto_tags_worker"
        )
        logger.info("Auto-tags worker started (mode=%s)", settings.features.auto_tags)

        # on_index: queue already-indexed files that don't have suggested tags yet
        if settings.features.auto_tags == "on_index":
            pending = await auto_tags_worker.enqueue_unprocessed()
            if pending > 0:
                logger.info("Auto-tags: queued %d previously indexed files", pending)

    # Initialize summaries worker (shares the same LLM client)
    summaries_worker = SummariesWorker(llm_client)
    dependencies._summaries_worker = summaries_worker
    summaries_task: asyncio.Task | None = None

    if settings.features.summaries != "false" and llm_client.enabled:
        summaries_task = asyncio.create_task(
            summaries_worker.run(), name="summaries_worker"
        )
        logger.info(
            "Summaries worker started (mode=%s)", settings.features.summaries
        )

        # on_index: queue already-indexed files that don't have summaries yet
        if settings.features.summaries == "on_index":
            pending = await summaries_worker.enqueue_unprocessed()
            if pending > 0:
                logger.info(
                    "Summaries: queued %d previously indexed files", pending
                )

    # Start index manager (pass workers for post-metadata hooks)
    index_manager = IndexManager(
        auto_tags_worker=auto_tags_worker,
        summaries_worker=summaries_worker,
    )
    dependencies._index_manager = index_manager
    try:
        await index_manager.start()
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
    if summaries_task is not None:
        summaries_task.cancel()
        try:
            await summaries_task
        except asyncio.CancelledError:
            pass
    if dependencies._index_manager is not None:
        await dependencies._index_manager.stop()
    logger.info("Semantic search service stopped")


app = FastAPI(
    title="HomeVault Semantic Search",
    version=settings.service_version,
    lifespan=lifespan,
)

# Register routers (no prefixes — paths are fully qualified in each router)
app.include_router(search.router)
app.include_router(webhooks.router)
app.include_router(queue.router)
app.include_router(similar.router)
app.include_router(files.router)
app.include_router(summaries.router)
app.include_router(rag.router)


@app.get("/status", response_model=StatusResponse, tags=["status"])
async def status_endpoint() -> StatusResponse:
    """Get current service and indexing status."""
    manager = dependencies.get_index_manager()
    index_status = manager.get_index_status()
    queue_status = manager.get_queue_status()

    llm_client = dependencies._llm_client

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
            summaries=settings.features.summaries,
            rag=settings.features.rag,
        ),
        llm=LLMStatus(
            provider=settings.llm.provider,
            model=settings.llm.model,
            enabled=llm_client.enabled if llm_client else False,
            output_language=settings.llm.output_language,
        ),
    )


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "ok", "version": settings.service_version}
