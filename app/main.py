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
from app.database import init_litloft_db, init_search_db
from app.indexer import IndexManager
from app.llm import create_llm_client
from app.routers import (
    files,
    queue,
    rag,
    refine,
    search,
    similar,
    summaries,
    vision,
    webhooks,
)
from app.schemas import FeaturesStatus, LLMStatus, StatusResponse
from app.workers.auto_tags import AutoTagsWorker
from app.workers.summaries import SummariesWorker
from app.workers.vision import VisionDescribeWorker


async def _warm_up_auto_tag_candidates() -> None:
    """Preload CLIP concepts + TF-IDF corpus IDF in the background.

    On a cold container, the first auto-tag request pays ~15s for the
    CLIP model load plus ~6s for the corpus IDF build. Running these
    eagerly at startup keeps the first user-triggered tag generation
    fast. Deliberately silent on failures: if the CLIP weights are
    missing or the corpus is empty, auto_tags already falls back
    gracefully, and a warm-up crash shouldn't take down the service.
    """
    from app.tfidf import _get_corpus_idf
    from app.workers.clip_concepts import get_concept_embeddings

    logger.info("Auto-tag warm-up starting in background")
    try:
        # Heavy numpy / torch / SQL work — offload from the event loop.
        await asyncio.to_thread(get_concept_embeddings)
        await asyncio.to_thread(_get_corpus_idf)
    except Exception as e:
        logger.warning("Auto-tag warm-up failed (non-fatal): %s", e)
        return
    logger.info("Auto-tag warm-up complete")

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
        init_litloft_db()
        logger.info("Litloft database connected (read-only)")
    except FileNotFoundError:
        logger.warning(
            "Litloft database not found at %s. "
            "Service will start but indexing will be unavailable until DB is accessible.",
            settings.litloft_db_path,
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

    # Per-drive policy: drop everything indexed for drives whose
    # intelligence.index has been turned off in drives.json. Runs on
    # every startup so flipping the policy and restarting the addon is
    # the documented "purge" workflow. Failure inside one drive is
    # logged but does not block the rest of startup.
    try:
        from app.purge import purge_disabled_drives
        purged = await purge_disabled_drives()
        if purged:
            for drive, count in purged.items():
                logger.info(
                    "Purged %d files from drive '%s' "
                    "(intelligence.index disabled in drives.json)",
                    count, drive,
                )
    except Exception:
        logger.exception("Per-drive policy purge failed; continuing startup")

    # Per-feature policy: drop vision_describe artefacts on drives whose
    # vision_describe policy flipped off (umbrella stays on).
    try:
        from app.purge import purge_disabled_vision_drives
        vision_purged = await purge_disabled_vision_drives()
        if vision_purged:
            for drive, count in vision_purged.items():
                logger.info(
                    "Purged vision_describe data for %d files on drive '%s' "
                    "(vision_describe disabled in drives.json)",
                    count, drive,
                )
    except Exception:
        logger.exception("vision_describe purge failed; continuing startup")

    # Initialize LLM client and auto-tags worker
    llm_client = create_llm_client(settings.llm)
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

    # Auto-tags runs even when the LLM is disabled — the worker falls
    # back to CLIP zero-shot + TF-IDF candidate generation in that
    # case and produces "clip+tfidf" sourced suggestions.
    if settings.features.auto_tags != "false":
        auto_tags_task = asyncio.create_task(
            auto_tags_worker.run(), name="auto_tags_worker"
        )
        logger.info(
            "Auto-tags worker started (mode=%s, llm=%s)",
            settings.features.auto_tags, llm_client.enabled,
        )

        # on_index: queue already-indexed files that don't have suggested tags yet
        if settings.features.auto_tags == "on_index":
            pending = await auto_tags_worker.enqueue_unprocessed()
            if pending > 0:
                logger.info("Auto-tags: queued %d previously indexed files", pending)

    # Initialize summaries worker (shares the same LLM client)
    summaries_worker = SummariesWorker(llm_client)
    dependencies._summaries_worker = summaries_worker
    summaries_task: asyncio.Task | None = None

    summaries_active = settings.features.summaries != "false"
    detailed_on_index = settings.features.detailed_summaries == "on_index"
    if (summaries_active or detailed_on_index) and llm_client.enabled:
        summaries_task = asyncio.create_task(
            summaries_worker.run(), name="summaries_worker"
        )
        logger.info(
            "Summaries worker started (summaries=%s, detailed=%s)",
            settings.features.summaries,
            settings.features.detailed_summaries,
        )

        # on_index: queue already-indexed files that still need summary work.
        # Triggered when either the short/long path or the detailed path is
        # on_index; enqueue_unprocessed walks both gaps in the search DB.
        if settings.features.summaries == "on_index" or detailed_on_index:
            pending = await summaries_worker.enqueue_unprocessed()
            if pending > 0:
                logger.info(
                    "Summaries: queued %d previously indexed files", pending
                )

    # Initialize vision_describe worker. Starts only when the feature is
    # live (mode != false) AND a vision model is configured AND the LLM
    # client itself is enabled. Any of those missing → worker stays idle
    # (the routers already 404 in that state, so no requests reach it).
    vision_worker = VisionDescribeWorker()
    dependencies._vision_worker = vision_worker
    vision_task: asyncio.Task | None = None

    from app.config import is_vision_describe_available

    if is_vision_describe_available(settings) and llm_client.enabled:
        vision_task = asyncio.create_task(
            vision_worker.run(), name="vision_describe_worker"
        )
        logger.info(
            "Vision describe worker started (mode=%s, model=%s)",
            settings.features.vision_describe,
            settings.llm.vision_model,
        )

        # on_index: queue already-indexed images that don't have a
        # description yet. Without this, switching the feature on for
        # an existing library would only describe newly-indexed files
        # (the CLIP-completion hook in indexer.py only fires once per
        # file). Mirrors the auto_tags / summaries on_index sweep.
        if settings.features.vision_describe == "on_index":
            pending = await vision_worker.enqueue_unprocessed()
            if pending > 0:
                logger.info(
                    "Vision: queued %d previously indexed files", pending
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

    # Kick off auto-tag warm-up as a background task so the first user
    # request doesn't pay the CLIP / TF-IDF cold-start cost. Runs only
    # when auto_tags is enabled — otherwise the caches are dead weight.
    warm_up_task: asyncio.Task | None = None
    if settings.features.auto_tags != "false":
        warm_up_task = asyncio.create_task(
            _warm_up_auto_tag_candidates(), name="auto_tags_warmup"
        )

    logger.info("Semantic search service ready on port %d", settings.port)
    yield

    # Shutdown
    if warm_up_task is not None and not warm_up_task.done():
        warm_up_task.cancel()
        try:
            await warm_up_task
        except asyncio.CancelledError:
            pass
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
    if vision_task is not None:
        vision_task.cancel()
        try:
            await vision_task
        except asyncio.CancelledError:
            pass
    if dependencies._index_manager is not None:
        await dependencies._index_manager.stop()
    logger.info("Semantic search service stopped")


app = FastAPI(
    title="Litloft Semantic Search",
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
app.include_router(refine.router)
app.include_router(vision.router)


@app.get("/status", response_model=StatusResponse, tags=["status"])
async def status_endpoint() -> StatusResponse:
    """Get current service and indexing status."""
    manager = dependencies.get_index_manager()
    index_status = manager.get_index_status()
    queue_status = manager.get_queue_status()

    llm_client = dependencies._llm_client

    # Per-task breakdown for the dashboard. Index manager owns the four
    # core indexing types (metadata / clip / whisper / text_content).
    # The LLM workers (auto_tags / summaries / vision_describe) and the
    # refine module each track their own in-flight files so they can be
    # surfaced alongside.
    tasks = manager.get_queue_breakdown()
    if dependencies._auto_tags_worker is not None:
        tasks["auto_tags"] = dependencies._auto_tags_worker.get_status()
    if dependencies._summaries_worker is not None:
        tasks["summaries"] = dependencies._summaries_worker.get_status()
    if dependencies._vision_worker is not None:
        tasks["vision_describe"] = dependencies._vision_worker.get_status()
    try:
        from app.workers.refine import get_refine_status
        tasks["transcript_refine"] = get_refine_status()
    except Exception:  # noqa: BLE001 — refine surface is non-critical
        pass

    # Resolve filenames for files currently being processed so the
    # dashboard can show "now: 文字起こし of foo.mp4". We hit the search
    # DB once with all unique ids to avoid an N+1.
    all_active_ids = {
        fid
        for entry in tasks.values()
        for fid in entry.get("processing", [])  # type: ignore[union-attr]
    }
    filenames: dict[str, str] = {}
    if all_active_ids:
        from app.database import get_search_db
        from app.models import IndexedFile

        with get_search_db() as session:
            rows = (
                session.query(IndexedFile.file_id, IndexedFile.filename)
                .filter(IndexedFile.file_id.in_(all_active_ids))
                .all()
            )
            filenames = {row[0]: row[1] for row in rows}

    # Annotate each processing entry with a {file_id, filename} pair so
    # the frontend doesn't need to re-query for names. Filename is
    # nullable — purged or not-yet-indexed ids fall through.
    for entry in tasks.values():
        raw_ids: list[str] = list(entry.get("processing", []))  # type: ignore[arg-type]
        entry["processing"] = [
            {"file_id": fid, "filename": filenames.get(fid)}
            for fid in raw_ids
        ]

    return StatusResponse(
        status="running",
        indexed={
            "total": index_status.total_indexed,
            "metadata": index_status.metadata_indexed,
            "clip": index_status.clip_indexed,
            "whisper": index_status.whisper_indexed,
            "text": index_status.text_indexed,
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
            "text": index_status.pending_text,
        },
        queue={
            "processing": queue_status.processing_count,
            "waiting": queue_status.waiting_count,
            "paused": queue_status.state == "paused",
            "tasks": tasks,
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
            transcript_refine=settings.features.transcript_refine,
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
