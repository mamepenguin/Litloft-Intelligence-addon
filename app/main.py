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
from app.loop_watchdog import LoopWatchdog
from app.routers import (
    admin,
    chapter_suggestions,
    files,
    pickup,
    queue,
    rag,
    refine,
    search,
    similar,
    summaries,
    video_visual,
    vision,
    webhooks,
)
from app.schemas import FeaturesStatus, LLMStatus, StatusResponse
from app.workers.auto_tags import AutoTagsWorker
from app.workers.chapter_suggestions import ChapterSuggestionsWorker
from app.workers.pickup import PickupWorker
from app.workers.retrieval_keywords import RetrievalKeywordsWorker
from app.workers.summaries import SummariesWorker
from app.workers.video_visual import VideoVisualWorker
from app.workers.vision import VisionDescribeWorker


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

    # Started first so it covers startup too: a stall during DB init or
    # the first reconcile is exactly as invisible as one at steady state.
    watchdog = LoopWatchdog(asyncio.get_running_loop())
    watchdog.start()

    # The webhook gate fails closed once configured, and it needs the other
    # end to agree: core only sends X-Webhook-Secret when the listener in
    # event-hooks.json declares secret_env. If the secret is set here but
    # not declared there, every webhook returns 403 and indexing stops with
    # no other symptom — so say which mode we are in, out loud, once.
    from app import dependencies as _deps

    if _deps._WEBHOOK_SECRET:
        logger.info(
            "Webhook secret gate ACTIVE. Core must declare "
            '"secret_env": "SEARCH_WEBHOOK_SECRET" for each intelligence '
            "listener in event-hooks.json, or webhooks will 403."
        )
    else:
        logger.info(
            "Webhook secret gate inactive (SEARCH_WEBHOOK_SECRET unset); "
            "webhook endpoints accept any Docker-network caller."
        )

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
    from app.indexer import (
        cleanup_orphaned_embeddings,
        reset_falsely_completed_clip,
        reset_falsely_completed_clip_thumbnail,
    )
    cleaned = cleanup_orphaned_embeddings()
    if cleaned > 0:
        logger.info("Cleaned up %d orphaned embeddings from previous run", cleaned)

    # Reset files marked as clip_indexed but missing actual vectors
    reset = reset_falsely_completed_clip()
    if reset > 0:
        logger.info("Reset %d falsely completed CLIP files for re-indexing", reset)

    # The thumbnail leg is tracked separately and a video's scene rows
    # hide its absence from the check above.
    reset_thumb = reset_falsely_completed_clip_thumbnail()
    if reset_thumb:
        logger.info(
            "Reopened %d thumbnail CLIP legs for re-indexing",
            len(reset_thumb),
        )

    from app.workers.clip import warn_if_thumbnails_unreachable
    warn_if_thumbnails_unreachable()

    # Video visual index: stale "running" scenes/runs left by a container
    # restart return to a resumable state (design doc §9).
    try:
        from app.workers.video_visual import recover_on_startup as recover_video_visual
        recover_video_visual()
    except Exception:
        logger.exception("video_visual startup recovery failed; continuing startup")

    # Phase 1C of cloud-transcription-providers spec: a container
    # restart leaves "running" JobRecord rows orphaned (the worker that
    # owned them is gone). Flip them to "failed" with
    # error_class="ContainerRestart" so operators can distinguish real
    # provider errors from stale rows, and purge any partial chunks /
    # words the dead worker wrote.
    try:
        from app.workers.whisper import fail_orphaned_running_jobs
        fail_orphaned_running_jobs()
    except Exception:
        logger.exception(
            "Failed to fail orphaned running JobRecords; continuing startup"
        )

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

    # Per-feature policy: drop video_visual_index artefacts on drives whose
    # video_visual_index policy flipped off (umbrella stays on).
    try:
        from app.purge import purge_disabled_video_visual_drives
        video_visual_purged = await purge_disabled_video_visual_drives()
        if video_visual_purged:
            for drive, count in video_visual_purged.items():
                logger.info(
                    "Purged video_visual_index data for %d files on drive '%s' "
                    "(video_visual_index disabled in drives.json)",
                    count, drive,
                )
    except Exception:
        logger.exception("video_visual_index purge failed; continuing startup")

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

    chapter_worker = ChapterSuggestionsWorker(llm_client)
    dependencies._chapter_suggestions_worker = chapter_worker
    chapter_task: asyncio.Task | None = None
    if settings.features.chapter_suggestions != "false" and llm_client.enabled:
        chapter_task = asyncio.create_task(
            chapter_worker.run(), name="chapter_suggestions_worker"
        )
        logger.info(
            "Chapter suggestions worker started (mode=%s)",
            settings.features.chapter_suggestions,
        )
        if settings.features.chapter_suggestions == "on_index":
            pending = await chapter_worker.enqueue_unprocessed()
            if pending:
                logger.info("Chapter suggestions: queued %d files", pending)

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

    # Initialize video-visual-index worker. Shares the same LLM client and
    # enable rule as vision_describe: needs vision_model configured AND the
    # LLM client itself enabled.
    video_visual_worker = VideoVisualWorker(llm_client)
    dependencies._video_visual_worker = video_visual_worker
    video_visual_task: asyncio.Task | None = None

    from app.config import is_video_visual_index_available

    if is_video_visual_index_available(settings) and llm_client.enabled:
        video_visual_task = asyncio.create_task(
            video_visual_worker.run(), name="video_visual_worker"
        )
        logger.info(
            "Video visual index worker started (mode=%s, model=%s)",
            settings.features.video_visual_index,
            settings.llm.vision_model,
        )

        if settings.features.video_visual_index == "on_index":
            pending = await video_visual_worker.enqueue_unprocessed()
            if pending > 0:
                logger.info(
                    "Video visual index: queued %d previously indexed files", pending
                )

    # Initialize retrieval_keywords worker (SIRA-style keyword expansion).
    # Same enable rule as auto_tags / summaries: needs LLM client AND a
    # non-false feature mode. opt-in by default.
    retrieval_keywords_worker = RetrievalKeywordsWorker(llm_client)
    dependencies._retrieval_keywords_worker = retrieval_keywords_worker
    retrieval_keywords_task: asyncio.Task | None = None

    if (
        settings.features.retrieval_keywords != "false"
        and llm_client.enabled
    ):
        retrieval_keywords_task = asyncio.create_task(
            retrieval_keywords_worker.run(),
            name="retrieval_keywords_worker",
        )
        logger.info(
            "Retrieval-keywords worker started (mode=%s)",
            settings.features.retrieval_keywords,
        )

        # on_index: queue already-indexed files that still need a
        # keyword expansion. Mirrors the auto_tags / summaries
        # on_index sweep.
        if settings.features.retrieval_keywords == "on_index":
            pending = await retrieval_keywords_worker.enqueue_unprocessed()
            if pending > 0:
                logger.info(
                    "Retrieval-keywords: queued %d previously indexed files",
                    pending,
                )

    # Initialize pickup worker (always runs — no feature flag needed)
    pickup_worker = PickupWorker()
    dependencies._pickup_worker = pickup_worker
    pickup_task = asyncio.create_task(pickup_worker.run(), name="pickup_worker")
    logger.info("Pickup recommendation worker started")

    # Start index manager (pass workers for post-metadata hooks)
    index_manager = IndexManager(
        auto_tags_worker=auto_tags_worker,
        summaries_worker=summaries_worker,
        retrieval_keywords_worker=retrieval_keywords_worker,
        chapter_suggestions_worker=chapter_worker,
    )
    dependencies._index_manager = index_manager
    try:
        await index_manager.start()
    except Exception as e:
        logger.error("Failed to start index manager: %s", e)

    logger.info("Semantic search service ready on port %d", settings.port)
    yield

    # Shutdown
    pickup_task.cancel()
    try:
        await pickup_task
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
    if chapter_task is not None:
        chapter_task.cancel()
        try:
            await chapter_task
        except asyncio.CancelledError:
            pass
    if vision_task is not None:
        vision_task.cancel()
        try:
            await vision_task
        except asyncio.CancelledError:
            pass
    if video_visual_task is not None:
        video_visual_task.cancel()
        try:
            await video_visual_task
        except asyncio.CancelledError:
            pass
    if retrieval_keywords_task is not None:
        retrieval_keywords_task.cancel()
        try:
            await retrieval_keywords_task
        except asyncio.CancelledError:
            pass
    if dependencies._index_manager is not None:
        await dependencies._index_manager.stop()
    watchdog.stop()
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
app.include_router(pickup.router)
app.include_router(summaries.router)
app.include_router(chapter_suggestions.router)
app.include_router(rag.router)
app.include_router(refine.router)
app.include_router(vision.router)
app.include_router(video_visual.router)
app.include_router(admin.router)


@app.get("/status", response_model=StatusResponse, tags=["status"])
def status_endpoint() -> StatusResponse:
    """Get current service and indexing status.

    Declared as a sync ``def`` (not ``async def``) on purpose: the body
    issues a dozen synchronous SQLite ``COUNT`` queries via
    ``IndexManager.get_index_status()`` plus a name-lookup query, which
    would block the event loop and starve unrelated endpoints during
    heavy indexing. FastAPI runs sync routes on its threadpool, so the
    loop stays free to serve search / other addon traffic while /status
    waits on disk.
    """
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
    if dependencies._video_visual_worker is not None:
        tasks["video_visual"] = dependencies._video_visual_worker.get_status()
    if dependencies._retrieval_keywords_worker is not None:
        tasks["retrieval_keywords"] = (
            dependencies._retrieval_keywords_worker.get_status()
        )
    if dependencies._chapter_suggestions_worker is not None:
        tasks["chapter_suggestions"] = (
            dependencies._chapter_suggestions_worker.get_status()
        )
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
            chapter_suggestions=settings.features.chapter_suggestions,
            video_visual_index=settings.features.video_visual_index,
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
