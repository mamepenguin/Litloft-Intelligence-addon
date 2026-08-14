"""Video Visual Index worker: durable run/scene pipeline.

Turns a native video into a seekable visual index: a small set of
representative frames, each carrying a timestamp, a short description
label for navigation, legible on-screen text, and an optional transcript
excerpt. Owned entirely by the intelligence addon — no core table, no
Internal API endpoint (design doc "Video Visual Index", 2026-08-13).

Durability model (design doc §9): ``video_visual_runs`` /
``video_visual_scenes`` rows are the source of truth. The in-memory
``asyncio.Event`` wake signal is only a hint to check the DB sooner;
losing it (container restart) is harmless because :func:`recover_on_startup`
resets stale ``running`` rows back to a resumable state and the worker
loop re-polls on a short timer regardless.

Non-goals (see design doc §16): every frame / every CLIP candidate sent
to Vision, `.loft` / remote frame extraction, multi-frame contact
sheets, a new core entity, automatic tag/chapter/note writes, default
search inclusion, a whole-video summary, manual scene editing, seek-bar
markers, library-wide auto-processing before the pilot review.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import UTC, datetime

import httpx
from sqlalchemy import or_, text as sql_text
from sqlalchemy.exc import OperationalError

import app.config as config
from app.config import is_video_visual_index_available, settings
from app.database import get_search_db, get_search_db_read, get_search_engine
from app.frame_cache import ensure_frame_cached
from app.llm import VISION_UNSUPPORTED
from app.models import Embedding, IndexedFile, TranscriptChunk, VideoVisualRun, VideoVisualScene
from app.output_language import configured_language_requirement
from app.policy_client import is_file_feature_enabled
from app.prompt_loader import render
from app.workers.clip import VIDEO_TYPES
from app.workers.video_visual_selection import (
    Candidate,
    compute_candidate_fingerprint,
    select_candidates,
)

logger = logging.getLogger(__name__)

# Bumped when the selection algorithm or scene contract changes in a way
# that should be visible on run rows (audit trail only; no code branches
# on this value today).
PIPELINE_VERSION = 2

# Size caps applied at validation/write time (design doc §5.2, §7.1).
# External model output never reaches the DB unbounded.
MAX_SCENE_LABEL_CHARS = 80
MAX_VISIBLE_TEXT_CHARS = 800
MAX_ERROR_MESSAGE_CHARS = 500

ALLOWED_SCENE_TYPES = frozenset(
    {"slide", "screen", "person", "demonstration", "environment", "object", "action", "other"}
)

# Transcript excerpt selection (design doc §7 step 4).
TRANSCRIPT_WINDOW_SECONDS = 15.0
TRANSCRIPT_EXCERPT_MAX_CHARS = 1200

_CORE_INTERNAL_API_DEFAULT = "http://backend:8000/api/internal"

_READY_EVENTS = {
    "succeeded": "intelligence.video_visual.succeeded",
    "partial": "intelligence.video_visual.partial",
    "failed": "intelligence.video_visual.failed",
}


# ---------------------------------------------------------------------------
# Scene prompt
# ---------------------------------------------------------------------------


def _build_scene_system_prompt(output_language: str | None) -> str:
    return render(
        "video_visual_scene/system.jinja2",
        language_requirement=configured_language_requirement(
            output_language,
            auto_requirement=(
                "Use the language indicated by the filename and nearby transcript "
                "context, defaulting to English."
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------


def _load_candidates(file_id: str) -> tuple[list[Candidate], float | None]:
    """Load the file's scene-CLIP candidate pool, ordered by timestamp.

    Reads ``Embedding`` metadata (id, timestamp) from the ORM and the
    matching vectors from ``vec_clip`` in one follow-up query, mirroring
    ``app.workers.clip_concepts.load_file_clip_vectors``.
    """
    with get_search_db_read() as session:
        rows = (
            session.query(Embedding.id, Embedding.timestamp_start)
            .filter(Embedding.file_id == file_id, Embedding.embedding_type == "clip")
            .order_by(Embedding.timestamp_start.asc())
            .all()
        )
        file_row = (
            session.query(IndexedFile.duration)
            .filter(IndexedFile.file_id == file_id)
            .first()
        )
    duration = file_row.duration if file_row is not None else None
    if not rows:
        return [], duration

    ids = [r[0] for r in rows]
    ts_by_id = {r[0]: r[1] for r in rows}
    placeholders = ",".join(f":id{i}" for i in range(len(ids)))
    params = {f"id{i}": eid for i, eid in enumerate(ids)}
    with get_search_engine().connect() as conn:
        vec_rows = conn.execute(
            sql_text(f"SELECT embedding_id, vector FROM vec_clip WHERE embedding_id IN ({placeholders})"),
            params,
        ).fetchall()
    import numpy as np

    vec_by_id = {r[0]: np.frombuffer(r[1], dtype=np.float32) for r in vec_rows if r[1]}

    candidates: list[Candidate] = []
    for eid in ids:
        vec = vec_by_id.get(eid)
        if vec is None:
            continue
        candidates.append(
            Candidate(embedding_id=eid, timestamp=float(ts_by_id[eid] or 0.0), vector=vec)
        )
    return candidates, duration


def _clip_candidates_exist(file_id: str) -> bool:
    with get_search_db_read() as session:
        return (
            session.query(Embedding.id)
            .filter(Embedding.file_id == file_id, Embedding.embedding_type == "clip")
            .first()
            is not None
        )


def _read_or_extract_frame(
    file_id: str,
    start_time: float,
    file_path: str,
) -> bytes:
    """Load one frame without exposing synchronous I/O to the event loop."""
    cache_path = ensure_frame_cached(file_id, start_time, file_path)
    return cache_path.read_bytes()


async def _load_frame_bytes(
    file_id: str,
    start_time: float,
    file_path: str,
) -> bytes:
    return await asyncio.to_thread(
        _read_or_extract_frame,
        file_id,
        start_time,
        file_path,
    )


# ---------------------------------------------------------------------------
# Transcript excerpt (mechanical, never model-generated — design doc §7.1)
# ---------------------------------------------------------------------------


def _select_transcript_excerpt(file_id: str, start_time: float, end_time: float | None) -> str:
    if end_time is not None and end_time > start_time:
        lo, hi = start_time, end_time
    else:
        lo = max(0.0, start_time - TRANSCRIPT_WINDOW_SECONDS / 2)
        hi = start_time + TRANSCRIPT_WINDOW_SECONDS / 2
    with get_search_db_read() as session:
        chunks = (
            session.query(TranscriptChunk)
            .filter(
                TranscriptChunk.file_id == file_id,
                TranscriptChunk.timestamp_start < hi,
                TranscriptChunk.timestamp_end > lo,
            )
            .order_by(TranscriptChunk.timestamp_start)
            .all()
        )
    text = " ".join(c.text.strip() for c in chunks if c.text and c.text.strip())
    return text[:TRANSCRIPT_EXCERPT_MAX_CHARS]


# ---------------------------------------------------------------------------
# Structured-output validation
# ---------------------------------------------------------------------------


def _validate_scene_output(raw: object) -> dict | None:
    """Normalise + bound-check the model's structured output (§7.1).

    Returns ``None`` when ``raw`` is not a usable scene object — the
    caller treats that as "malformed" and drives the repair-retry /
    scene-failure path. ``scene_label`` is required (non-blank);
    ``visible_text`` and ``scene_type`` are optional and omitted rather
    than guessed when uncertain.
    """
    if not isinstance(raw, dict):
        return None
    label = raw.get("scene_label")
    if not isinstance(label, str) or not label.strip():
        return None
    visible_text_raw = raw.get("visible_text")
    visible_text = visible_text_raw.strip() if isinstance(visible_text_raw, str) else ""
    scene_type = raw.get("scene_type")
    if scene_type not in ALLOWED_SCENE_TYPES:
        scene_type = None
    return {
        "scene_label": label.strip()[:MAX_SCENE_LABEL_CHARS],
        "visible_text": visible_text[:MAX_VISIBLE_TEXT_CHARS],
        "scene_type": scene_type,
    }


# ---------------------------------------------------------------------------
# Scene text embedding (embedding_type="video_visual_scene", vec_text)
# ---------------------------------------------------------------------------


def _scene_embedding_id(scene_id: int) -> str:
    return f"vvs_{scene_id}_{uuid.uuid4().hex[:8]}"


def _embed_scene(
    file_id: str,
    scene_id: int,
    scene_label: str,
    visible_text: str,
    start_time: float,
    end_time: float | None,
) -> None:
    """Embed the scene's text and register a ``video_visual_scene`` row.

    Excluded from default search/RAG at read time (design doc §8); the
    write side has no special-casing beyond the ``embedding_type`` tag.
    """
    combined = " ".join(p for p in (scene_label, visible_text) if p).strip()
    if not combined:
        return
    try:
        from app.workers.embedder import embed_passages
    except Exception as e:
        logger.warning("video_visual: embedder unavailable (%s)", e)
        return
    try:
        vectors = embed_passages([combined])
    except Exception as e:
        logger.warning("video_visual: embed_passages failed (%s)", type(e).__name__)
        return
    if not vectors:
        return
    vec = vectors[0]
    embedding_id = _scene_embedding_id(scene_id)
    with get_search_db() as session:
        session.add(
            Embedding(
                id=embedding_id,
                file_id=file_id,
                embedding_type="video_visual_scene",
                timestamp_start=start_time,
                timestamp_end=end_time,
                content_preview=combined[:200],
                vector_table="vec_text",
            )
        )
        session.flush()
        try:
            session.execute(
                sql_text("INSERT INTO vec_text(embedding_id, vector) VALUES(:id, :vec)"),
                {"id": embedding_id, "vec": vec.tobytes()},
            )
        except Exception as e:
            logger.warning("video_visual: vec_text insert failed (%s)", type(e).__name__)


def _purge_scene_embeddings(session, scene_ids: list[int]) -> None:
    """Delete ``video_visual_scene`` embeddings tied to ``scene_ids``.

    Used when superseding a run (§5.3) so a superseded run's stale
    vectors don't linger alongside the new active run's for the same
    file. Matches by the ``vvs_{scene_id}_`` embedding-id prefix (the
    only linkage available — ``Embedding`` has no ``scene_id`` column).
    """
    for sid in scene_ids:
        rows = (
            session.query(Embedding)
            .filter(
                Embedding.embedding_type == "video_visual_scene",
                Embedding.id.like(f"vvs_{sid}_%"),
            )
            .all()
        )
        for emb in rows:
            table = emb.vector_table or ""
            if table.startswith("vec_"):
                try:
                    session.execute(
                        sql_text(f"DELETE FROM {table} WHERE embedding_id = :id"),
                        {"id": emb.id},
                    )
                except OperationalError as e:
                    # Missing vec table (narrow test harness) — log and
                    # keep deleting the metadata row below.
                    logger.warning(
                        "video_visual: vec delete failed for %s (%s)",
                        table, type(e).__name__,
                    )
            session.delete(emb)


# ---------------------------------------------------------------------------
# WebSocket / core events
# ---------------------------------------------------------------------------


async def emit_video_visual_event(event: str, data: dict) -> None:
    """Best-effort, drive-scoped completion event through the core WS bridge.

    Mirrors ``app.workers.chapter_suggestions.emit_chapter_suggestions_event``.
    """
    base = os.environ.get("HOMEVAULT_INTERNAL_API_URL", _CORE_INTERNAL_API_DEFAULT).rstrip("/")
    secret = os.environ.get("CORE_INTERNAL_SECRET", "")
    headers = {"X-Internal-Secret": secret} if secret else {}
    payload: dict = {"event": event, "data": data}
    if drive := data.get("drive"):
        payload["drive"] = drive
    try:
        async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
            response = await client.post(f"{base}/addon-events", headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPError:
        logger.warning("video_visual: could not emit event %s", event)


# ---------------------------------------------------------------------------
# Startup recovery (design doc §9)
# ---------------------------------------------------------------------------


def recover_on_startup() -> dict[str, int]:
    """Reset orphaned ``running`` rows left by a container restart.

    Stale ``running`` scenes return to ``pending``; stale ``running``
    runs return to ``queued``. Succeeded scene rows are never touched —
    the worker's normal processing loop already skips non-``pending``
    scenes. Queued/nonterminal runs need no explicit re-enqueue: the
    worker claims the next ``queued`` run straight from the DB.
    """
    with get_search_db() as session:
        scenes_reset = (
            session.query(VideoVisualScene)
            .filter(VideoVisualScene.status == "running")
            .update({"status": "pending"}, synchronize_session=False)
        )
        runs_reset = (
            session.query(VideoVisualRun)
            .filter(VideoVisualRun.status == "running")
            .update({"status": "queued"}, synchronize_session=False)
        )
    if scenes_reset or runs_reset:
        logger.info(
            "video_visual: startup recovery reset %d scene(s), %d run(s)",
            scenes_reset, runs_reset,
        )
    return {"scenes_reset": scenes_reset, "runs_reset": runs_reset}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class VideoVisualWorker:
    """Single-concurrency durable worker for the video visual index.

    Run/scene rows in the DB are the source of truth (§9); the internal
    queue is only a wake-up signal so the worker doesn't poll on a tight
    busy loop while idle.
    """

    def __init__(self, llm_client) -> None:
        self._llm_client = llm_client
        self._wake = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # start unpaused
        self._current_file_id: str | None = None
        self._current_run_id: str | None = None

    # -- Status / pause ---------------------------------------------------

    def get_status(self) -> dict[str, object]:
        with get_search_db_read() as session:
            waiting = (
                session.query(VideoVisualRun.id)
                .filter(VideoVisualRun.status == "queued")
                .count()
            )
        return {
            "waiting": waiting,
            "processing": [self._current_file_id] if self._current_file_id else [],
            "current_run_id": self._current_run_id,
        }

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    # -- Enqueue ------------------------------------------------------------

    async def _should_accept(self, file_id: str) -> tuple[bool, str]:
        """Shared gate: feature flag + vision_model + drive policy + mime + CLIP + stickiness."""
        if not is_video_visual_index_available(settings):
            return False, "disabled"

        try:
            enabled = await is_file_feature_enabled(
                file_id,
                "video_visual_index",
                default_on_failure=False,
            )
        except Exception as e:
            logger.warning(
                "video_visual: policy lookup failed for %s (%s); refusing (fail-closed)",
                file_id, type(e).__name__,
            )
            return False, "policy_lookup_failed"
        if not enabled:
            return False, "disabled"

        with get_search_db_read() as session:
            file_row = (
                session.query(IndexedFile)
                .filter(IndexedFile.file_id == file_id, IndexedFile.active.is_(True))
                .first()
            )
            if file_row is None:
                return False, "not_found"
            if file_row.mime_type not in VIDEO_TYPES:
                return False, "not_eligible"

            last_run = (
                session.query(VideoVisualRun)
                .filter(VideoVisualRun.file_id == file_id)
                .order_by(VideoVisualRun.created_at.desc())
                .first()
            )
            # Sticky "provider/model can't do this" — same shape as
            # VisionDescribeWorker._should_accept. A model swap clears it.
            if (
                last_run is not None
                and last_run.status == "failed"
                and last_run.error_class == "Unsupported"
                and (last_run.vision_model or "") == (settings.llm.vision_model or "")
            ):
                return False, "unsupported_sticky"

            in_flight = (
                session.query(VideoVisualRun.id)
                .filter(
                    VideoVisualRun.file_id == file_id,
                    VideoVisualRun.status.in_(("queued", "running")),
                )
                .first()
            )
            if in_flight is not None:
                return False, "already_queued"

        if not _clip_candidates_exist(file_id):
            return False, "waiting_clip"

        return True, "ok"

    async def enqueue(self, file_id: str, *, requested_by: str = "manual") -> dict:
        """Stage a new run for ``file_id``.

        ``requested_by`` is ``"manual"`` (priority 100) or ``"on_index"``
        (priority 0) — execution provenance, not user identity (§5.1).
        When scene CLIP isn't ready yet, a manual request prioritizes the
        file's CLIP task (§6.1); an automatic request is simply skipped
        (the periodic on_index sweep / CLIP-completion hook retries it).
        """
        accepted, reason = await self._should_accept(file_id)
        if not accepted:
            if reason == "waiting_clip" and requested_by == "manual":
                try:
                    from app.dependencies import get_index_manager

                    await get_index_manager().prioritize(file_id)
                except Exception:
                    pass
            return {"accepted": False, "reason": reason}

        if requested_by == "on_index":
            # Skip a pointless replacement when the active run's
            # candidate set hasn't actually changed (§12 "Source CLIP
            # fingerprint changes"). Manual "Generate again" always
            # creates a new run regardless (§3.4).
            with get_search_db_read() as session:
                active = (
                    session.query(VideoVisualRun)
                    .filter(VideoVisualRun.file_id == file_id, VideoVisualRun.is_active.is_(True))
                    .first()
                )
            if active is not None:
                candidates, _ = _load_candidates(file_id)
                if (
                    active.pipeline_version == PIPELINE_VERSION
                    and compute_candidate_fingerprint(candidates)
                    == active.candidate_fingerprint
                ):
                    return {"accepted": False, "reason": "up_to_date"}

        priority = 100 if requested_by == "manual" else 0
        with get_search_db() as session:
            run = VideoVisualRun(
                file_id=file_id,
                status="queued",
                requested_by=requested_by,
                priority=priority,
                vision_model=settings.llm.vision_model or "",
                pipeline_version=PIPELINE_VERSION,
                candidate_fingerprint="",
                created_at=datetime.now(UTC),
            )
            session.add(run)
            session.flush()
            run_id = run.id

        self._wake.set()
        return {"accepted": True, "reason": "queued", "run_id": run_id}

    async def retry(self, file_id: str) -> dict:
        """Re-queue only the failed scenes of the file's most recent
        retryable run (§3.4, §9 "Retry operates on the same staged run
        and failed scene rows")."""
        with get_search_db() as session:
            active = (
                session.query(VideoVisualRun)
                .filter(
                    VideoVisualRun.file_id == file_id,
                    VideoVisualRun.is_active.is_(True),
                )
                .first()
            )
            retryable = session.query(VideoVisualRun).filter(
                VideoVisualRun.file_id == file_id,
                VideoVisualRun.status.in_(("partial", "failed")),
            )
            if active is not None:
                retryable = retryable.filter(
                    or_(
                        VideoVisualRun.id == active.id,
                        VideoVisualRun.created_at > active.created_at,
                    )
                )
            run = (
                retryable
                .order_by(VideoVisualRun.created_at.desc())
                .first()
            )
            if run is None:
                return {"accepted": False, "reason": "no_run"}

            failed_scenes = (
                session.query(VideoVisualScene)
                .filter(VideoVisualScene.run_id == run.id, VideoVisualScene.status == "failed")
                .all()
            )
            if not failed_scenes:
                return {"accepted": False, "reason": "no_failed_scenes"}

            reset_count = 0
            for scene in failed_scenes:
                scene.status = "pending"
                scene.error_class = None
                scene.error_message = None
                reset_count += 1

            run.failed_count = max(0, run.failed_count - reset_count)
            run.completed_count = max(0, run.completed_count - reset_count)
            run.status = "queued"
            run.priority = 100
            run.completed_at = None
            run_id = run.id

        self._wake.set()
        return {"accepted": True, "run_id": run_id, "reset_count": reset_count}

    async def enqueue_unprocessed(self) -> int:
        """on_index coverage sweep: queue native videos with no run yet."""
        with get_search_db_read() as session:
            existing_ids = {r[0] for r in session.query(VideoVisualRun.file_id).distinct().all()}
            rows = (
                session.query(IndexedFile.file_id)
                .filter(
                    IndexedFile.active.is_(True),
                    IndexedFile.mime_type.in_(list(VIDEO_TYPES)),
                )
                .all()
            )
        queued = 0
        for (file_id,) in rows:
            if file_id in existing_ids:
                continue
            result = await self.enqueue(file_id, requested_by="on_index")
            if result.get("accepted"):
                queued += 1
        return queued

    # -- Processing -----------------------------------------------------

    def _claim_next_run(self) -> VideoVisualRun | None:
        with get_search_db() as session:
            run = (
                session.query(VideoVisualRun)
                .filter(VideoVisualRun.status == "queued")
                .order_by(VideoVisualRun.priority.desc(), VideoVisualRun.created_at.asc())
                .first()
            )
            if run is None:
                return None
            run.status = "running"
            if run.started_at is None:
                run.started_at = datetime.now(UTC)
            session.flush()
            return run

    async def run(self) -> None:
        """Main loop: claim one run at a time from the DB and drive it."""
        while True:
            try:
                await self._pause_event.wait()
                run_row = await asyncio.to_thread(self._claim_next_run)
                if run_row is None:
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=5.0)
                    except TimeoutError:
                        pass
                    continue

                self._current_run_id = run_row.id
                self._current_file_id = run_row.file_id
                try:
                    await self._process_run(run_row.id, run_row.file_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "video_visual run %s interrupted; re-queueing",
                        run_row.id,
                    )
                    await asyncio.to_thread(
                        self._requeue_interrupted_run,
                        run_row.id,
                    )
                    self._wake.set()
                    await asyncio.sleep(5)
                finally:
                    self._current_run_id = None
                    self._current_file_id = None
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("video_visual worker error")
                await asyncio.sleep(5)

    def _requeue_interrupted_run(self, run_id: str) -> None:
        """Restore a claimed run after an unexpected in-process error."""
        with get_search_db() as session:
            session.query(VideoVisualScene).filter(
                VideoVisualScene.run_id == run_id,
                VideoVisualScene.status == "running",
            ).update({"status": "pending"}, synchronize_session=False)
            session.query(VideoVisualRun).filter(
                VideoVisualRun.id == run_id,
                VideoVisualRun.status == "running",
            ).update({"status": "queued"}, synchronize_session=False)

    async def _process_run(self, run_id: str, file_id: str) -> None:
        with get_search_db_read() as session:
            file_row = (
                session.query(IndexedFile)
                .filter(IndexedFile.file_id == file_id, IndexedFile.active.is_(True))
                .first()
            )
        if file_row is None:
            self._fail_run(run_id, "FileUnavailable")
            return
        drive, file_path, filename = file_row.drive, file_row.file_path, file_row.filename

        try:
            enabled = await is_file_feature_enabled(
                file_id,
                "video_visual_index",
                default_on_failure=False,
            )
        except Exception:
            enabled = False
        if not enabled:
            self._fail_run(run_id, "PolicyDisabled")
            return

        await emit_video_visual_event(
            "intelligence.video_visual.started",
            {"file_id": file_id, "run_id": run_id, "drive": drive},
        )

        with get_search_db_read() as session:
            has_scenes = (
                session.query(VideoVisualScene.id).filter(VideoVisualScene.run_id == run_id).first()
                is not None
            )

        if not has_scenes:
            built = self._build_scenes(run_id, file_id)
            if not built:
                self._fail_run(run_id, "NoCandidates")
                await emit_video_visual_event(
                    _READY_EVENTS["failed"],
                    {"file_id": file_id, "run_id": run_id, "drive": drive, "reason": "no_candidates"},
                )
                return

        with get_search_db_read() as session:
            pending_ids = [
                r[0]
                for r in session.query(VideoVisualScene.id)
                .filter(VideoVisualScene.run_id == run_id, VideoVisualScene.status == "pending")
                .order_by(VideoVisualScene.ordering)
                .all()
            ]

        for scene_id in pending_ids:
            await self._pause_event.wait()  # checked between scenes only
            outcome = await self._process_scene(scene_id, run_id, file_id, file_path, filename)
            if outcome == "unsupported":
                self._fail_run(run_id, "Unsupported")
                await emit_video_visual_event(
                    _READY_EVENTS["failed"],
                    {"file_id": file_id, "run_id": run_id, "drive": drive, "reason": "unsupported"},
                )
                return
            await self._emit_progress(run_id, file_id, drive)

        await self._finalize_run(run_id, file_id, drive)

    def _build_scenes(self, run_id: str, file_id: str) -> bool:
        candidates, duration = _load_candidates(file_id)
        fingerprint = compute_candidate_fingerprint(candidates)
        selected = select_candidates(candidates, duration_seconds=duration)
        with get_search_db() as session:
            run = session.query(VideoVisualRun).filter_by(id=run_id).first()
            if run is None:
                return False
            run.candidate_fingerprint = fingerprint
            if not selected:
                return False
            run.selected_count = len(selected)
            for i, c in enumerate(selected):
                session.add(
                    VideoVisualScene(
                        run_id=run_id,
                        ordering=i,
                        clip_embedding_id=c.embedding_id,
                        start_time=c.timestamp,
                        status="pending",
                    )
                )
        return bool(selected)

    def _fail_run(self, run_id: str, error_class: str, error_message: str = "") -> None:
        with get_search_db() as session:
            run = session.query(VideoVisualRun).filter_by(id=run_id).first()
            if run is None:
                return
            run.status = "failed"
            run.error_class = error_class
            run.error_message = error_message[:MAX_ERROR_MESSAGE_CHARS] or None
            run.completed_at = datetime.now(UTC)

    def _fail_scene(self, scene_id: int, run_id: str, error_class: str, error_message: str) -> str:
        with get_search_db() as session:
            scene = session.query(VideoVisualScene).filter_by(id=scene_id).first()
            if scene is not None:
                scene.status = "failed"
                scene.error_class = error_class
                scene.error_message = (error_message or "")[:MAX_ERROR_MESSAGE_CHARS]
                scene.completed_at = datetime.now(UTC)
            run = session.query(VideoVisualRun).filter_by(id=run_id).first()
            if run is not None:
                run.completed_count += 1
                run.failed_count += 1
        return "failed"

    async def _process_scene(
        self, scene_id: int, run_id: str, file_id: str, file_path: str, filename: str
    ) -> str:
        with get_search_db() as session:
            scene = session.query(VideoVisualScene).filter_by(id=scene_id).first()
            if scene is None:
                return "failed"
            scene.status = "running"
            scene.attempt_count += 1
            start_time = scene.start_time
            end_time = scene.end_time

        if not config.validate_file_path(file_path):
            return self._fail_scene(scene_id, run_id, "InvalidPath", "file path failed validation")

        try:
            raw_bytes = await _load_frame_bytes(file_id, start_time, file_path)
        except Exception as e:
            return self._fail_scene(scene_id, run_id, "FrameExtraction", str(e)[:200])

        # Reuse the existing image preprocessing path (bound dimensions/
        # bytes, strip metadata) even though ffmpeg frames normally carry
        # no EXIF — design doc §7 step 3.
        from app.workers.vision import _preprocess_image

        preprocessed = _preprocess_image(raw_bytes, "image/webp")
        if preprocessed is None:
            return self._fail_scene(scene_id, run_id, "Decode", "frame preprocessing failed")
        image_bytes, image_mime = preprocessed

        transcript_excerpt = _select_transcript_excerpt(file_id, start_time, end_time)

        system_prompt = _build_scene_system_prompt(settings.llm.output_language)
        user_prompt = render(
            "video_visual_scene/user.jinja2",
            filename=filename,
            timestamp_seconds=start_time,
            transcript_context=transcript_excerpt,
        )

        try:
            result = await self._llm_client.generate_video_scene_json(
                image_bytes, image_mime, system_prompt, user_prompt
            )
        except Exception as e:
            logger.warning("video_visual: LLM call raised for scene %s (%s)", scene_id, type(e).__name__)
            result = None

        if result is VISION_UNSUPPORTED:
            return "unsupported"

        parsed = _validate_scene_output(result)
        if parsed is None:
            # One repair attempt, same frame, no resend of a different
            # image (design doc §12).
            retry_prompt = render(
                "video_visual_scene/retry_user.jinja2", original_prompt=user_prompt
            )
            try:
                result2 = await self._llm_client.generate_video_scene_json(
                    image_bytes, image_mime, system_prompt, retry_prompt
                )
            except Exception as e:
                logger.warning(
                    "video_visual: repair LLM call raised for scene %s (%s)",
                    scene_id, type(e).__name__,
                )
                result2 = None
            if result2 is VISION_UNSUPPORTED:
                return "unsupported"
            parsed = _validate_scene_output(result2)
            if parsed is None:
                return self._fail_scene(
                    scene_id, run_id, "MalformedOutput",
                    "structured output invalid after repair attempt",
                )

        with get_search_db() as session:
            scene = session.query(VideoVisualScene).filter_by(id=scene_id).first()
            if scene is not None:
                scene.status = "succeeded"
                scene.scene_label = parsed["scene_label"]
                scene.visible_text = parsed["visible_text"] or None
                scene.scene_type = parsed["scene_type"]
                scene.transcript_excerpt = transcript_excerpt or None
                scene.completed_at = datetime.now(UTC)
            run = session.query(VideoVisualRun).filter_by(id=run_id).first()
            if run is not None:
                run.completed_count += 1
                run.succeeded_count += 1

        try:
            _embed_scene(
                file_id, scene_id,
                parsed["scene_label"], parsed["visible_text"],
                start_time, end_time,
            )
        except Exception as e:
            logger.warning("video_visual: embed_scene failed for %s (%s)", scene_id, type(e).__name__)

        return "succeeded"

    async def _emit_progress(self, run_id: str, file_id: str, drive: str) -> None:
        with get_search_db_read() as session:
            run = session.query(VideoVisualRun).filter_by(id=run_id).first()
            if run is None:
                return
            counts = {
                "selected_count": run.selected_count,
                "completed_count": run.completed_count,
                "succeeded_count": run.succeeded_count,
                "failed_count": run.failed_count,
            }
        await emit_video_visual_event(
            "intelligence.video_visual.progress",
            {"file_id": file_id, "run_id": run_id, "drive": drive, **counts},
        )

    async def _finalize_run(self, run_id: str, file_id: str, drive: str) -> None:
        with get_search_db() as session:
            run = session.query(VideoVisualRun).filter_by(id=run_id).first()
            if run is None:
                return

            if run.succeeded_count > 0 and run.failed_count == 0:
                final_status = "succeeded"
            elif run.succeeded_count > 0:
                final_status = "partial"
            else:
                final_status = "failed"

            run.status = final_status
            run.completed_at = datetime.now(UTC)

            if final_status == "succeeded":
                old_runs = (
                    session.query(VideoVisualRun)
                    .filter(
                        VideoVisualRun.file_id == file_id,
                        VideoVisualRun.id != run_id,
                        VideoVisualRun.status.in_(("succeeded", "partial", "failed")),
                    )
                    .all()
                )
                old_run_ids = [r.id for r in old_runs]
                if old_run_ids:
                    old_scene_ids = [
                        s[0]
                        for s in session.query(VideoVisualScene.id)
                        .filter(VideoVisualScene.run_id.in_(old_run_ids))
                        .all()
                    ]
                    _purge_scene_embeddings(session, old_scene_ids)
                    session.query(VideoVisualRun).filter(
                        VideoVisualRun.id.in_(old_run_ids)
                    ).update(
                        {"is_active": False, "status": "superseded"},
                        synchronize_session=False,
                    )
                    session.flush()
                run.is_active = True
            elif final_status == "partial":
                has_other_active = (
                    session.query(VideoVisualRun.id)
                    .filter(
                        VideoVisualRun.file_id == file_id,
                        VideoVisualRun.is_active.is_(True),
                        VideoVisualRun.id != run_id,
                    )
                    .first()
                    is not None
                )
                if not has_other_active:
                    run.is_active = True
            # failed: is_active stays False, active run (if any) untouched.

            counts = {
                "selected_count": run.selected_count,
                "succeeded_count": run.succeeded_count,
                "failed_count": run.failed_count,
            }

        await emit_video_visual_event(
            _READY_EVENTS[final_status],
            {"file_id": file_id, "run_id": run_id, "drive": drive, **counts},
        )


__all__ = [
    "ALLOWED_SCENE_TYPES",
    "MAX_ERROR_MESSAGE_CHARS",
    "MAX_SCENE_LABEL_CHARS",
    "MAX_VISIBLE_TEXT_CHARS",
    "PIPELINE_VERSION",
    "VideoVisualWorker",
    "emit_video_visual_event",
    "recover_on_startup",
]
