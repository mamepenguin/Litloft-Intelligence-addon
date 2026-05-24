"""Index manager: diff detection, queue management, and priority control.

Compares Litloft DB with the search index to detect new, updated,
and removed files. Manages an asyncio-based task queue for processing
files through the indexing pipeline.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sqlalchemy import text as sql_text

from app.config import resolve_file_path, settings
from app.database import (
    delete_fts_file,
    delete_fts_text_content,
    delete_fts_transcripts,
    get_litloft_db,
    get_search_db,
    upsert_fts_file,
    validate_vector_table,
)
from app.models import Embedding, IndexedFile, TranscriptChunk, TranscriptWord
from app.workers.blip import check_idle_unload as check_blip_idle_unload
from app.workers.clip_concepts import check_idle_unload as check_clip_concepts_idle_unload
from app.workers.clip import (
    index_clip,
    IMAGE_TYPES,
    THUMBNAIL_FALLBACK_TYPES,
    VIDEO_TYPES,
)
from app.workers.metadata import index_metadata_batch, index_text_content
from app.workers.whisper import (
    check_idle_unload as check_whisper_idle_unload,
    index_tfidf_keywords_backfill,
    index_whisper,
    TRANSCRIBABLE_TYPES,
    LOFT_MIME,
)

logger = logging.getLogger(__name__)

# MIME types for text content extraction (shared across all indexing logic)
TEXT_MIMES = frozenset({
    "text/plain", "text/markdown", "text/csv",
    "application/json", "application/pdf",
    "text/srt", "text/vtt",
    # Office Open XML formats
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # HTML / XHTML (spec 2026-05-12-html-indexing)
    "text/html", "application/xhtml+xml",
})


class TaskType(str, Enum):
    METADATA = "metadata"
    CLIP = "clip"
    WHISPER = "whisper"
    TEXT_CONTENT = "text_content"
    TFIDF_KEYWORDS = "tfidf_keywords"


class QueueState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"


@dataclass(frozen=True)
class IndexTask:
    """A single indexing task in the queue."""

    file_id: str
    task_type: TaskType
    priority: int = 0  # Higher = processed first


@dataclass
class QueueStatus:
    """Current state of the indexing queue."""

    state: QueueState = QueueState.RUNNING
    processing_count: int = 0
    waiting_count: int = 0


@dataclass
class IndexStatus:
    """Overall indexing status."""

    total_indexed: int = 0
    metadata_indexed: int = 0
    clip_indexed: int = 0
    clip_thumbnail_indexed: int = 0
    whisper_indexed: int = 0
    text_indexed: int = 0
    tfidf_keywords_indexed: int = 0
    pending_metadata: int = 0
    pending_clip: int = 0
    pending_clip_thumbnail: int = 0
    pending_whisper: int = 0
    pending_text: int = 0
    pending_tfidf_keywords: int = 0


class IndexManager:
    """Manages the indexing pipeline and task queue.

    Provides methods for:
    - Detecting file differences between Litloft DB and search index
    - Queuing files for processing through the pipeline
    - Priority control and pause/resume
    - Periodic reconciliation
    """

    def __init__(
        self,
        auto_tags_worker: object | None = None,
        summaries_worker: object | None = None,
        retrieval_keywords_worker: object | None = None,
    ) -> None:
        self._auto_tags_worker = auto_tags_worker
        self._summaries_worker = summaries_worker
        self._retrieval_keywords_worker = retrieval_keywords_worker
        self._queues: dict[TaskType, asyncio.PriorityQueue[tuple[int, float, IndexTask]]] = {
            task_type: asyncio.PriorityQueue()
            for task_type in TaskType
        }
        self._state = QueueState.RUNNING
        self._processing_count = 0
        # Per-task list of file_ids currently being processed, so /status
        # can render "now doing: 文字起こし of foo.mp4" instead of a single
        # opaque count. Lists (not sets) preserve enqueue order which makes
        # the dashboard view stable across polls.
        self._processing_by_type: dict[TaskType, list[str]] = {
            task_type: [] for task_type in TaskType
        }
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Start unpaused
        self._whisper_parallel = max(1, int(settings.workers.whisper_parallel))
        self._whisper_semaphore = asyncio.Semaphore(self._whisper_parallel)
        # Tracks file_ids already sitting in each per-type queue so that
        # repeated reconcile() / _resume_incomplete() calls (e.g. one per
        # scan-complete webhook) don't pile up duplicate entries.
        self._queued_by_type: dict[TaskType, set[str]] = {
            task_type: set() for task_type in TaskType
        }
        self._background_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start the background processing workers."""
        logger.info("Starting index manager")

        # Start worker tasks
        clip_count = settings.workers.clip_parallel
        self._background_tasks = [
            asyncio.create_task(self._metadata_worker(), name="metadata_worker"),
            asyncio.create_task(
                self._text_content_worker(), name="text_content_worker"
            ),
            *[
                asyncio.create_task(
                    self._clip_worker(), name=f"clip_worker_{i}"
                )
                for i in range(clip_count)
            ],
            *[
                asyncio.create_task(
                    self._whisper_worker(), name=f"whisper_worker_{i}"
                )
                for i in range(self._whisper_parallel)
            ],
            asyncio.create_task(
                self._tfidf_keywords_worker(), name="tfidf_keywords_worker"
            ),
            asyncio.create_task(
                self._reconciliation_worker(), name="reconciliation_worker"
            ),
            asyncio.create_task(
                self._idle_unload_worker(), name="idle_unload_worker"
            ),
        ]

        # Initial reconciliation
        await self.reconcile()

    async def stop(self) -> None:
        """Stop all background workers gracefully."""
        logger.info("Stopping index manager")
        for task in self._background_tasks:
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks = []

    def pause(self) -> None:
        """Pause queue processing."""
        self._state = QueueState.PAUSED
        self._pause_event.clear()
        logger.info("Queue paused")

    def resume(self) -> None:
        """Resume queue processing."""
        self._state = QueueState.RUNNING
        self._pause_event.set()
        logger.info("Queue resumed")

    def get_queue_status(self) -> QueueStatus:
        """Get current queue status."""
        return QueueStatus(
            state=self._state,
            processing_count=self._processing_count,
            waiting_count=sum(q.qsize() for q in self._queues.values()),
        )

    def get_queue_breakdown(self) -> dict[str, dict[str, Any]]:
        """Per-task queue snapshot for the dashboard.

        Returns a mapping ``{task_type.value: {"waiting": int,
        "processing": [file_id, ...]}}``. ``processing`` is a copy so
        callers can iterate without holding the worker's reference.
        """
        return {
            task_type.value: {
                "waiting": self._queues[task_type].qsize(),
                "processing": list(self._processing_by_type[task_type]),
            }
            for task_type in TaskType
        }

    def get_index_status(self) -> IndexStatus:
        """Get overall indexing status from the database."""
        with get_search_db() as session:
            active_files = session.query(IndexedFile).filter(
                IndexedFile.active.is_(True)
            )

            total = active_files.count()
            metadata = active_files.filter(
                IndexedFile.metadata_indexed.is_(True)
            ).count()

            # Filter by applicable MIME types for type-specific counts
            clip_types = list(IMAGE_TYPES | VIDEO_TYPES | THUMBNAIL_FALLBACK_TYPES)
            clip = active_files.filter(
                IndexedFile.clip_indexed.is_(True),
                IndexedFile.mime_type.in_(clip_types),
            ).count()
            clip_thumbnail = active_files.filter(
                IndexedFile.clip_thumbnail_indexed.is_(True),
                IndexedFile.mime_type.in_(clip_types),
            ).count()

            whisper_types = list(TRANSCRIBABLE_TYPES) + [LOFT_MIME]
            whisper = active_files.filter(
                IndexedFile.whisper_indexed.is_(True),
                IndexedFile.mime_type.in_(whisper_types),
            ).count()

            text = active_files.filter(
                IndexedFile.text_indexed.is_(True),
                IndexedFile.mime_type.in_(list(TEXT_MIMES)),
            ).count()

            tfidf_kw_types = whisper_types  # same MIME scope
            tfidf_kw = active_files.filter(
                IndexedFile.tfidf_keywords_indexed.is_(True),
                IndexedFile.mime_type.in_(tfidf_kw_types),
            ).count()

            # Count pending by type
            pending_metadata = active_files.filter(
                IndexedFile.metadata_indexed.is_(False),
            ).count()

            pending_clip = active_files.filter(
                IndexedFile.clip_indexed.is_(False),
                IndexedFile.mime_type.in_(clip_types),
            ).count()
            pending_clip_thumbnail = active_files.filter(
                IndexedFile.clip_thumbnail_indexed.is_(False),
                IndexedFile.mime_type.in_(clip_types),
            ).count()

            pending_whisper = active_files.filter(
                IndexedFile.whisper_indexed.is_(False),
                IndexedFile.mime_type.in_(whisper_types),
            ).count()

            pending_text = active_files.filter(
                IndexedFile.text_indexed.is_(False),
                IndexedFile.mime_type.in_(list(TEXT_MIMES)),
            ).count()

            pending_tfidf_kw = active_files.filter(
                IndexedFile.tfidf_keywords_indexed.is_(False),
                IndexedFile.whisper_indexed.is_(True),
                IndexedFile.mime_type.in_(tfidf_kw_types),
            ).count()

            return IndexStatus(
                total_indexed=total,
                metadata_indexed=metadata,
                clip_indexed=clip,
                clip_thumbnail_indexed=clip_thumbnail,
                whisper_indexed=whisper,
                text_indexed=text,
                tfidf_keywords_indexed=tfidf_kw,
                pending_metadata=pending_metadata,
                pending_clip=pending_clip,
                pending_clip_thumbnail=pending_clip_thumbnail,
                pending_whisper=pending_whisper,
                pending_text=pending_text,
                pending_tfidf_keywords=pending_tfidf_kw,
            )

    async def reconcile(self) -> dict[str, int]:
        """Reconcile search index with Litloft DB.

        Detects new files, removed files, and soft-deleted files. Also
        repairs IndexedFile snapshots whose ``(drive, file_path,
        filename)`` drifted from core (a missed ``files.moved`` webhook
        leaves the index pointing at the old path).

        Returns:
            Dict with counts of added, deactivated, purged, and
            drift_repaired files.
        """
        logger.info("Starting reconciliation with Litloft DB")
        added = 0
        deactivated = 0
        purged = 0
        drift_repaired = 0

        try:
            litloft_files = _get_litloft_files()
            indexed_files = _get_indexed_file_ids()

            litloft_ids = {f["id"] for f in litloft_files}
            # A file is "active" for indexing purposes only if it is neither
            # trashed (deleted_at) nor missing (missing_since). Missing files
            # keep their embeddings but are marked inactive so they don't
            # appear in search results — matching the soft-delete behaviour.
            litloft_active = {
                f["id"]
                for f in litloft_files
                if f["deleted_at"] is None and f.get("missing_since") is None
            }
            litloft_inactive = {
                f["id"]
                for f in litloft_files
                if f["deleted_at"] is not None or f.get("missing_since") is not None
            }
            litloft_by_id = {f["id"]: f for f in litloft_files}

            # New files: in Litloft (active) but not indexed
            new_ids = litloft_active - indexed_files
            if new_ids:
                added = await self._add_new_files(
                    [f for f in litloft_files if f["id"] in new_ids]
                )

            # Inactive (trashed or missing): deactivate in the index
            for file_id in litloft_inactive & indexed_files:
                _set_file_active(file_id, active=False)
                deactivated += 1

            # Active: reactivate (covers "missing → recovered" and
            # "trash → restored" since both flow through this branch)
            # and self-heal IndexedFile snapshot drift (webhook fallback).
            indexed_meta = _get_indexed_metadata()
            drifted: list[str] = []
            for file_id in litloft_active & indexed_files:
                _set_file_active(file_id, active=True)

                core = litloft_by_id.get(file_id)
                snap = indexed_meta.get(file_id)
                if core is None or snap is None:
                    continue
                expected_path = resolve_file_path(
                    core["drive"], core["file_path"]
                )
                if expected_path is None:
                    continue
                if (
                    snap["drive"] != core["drive"]
                    or snap["file_path"] != expected_path
                    or snap["filename"] != core["filename"]
                    or snap.get("file_type") != core.get("file_type")
                    or snap.get("mime_type") != core.get("mime_type")
                ):
                    drifted.append(file_id)

            if drifted:
                await self.handle_files_moved(drifted)
                drift_repaired = len(drifted)
                logger.warning(
                    "reconcile() repaired drift on %d IndexedFile rows "
                    "(webhook may be unhealthy): drift_repaired=%d",
                    drift_repaired, drift_repaired,
                )

            # Purged: in index but not in Litloft DB at all
            # (user explicitly called DELETE /purge)
            orphaned = indexed_files - litloft_ids
            for file_id in orphaned:
                _purge_file(file_id)
                purged += 1

            # Reset loft refs that were marked complete but have no transcript
            # (caption download may have failed initially and succeeded later)
            self._reset_loft_refs_with_new_vtt()

            # Resume incomplete: re-queue files that were interrupted mid-indexing
            resumed = await self._resume_incomplete()

            logger.info(
                "Reconciliation complete: added=%d, deactivated=%d, purged=%d, "
                "drift_repaired=%d, resumed=%d",
                added, deactivated, purged, drift_repaired, resumed,
            )

        except Exception as e:
            logger.error("Reconciliation failed: %s", e)

        return {
            "added": added,
            "deactivated": deactivated,
            "purged": purged,
            "drift_repaired": drift_repaired,
        }

    async def _add_new_files(self, files: list[dict]) -> int:
        """Add new files to the index and queue them for processing.

        Args:
            files: List of file dicts from Litloft DB.

        Returns:
            Number of files added.
        """
        from app.policy_client import is_feature_enabled

        # Per-drive policy gate: drop files whose drive disables the
        # ``index`` feature in drives.json before they touch the search
        # DB. The host already filters the scan-complete webhook for
        # listeners that opt in, but workers run reconcile() unprompted
        # too — so we re-check here. is_feature_enabled fails open on
        # network errors; this is a worker optimisation, not the
        # security boundary.
        permitted: list[dict] = []
        for f in files:
            try:
                if await is_feature_enabled(f["drive"], "index"):
                    permitted.append(f)
                else:
                    logger.debug(
                        "Skipping %s: intelligence.index disabled for drive %s",
                        f["id"], f["drive"],
                    )
            except Exception:
                # Defensive: never let policy lookup block real work.
                permitted.append(f)
        files = permitted

        added = 0

        with get_search_db() as session:
            for file_data in files:
                try:
                    # Build tags text from Litloft DB
                    tags_text = _get_file_tags(file_data["id"])

                    # Resolve relative file_path to absolute using drive mounts
                    abs_path = resolve_file_path(
                        file_data["drive"], file_data["file_path"]
                    )
                    if not abs_path:
                        logger.warning(
                            "No mount configured for drive %s, skipping %s",
                            file_data["drive"], file_data["id"],
                        )
                        continue

                    filename = file_data["filename"]
                    title = file_data.get("title", "")
                    description = file_data.get("description", "")

                    indexed_file = IndexedFile(
                        file_id=file_data["id"],
                        drive=file_data["drive"],
                        filename=filename,
                        file_path=abs_path,
                        file_type=file_data["file_type"],
                        mime_type=file_data["mime_type"],
                        file_size=file_data["file_size"],
                        duration=file_data.get("duration"),
                        thumbnail_path=file_data.get("thumbnail_path"),
                        title=title,
                        description=description,
                        tags_text=tags_text,
                    )
                    session.add(indexed_file)

                    # Keep FTS5 trigram index in sync
                    upsert_fts_file(
                        session, file_data["id"],
                        filename, title, description, tags_text,
                    )

                    added += 1
                except Exception as e:
                    logger.error(
                        "Failed to add file %s to index: %s",
                        file_data["id"], e,
                    )

        # Queue all new files for processing
        for file_data in files:
            await self._queue_file_tasks(file_data)

        return added

    def _reset_loft_refs_with_new_vtt(self) -> None:
        """Reset whisper_indexed for loft refs that gained VTT or temp audio.

        When caption download fails initially, the loft ref is marked
        whisper_indexed=True with no TranscriptChunks. If captions are
        later downloaded (via retry or refresh), this resets the flag so
        _resume_incomplete will re-queue them. Media Import can also place
        an adjacent ``*.stt_temp.m4a`` after the loft has already been
        indexed; that temp audio intentionally wins even if VTT chunks
        already exist (manual/always STT).
        """
        with get_search_db() as session:
            loft_refs = (
                session.query(IndexedFile)
                .filter(
                    IndexedFile.active.is_(True),
                    IndexedFile.mime_type == LOFT_MIME,
                    IndexedFile.whisper_indexed.is_(True),
                )
                .all()
            )

            reset_count = 0
            for f in loft_refs:
                # Temp STT audio is an explicit request from Media Import
                # and must re-run even when the loft already has VTT chunks.
                from pathlib import Path
                loft_path = Path(f.file_path)
                stem = loft_path.stem
                parent = loft_path.parent
                temp_audio = parent / f"{stem}.stt_temp.m4a"
                if temp_audio.is_file():
                    f.whisper_indexed = False
                    reset_count += 1
                    continue

                has_chunks = (
                    session.query(TranscriptChunk)
                    .filter_by(file_id=f.file_id)
                    .first()
                    is not None
                )
                if has_chunks:
                    continue

                # Check if VTT file now exists on disk
                if any(parent.glob(f"{stem}*.vtt")):
                    f.whisper_indexed = False
                    reset_count += 1

            if reset_count:
                logger.info(
                    "Reset %d loft ref(s) with new VTT/STT temp for re-indexing",
                    reset_count,
                )

    async def _resume_incomplete(self) -> int:
        """Re-queue active files that have incomplete indexing.

        Finds files where any *_indexed flag is False and queues only
        the missing task types. For index types that don't apply to a
        file's mime_type (e.g., CLIP for audio), the flag is set to True
        directly. This handles recovery after a crash or container restart.

        Returns:
            Number of files re-queued.
        """
        resumed = 0

        clip_mimes = IMAGE_TYPES | VIDEO_TYPES | THUMBNAIL_FALLBACK_TYPES

        with get_search_db() as session:
            from sqlalchemy import or_

            incomplete = (
                session.query(IndexedFile)
                .filter(
                    IndexedFile.active.is_(True),
                    or_(
                        IndexedFile.metadata_indexed.is_(False),
                        IndexedFile.clip_indexed.is_(False),
                        # Phase 4: existing files predating the
                        # clip_thumbnail rollout (spec
                        # 2026-05-02-thumbnail-clip-default-shallow-search.md)
                        # surface here so the next CLIP pass will fill
                        # the thumbnail leg.
                        IndexedFile.clip_thumbnail_indexed.is_(False),
                        IndexedFile.whisper_indexed.is_(False),
                        IndexedFile.text_indexed.is_(False),
                        IndexedFile.tfidf_keywords_indexed.is_(False),
                    ),
                )
                .all()
            )

            # Mark inapplicable index types as done so they don't appear
            # as permanently incomplete (e.g., clip_indexed for audio files)
            for f in incomplete:
                if not f.clip_indexed and f.mime_type not in clip_mimes:
                    f.clip_indexed = True
                if (
                    not f.clip_thumbnail_indexed
                    and f.mime_type not in clip_mimes
                ):
                    f.clip_thumbnail_indexed = True
                if not f.whisper_indexed and f.mime_type not in TRANSCRIBABLE_TYPES and f.mime_type != LOFT_MIME:
                    f.whisper_indexed = True
                if not f.text_indexed and f.mime_type not in TEXT_MIMES:
                    f.text_indexed = True
                # tfidf_keywords: only video/loft files with completed whisper
                if not f.tfidf_keywords_indexed and not f.whisper_indexed:
                    f.tfidf_keywords_indexed = True
                if (
                    not f.tfidf_keywords_indexed
                    and f.mime_type not in TRANSCRIBABLE_TYPES
                    and f.mime_type != LOFT_MIME
                ):
                    f.tfidf_keywords_indexed = True

            # Snapshot what we need for queuing
            file_tasks: list[tuple[str, str, bool, bool, bool, bool, bool, bool]] = [
                (
                    f.file_id,
                    f.mime_type,
                    f.metadata_indexed,
                    f.clip_indexed,
                    f.clip_thumbnail_indexed,
                    f.whisper_indexed,
                    f.text_indexed,
                    f.tfidf_keywords_indexed,
                )
                for f in incomplete
            ]

        for (
            file_id, mime_type,
            meta_done, clip_done, thumb_done, whisper_done, text_done,
            tfidf_kw_done,
        ) in file_tasks:
            queued_any = False

            if not meta_done:
                await self._enqueue(IndexTask(
                    file_id=file_id, task_type=TaskType.METADATA
                ))
                queued_any = True

            # CLIP queue runs ``index_clip`` which handles both the
            # scene route (clip_indexed) and the thumbnail route
            # (clip_thumbnail_indexed) in one job — re-enqueue if
            # *either* leg is incomplete.
            if not clip_done or not thumb_done:
                await self._enqueue(IndexTask(
                    file_id=file_id, task_type=TaskType.CLIP
                ))
                queued_any = True

            if not whisper_done:
                await self._enqueue(IndexTask(
                    file_id=file_id, task_type=TaskType.WHISPER
                ))
                queued_any = True

            if not text_done:
                await self._enqueue(IndexTask(
                    file_id=file_id, task_type=TaskType.TEXT_CONTENT
                ))
                queued_any = True

            # Backfill: file has transcript but no tfidf_keywords embedding.
            # New files get this automatically from _persist_transcript hook.
            if not tfidf_kw_done and whisper_done:
                await self._enqueue(IndexTask(
                    file_id=file_id, task_type=TaskType.TFIDF_KEYWORDS
                ))
                queued_any = True

            if queued_any:
                resumed += 1

        return resumed

    async def _queue_file_tasks(self, file_data: dict) -> None:
        """Queue appropriate indexing tasks for a file.

        Args:
            file_data: File data dict from Litloft DB.
        """
        file_id = file_data["id"]
        mime_type = file_data.get("mime_type", "")

        # Always queue metadata embedding
        await self._enqueue(IndexTask(
            file_id=file_id, task_type=TaskType.METADATA
        ))

        # Queue CLIP for images and videos
        if mime_type in IMAGE_TYPES or mime_type in VIDEO_TYPES:
            await self._enqueue(IndexTask(
                file_id=file_id, task_type=TaskType.CLIP
            ))

        # Vision-describe for images runs in parallel with CLIP — they
        # share no intermediate state. Worker-side ``_should_accept``
        # filters by mime/policy/stickiness; this just kicks the file
        # at the same point CLIP is enqueued.
        if (
            mime_type in IMAGE_TYPES
            and settings.features.vision_describe == "on_index"
        ):
            await self._enqueue_vision_describe(file_id)

        # Queue Whisper for audio/video (or VTT indexing for .loft)
        if mime_type in TRANSCRIBABLE_TYPES or mime_type == LOFT_MIME:
            await self._enqueue(IndexTask(
                file_id=file_id, task_type=TaskType.WHISPER
            ))

        # Queue text content extraction for documents
        if mime_type in TEXT_MIMES:
            await self._enqueue(IndexTask(
                file_id=file_id, task_type=TaskType.TEXT_CONTENT
            ))

    async def _enqueue(self, task: IndexTask, *, force: bool = False) -> None:
        """Add a task to the per-type priority queue.

        Skips if the same (file_id, task_type) is already queued or being
        processed — prevents reconcile() / _resume_incomplete() from
        accumulating duplicates across repeated webhook calls.

        Pass ``force=True`` (used by prioritize()) to bypass the dedup check
        so a file can be re-inserted at higher priority.

        Args:
            task: The indexing task to queue.
            force: If True, enqueue even if the file_id is already present.
        """
        queued = self._queued_by_type[task.task_type]
        processing = self._processing_by_type[task.task_type]
        if not force and (task.file_id in queued or task.file_id in processing):
            return
        queued.add(task.file_id)
        queue = self._queues[task.task_type]
        await queue.put((-task.priority, time.monotonic(), task))

    async def prioritize(self, file_id: str) -> bool:
        """Prioritize a specific file for immediate processing.

        Re-queues all pending tasks for the file with high priority.

        Args:
            file_id: The file to prioritize.

        Returns:
            True if the file was found and prioritized.
        """
        with get_search_db() as session:
            file = session.query(IndexedFile).filter_by(
                file_id=file_id, active=True
            ).first()

            if file is None:
                return False

            # Queue with high priority (force=True allows re-inserting even if
            # already queued at normal priority — a duplicate entry is acceptable
            # here since index_* functions are idempotent on the *_indexed flag)
            if not file.metadata_indexed:
                await self._enqueue(IndexTask(
                    file_id=file_id, task_type=TaskType.METADATA, priority=100
                ), force=True)
            if not file.clip_indexed and file.mime_type in (IMAGE_TYPES | VIDEO_TYPES):
                await self._enqueue(IndexTask(
                    file_id=file_id, task_type=TaskType.CLIP, priority=100
                ), force=True)
            if not file.whisper_indexed and (file.mime_type in TRANSCRIBABLE_TYPES or file.mime_type == LOFT_MIME):
                await self._enqueue(IndexTask(
                    file_id=file_id, task_type=TaskType.WHISPER, priority=100
                ), force=True)
            if not file.text_indexed:
                await self._enqueue(IndexTask(
                    file_id=file_id, task_type=TaskType.TEXT_CONTENT, priority=100
                ), force=True)

            return True

    # ------------------------------------------------------------------
    # Per-file × per-task reindex hooks
    # ------------------------------------------------------------------
    #
    # Spec ``2026-05-24-intelligence-reindex-controls.md`` §2.1. The
    # router-side handler validates the task name and flips the
    # ``*_indexed`` flag to ``False`` itself (so a single transaction
    # covers all tasks for the request); these helpers exist so the
    # handler can talk to the queue without poking the manager's
    # private state.

    # Spec-name → internal TaskType. The spec uses ``"text"`` for the
    # text-content task; the internal enum keeps ``TEXT_CONTENT`` for
    # parity with the existing worker / queue naming.
    _TASK_NAME_TO_TYPE: dict[str, TaskType] = {
        "metadata": TaskType.METADATA,
        "clip": TaskType.CLIP,
        "whisper": TaskType.WHISPER,
        "text": TaskType.TEXT_CONTENT,
    }

    def is_queued(self, file_id: str, task: str) -> bool:
        """Return ``True`` if ``(file_id, task)`` is already in the queue.

        Used by ``POST /files/{id}/reindex`` to short-circuit with HTTP
        202 ``already_queued`` so connect-clicks from the failed-jobs
        modal don't pile up duplicate flag flips + enqueue entries.
        """
        task_type = self._TASK_NAME_TO_TYPE.get(task)
        if task_type is None:
            return False
        return file_id in self._queued_by_type[task_type]

    async def enqueue_task_for_file(self, file_id: str, task: str) -> None:
        """Enqueue a single ``(file_id, task)`` pair at high priority.

        The caller (``reindex_file`` handler) has already flipped the
        flag to ``False`` and verified the file lives in the caller's
        drive. We enqueue with ``priority=100`` so the user-initiated
        reindex is processed ahead of the regular background sweep.
        ``force=True`` lets us re-enqueue if a stale entry already
        exists; ``is_queued`` is the front-line guard against
        double-clicks.
        """
        task_type = self._TASK_NAME_TO_TYPE.get(task)
        if task_type is None:
            return
        await self._enqueue(
            IndexTask(file_id=file_id, task_type=task_type, priority=100),
            force=True,
        )

    async def requeue_after_whisper(self, file_id: str) -> None:
        """Re-enqueue summaries / auto_tags after WHISPER completion.

        Closes the race where METADATA-driven enqueue fires before
        TranscriptChunk rows exist: the LLM workers `insufficient_content`
        silent-return without writing a `file_summaries` / `suggested_tags`
        row, leaving the file stuck until the next intelligence restart
        sweep.

        Conditions match the existing `enqueue_unprocessed` sweep so
        user-deleted summaries / tags are not regenerated — only the
        silent-return trace is rescued.
        """
        with get_search_db() as session:
            row = session.execute(
                sql_text(
                    "SELECT metadata_indexed FROM indexed_files "
                    "WHERE file_id = :fid AND active = 1"
                ),
                {"fid": file_id},
            ).fetchone()
            if row is None or not row[0]:
                return

            no_summary = session.execute(
                sql_text(
                    "SELECT 1 FROM file_summaries "
                    "WHERE file_id = :fid LIMIT 1"
                ),
                {"fid": file_id},
            ).fetchone() is None
            no_tags = session.execute(
                sql_text(
                    "SELECT 1 FROM suggested_tags "
                    "WHERE file_id = :fid LIMIT 1"
                ),
                {"fid": file_id},
            ).fetchone() is None

        # SummariesWorker.enqueue gates per-layer (short/long vs detailed)
        # internally — hand it the file when either layer is on_index.
        summaries_on = (
            settings.features.summaries == "on_index"
            or settings.features.detailed_summaries == "on_index"
        )
        if (
            self._summaries_worker is not None
            and no_summary
            and summaries_on
        ):
            await self._summaries_worker.enqueue(file_id)

        if (
            self._auto_tags_worker is not None
            and no_tags
            and settings.features.auto_tags == "on_index"
        ):
            await self._auto_tags_worker.enqueue(file_id)

    # ``reindex_all`` permanently removed per spec
    # ``2026-05-24-intelligence-reindex-controls.md`` §1. The single
    # caller (``POST /queue/reindex``) was a global blast-radius bug
    # (hako WmAMUDZSsMHlutJFKsyAe). Per-file × per-task reindex lives
    # in ``app.routers.files.reindex_file`` which talks to
    # :meth:`is_queued` + :meth:`enqueue_task_for_file` below.

    async def handle_scan_complete(self, drive: str) -> None:
        """Handle scan-complete webhook from Litloft.

        Args:
            drive: The drive that was scanned.
        """
        from app.policy_client import is_feature_enabled

        if not await is_feature_enabled(drive, "index"):
            logger.info(
                "Skipping reconcile for drive %s (intelligence.index disabled)",
                drive,
            )
            return
        logger.info("Handling scan-complete for drive: %s", drive)
        await self.reconcile()

    async def handle_files_deleted(
        self, file_ids: list[str], delete_type: str
    ) -> None:
        """Handle file deletion webhook.

        Args:
            file_ids: List of deleted file IDs.
            delete_type: "soft_delete" or "hard_delete".
        """
        for file_id in file_ids:
            if delete_type == "soft_delete":
                _set_file_active(file_id, active=False)
            else:
                _purge_file(file_id)

    async def handle_files_restored(self, file_ids: list[str]) -> None:
        """Handle file restoration webhook.

        Args:
            file_ids: List of restored file IDs.
        """
        for file_id in file_ids:
            _set_file_active(file_id, active=True)

    async def handle_files_purged(self, file_ids: list[str]) -> None:
        """Handle file purge webhook (permanent deletion).

        Args:
            file_ids: List of purged file IDs.
        """
        for file_id in file_ids:
            _purge_file(file_id)

    async def handle_files_missing(self, file_ids: list[str]) -> None:
        """Handle files-missing webhook.

        Missing files are kept in the search index but marked inactive so
        they don't appear in search results. Embeddings, transcripts and
        CLIP vectors are preserved for fast reactivation on recovery.

        Args:
            file_ids: IDs of files that vanished from the Litloft filesystem.
        """
        for file_id in file_ids:
            _set_file_active(file_id, active=False)

    async def handle_files_recovered(self, file_ids: list[str]) -> None:
        """Handle files-recovered webhook.

        Reactivates files that were previously marked missing.

        Args:
            file_ids: IDs of files that reappeared on the Litloft filesystem.
        """
        for file_id in file_ids:
            _set_file_active(file_id, active=True)

    async def handle_files_moved(self, file_ids: list[str]) -> None:
        """Sync IndexedFile snapshot after rename / move / folder ops in core.

        Refreshes ``drive`` / ``file_path`` / ``filename`` / ``title`` and
        re-upserts the FTS5 row from Litloft DB. ``*_indexed`` flags and
        embeddings are preserved — file content is unchanged.

        Args:
            file_ids: IDs whose path / name changed in core.
        """
        if not file_ids:
            return

        from app.policy_client import is_feature_enabled

        litloft_meta = _get_litloft_files_by_ids(file_ids)
        if not litloft_meta:
            return

        with get_search_db() as session:
            for file_id, meta in litloft_meta.items():
                # Per-drive policy gate (fail open). Workers also re-check
                # so this is a worker-fast-path optimisation, not a security
                # boundary.
                try:
                    if not await is_feature_enabled(meta["drive"], "index"):
                        continue
                except Exception:
                    pass

                indexed = (
                    session.query(IndexedFile)
                    .filter_by(file_id=file_id)
                    .first()
                )
                if indexed is None:
                    # Not indexed yet; reconcile() will pick it up as a
                    # new file when the drive's policy permits.
                    continue

                abs_path = resolve_file_path(meta["drive"], meta["file_path"])
                if abs_path is None:
                    logger.warning(
                        "Cannot resolve path for %s after move (drive=%s)",
                        file_id, meta["drive"],
                    )
                    continue

                title = meta.get("title") or ""
                indexed.drive = meta["drive"]
                indexed.file_path = abs_path
                indexed.filename = meta["filename"]
                indexed.title = title
                # file_type / mime_type drift can come from a core
                # ``classify()`` rule update (e.g. .loft promoted from
                # ``other`` to ``video``). Sync them here so reconcile's
                # drift repair fixes the search-time file_type filter.
                core_file_type = meta.get("file_type")
                core_mime_type = meta.get("mime_type")
                if core_file_type is not None:
                    indexed.file_type = core_file_type
                if core_mime_type is not None:
                    indexed.mime_type = core_mime_type

                # FTS5 has no UNIQUE on file_id, so ``INSERT OR REPLACE``
                # would leave the old row alongside the new one. Delete
                # first to keep the index single-rowed per file.
                delete_fts_file(session, file_id)
                upsert_fts_file(
                    session, file_id,
                    meta["filename"],
                    title,
                    indexed.description,
                    indexed.tags_text,
                )

    # --- Background workers ---

    async def _metadata_worker(self) -> None:
        """Process metadata embedding tasks in batches."""
        batch_size = settings.workers.metadata_batch_size

        while True:
            try:
                await self._pause_event.wait()
                batch = await self._collect_batch(TaskType.METADATA, batch_size)

                if not batch:
                    await asyncio.sleep(2)
                    continue

                self._processing_count += len(batch)
                file_ids = [t.file_id for t in batch]
                self._processing_by_type[TaskType.METADATA].extend(file_ids)

                try:
                    count = await asyncio.to_thread(
                        index_metadata_batch, file_ids
                    )
                    logger.info("Metadata batch indexed: %d/%d", count, len(batch))

                    # text_content is processed by `_text_content_worker`
                    # off the dedicated TEXT_CONTENT queue. The metadata
                    # worker no longer runs `index_text_content` inline so
                    # files whose metadata is already indexed (e.g. after
                    # an extractor change) still get re-extracted when the
                    # reconciliation worker requeues just the TEXT_CONTENT
                    # task.

                    # Queue auto-tagging for successfully indexed files (on_index mode only)
                    if (
                        self._auto_tags_worker is not None
                        and settings.features.auto_tags == "on_index"
                        and count > 0
                    ):
                        for file_id in file_ids:
                            await self._auto_tags_worker.enqueue(file_id)

                    # Queue summary generation for successfully indexed files.
                    # Triggered for on_index mode on either the short/long
                    # path or the detailed path — the worker decides which
                    # layer actually runs per file.
                    if (
                        self._summaries_worker is not None
                        and count > 0
                        and (
                            settings.features.summaries == "on_index"
                            or settings.features.detailed_summaries == "on_index"
                        )
                    ):
                        for file_id in file_ids:
                            await self._summaries_worker.enqueue(file_id)

                    # Queue retrieval-keywords generation for newly indexed
                    # files (on_index mode only). The worker enqueue gate
                    # rechecks per-drive policy and the unsupported
                    # context-type filter, so this fan-out is safe to call
                    # for every file_id without pre-filtering by type.
                    if (
                        self._retrieval_keywords_worker is not None
                        and count > 0
                        and settings.features.retrieval_keywords == "on_index"
                    ):
                        for file_id in file_ids:
                            await self._retrieval_keywords_worker.enqueue(file_id)

                except Exception as e:
                    logger.error("Metadata batch failed: %s", e)
                    # Re-queue failed batch so they're retried. Use force=True
                    # because the files are still in _processing_by_type at this
                    # point (finally runs after except) and would otherwise be
                    # silently dropped by the dedup check.
                    for task in batch:
                        await self._enqueue(task, force=True)
                finally:
                    self._processing_count -= len(batch)
                    for fid in file_ids:
                        try:
                            self._processing_by_type[TaskType.METADATA].remove(fid)
                        except ValueError:
                            pass

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Metadata worker error: %s", e)
                await asyncio.sleep(5)

    async def _text_content_worker(self) -> None:
        """Process TEXT_CONTENT extraction tasks one file at a time.

        Independent of the metadata worker so files whose metadata is
        already indexed can still be re-extracted (e.g. after switching
        the PDF extractor to PyMuPDF4LLM, the migration only resets
        ``text_indexed`` and the TEXT_CONTENT queue is the path through
        which the new extraction runs).

        ``index_text_content`` is idempotent and skips files whose
        ``text_indexed`` flag is already True, so concurrent enqueues
        are safe.
        """
        while True:
            try:
                await self._pause_event.wait()
                batch = await self._collect_batch(TaskType.TEXT_CONTENT, 1)

                if not batch:
                    await asyncio.sleep(2)
                    continue

                task = batch[0]
                self._processing_count += 1
                self._processing_by_type[TaskType.TEXT_CONTENT].append(task.file_id)

                try:
                    await asyncio.to_thread(index_text_content, task.file_id)
                except Exception as e:
                    logger.error(
                        "Text content indexing failed for %s: %s",
                        task.file_id, e,
                    )
                finally:
                    self._processing_count -= 1
                    try:
                        self._processing_by_type[TaskType.TEXT_CONTENT].remove(task.file_id)
                    except ValueError:
                        pass

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Text content worker error: %s", e)
                await asyncio.sleep(5)

    async def _enqueue_vision_describe(self, file_id: str) -> None:
        """Hand the file to the VisionDescribeWorker (on_index hook).

        Looks up the worker singleton from ``app.dependencies`` lazily
        so this module stays importable before the lifespan has wired
        the global. Silently no-ops when the singleton is missing (early
        startup, partial test harness, feature disabled mid-run).
        """
        try:
            from app.dependencies import get_vision_worker

            worker = get_vision_worker()
        except Exception:
            return
        try:
            await worker.enqueue(file_id)
        except Exception as e:
            logger.warning(
                "vision_describe on_index enqueue failed for %s (%s)",
                file_id, type(e).__name__,
            )

    async def _clip_worker(self) -> None:
        """Process CLIP embedding tasks."""
        while True:
            try:
                await self._pause_event.wait()
                batch = await self._collect_batch(TaskType.CLIP, 1)

                if not batch:
                    await asyncio.sleep(2)
                    continue

                task = batch[0]
                self._processing_count += 1
                self._processing_by_type[TaskType.CLIP].append(task.file_id)

                try:
                    success = await index_clip(task.file_id)
                    if success:
                        logger.info("CLIP indexed: %s", task.file_id)
                    else:
                        logger.warning(
                            "CLIP indexing failed for %s, will retry on next reconciliation",
                            task.file_id,
                        )
                except Exception as e:
                    logger.error(
                        "CLIP indexing failed for %s: %s, will retry on next reconciliation",
                        task.file_id, e,
                    )
                finally:
                    self._processing_count -= 1
                    try:
                        self._processing_by_type[TaskType.CLIP].remove(task.file_id)
                    except ValueError:
                        pass

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("CLIP worker error: %s", e)
                await asyncio.sleep(5)

    async def _whisper_worker(self) -> None:
        """Process transcription tasks, one file per worker."""
        while True:
            try:
                await self._pause_event.wait()
                batch = await self._collect_batch(TaskType.WHISPER, 1)

                if not batch:
                    await asyncio.sleep(5)
                    continue

                task = batch[0]
                self._processing_count += 1
                self._processing_by_type[TaskType.WHISPER].append(task.file_id)

                try:
                    async with self._whisper_semaphore:
                        success = await index_whisper(task.file_id)
                        if success:
                            logger.info("Whisper indexed: %s", task.file_id)
                            # Wake the LLM workers now that transcript
                            # chunks exist — METADATA-driven enqueue may
                            # have run before they were available. Must
                            # stay light: still inside whisper_semaphore
                            # to keep ordering simple, so this hook is
                            # restricted to a few SELECTs + a non-blocking
                            # queue put per worker.
                            await self.requeue_after_whisper(task.file_id)
                except Exception as e:
                    logger.error(
                        "Whisper indexing failed for %s: %s",
                        task.file_id, e,
                    )
                finally:
                    self._processing_count -= 1
                    try:
                        self._processing_by_type[TaskType.WHISPER].remove(task.file_id)
                    except ValueError:
                        pass

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Whisper worker error: %s", e)
                await asyncio.sleep(10)

    async def _tfidf_keywords_worker(self) -> None:
        """Process TF-IDF keyword embedding tasks."""
        while True:
            try:
                await self._pause_event.wait()
                batch = await self._collect_batch(TaskType.TFIDF_KEYWORDS, 1)

                if not batch:
                    await asyncio.sleep(5)
                    continue

                task = batch[0]
                self._processing_count += 1
                self._processing_by_type[TaskType.TFIDF_KEYWORDS].append(task.file_id)

                try:
                    success = await asyncio.to_thread(
                        index_tfidf_keywords_backfill, task.file_id
                    )
                    if success:
                        logger.info("TF-IDF keywords indexed: %s", task.file_id)
                except Exception as e:
                    logger.error(
                        "TF-IDF keyword indexing failed for %s: %s",
                        task.file_id, e,
                    )
                finally:
                    self._processing_count -= 1
                    try:
                        self._processing_by_type[TaskType.TFIDF_KEYWORDS].remove(task.file_id)
                    except ValueError:
                        pass

            except Exception as e:
                logger.error("TF-IDF keywords worker error: %s", e)
                await asyncio.sleep(5)

    async def _reconciliation_worker(self) -> None:
        """Periodically reconcile index with Litloft DB."""
        interval = settings.indexing.reconciliation_interval

        while True:
            try:
                await asyncio.sleep(interval)
                await self.reconcile()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Reconciliation worker error: %s", e)
                await asyncio.sleep(60)

    async def _idle_unload_worker(self) -> None:
        """Periodically check if heavy models should be unloaded.

        Models used only during indexing (Whisper, BLIP) are unloaded
        after their configured idle timeout to free RAM. CLIP stays
        loaded because it is also used in the search query path.
        """
        while True:
            try:
                await asyncio.sleep(60)
                check_whisper_idle_unload()
                check_blip_idle_unload()
                check_clip_concepts_idle_unload()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Idle unload worker error: %s", e)

    async def _collect_batch(
        self, task_type: TaskType, max_size: int
    ) -> list[IndexTask]:
        """Collect tasks from the per-type queue.

        Args:
            task_type: The type of tasks to collect.
            max_size: Maximum batch size.

        Returns:
            List of collected tasks.
        """
        queue = self._queues[task_type]
        collected: list[IndexTask] = []

        queued = self._queued_by_type[task_type]
        processing = self._processing_by_type[task_type]
        while len(collected) < max_size:
            try:
                _, _, task = queue.get_nowait()
                queued.discard(task.file_id)
                if task.file_id in processing:
                    continue
                collected = [*collected, task]
            except asyncio.QueueEmpty:
                break

        return collected


# --- Startup cleanup ---


def reset_falsely_completed_clip() -> int:
    """Reset clip_indexed for files marked complete but missing vectors.

    Detects files where clip_indexed=True but no CLIP embeddings exist.
    This happens when a previous run failed mid-indexing but still
    marked the file as complete. Called once at startup.

    Images store ``embedding_type='clip_thumbnail'`` (one per file;
    spec ``2026-05-02-thumbnail-clip-default-shallow-search.md``) and
    videos store ``embedding_type='clip'`` (per scene frame), so a
    valid completion is "at least one row of either type exists".

    Returns:
        Number of files reset.
    """
    reset = 0
    clip_mimes = list(IMAGE_TYPES | VIDEO_TYPES)

    with get_search_db() as session:
        placeholders = ", ".join(f":m{i}" for i in range(len(clip_mimes)))
        params = {f"m{i}": m for i, m in enumerate(clip_mimes)}
        falsely_completed = session.execute(
            sql_text(
                "SELECT f.file_id FROM indexed_files f "
                "WHERE f.clip_indexed = 1 AND f.active = 1 "
                f"AND f.mime_type IN ({placeholders}) "
                "AND f.file_id NOT IN ("
                "  SELECT DISTINCT e.file_id FROM embeddings e "
                "  WHERE e.embedding_type IN ('clip', 'clip_thumbnail')"
                ")"
            ),
            params,
        ).fetchall()

        # Also fetch filename + path for operator-visible logging so stuck
        # files can be identified without diving into the DB directly.
        file_meta: dict[str, tuple[str, str]] = {}
        if falsely_completed:
            ids = [row[0] for row in falsely_completed]
            placeholders2 = ", ".join(f":fid{i}" for i in range(len(ids)))
            rows = session.execute(
                sql_text(
                    f"SELECT file_id, filename, file_path FROM indexed_files "
                    f"WHERE file_id IN ({placeholders2})"
                ),
                {f"fid{i}": fid for i, fid in enumerate(ids)},
            ).fetchall()
            file_meta = {row[0]: (row[1], row[2]) for row in rows}

        for (file_id,) in falsely_completed:
            filename, fpath = file_meta.get(file_id, ("?", "?"))
            logger.warning(
                "reset_falsely_completed_clip: resetting %s (%s, path=%s) "
                "— clip_indexed=True but no CLIP embeddings found; "
                "will retry on next reconcile",
                file_id, filename, fpath,
            )
            session.execute(
                sql_text(
                    "UPDATE indexed_files SET clip_indexed = 0 "
                    "WHERE file_id = :file_id"
                ),
                {"file_id": file_id},
            )
            reset += 1

    return reset


def cleanup_orphaned_embeddings() -> int:
    """Remove embeddings whose vectors are missing from vec tables.

    This handles the case where a crash occurred between writing the
    embedding record and inserting/committing the vector, or vice versa.
    Called once at startup before the index manager starts.

    Returns:
        Number of orphaned embedding records removed.
    """
    cleaned = 0

    with get_search_db() as session:
        # Find embedding records with no matching vector
        for vec_table_name in ("vec_text", "vec_clip"):
            table = validate_vector_table(vec_table_name)
            orphaned = session.execute(
                sql_text(
                    f"SELECT e.id FROM embeddings e "
                    f"WHERE e.vector_table = :table "
                    f"AND e.id NOT IN (SELECT embedding_id FROM {table})"
                ),
                {"table": table},
            ).fetchall()

            for (emb_id,) in orphaned:
                session.execute(
                    sql_text("DELETE FROM embeddings WHERE id = :id"),
                    {"id": emb_id},
                )
                cleaned += 1

        # Find orphaned vectors with no embedding record
        for vec_table_name in ("vec_text", "vec_clip"):
            table = validate_vector_table(vec_table_name)
            orphaned_vecs = session.execute(
                sql_text(
                    f"SELECT v.embedding_id FROM {table} v "
                    f"WHERE v.embedding_id NOT IN (SELECT id FROM embeddings)"
                ),
            ).fetchall()

            for (vec_id,) in orphaned_vecs:
                session.execute(
                    sql_text(
                        f"DELETE FROM {table} WHERE embedding_id = :id"
                    ),
                    {"id": vec_id},
                )
                cleaned += 1

    return cleaned


# --- Helper functions for Litloft DB interaction ---


def _get_litloft_files_by_ids(file_ids: list[str]) -> dict[str, dict]:
    """Look up core file metadata for the given ids.

    Returns ``{file_id: {drive, file_path, filename, title}}``. Missing
    ids (e.g. purged before the webhook arrived) are silently dropped —
    the caller treats absence as "no-op for this id".
    """
    if not file_ids:
        return {}

    from sqlalchemy import bindparam

    with get_litloft_db() as session:
        # ``thumbnail_path`` was added to core's File model long ago, but
        # some test fixtures still create a slimmer ``files`` table. Probe
        # via PRAGMA so the addon stays compatible with both.
        has_thumbnail_path = any(
            row[1] == "thumbnail_path"
            for row in session.execute(sql_text("PRAGMA table_info(files)")).fetchall()
        )

        thumb_select = "thumbnail_path" if has_thumbnail_path else "NULL AS thumbnail_path"
        rows = session.execute(
            sql_text(
                f"SELECT id, drive, file_path, filename, title, {thumb_select}, "
                "file_type, mime_type "
                "FROM files WHERE id IN :ids"
            ).bindparams(bindparam("ids", expanding=True)),
            {"ids": list(file_ids)},
        ).fetchall()

        return {
            row[0]: {
                "drive": row[1],
                "file_path": row[2],
                "filename": row[3],
                "title": row[4],
                "thumbnail_path": row[5],
                "file_type": row[6],
                "mime_type": row[7],
            }
            for row in rows
        }


def _get_litloft_files() -> list[dict]:
    """Get all files from Litloft DB.

    Returns a list of file dicts including the ``missing_since`` column
    so reconcile can distinguish active / missing / trashed states. The
    column is read via PRAGMA check so older Litloft DBs without it
    still work.
    """
    with get_litloft_db() as session:
        cols = {
            row[1]
            for row in session.execute(sql_text("PRAGMA table_info(files)")).fetchall()
        }
        missing_since_select = (
            "missing_since" if "missing_since" in cols else "NULL AS missing_since"
        )
        # Same defensive PRAGMA pattern as ``missing_since``: keeps slim
        # test fixtures (and any pre-thumbnail core schema) working.
        thumb_select = (
            "thumbnail_path" if "thumbnail_path" in cols else "NULL AS thumbnail_path"
        )
        query = (
            "SELECT id, filename, title, description, drive, folder_path, "
            "file_path, file_size, file_type, mime_type, duration, deleted_at, "
            f"{missing_since_select}, {thumb_select} FROM files"
        )

        rows = session.execute(sql_text(query)).fetchall()

        return [
            {
                "id": row[0],
                "filename": row[1],
                "title": row[2],
                "description": row[3],
                "drive": row[4],
                "folder_path": row[5],
                "file_path": row[6],
                "file_size": row[7],
                "file_type": row[8],
                "mime_type": row[9],
                "duration": row[10],
                "deleted_at": row[11],
                "missing_since": row[12],
                "thumbnail_path": row[13],
            }
            for row in rows
        ]


def _get_file_tags(file_id: str) -> str:
    """Get tags for a file from Litloft DB.

    Args:
        file_id: The file ID.

    Returns:
        Space-separated tag names.
    """
    try:
        with get_litloft_db() as session:
            rows = session.execute(
                sql_text(
                    "SELECT t.name FROM tags t "
                    "JOIN file_tags ft ON t.id = ft.tag_id "
                    "WHERE ft.file_id = :file_id"
                ),
                {"file_id": file_id},
            ).fetchall()

            return " ".join(row[0] for row in rows)
    except Exception:
        return ""


def _get_indexed_file_ids() -> set[str]:
    """Get all file IDs currently in the search index.

    Returns:
        Set of indexed file IDs.
    """
    with get_search_db() as session:
        rows = session.query(IndexedFile.file_id).all()
        return {row[0] for row in rows}


def _get_indexed_metadata() -> dict[str, dict]:
    """Get the snapshot fields used by reconcile drift detection.

    Returns drive / file_path / filename / file_type / mime_type per
    IndexedFile so reconcile can repair both path-class drift
    (rename / move missed by webhook) and classification drift
    (core's ``classify()`` rule changed — e.g. a vendor wrapper mime
    promoted from ``other`` to ``video``).
    """
    with get_search_db() as session:
        rows = session.query(
            IndexedFile.file_id,
            IndexedFile.drive,
            IndexedFile.file_path,
            IndexedFile.filename,
            IndexedFile.file_type,
            IndexedFile.mime_type,
        ).all()
        return {
            row[0]: {
                "drive": row[1],
                "file_path": row[2],
                "filename": row[3],
                "file_type": row[4],
                "mime_type": row[5],
            }
            for row in rows
        }


def _set_file_active(file_id: str, *, active: bool) -> None:
    """Set the active flag for an indexed file.

    Args:
        file_id: The file ID.
        active: Whether the file should be active.
    """
    with get_search_db() as session:
        file = session.query(IndexedFile).filter_by(file_id=file_id).first()
        if file is not None:
            file.active = active


def _purge_file(file_id: str) -> None:
    """Permanently remove a file and all its data from the search index.

    Deletes embeddings, transcript chunks, vector entries, FTS rows,
    LLM-generated artifacts (suggested tags and summaries), and the
    indexed file record.

    Args:
        file_id: The file ID to purge.
    """
    with get_search_db() as session:
        # Delete vector entries
        embeddings = session.query(Embedding).filter_by(file_id=file_id).all()

        if embeddings:
            for emb in embeddings:
                table_raw = emb.vector_table or ""
                # Tolerate non-canonical vector_table values (e.g. the
                # short "text"/"clip" form used by vision_description
                # embeddings in some test fixtures). validate_vector_table
                # would otherwise raise and abort the whole purge.
                if table_raw in ("vec_text", "vec_clip"):
                    table = validate_vector_table(table_raw)
                    session.execute(
                        sql_text(
                            f"DELETE FROM {table} WHERE embedding_id = :id"
                        ),
                        {"id": emb.id},
                    )

            for emb in embeddings:
                session.delete(emb)

        # Delete transcript chunks and word-level rows
        session.query(TranscriptChunk).filter_by(file_id=file_id).delete()
        session.query(TranscriptWord).filter_by(file_id=file_id).delete()

        # Remove from FTS5 indexes
        delete_fts_file(session, file_id)
        delete_fts_transcripts(session, file_id)
        delete_fts_text_content(session, file_id)

        # Delete LLM-generated artifacts. These rows outlive embeddings
        # today (no foreign key) so they must be cleaned explicitly —
        # otherwise synthesized content from deleted files survives in DB.
        session.execute(
            sql_text("DELETE FROM suggested_tags WHERE file_id = :fid"),
            {"fid": file_id},
        )
        session.execute(
            sql_text("DELETE FROM file_summaries WHERE file_id = :fid"),
            {"fid": file_id},
        )
        # retrieval_keywords + its FTS mirror. Older DBs that pre-date
        # Phase 1 of the SIRA retrieval_keywords spec may not have the
        # tables yet — probe sqlite_master before deleting so a missing
        # table doesn't abort the whole purge. Once every install has
        # been started on the current migration, this probe becomes
        # noise and can be removed.
        has_retrieval_kw = session.execute(
            sql_text(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='retrieval_keywords'"
            )
        ).fetchone()
        if has_retrieval_kw is not None:
            from app.database import delete_retrieval_keywords
            delete_retrieval_keywords(session, file_id)
        # detailed_summary_citations: linked 1:N to file_summaries via
        # file_id, so we drop them when the file leaves the index. The
        # table may not exist on very old DBs or in narrow unit-test
        # harnesses that skip its creation — probe sqlite_master first
        # so a missing table doesn't abort the whole purge.
        has_citations = session.execute(
            sql_text(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='detailed_summary_citations'"
            )
        ).fetchone()
        if has_citations is not None:
            session.execute(
                sql_text(
                    "DELETE FROM detailed_summary_citations "
                    "WHERE file_id = :fid"
                ),
                {"fid": file_id},
            )

        # file_insights: cross-DB reference to core File.id with no FK,
        # same pattern as file_summaries. Probe the table first — older
        # DBs / narrow test harnesses may not have it.
        has_insights = session.execute(
            sql_text(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='file_insights'"
            )
        ).fetchone()
        if has_insights is not None:
            session.execute(
                sql_text(
                    "DELETE FROM file_insights WHERE file_id = :fid"
                ),
                {"fid": file_id},
            )

        # Delete indexed file record
        session.query(IndexedFile).filter_by(file_id=file_id).delete()

    # Drop the on-disk CLIP frame thumbnail cache. Done outside the DB
    # session so a filesystem hiccup doesn't roll back the index purge.
    try:
        from app.routers.files import purge_frame_cache

        purge_frame_cache(file_id)
    except Exception as exc:  # noqa: BLE001 — best-effort cache cleanup
        logger.warning(
            "purge_file: frame cache cleanup failed for %s (%s)",
            file_id, exc,
        )
