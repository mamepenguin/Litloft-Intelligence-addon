"""Index manager: diff detection, queue management, and priority control.

Compares HomeVault DB with the search index to detect new, updated,
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
    get_homevault_db,
    get_search_db,
    upsert_fts_file,
    validate_vector_table,
)
from app.models import Embedding, IndexedFile, TranscriptChunk
from app.workers.blip import check_idle_unload as check_blip_idle_unload
from app.workers.clip import index_clip, IMAGE_TYPES, VIDEO_TYPES
from app.workers.metadata import index_metadata_batch, index_text_content
from app.workers.whisper import (
    check_idle_unload as check_whisper_idle_unload,
    index_whisper,
    TRANSCRIBABLE_TYPES,
)

logger = logging.getLogger(__name__)

# MIME types for text content extraction (shared across all indexing logic)
TEXT_MIMES = frozenset({
    "text/plain", "text/markdown", "text/csv",
    "application/json", "application/pdf",
    "text/srt", "text/vtt",
})


class TaskType(str, Enum):
    METADATA = "metadata"
    CLIP = "clip"
    WHISPER = "whisper"
    TEXT_CONTENT = "text_content"


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
    whisper_indexed: int = 0
    text_indexed: int = 0
    pending_metadata: int = 0
    pending_clip: int = 0
    pending_whisper: int = 0
    pending_text: int = 0


class IndexManager:
    """Manages the indexing pipeline and task queue.

    Provides methods for:
    - Detecting file differences between HomeVault DB and search index
    - Queuing files for processing through the pipeline
    - Priority control and pause/resume
    - Periodic reconciliation
    """

    def __init__(
        self,
        auto_tags_worker: object | None = None,
        summaries_worker: object | None = None,
    ) -> None:
        self._auto_tags_worker = auto_tags_worker
        self._summaries_worker = summaries_worker
        self._queues: dict[TaskType, asyncio.PriorityQueue[tuple[int, float, IndexTask]]] = {
            task_type: asyncio.PriorityQueue()
            for task_type in TaskType
        }
        self._state = QueueState.RUNNING
        self._processing_count = 0
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Start unpaused
        self._whisper_semaphore = asyncio.Semaphore(
            settings.workers.whisper_parallel
        )
        self._running_tasks: set[str] = set()
        self._background_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start the background processing workers."""
        logger.info("Starting index manager")

        # Start worker tasks
        clip_count = settings.workers.clip_parallel
        self._background_tasks = [
            asyncio.create_task(self._metadata_worker(), name="metadata_worker"),
            *[
                asyncio.create_task(
                    self._clip_worker(), name=f"clip_worker_{i}"
                )
                for i in range(clip_count)
            ],
            asyncio.create_task(self._whisper_worker(), name="whisper_worker"),
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
            clip_types = list(IMAGE_TYPES | VIDEO_TYPES)
            clip = active_files.filter(
                IndexedFile.clip_indexed.is_(True),
                IndexedFile.mime_type.in_(clip_types),
            ).count()

            whisper_types = list(TRANSCRIBABLE_TYPES)
            whisper = active_files.filter(
                IndexedFile.whisper_indexed.is_(True),
                IndexedFile.mime_type.in_(whisper_types),
            ).count()

            text = active_files.filter(
                IndexedFile.text_indexed.is_(True),
                IndexedFile.mime_type.in_(list(TEXT_MIMES)),
            ).count()

            # Count pending by type
            pending_metadata = active_files.filter(
                IndexedFile.metadata_indexed.is_(False),
            ).count()

            pending_clip = active_files.filter(
                IndexedFile.clip_indexed.is_(False),
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

            return IndexStatus(
                total_indexed=total,
                metadata_indexed=metadata,
                clip_indexed=clip,
                whisper_indexed=whisper,
                text_indexed=text,
                pending_metadata=pending_metadata,
                pending_clip=pending_clip,
                pending_whisper=pending_whisper,
                pending_text=pending_text,
            )

    async def reconcile(self) -> dict[str, int]:
        """Reconcile search index with HomeVault DB.

        Detects new files, removed files, and soft-deleted files.

        Returns:
            Dict with counts of added, deactivated, and purged files.
        """
        logger.info("Starting reconciliation with HomeVault DB")
        added = 0
        deactivated = 0
        purged = 0

        try:
            homevault_files = _get_homevault_files()
            indexed_files = _get_indexed_file_ids()

            homevault_ids = {f["id"] for f in homevault_files}
            homevault_active = {
                f["id"] for f in homevault_files if f["deleted_at"] is None
            }
            homevault_deleted = {
                f["id"] for f in homevault_files if f["deleted_at"] is not None
            }

            # New files: in HomeVault but not indexed
            new_ids = homevault_active - indexed_files
            if new_ids:
                added = await self._add_new_files(
                    [f for f in homevault_files if f["id"] in new_ids]
                )

            # Soft-deleted: mark as inactive
            for file_id in homevault_deleted & indexed_files:
                _set_file_active(file_id, active=False)
                deactivated += 1

            # Restored: mark as active again
            for file_id in homevault_active & indexed_files:
                _set_file_active(file_id, active=True)

            # Purged: in index but not in HomeVault at all
            orphaned = indexed_files - homevault_ids
            for file_id in orphaned:
                _purge_file(file_id)
                purged += 1

            # Resume incomplete: re-queue files that were interrupted mid-indexing
            resumed = await self._resume_incomplete()

            logger.info(
                "Reconciliation complete: added=%d, deactivated=%d, purged=%d, resumed=%d",
                added, deactivated, purged, resumed,
            )

        except Exception as e:
            logger.error("Reconciliation failed: %s", e)

        return {"added": added, "deactivated": deactivated, "purged": purged}

    async def _add_new_files(self, files: list[dict]) -> int:
        """Add new files to the index and queue them for processing.

        Args:
            files: List of file dicts from HomeVault DB.

        Returns:
            Number of files added.
        """
        added = 0

        with get_search_db() as session:
            for file_data in files:
                try:
                    # Build tags text from HomeVault DB
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

        clip_mimes = IMAGE_TYPES | VIDEO_TYPES

        with get_search_db() as session:
            from sqlalchemy import or_

            incomplete = (
                session.query(IndexedFile)
                .filter(
                    IndexedFile.active.is_(True),
                    or_(
                        IndexedFile.metadata_indexed.is_(False),
                        IndexedFile.clip_indexed.is_(False),
                        IndexedFile.whisper_indexed.is_(False),
                        IndexedFile.text_indexed.is_(False),
                    ),
                )
                .all()
            )

            # Mark inapplicable index types as done so they don't appear
            # as permanently incomplete (e.g., clip_indexed for audio files)
            for f in incomplete:
                if not f.clip_indexed and f.mime_type not in clip_mimes:
                    f.clip_indexed = True
                if not f.whisper_indexed and f.mime_type not in TRANSCRIBABLE_TYPES:
                    f.whisper_indexed = True
                if not f.text_indexed and f.mime_type not in TEXT_MIMES:
                    f.text_indexed = True

            # Snapshot what we need for queuing
            file_tasks: list[tuple[str, str, bool, bool, bool, bool]] = [
                (
                    f.file_id,
                    f.mime_type,
                    f.metadata_indexed,
                    f.clip_indexed,
                    f.whisper_indexed,
                    f.text_indexed,
                )
                for f in incomplete
            ]

        for file_id, mime_type, meta_done, clip_done, whisper_done, text_done in file_tasks:
            queued_any = False

            if not meta_done:
                await self._enqueue(IndexTask(
                    file_id=file_id, task_type=TaskType.METADATA
                ))
                queued_any = True

            if not clip_done:
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

            if queued_any:
                resumed += 1

        return resumed

    async def _queue_file_tasks(self, file_data: dict) -> None:
        """Queue appropriate indexing tasks for a file.

        Args:
            file_data: File data dict from HomeVault DB.
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

        # Queue Whisper for audio/video
        if mime_type in TRANSCRIBABLE_TYPES:
            await self._enqueue(IndexTask(
                file_id=file_id, task_type=TaskType.WHISPER
            ))

        # Queue text content extraction for documents
        if mime_type in TEXT_MIMES:
            await self._enqueue(IndexTask(
                file_id=file_id, task_type=TaskType.TEXT_CONTENT
            ))

    async def _enqueue(self, task: IndexTask) -> None:
        """Add a task to the per-type priority queue.

        Args:
            task: The indexing task to queue.
        """
        # Priority queue: lower number = higher priority
        # Negate priority so higher values are processed first
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

            # Queue with high priority
            if not file.metadata_indexed:
                await self._enqueue(IndexTask(
                    file_id=file_id, task_type=TaskType.METADATA, priority=100
                ))
            if not file.clip_indexed and file.mime_type in (IMAGE_TYPES | VIDEO_TYPES):
                await self._enqueue(IndexTask(
                    file_id=file_id, task_type=TaskType.CLIP, priority=100
                ))
            if not file.whisper_indexed and file.mime_type in TRANSCRIBABLE_TYPES:
                await self._enqueue(IndexTask(
                    file_id=file_id, task_type=TaskType.WHISPER, priority=100
                ))
            if not file.text_indexed:
                await self._enqueue(IndexTask(
                    file_id=file_id, task_type=TaskType.TEXT_CONTENT, priority=100
                ))

            return True

    async def reindex_all(self) -> None:
        """Trigger a full reindex of all files.

        Resets all indexing flags and re-queues everything.
        """
        logger.info("Starting full reindex")

        from app.search import invalidate_similar_cache
        invalidate_similar_cache()

        with get_search_db() as session:
            session.query(IndexedFile).filter(
                IndexedFile.active.is_(True)
            ).update({
                "metadata_indexed": False,
                "clip_indexed": False,
                "whisper_indexed": False,
                "text_indexed": False,
            })

        await self.reconcile()

    async def handle_scan_complete(self, drive: str) -> None:
        """Handle scan-complete webhook from HomeVault.

        Args:
            drive: The drive that was scanned.
        """
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

                try:
                    count = await asyncio.to_thread(
                        index_metadata_batch, file_ids
                    )
                    logger.info("Metadata batch indexed: %d/%d", count, len(batch))

                    # Also process text content for applicable files
                    for file_id in file_ids:
                        await asyncio.to_thread(index_text_content, file_id)

                    # Queue auto-tagging for successfully indexed files (on_index mode only)
                    if (
                        self._auto_tags_worker is not None
                        and settings.features.auto_tags == "on_index"
                        and count > 0
                    ):
                        for file_id in file_ids:
                            await self._auto_tags_worker.enqueue(file_id)

                    # Queue summary generation for successfully indexed files (on_index mode only)
                    if (
                        self._summaries_worker is not None
                        and settings.features.summaries == "on_index"
                        and count > 0
                    ):
                        for file_id in file_ids:
                            await self._summaries_worker.enqueue(file_id)

                except Exception as e:
                    logger.error("Metadata batch failed: %s", e)
                    # Re-queue failed batch so they're retried
                    for task in batch:
                        await self._enqueue(task)
                finally:
                    self._processing_count -= len(batch)

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Metadata worker error: %s", e)
                await asyncio.sleep(5)

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

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("CLIP worker error: %s", e)
                await asyncio.sleep(5)

    async def _whisper_worker(self) -> None:
        """Process Whisper transcription tasks (1 at a time)."""
        while True:
            try:
                await self._pause_event.wait()
                batch = await self._collect_batch(TaskType.WHISPER, 1)

                if not batch:
                    await asyncio.sleep(5)
                    continue

                task = batch[0]
                self._processing_count += 1

                try:
                    async with self._whisper_semaphore:
                        success = await index_whisper(task.file_id)
                        if success:
                            logger.info("Whisper indexed: %s", task.file_id)
                except Exception as e:
                    logger.error(
                        "Whisper indexing failed for %s: %s",
                        task.file_id, e,
                    )
                finally:
                    self._processing_count -= 1

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Whisper worker error: %s", e)
                await asyncio.sleep(10)

    async def _reconciliation_worker(self) -> None:
        """Periodically reconcile index with HomeVault DB."""
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

        while len(collected) < max_size:
            try:
                _, _, task = queue.get_nowait()
                collected = [*collected, task]
            except asyncio.QueueEmpty:
                break

        return collected


# --- Startup cleanup ---


def reset_falsely_completed_clip() -> int:
    """Reset clip_indexed for files marked complete but missing vectors.

    Detects files where clip_indexed=True but no CLIP embeddings exist
    in vec_clip. This happens when a previous run failed mid-indexing
    but still marked the file as complete. Called once at startup.

    Returns:
        Number of files reset.
    """
    reset = 0
    clip_mimes = list(IMAGE_TYPES | VIDEO_TYPES)

    with get_search_db() as session:
        # Find files marked as clip_indexed but with no CLIP embeddings
        # Build IN clause with positional placeholders for SQLite
        placeholders = ", ".join(f":m{i}" for i in range(len(clip_mimes)))
        params = {f"m{i}": m for i, m in enumerate(clip_mimes)}
        falsely_completed = session.execute(
            sql_text(
                "SELECT f.file_id FROM indexed_files f "
                "WHERE f.clip_indexed = 1 AND f.active = 1 "
                f"AND f.mime_type IN ({placeholders}) "
                "AND f.file_id NOT IN ("
                "  SELECT DISTINCT e.file_id FROM embeddings e "
                "  WHERE e.embedding_type = 'clip'"
                ")"
            ),
            params,
        ).fetchall()

        for (file_id,) in falsely_completed:
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


# --- Helper functions for HomeVault DB interaction ---


def _get_homevault_files() -> list[dict]:
    """Get all files from HomeVault DB.

    Returns:
        List of file data dicts.
    """
    with get_homevault_db() as session:
        rows = session.execute(
            sql_text(
                "SELECT id, filename, title, description, drive, folder_path, "
                "file_path, file_size, file_type, mime_type, duration, deleted_at "
                "FROM files"
            )
        ).fetchall()

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
            }
            for row in rows
        ]


def _get_file_tags(file_id: str) -> str:
    """Get tags for a file from HomeVault DB.

    Args:
        file_id: The file ID.

    Returns:
        Space-separated tag names.
    """
    try:
        with get_homevault_db() as session:
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
                table = validate_vector_table(emb.vector_table)
                session.execute(
                    sql_text(
                        f"DELETE FROM {table} WHERE embedding_id = :id"
                    ),
                    {"id": emb.id},
                )

            for emb in embeddings:
                session.delete(emb)

        # Delete transcript chunks
        session.query(TranscriptChunk).filter_by(file_id=file_id).delete()

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

        # Delete indexed file record
        session.query(IndexedFile).filter_by(file_id=file_id).delete()
