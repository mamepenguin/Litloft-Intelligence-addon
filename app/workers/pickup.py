"""Background worker that precomputes pickup (recommendation) caches.

Triggered by:
  - scan.complete webhook (per drive)
  - 1-hour periodic sweep over all indexed drives

Logic per viewer × drive:
  1. Query WatchHistory from the Litloft read-only DB.
  2. Compute a checkpoint hash of the recent file_id list.
  3. If unchanged since the last run, return immediately.
  4. Run KNN (find_similar) for up to SEED_LIMIT seed files.
  5. Filter out recently-watched files and missing/trash files.
  6. Store the top RESULT_LIMIT file_ids in pickup_cache.
"""

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime

from sqlalchemy import text

from app.database import get_litloft_db, get_search_db
from app.models import IndexedFile, PickupCache

logger = logging.getLogger(__name__)

_SEED_LIMIT = 5
_CANDIDATES_PER_SEED = 12
_RESULT_LIMIT = 12
_HISTORY_LOOKBACK = 50
_PERIODIC_INTERVAL_SECONDS = 3600


class PickupWorker:
    """Precomputes per-viewer pickup recommendations for all indexed drives."""

    def __init__(self) -> None:
        self._running = False
        self._drive_locks: dict[str, asyncio.Lock] = {}
        self._pending_tasks: set[asyncio.Task] = set()

    def _lock_for(self, drive: str) -> asyncio.Lock:
        if drive not in self._drive_locks:
            self._drive_locks[drive] = asyncio.Lock()
        return self._drive_locks[drive]

    async def run(self) -> None:
        self._running = True
        logger.info("Pickup worker started")
        while self._running:
            try:
                await self._sweep_all_drives()
            except Exception:
                logger.exception("Pickup worker sweep failed")
            await asyncio.sleep(_PERIODIC_INTERVAL_SECONDS)

    async def schedule_drive(self, drive: str) -> None:
        """Trigger computation for one drive without blocking the caller."""
        task = asyncio.create_task(
            self._guarded_compute(drive),
            name=f"pickup_compute_{drive}",
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _guarded_compute(self, drive: str) -> None:
        lock = self._lock_for(drive)
        if lock.locked():
            return
        async with lock:
            try:
                await self._compute_for_drive(drive)
            except Exception:
                logger.exception("Pickup worker failed for drive=%s", drive)

    async def _sweep_all_drives(self) -> None:
        with get_search_db() as session:
            rows = (
                session.query(IndexedFile.drive)
                .filter(IndexedFile.active.is_(True))
                .distinct()
                .all()
            )
        drives = [r[0] for r in rows]
        for drive in drives:
            await self._compute_for_drive(drive)

    async def _compute_for_drive(self, drive: str) -> None:
        try:
            history_rows = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _fetch_watch_history(drive)
            )
        except Exception:
            logger.warning("Pickup: could not read watch_history for drive=%s", drive)
            return

        if not history_rows:
            return

        viewer_files: dict[str, list[str]] = defaultdict(list)
        for viewer_id, file_id in history_rows:
            viewer_files[viewer_id].append(file_id)

        for viewer_id, file_ids in viewer_files.items():
            await self._compute_for_viewer(drive, viewer_id, file_ids)

    async def _compute_for_viewer(
        self, drive: str, viewer_id: str, recent_file_ids: list[str]
    ) -> None:
        checkpoint = _checkpoint(recent_file_ids)

        with get_search_db() as session:
            existing = (
                session.query(PickupCache)
                .filter_by(drive_id=drive, viewer_id=viewer_id)
                .first()
            )
            if existing and existing.watch_history_checkpoint == checkpoint:
                return

        seed_ids = recent_file_ids[:_SEED_LIMIT]
        recently_watched = set(recent_file_ids)

        candidates: dict[str, float] = {}
        for seed_id in seed_ids:
            try:
                results = await asyncio.get_event_loop().run_in_executor(
                    None, lambda sid=seed_id, d=drive: _find_similar_sync(sid, d)
                )
            except Exception:
                logger.debug("Pickup: find_similar failed for file=%s", seed_id)
                continue
            for file_id, score in results:
                if file_id not in recently_watched:
                    candidates[file_id] = max(candidates.get(file_id, 0.0), score)

        if not candidates:
            return

        top_ids = sorted(candidates, key=lambda x: candidates[x], reverse=True)[
            :_RESULT_LIMIT
        ]

        with get_search_db() as session:
            existing = (
                session.query(PickupCache)
                .filter_by(drive_id=drive, viewer_id=viewer_id)
                .first()
            )
            now = datetime.now(UTC)
            if existing:
                existing.file_ids = json.dumps(top_ids)
                existing.computed_at = now
                existing.watch_history_checkpoint = checkpoint
            else:
                session.add(
                    PickupCache(
                        drive_id=drive,
                        viewer_id=viewer_id,
                        file_ids=json.dumps(top_ids),
                        computed_at=now,
                        watch_history_checkpoint=checkpoint,
                    )
                )
            session.commit()

        logger.debug(
            "Pickup: computed %d recommendations for viewer=%s drive=%s",
            len(top_ids), viewer_id, drive,
        )


def _fetch_watch_history(drive: str) -> list[tuple[str, str]]:
    """Return [(viewer_id, file_id)] ordered by last_played_at DESC for a drive."""
    try:
        with get_litloft_db() as session:
            rows = session.execute(
                text(
                    """
                    SELECT wh.viewer_id, wh.file_id
                    FROM watch_history wh
                    JOIN files f ON wh.file_id = f.id
                    WHERE f.drive = :drive
                      AND f.deleted_at IS NULL
                      AND f.missing_since IS NULL
                    ORDER BY wh.last_played_at DESC
                    LIMIT :limit
                    """
                ),
                {"drive": drive, "limit": _HISTORY_LOOKBACK},
            ).fetchall()
        return [(r[0], r[1]) for r in rows]
    except Exception:
        logger.warning("Pickup: litloft DB unavailable for drive=%s", drive)
        return []


def _find_similar_sync(file_id: str, drive: str) -> list[tuple[str, float]]:
    """Synchronous KNN call. Run in executor to avoid blocking the event loop."""
    from app.search import find_similar

    result = find_similar(file_id=file_id, limit=_CANDIDATES_PER_SEED, drive=drive)
    return [(r.file_id, r.score) for r in result.results]


def _checkpoint(file_ids: list[str]) -> str:
    """16-char MD5 of the first N file_ids — cheap change detection."""
    key = ",".join(file_ids[:20])
    return hashlib.md5(key.encode()).hexdigest()[:16]
