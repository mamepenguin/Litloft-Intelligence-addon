"""Background worker that builds each viewer's Pickup feed.

Triggered by the ``scan.complete`` webhook (per drive) and by an hourly
sweep over every indexed drive.

Per drive:

  1. Load the candidate rows for each channel, once. They do not depend
     on the viewer, so every viewer of the drive scores against the same
     matrices.
  2. Per viewer, read their history twice — the whole set of file ids
     they have opened, which bounds nothing, and the vectors for the
     window, which is capped.
  3. Skip if their history has not moved since the last run.
  4. Cluster the window into weighted lanes, score every lane in one
     matmul per channel, interleave, and store the result as rows.

The shape this replaced seeded a k-nearest-neighbour search from the
five most recently watched files. Two things were wrong with it: the
query was "more of what I just watched", so a binge decided everything
after it; and sqlite-vec caps k at 4096 rows, which against this index
is under one percent of either vector table, nearly all of it already
watched. Neither was a tuning problem.
"""

import asyncio
import hashlib
import logging
from datetime import UTC, datetime

from app.database import get_search_db
from app.models import IndexedFile, PickupItem, PickupProfile
from app.pickup import feed as feed_mod
from app.pickup import profile as profile_mod
from app.pickup.retrieval import CandidateSet, load_candidates, score_lanes

logger = logging.getLogger(__name__)

#: Rows held per viewer. Deeper than anyone scrolls, and cheap: the feed
#: is a few hundred short rows.
_FEED_DEPTH = 300

#: Candidates asked of each lane before interleaving. A lane can only
#: contribute what it is handed, so this is the ceiling on how much of
#: the feed a single interest could fill.
_CANDIDATES_PER_LANE = 120

_PERIODIC_INTERVAL_SECONDS = 3600


class PickupWorker:
    """Builds per-viewer Pickup feeds for every indexed drive."""

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
        for (drive,) in rows:
            await self._guarded_compute(drive)

    async def _compute_for_drive(self, drive: str) -> None:
        loop = asyncio.get_event_loop()

        try:
            viewers = await loop.run_in_executor(
                None, lambda: profile_mod.viewer_ids(drive)
            )
        except Exception:
            logger.warning("Pickup: could not read watch_history for drive=%s", drive)
            return
        if not viewers:
            return

        # Viewer-independent, so built once and scored against by all of
        # them. On a drive with several viewers this is the difference
        # between one read per channel and one per viewer per lane.
        candidates: dict[str, CandidateSet] = {}
        for channel in profile_mod.CHANNELS:
            try:
                candidates[channel] = await loop.run_in_executor(
                    None, lambda c=channel: load_candidates(drive=drive, channel=c)
                )
            except Exception:
                logger.exception(
                    "Pickup: could not load candidates drive=%s channel=%s",
                    drive, channel,
                )

        for viewer_id in viewers:
            try:
                await loop.run_in_executor(
                    None,
                    lambda v=viewer_id: self._compute_for_viewer(
                        drive, v, candidates
                    ),
                )
            except Exception:
                logger.exception(
                    "Pickup: failed for drive=%s viewer=%s", drive, viewer_id,
                )

    def _compute_for_viewer(
        self,
        drive: str,
        viewer_id: str,
        candidates: dict[str, CandidateSet],
    ) -> None:
        watched = profile_mod.watched_file_ids(drive, viewer_id)
        if not watched:
            return

        checkpoint = _checkpoint(watched)
        with get_search_db() as session:
            existing = (
                session.query(PickupProfile)
                .filter_by(drive_id=drive, viewer_id=viewer_id)
                .first()
            )
            if existing and existing.watch_history_checkpoint == checkpoint:
                return

        history = profile_mod.profile_history(drive, viewer_id)
        lanes = profile_mod.build_lanes(history, key=f"{drive}\x1f{viewer_id}")
        if not lanes:
            _store(drive, viewer_id, [], checkpoint)
            return

        scored: dict[str, list[tuple[str, float]]] = {}
        for channel, channel_lanes in _by_channel(lanes).items():
            candidate_set = candidates.get(channel)
            if candidate_set is None or len(candidate_set) == 0:
                continue
            results = score_lanes(
                candidate_set,
                [lane.centroid for lane in channel_lanes],
                exclude_file_ids=watched,
                limit=_CANDIDATES_PER_LANE,
            )
            for lane, hits in zip(channel_lanes, results, strict=True):
                scored[lane.cluster_id] = list(hits)

        items = feed_mod.interleave(lanes, scored, depth=_FEED_DEPTH)
        _store(drive, viewer_id, items, checkpoint)
        logger.debug(
            "Pickup: %d items across %d lanes for viewer=%s drive=%s",
            len(items), len(lanes), viewer_id, drive,
        )


def _by_channel(lanes):
    """Group lanes by channel, preserving their order within each."""
    grouped: dict[str, list] = {}
    for lane in lanes:
        grouped.setdefault(lane.channel, []).append(lane)
    return grouped


def _store(drive: str, viewer_id: str, items, checkpoint: str) -> None:
    """Replace this viewer's feed in one transaction.

    Delete-then-insert rather than a diff: ranks are positional, so a
    partial update would leave rows from two different rankings
    interleaved under one ordering.
    """
    now = datetime.now(UTC)
    with get_search_db() as session:
        session.query(PickupItem).filter_by(
            drive_id=drive, viewer_id=viewer_id,
        ).delete(synchronize_session=False)

        for position, item in enumerate(items, start=1):
            session.add(PickupItem(
                drive_id=drive,
                viewer_id=viewer_id,
                rank=position,
                file_id=item.file_id,
                cluster_id=item.cluster_id,
                channel=item.channel,
                score=item.score,
            ))

        header = (
            session.query(PickupProfile)
            .filter_by(drive_id=drive, viewer_id=viewer_id)
            .first()
        )
        if header is None:
            header = PickupProfile(drive_id=drive, viewer_id=viewer_id)
            session.add(header)
        header.total = len(items)
        header.computed_at = now
        header.watch_history_checkpoint = checkpoint
        session.commit()


def _checkpoint(watched_file_ids) -> str:
    """A hash of what this viewer has opened.

    Over the whole set rather than a recent slice, and sorted, so it
    moves when anything is watched and not merely when the newest
    changes. The previous version hashed the first twenty of a
    drive-wide list, which for a second viewer barely moved at all.
    """
    joined = ",".join(sorted(watched_file_ids))
    return hashlib.md5(joined.encode()).hexdigest()[:32]
