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
from collections.abc import Sequence
from datetime import UTC, datetime

from app.database import get_search_db, get_search_db_read
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
        #: Drives whose trigger arrived while a run held their lock.
        self._missed_triggers: set[str] = set()

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
            # A scan finishing mid-sweep is exactly when the feed is
            # most out of date, so dropping the trigger silently is the
            # wrong kind of quiet. The run in progress may already have
            # read the index before the scan committed, and the next
            # periodic sweep is up to an hour away.
            self._missed_triggers.add(drive)
            logger.info(
                "Pickup: drive=%s already computing; will recompute after",
                drive,
            )
            return
        async with lock:
            while True:
                self._missed_triggers.discard(drive)
                try:
                    await self._compute_for_drive(drive)
                except Exception:
                    logger.exception("Pickup worker failed for drive=%s", drive)
                    return
                if drive not in self._missed_triggers:
                    return
                logger.info("Pickup: recomputing drive=%s for a missed trigger", drive)

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

        # Decide who needs rebuilding *before* touching the index. The
        # candidate matrices cost a read of every vector in the drive,
        # and on a quiet hour — the common case — none of it would be
        # used. The staleness check is one small query per viewer.
        pending: list[tuple[str, set[str], str]] = []
        for viewer_id in viewers:
            try:
                stale = await loop.run_in_executor(
                    None, lambda v=viewer_id: self._stale_work(drive, v)
                )
            except Exception:
                logger.exception(
                    "Pickup: could not read history drive=%s viewer=%s",
                    drive, viewer_id,
                )
                continue
            if stale is not None:
                pending.append(stale)

        if not pending:
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
                # Abandon the whole sweep for this drive rather than
                # build feeds from the channels that happened to load.
                # A partial feed is not merely thinner: it would be
                # stored with a fresh checkpoint, and the next sweep
                # would then skip the viewer as up to date. One
                # transient error would outlive itself until the viewer
                # watched something new.
                logger.exception(
                    "Pickup: could not load candidates drive=%s channel=%s; "
                    "skipping this sweep",
                    drive, channel,
                )
                return

        for viewer_id, watched, checkpoint in pending:
            try:
                await loop.run_in_executor(
                    None,
                    lambda v=viewer_id, w=watched, c=checkpoint:
                        self._compute_for_viewer(drive, v, w, c, candidates),
                )
            except Exception:
                logger.exception(
                    "Pickup: failed for drive=%s viewer=%s", drive, viewer_id,
                )

    def _stale_work(
        self, drive: str, viewer_id: str,
    ) -> tuple[str, set[str], str] | None:
        """Return this viewer's pending work, or None if it is current."""
        signature = profile_mod.watch_signature(drive, viewer_id)
        if not signature:
            return None
        watched = {file_id for file_id, _ in signature}

        checkpoint = _checkpoint(signature)
        with get_search_db_read() as session:
            existing = (
                session.query(PickupProfile)
                .filter_by(drive_id=drive, viewer_id=viewer_id)
                .first()
            )
        if existing and existing.watch_history_checkpoint == checkpoint:
            return None
        return viewer_id, watched, checkpoint

    def _compute_for_viewer(
        self,
        drive: str,
        viewer_id: str,
        watched: set[str],
        checkpoint: str,
        candidates: dict[str, CandidateSet],
    ) -> None:
        history = profile_mod.profile_history(drive, viewer_id)
        lanes = profile_mod.build_lanes(history, key=f"{drive}\x1f{viewer_id}")
        if not lanes:
            _store(drive, viewer_id, [], None)
            return

        scored: dict[str, list[tuple[str, float]]] = {}
        for channel, channel_lanes in _by_channel(lanes).items():
            candidate_set = candidates.get(channel)
            if candidate_set is None or len(candidate_set) == 0:
                continue
            results = score_lanes(
                candidate_set,
                [lane.centroid for lane in channel_lanes],
                channel=channel,
                exclude_file_ids=watched,
                limit=_CANDIDATES_PER_LANE,
            )
            for lane, hits in zip(channel_lanes, results, strict=True):
                scored[lane.cluster_id] = list(hits)

        items = feed_mod.interleave(lanes, scored, depth=_FEED_DEPTH)
        # An empty feed is never settled. The checkpoint tracks the
        # viewer's history, not the state of the index, so storing one
        # against a checkpoint would strand a viewer whose files simply
        # had not been embedded yet: their history would not move, and
        # the sweep would keep skipping them. Recomputing an empty feed
        # each hour is one matmul against an already-loaded matrix.
        _store(drive, viewer_id, items, checkpoint if items else None)
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


def _checkpoint(signature: Sequence[tuple[str, str]]) -> str:
    """A hash of what this viewer has opened, and when.

    Three things have to move it, and each was missed by an earlier
    version of this function.

    *Anything watched*, not just the newest — a hash of the first twenty
    ids of a drive-wide recency list barely moved for a second viewer.
    So the whole set, sorted.

    *Anything re-watched.* Reopening a file already in the set changes
    no ids, but moves its recency, and recency is what the lane weights
    are made of. So the timestamps are in the hash too.

    *Time itself.* The profile reads a rolling year and weights it by a
    60-day half-life, both of which drift while the viewer does
    nothing at all. Without the date a viewer who stops watching is
    frozen at whatever their profile said the day they stopped. With
    it, a quiet viewer is recomputed once a day rather than never — an
    hourly sweep still skips them the other twenty-three times.
    """
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    joined = ";".join(f"{file_id}@{played}" for file_id, played in sorted(signature))
    return hashlib.md5(f"{stamp}|{joined}".encode()).hexdigest()[:32]
