"""Measure how well Pickup ranks, from the log or by replaying history.

Two readings of the same question, "did the feed hold what the viewer
went on to watch":

- ``online`` scores the feeds the worker actually stored, against the
  videos the viewer opened in the days after each was built. It answers
  what happened, and cannot be flattered by a simulation.
- ``replay`` rebuilds the feed at a past cut point and scores it against
  the plays that followed. It can compare configurations — documents in
  or out, engagement or not, and a grid of freshness settings on the
  worker's own lanes — because it computes all of them from one history.
  History keeps one row per file and engagement is today's, which makes
  it optimistic about content-based rankings; ``online`` is the check
  that the optimism did not decide the answer.

The replay must rank as the worker ranks, or the numbers describe a
program nobody runs. So it calls the worker's own profile, clustering,
scoring and interleaving functions, passing the cut point as the
reference time and video-only candidates where a configuration says so.

``interests`` asks a third question: which interests the profile found
in one viewer's history, and how that viewer treated each one. It prints
the lanes the worker would build at one instant, with the rows glanced
at, set aside or marked attributed to the lane they resemble, and the
candidates each lane would lead with. It measures; it decides nothing.

Everything is read: the commands open both databases ``mode=ro`` and
never migrate, create or log.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import text as sql_text

from app.database import (
    get_litloft_db,
    get_search_db_read,
    init_readonly_databases,
)
from app.pickup import feed as feed_mod
from app.pickup import freshness as freshness_mod
from app.pickup import profile as profile_mod
from app.pickup.retrieval import (
    CandidateSet,
    load_candidates,
    restrict_candidates,
    score_lanes,
)
from app.workers.pickup import (
    _CANDIDATES_PER_LANE,
    _FEED_DEPTH,
    _not_interested_ids,
)

#: The feeds compared by ``replay``, in report order. ``after`` is the
#: worker's path with freshness off: videos only, weighted by engagement.
_CONFIGS = (
    ("before", False, False),
    ("depth", False, True),
    ("video", True, False),
    ("after", True, True),
)

#: The freshness settings ``replay`` reports on top of the configurations:
#: ``(HALF_LIFE_DAYS, FLOOR)``. Each row re-scores ``after``'s lanes at the
#: cut point, so the grid costs no profile or clustering.
_GRID = tuple(
    (half_life, floor)
    for half_life in (7, 30, 90, 365)
    for floor in (0, 0.5)
)

_RECALL_DEPTHS = (12, 40)

#: A feed counts for a target opened no later than this after it was
#: built. Longer than the progress-save interval, short enough that the
#: feed still resembles the one the viewer was looking at.
_FEED_WINDOW = timedelta(hours=48)


@dataclass(frozen=True)
class _FileRow:
    """One core ``files`` row the commands need to place a candidate."""

    file_id: str
    file_type: str
    created_at: datetime | None
    live: bool


@dataclass(frozen=True)
class _HistoryRow:
    """One watch-history row with everything engagement and windows read."""

    file_id: str
    played_at: datetime | None
    engagement: float
    watched: bool
    file_type: str
    created_at: datetime | None
    live: bool

    @property
    def is_video(self) -> bool:
        return self.live and self.file_type == "video"


@dataclass
class _Accumulator:
    """Sums over the evaluated cut points of one configuration."""

    cuts: int = 0
    targets: int = 0
    recall12: float = 0.0
    recall40: float = 0.0
    hit40: float = 0.0


# ---------------------------------------------------------------------------
# Reading the core database
# ---------------------------------------------------------------------------

_FILES_SQL = """
SELECT id,
       file_type,
       created_at,
       CASE WHEN deleted_at IS NULL AND missing_since IS NULL
            THEN 1 ELSE 0 END AS live
FROM files
WHERE drive = :drive
"""


def _read_files(drive: str) -> dict[str, _FileRow]:
    """Every core file row of the drive, keyed by id."""
    with get_litloft_db() as session:
        rows = session.execute(
            sql_text(_FILES_SQL), {"drive": drive},
        ).fetchall()
    return {
        row[0]: _FileRow(
            file_id=row[0],
            file_type=row[1],
            created_at=profile_mod._parse_naive_utc(row[2]),
            live=bool(row[3]),
        )
        for row in rows
    }


_HISTORY_SQL = """
SELECT wh.file_id,
       wh.last_played_at,
       wh.dismissed_at,
       wh.playback_position,
       wh.duration,
       f.file_type,
       f.mime_type,
       f.created_at,
       CASE WHEN f.deleted_at IS NULL AND f.missing_since IS NULL
            THEN 1 ELSE 0 END AS live,
       CASE WHEN f.liked_at IS NOT NULL THEN 1 ELSE 0 END AS liked,
       COALESCE(f.is_favorite, 0) AS favourite,
       ps.segment_index,
       ps.played_seconds
FROM watch_history wh
JOIN files f ON wh.file_id = f.id
LEFT JOIN played_segments ps
       ON ps.file_id = wh.file_id AND ps.viewer_id = wh.viewer_id
WHERE f.drive = :drive AND wh.viewer_id = :viewer
ORDER BY wh.file_id, ps.segment_index
"""


def _read_history(drive: str, viewer_id: str) -> list[_HistoryRow]:
    """This viewer's history in the drive, engagement computed once."""
    with get_litloft_db() as session:
        rows = session.execute(
            sql_text(_HISTORY_SQL),
            {"drive": drive, "viewer": viewer_id},
        ).fetchall()

    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)

    history: list[_HistoryRow] = []
    for file_id, group in grouped.items():
        head = group[0]
        watch_row = profile_mod.WatchRow(
            file_id=file_id,
            played_at=profile_mod._parse_naive_utc(head[1]),
            dismissed=profile_mod._is_dismissed(head[2], head[1]),
            position=profile_mod._finite(head[3]),
            duration=profile_mod._finite(head[4]),
            file_type=head[5],
            mime_type=head[6],
            live=bool(head[8]),
            liked=bool(head[9]),
            favourite=bool(head[10]),
            played_seconds=sum(
                profile_mod._finite(row[12]) or 0.0 for row in group
            ),
            segments=tuple(
                (int(row[11]), profile_mod._finite(row[12]) or 0.0)
                for row in group
                if row[11] is not None
            ),
        )
        state = profile_mod.watch_state(watch_row)
        history.append(_HistoryRow(
            file_id=file_id,
            played_at=watch_row.played_at,
            engagement=state.engagement,
            watched=state.watched,
            file_type=watch_row.file_type,
            created_at=profile_mod._parse_naive_utc(head[7]),
            live=watch_row.live,
        ))
    return history


def _viewer_ids(drive: str, viewer_id: str | None) -> list[str]:
    """The viewers to report on, ordered by id."""
    ids = set(profile_mod.viewer_ids(drive))
    if viewer_id is not None:
        ids &= {viewer_id}
    return sorted(ids)


def _midnight_utc() -> datetime:
    return datetime.now(UTC).replace(
        tzinfo=None, hour=0, minute=0, second=0, microsecond=0,
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


_MARKS_SQL = """
SELECT file_id, marked_at
FROM pickup_not_interested
WHERE drive_id = :drive AND viewer_id = :viewer
"""


def _read_marks(drive: str, viewer_id: str) -> list[tuple[str, datetime | None]]:
    """This viewer's not-interested marks, with when each was made.

    A failure here aborts the command: a replay that ignored the marks
    would score a feed nobody was served.
    """
    with get_search_db_read() as session:
        rows = session.execute(
            sql_text(_MARKS_SQL),
            {"drive": drive, "viewer": viewer_id},
        ).fetchall()
    return [
        (row[0], profile_mod._parse_naive_utc(row[1]))
        for row in rows
    ]


def _opened_before(rows: list[_HistoryRow], t: datetime) -> set[str]:
    """Files the viewer opened before ``t``.

    An unparseable timestamp keeps its file in the exclusion set, and
    before every cut point: the row cannot date the open, and admitting
    it as a candidate would offer the viewer something they may have
    watched.
    """
    return {
        row.file_id
        for row in rows
        if row.played_at is None or row.played_at < t
    }


def _excluded_before(rows: list[_HistoryRow], t: datetime) -> set[str]:
    """Files the feed must not hold at ``t`` under SPEC-ADDON-149.

    A row whose timestamp does not parse cannot date the open, so it is
    judged as already opened: the same treatment ``_opened_before``
    gives it.
    """
    return {
        row.file_id
        for row in rows
        if (row.played_at is None or row.played_at < t)
        and (row.file_type != "video" or row.watched)
    }


def _targets(
    rows: list[_HistoryRow], t: datetime, horizon: timedelta,
) -> list[_HistoryRow]:
    """Live videos created before ``t`` and first opened in ``[t, t+horizon)``."""
    end = t + horizon
    return [
        row
        for row in rows
        if row.is_video
        and row.created_at is not None
        and row.created_at < t
        and row.played_at is not None
        and t <= row.played_at < end
        and row.engagement > 0.0
    ]


def _allowed_files(
    files: dict[str, _FileRow], opened: set[str], t: datetime,
    *, videos_only: bool,
) -> set[str]:
    """Files that may be ranked at ``t``: present, not opened, video by need."""
    return {
        file_id
        for file_id, file_row in files.items()
        if file_row.live
        and file_row.created_at is not None
        and file_row.created_at < t
        and file_id not in opened
        and (not videos_only or file_row.file_type == "video")
    }


def _universe(
    candidates: dict[str, CandidateSet], allowed: set[str],
) -> dict[str, CandidateSet]:
    return {
        channel: restrict_candidates(candidate_set, allowed)
        for channel, candidate_set in candidates.items()
    }


def _history_vectors(
    rows: list[_HistoryRow],
) -> dict[str, dict[str, np.ndarray]]:
    """One vector per history file and channel, loaded once per viewer.

    The replay walks dozens of cut points over one history. Reloading
    the vectors at each would repeat the same reads from the virtual
    tables and, for ``text_content``, the same per-file chunk means.
    """
    file_ids = {
        row.file_id
        for row in rows
        if row.live and row.played_at is not None
    }
    return {
        channel: profile_mod.representative_vectors(file_ids, channel)
        for channel in profile_mod.CHANNELS
    }


def _vector_lookup(cache: dict[str, dict[str, np.ndarray]]):
    def lookup(file_ids, channel):
        loaded = cache.get(channel, {})
        return {
            file_id: loaded[file_id]
            for file_id in file_ids
            if file_id in loaded
        }

    return lookup


def _rank_lanes(
    lanes,
    candidates: dict[str, CandidateSet],
    *,
    opened: set[str],
    freshness: dict[str, float] | None = None,
) -> list[str]:
    """The feed the worker would store from ``lanes`` at this cut point."""
    if not lanes:
        return []

    grouped: dict[str, list] = {}
    for lane in lanes:
        grouped.setdefault(lane.channel, []).append(lane)

    scored: dict[str, list[tuple[str, float, float]]] = {}
    for channel, channel_lanes in grouped.items():
        candidate_set = candidates.get(channel)
        if candidate_set is None or len(candidate_set) == 0:
            continue
        results = score_lanes(
            candidate_set,
            [lane.centroid for lane in channel_lanes],
            channel=channel,
            exclude_file_ids=opened,
            limit=_CANDIDATES_PER_LANE,
            freshness=freshness,
        )
        for lane, hits in zip(channel_lanes, results, strict=True):
            scored[lane.cluster_id] = list(hits)

    return [
        item.file_id
        for item in feed_mod.interleave(lanes, scored, depth=_FEED_DEPTH)
    ]


def _grid_names() -> list[str]:
    return [f"fresh_h{half_life:g}_f{floor:g}" for half_life, floor in _GRID]


def _marked_name() -> str:
    """The report row whose constants the worker ships with."""
    if freshness_mod.FLOOR == 1:
        return "after"
    return (
        f"fresh_h{freshness_mod.HALF_LIFE_DAYS:g}"
        f"_f{freshness_mod.FLOOR:g}"
    )


def _reported_names() -> tuple[str, ...]:
    """Every row the report prints, with the shipped one marked."""
    names = (
        *(name for name, *_ in _CONFIGS),
        *_grid_names(),
        "random",
        "newest",
    )
    marked = _marked_name()
    return tuple(name + "*" if name == marked else name for name in names)


def _random_order(
    file_ids: set[str], *, drive: str, viewer_id: str, t: datetime,
) -> list[str]:
    """A reproducible shuffle, so two runs of a report agree."""
    ordered = sorted(file_ids)
    seed = profile_mod._seed_from(drive, viewer_id, t.isoformat(sep=" "))
    random.Random(seed).shuffle(ordered)
    return ordered


def _newest_order(
    file_ids: set[str], files: dict[str, _FileRow],
) -> list[str]:
    """Newest core row first, ties by file id ascending."""
    ordered = sorted(file_ids)
    ordered.sort(key=lambda file_id: files[file_id].created_at, reverse=True)
    return ordered


def _rank_configs(
    drive: str,
    viewer_id: str,
    rows: list[_HistoryRow],
    files: dict[str, _FileRow],
    candidates: dict[str, CandidateSet],
    vectors,
    *,
    t: datetime,
    marks: list[tuple[str, datetime | None]] = (),
) -> tuple[dict[str, list[str]], set[str]]:
    """One ranking per configuration at ``t``, and the video universe."""
    opened = _opened_before(rows, t)
    allowed_all = _allowed_files(files, opened, t, videos_only=False)
    universe_all = _universe(candidates, allowed_all)
    allowed_videos = _allowed_files(files, opened, t, videos_only=True)
    universe_video = (
        universe_all if allowed_videos == allowed_all
        else _universe(candidates, allowed_videos)
    )
    excluded_now = _excluded_before(rows, t)
    # A mark hides its file only for cut points after it was made: an
    # earlier feed could not have known about it, and scoring one as if
    # it had would flatter the row. An unparseable mark's time cannot
    # place it, so it is not used to exclude anything.
    refused = {
        file_id
        for file_id, marked_at in marks
        if marked_at is not None and marked_at < t
    }
    excluded_after = excluded_now | refused
    allowed_after = _allowed_files(files, excluded_after, t, videos_only=True)
    universe_after = (
        universe_video if allowed_after == allowed_videos
        else _universe(candidates, allowed_after)
    )

    rankings: dict[str, list[str]] = {}
    lanes_by_config: dict[str, list] = {}
    for name, videos_only, weighted in _CONFIGS:
        if name == "after":
            exclusion, universe = excluded_after, universe_after
        else:
            exclusion = opened
            universe = universe_video if videos_only else universe_all
        history = profile_mod.profile_history(
            drive, viewer_id, now=t, until=t,
            videos_only=videos_only, weighted=weighted,
            version_4=(name == "depth"),
        )
        lanes = profile_mod.build_lanes(
            history, key=f"{drive}\x1f{viewer_id}", now=t, vectors=vectors,
        )
        lanes_by_config[name] = lanes
        rankings[name] = _rank_lanes(lanes, universe, opened=exclusion)

    voted = {
        file_id
        for candidate_set in universe_video.values()
        for file_id in candidate_set.file_ids
    }

    # Freshness does not touch the profile, so every grid row re-scoring
    # ``after``'s lanes is the same ranking the worker would build with
    # those constants at this cut point.
    for half_life, floor in _GRID:
        mapping = {
            file_id: freshness_mod.freshness(
                files[file_id].created_at, t,
                half_life_days=half_life, floor=floor,
            )
            for file_id in voted
        }
        rankings[f"fresh_h{half_life:g}_f{floor:g}"] = _rank_lanes(
            lanes_by_config["after"],
            universe_after,
            opened=excluded_after,
            freshness=mapping,
        )

    marked = _marked_name()
    if marked in rankings:
        rankings[f"{marked}*"] = rankings.pop(marked)

    rankings["random"] = _random_order(
        voted, drive=drive, viewer_id=viewer_id, t=t,
    )
    rankings["newest"] = _newest_order(voted, files)
    return rankings, voted


def _cut_rankings(
    drive: str,
    viewer_id: str,
    rows: list[_HistoryRow],
    files: dict[str, _FileRow],
    candidates: dict[str, CandidateSet],
    vectors,
    *,
    t: datetime,
    horizon: timedelta,
    marks: list[tuple[str, datetime | None]] = (),
) -> tuple[dict[str, list[str]], list[_HistoryRow], int] | None:
    """One ranking per configuration at ``t``, or None with no target."""
    targets = _targets(rows, t, horizon)
    if not targets:
        return None
    rankings, voted = _rank_configs(
        drive, viewer_id, rows, files, candidates, vectors, t=t, marks=marks,
    )
    return rankings, targets, len(voted)


def _metrics(
    targets: list[_HistoryRow], ranking: list[str],
) -> tuple[int, float, float, float]:
    target_ids = {row.file_id for row in targets}
    found12 = len(target_ids & set(ranking[:_RECALL_DEPTHS[0]]))
    found40 = len(target_ids & set(ranking[:_RECALL_DEPTHS[1]]))
    count = len(target_ids)
    return (
        count,
        found12 / count,
        found40 / count,
        1.0 if found40 else 0.0,
    )


def _replay_viewer(
    drive: str,
    viewer_id: str,
    files: dict[str, _FileRow],
    candidates: dict[str, CandidateSet],
    *,
    t0: datetime,
    days: int,
    horizon: timedelta,
) -> str:
    rows = _read_history(drive, viewer_id)
    marks = _read_marks(drive, viewer_id)
    vectors = _vector_lookup(_history_vectors(rows))

    cut_points = [
        t0 - horizon - timedelta(days=offset)
        for offset in range(days)
    ]
    names = _reported_names()
    stats = {name: _Accumulator() for name in names}
    skipped = 0
    universe_sizes: dict[datetime, int] = {}

    for t in cut_points:
        result = _cut_rankings(
            drive, viewer_id, rows, files, candidates, vectors,
            t=t, horizon=horizon, marks=marks,
        )
        if result is None:
            skipped += 1
            continue
        rankings, targets, universe_size = result
        universe_sizes[t] = universe_size
        for name in names:
            count, recall12, recall40, hit40 = _metrics(
                targets, rankings[name],
            )
            accumulator = stats[name]
            accumulator.cuts += 1
            accumulator.targets += count
            accumulator.recall12 += recall12
            accumulator.recall40 += recall40
            accumulator.hit40 += hit40

    earliest = (
        str(universe_sizes[min(universe_sizes)]) if universe_sizes else "-"
    )
    latest = (
        str(universe_sizes[max(universe_sizes)]) if universe_sizes else "-"
    )

    lines = [
        f"viewer {viewer_id}",
        f"candidate videos: {earliest} .. {latest}",
        "",
        "| config | cuts | skipped | targets | recall@12 | recall@40 | hit@40 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in names:
        accumulator = stats[name]
        lines.append(
            f"| {name} | {accumulator.cuts} | {skipped} | "
            f"{accumulator.targets} | "
            f"{_mean(accumulator.recall12, accumulator.cuts)} | "
            f"{_mean(accumulator.recall40, accumulator.cuts)} | "
            f"{_mean(accumulator.hit40, accumulator.cuts)} |"
        )
    return "\n".join(lines)


def _mean(total: float, count: int) -> str:
    """A ratio with three decimals, or ``-`` when nothing was evaluated."""
    return "-" if count == 0 else f"{total / count:.3f}"


# ---------------------------------------------------------------------------
# Online
# ---------------------------------------------------------------------------

_FEED_LOG_SQL = """
SELECT computed_at, profile_version, file_ids
FROM pickup_feed_log
WHERE drive_id = :drive AND viewer_id = :viewer
ORDER BY computed_at
"""


def _feed_log_exists() -> bool:
    with get_search_db_read() as session:
        row = session.execute(
            sql_text(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'pickup_feed_log'"
            )
        ).fetchone()
    return row is not None


def _read_feeds(
    drive: str, viewer_id: str,
) -> list[tuple[datetime, int, list[str]]]:
    """This viewer's logged feeds for the drive, oldest first."""
    with get_search_db_read() as session:
        rows = session.execute(
            sql_text(_FEED_LOG_SQL),
            {"drive": drive, "viewer": viewer_id},
        ).fetchall()

    feeds: list[tuple[datetime, int, list[str]]] = []
    for computed_at, version, file_ids in rows:
        parsed = profile_mod._parse_naive_utc(computed_at)
        if parsed is None:
            continue
        try:
            ids = json.loads(file_ids)
        except (TypeError, ValueError):
            continue
        if not isinstance(ids, list):
            continue
        feeds.append((parsed, int(version), [str(file_id) for file_id in ids]))
    return feeds


def _online_targets(
    drive: str, viewer_id: str, *, now: datetime, days: int,
) -> list[_HistoryRow]:
    """Live videos opened inside the window with engagement above zero."""
    lower = now - timedelta(days=days)
    upper = now + timedelta(days=profile_mod._FUTURE_TOLERANCE_DAYS)
    return [
        row
        for row in _read_history(drive, viewer_id)
        if row.is_video
        and row.played_at is not None
        and lower <= row.played_at <= upper
        and row.engagement > 0.0
    ]


def _week_of(played_at: datetime) -> str:
    year, week, _ = played_at.isocalendar()
    return f"{year}-W{week:02d}"


def _online_row(
    targets: list[_HistoryRow],
    feeds: list[tuple[datetime, int, list[str]]],
    *,
    week: str,
) -> tuple[str, int, int, str, str, str]:
    """Score one week, or the total, of ``targets`` against ``feeds``."""
    evaluated = 0
    unscored = 0
    hits12 = 0
    hits40 = 0
    versions: set[int] = set()

    for target in targets:
        relevant = [
            (version, ids)
            for computed_at, version, ids in feeds
            if target.played_at - _FEED_WINDOW <= computed_at < target.played_at
        ]
        if not relevant:
            unscored += 1
            continue
        evaluated += 1
        if any(target.file_id in ids[:12] for _, ids in relevant):
            hits12 += 1
        if any(target.file_id in ids for _, ids in relevant):
            hits40 += 1
        versions.update(version for version, _ in relevant)

    version_cell = (
        ",".join(str(version) for version in sorted(versions))
        if versions else "-"
    )
    return (
        week,
        evaluated,
        unscored,
        _mean(float(hits12), evaluated),
        _mean(float(hits40), evaluated),
        version_cell,
    )


def _online_viewer(
    drive: str, viewer_id: str, *, days: int, now: datetime,
) -> str:
    targets = _online_targets(drive, viewer_id, now=now, days=days)
    feeds = _read_feeds(drive, viewer_id)

    by_week: dict[tuple[int, int], list[_HistoryRow]] = {}
    for target in targets:
        iso = target.played_at.isocalendar()
        by_week.setdefault((iso.year, iso.week), []).append(target)

    lines = [
        f"viewer {viewer_id}",
        "",
        "| week | targets | unscored | recall@12 | recall@40 | versions |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    rows: list[tuple[str, int, int, str, str, str]] = []
    for key in sorted(by_week):
        week_targets = by_week[key]
        rows.append(_online_row(
            week_targets, feeds, week=_week_of(week_targets[0].played_at),
        ))
    rows.append(_online_row(targets, feeds, week="total"))
    for row in rows:
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | "
            f"{row[3]} | {row[4]} | {row[5]} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interests
# ---------------------------------------------------------------------------

#: The classes a kept history row falls into, in the counts line's order.
#: Each row is in exactly one, and the class is not the same thing as
#: ``watch_state``: a row with no duration cannot be judged by it, and a
#: dismissal is read before one is tried.
_ROW_CLASSES = ("watched", "glanced", "set aside", "no length", "legacy zero")

#: The classes attributed to a lane, in the lane table's order.
_ATTRIBUTED = ("glanced", "set aside", "marked")

_TITLES_SQL = """
SELECT id, title, filename
FROM files
WHERE drive = :drive
"""


def _read_titles(drive: str) -> dict[str, str]:
    """What to print for each file of the drive: title, filename, else id."""
    with get_litloft_db() as session:
        rows = session.execute(
            sql_text(_TITLES_SQL), {"drive": drive},
        ).fetchall()
    return {row[0]: row[1] or row[2] or row[0] for row in rows}


def _row_class(row: profile_mod.WatchRow, now: datetime) -> str | None:
    """The one class this history row belongs to, or None if it is not kept.

    The filters before the class are exactly ``profile_history``'s before
    weighting, and are applied without its cap: the report says how much
    of the history is which class, and a cap would hide the rest. A
    liked or favourite row is watched before anything else is asked, and
    a row played before recording began is judged by engagement alone:
    neither the dismissal nor the duration rule decides those.
    """
    if not row.live or row.file_type != "video":
        return None
    played = row.played_at
    if played is None:
        return None
    age = profile_mod.age_days(played, now=now)
    if (
        age < -profile_mod._FUTURE_TOLERANCE_DAYS
        or age > profile_mod.PROFILE_WINDOW_DAYS
    ):
        return None

    if row.liked or row.favourite:
        return "watched"
    if played < profile_mod.RECORDING_START:
        engagement = profile_mod.watch_state(row).engagement
        return "watched" if engagement > 0.0 else "legacy zero"
    if row.dismissed:
        return "set aside"
    if profile_mod._known_duration(row.duration) is None:
        return "no length"
    if profile_mod.watch_state(row).watched:
        return "watched"
    return "glanced"


def _interests_viewer(
    drive: str,
    viewer_id: str,
    *,
    now: datetime,
    live_ids: set[str],
    candidates: dict[str, CandidateSet],
    freshness: dict[str, float],
    titles: dict[str, str],
    titles_limit: int,
    picks_limit: int,
) -> str:
    """One viewer's report: counts, the lane table, and a block per lane."""
    history = profile_mod.profile_history(drive, viewer_id, now=now)
    lanes = profile_mod.cluster_lanes(
        history, key=f"{drive}\x1f{viewer_id}", now=now,
    )
    engagements = {entry.file_id: entry.engagement for entry in history}
    played = {entry.file_id: entry.last_played_at for entry in history}

    counts = dict.fromkeys(_ROW_CLASSES, 0)
    glanced: list[str] = []
    set_aside: list[str] = []
    for row in profile_mod.watch_rows(drive, viewer_id):
        row_class = _row_class(row, now)
        if row_class is None:
            continue
        counts[row_class] += 1
        if row_class == "glanced":
            glanced.append(row.file_id)
        elif row_class == "set aside":
            set_aside.append(row.file_id)

    marked = _not_interested_ids(drive, viewer_id)
    marked_videos = sorted(marked & live_ids)

    counts_line = (
        f"watched {counts['watched']}, glanced {counts['glanced']}, "
        f"set aside {counts['set aside']}, no length {counts['no length']}, "
        f"legacy zero {counts['legacy zero']}, marked {len(marked_videos)}"
    )
    if len(history) == profile_mod.PROFILE_VECTOR_CAP:
        counts_line += f"\nmembers capped at {profile_mod.PROFILE_VECTOR_CAP}"
    blocks = [f"viewer {viewer_id}", counts_line]
    if not lanes:
        blocks.append("no lanes")
        return "\n\n".join(blocks)

    # One read per type over every attributed class: a file can be both
    # glanced at and marked.
    groups = {"glanced": glanced, "set aside": set_aside, "marked": marked_videos}
    attributed = {
        lane.cluster_id: dict.fromkeys(_ATTRIBUTED, 0) for lane, _ in lanes
    }
    for channel in profile_mod.CHANNELS:
        channel_lanes = [pair for pair in lanes if pair[0].channel == channel]
        if not channel_lanes:
            continue
        file_ids = sorted({f for ids in groups.values() for f in ids})
        vectors = profile_mod.representative_vectors(file_ids, channel)
        if not vectors:
            continue
        centroids = np.stack([lane.centroid for lane, _ in channel_lanes])
        nearest = {
            file_id: channel_lanes[int(np.argmax(centroids @ vector))][0]
            for file_id, vector in vectors.items()
        }
        for label, ids in groups.items():
            for file_id in ids:
                lane = nearest.get(file_id)
                if lane is not None:
                    attributed[lane.cluster_id][label] += 1

    scored: dict[str, list[tuple[str, float, float]]] = {}
    exclusions = profile_mod.watched_file_ids(drive, viewer_id) | marked
    for channel in profile_mod.CHANNELS:
        channel_lanes = [pair for pair in lanes if pair[0].channel == channel]
        candidate_set = candidates.get(channel)
        if not channel_lanes or candidate_set is None or len(candidate_set) == 0:
            continue
        results = score_lanes(
            candidate_set,
            [lane.centroid for lane, _ in channel_lanes],
            channel=channel,
            exclude_file_ids=exclusions,
            limit=_CANDIDATES_PER_LANE,
            freshness=freshness,
        )
        for (lane, _), hits in zip(channel_lanes, results, strict=True):
            scored[lane.cluster_id] = hits

    table = [
        "| # | type | weight | members | 14d | 90d | median age | recorded"
        " | glanced | glance rate | set aside | marked |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lane_blocks: list[str] = []
    for number, (lane, member_vectors) in enumerate(lanes, start=1):
        ages = [profile_mod.age_days(played[f], now=now) for f in member_vectors]
        recorded = sum(
            1
            for f in member_vectors
            if played[f] >= profile_mod.RECORDING_START
        )
        lane_counts = attributed[lane.cluster_id]
        judged = recorded + lane_counts["glanced"]
        rate = f"{lane_counts['glanced'] / judged:.3f}" if judged else "-"
        table.append(
            f"| {number} | {lane.channel} | {lane.weight:.3f} | "
            f"{lane.member_count} | "
            f"{sum(1 for age in ages if age <= 14)} | "
            f"{sum(1 for age in ages if age <= 90)} | "
            f"{statistics.median(ages):.1f} | {recorded} | "
            f"{lane_counts['glanced']} | {rate} | "
            f"{lane_counts['set aside']} | {lane_counts['marked']} |"
        )

        lines = [f"### {number} {lane.channel}"]
        if titles_limit:
            ordered = sorted(
                member_vectors.items(),
                key=lambda item: (-float(lane.centroid @ item[1]), item[0]),
            )
            chosen = [
                f"- {titles.get(f, f)} ({engagements[f]:.2f})"
                for f, _ in ordered[:titles_limit]
            ]
            lines += ["members", *(chosen or ["- none"])]
        if picks_limit:
            chosen = [
                f"- {titles.get(hit[0], hit[0])}"
                for hit in scored.get(lane.cluster_id, [])[:picks_limit]
            ]
            lines += ["picks", *(chosen or ["- none"])]
        lane_blocks.append("\n".join(lines))

    blocks.append("\n".join(table))
    blocks += lane_blocks
    return "\n\n".join(blocks)


def _interests(
    drive: str, viewer_id: str | None, *, titles: int, picks: int,
) -> int:
    viewers = _viewer_ids(drive, viewer_id)
    if not viewers:
        print("no viewers")
        return 0

    # One instant for the window, the lanes, the ages and freshness: the
    # report compares lanes against the ages of the rows that made them.
    now = freshness_mod.reference_now()
    videos = profile_mod.live_videos(drive)
    live_ids = set(videos)
    candidates = {
        channel: restrict_candidates(
            load_candidates(drive=drive, channel=channel), live_ids,
        )
        for channel in profile_mod.CHANNELS
    }
    freshness = {
        file_id: freshness_mod.freshness(created_at, now)
        for file_id, created_at in videos.items()
    }
    names = _read_titles(drive)

    reports = [
        _interests_viewer(
            drive, viewer,
            now=now, live_ids=live_ids, candidates=candidates,
            freshness=freshness, titles=names,
            titles_limit=titles, picks_limit=picks,
        )
        for viewer in viewers
    ]
    print("\n\n".join(reports))
    return 0


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def _replay(drive: str, viewer_id: str | None, *, days: int, horizon: int) -> int:
    viewers = _viewer_ids(drive, viewer_id)
    if not viewers:
        print("no viewers")
        return 0

    files = _read_files(drive)
    candidates = {
        channel: load_candidates(drive=drive, channel=channel)
        for channel in profile_mod.CHANNELS
    }
    t0 = _midnight_utc()
    reports = [
        _replay_viewer(
            drive, viewer, files, candidates,
            t0=t0, days=days, horizon=timedelta(days=horizon),
        )
        for viewer in viewers
    ]
    print("\n\n".join(reports))
    return 0


def _online(drive: str, viewer_id: str | None, *, days: int) -> int:
    if not _feed_log_exists():
        print("no feed log", file=sys.stderr)
        return 0

    viewers = _viewer_ids(drive, viewer_id)
    if not viewers:
        print("no viewers")
        return 0

    now = datetime.now(UTC).replace(tzinfo=None)
    reports = [
        _online_viewer(drive, viewer, days=days, now=now)
        for viewer in viewers
    ]
    print("\n\n".join(reports))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.pickup.evaluate",
        description="Measure Pickup's accuracy from the feed log or by replay.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    replay = subcommands.add_parser(
        "replay", help="Rebuild the feed at past cut points and score it.",
    )
    replay.add_argument("--drive", required=True)
    replay.add_argument(
        "--days", type=int, default=60,
        help="How many daily cut points to evaluate.",
    )
    replay.add_argument(
        "--horizon", type=int, default=7,
        help="Days after a cut point that count as its target window.",
    )
    replay.add_argument("--viewer")

    online = subcommands.add_parser(
        "online", help="Score stored feeds against the plays that followed.",
    )
    online.add_argument("--drive", required=True)
    online.add_argument(
        "--days", type=int, default=30,
        help="How far back to collect targets.",
    )
    online.add_argument("--viewer")

    interests_parser = subcommands.add_parser(
        "interests",
        help="List the interests Pickup learned, and how they were treated.",
    )
    interests_parser.add_argument("--drive", required=True)
    interests_parser.add_argument(
        "--titles", type=int, default=5, choices=range(0, 21),
        help="Members to list per lane, from 0 to 20.",
    )
    interests_parser.add_argument(
        "--picks", type=int, default=3, choices=range(0, 21),
        help="Candidates to list per lane, from 0 to 20.",
    )
    interests_parser.add_argument("--viewer")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        init_readonly_databases()
        if args.command == "replay":
            return _replay(
                args.drive, args.viewer,
                days=args.days, horizon=args.horizon,
            )
        if args.command == "online":
            return _online(args.drive, args.viewer, days=args.days)
        return _interests(
            args.drive, args.viewer, titles=args.titles, picks=args.picks,
        )
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
