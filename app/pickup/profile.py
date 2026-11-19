"""The taste profile the Pickup feed retrieves against.

The lane this replaced asked "what resembles the last five files I
watched". That shape cannot be fixed by asking for more seeds: a
forty-episode binge still supplies most of any recent window, so
whatever follows it collapses onto one subject. Recent entries are
*part of* a history of several hundred, and it is the whole history that
has to be analysed.

So the history is clustered into a handful of interests, and recency
becomes a weight on an interest rather than a filter on which interests
exist. A binge can then only make one cluster denser, and the
interleave bounds what one cluster is worth.

Pickup recommends videos, so the profile reads videos only. Each one
enters with an engagement between 0 and 1: how much of it was watched,
lifted to 1 by a like or a favourite, and 0 for a glance or a dismissal.
That engagement weights every mean of member vectors and the lane's raw
weight, so one video watched to the end shapes an interest more than five
opened for a few seconds.

Two reads of the history, not one. The exclusion set — every non-video
file the viewer has opened, and every video they watched — is never
capped, because a cap on it would start recommending watched files to
anyone past the cap. The profile's vector load is capped, because
clustering does not get better past a couple of thousand points and the
cost is real.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import text as sql_text

from app.database import get_litloft_db, get_search_db_read
from app.models import Embedding
from app.pickup.retrieval import vector_table_for

logger = logging.getLogger(__name__)

#: Bumped whenever anything that changes the ranking changes: these
#: constants, the clustering, the weighting, the channels.
#:
#: The worker's checkpoint asks "has this viewer's history moved", and
#: the honest answer after a change like this is that it has not — the
#: *ranking of it* has. Without a version in that hash, a deployment
#: lands and every viewer whose history is quiet keeps yesterday's
#: ranking until the date rolls, up to a day later. Cheap to get right,
#: and invisible when it is wrong.
PROFILE_VERSION = 5

#: Watch history played before this instant is judged by the rule that
#: predates recorded coverage: opened at all meant seen, and the position
#: was the engagement. A margin after the core's playback recording
#: landed, so no row that predates it is read as a glance.
RECORDING_START = datetime(2026, 11, 5, 0, 0, 0)

#: The core's playback segment width (``PLAYED_SEGMENT_SECONDS``). A
#: segment records that playback happened inside those five seconds, not
#: how much of them, so it stands for the whole bucket.
PLAYED_SEGMENT_SECONDS = 5.0

#: Below this length a video records too few segments to say anything:
#: played stretches under ten seconds are never written, so a short
#: video's coverage reads as zero however it was watched.
SHORT_VIDEO_SECONDS = 20.0

#: The core's mime type for a ``.loft`` embed. Some embed providers
#: cannot report position or duration, and an embed that reports
#: neither was not necessarily abandoned.
LOFT_MIME = "application/vnd.litloft.loft+json"

#: On history recorded before :data:`RECORDING_START`, watched fractions
#: below this are a brief open, not interest.
F_MIN = 0.10

#: On the same history, reaching this fraction of a video counts as fully
#: watched, so the difference between a third and eighty percent does not
#: decide the profile.
F_FULL = 0.5

#: On the same history, what an embed that never reported playback
#: counts. Absence of a position from a provider that cannot send one is
#: unknown, not zero.
E_UNKNOWN = 0.5

#: Embedding types the profile is built on. ``metadata`` is absent for
#: the same reason ``find_similar`` refuses it as a second opinion: it
#: embeds the filename, which for "IMG_1234.jpg" or a UUID names
#: nothing, and a feed has no user in the loop to catch it.
CHANNELS = ("clip_thumbnail", "tfidf_keywords", "text_content")

#: How far back the profile looks.
#:
#: An earlier draft claimed the profile read the whole history with no
#: recency cut-off, and that claim was false the moment a cap was put on
#: the read: ``ORDER BY last_played_at DESC LIMIT`` is a recency filter
#: wearing a sanity bound's clothes, and past the cap an old interest
#: does not go quiet, it disappears. The claim is withdrawn rather than
#: the bound removed — a year is enough to describe what someone is
#: interested in, and saying so is honest where the previous wording was
#: not.
#:
#: The exclusion set is a separate read and has neither bound.
PROFILE_WINDOW_DAYS = 365

#: Belt to the window's braces: a viewer who opened thousands of files
#: inside the window still clusters in bounded time.
PROFILE_VECTOR_CAP = 2000

#: How fast an interest goes quiet.
#:
#: Set from a requirement rather than a preference: a fortnight without
#: touching an interest should visibly cost it, which means at least
#: three quarters of its mass gone by day 14.
#:
#: Measured against the production index, on a history that ran 180 days
#: at 80% one series and then stopped it entirely — feed share of that
#: series, by half-life:
#:
#:     half-life     +0d    +14d    +30d    +60d    +90d
#:            7d     77%     37%     38%     16%     15%
#:           14d     77%     46%     48%     20%     17%
#:           60d     77%     52%     60%     39%     33%
#:
#: Three things that might have argued against a short half-life were
#: measured and did not:
#:
#: - Over-reaction. Three days of watching nothing else moves the share
#:   by +5 points at 7 days against +1 at 60. Not a risk at this scale.
#: - Steady state. An interest held at 20% of viewing settles at 13% of
#:   the feed for *every* half-life; the ratio is stationary, so "still
#:   watching it, still see it" holds regardless.
#: - Losing old interests. An interest last touched 200-300 days ago
#:   keeps 67% of the feed at a 7-day half-life against 76% at 60. Its
#:   raw weight collapses to 0.011 of the loudest, and ``W_MIN`` catches
#:   it. Old interests are kept visible by the floor, not by the decay.
#:
#: The cost that is real: more lanes pin to the floor (4 of 7 at 7 days
#: against 2 at 14), so interests of quite different ages come out
#: weighted the same. Resolution among old interests is traded for
#: responsiveness to recent ones.
#:
#: Temper the expectation either way — this is a weak lever. What
#: dominates the share is how many lanes an interest occupies, which
#: clustering decides without consulting these weights at all.
HALF_LIFE_DAYS = 7.0

#: The quietest lane's share of the turns the loudest one gets.
#:
#: Decay alone cannot express "quieter but not gone". Worked against the
#: interleave's ``key = j / weight``, a cluster of fifty files watched
#: three years ago earns a raw weight of 1.6e-4 against 1.79 for five
#: files watched today, which places its first item around position 6250
#: of a feed that holds a few hundred. That is deletion wearing the
#: costume of decay, and it contradicts the point of clustering the full
#: history. Normalising onto ``[W_MIN, 1.0]`` makes the floor a
#: constraint rather than something the arithmetic might happen to
#: honour.
#:
#: The cost is the other side of the same dial: the loudest lane's share
#: of the feed is bounded by ``1 / (1 + (L - 1) * W_MIN)`` for ``L``
#: lanes — 44% at six, 21% at sixteen. Lowering W_MIN contains a binge
#: harder and buries old interests faster.
W_MIN = 0.25

#: There is deliberately no pass that folds near-identical lanes back
#: together, and the absence is measured rather than assumed.
#:
#: K comes from how many files the history holds, not how many subjects
#: are in it, so k-means splits a binge — a dense blob — into several
#: lanes, and those lanes then take several lanes' worth of turns. The
#: obvious repair is to merge clusters whose centroids are close. It
#: does not work, at any threshold. Against the production index, using
#: folder membership as ground truth:
#:
#:   same subject, sub-cluster centroids   p5 0.602  p25 0.668  p50 0.797
#:   different subjects, centroids                   p50 0.713  p90 0.859  max 0.942
#:
#: The two populations overlap almost entirely, and mean-centring does
#: not separate them. At 0.90 such a pass folds about a quarter of
#: same-subject splits while already merging distinct subjects at the
#: top of their range.
#:
#: Choosing K from the data instead — by silhouette — was measured too,
#: and keeps a binge whole in 3 of 12 production cases. The reason is
#: not a bad criterion: a 680-episode series has real internal structure
#: (eras, openings, formats), so the split is genuine. "Sub-structure of
#: one interest" and "two interests" are a semantic distinction, not a
#: geometric one, and no clustering parameter recovers it.
#:
#: So the profile does not pretend to. A binge is reported as several
#: lanes because that is what the history looks like, the weight floor
#: keeps quieter interests present, and narrowing a subject the viewer
#: is tired of is left to an explicit control, where the person supplies
#: the meaning the vectors do not carry.

#: Below this a history is too small to have distinguishable interests.
_MIN_FILES_FOR_CLUSTERING = 12
_K_MIN = 2
_K_MAX = 8

_KMEANS_ITERATIONS = 25



@dataclass(frozen=True)
class WatchedFile:
    """One entry of the viewer's history, with a naive-UTC timestamp.

    ``engagement`` is how much this entry shapes the profile, from 0 to
    1. See :func:`engagement`.
    """

    file_id: str
    last_played_at: datetime
    engagement: float


@dataclass(frozen=True)
class Lane:
    """One interest, and its share of the feed's turns."""

    channel: str
    cluster_id: str
    centroid: np.ndarray
    member_count: int
    raw_weight: float
    weight: float


# ---------------------------------------------------------------------------
# Reading the history
# ---------------------------------------------------------------------------


def _parse_naive_utc(value: object) -> datetime | None:
    """Read ``watch_history.last_played_at`` as a naive UTC datetime.

    Core writes ``datetime.now(UTC)`` into a column declared without a
    timezone, so the stored text is UTC wall-clock with no offset. Rows
    written through a path that bound an aware value carry a ``+00:00``
    suffix instead; those are converted rather than rejected.

    Returns None for anything unparseable. The caller drops the row: a
    fabricated timestamp would put a broken row at the top of the
    recency order, which is worse than losing it.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


#: A row dated further ahead than this is treated as unreadable rather
#: than as brand new. Clock skew between a client and the host is
#: ordinary; clamping it to zero would rank the skewed row above every
#: genuine one, which is the same fabrication ``_parse_naive_utc``
#: refuses when the text will not parse.
_FUTURE_TOLERANCE_DAYS = 1.0


def age_days(last_played_at: datetime, now: datetime | None = None) -> float:
    """Days since ``last_played_at``. Negative for a future timestamp."""
    reference = datetime.now(UTC).replace(tzinfo=None) if now is None else now
    if reference.tzinfo is not None:
        reference = reference.astimezone(UTC).replace(tzinfo=None)
    return (reference - last_played_at).total_seconds() / 86400.0


def _finite(value: object) -> float | None:
    """The value as a float, or None when it is absent or not finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_dismissed(dismissed_at: object, last_played_at: object) -> bool:
    """The core's reading of a dismissal: set, and not undone by a play.

    A play after the dismissal clears its effect, so a dismissal earlier
    than ``last_played_at`` is not one. An unparseable dismissal is not
    a dismissal either.
    """
    dismissed = _parse_naive_utc(dismissed_at)
    played = _parse_naive_utc(last_played_at)
    if dismissed is None or played is None:
        return False
    return dismissed >= played


@dataclass(frozen=True)
class WatchRow:
    """One watch-history row with everything watched-ness and engagement read.

    ``segments`` holds ``(segment_index, played_seconds)`` for this
    viewer and file; ``played_seconds`` is their sum, which the old
    engagement reads.
    """

    file_id: str
    played_at: datetime | None
    dismissed: bool
    position: float | None
    duration: float | None
    file_type: str
    mime_type: str | None
    live: bool
    liked: bool
    favourite: bool
    played_seconds: float
    segments: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class WatchState:
    """Whether one video counts as watched, and how much it seeds the profile."""

    watched: bool
    engagement: float
    coverage: float


def _known_duration(duration: float | None) -> float | None:
    """The reported duration when it can bound a video, else None."""
    if duration is None or duration <= 0.0:
        return None
    return duration


def _reported_position(position: float | None) -> float:
    """The reported position, with an absent, non-finite or negative one as 0."""
    return position if position is not None and position > 0.0 else 0.0


def playback_coverage(
    segments: Iterable[tuple[int, float]], duration: float | None,
) -> float:
    """Seconds of ``duration`` covered by the viewer's played segments.

    Each segment with any playback stands for its whole bucket, and a
    bucket that starts past the end of the video is ignored: the duration
    can shrink after recording, and the rows left beyond it describe a
    longer cut than the one the viewer has.
    """
    known = _known_duration(duration)
    if known is None:
        return 0.0
    total = 0.0
    for index, played in segments:
        if played <= 0.0:
            continue
        start = PLAYED_SEGMENT_SECONDS * index
        if start >= known:
            continue
        total += min(PLAYED_SEGMENT_SECONDS, known - start)
    return total


def _version_4_engagement(row: WatchRow) -> float:
    """How much the row shaped the profile at ``PROFILE_VERSION`` 4.

    Kept for the evaluation rows that must not move with this rule.
    """
    if row.liked or row.favourite:
        return 1.0
    if row.dismissed:
        return 0.0

    reported = _known_duration(row.duration)
    if reported is not None:
        position = _reported_position(row.position)
        played = row.played_seconds
        fraction = max(position, min(played, reported)) / reported
        fraction = min(1.0, max(0.0, fraction))
        if fraction < F_MIN:
            return 0.0
        return min(1.0, fraction / F_FULL)

    if row.mime_type == LOFT_MIME:
        return E_UNKNOWN
    return 0.0


def watch_state(row: WatchRow) -> WatchState:
    """Judge one video by what was played, not by where the page was left.

    A video is watched when it is liked or favourite, when its history
    predates recording, when it is short and its position is at least
    half, or when its played coverage is at least half its duration. Every
    other video with a row is a glance: it stays a candidate, and it is
    no seed.
    """
    duration = _known_duration(row.duration)
    coverage = playback_coverage(row.segments, row.duration)
    position = _reported_position(row.position)
    legacy = row.played_at is None or row.played_at < RECORDING_START

    if row.liked or row.favourite:
        return WatchState(True, 1.0, coverage)
    if legacy:
        watched, engagement = True, _version_4_engagement(row)
    elif duration is not None and (
        2.0 * position >= duration
        if duration < SHORT_VIDEO_SECONDS
        else 2.0 * coverage >= duration
    ):
        watched, engagement = True, 1.0
    else:
        watched, engagement = False, 0.0

    if row.dismissed:
        engagement = 0.0
    return WatchState(watched, engagement, coverage)


def viewer_ids(drive: str) -> list[str]:
    """Every viewer with history in this drive."""
    with get_litloft_db() as session:
        rows = session.execute(
            sql_text(
                "SELECT DISTINCT wh.viewer_id "
                "FROM watch_history wh "
                "JOIN files f ON wh.file_id = f.id "
                "WHERE f.drive = :drive"
            ),
            {"drive": drive},
        ).fetchall()
    return [row[0] for row in rows if row[0]]


#: Every watched row the profile depends on, one row per played segment.
_WATCH_ROWS_SQL = """
SELECT wh.file_id,
       wh.last_played_at,
       wh.dismissed_at,
       wh.playback_position,
       wh.duration,
       f.file_type,
       f.mime_type,
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


def watch_rows(drive: str, viewer_id: str) -> list[WatchRow]:
    """This viewer's history in the drive, one entry per file, live or not."""
    with get_litloft_db() as session:
        rows = session.execute(
            sql_text(_WATCH_ROWS_SQL),
            {"drive": drive, "viewer": viewer_id},
        ).fetchall()

    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)

    out: list[WatchRow] = []
    for file_id, group in grouped.items():
        head = group[0]
        out.append(WatchRow(
            file_id=file_id,
            played_at=_parse_naive_utc(head[1]),
            dismissed=_is_dismissed(head[2], head[1]),
            position=_finite(head[3]),
            duration=_finite(head[4]),
            file_type=head[5],
            mime_type=head[6],
            live=bool(head[7]),
            liked=bool(head[8]),
            favourite=bool(head[9]),
            played_seconds=sum(_finite(row[11]) or 0.0 for row in group),
            segments=tuple(
                (int(row[10]), _finite(row[11]) or 0.0)
                for row in group
                if row[10] is not None
            ),
        ))
    out.sort(key=lambda row: row.file_id)
    return out


#: Positions in a signature row, for the two fields the exclusion set
#: reads: ``(file_id, played_at, live, liked, favourite, dismissed,
#: file_type, mime_type, played_seconds, duration, position, coverage,
#: watched)``.
_SIGNATURE_FILE_TYPE = 6
_SIGNATURE_WATCHED = 12


def watch_signature(drive: str, viewer_id: str) -> list[tuple]:
    """Everything about this viewer's history that the profile depends on.

    The **timestamps** are what make this a signature rather than a set.
    Reopening something already watched changes no ids at all, but moves
    that file's recency — which is what the lane weights and the window
    are computed from.

    The **lifecycle flag** is here because ``profile_history`` filters on
    it. Trashing a watched file, or restoring one, changes which entries
    the profile is built from while leaving ids and timestamps
    untouched, and the feed would go on describing the old set.

    The **liked, favourite and dismissal flags**, the **file type and
    mime type**, the **played seconds**, and the **duration, position and
    coverage** are here because engagement and watched-ness read them.
    Liking a video, favouriting it, recording a played range, reporting a
    longer duration or seeking changes how much it shapes the profile
    without moving ``last_played_at``.

    The **watched flag** is carried beside the rounded coverage and
    position: a flip between watched and glanced inside one decimal still
    changes the checkpoint.

    Rows are *not* filtered by any of these flags: the exclusion set is
    derived from this too, and a file in the trash is still a file this
    viewer has opened. Rows are sorted by file id so the signature is
    stable, and every number is written to one decimal so float noise
    does not move the checkpoint.
    """
    signature = []
    for row in watch_rows(drive, viewer_id):
        state = watch_state(row)
        signature.append((
            row.file_id,
            str(row.played_at) if row.played_at is not None else "",
            int(row.live),
            int(row.liked),
            int(row.favourite),
            int(row.dismissed),
            row.file_type,
            row.mime_type or "",
            f"{row.played_seconds:.1f}",
            f"{row.duration if row.duration is not None else 0.0:.1f}",
            f"{_reported_position(row.position):.1f}",
            f"{state.coverage:.1f}",
            int(state.watched),
        ))
    return signature


def signature_exclusions(signature: Sequence[tuple]) -> set[str]:
    """The exclusion set a ``watch_signature`` describes.

    Rule 1: every non-video file the viewer has a row for, and every
    watched video. Read from the signature so the sweep needs no second
    pass over the history.
    """
    return {
        row[0]
        for row in signature
        if row[_SIGNATURE_FILE_TYPE] != "video" or row[_SIGNATURE_WATCHED]
    }


def live_videos(drive: str) -> dict[str, datetime | None]:
    """The drive's live videos, each with its parsed ``created_at``.

    The values are naive UTC, or ``None`` when the stored text will not
    parse. Videos are the only files Pickup recommends, so this doubles
    as the candidate restriction and as the one read freshness is built
    from: age is a property of the file, not of the viewer.
    """
    with get_litloft_db() as session:
        rows = session.execute(
            sql_text(
                "SELECT f.id, f.created_at FROM files f "
                "WHERE f.drive = :drive AND f.file_type = 'video' "
                "  AND f.deleted_at IS NULL AND f.missing_since IS NULL"
            ),
            {"drive": drive},
        ).fetchall()
    return {row[0]: _parse_naive_utc(row[1]) for row in rows}


def watched_file_ids(drive: str, viewer_id: str) -> set[str]:
    """Every file this viewer's feed must not hold. Never capped.

    A non-video file the viewer has a row for, and a video they watched;
    a glanced video stays a candidate, like one never opened. Trashed and
    missing files stay in: the question is what the viewer has already
    seen, not what is currently on disk. They cannot be candidates
    anyway, so including them costs nothing, and dropping them would risk
    recommending a file whose row came back.
    """
    return signature_exclusions(watch_signature(drive, viewer_id))


def profile_history(
    drive: str,
    viewer_id: str,
    cap: int = PROFILE_VECTOR_CAP,
    window_days: float = PROFILE_WINDOW_DAYS,
    *,
    now: datetime | None = None,
    until: datetime | None = None,
    videos_only: bool = True,
    weighted: bool = True,
    version_4: bool = False,
) -> list[WatchedFile]:
    """This viewer's kept entries inside the window, newest first.

    Every filter runs before the cap: a document, an unparseable
    timestamp or a video with no engagement must not push a kept row out
    of the profile.

    Args:
        drive: The drive to read, and the only one.
        viewer_id: Whose history to read.
        cap: Most entries to return. Applied after every filter.
        window_days: How far back from ``now`` entries may be.
        now: Reference time for ages, the window and "newest". None
            means the current time.
        until: Exclusive upper bound on ``last_played_at``, for replay.
        videos_only: Keep live videos only, or live files of every kind.
        weighted: Give video rows their engagement and drop those with
            none. Without it every row counts fully.
        version_4: Weight every video by ``PROFILE_VERSION`` 4's
            engagement, for the evaluation rows that must not move.

    Returns:
        ``WatchedFile`` rows, newest first.
    """
    if now is None:
        reference = datetime.now(UTC).replace(tzinfo=None)
    elif now.tzinfo is not None:
        reference = now.astimezone(UTC).replace(tzinfo=None)
    else:
        reference = now
    upper = _parse_naive_utc(until) if until is not None else None

    history: list[WatchedFile] = []
    unweighted = 0
    for row in watch_rows(drive, viewer_id):
        if not row.live:
            continue
        if videos_only and row.file_type != "video":
            continue
        parsed = row.played_at
        if parsed is None:
            logger.debug(
                "Pickup: unreadable last_played_at for file=%s viewer=%s",
                row.file_id, viewer_id,
            )
            continue
        age = (reference - parsed).total_seconds() / 86400.0
        if age < -_FUTURE_TOLERANCE_DAYS:
            logger.debug(
                "Pickup: last_played_at is in the future for file=%s", row.file_id,
            )
            continue
        if age > window_days:
            continue
        if upper is not None and parsed >= upper:
            continue

        if weighted and row.file_type == "video":
            value = (
                _version_4_engagement(row)
                if version_4
                else watch_state(row).engagement
            )
            if value <= 0.0:
                unweighted += 1
                continue
        else:
            value = 1.0
        history.append(
            WatchedFile(
                file_id=row.file_id, last_played_at=parsed, engagement=value,
            )
        )

    history.sort(key=lambda entry: entry.last_played_at, reverse=True)
    kept = history[:cap]
    logger.debug(
        "Pickup: kept %d rows (dropped %d with no engagement) viewer=%s",
        len(kept), unweighted, viewer_id,
    )
    return kept


# ---------------------------------------------------------------------------
# Representative vectors
# ---------------------------------------------------------------------------


def _load_vectors(
    embedding_ids: Sequence[str],
    table: str,
) -> dict[str, np.ndarray]:
    """Fetch raw vectors by embedding id from one vector table.

    One statement per id. sqlite-vec does not decompose ``IN`` into
    point lookups — each such statement scans the virtual table — so
    batching multiplies whole scans instead of amortising them. See the
    measurement in ``app.pickup.retrieval``.
    """
    if not embedding_ids:
        return {}

    from app.database import get_search_engine

    out: dict[str, np.ndarray] = {}
    engine = get_search_engine()
    statement = sql_text(f"SELECT vector FROM {table} WHERE embedding_id = :eid")
    with engine.connect() as conn:
        for embedding_id in embedding_ids:
            row = conn.execute(statement, {"eid": embedding_id}).fetchone()
            if row and row[0]:
                out[embedding_id] = np.frombuffer(row[0], dtype=np.float32)
    return out


def representative_vectors(
    file_ids: Iterable[str],
    channel: str,
) -> dict[str, np.ndarray]:
    """One normalized vector per file for ``channel``.

    ``clip_thumbnail`` and ``tfidf_keywords`` store a single row per
    file. ``text_content`` stores one per chunk, and their mean is what
    "this document, generally" means here. Files with no embedding of
    this channel are simply absent from the result.
    """
    ids = list(file_ids)
    if not ids:
        return {}

    table = vector_table_for(channel)
    with get_search_db_read() as session:
        rows = (
            session.query(Embedding.id, Embedding.file_id)
            .filter(
                Embedding.file_id.in_(ids),
                Embedding.embedding_type == channel,
            )
            .all()
        )
    if not rows:
        return {}

    by_file: dict[str, list[str]] = {}
    for embedding_id, file_id in rows:
        by_file.setdefault(file_id, []).append(embedding_id)

    # Reduced per file rather than loaded all at once. The cap is on
    # *files*, and ``text_content`` averages 53 chunks each, so holding
    # every chunk would keep six figures of vectors resident — hundreds
    # of megabytes at 384 or 768 dimensions — beside the drive-wide
    # candidate matrices the worker is already carrying. Only one file's
    # chunks are live at a time here.
    out: dict[str, np.ndarray] = {}
    for file_id, embedding_ids in by_file.items():
        vectors = _load_vectors(embedding_ids, table)
        if not vectors:
            continue
        total = None
        count = 0
        for embedding_id in embedding_ids:
            vector = vectors.get(embedding_id)
            if vector is None:
                continue
            total = vector.astype(np.float32) if total is None else total + vector
            count += 1
        if total is None or count == 0:
            continue
        mean = total / count
        norm = float(np.linalg.norm(mean))
        if norm == 0.0:
            continue
        out[file_id] = (mean / norm).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def choose_k(n: int) -> int:
    """How many interests to look for in ``n`` files.

    The lower bound is 2 rather than 3 so K steps 1 -> 2 at the
    threshold; three clusters over twelve files is noise, and the jump
    would change the feed's shape on the twelfth watched file.
    """
    if n < _MIN_FILES_FOR_CLUSTERING:
        return 1
    return min(_K_MAX, max(_K_MIN, round(math.sqrt(n / 2))))


def _seed_from(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _weighted_mean(
    matrix: np.ndarray, weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """The mean of ``matrix``'s rows, scaled by ``weights`` when given."""
    if weights is None:
        return matrix.mean(axis=0)
    scaled = np.asarray(weights, dtype=np.float64)
    return (matrix.astype(np.float64) * scaled[:, None]).sum(axis=0)


def _kmeans(
    matrix: np.ndarray,
    k: int,
    seed: int,
    weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Spherical k-means. Returns a label per row.

    The vectors are unit length, so cosine similarity is a dot product
    and a cluster's centre is the normalized weighted mean of its
    members. Seeding ignores the weights: it only picks starting
    directions, and a k-means++ spread that already covers the data is
    as good a start for weighted updates. Seeded from the caller's key
    so a run is reproducible — an unreproducible profile would reshuffle
    the feed on every sweep for no reason the viewer could see.
    """
    n = matrix.shape[0]
    if k <= 1:
        return np.zeros(n, dtype=int)
    if n <= k:
        return np.arange(n)

    rng = np.random.default_rng(seed)

    # k-means++ seeding: spread the initial centres out, so a dense
    # binge cannot capture several of them.
    centres = [matrix[rng.integers(n)]]
    for _ in range(1, k):
        similarity = np.max(matrix @ np.stack(centres).T, axis=1)
        distance = np.clip(1.0 - similarity, 0.0, None) ** 2
        total = float(distance.sum())
        if total <= 0.0:
            centres.append(matrix[rng.integers(n)])
            continue
        centres.append(matrix[rng.choice(n, p=distance / total)])
    centroids = np.stack(centres)

    labels = np.full(n, -1, dtype=int)
    for _ in range(_KMEANS_ITERATIONS):
        new_labels = np.argmax(matrix @ centroids.T, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for index in range(k):
            mask = labels == index
            if not mask.any():
                continue
            mean = _weighted_mean(
                matrix[mask], None if weights is None else np.asarray(weights)[mask],
            )
            norm = float(np.linalg.norm(mean))
            if norm > 0:
                centroids[index] = mean / norm
    return labels


#: A cluster whose members cancel out has a mean with no direction, and
#: normalising it is not possible. Passed on as a query it would sit at
#: an equal distance from everything, scoring a flat 0.5 — above the
#: retrieval floor — and fill its lane with files related to nothing.
_MIN_CENTROID_NORM = 1e-6


def _centroid(
    matrix: np.ndarray, weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray | None:
    """The cluster's direction, or None if it has none."""
    mean = _weighted_mean(matrix, weights)
    norm = float(np.linalg.norm(mean))
    if norm < _MIN_CENTROID_NORM:
        return None
    return (mean / norm).astype(np.float32)


def _merge_singletons(
    groups: list[list[int]],
    matrix: np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> list[list[int]]:
    """Fold one-member clusters into their nearest neighbour.

    Every lane is guaranteed a share of the feed (see ``W_MIN``), so a
    cluster holding one stray file would buy that file's neighbourhood
    real airtime. Merging beats dropping: the file was watched, and its
    subject still belongs somewhere.

    A profile that is a single cluster keeps it however small — with one
    watched file, "more like that one" is the honest profile.
    """
    if len(groups) <= 1:
        return groups

    survivors = [g for g in groups if len(g) > 1]
    strays = [g for g in groups if len(g) == 1]
    if not strays:
        return groups
    if not survivors:
        # Everything is a singleton: one lane holding all of them.
        return [[index for group in groups for index in group]]

    for stray in strays:
        # Recomputed each round: absorbing a stray moves the centroid it
        # went into, and the next stray must be judged against where
        # that lane now sits.
        directions = [
            _centroid(
                matrix[g], None if weights is None else np.asarray(weights)[g],
            )
            for g in survivors
        ]
        usable = [i for i, d in enumerate(directions) if d is not None]
        if not usable:
            return [[index for group in groups for index in group]]
        scores = [float(directions[i] @ matrix[stray[0]]) for i in usable]
        survivors[usable[int(np.argmax(scores))]] += stray
    return survivors


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def decay(days: float) -> float:
    """How much a file watched ``days`` ago still counts."""
    return 0.5 ** (days / HALF_LIFE_DAYS)


def raw_weight(
    ages: Sequence[float], weights: Sequence[float] | None = None,
) -> float:
    """Decayed mass of a cluster, log-compressed.

    The logarithm is what stops a binge dominating: forty episodes must
    not outweigh a five-file interest by a factor of eight.

    ``weights`` is one engagement per age, and defaults to 1 for every
    entry: a lane's mass is what the viewer actually watched, not how
    many rows it holds.
    """
    if weights is None:
        weights = [1.0] * len(ages)
    return math.log1p(
        sum(weight * decay(age) for age, weight in zip(ages, weights, strict=True))
    )


def _normalise(raws: list[float]) -> list[float]:
    """Scale raw weights against the loudest lane, with a floor.

    Across every lane of every channel together: the interleave treats
    them as one sequence, so a lane's share must not depend on which
    other lanes happened to share its embedding type.

    Deliberately not min-max. Mapping the minimum onto ``W_MIN`` and the
    maximum onto 1.0 does not only impose a floor, it *stretches the
    range*: three interests watched in the same week, whose decayed mass
    differs by one percent, come out as 1.0 / 0.625 / 0.25 and take
    turns in a 4 : 2.5 : 1 ratio. There is no way to say "these are
    equally interesting" under it, and the special case for
    ``max == min`` is the seam where that shows.

    Dividing by the maximum leaves near-ties as near-ties and still
    guarantees the floor.
    """
    if not raws:
        return []
    high = max(raws)
    if high <= 0.0:
        return [1.0] * len(raws)
    return [max(W_MIN, raw / high) for raw in raws]


# ---------------------------------------------------------------------------
# Assembling the profile
# ---------------------------------------------------------------------------


def build_lanes(
    history: Sequence[WatchedFile],
    *,
    key: str,
    now: datetime | None = None,
    vectors: Callable[[Iterable[str], str], dict[str, np.ndarray]] = representative_vectors,
) -> list[Lane]:
    """Cluster ``history`` into weighted lanes across every channel.

    Every mean of member vectors is weighted by engagement and then
    normalized, so the lane points where the watched time went rather
    than where the count is.

    Args:
        history: The viewer's kept entries, newest first.
        key: Stable identity for this (drive, viewer), used to seed
            clustering so repeated runs agree.
        now: Reference time for the ages in the lane weights. None means
            the current time.
        vectors: ``(file_ids, channel) -> {file_id: unit vector}``. The
            worker reads the index through the default; replay passes
            vectors it loaded once.

    Returns:
        Lanes with normalized weights, heaviest first.
    """
    if not history:
        return []

    ages = {w.file_id: age_days(w.last_played_at, now=now) for w in history}
    engagements = {w.file_id: w.engagement for w in history}
    file_ids = [w.file_id for w in history]

    pending: list[tuple[str, str, np.ndarray, int, float]] = []
    for channel in CHANNELS:
        loaded = vectors(file_ids, channel)
        if not loaded:
            continue
        # Fixed order, so the seeded clustering below is reproducible.
        members = [f for f in file_ids if f in loaded]
        matrix = np.stack([loaded[f] for f in members])
        weights = np.asarray([engagements[f] for f in members], dtype=np.float64)

        labels = _kmeans(
            matrix, choose_k(len(members)), _seed_from(key, channel), weights,
        )
        groups = [
            [i for i in range(len(members)) if labels[i] == index]
            for index in range(int(labels.max()) + 1)
        ]
        groups = _merge_singletons([g for g in groups if g], matrix, weights)

        for index, group in enumerate(groups):
            centroid = _centroid(matrix[group], weights[group])
            if centroid is None:
                logger.debug(
                    "Pickup: dropping directionless cluster %s:%d",
                    channel, index,
                )
                continue
            pending.append((
                channel,
                f"{channel}:{index}",
                centroid,
                len(group),
                raw_weight(
                    [ages[members[i]] for i in group],
                    [weights[i] for i in group],
                ),
            ))

    if not pending:
        return []

    weights = _normalise([raw for *_, raw in pending])
    lanes = [
        Lane(
            channel=channel,
            cluster_id=cluster_id,
            centroid=centroid,
            member_count=count,
            raw_weight=raw,
            weight=weight,
        )
        for (channel, cluster_id, centroid, count, raw), weight
        in zip(pending, weights, strict=True)
    ]
    return sorted(lanes, key=lambda lane: (-lane.weight, lane.cluster_id))
