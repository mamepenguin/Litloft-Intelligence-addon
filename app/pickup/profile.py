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

Two reads of the history, not one. The exclusion set — every file the
viewer has opened — is never capped, because a cap on it would start
recommending watched files to anyone past the cap. The profile's vector
load is capped, because clustering does not get better past a couple of
thousand points and the cost is real.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import text as sql_text

from app.database import get_litloft_db, get_search_db_read
from app.models import Embedding
from app.pickup.retrieval import vector_table_for

logger = logging.getLogger(__name__)

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

#: How fast an interest goes quiet. A guess, expected to need tuning.
HALF_LIFE_DAYS = 60.0

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
    """One entry of the viewer's history, with a naive-UTC timestamp."""

    file_id: str
    last_played_at: datetime


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


def age_days(last_played_at: datetime) -> float:
    """Days since ``last_played_at``. Negative for a future timestamp."""
    now = datetime.now(UTC).replace(tzinfo=None)
    return (now - last_played_at).total_seconds() / 86400.0


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


def watched_file_ids(drive: str, viewer_id: str) -> set[str]:
    """Every file this viewer has opened in this drive. Never capped.

    Trashed and missing files stay in: the question is what the viewer
    has already seen, not what is currently on disk. They cannot be
    candidates anyway, so including them costs nothing, and dropping
    them would risk recommending a file whose row came back.
    """
    with get_litloft_db() as session:
        rows = session.execute(
            sql_text(
                "SELECT wh.file_id "
                "FROM watch_history wh "
                "JOIN files f ON wh.file_id = f.id "
                "WHERE f.drive = :drive AND wh.viewer_id = :viewer"
            ),
            {"drive": drive, "viewer": viewer_id},
        ).fetchall()
    return {row[0] for row in rows}


def profile_history(
    drive: str,
    viewer_id: str,
    cap: int = PROFILE_VECTOR_CAP,
    window_days: float = PROFILE_WINDOW_DAYS,
) -> list[WatchedFile]:
    """This viewer's live entries inside the window, newest first.

    Both bounds are per viewer. The worker this replaces took the
    drive's 50 most recent rows and split them by viewer afterwards, so
    whoever had been active most recently consumed the whole budget.
    """
    with get_litloft_db() as session:
        rows = session.execute(
            sql_text(
                "SELECT wh.file_id, wh.last_played_at "
                "FROM watch_history wh "
                "JOIN files f ON wh.file_id = f.id "
                "WHERE f.drive = :drive AND wh.viewer_id = :viewer "
                "  AND f.deleted_at IS NULL AND f.missing_since IS NULL "
                "ORDER BY wh.last_played_at DESC "
                "LIMIT :cap"
            ),
            {"drive": drive, "viewer": viewer_id, "cap": cap},
        ).fetchall()

    history: list[WatchedFile] = []
    for file_id, played_at in rows:
        parsed = _parse_naive_utc(played_at)
        if parsed is None:
            logger.debug(
                "Pickup: unreadable last_played_at for file=%s viewer=%s",
                file_id, viewer_id,
            )
            continue
        age = age_days(parsed)
        if age < -_FUTURE_TOLERANCE_DAYS:
            logger.debug(
                "Pickup: last_played_at is in the future for file=%s", file_id,
            )
            continue
        if age > window_days:
            continue
        history.append(WatchedFile(file_id=file_id, last_played_at=parsed))
    return history


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

    raw = _load_vectors(
        [e for embedding_ids in by_file.values() for e in embedding_ids],
        table,
    )

    out: dict[str, np.ndarray] = {}
    for file_id, embedding_ids in by_file.items():
        vectors = [raw[e] for e in embedding_ids if e in raw]
        if not vectors:
            continue
        mean = np.mean(np.stack(vectors), axis=0)
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


def _kmeans(matrix: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Spherical k-means. Returns a label per row.

    The vectors are unit length, so cosine similarity is a dot product
    and a cluster's centre is the normalized mean of its members.
    Seeded from the caller's key so a run is reproducible — an
    unreproducible profile would reshuffle the feed on every sweep for
    no reason the viewer could see.
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
            members = matrix[labels == index]
            if members.size == 0:
                continue
            mean = members.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm > 0:
                centroids[index] = mean / norm
    return labels


#: A cluster whose members cancel out has a mean with no direction, and
#: normalising it is not possible. Passed on as a query it would sit at
#: an equal distance from everything, scoring a flat 0.5 — above the
#: retrieval floor — and fill its lane with files related to nothing.
_MIN_CENTROID_NORM = 1e-6


def _centroid(matrix: np.ndarray) -> np.ndarray | None:
    """The cluster's direction, or None if it has none."""
    mean = matrix.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < _MIN_CENTROID_NORM:
        return None
    return (mean / norm).astype(np.float32)


def _merge_singletons(
    groups: list[list[int]],
    matrix: np.ndarray,
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
        directions = [_centroid(matrix[g]) for g in survivors]
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


def raw_weight(ages: Sequence[float]) -> float:
    """Decayed mass of a cluster, log-compressed.

    The logarithm is what stops a binge dominating: forty episodes must
    not outweigh a five-file interest by a factor of eight.
    """
    return math.log1p(sum(decay(age) for age in ages))


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


def build_lanes(history: Sequence[WatchedFile], *, key: str) -> list[Lane]:
    """Cluster ``history`` into weighted lanes across every channel.

    Args:
        history: The viewer's entries, newest first.
        key: Stable identity for this (drive, viewer), used to seed
            clustering so repeated runs agree.

    Returns:
        Lanes with normalized weights, heaviest first.
    """
    if not history:
        return []

    ages = {w.file_id: age_days(w.last_played_at) for w in history}
    file_ids = [w.file_id for w in history]

    pending: list[tuple[str, str, np.ndarray, int, float]] = []
    for channel in CHANNELS:
        vectors = representative_vectors(file_ids, channel)
        if not vectors:
            continue
        # Fixed order, so the seeded clustering below is reproducible.
        members = [f for f in file_ids if f in vectors]
        matrix = np.stack([vectors[f] for f in members])

        labels = _kmeans(matrix, choose_k(len(members)), _seed_from(key, channel))
        groups = [
            [i for i in range(len(members)) if labels[i] == index]
            for index in range(int(labels.max()) + 1)
        ]
        groups = _merge_singletons([g for g in groups if g], matrix)

        for index, group in enumerate(groups):
            centroid = _centroid(matrix[group])
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
                raw_weight([ages[members[i]] for i in group]),
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
