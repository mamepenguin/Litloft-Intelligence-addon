"""Candidate generation for the Pickup feed.

Deliberately not built on ``app.search.find_similar``. That function
serves the file-detail "similar files" section, where saying nothing is
better than saying something thin, and it carries five filters that make
it honest about weak evidence: a hard 0.70 floor, a gap check and a
coefficient-of-variation check that each discard *every* candidate, and
a margin cutoff that keeps only what scores within 0.05 of the best
match.

The feed needs the opposite. It exists to produce quantity, and the
neighbourhood around a cluster centroid is legitimately flat — files
that resemble each other is what a cluster *is* — so the two
non-discrimination guards read normal input as a fault. The margin
cutoff is worse still: it caps a lane's contribution at however many
files happen to land in one narrow band, which is why the seed-based
Pickup could not fill twelve slots.

Only the near-identical exclusion is worth keeping, and one low floor
takes the place of the rest.

The other reason for a separate path is ordering. ``find_similar``
resolves ``IndexedFile.drive`` last, after all five filters have already
scored and pruned a cross-drive candidate pool, so a file whose nearest
neighbours live in another drive can come back empty while good in-drive
candidates sat further down. A drive is a security boundary: the
restriction is a scope, not a quality judgement, and the two must not be
interleaved.
"""

from __future__ import annotations

import logging
from collections.abc import Collection

import numpy as np
from sqlalchemy import text as sql_text

from app.database import (
    get_search_db_read,
    get_search_engine,
    validate_vector_table,
)
from app.models import Embedding, IndexedFile

logger = logging.getLogger(__name__)

#: Embedding types the feed profiles and retrieves on, and the vector
#: table each lives in. ``metadata`` is absent on purpose: it embeds the
#: filename, which for "IMG_1234.jpg" or a UUID names nothing, and a feed
#: has no user in the loop to notice.
_CHANNEL_TABLES = {
    "clip_thumbnail": "vec_clip",
    "tfidf_keywords": "vec_text",
    "text_content": "vec_text",
}

#: Below this cosine similarity a candidate is not about anything the
#: cluster is about, and the tail turns into arbitrary rows. Far below
#: ``find_similar``'s 0.70, which exists to answer a different question.
_FEED_MIN_SCORE = 0.45

#: A score this high means the vectors are effectively identical, which
#: in practice means a duplicate or a trivial embedding (Whisper
#: emitting one word over a music track) rather than a real match.
_NEAR_IDENTICAL_SCORE = 0.999

#: Both vector tables are global indexes: MATCH ranks across every drive
#: and every embedding type, and knows about neither. The filters are
#: applied to what comes back and the fetch widens until enough rows
#: survive them. Same shape as the neighbour fetch in ``app.search`` and
#: ``app.workers.tag_knn``.
_FETCH_FACTORS = (1, 4, 16)
#: Ceiling on a single KNN, so a drive that is a small minority of the
#: library cannot turn one lane into a table scan.
_FETCH_MAX = 4096


def vector_table_for(channel: str) -> str:
    """Return the vector table holding ``channel``'s embeddings."""
    try:
        return validate_vector_table(_CHANNEL_TABLES[channel])
    except KeyError:
        raise ValueError(f"Not a pickup channel: {channel}") from None


def _fetch_schedule(base_limit: int) -> list[int]:
    """Ascending KNN fetch sizes, always ending at the ceiling.

    A neighbour search seeded from one file discards only that file's
    own embeddings, so its first page is nearly all usable and the
    widening factors are generous. This search discards every file the
    viewer has ever opened — which, since the query vector is built out
    of exactly those files, is most of what sits nearest it. Paging
    through thousands of already-watched rows is the expected path here,
    not an edge case.

    So the schedule cannot be left to bottom out at ``base_limit * 16``:
    with a small base that is well under ``_FETCH_MAX``, and the loop
    would give up while the index still had rows to give. The ceiling is
    appended explicitly.
    """
    sizes: list[int] = []
    for factor in _FETCH_FACTORS:
        size = min(base_limit * factor, _FETCH_MAX)
        if size not in sizes:
            sizes.append(size)
    if sizes[-1] < _FETCH_MAX:
        sizes.append(_FETCH_MAX)
    return sizes


def _surviving_file_ids(
    embedding_ids: list[str],
    *,
    channel: str,
    drive: str,
) -> dict[str, str]:
    """Map embedding id -> file id for rows that clear scope and type.

    Resolved in a second statement rather than joined into the KNN:
    the vector tables store neither drive nor embedding type, and
    sqlite-vec's planner requires the LIMIT to apply directly to the
    virtual table — joining anything into that statement moves the LIMIT
    out of its reach and raises "A LIMIT or 'k = ?' constraint is
    required".
    """
    if not embedding_ids:
        return {}

    with get_search_db_read() as session:
        rows = (
            session.query(Embedding.id, Embedding.file_id)
            .join(IndexedFile, IndexedFile.file_id == Embedding.file_id)
            .filter(
                Embedding.id.in_(embedding_ids),
                Embedding.embedding_type == channel,
                IndexedFile.drive == drive,
                IndexedFile.active.is_(True),
            )
            .all()
        )
    return {embedding_id: file_id for embedding_id, file_id in rows}


def centroid_neighbours(
    centroid: np.ndarray,
    *,
    channel: str,
    drive: str,
    exclude_file_ids: Collection[str],
    limit: int,
) -> list[tuple[str, float]]:
    """Return up to ``limit`` (file_id, score) nearest ``centroid``.

    Every result belongs to ``drive``, is active, carries an embedding of
    ``channel``, and is absent from ``exclude_file_ids``.

    ``drive`` is keyword-only with no default so that a missed call site
    is a ``TypeError`` rather than a silent read of the whole library.

    Args:
        centroid: A normalized query vector — a cluster centre, not a
            file's own embedding.
        channel: The embedding type to profile on. See
            ``_CHANNEL_TABLES``.
        drive: Only this drive's files are candidates.
        exclude_file_ids: Files the viewer has already opened. This is
            the viewer's *whole* history and is never truncated; the
            2000-file cap on the profile's vector load must not be
            shared with it, or a viewer past the cap starts being
            recommended what they have already seen.
        limit: Maximum candidates to return.

    Returns:
        (file_id, cosine similarity), best first.
    """
    table = vector_table_for(channel)
    if limit <= 0:
        return []

    query = np.asarray(centroid, dtype=np.float32)
    excluded = frozenset(exclude_file_ids)

    # A file can hold several embeddings of one channel (``text_content``
    # is per chunk), the excluded set can be most of the drive, and the
    # other drives share the index — so ask for well over ``limit``.
    base_limit = max(limit * 10 + 50, 100)

    scores: dict[str, float] = {}
    engine = get_search_engine()

    for asked in _fetch_schedule(base_limit):
        with engine.connect() as conn:
            rows = conn.execute(
                sql_text(
                    f"SELECT embedding_id, distance FROM {table} "
                    "WHERE vector MATCH :vec "
                    "ORDER BY distance "
                    "LIMIT :limit"
                ),
                {"vec": query.tobytes(), "limit": asked},
            ).fetchall()

        if not rows:
            return []

        distances = {row[0]: float(row[1]) for row in rows}
        file_ids = _surviving_file_ids(
            list(distances), channel=channel, drive=drive,
        )

        scores = {}
        for embedding_id, file_id in file_ids.items():
            if file_id in excluded:
                continue
            # L2 distance on normalized vectors: ||a-b||² = 2 - 2·cos,
            # so cos = 1 - d²/2.
            distance = distances[embedding_id]
            similarity = 1.0 - (distance * distance) / 2.0
            if similarity >= _NEAR_IDENTICAL_SCORE:
                continue
            if similarity < _FEED_MIN_SCORE:
                continue
            # Several chunks of one file: keep its best.
            if similarity > scores.get(file_id, -1.0):
                scores[file_id] = similarity

        # Widen only while there is more to find. The count tested here
        # is of candidates that already cleared every filter, so a lane
        # whose in-drive neighbours all sit below the floor exhausts the
        # index rather than stopping at a page that looked full.
        if len(scores) >= limit or asked >= _FETCH_MAX or len(rows) < asked:
            break

    if not scores:
        logger.debug(
            "Pickup: no candidates for channel=%s drive=%s", channel, drive,
        )

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
