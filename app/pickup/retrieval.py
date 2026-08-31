"""Candidate generation for the Pickup feed.

Deliberately not built on ``app.search.find_similar``. That function
serves the file-detail "similar files" section, where saying nothing is
better than saying something thin, and it carries five filters that make
it honest about weak evidence: a 0.70 floor, a gap check and a
coefficient-of-variation check that each discard *every* candidate, a
margin cutoff keeping only what scores within 0.05 of the best, and an
exclusion of near-identical matches. Four of the five are wrong for a
feed, which exists to produce quantity — a cluster's neighbourhood is
flat by definition, so both non-discrimination guards read normal input
as a fault, and the margin cutoff caps a lane's contribution at whatever
lands in one narrow band.

It is also not built on a k-nearest-neighbour query at all.

sqlite-vec caps ``k`` at 4096 rows, and that is a compile-time constant
in the extension, not a number we chose:

    k=4096  OK
    k=4097  OperationalError: k value in knn query too large,
            provided 4097 and the limit is 4096

Measured against the real index, ``vec_text`` holds 56,422 rows of which
65.7% are ``whisper`` — discarded by the type filter — and 33.2% are
``text_content``, at 53.3 chunks per file. A 4096-row window is 7.3% of
the table and resolves to roughly twenty-five documents before the drive
filter and before removing what the viewer has already read. The
exclusion set here *is* the viewer's history, and the query vector is
built out of that same history, so the nearest rows are overwhelmingly
rows we must drop. A KNN cannot be widened far enough to survive that.

So the shape is inverted. Scope and channel are settled in one indexed
join, the vectors for exactly those rows are read, and a single matmul
scores every lane at once. There is no ceiling, no widening loop, and no
page at which the search gives up.

Two consequences worth naming:

- The drive boundary becomes structural. Rows outside the drive never
  enter the matrix, so there is no ordering in which a quality filter
  could run ahead of the scope restriction — the defect this replaces,
  where ``find_similar`` scored and pruned a cross-drive pool and only
  then cut the survivors down to the requested drive.
- Scoring stays per chunk and reduces per file by ``max``. A document is
  a candidate because *some part of it* is about the lane's subject; its
  mean over 53 chunks is a blur of 53 topics that scores mediocre
  everywhere. The profile takes the mean, because there the question is
  what a document is about as a whole. The asymmetry is deliberate.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass

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
#: lane is about, and the tail turns into arbitrary rows. Far below
#: ``find_similar``'s 0.70, which answers a different question.
_FEED_MIN_SCORE = 0.45

#: A score this high means the vectors are effectively identical, which
#: in practice means a duplicate or a trivial embedding (Whisper
#: emitting one word over a music track) rather than a real match.
#:
#: Applied to a file's reduced best score, not to individual chunks.
#: Per chunk it would only skip one row of a duplicated document and let
#: the file through on its next-best chunk.
_NEAR_IDENTICAL_SCORE = 0.999

#: Vectors are read one statement per id, and that is the fast path.
#:
#: sqlite-vec does not decompose ``IN`` into point lookups: every such
#: statement scans the virtual table, so batching multiplies whole scans
#: rather than amortising them. Measured against the real index
#: (``vec_text``, 56,422 rows):
#:
#:     n_ids     IN/500    IN/2000   point lookup   full scan
#:      2,000     0.26s      0.08s          0.03s       0.71s
#:      5,000     0.67s      0.23s          0.07s       0.71s
#:     12,000     1.57s      0.47s          0.16s       0.71s
#:
#: An earlier revision batched at 500 and cited a measurement of a
#: *single* 2,000-id statement as justification — which is not what the
#: code did. Point lookups cost about 13 microseconds each here and,
#: unlike the scans, do not grow with the size of the table; production
#: holds 463,350 rows in this one.


def vector_table_for(channel: str) -> str:
    """Return the vector table holding ``channel``'s embeddings."""
    try:
        return validate_vector_table(_CHANNEL_TABLES[channel])
    except KeyError:
        raise ValueError(f"Not a pickup channel: {channel}") from None


@dataclass(frozen=True)
class CandidateSet:
    """Every scorable row of one drive and channel.

    One row per embedding, so a document appears once per chunk. The
    ``file_ids`` tuple is parallel to ``matrix``'s rows.
    """

    channel: str
    drive: str
    file_ids: tuple[str, ...]
    matrix: np.ndarray

    def __len__(self) -> int:
        return len(self.file_ids)


def _row_identity(drive: str, channel: str) -> list[tuple[str, str]]:
    """(embedding_id, file_id) for one drive and channel.

    Scope and channel are settled here, in one indexed join, before any
    vector is read or any score computed. Nothing outside the drive
    reaches the matrix, so no later filter can be ordered ahead of the
    boundary.
    """
    with get_search_db_read() as session:
        rows = (
            session.query(Embedding.id, Embedding.file_id)
            .join(IndexedFile, IndexedFile.file_id == Embedding.file_id)
            .filter(
                Embedding.embedding_type == channel,
                IndexedFile.drive == drive,
                IndexedFile.active.is_(True),
            )
            .all()
        )
    return [(embedding_id, file_id) for embedding_id, file_id in rows]


def _read_vectors(embedding_ids: Sequence[str], table: str) -> dict[str, bytes]:
    """Fetch raw vector blobs by embedding id, one lookup each.

    See the module note above for why this is not batched.
    """
    if not embedding_ids:
        return {}

    out: dict[str, bytes] = {}
    engine = get_search_engine()
    statement = sql_text(
        f"SELECT vector FROM {table} WHERE embedding_id = :eid"
    )
    with engine.connect() as conn:
        for embedding_id in embedding_ids:
            row = conn.execute(statement, {"eid": embedding_id}).fetchone()
            if row and row[0]:
                out[embedding_id] = row[0]
    return out


def load_candidates(*, drive: str, channel: str) -> CandidateSet:
    """Load every scorable row for one drive and channel.

    Keyword-only with no defaults, so a missed call site is a
    ``TypeError`` rather than a silent read of the whole library.

    The result is independent of the viewer, so one sweep can build it
    once per (drive, channel) and score every viewer's lanes against it.
    """
    table = vector_table_for(channel)
    identity = _row_identity(drive, channel)
    if not identity:
        return CandidateSet(
            channel=channel, drive=drive, file_ids=(),
            matrix=np.zeros((0, 0), dtype=np.float32),
        )

    blobs = _read_vectors([embedding_id for embedding_id, _ in identity], table)

    file_ids: list[str] = []
    vectors: list[np.ndarray] = []
    dimension: int | None = None
    for embedding_id, file_id in identity:
        blob = blobs.get(embedding_id)
        if blob is None:
            continue
        vector = np.frombuffer(blob, dtype=np.float32)
        # sqlite-vec validated width on the way in; reading the blobs
        # directly means we validate it on the way out, or np.stack
        # would fail later with nothing pointing at the bad row.
        if dimension is None:
            dimension = vector.shape[0]
        elif vector.shape[0] != dimension:
            raise ValueError(
                f"{table} row {embedding_id} has width {vector.shape[0]}, "
                f"expected {dimension}"
            )
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            continue
        file_ids.append(file_id)
        vectors.append(vector / norm)

    if not vectors:
        return CandidateSet(
            channel=channel, drive=drive, file_ids=(),
            matrix=np.zeros((0, 0), dtype=np.float32),
        )

    # ``np.stack`` of float32 rows is already float32, and an
    # ``astype`` without ``copy=False`` would duplicate the whole matrix
    # while the blobs and per-row arrays are still live — about 146 MiB
    # extra for 100,000 rows at 384 dimensions, on the sweep that is
    # also holding the profile.
    return CandidateSet(
        channel=channel,
        drive=drive,
        file_ids=tuple(file_ids),
        matrix=np.stack(vectors).astype(np.float32, copy=False),
    )


def score_lanes(
    candidates: CandidateSet,
    centroids: Sequence[np.ndarray],
    *,
    channel: str,
    exclude_file_ids: Collection[str],
    limit: int,
) -> list[list[tuple[str, float]]]:
    """Score every lane against ``candidates`` in one pass.

    Args:
        candidates: Rows for one drive and channel.
        centroids: One normalized query vector per lane.
        channel: The channel those centroids were built from. Checked
            against the candidate set, because comparing widths cannot
            do it: ``tfidf_keywords`` and ``text_content`` share
            ``vec_text``, which declares one width for the table, so
            those two are *always* the same width and swapping them
            would score silently and meaninglessly.
        exclude_file_ids: Files the viewer has opened. The whole
            history, never truncated — a cap here would start
            recommending watched files to anyone past it.
        limit: Maximum candidates to return per lane.

    Returns:
        Per lane, ``(file_id, score)`` best first.
    """
    if candidates.channel != channel:
        raise ValueError(
            f"{channel} lanes scored against {candidates.channel} rows"
        )
    if not centroids or limit <= 0 or len(candidates) == 0:
        return [[] for _ in centroids]

    query = np.stack([np.asarray(c, dtype=np.float32) for c in centroids])
    if query.shape[1] != candidates.matrix.shape[1]:
        raise ValueError(
            f"lane centroids are {query.shape[1]}-wide but "
            f"{candidates.channel} rows are {candidates.matrix.shape[1]}-wide"
        )

    similarity = candidates.matrix @ query.T

    # Reduce chunks to files by ``max``: a document is a candidate
    # because some part of it is about the subject.
    unique_files: list[str] = []
    index_of: dict[str, int] = {}
    row_index = np.empty(len(candidates), dtype=np.intp)
    for position, file_id in enumerate(candidates.file_ids):
        index = index_of.get(file_id)
        if index is None:
            index = len(unique_files)
            index_of[file_id] = index
            unique_files.append(file_id)
        row_index[position] = index

    best = np.full((len(unique_files), query.shape[0]), -np.inf, dtype=np.float32)
    np.maximum.at(best, row_index, similarity)

    excluded = np.fromiter(
        (file_id in exclude_file_ids for file_id in unique_files),
        dtype=bool,
        count=len(unique_files),
    )
    keep = ~excluded

    results: list[list[tuple[str, float]]] = []
    for lane in range(query.shape[0]):
        scores = best[:, lane]
        usable = (
            keep
            & (scores >= _FEED_MIN_SCORE)
            & (scores < _NEAR_IDENTICAL_SCORE)
        )
        positions = np.flatnonzero(usable)
        if positions.size == 0:
            results.append([])
            continue
        order = positions[np.argsort(-scores[positions], kind="stable")][:limit]
        results.append([(unique_files[i], float(scores[i])) for i in order])
    return results
