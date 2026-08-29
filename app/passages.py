"""Passage-level links between a file and the sources the viewer vouched for.

Answers "which passages of what I am reading connect to what I already
have?" — a reading aid, not a judgement. Each row pairs one chunk of the
file being read with one chunk of a **verified** file, both reproduced
verbatim. No LLM is called and nothing is summarised: the feature points
at places, it does not write words (hako ``DPcjrRgspKAXqHjHOkJ8L``).

Two stages, so the cost stays bounded:

1. Average this file's chunk vectors into a centroid and run **one** KNN
   over ``vec_text`` to pick candidate files. Drive scoping happens in
   that query; trust and access narrowing happen right after, through
   the same Internal API call Ask uses.
2. Load the surviving candidates' chunk vectors and compute every
   source×candidate similarity as a single matrix product. Vectors are
   L2-normalised at write time, so a dot product *is* cosine similarity.

The alternative — one semantic search per paragraph — is what this
replaces. It cost one round trip per paragraph, which forced a cap on
how many paragraphs were looked at, which meant only a document's
opening ever got examined.

Spec ``2026-08-29-related-passages.md``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy import text as sql_text

from app.credentials import CallerCredential
from app.database import get_search_engine
# Reused rather than reimplemented: this function's fail-closed handling
# (network error, non-200, and above all a response that does not confirm
# the trust filter) is exactly the part that must never drift between
# callers.
from app.rag.retriever import _filter_file_ids_via_internal_api

logger = logging.getLogger(__name__)

#: The chunk kinds that carry prose. ``vec_text`` also holds metadata,
#: tfidf_keywords and vision_description vectors; none of those is a
#: passage a reader can be pointed at.
_PASSAGE_TYPES = ("text_content", "whisper")

#: Eligible rows the KNN aims to return after drive, kind and
#: self-exclusion narrow what it fetched. ``MATCH`` is a **global** KNN
#: and every joined predicate is applied post-fetch (the same behaviour
#: ``app.citations`` documents), so this is headroom, not a guarantee.
_KNN_POOL = 400

#: Ceiling sqlite-vec puts on ``k``.
_KNN_K_MAX = 4096

#: Files carried into stage 2.
_CANDIDATE_FILES = 20

#: Files asked of the KNN, before trust and access drop rows. The filters
#: run after ranking, so without headroom a run of unverified neighbours
#: at the top empties the list while verified files sit just below the
#: cut — the reason ``retriever._search_pool_size`` exists.
_CANDIDATE_POOL = _CANDIDATE_FILES * 4

#: Chunks scored per file. A book runs to thousands of chunks and the
#: matrix is the product of both sides, so both are capped — by
#: sampling across the whole file, never by taking its opening (see
#: ``_sample``).
_MAX_SOURCE_CHUNKS = 400
_MAX_CANDIDATE_CHUNKS = 200

#: Cosine floor for a pair. Provisional: a floor set too low is worse
#: than an empty section, because a reader who is shown spurious links
#: learns to ignore the real ones. Tuned against real files before this
#: ships, and moved into ``search-config.yml`` there.
_MIN_SCORE = 0.80

#: Rows returned when the caller does not say.
_DEFAULT_LIMIT = 5


@dataclass(frozen=True)
class _Chunk:
    """One embedded passage, with everything needed to find its text again."""

    embedding_id: str
    file_id: str
    embedding_type: str
    chunk_index: int | None
    timestamp_start: float | None
    page: int | None
    vector: np.ndarray


@dataclass(frozen=True)
class PassagePair:
    """A passage of the file being read, beside one it echoes."""

    text: str
    page: int | None
    timestamp: float | None
    other_file_id: str
    other_drive: str
    other_filename: str
    other_text: str
    other_page: int | None
    other_timestamp: float | None
    score: float


def _sample(rows: list, cap: int) -> list:
    """At most ``cap`` rows, spread evenly across the whole file.

    Taking the first ``cap`` instead would put the opening of a long
    document in front of a reader and leave its middle unexamined —
    which is the exact failure this feature was built to remove.
    """
    if len(rows) <= cap:
        return rows
    stride = len(rows) / cap
    return [rows[int(i * stride)] for i in range(cap)]


def _passage_count(file_id: str) -> int:
    """How many passage chunks a file has, before any sampling.

    The KNN budget is computed from this: a file's own chunks are, by
    construction, the nearest things to its own centroid, so they fill
    the global top-k before the ``file_id != :self`` predicate ever runs.
    """
    engine = get_search_engine()
    types = ",".join(f"'{t}'" for t in _PASSAGE_TYPES)
    with engine.connect() as conn:
        return int(
            conn.execute(
                sql_text(
                    "SELECT COUNT(*) FROM embeddings "
                    f"WHERE file_id = :f AND embedding_type IN ({types})"
                ),
                {"f": file_id},
            ).scalar()
            or 0
        )


def _load_chunks(file_ids: list[str], cap: int) -> dict[str, list[_Chunk]]:
    """Read each file's passage chunks, vectors included.

    Metadata and vectors are fetched separately: ``vec_text`` is a
    virtual table and the proven way to read vectors out of one is an
    ``embedding_id IN (...)`` lookup, not a join.
    """
    if not file_ids:
        return {}

    engine = get_search_engine()
    placeholders = ",".join(f":f{i}" for i in range(len(file_ids)))
    params = {f"f{i}": fid for i, fid in enumerate(file_ids)}
    types = ",".join(f"'{t}'" for t in _PASSAGE_TYPES)

    with engine.connect() as conn:
        rows = conn.execute(
            sql_text(
                "SELECT id, file_id, embedding_type, chunk_index, "
                "       timestamp_start, page "
                "FROM embeddings "
                f"WHERE file_id IN ({placeholders}) "
                f"  AND embedding_type IN ({types}) "
                "ORDER BY file_id, chunk_index, timestamp_start"
            ),
            params,
        ).fetchall()

        by_file: dict[str, list[tuple]] = {}
        for row in rows:
            by_file.setdefault(row[1], []).append(row)

        kept = [
            row
            for bucket in by_file.values()
            for row in _sample(bucket, cap)
        ]
        if not kept:
            return {}

        vec_placeholders = ",".join(f":v{i}" for i in range(len(kept)))
        vec_params = {f"v{i}": row[0] for i, row in enumerate(kept)}
        vec_rows = conn.execute(
            sql_text(
                "SELECT embedding_id, vector FROM vec_text "
                f"WHERE embedding_id IN ({vec_placeholders})"
            ),
            vec_params,
        ).fetchall()

    vectors = {
        row[0]: np.frombuffer(row[1], dtype=np.float32)
        for row in vec_rows
        if row[1]
    }

    chunks: dict[str, list[_Chunk]] = {}
    for row in kept:
        vector = vectors.get(row[0])
        if vector is None:
            # The metadata row outlived its vector. Nothing can be
            # scored against it.
            continue
        chunks.setdefault(row[1], []).append(
            _Chunk(
                embedding_id=row[0],
                file_id=row[1],
                embedding_type=row[2],
                chunk_index=row[3],
                timestamp_start=row[4],
                page=row[5],
                vector=vector,
            )
        )
    return chunks


def _knn_budgets(source_rows: int) -> list[int]:
    """The ``k`` values to try, in order.

    ``MATCH`` is a global KNN: sqlite-vec picks ``k`` rows across the
    whole table and only then does the join apply drive, kind and
    ``!= :self``. The source's own chunks are the nearest things to
    their own centroid, so on a long document they occupy the entire
    budget and the self-exclusion empties it. Budget for them.

    A second, wider attempt covers the rest of the post-filter loss
    (other drives, metadata rows). Two queries is the ceiling: past
    sqlite-vec's own ``k`` cap there is nothing further to ask for.
    """
    first = min(_KNN_K_MAX, source_rows + _KNN_POOL)
    if first >= _KNN_K_MAX:
        return [_KNN_K_MAX]
    return [first, _KNN_K_MAX]


def _nearest_files(
    centroid: np.ndarray,
    *,
    file_id: str,
    drive: str,
    limit: int = _CANDIDATE_POOL,
    source_rows: int = 0,
) -> list[str]:
    """Files whose passages sit closest to this file's centre of mass.

    A drive is a security boundary, so the candidate set never leaves
    the request's drive. The source file is excluded: a document is
    trivially closest to itself.
    """
    engine = get_search_engine()
    types = ",".join(f"'{t}'" for t in _PASSAGE_TYPES)

    found: list[str] = []
    with engine.connect() as conn:
        for k in _knn_budgets(source_rows):
            rows = conn.execute(
                sql_text(
                    "SELECT e.file_id, MIN(v.distance) AS d "
                    "FROM vec_text v "
                    "JOIN embeddings e ON CAST(e.id AS TEXT) = v.embedding_id "
                    "JOIN indexed_files f ON f.file_id = e.file_id "
                    "WHERE v.vector MATCH :vec AND k = :k "
                    f"  AND e.embedding_type IN ({types}) "
                    "  AND e.file_id != :self "
                    "  AND f.drive = :drive "
                    "  AND f.active = 1 "
                    "GROUP BY e.file_id "
                    "ORDER BY d "
                    "LIMIT :limit"
                ),
                {
                    "vec": centroid.tobytes(),
                    "k": k,
                    "self": file_id,
                    "drive": drive,
                    "limit": limit,
                },
            ).fetchall()
            found = [row[0] for row in rows]
            if len(found) >= limit:
                break

    return found


def _file_meta(file_ids: list[str]) -> dict[str, tuple[str, str]]:
    """``file_id -> (drive, filename)`` for the files being linked to."""
    if not file_ids:
        return {}
    engine = get_search_engine()
    placeholders = ",".join(f":f{i}" for i in range(len(file_ids)))
    params = {f"f{i}": fid for i, fid in enumerate(file_ids)}
    with engine.connect() as conn:
        rows = conn.execute(
            sql_text(
                "SELECT file_id, drive, filename FROM indexed_files "
                f"WHERE file_id IN ({placeholders})"
            ),
            params,
        ).fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def _resolve_text(chunk: _Chunk) -> str | None:
    """The chunk's full text, or None when it cannot be found.

    None is a real answer, and the caller drops the pair rather than
    substituting ``content_preview``: that column is truncated at 200
    characters, so showing it would put a prefix on screen while the
    hidden remainder is what actually produced the score.
    """
    engine = get_search_engine()
    with engine.connect() as conn:
        if chunk.embedding_type == "text_content":
            if chunk.chunk_index is None:
                # Indexed before the chunk key existed; re-indexing fills
                # it in.
                return None
            row = conn.execute(
                sql_text(
                    "SELECT text FROM fts_text_content "
                    "WHERE file_id = :f AND chunk_index = :ci LIMIT 1"
                ),
                # Every FTS5 column is text, chunk_index included.
                {"f": chunk.file_id, "ci": str(chunk.chunk_index)},
            ).fetchone()
        else:
            if chunk.timestamp_start is None:
                return None
            row = conn.execute(
                sql_text(
                    "SELECT text FROM transcript_chunks "
                    "WHERE file_id = :f AND timestamp_start = :s LIMIT 1"
                ),
                {"f": chunk.file_id, "s": chunk.timestamp_start},
            ).fetchone()

    return row[0] if row else None


def _rank_pairs(
    source: list[_Chunk],
    candidates: list[_Chunk],
    limit: int,
) -> list[tuple[float, _Chunk, _Chunk]]:
    """Best pairs by cosine similarity, spread across the material.

    Vectors are L2-normalised at write time, so the matrix product is
    already cosine similarity.

    At most one row per source passage and one per other file. Without
    that, a single paragraph with several close matches fills the whole
    list and everything else in the document goes unmentioned.
    """
    matrix = np.stack([c.vector for c in source]) @ np.stack(
        [c.vector for c in candidates]
    ).T
    hits = np.argwhere(matrix >= _MIN_SCORE)
    if hits.size == 0:
        return []

    order = np.argsort(-matrix[hits[:, 0], hits[:, 1]])
    used_source: set[int] = set()
    used_file: set[str] = set()
    pairs: list[tuple[float, _Chunk, _Chunk]] = []

    for idx in order:
        i, j = int(hits[idx][0]), int(hits[idx][1])
        other = candidates[j]
        if i in used_source or other.file_id in used_file:
            continue
        used_source.add(i)
        used_file.add(other.file_id)
        pairs.append((float(matrix[i][j]), source[i], other))
        if len(pairs) >= limit:
            break

    return pairs


def _build_pairs(
    source: list[_Chunk], candidate_ids: list[str], limit: int
) -> list[PassagePair]:
    """Stage 2, plus text resolution. Runs off the event loop."""
    by_file = _load_chunks(candidate_ids, _MAX_CANDIDATE_CHUNKS)
    candidates = [c for fid in candidate_ids for c in by_file.get(fid, [])]
    if not candidates:
        return []

    ranked = _rank_pairs(source, candidates, limit)
    if not ranked:
        return []

    meta = _file_meta([other.file_id for _, _, other in ranked])

    pairs: list[PassagePair] = []
    for score, mine, other in ranked:
        my_text = _resolve_text(mine)
        other_text = _resolve_text(other)
        if my_text is None or other_text is None:
            continue
        drive, filename = meta.get(other.file_id, ("", ""))
        pairs.append(
            PassagePair(
                text=my_text,
                page=mine.page,
                timestamp=mine.timestamp_start,
                other_file_id=other.file_id,
                other_drive=drive,
                other_filename=filename,
                other_text=other_text,
                other_page=other.page,
                other_timestamp=other.timestamp_start,
                score=score,
            )
        )
    return pairs


def _source_and_candidates(file_id: str, drive: str) -> tuple[list[_Chunk], list[str]]:
    """Stage 1. Runs off the event loop."""
    source = _load_chunks([file_id], _MAX_SOURCE_CHUNKS).get(file_id, [])
    if not source:
        return [], []

    centroid = np.mean(np.stack([c.vector for c in source]), axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        return source, []
    centroid = (centroid / norm).astype(np.float32)

    return source, _nearest_files(
        centroid,
        file_id=file_id,
        drive=drive,
        source_rows=_passage_count(file_id),
    )


async def find_related_passages(
    file_id: str,
    drive: str,
    credential: CallerCredential | None,
    limit: int = _DEFAULT_LIMIT,
) -> list[PassagePair]:
    """Passages of ``file_id`` paired with passages of verified files.

    Returns an empty list rather than an error whenever there is nothing
    to say: an unindexed file, a file whose closest neighbours are all
    unvouched, a library with nothing similar in it.
    """
    source, candidate_ids = await asyncio.to_thread(
        _source_and_candidates, file_id, drive
    )
    if not source or not candidate_ids:
        return []

    allowed = await _filter_file_ids_via_internal_api(
        candidate_ids, credential, trust_tier="verified"
    )
    # The cap lands here, not on the KNN: a verified file just below a
    # run of unverified neighbours has to survive long enough to be
    # asked about.
    verified = [fid for fid in candidate_ids if fid in allowed][:_CANDIDATE_FILES]
    if not verified:
        return []

    return await asyncio.to_thread(_build_pairs, source, verified, limit)
