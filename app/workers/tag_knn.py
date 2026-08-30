"""k-NN tag recommendation using CLIP embedding similarity.

For a source file, finds the CLIP-most-similar already-tagged files
and aggregates their tags into a ranked suggestion list. This is the
"if it looks like your previous cooking videos, tag it with the same
things you tagged those with" pathway — the most effective way to
make local tagging smarter as the user keeps tagging files.

The quality of the suggestions depends on:

- Having some tagged files at all (cold start: returns nothing)
- CLIP embeddings existing for the source file (no image/video CLIP
  vector → no k-NN result)

Unlike the CLIP zero-shot concept scorer, this doesn't require a
curated vocabulary; the "vocabulary" is whatever tags the user has
already applied *in the drive being tagged*. Neighbors and their tags
stop at the drive boundary.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from sqlalchemy import text as sql_text

from app.database import get_litloft_db, get_search_db_read, get_search_engine
from app.models import Embedding, IndexedFile
from app.workers.clip_concepts import load_file_clip_vectors

logger = logging.getLogger(__name__)

# Tags used on only one other file are rarely worth suggesting — one
# co-occurrence could just mean the user made an ad-hoc tag once.
# Two files is the "this tag is used like a category" threshold.
_MIN_SUPPORT = 2

# vec_clip is a single global index: MATCH ranks across every drive and
# knows nothing about the boundary between them, so the drive filter is
# applied to what comes back and the fetch is widened until enough
# in-drive neighbors survive it. Same shape as the neighbour fetch in
# ``app.search``.
_FETCH_FACTORS = (1, 4, 16)
# Ceiling on a single KNN, so a library where the current drive is a
# small minority cannot turn one recommendation into a table scan.
_FETCH_MAX = 4096

# Drives already reported for tag/file drive drift. The condition is a
# property of the library, not of one file, so reporting it per tagged
# file would bury the log during an on_index sweep.
_drift_reported: set[str] = set()


def _average_clip_vector(
    file_id: str,
    vectors: list[np.ndarray] | None = None,
) -> np.ndarray | None:
    """Return the averaged+normalized CLIP vector for a file.

    Videos store one vector per extracted key frame; the average
    approximates "what does this video generally look like". Image
    files have a single vector so averaging is a no-op.

    Args:
        file_id: The file to load vectors for.
        vectors: Optional pre-loaded CLIP vectors (see
            ``clip_concepts.load_file_clip_vectors``). Pass this when
            the caller already fetched the vectors for another
            pipeline (e.g. CLIP concept scoring) to avoid a redundant
            DB round trip; when None, this function loads them itself.
    """
    if vectors is None:
        vectors = load_file_clip_vectors(file_id)
    if not vectors:
        return None

    avg = np.mean(vectors, axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else avg


def _in_drive_file_ids(
    embedding_ids: list[str],
    source_file_id: str,
    drive: str,
) -> dict[str, str]:
    """Map embedding id → file id, keeping only this drive's active files.

    Resolved in a second statement rather than a join: ``vec_clip``
    stores no drive, and sqlite-vec's KNN planner requires the LIMIT
    (or ``k = ?``) to apply directly to the virtual table — joining
    anything into that statement moves the LIMIT out of the
    optimizer's reach and raises "A LIMIT or 'k = ?' constraint is
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
                Embedding.file_id != source_file_id,
                IndexedFile.drive == drive,
                IndexedFile.active.is_(True),
            )
            .all()
        )
    return {embedding_id: file_id for embedding_id, file_id in rows}


def _query_nearest_file_ids(
    query_vector: np.ndarray,
    source_file_id: str,
    k: int,
    drive: str,
) -> list[tuple[str, float]]:
    """Return the k nearest *distinct* in-drive file IDs by CLIP similarity.

    vec_clip stores per-frame embeddings so a single similar video
    can occupy several top results; we dedupe to one entry per file,
    keeping the best (lowest-distance) frame's similarity as the
    file's score. Fetches extra rows up front to absorb the source
    file's own frames, the dedup churn, and every neighbor that
    belongs to another drive.
    """
    # Heuristic fetch size: source frames could be hundreds for long
    # videos, and each candidate can contribute several frames too.
    # Pulling k * 10 + 50 keeps things manageable even on large libraries.
    base_limit = max(k * 10 + 50, 100)

    scores: dict[str, float] = {}
    engine = get_search_engine()
    for factor in _FETCH_FACTORS:
        asked = min(base_limit * factor, _FETCH_MAX)
        with engine.connect() as conn:
            rows = conn.execute(
                sql_text(
                    "SELECT embedding_id, distance FROM vec_clip "
                    "WHERE vector MATCH :vec "
                    "ORDER BY distance "
                    "LIMIT :limit"
                ),
                {
                    "vec": query_vector.astype(np.float32).tobytes(),
                    "limit": asked,
                },
            ).fetchall()

        if not rows:
            return []

        distances = {row[0]: float(row[1]) for row in rows}
        file_ids = _in_drive_file_ids(list(distances), source_file_id, drive)

        scores = {}
        for embedding_id, file_id in file_ids.items():
            # Convert L2 distance on normalized vectors to cosine similarity:
            # for unit vectors, ||a-b||² = 2 - 2·cos(a,b), so cos = 1 - d²/2.
            sim = 1.0 - (distances[embedding_id] ** 2) / 2.0
            # Keep the best score we've seen for each file.
            if sim > scores.get(file_id, -1.0):
                scores[file_id] = sim

        if len(scores) >= k or asked >= _FETCH_MAX or len(rows) < asked:
            # Enough in-drive neighbors, or the index has nothing further
            # to give.
            break

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def _load_tags_for_files(
    file_ids: list[str],
    drive: str,
) -> dict[str, list[str]]:
    """Fetch the Litloft tags applied to each given file ID.

    The neighbors are already drive-scoped; the tag-side drive check is
    the second layer, so a tag row mis-attributed to another drive
    still cannot reach this drive's suggestions.

    A dropped row means the tag rows and the files they hang off
    disagree about which drive they are in — core's tags-table
    migration stamps every pre-existing tag with a single drive name,
    so a library that was tagged before drives were partitioned has
    exactly that disagreement. Suggestions then go quiet with nothing
    to explain why, hence the count in the log.
    """
    if not file_ids:
        return {}

    try:
        with get_litloft_db() as session:
            # Parameterize every id explicitly so SQLite receives
            # literals it can cache (no IN-list reuse headaches).
            placeholders = ",".join(f":id{i}" for i in range(len(file_ids)))
            params: dict[str, str] = {
                f"id{i}": fid for i, fid in enumerate(file_ids)
            }
            rows = session.execute(
                sql_text(
                    "SELECT ft.file_id, t.name, t.drive "
                    "FROM file_tags ft "
                    "JOIN tags t ON t.id = ft.tag_id "
                    f"WHERE ft.file_id IN ({placeholders})"
                ),
                params,
            ).fetchall()
    except Exception as e:
        logger.warning("k-NN tag lookup failed: %s", e)
        return {}

    grouped: dict[str, list[str]] = defaultdict(list)
    dropped = 0
    for file_id, tag_name, tag_drive in rows:
        if tag_drive != drive:
            dropped += 1
            continue
        grouped[file_id].append(tag_name)

    if dropped and drive not in _drift_reported:
        _drift_reported.add(drive)
        logger.warning(
            "Drive %s has tag rows labelled for another drive (%d ignored "
            "while recommending tags). Tag suggestions from similar files "
            "stay quiet until those rows carry the right drive. Reported "
            "once per drive.",
            drive, dropped,
        )
    return dict(grouped)


def recommend_tags_by_similarity(
    file_id: str,
    *,
    drive: str,
    k_neighbors: int = 20,
    top_tags: int = 10,
    min_support: int = _MIN_SUPPORT,
    vectors: list[np.ndarray] | None = None,
) -> list[tuple[str, float]]:
    """Suggest tags by looking at already-tagged visually similar files.

    Returns an empty list when there are no CLIP embeddings for the
    source file (e.g. documents) or when no neighbor in the same drive
    has any tags (cold start). Scores are in ``[0, k_neighbors]`` range —
    roughly the weighted neighbor count, higher is better.

    Args:
        file_id: The file to recommend tags for.
        drive: Only this drive's files and tags may be consulted — a
            drive is a security boundary, so a neighbor outside it is
            not a neighbor at all.
        k_neighbors: How many similar files to consider.
        top_tags: Max tag recommendations to return.
        min_support: Require at least this many neighbors to use a tag
            before it qualifies as a recommendation.
        vectors: Optional pre-loaded CLIP vectors for this file, passed
            through to ``_average_clip_vector`` to avoid a redundant
            DB round trip when another pipeline already fetched them.

    Returns:
        List of (tag_name, confidence_score) sorted by score desc.
    """
    query_vec = _average_clip_vector(file_id, vectors)
    if query_vec is None:
        return []

    neighbors = _query_nearest_file_ids(query_vec, file_id, k_neighbors, drive)
    if not neighbors:
        return []

    neighbor_ids = [fid for fid, _ in neighbors]
    tags_by_file = _load_tags_for_files(neighbor_ids, drive)
    if not tags_by_file:
        return []

    # Similarity-weighted vote: tags from more-similar neighbors count more.
    tag_score: dict[str, float] = defaultdict(float)
    tag_support: dict[str, int] = defaultdict(int)
    for fid, sim in neighbors:
        for tag in tags_by_file.get(fid, []):
            tag_score[tag] += sim
            tag_support[tag] += 1

    # Filter by minimum support so single-occurrence tags don't bubble up.
    ranked = [
        (tag, score)
        for tag, score in tag_score.items()
        if tag_support[tag] >= min_support
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_tags]
