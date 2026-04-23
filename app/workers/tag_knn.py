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
already applied.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from sqlalchemy import text as sql_text

from app.database import get_litloft_db, get_search_db, get_search_engine
from app.models import Embedding

logger = logging.getLogger(__name__)

# Tags used on only one other file are rarely worth suggesting — one
# co-occurrence could just mean the user made an ad-hoc tag once.
# Two files is the "this tag is used like a category" threshold.
_MIN_SUPPORT = 2


def _average_clip_vector(file_id: str) -> np.ndarray | None:
    """Return the averaged+normalized CLIP vector for a file.

    Videos store one vector per extracted key frame; the average
    approximates "what does this video generally look like". Image
    files have a single vector so averaging is a no-op.
    """
    with get_search_db() as session:
        rows = (
            session.query(Embedding.id)
            .filter(
                Embedding.file_id == file_id,
                Embedding.embedding_type == "clip",
            )
            .all()
        )
        embedding_ids = [r.id for r in rows]

    if not embedding_ids:
        return None

    vectors: list[np.ndarray] = []
    with get_search_engine().connect() as conn:
        for eid in embedding_ids:
            row = conn.execute(
                sql_text("SELECT vector FROM vec_clip WHERE embedding_id = :eid"),
                {"eid": eid},
            ).fetchone()
            if row and row[0]:
                vectors.append(np.frombuffer(row[0], dtype=np.float32))

    if not vectors:
        return None

    avg = np.mean(vectors, axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else avg


def _query_nearest_file_ids(
    query_vector: np.ndarray,
    source_file_id: str,
    k: int,
) -> list[tuple[str, float]]:
    """Return the k nearest *distinct* file IDs by CLIP similarity.

    vec_clip stores per-frame embeddings so a single similar video
    can occupy several top results; we dedupe to one entry per file,
    keeping the best (lowest-distance) frame's similarity as the
    file's score. Fetches extra rows up front to absorb both the
    source file's own frames and the dedup churn.
    """
    # Heuristic fetch size: source frames could be hundreds for long
    # videos, and each candidate can contribute several frames too.
    # Pulling k * 10 + 50 keeps things manageable even on large libraries.
    fetch_limit = max(k * 10 + 50, 100)

    # sqlite-vec's KNN planner requires LIMIT (or `k = ?`) to apply
    # directly to the virtual vec_clip table. Joining to embeddings in
    # the same statement moves the LIMIT outside the optimizer's reach
    # and raises "A LIMIT or 'k = ?' constraint is required". Do the
    # KNN first in a subquery, then JOIN + filter on the result — same
    # pattern app.search uses for its CLIP similarity lookups.
    scores: dict[str, float] = {}
    with get_search_engine().connect() as conn:
        rows = conn.execute(
            sql_text(
                "SELECT e.file_id, v.distance FROM ("
                "  SELECT embedding_id, distance FROM vec_clip "
                "  WHERE vector MATCH :vec "
                "  ORDER BY distance "
                "  LIMIT :limit"
                ") v "
                "JOIN embeddings e ON v.embedding_id = e.id "
                "WHERE e.file_id != :src "
                "ORDER BY v.distance"
            ),
            {
                "vec": query_vector.astype(np.float32).tobytes(),
                "src": source_file_id,
                "limit": fetch_limit,
            },
        ).fetchall()

    for file_id, distance in rows:
        # Convert L2 distance on normalized vectors to cosine similarity:
        # for unit vectors, ||a-b||² = 2 - 2·cos(a,b), so cos = 1 - d²/2.
        sim = 1.0 - (float(distance) ** 2) / 2.0
        # Keep the best score we've seen for each file.
        if sim > scores.get(file_id, -1.0):
            scores[file_id] = sim
        if len(scores) >= k:
            break

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def _load_tags_for_files(file_ids: list[str]) -> dict[str, list[str]]:
    """Fetch the Litloft tags applied to each given file ID."""
    if not file_ids:
        return {}

    try:
        with get_litloft_db() as session:
            # Parameterize every id explicitly so SQLite receives
            # literals it can cache (no IN-list reuse headaches).
            placeholders = ",".join(f":id{i}" for i in range(len(file_ids)))
            params = {f"id{i}": fid for i, fid in enumerate(file_ids)}
            rows = session.execute(
                sql_text(
                    "SELECT ft.file_id, t.name "
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
    for file_id, tag_name in rows:
        grouped[file_id].append(tag_name)
    return dict(grouped)


def recommend_tags_by_similarity(
    file_id: str,
    *,
    k_neighbors: int = 20,
    top_tags: int = 10,
    min_support: int = _MIN_SUPPORT,
) -> list[tuple[str, float]]:
    """Suggest tags by looking at already-tagged visually similar files.

    Returns an empty list when there are no CLIP embeddings for the
    source file (e.g. documents) or when no neighbor has any tags
    (cold start). Scores are in ``[0, k_neighbors]`` range —
    roughly the weighted neighbor count, higher is better.

    Args:
        file_id: The file to recommend tags for.
        k_neighbors: How many similar files to consider.
        top_tags: Max tag recommendations to return.
        min_support: Require at least this many neighbors to use a tag
            before it qualifies as a recommendation.

    Returns:
        List of (tag_name, confidence_score) sorted by score desc.
    """
    query_vec = _average_clip_vector(file_id)
    if query_vec is None:
        return []

    neighbors = _query_nearest_file_ids(query_vec, file_id, k_neighbors)
    if not neighbors:
        return []

    neighbor_ids = [fid for fid, _ in neighbors]
    tags_by_file = _load_tags_for_files(neighbor_ids)
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
