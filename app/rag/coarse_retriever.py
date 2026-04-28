"""Stage 1 coarse retrieval: per-file shortlist by metadata embedding.

Drives the hierarchical RAG pipeline's first stage. Embeds the user's
raw natural-language query (NOT the keyword-transformed form — file
summaries are already domain vocabulary, see spec §4.2) and pulls the
top-K closest files from the ``metadata`` embedding channel,
restricted to a single drive (drive == security boundary).

Phase 1 logs the result without acting on it; Phase 2 feeds the
returned ``file_ids`` into ``app.search.search()`` as a scope filter.
"""

import asyncio
from dataclasses import dataclass

from sqlalchemy import text as sql_text

from app.database import get_search_db, get_search_engine
from app.search import _l2_to_cosine_similarity
from app.workers.embedder import embed_query


@dataclass(frozen=True)
class ShortlistResult:
    """Result of Stage 1 coarse retrieval.

    ``file_ids`` is ordered by descending coarse score; ``scores`` is
    parallel to it. ``top_score`` is the maximum — 0.0 when the
    shortlist is empty so callers can compare against
    ``coarse_score_threshold`` without special-casing emptiness.
    ``drive_file_count`` is the active-file count for the drive,
    used by the small-drive bypass.
    """

    file_ids: tuple[str, ...]
    scores: tuple[float, ...]
    top_score: float
    drive_file_count: int


async def coarse_retrieve(
    query: str,
    drive: str,
    *,
    top_k: int,
) -> ShortlistResult:
    """Compute a Stage 1 shortlist for ``drive`` from a raw query.

    Args:
        query: User's natural-language question. Embedded as-is —
            keyword extraction would strip the contextual phrasing
            that makes the summary embedding match meaningful.
        drive: Target drive name. Coarse retrieval is drive-scoped to
            preserve the drive-as-security-boundary invariant.
        top_k: Maximum shortlist size.

    Returns:
        A ``ShortlistResult`` with file_ids ordered by descending
        cosine similarity. Empty file_ids + ``top_score=0.0`` when no
        metadata embeddings exist for this drive.
    """
    # Embedding model load is sync + CPU-bound; keep the event loop
    # responsive for concurrent SSE streams.
    query_vector = await asyncio.to_thread(embed_query, query)
    vec_bytes = query_vector.tobytes()

    # sqlite-vec MATCH returns the GLOBAL top-K nearest neighbors across
    # the entire vec_text table. When multiple drives × multiple
    # embedding types (transcript / text_content / blip_caption /
    # metadata) share the table, a naive ``LIMIT :top_k`` followed by
    # post-WHERE filters can yield zero target-drive metadata rows even
    # when the drive's metadata channel has plenty of strong matches —
    # the global neighborhood is dominated by other drives / channels.
    #
    # Mitigation: over-fetch by a wide constant so the post-filter has
    # a generous candidate pool. The pattern mirrors the per-file
    # similar-files code path in ``app/search.py`` (~line 1640) which
    # over-fetches by ``source_count + limit + 20``. Here we use a
    # multiplicative bump because the post-filter is more aggressive
    # (drives × types vs. just type).
    over_fetch = max(top_k * 50, 500)

    engine = get_search_engine()
    with engine.connect() as conn:
        # sqlite-vec rejects ``ORDER BY ... LIMIT`` on the outer SELECT
        # when JOINs sit between the MATCH and the LIMIT — the LIMIT
        # cannot be pushed into the vec0 KNN scan, and vec0 fails with
        # "A LIMIT or 'k = ?' constraint is required on vec0 knn
        # queries". Isolate the KNN scan in a CTE so the limit binds
        # directly to vec_text, then JOIN the post-filter on the
        # over-fetched id set.
        rows = conn.execute(
            sql_text(
                "WITH knn AS ("
                "  SELECT embedding_id, distance "
                "  FROM vec_text "
                "  WHERE vector MATCH :vec "
                "  ORDER BY distance "
                "  LIMIT :over_fetch"
                ") "
                "SELECT i.file_id, knn.distance "
                "FROM knn "
                "JOIN embeddings e ON e.id = knn.embedding_id "
                "JOIN indexed_files i ON i.file_id = e.file_id "
                "WHERE e.embedding_type = 'metadata' "
                "  AND i.drive = :drive "
                "  AND i.active = 1 "
                "ORDER BY knn.distance"
            ),
            {"vec": vec_bytes, "drive": drive, "over_fetch": over_fetch},
        ).fetchall()

    with get_search_db() as session:
        count_row = session.execute(
            sql_text(
                "SELECT COUNT(*) FROM indexed_files "
                "WHERE drive = :drive AND active = 1"
            ),
            {"drive": drive},
        ).fetchone()
        drive_file_count = int(count_row[0]) if count_row else 0

    if not rows:
        return ShortlistResult(
            file_ids=(),
            scores=(),
            top_score=0.0,
            drive_file_count=drive_file_count,
        )

    # Dedupe per file_id keeping the smallest distance (best score).
    # The MATCH-against-metadata path normally gives one row per file
    # already, but the JOIN doesn't enforce that — defensive dedup.
    best_by_file: dict[str, float] = {}
    for fid, distance in rows:
        if fid not in best_by_file or distance < best_by_file[fid]:
            best_by_file[fid] = distance

    sorted_pairs = sorted(best_by_file.items(), key=lambda kv: kv[1])
    # Trim the over-fetched, post-filtered candidate pool down to the
    # caller-requested top_k. The over-fetch above is a defence against
    # sqlite-vec's global neighborhood selection — once the WHERE has
    # restricted us to target-drive metadata rows, the contractual
    # shortlist size is still ``top_k``.
    sorted_pairs = sorted_pairs[:top_k]
    file_ids = tuple(fid for fid, _ in sorted_pairs)
    scores = tuple(_l2_to_cosine_similarity(d) for _, d in sorted_pairs)
    top_score = scores[0] if scores else 0.0

    return ShortlistResult(
        file_ids=file_ids,
        scores=scores,
        top_score=top_score,
        drive_file_count=drive_file_count,
    )
