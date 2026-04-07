"""Hybrid search logic: vector similarity + keyword matching.

Combines semantic vector search with traditional keyword matching
to provide high-quality search results ranked by combined score.
Results are grouped at file level with segment timestamps.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_search_db, get_search_engine
from app.models import Embedding, IndexedFile
from app.workers.clip import embed_text_clip, CLIP_DIM
from app.workers.embedder import embed_query, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Time window for grouping co-located matches (seconds)
SEGMENT_GROUP_WINDOW = 30


def _l2_to_cosine_similarity(distance: float) -> float:
    """Convert L2 distance to cosine similarity for normalized vectors.

    For L2-normalized vectors: L2² = 2 - 2·cos(θ)
    Therefore: cos(θ) = 1 - L2²/2

    Args:
        distance: L2 (Euclidean) distance from sqlite-vec.

    Returns:
        Cosine similarity in range [-1, 1].
    """
    return 1.0 - (distance * distance) / 2.0


@dataclass(frozen=True)
class MatchInfo:
    """A single matching segment within a file."""

    match_type: str  # "metadata", "transcript", "clip", "text_content"
    text: str
    score: float
    timestamp_start: float | None = None
    timestamp_end: float | None = None


@dataclass(frozen=True)
class SegmentGroup:
    """A group of matches within the same time range."""

    time_range: tuple[float, float] | None
    matches: tuple[MatchInfo, ...]


@dataclass(frozen=True)
class SearchResult:
    """A file-level search result with grouped segment matches."""

    file_id: str
    drive: str
    filename: str
    file_type: str
    score: float
    match_types: tuple[str, ...]
    segments: tuple[SegmentGroup, ...]


@dataclass(frozen=True)
class SearchResponse:
    """Complete search response."""

    results: tuple[SearchResult, ...]
    total: int
    indexed_files: int
    service_version: str


def search(
    query: str,
    limit: int | None = None,
    file_type: str | None = None,
    drive: str | None = None,
) -> SearchResponse:
    """Execute a hybrid search query.

    Performs vector similarity search across text and CLIP embeddings,
    combines with keyword matching, and returns file-level results.

    Args:
        query: The search query string.
        limit: Maximum number of results.
        file_type: Optional file type filter.
        drive: Optional drive name filter.

    Returns:
        SearchResponse with ranked results.
    """
    search_config = settings.search
    effective_limit = min(
        limit or search_config.default_limit,
        search_config.max_limit,
    )

    # Generate query embeddings
    text_vector = embed_query(query)
    clip_vector = _safe_clip_embed(query)

    # Vector search across both tables
    text_matches = _vector_search_text(text_vector, effective_limit * 5)
    clip_matches = _vector_search_clip(clip_vector, effective_limit * 5) if clip_vector is not None else []

    # Keyword matching
    keyword_matches = _keyword_search(query, effective_limit * 3)

    # Combine and rank results
    file_scores = _combine_scores(
        text_matches=text_matches,
        clip_matches=clip_matches,
        keyword_matches=keyword_matches,
        alpha=search_config.alpha,
    )

    # Apply filters and build results
    results = _build_results(
        file_scores=file_scores,
        file_type=file_type,
        drive=drive,
        limit=effective_limit,
    )

    # Get total indexed count
    with get_search_db() as session:
        indexed_count = session.query(IndexedFile).filter(
            IndexedFile.active.is_(True)
        ).count()

    return SearchResponse(
        results=tuple(results),
        total=len(results),
        indexed_files=indexed_count,
        service_version=settings.service_version,
    )


def _safe_clip_embed(query: str) -> np.ndarray | None:
    """Attempt CLIP text embedding, returning None on failure.

    Args:
        query: The search query.

    Returns:
        CLIP embedding vector or None.
    """
    try:
        return embed_text_clip(query)
    except Exception as e:
        logger.warning("CLIP text embedding failed: %s", e)
        return None


@dataclass
class _VectorMatch:
    """Internal vector search match."""

    embedding_id: str
    file_id: str
    score: float
    embedding_type: str
    content_preview: str
    timestamp_start: float | None
    timestamp_end: float | None


def _vector_search_text(
    query_vector: np.ndarray, limit: int
) -> list[_VectorMatch]:
    """Search the text vector table for similar embeddings.

    Args:
        query_vector: Query embedding vector.
        limit: Maximum results.

    Returns:
        List of vector matches.
    """
    engine = get_search_engine()
    vec_bytes = query_vector.tobytes()

    with engine.connect() as conn:
        rows = conn.execute(
            sql_text(
                "SELECT embedding_id, distance "
                "FROM vec_text "
                "WHERE vector MATCH :vec "
                "ORDER BY distance "
                "LIMIT :limit"
            ),
            {"vec": vec_bytes, "limit": limit},
        ).fetchall()

    if not rows:
        return []

    embedding_ids = [row[0] for row in rows]
    distances = {row[0]: row[1] for row in rows}

    min_score = settings.search.min_score_text

    with get_search_db() as session:
        embeddings = (
            session.query(Embedding)
            .filter(Embedding.id.in_(embedding_ids))
            .all()
        )

        results: list[_VectorMatch] = []
        for emb in embeddings:
            score = _l2_to_cosine_similarity(distances.get(emb.id, 2.0))
            if score < min_score:
                continue
            results = [
                *results,
                _VectorMatch(
                    embedding_id=emb.id,
                    file_id=emb.file_id,
                    score=score,
                    embedding_type=emb.embedding_type,
                    content_preview=emb.content_preview,
                    timestamp_start=emb.timestamp_start,
                    timestamp_end=emb.timestamp_end,
                ),
            ]
        return results


def _vector_search_clip(
    query_vector: np.ndarray, limit: int
) -> list[_VectorMatch]:
    """Search the CLIP vector table for similar embeddings.

    Args:
        query_vector: CLIP query embedding vector.
        limit: Maximum results.

    Returns:
        List of vector matches.
    """
    engine = get_search_engine()
    vec_bytes = query_vector.tobytes()

    with engine.connect() as conn:
        rows = conn.execute(
            sql_text(
                "SELECT embedding_id, distance "
                "FROM vec_clip "
                "WHERE vector MATCH :vec "
                "ORDER BY distance "
                "LIMIT :limit"
            ),
            {"vec": vec_bytes, "limit": limit},
        ).fetchall()

    if not rows:
        return []

    embedding_ids = [row[0] for row in rows]
    distances = {row[0]: row[1] for row in rows}

    min_score = settings.search.min_score_clip

    with get_search_db() as session:
        embeddings = (
            session.query(Embedding)
            .filter(Embedding.id.in_(embedding_ids))
            .all()
        )

        results: list[_VectorMatch] = []
        for emb in embeddings:
            score = _l2_to_cosine_similarity(distances.get(emb.id, 2.0))
            if score < min_score:
                continue
            results = [
                *results,
                _VectorMatch(
                    embedding_id=emb.id,
                    file_id=emb.file_id,
                    score=score,
                    embedding_type=emb.embedding_type,
                    content_preview=emb.content_preview,
                    timestamp_start=emb.timestamp_start,
                    timestamp_end=emb.timestamp_end,
                ),
            ]
        return results


@dataclass
class _KeywordMatch:
    """Internal keyword search match."""

    file_id: str
    score: float
    matched_field: str


def _build_fts_query(query: str) -> str:
    """Build an FTS5 trigram query string.

    For terms with 2 or fewer characters, wraps in double quotes so
    FTS5 trigram tokenizer matches the literal substring. Longer terms
    are also quoted to ensure exact substring matching.

    Args:
        query: Raw search query.

    Returns:
        FTS5 query string safe for the trigram tokenizer.
    """
    terms = [t.strip() for t in query.split() if t.strip()]
    if not terms:
        return ""
    # Quote each term for literal substring matching with trigram tokenizer.
    # Strip double quotes to prevent FTS5 syntax errors from user input.
    quoted = [f'"{t.replace(chr(34), "")}"' for t in terms]
    return " AND ".join(quoted)


# Field weight mapping for FTS5 keyword scoring
_FIELD_WEIGHTS: dict[str, float] = {
    "filename": 1.0,
    "title": 0.9,
    "tags_text": 0.8,
    "description": 0.6,
}


def _keyword_search(query: str, limit: int) -> list[_KeywordMatch]:
    """Search indexed files using FTS5 trigram index.

    Uses SQLite FTS5 with trigram tokenizer for efficient substring
    matching across filename, title, description, and tags.

    Args:
        query: The search query.
        limit: Maximum results.

    Returns:
        List of keyword matches with relevance scores.
    """
    fts_query = _build_fts_query(query)
    if not fts_query:
        return []

    engine = get_search_engine()

    with engine.connect() as conn:
        # FTS5 rank is negative (more negative = more relevant)
        rows = conn.execute(
            sql_text(
                "SELECT f.file_id, -f.rank AS relevance "
                "FROM fts_files f "
                "WHERE fts_files MATCH :query "
                "ORDER BY f.rank "
                "LIMIT :limit"
            ),
            {"query": fts_query, "limit": limit},
        ).fetchall()

    if not rows:
        return []

    # Normalize scores to 0-1 range
    max_relevance = max(row[1] for row in rows) if rows else 1.0
    normalizer = max_relevance if max_relevance > 0 else 1.0

    # Filter to only active files
    file_ids = [row[0] for row in rows]
    with get_search_db() as session:
        active_ids = {
            f.file_id
            for f in session.query(IndexedFile.file_id)
            .filter(
                IndexedFile.file_id.in_(file_ids),
                IndexedFile.active.is_(True),
            )
            .all()
        }

    return [
        _KeywordMatch(
            file_id=row[0],
            score=min(row[1] / normalizer, 1.0),
            matched_field="fts",
        )
        for row in rows
        if row[0] in active_ids
    ]


@dataclass
class _FileScore:
    """Aggregated score for a file across all match sources."""

    file_id: str
    combined_score: float
    matches: list[MatchInfo] = field(default_factory=list)
    match_types: set[str] = field(default_factory=set)


def _get_type_weight(embedding_type: str) -> float:
    """Get the scoring weight for an embedding type.

    Args:
        embedding_type: One of "metadata", "transcript", "text_content", "clip".

    Returns:
        Weight multiplier from config.
    """
    search_config = settings.search
    weights = {
        "metadata": search_config.type_weight_metadata,
        "transcript": search_config.type_weight_transcript,
        "text_content": search_config.type_weight_text_content,
        "clip": search_config.type_weight_clip,
    }
    return weights.get(embedding_type, 1.0)


def _combine_scores(
    text_matches: list[_VectorMatch],
    clip_matches: list[_VectorMatch],
    keyword_matches: list[_KeywordMatch],
    alpha: float,
) -> dict[str, _FileScore]:
    """Combine vector and keyword scores into file-level rankings.

    Uses per-type best scores with type weights to prevent content-heavy
    files from dominating through sheer volume of embeddings.

    Args:
        text_matches: Text vector search results.
        clip_matches: CLIP vector search results.
        keyword_matches: Keyword search results.
        alpha: Weight for vector similarity (0-1).

    Returns:
        Dict mapping file_id to aggregated scores.
    """
    # Phase 1: Collect best score per (file, embedding_type)
    file_type_best: dict[str, dict[str, float]] = {}
    file_matches: dict[str, list[MatchInfo]] = {}
    file_match_types: dict[str, set[str]] = {}

    all_vector_matches = [
        *((m, m.embedding_type) for m in text_matches),
        *((m, "clip") for m in clip_matches),
    ]

    for match, match_type in all_vector_matches:
        fid = match.file_id

        type_best = file_type_best.setdefault(fid, {})
        type_best[match_type] = max(type_best.get(match_type, 0.0), match.score)

        file_match_types.setdefault(fid, set()).add(match_type)
        file_matches.setdefault(fid, []).append(MatchInfo(
            match_type=match_type,
            text=match.content_preview,
            score=match.score,
            timestamp_start=match.timestamp_start,
            timestamp_end=match.timestamp_end,
        ))

    # Phase 2: Compute weighted vector score per file
    file_scores: dict[str, _FileScore] = {}

    for fid, type_best in file_type_best.items():
        best_weighted = max(
            score * _get_type_weight(etype)
            for etype, score in type_best.items()
        )
        vector_contribution = best_weighted * alpha

        file_scores[fid] = _FileScore(
            file_id=fid,
            combined_score=vector_contribution,
            matches=file_matches.get(fid, []),
            match_types=file_match_types.get(fid, set()),
        )

    # Phase 3: Add keyword contribution
    for match in keyword_matches:
        fs = file_scores.setdefault(
            match.file_id,
            _FileScore(file_id=match.file_id, combined_score=0.0),
        )
        keyword_contribution = match.score * (1.0 - alpha)
        fs.combined_score = fs.combined_score + keyword_contribution
        fs.match_types.add("keyword")

    return file_scores


def _build_results(
    file_scores: dict[str, _FileScore],
    file_type: str | None,
    drive: str | None,
    limit: int,
) -> list[SearchResult]:
    """Build final search results from aggregated scores.

    Applies filters, groups matches into segments, and sorts by score.

    Args:
        file_scores: Aggregated file scores.
        file_type: Optional file type filter.
        drive: Optional drive filter.
        limit: Maximum results.

    Returns:
        Sorted list of SearchResult objects.
    """
    if not file_scores:
        return []

    file_ids = list(file_scores.keys())

    with get_search_db() as session:
        query = session.query(IndexedFile).filter(
            IndexedFile.file_id.in_(file_ids),
            IndexedFile.active.is_(True),
        )

        if file_type:
            query = query.filter(IndexedFile.file_type == file_type)
        if drive:
            query = query.filter(IndexedFile.drive == drive)

        files = {f.file_id: f for f in query.all()}

    results: list[SearchResult] = []

    for file_id, fs in file_scores.items():
        file = files.get(file_id)
        if file is None:
            continue

        segments = _group_matches_into_segments(fs.matches)

        result = SearchResult(
            file_id=file_id,
            drive=file.drive,
            filename=file.filename,
            file_type=file.file_type,
            score=fs.combined_score,
            match_types=tuple(sorted(fs.match_types)),
            segments=tuple(segments),
        )
        results = [*results, result]

    # Sort by score descending
    results = sorted(results, key=lambda r: r.score, reverse=True)

    # Dynamic cutoff: discard results far below the top score
    if results:
        cutoff_ratio = settings.search.score_cutoff_ratio
        top_score = results[0].score
        min_combined = top_score * cutoff_ratio
        results = [r for r in results if r.score >= min_combined]

    return results[:limit]


def _group_matches_into_segments(
    matches: list[MatchInfo],
) -> list[SegmentGroup]:
    """Group co-located matches into time-based segments.

    Matches without timestamps are grouped as "general" matches.
    Timed matches are grouped if they fall within SEGMENT_GROUP_WINDOW.

    Args:
        matches: List of individual matches.

    Returns:
        List of grouped segments.
    """
    timed: list[MatchInfo] = []
    general: list[MatchInfo] = []

    for match in matches:
        if match.timestamp_start is not None:
            timed = [*timed, match]
        else:
            general = [*general, match]

    segments: list[SegmentGroup] = []

    # Group general matches
    if general:
        segments = [
            *segments,
            SegmentGroup(time_range=None, matches=tuple(general)),
        ]

    # Group timed matches by proximity
    if timed:
        sorted_timed = sorted(timed, key=lambda m: m.timestamp_start or 0)
        current_group: list[MatchInfo] = [sorted_timed[0]]
        group_start = sorted_timed[0].timestamp_start or 0
        group_end = sorted_timed[0].timestamp_end or group_start + SEGMENT_GROUP_WINDOW

        for match in sorted_timed[1:]:
            match_start = match.timestamp_start or 0

            if match_start <= group_end + SEGMENT_GROUP_WINDOW:
                # Extend current group
                current_group = [*current_group, match]
                match_end = match.timestamp_end or match_start + SEGMENT_GROUP_WINDOW
                group_end = max(group_end, match_end)
            else:
                # Flush current group and start new one
                segments = [
                    *segments,
                    SegmentGroup(
                        time_range=(group_start, group_end),
                        matches=tuple(current_group),
                    ),
                ]
                current_group = [match]
                group_start = match_start
                group_end = match.timestamp_end or match_start + SEGMENT_GROUP_WINDOW

        # Flush final group
        if current_group:
            segments = [
                *segments,
                SegmentGroup(
                    time_range=(group_start, group_end),
                    matches=tuple(current_group),
                ),
            ]

    return segments
