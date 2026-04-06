"""Hybrid search logic: vector similarity + keyword matching.

Combines semantic vector search with traditional keyword matching
to provide high-quality search results ranked by combined score.
Results are grouped at file level with segment timestamps.
"""

import logging
import re
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

    with get_search_db() as session:
        embeddings = (
            session.query(Embedding)
            .filter(Embedding.id.in_(embedding_ids))
            .all()
        )

        return [
            _VectorMatch(
                embedding_id=emb.id,
                file_id=emb.file_id,
                score=1.0 - distances.get(emb.id, 1.0),  # Convert distance to similarity
                embedding_type=emb.embedding_type,
                content_preview=emb.content_preview,
                timestamp_start=emb.timestamp_start,
                timestamp_end=emb.timestamp_end,
            )
            for emb in embeddings
        ]


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

    with get_search_db() as session:
        embeddings = (
            session.query(Embedding)
            .filter(Embedding.id.in_(embedding_ids))
            .all()
        )

        return [
            _VectorMatch(
                embedding_id=emb.id,
                file_id=emb.file_id,
                score=1.0 - distances.get(emb.id, 1.0),
                embedding_type=emb.embedding_type,
                content_preview=emb.content_preview,
                timestamp_start=emb.timestamp_start,
                timestamp_end=emb.timestamp_end,
            )
            for emb in embeddings
        ]


@dataclass
class _KeywordMatch:
    """Internal keyword search match."""

    file_id: str
    score: float
    matched_field: str


def _keyword_search(query: str, limit: int) -> list[_KeywordMatch]:
    """Search indexed files by keyword matching.

    Performs case-insensitive substring matching against
    filename, title, description, and tags.

    Args:
        query: The search query.
        limit: Maximum results.

    Returns:
        List of keyword matches.
    """
    query_lower = query.lower()
    query_terms = [t.strip() for t in query_lower.split() if t.strip()]

    if not query_terms:
        return []

    with get_search_db() as session:
        files = (
            session.query(IndexedFile)
            .filter(IndexedFile.active.is_(True))
            .all()
        )

        matches: list[_KeywordMatch] = []

        for file in files:
            score, matched_field = _calculate_keyword_score(file, query_terms)
            if score > 0:
                matches = [*matches, _KeywordMatch(
                    file_id=file.file_id,
                    score=score,
                    matched_field=matched_field,
                )]

        # Sort by score descending, take top N
        matches = sorted(matches, key=lambda m: m.score, reverse=True)
        return matches[:limit]


def _calculate_keyword_score(
    file: IndexedFile, terms: list[str]
) -> tuple[float, str]:
    """Calculate keyword match score for a file.

    Args:
        file: The indexed file.
        terms: Lowercased search terms.

    Returns:
        Tuple of (score, best_matched_field).
    """
    fields = {
        "filename": (file.filename or "").lower(),
        "title": (file.title or "").lower(),
        "description": (file.description or "").lower(),
        "tags": (file.tags_text or "").lower(),
    }

    # Weight by field importance
    weights = {
        "filename": 1.0,
        "title": 0.9,
        "tags": 0.8,
        "description": 0.6,
    }

    best_score = 0.0
    best_field = ""

    for field_name, field_text in fields.items():
        if not field_text:
            continue

        matched_terms = sum(1 for t in terms if t in field_text)
        if matched_terms == 0:
            continue

        # Score based on percentage of terms matched, weighted by field
        field_score = (matched_terms / len(terms)) * weights[field_name]

        # Bonus for exact match
        if all(t in field_text for t in terms):
            field_score *= 1.2

        if field_score > best_score:
            best_score = field_score
            best_field = field_name

    return min(best_score, 1.0), best_field


@dataclass
class _FileScore:
    """Aggregated score for a file across all match sources."""

    file_id: str
    combined_score: float
    matches: list[MatchInfo] = field(default_factory=list)
    match_types: set[str] = field(default_factory=set)


def _combine_scores(
    text_matches: list[_VectorMatch],
    clip_matches: list[_VectorMatch],
    keyword_matches: list[_KeywordMatch],
    alpha: float,
) -> dict[str, _FileScore]:
    """Combine vector and keyword scores into file-level rankings.

    Args:
        text_matches: Text vector search results.
        clip_matches: CLIP vector search results.
        keyword_matches: Keyword search results.
        alpha: Weight for vector similarity (0-1).

    Returns:
        Dict mapping file_id to aggregated scores.
    """
    file_scores: dict[str, _FileScore] = {}

    # Process text vector matches
    for match in text_matches:
        fs = file_scores.setdefault(
            match.file_id,
            _FileScore(file_id=match.file_id, combined_score=0.0),
        )
        weighted_score = match.score * alpha
        fs.combined_score = max(fs.combined_score, weighted_score)
        fs.match_types.add(match.embedding_type)
        fs.matches.append(MatchInfo(
            match_type=match.embedding_type,
            text=match.content_preview,
            score=match.score,
            timestamp_start=match.timestamp_start,
            timestamp_end=match.timestamp_end,
        ))

    # Process CLIP vector matches
    for match in clip_matches:
        fs = file_scores.setdefault(
            match.file_id,
            _FileScore(file_id=match.file_id, combined_score=0.0),
        )
        weighted_score = match.score * alpha
        fs.combined_score = max(fs.combined_score, weighted_score)
        fs.match_types.add("clip")
        fs.matches.append(MatchInfo(
            match_type="clip",
            text=match.content_preview,
            score=match.score,
            timestamp_start=match.timestamp_start,
            timestamp_end=match.timestamp_end,
        ))

    # Process keyword matches
    for match in keyword_matches:
        fs = file_scores.setdefault(
            match.file_id,
            _FileScore(file_id=match.file_id, combined_score=0.0),
        )
        keyword_contribution = match.score * (1.0 - alpha)
        fs.combined_score = max(
            fs.combined_score,
            fs.combined_score + keyword_contribution,
        )
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
