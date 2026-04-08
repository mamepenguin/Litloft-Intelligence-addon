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
from app.models import Embedding, IndexedFile, TranscriptChunk
from app.workers.clip import embed_text_clip
from app.workers.embedder import embed_query

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
    page: int | None = None


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
    combines with keyword matching using weighted cosine similarity,
    and returns file-level results.

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
    candidates = search_config.rrf_candidates

    # Generate query embeddings
    text_vector = embed_query(query)
    clip_vector = _safe_clip_embed(query)

    # Five retrieval systems, each returning top-N candidates
    text_matches = _vector_search_text(text_vector, candidates)
    clip_matches = _vector_search_clip(clip_vector, candidates) if clip_vector is not None else []
    keyword_matches = _keyword_search(query, candidates)
    transcript_keyword_matches = _keyword_search_transcripts(query, candidates)
    text_content_keyword_matches = _keyword_search_text_content(query, candidates)

    # Combine via weighted cosine similarity
    file_scores = _combine_scores_cosine(
        text_matches=text_matches,
        clip_matches=clip_matches,
        keyword_matches=keyword_matches,
        transcript_keyword_matches=transcript_keyword_matches,
        text_content_keyword_matches=text_content_keyword_matches,
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

    Applies three filtering stages:
    1. Absolute threshold (min_score_text): discard low-similarity matches
    2. Score gap analysis: if scores are flat (no standout), discard all
    3. Margin cutoff: keep only matches within margin of top score

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

    # Convert all distances to cosine similarities for gap analysis
    all_scores = [_l2_to_cosine_similarity(row[1]) for row in rows]

    search_config = settings.search
    min_score = search_config.min_score_text

    # Score gap analysis: check if results are "flat" (no standout match)
    # If top score barely exceeds the mean, everything is equally irrelevant
    if all_scores:
        top_score = all_scores[0]  # already sorted by distance
        mean_score = sum(all_scores) / len(all_scores)
        gap = top_score - mean_score

        if gap < search_config.score_gap_threshold:
            logger.debug(
                "Text vector score gap too small (%.4f < %.4f), "
                "discarding all %d candidates",
                gap, search_config.score_gap_threshold, len(all_scores),
            )
            return []

    with get_search_db() as session:
        embeddings = (
            session.query(Embedding)
            .filter(Embedding.id.in_(embedding_ids))
            .all()
        )

        # Stage 1: absolute threshold filter
        candidates: list[_VectorMatch] = []
        for emb in embeddings:
            score = _l2_to_cosine_similarity(distances.get(emb.id, 2.0))
            if score < min_score:
                continue
            candidates = [
                *candidates,
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

        if not candidates:
            return []

        # Stage 3: margin cutoff — keep only within margin of top score
        best_score = max(c.score for c in candidates)
        margin = search_config.score_cutoff_margin
        results = [c for c in candidates if c.score >= best_score - margin]

        return results


def _vector_search_clip(
    query_vector: np.ndarray, limit: int
) -> list[_VectorMatch]:
    """Search the CLIP vector table for similar embeddings.

    Applies the same filtering stages as text vector search:
    absolute threshold, score gap analysis, and margin cutoff.

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

    # Score gap analysis for CLIP
    all_scores = [_l2_to_cosine_similarity(row[1]) for row in rows]
    search_config = settings.search

    if all_scores:
        top_score = all_scores[0]
        mean_score = sum(all_scores) / len(all_scores)
        gap = top_score - mean_score

        if gap < search_config.score_gap_threshold:
            logger.debug(
                "CLIP score gap too small (%.4f < %.4f), "
                "discarding all %d candidates",
                gap, search_config.score_gap_threshold, len(all_scores),
            )
            return []

    min_score = search_config.min_score_clip

    with get_search_db() as session:
        embeddings = (
            session.query(Embedding)
            .filter(Embedding.id.in_(embedding_ids))
            .all()
        )

        candidates: list[_VectorMatch] = []
        for emb in embeddings:
            score = _l2_to_cosine_similarity(distances.get(emb.id, 2.0))
            if score < min_score:
                continue
            candidates = [
                *candidates,
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

        if not candidates:
            return []

        # Margin cutoff
        best_score = max(c.score for c in candidates)
        margin = search_config.score_cutoff_margin
        results = [c for c in candidates if c.score >= best_score - margin]

        return results


@dataclass
class _KeywordMatch:
    """Internal keyword search match."""

    file_id: str
    score: float
    matched_field: str


@dataclass(frozen=True)
class _TranscriptKeywordMatch:
    """Internal transcript keyword search match."""

    file_id: str
    score: float
    text: str
    timestamp_start: float | None
    timestamp_end: float | None


@dataclass(frozen=True)
class _TextContentKeywordMatch:
    """Internal text content keyword search match."""

    file_id: str
    score: float
    text: str
    page: int | None


def _keyword_search_transcripts(
    query: str, limit: int
) -> list[_TranscriptKeywordMatch]:
    """Search transcript chunks using FTS5 trigram index.

    Returns chunk-level matches with timestamps for segment grouping.

    Args:
        query: The search query.
        limit: Maximum results.

    Returns:
        List of transcript keyword matches with timestamps.
    """
    fts_query = _build_fts_query(query)
    if not fts_query:
        return []

    engine = get_search_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            sql_text(
                "SELECT file_id, chunk_index, text, -rank AS relevance "
                "FROM fts_transcripts "
                "WHERE fts_transcripts MATCH :query "
                "ORDER BY rank "
                "LIMIT :limit"
            ),
            {"query": fts_query, "limit": limit},
        ).fetchall()

    if not rows:
        return []

    # Look up timestamps from transcript_chunks table
    file_chunk_pairs = [(row[0], int(row[1])) for row in rows]
    file_ids = {pair[0] for pair in file_chunk_pairs}

    with get_search_db() as session:
        # Verify files are active
        active_ids = {
            f.file_id
            for f in session.query(IndexedFile.file_id)
            .filter(
                IndexedFile.file_id.in_(file_ids),
                IndexedFile.active.is_(True),
            )
            .all()
        }

        # Get timestamps for matched chunks
        chunks_by_key: dict[tuple[str, int], TranscriptChunk] = {}
        if file_chunk_pairs:
            all_chunks = (
                session.query(TranscriptChunk)
                .filter(TranscriptChunk.file_id.in_(file_ids))
                .all()
            )
            for chunk in all_chunks:
                chunks_by_key[(chunk.file_id, chunk.chunk_index)] = chunk

    # Normalize scores
    max_relevance = max(row[3] for row in rows) if rows else 1.0
    normalizer = max_relevance if max_relevance > 0 else 1.0

    results: list[_TranscriptKeywordMatch] = []
    for row in rows:
        fid, chunk_idx, matched_text, relevance = row[0], int(row[1]), row[2], row[3]
        if fid not in active_ids:
            continue

        chunk = chunks_by_key.get((fid, chunk_idx))
        results = [
            *results,
            _TranscriptKeywordMatch(
                file_id=fid,
                score=min(relevance / normalizer, 1.0),
                text=matched_text[:200],
                timestamp_start=chunk.timestamp_start if chunk else None,
                timestamp_end=chunk.timestamp_end if chunk else None,
            ),
        ]

    return results


def _to_katakana(text: str) -> str:
    """Convert hiragana characters to katakana."""
    return "".join(
        chr(ord(c) + 0x60) if "\u3041" <= c <= "\u3096" else c
        for c in text
    )


def _to_hiragana(text: str) -> str:
    """Convert katakana characters to hiragana."""
    return "".join(
        chr(ord(c) - 0x60) if "\u30A1" <= c <= "\u30F6" else c
        for c in text
    )


def _build_fts_query(query: str) -> str:
    """Build an FTS5 trigram query string with kana normalization.

    Each term is expanded to match both hiragana and katakana variants.
    For example, "みかん先輩" matches both "みかん先輩" and "ミカン先輩".

    Args:
        query: Raw search query.

    Returns:
        FTS5 query string safe for the trigram tokenizer.
    """
    terms = [t.strip() for t in query.split() if t.strip()]
    if not terms:
        return ""

    term_clauses: list[str] = []
    for term in terms:
        sanitized = term.replace(chr(34), "")
        if not sanitized:
            continue

        katakana = _to_katakana(sanitized)
        hiragana = _to_hiragana(sanitized)

        # Collect unique variants (original, katakana, hiragana)
        variants = list(dict.fromkeys([sanitized, katakana, hiragana]))

        if len(variants) == 1:
            term_clauses = [*term_clauses, f'"{variants[0]}"']
        else:
            # OR across kana variants for this term
            or_clause = " OR ".join(f'"{v}"' for v in variants)
            term_clauses = [*term_clauses, f"({or_clause})"]

    if not term_clauses:
        return ""
    return " AND ".join(term_clauses)


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


def _keyword_search_text_content(
    query: str, limit: int
) -> list[_TextContentKeywordMatch]:
    """Search text content chunks using FTS5 trigram index.

    Returns chunk-level matches with page numbers for PDF and similar docs.

    Args:
        query: The search query.
        limit: Maximum results.

    Returns:
        List of text content keyword matches with page info.
    """
    fts_query = _build_fts_query(query)
    if not fts_query:
        return []

    engine = get_search_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            sql_text(
                "SELECT file_id, chunk_index, page, text, -rank AS relevance "
                "FROM fts_text_content "
                "WHERE fts_text_content MATCH :query "
                "ORDER BY rank "
                "LIMIT :limit"
            ),
            {"query": fts_query, "limit": limit},
        ).fetchall()

    if not rows:
        return []

    # Verify files are active
    file_ids = {row[0] for row in rows}
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

    # Normalize scores
    max_relevance = max(row[4] for row in rows) if rows else 1.0
    normalizer = max_relevance if max_relevance > 0 else 1.0

    results: list[_TextContentKeywordMatch] = []
    for row in rows:
        fid = row[0]
        if fid not in active_ids:
            continue

        page_str = row[2]
        page_num = int(page_str) if page_str and page_str.isdigit() else None

        results = [
            *results,
            _TextContentKeywordMatch(
                file_id=fid,
                score=min(row[4] / normalizer, 1.0),
                text=row[3][:200],
                page=page_num,
            ),
        ]

    return results


@dataclass
class _FileScore:
    """Aggregated score for a file across all match sources."""

    file_id: str
    combined_score: float
    matches: list[MatchInfo] = field(default_factory=list)
    match_types: set[str] = field(default_factory=set)


def _file_ranking_from_vector(
    matches: list[_VectorMatch],
) -> tuple[dict[str, int], dict[str, float]]:
    """Aggregate vector matches to file level and produce a ranking.

    Takes the best score per file, then ranks files by that best score.

    Args:
        matches: Embedding-level vector matches.

    Returns:
        Tuple of (file_id → 0-based rank, file_id → best score).
    """
    file_best: dict[str, float] = {}
    for m in matches:
        file_best[m.file_id] = max(file_best.get(m.file_id, 0.0), m.score)

    sorted_files = sorted(file_best.items(), key=lambda x: x[1], reverse=True)
    ranking = {fid: rank for rank, (fid, _) in enumerate(sorted_files)}
    return ranking, file_best


def _file_ranking_from_keywords(
    matches: (
        list[_KeywordMatch]
        | list[_TranscriptKeywordMatch]
        | list[_TextContentKeywordMatch]
    ),
) -> dict[str, int]:
    """Produce a file-level ranking from keyword matches.

    For transcript/text-content keywords, multiple chunks per file are
    collapsed to the best score before ranking.

    Args:
        matches: Keyword match list (already ordered by relevance).

    Returns:
        Dict mapping file_id to 0-based rank.
    """
    file_best: dict[str, float] = {}
    for m in matches:
        file_best[m.file_id] = max(file_best.get(m.file_id, 0.0), m.score)

    sorted_files = sorted(file_best.items(), key=lambda x: x[1], reverse=True)
    return {fid: rank for rank, (fid, _) in enumerate(sorted_files)}


def _combine_scores_rrf(
    text_matches: list[_VectorMatch],
    clip_matches: list[_VectorMatch],
    keyword_matches: list[_KeywordMatch],
    transcript_keyword_matches: list[_TranscriptKeywordMatch],
    text_content_keyword_matches: list[_TextContentKeywordMatch] | None = None,
    *,
    k: int,
) -> dict[str, _FileScore]:
    """Combine search results using Reciprocal Rank Fusion (RRF).

    Each retrieval system produces a file-level ranking. RRF merges them:
        score = Σ 1/(k + rank + 1)
    Files appearing in multiple systems are naturally boosted.

    Args:
        text_matches: Text vector search results.
        clip_matches: CLIP vector search results.
        keyword_matches: Keyword search results (metadata fields).
        transcript_keyword_matches: Keyword search results (transcript text).
        text_content_keyword_matches: Keyword search results (text content).
        k: RRF smoothing constant (typically 60).

    Returns:
        Dict mapping file_id to aggregated scores.
    """
    # --- Collect match info for display (separate from scoring) ---
    file_matches: dict[str, list[MatchInfo]] = {}
    file_match_types: dict[str, set[str]] = {}

    for m in text_matches:
        file_match_types.setdefault(m.file_id, set()).add(m.embedding_type)
        file_matches.setdefault(m.file_id, []).append(MatchInfo(
            match_type=m.embedding_type,
            text=m.content_preview,
            score=m.score,
            timestamp_start=m.timestamp_start,
            timestamp_end=m.timestamp_end,
        ))

    for m in clip_matches:
        file_match_types.setdefault(m.file_id, set()).add("clip")
        file_matches.setdefault(m.file_id, []).append(MatchInfo(
            match_type="clip",
            text=m.content_preview,
            score=m.score,
            timestamp_start=m.timestamp_start,
            timestamp_end=m.timestamp_end,
        ))

    for m in keyword_matches:
        file_match_types.setdefault(m.file_id, set()).add("keyword")

    for m in transcript_keyword_matches:
        file_match_types.setdefault(m.file_id, set()).add("transcript_keyword")
        file_matches.setdefault(m.file_id, []).append(MatchInfo(
            match_type="transcript",
            text=m.text,
            score=m.score,
            timestamp_start=m.timestamp_start,
            timestamp_end=m.timestamp_end,
        ))

    for m in (text_content_keyword_matches or []):
        file_match_types.setdefault(m.file_id, set()).add("text_content_keyword")
        file_matches.setdefault(m.file_id, []).append(MatchInfo(
            match_type="text_content",
            text=m.text,
            score=m.score,
            page=m.page,
        ))

    # --- Build per-system file rankings ---
    text_ranking, _ = _file_ranking_from_vector(text_matches)
    clip_ranking, _ = _file_ranking_from_vector(clip_matches)
    keyword_ranking = _file_ranking_from_keywords(keyword_matches)
    transcript_kw_ranking = _file_ranking_from_keywords(transcript_keyword_matches)
    text_content_kw_ranking = _file_ranking_from_keywords(
        text_content_keyword_matches or []
    )

    # Weighted RRF: CLIP gets reduced weight (0.3) because cross-modal
    # text→image similarity is noisy for text queries. Other systems
    # use full weight (1.0). This lets CLIP still surface visual-only
    # matches while preventing it from overwhelming text-based results.
    weighted_rankings: list[tuple[dict[str, int], float]] = [
        (text_ranking, 1.0),
        (clip_ranking, settings.search.rrf_weight_clip),
        (keyword_ranking, 1.0),
        (transcript_kw_ranking, 1.0),
        (text_content_kw_ranking, 1.0),
    ]

    # --- RRF fusion ---
    all_file_ids: set[str] = set()
    for ranking, _ in weighted_rankings:
        all_file_ids.update(ranking.keys())

    file_scores: dict[str, _FileScore] = {}

    for fid in all_file_ids:
        rrf_score = 0.0
        for ranking, weight in weighted_rankings:
            if fid in ranking:
                rrf_score = rrf_score + weight / (k + ranking[fid] + 1)

        file_scores[fid] = _FileScore(
            file_id=fid,
            combined_score=rrf_score,
            matches=file_matches.get(fid, []),
            match_types=file_match_types.get(fid, set()),
        )

    return file_scores


def _build_results(
    file_scores: dict[str, _FileScore],
    file_type: str | None,
    drive: str | None,
    limit: int,
    *,
    skip_cutoff: bool = False,
) -> list[SearchResult]:
    """Build final search results from aggregated scores.

    Applies filters, groups matches into segments, and sorts by score.

    Args:
        file_scores: Aggregated file scores.
        file_type: Optional file type filter.
        drive: Optional drive filter.
        limit: Maximum results.
        skip_cutoff: If True, skip the dynamic score cutoff.

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

    # Dynamic cutoff: discard results far below the top score.
    # Ratio-based cutoff works well for RRF scores (which vary proportionally,
    # unlike cosine similarities that cluster in narrow absolute ranges).
    if results and not skip_cutoff:
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


# ---------------------------------------------------------------------------
# Cosine-similarity-based scoring (alternative to RRF)
# ---------------------------------------------------------------------------

_TYPE_WEIGHTS: dict[str, str] = {
    "metadata": "type_weight_metadata",
    "transcript": "type_weight_transcript",
    "text_content": "type_weight_text_content",
    "clip": "type_weight_clip",
}


def _combine_scores_cosine(
    text_matches: list[_VectorMatch],
    clip_matches: list[_VectorMatch],
    keyword_matches: list[_KeywordMatch],
    transcript_keyword_matches: list[_TranscriptKeywordMatch],
    text_content_keyword_matches: list[_TextContentKeywordMatch] | None = None,
) -> dict[str, _FileScore]:
    """Combine search results using weighted cosine similarity scores.

    Each file's score is the weighted maximum similarity across all sources.
    Uses config weights (alpha for vector vs keyword balance, type_weight_*
    for per-source weighting).
    """
    search_config = settings.search

    # Collect match info and per-source best scores
    file_matches: dict[str, list[MatchInfo]] = {}
    file_match_types: dict[str, set[str]] = {}
    # source → file_id → best score
    file_source_best: dict[str, dict[str, float]] = {
        "text_vector": {},
        "clip_vector": {},
        "keyword": {},
        "transcript_keyword": {},
        "text_content_keyword": {},
    }

    for m in text_matches:
        file_match_types.setdefault(m.file_id, set()).add(m.embedding_type)
        file_matches.setdefault(m.file_id, []).append(MatchInfo(
            match_type=m.embedding_type,
            text=m.content_preview,
            score=m.score,
            timestamp_start=m.timestamp_start,
            timestamp_end=m.timestamp_end,
        ))
        weight = getattr(search_config, _TYPE_WEIGHTS.get(m.embedding_type, ""), 1.0)
        weighted = m.score * weight
        src = file_source_best["text_vector"]
        src[m.file_id] = max(src.get(m.file_id, 0.0), weighted)

    for m in clip_matches:
        file_match_types.setdefault(m.file_id, set()).add("clip")
        file_matches.setdefault(m.file_id, []).append(MatchInfo(
            match_type="clip",
            text=m.content_preview,
            score=m.score,
            timestamp_start=m.timestamp_start,
            timestamp_end=m.timestamp_end,
        ))
        weighted = m.score * search_config.type_weight_clip
        src = file_source_best["clip_vector"]
        src[m.file_id] = max(src.get(m.file_id, 0.0), weighted)

    for m in keyword_matches:
        file_match_types.setdefault(m.file_id, set()).add("keyword")
        src = file_source_best["keyword"]
        src[m.file_id] = max(src.get(m.file_id, 0.0), m.score)

    for m in transcript_keyword_matches:
        file_match_types.setdefault(m.file_id, set()).add("transcript_keyword")
        file_matches.setdefault(m.file_id, []).append(MatchInfo(
            match_type="transcript",
            text=m.text,
            score=m.score,
            timestamp_start=m.timestamp_start,
            timestamp_end=m.timestamp_end,
        ))
        src = file_source_best["transcript_keyword"]
        src[m.file_id] = max(src.get(m.file_id, 0.0), m.score)

    for m in (text_content_keyword_matches or []):
        file_match_types.setdefault(m.file_id, set()).add("text_content_keyword")
        file_matches.setdefault(m.file_id, []).append(MatchInfo(
            match_type="text_content",
            text=m.text,
            score=m.score,
            page=m.page,
        ))
        src = file_source_best["text_content_keyword"]
        src[m.file_id] = max(src.get(m.file_id, 0.0), m.score)

    # Combine: alpha * best_vector + (1 - alpha) * best_keyword
    alpha = search_config.alpha
    all_file_ids: set[str] = set()
    for src_scores in file_source_best.values():
        all_file_ids.update(src_scores.keys())

    file_scores: dict[str, _FileScore] = {}
    for fid in all_file_ids:
        best_vector = max(
            file_source_best["text_vector"].get(fid, 0.0),
            file_source_best["clip_vector"].get(fid, 0.0),
        )
        best_keyword = max(
            file_source_best["keyword"].get(fid, 0.0),
            file_source_best["transcript_keyword"].get(fid, 0.0),
            file_source_best["text_content_keyword"].get(fid, 0.0),
        )
        combined = alpha * best_vector + (1.0 - alpha) * best_keyword
        file_scores[fid] = _FileScore(
            file_id=fid,
            combined_score=combined,
            matches=file_matches.get(fid, []),
            match_types=file_match_types.get(fid, set()),
        )

    return file_scores


@dataclass(frozen=True)
class SourceCounts:
    """Hit counts from each retrieval system (post-filter)."""

    text_vector: int
    clip_vector: int
    keyword: int
    transcript_keyword: int
    text_content_keyword: int = 0


@dataclass(frozen=True)
class CompareResponse:
    """Side-by-side comparison of RRF and cosine scoring."""

    rrf: SearchResponse
    cosine: SearchResponse
    rrf_no_cutoff: SearchResponse
    cosine_no_cutoff: SearchResponse
    source_counts: SourceCounts


def execute_search_compare(
    query: str,
    limit: int | None = None,
    file_type: str | None = None,
    drive: str | None = None,
) -> CompareResponse:
    """Execute search with both RRF and cosine scoring, for comparison."""
    search_config = settings.search
    effective_limit = min(
        limit or search_config.default_limit,
        search_config.max_limit,
    )
    candidates = search_config.rrf_candidates

    text_vector = embed_query(query)
    clip_vector = _safe_clip_embed(query)

    text_matches = _vector_search_text(text_vector, candidates)
    clip_matches = (
        _vector_search_clip(clip_vector, candidates)
        if clip_vector is not None
        else []
    )
    keyword_matches = _keyword_search(query, candidates)
    transcript_kw_matches = _keyword_search_transcripts(query, candidates)
    text_content_kw_matches = _keyword_search_text_content(query, candidates)

    source_counts = SourceCounts(
        text_vector=len(text_matches),
        clip_vector=len(clip_matches),
        keyword=len(keyword_matches),
        transcript_keyword=len(transcript_kw_matches),
        text_content_keyword=len(text_content_kw_matches),
    )

    # RRF scoring
    rrf_scores = _combine_scores_rrf(
        text_matches=text_matches,
        clip_matches=clip_matches,
        keyword_matches=keyword_matches,
        transcript_keyword_matches=transcript_kw_matches,
        text_content_keyword_matches=text_content_kw_matches,
        k=search_config.rrf_k,
    )
    rrf_results = _build_results(rrf_scores, file_type, drive, effective_limit)
    rrf_results_no_cutoff = _build_results(
        rrf_scores, file_type, drive, effective_limit, skip_cutoff=True,
    )

    # Cosine scoring
    cosine_scores = _combine_scores_cosine(
        text_matches=text_matches,
        clip_matches=clip_matches,
        keyword_matches=keyword_matches,
        transcript_keyword_matches=transcript_kw_matches,
        text_content_keyword_matches=text_content_kw_matches,
    )
    cosine_results = _build_results(cosine_scores, file_type, drive, effective_limit)
    cosine_results_no_cutoff = _build_results(
        cosine_scores, file_type, drive, effective_limit, skip_cutoff=True,
    )

    with get_search_db() as session:
        indexed_count = session.query(IndexedFile).filter(
            IndexedFile.active.is_(True)
        ).count()

    def _make_response(results: list[SearchResult]) -> SearchResponse:
        return SearchResponse(
            results=tuple(results),
            total=len(results),
            indexed_files=indexed_count,
            service_version=settings.service_version,
        )

    return CompareResponse(
        rrf=_make_response(rrf_results),
        cosine=_make_response(cosine_results),
        rrf_no_cutoff=_make_response(rrf_results_no_cutoff),
        cosine_no_cutoff=_make_response(cosine_results_no_cutoff),
        source_counts=source_counts,
    )
