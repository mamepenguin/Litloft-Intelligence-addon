"""Hybrid search logic: vector similarity + keyword matching.

Combines semantic vector search with traditional keyword matching
to provide high-quality search results ranked by combined score.
Results are grouped at file level with segment timestamps.
"""

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from sqlalchemy import text as sql_text

from app.config import settings
from concurrent.futures import ThreadPoolExecutor

from app.database import get_search_db, get_search_db_read, get_search_engine, validate_vector_table
from app.models import Embedding, IndexedFile, TranscriptChunk
from app.workers.clip import embed_text_clip
from app.workers.embedder import embed_query

if TYPE_CHECKING:
    from app.rag.query_transform import RequiredTerm

logger = logging.getLogger(__name__)

# Time window for grouping co-located matches (seconds)
SEGMENT_GROUP_WINDOW = 30

# Search mode:
# - "precision": tuned for human-facing search UI. Uses cosine combiner,
#   strict thresholds, narrow cutoffs. Default.
# - "recall": tuned for RAG / Ask pipeline. LLM can tolerate noise, so
#   we widen thresholds, relax cutoffs, rebalance channel weights toward
#   transcript/text-content (LLM-quotable sources) and downweight CLIP
#   (since the LLM only sees BLIP captions, not raw image bytes).
#   Uses the weighted RRF combiner.
SearchMode = Literal["precision", "recall"]


@dataclass(frozen=True)
class _RecallParams:
    """Threshold + weight overrides for recall (RAG) mode.

    Kept as a frozen dataclass so the recall knobs live in one place.
    Not exposed to config: these values are empirically chosen for the
    LLM's tolerance of noisy candidates and are not meant to be tuned
    by end users the way precision-mode thresholds are.
    """

    # Vector thresholds — looser than precision defaults so borderline
    # true-positives are not dropped at the channel level before RRF
    # has a chance to reinforce them across channels.
    min_score_text: float = 0.75
    score_gap_threshold: float = 0.0  # disabled — flat-set rejection off
    score_cutoff_margin: float = 0.15
    # Combined-score ratio cutoff — applied after RRF fusion.
    # Much more permissive than the precision default so the top_k=5
    # window keeps borderline candidates the LLM might cite.
    score_cutoff_ratio: float = 0.2
    # RRF channel weights. transcript/text_content upweighted because
    # they produce direct quotations the LLM can cite. CLIP downweighted
    # because the LLM receives BLIP captions at best (never raw pixels).
    rrf_weight_text: float = 1.0
    rrf_weight_keyword: float = 1.0
    rrf_weight_transcript_keyword: float = 1.5
    rrf_weight_text_content_keyword: float = 1.5
    rrf_weight_clip: float = 0.2
    # SIRA-style LLM-expanded retrieval keywords (fts_retrieval_keywords).
    # Slightly below other keyword channels because the hits are tier-3
    # (LLM-generated, not first-source) — they should reinforce real
    # signals rather than dominate them on their own. Operators who
    # observe the LLM producing low-quality expansions can dial this to
    # 0 to drop the channel without touching the on_index data.
    rrf_weight_retrieval_keywords: float = 0.8


_RECALL_PARAMS = _RecallParams()


def _recall_clip_enabled() -> bool:
    """Return True when CLIP ranking should participate in recall mode.

    In recall mode the CLIP channel is skipped entirely when BLIP is
    disabled, because without BLIP captions there is nothing readable
    the LLM can quote from an image match — the CLIP hit would just
    steal a top_k slot from a text candidate the LLM could actually use.
    """
    blip = getattr(settings.models, "blip", "") or ""
    return bool(blip.strip())


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
    *,
    mode: SearchMode = "precision",
    semantic_query: str | None = None,
    file_id_scope: list[str] | None = None,
    required: "tuple[RequiredTerm, ...] | None" = None,
    include_scene_clip: bool = False,
) -> SearchResponse:
    """Execute a hybrid search query.

    Performs vector similarity search across text and CLIP embeddings,
    combines with keyword matching, and returns file-level results.

    Args:
        query: The search query string.
        limit: Maximum number of results.
        file_type: Optional file type filter.
        drive: Optional drive name filter.
        mode: "precision" (default, human-facing UI) uses the weighted
            cosine combiner with strict thresholds; "recall" (RAG /
            Ask pipeline) uses RRF with loosened thresholds and
            rebalanced channel weights. See SearchMode docstring and
            the RAG redesign spec for rationale.
        file_id_scope: Optional allow-list of file_ids. When provided,
            results outside this set are dropped before ranking
            cutoffs are applied. Used by hierarchical RAG (Stage 2)
            to scope chunk-level retrieval to the Stage 1 shortlist.
            ``None`` disables the filter; an empty list short-circuits
            to zero results.
        required: Optional tuple of ``RequiredTerm`` from the structured
            query transform. Each term contributes an OR-of-aliases
            FTS clause; the running intersection across terms narrows
            the effective ``file_id_scope`` so every retrieval channel
            ranks only among files that pass the hard filter. ``None``
            (default) is the legacy behaviour. An empty tuple is also
            no-op so callers do not have to special-case the
            "transform succeeded but found no required terms" path.
        include_scene_clip: When ``False`` (default), CLIP retrieval
            is restricted to ``embedding_type="clip_thumbnail"`` (the
            "video about X" / 1-frame route, spec
            2026-05-02-thumbnail-clip-default-shallow-search.md). When
            ``True``, scene-frame ``embedding_type="clip"`` rows are
            unioned in — the explicit "scene with X" / "シーン検索"
            toggle. The toggle is meant for cases where the user
            specifically wants to find a moment inside a long video
            rather than a video whose subject matches the query.

    Returns:
        SearchResponse with ranked results.
    """
    search_config = settings.search
    effective_limit = min(
        limit or search_config.default_limit,
        search_config.max_limit,
    )
    candidates = search_config.rrf_candidates

    # Required-keyword hard filter (Phase 2 of the structured retriever).
    # Computed before any retrieval channel runs so the filter narrows
    # the ranking pool, not just the final ordering. ``None`` from the
    # filter means "no usable required terms" — leave file_id_scope
    # alone. An empty set means the hard filter dropped everything;
    # we feed [] into _build_results which short-circuits to no results
    # without firing vector / RRF work pointlessly.
    required_passing_ids: set[str] = set()
    if required:
        filtered = _required_keyword_filter(required)
        if filtered is not None:
            required_passing_ids = set(filtered)
            if file_id_scope is None:
                file_id_scope = list(required_passing_ids)
            else:
                file_id_scope = list(set(file_id_scope) & required_passing_ids)
            if not file_id_scope:
                # Hard filter dropped everything. Skip retrieval and
                # return an empty response immediately. The caller
                # (retriever) is responsible for triggering the Tier 3
                # fallback (re-search with required=None).
                with get_search_db_read() as session:
                    indexed_count = session.query(IndexedFile).filter(
                        IndexedFile.active.is_(True)
                    ).count()
                return SearchResponse(
                    results=(),
                    total=0,
                    indexed_files=indexed_count,
                    service_version=settings.service_version,
                )

    # Generate query embeddings. ``semantic_query`` is the natural-language
    # phrasing used for vector channels, which benefit from full context
    # (question form, particles, etc.). ``query`` is used unchanged for
    # keyword/FTS channels, which benefit from noise-free tokens. When the
    # caller does not pass ``semantic_query`` they get the legacy behaviour
    # of the same string driving both (suitable for direct UI input).
    effective_semantic = semantic_query if semantic_query is not None else query
    text_vector = embed_query(effective_semantic)
    clip_vector = _safe_clip_embed(effective_semantic)

    # Five retrieval systems, each returning top-N candidates.
    # Mode is threaded into the vector channels because their per-channel
    # thresholds differ between precision and recall. Keyword channels
    # are threshold-free so they do not need the mode argument.
    # In recall mode, skip CLIP retrieval entirely when BLIP is disabled —
    # without BLIP captions the LLM cannot read the image, so CLIP hits
    # are candidate-slot noise. See _recall_clip_enabled() docstring.
    skip_clip = mode == "recall" and not _recall_clip_enabled()

    # Six retrieval channels run concurrently. Each opens its own
    # get_search_db_read() session (no write lock) and its own
    # engine.connect() handle. WAL mode allows parallel readers.
    # The retrieval_keywords channel is intentionally additive: a fresh
    # DB without Phase 1 data simply returns [] and contributes nothing.
    with ThreadPoolExecutor(max_workers=6) as _pool:
        _f_text = _pool.submit(_vector_search_text, text_vector, candidates, mode=mode)
        _f_clip = (
            _pool.submit(
                _vector_search_clip, clip_vector, candidates,
                mode=mode, include_scene_clip=include_scene_clip,
            )
            if clip_vector is not None and not skip_clip
            else None
        )
        _f_kw = _pool.submit(_keyword_search, query, candidates)
        _f_tr = _pool.submit(_keyword_search_transcripts, query, candidates)
        _f_tc = _pool.submit(_keyword_search_text_content, query, candidates)
        _f_rk = _pool.submit(
            _keyword_search_retrieval_keywords, query, candidates
        )

        text_matches = _f_text.result()
        clip_matches = _f_clip.result() if _f_clip is not None else []
        keyword_matches = _f_kw.result()
        transcript_keyword_matches = _f_tr.result()
        text_content_keyword_matches = _f_tc.result()
        retrieval_keyword_matches = _f_rk.result()

    if mode == "recall":
        # Weighted RRF with recall-tuned channel weights. We drop the
        # CLIP channel weight to near-zero (or zero if BLIP disabled)
        # and upweight transcript/text_content because those are the
        # only channels the LLM can directly quote from.
        file_scores = _combine_scores_rrf(
            text_matches=text_matches,
            clip_matches=clip_matches,
            keyword_matches=keyword_matches,
            transcript_keyword_matches=transcript_keyword_matches,
            text_content_keyword_matches=text_content_keyword_matches,
            retrieval_keyword_matches=retrieval_keyword_matches,
            k=search_config.rrf_k,
            recall_params=_RECALL_PARAMS,
            include_clip=not skip_clip,
        )
    else:
        # Precision: existing weighted cosine combiner.
        file_scores = _combine_scores_cosine(
            text_matches=text_matches,
            clip_matches=clip_matches,
            keyword_matches=keyword_matches,
            transcript_keyword_matches=transcript_keyword_matches,
            text_content_keyword_matches=text_content_keyword_matches,
            retrieval_keyword_matches=retrieval_keyword_matches,
        )

    # Required-pass floor: any file that survived the hard filter but
    # did not surface in any other channel still needs to appear in
    # the result list — otherwise the AND-joined keyword path can drop
    # a Latin proper-noun-only file (e.g. "ViT" mentioned in the doc
    # but no other query term) even though it is the canonical answer
    # to the user's question. We synthesize a small floor score so
    # such files rank below genuine multi-channel hits but still
    # survive the dynamic cutoff downstream.
    if required_passing_ids:
        # The floor must clear the dynamic cutoff in ``_build_results``
        # (``top_score * cutoff_ratio``) — otherwise the entry is
        # added here and immediately discarded. A small buffer above
        # the cutoff guarantees survival without nudging the floor
        # high enough to outrank low-scoring real hits.
        if mode == "recall":
            cutoff_ratio = _RECALL_PARAMS.score_cutoff_ratio
        else:
            cutoff_ratio = settings.search.score_cutoff_ratio
        top_real_score = max(
            (fs.combined_score for fs in file_scores.values()),
            default=0.0,
        )
        # When no other channel produced anything, top_real_score is
        # 0; pick a small absolute fallback so the synthetic entries
        # at least make it into the result list.
        floor_buffer = 0.01  # 1% of top-or-1.0 buffer
        synthetic_floor = max(
            top_real_score * (cutoff_ratio + floor_buffer),
            0.001,
        )
        for fid in required_passing_ids:
            if fid in file_scores:
                continue
            file_scores[fid] = _FileScore(
                file_id=fid,
                combined_score=synthetic_floor,
                matches=[],
                match_types={"required"},
            )

    # Apply filters and build results
    results = _build_results(
        file_scores=file_scores,
        file_type=file_type,
        drive=drive,
        limit=effective_limit,
        mode=mode,
        file_id_scope=file_id_scope,
    )

    # Get total indexed count
    with get_search_db_read() as session:
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
    page: int | None = None


def _vector_search_text(
    query_vector: np.ndarray,
    limit: int,
    *,
    mode: SearchMode = "precision",
    include_video_visual_scene: bool = False,
) -> list[_VectorMatch]:
    """Search the text vector table for similar embeddings.

    Applies three filtering stages:
    1. Absolute threshold (min_score_text): discard low-similarity matches
    2. Score gap analysis: if scores are flat (no standout), discard all
    3. Margin cutoff: keep only matches within margin of top score

    In ``mode="recall"``, all three thresholds are loosened: the absolute
    threshold drops from 0.85 to 0.75, the gap check is disabled, and
    the margin cutoff widens from 0.05 to 0.15. This is to avoid
    dropping borderline true-positives before the RAG LLM sees them.

    Args:
        query_vector: Query embedding vector.
        limit: Maximum results.
        mode: precision (default) or recall.
        include_video_visual_scene: When ``False`` (default),
            ``embedding_type="video_visual_scene"`` rows are excluded
            from this channel (design doc "Video Visual Index" §8): a
            long video must not rank as being "about" an object merely
            because it appears in one incidental scene, so scene-level
            evidence is excluded from default file search / shallow Ask
            and reserved for explicit scene search / file-scoped
            retrieval (a future caller opts in here).

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
    if mode == "recall":
        min_score = _RECALL_PARAMS.min_score_text
        gap_threshold = _RECALL_PARAMS.score_gap_threshold
        margin = _RECALL_PARAMS.score_cutoff_margin
    else:
        min_score = search_config.min_score_text
        gap_threshold = search_config.score_gap_threshold
        margin = search_config.score_cutoff_margin

    # Score gap analysis: check if results are "flat" (no standout match)
    # If top score barely exceeds the mean, everything is equally irrelevant.
    # Recall mode disables this (gap_threshold=0) so borderline candidates
    # survive for the LLM to filter.
    if all_scores and gap_threshold > 0:
        top_score = all_scores[0]  # already sorted by distance
        mean_score = sum(all_scores) / len(all_scores)
        gap = top_score - mean_score

        if gap < gap_threshold:
            logger.debug(
                "Text vector score gap too small (%.4f < %.4f), "
                "discarding all %d candidates",
                gap, gap_threshold, len(all_scores),
            )
            return []

        # Coefficient of variation check: if scores are bunched in a
        # narrow band the embedding cannot meaningfully distinguish them.
        if len(all_scores) >= 5 and mean_score > 0:
            std = (sum((s - mean_score) ** 2 for s in all_scores) / len(all_scores)) ** 0.5
            cv = std / mean_score
            if cv < 0.01:
                logger.debug(
                    "Text vector CV too small (%.4f), "
                    "discarding all %d candidates",
                    cv, len(all_scores),
                )
                return []

    with get_search_db_read() as session:
        embeddings = (
            session.query(Embedding)
            .filter(Embedding.id.in_(embedding_ids))
            .all()
        )

        # Stage 1: type filter + absolute threshold filter
        candidates: list[_VectorMatch] = []
        for emb in embeddings:
            if (
                not include_video_visual_scene
                and emb.embedding_type == "video_visual_scene"
            ):
                continue
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
                    page=emb.page,
                ),
            ]

        if not candidates:
            return []

        # Stage 3: margin cutoff — keep only within margin of top score
        best_score = max(c.score for c in candidates)
        results = [c for c in candidates if c.score >= best_score - margin]

        return results


def _vector_search_clip(
    query_vector: np.ndarray,
    limit: int,
    *,
    mode: SearchMode = "precision",
    include_scene_clip: bool = False,
) -> list[_VectorMatch]:
    """Search the CLIP vector table for similar embeddings.

    Applies the same filtering stages as text vector search:
    absolute threshold, score gap analysis, and margin cutoff.

    In ``mode="recall"`` the ``min_score_clip`` absolute threshold is
    deliberately **not** loosened — the CLIP channel weight is already
    reduced in the RRF combiner, and keeping the entry threshold firm
    prevents low-quality image matches from crowding out text hits in
    the top_k window. Only the gap check (disabled) and the margin
    cutoff (widened) are relaxed.

    Args:
        query_vector: CLIP query embedding vector.
        limit: Maximum results.
        mode: precision (default) or recall.
        include_scene_clip: When ``False`` (default), only
            ``embedding_type="clip_thumbnail"`` rows are returned
            (the "video about X" / 1-frame route, spec
            2026-05-02-thumbnail-clip-default-shallow-search.md).
            When ``True``, scene-frame ``embedding_type="clip"`` rows
            are added — the "scene with X" / shipped-from-the-toggle
            route. Both types share ``vec_clip`` (same dim/model) so
            this is a post-MATCH filter on ``embeddings.embedding_type``.

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
    # min_score_clip is NOT loosened in recall mode (see docstring).
    # Only the gap check is disabled and the margin widened.
    if mode == "recall":
        gap_threshold = _RECALL_PARAMS.score_gap_threshold
        margin = _RECALL_PARAMS.score_cutoff_margin
    else:
        gap_threshold = search_config.score_gap_threshold
        margin = search_config.score_cutoff_margin

    if all_scores and gap_threshold > 0:
        top_score = all_scores[0]
        mean_score = sum(all_scores) / len(all_scores)
        gap = top_score - mean_score

        if gap < gap_threshold:
            logger.debug(
                "CLIP score gap too small (%.4f < %.4f), "
                "discarding all %d candidates",
                gap, gap_threshold, len(all_scores),
            )
            return []

        # Coefficient of variation check (same as text vector)
        if len(all_scores) >= 5 and mean_score > 0:
            std = (sum((s - mean_score) ** 2 for s in all_scores) / len(all_scores)) ** 0.5
            cv = std / mean_score
            if cv < 0.01:
                logger.debug(
                    "CLIP vector CV too small (%.4f), "
                    "discarding all %d candidates",
                    cv, len(all_scores),
                )
                return []

    # Default: ``clip_thumbnail`` only. Scene CLIP is opt-in via
    # ``include_scene_clip`` (search-modes "シーン検索" toggle, Ask
    # ``include_scenes``). Spec 2026-05-02-thumbnail-clip-default-
    # shallow-search.md.
    allowed_types = (
        ("clip_thumbnail", "clip") if include_scene_clip else ("clip_thumbnail",)
    )
    min_score_by_type = {
        "clip": search_config.min_score_clip,
        "clip_thumbnail": search_config.min_score_clip_thumbnail,
    }

    with get_search_db_read() as session:
        embeddings = (
            session.query(Embedding)
            .filter(
                Embedding.id.in_(embedding_ids),
                Embedding.embedding_type.in_(allowed_types),
            )
            .all()
        )

        candidates: list[_VectorMatch] = []
        for emb in embeddings:
            score = _l2_to_cosine_similarity(distances.get(emb.id, 2.0))
            min_score = min_score_by_type.get(
                emb.embedding_type,
                search_config.min_score_clip,
            )
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
                    page=emb.page,
                ),
            ]

        if not candidates:
            return []

        # Margin cutoff
        best_score = max(c.score for c in candidates)
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


@dataclass(frozen=True)
class _RetrievalKeywordMatch:
    """Internal retrieval-keywords search match (SIRA-style expansion).

    Carries the term that surfaced the hit so the UI can later annotate
    the result (Phase 2.5+); the time_range / page positional fields
    of the other keyword matches are intentionally absent because the
    LLM expansion does not point at a body location.
    """

    file_id: str
    score: float
    matched_keyword: str


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

    with get_search_db_read() as session:
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
    with get_search_db_read() as session:
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


def _keyword_search_retrieval_keywords(
    query: str, limit: int
) -> list[_RetrievalKeywordMatch]:
    """Search SIRA-style LLM-expanded retrieval keywords (fts_retrieval_keywords).

    Returns file-level matches with the matched keyword surfaced for
    UI annotation. Falls back to an empty list whenever the table is
    missing (legacy DBs that haven't run the Phase 1 migration) or
    when the query produces no FTS hits.

    Args:
        query: The raw user query — the same string used for the other
            keyword channels. ``_build_fts_query`` is applied for FTS5
            tokenisation so katakana/hiragana variants land alongside
            the literal form.
        limit: Maximum results returned to the caller (the combiner
            slices further if needed).

    Returns:
        Ranked list of ``_RetrievalKeywordMatch``. Scores are
        normalised 0-1 against the top hit, matching the convention
        of the other FTS channels so RRF / cosine combiners can mix
        them without extra rescaling.
    """
    fts_query = _build_fts_query(query)
    if not fts_query:
        return []

    engine = get_search_engine()

    with engine.connect() as conn:
        # Defensive: fts_retrieval_keywords lands in the Phase 1
        # migration. A search service started against an old DB before
        # _create_vec_tables has run would 500 here; treat the table
        # missing as "no expansions yet" instead.
        try:
            rows = conn.execute(
                sql_text(
                    "SELECT file_id, keywords, -rank AS relevance "
                    "FROM fts_retrieval_keywords "
                    "WHERE fts_retrieval_keywords MATCH :query "
                    "ORDER BY rank "
                    "LIMIT :limit"
                ),
                {"query": fts_query, "limit": limit},
            ).fetchall()
        except Exception:
            return []

    if not rows:
        return []

    # Verify files are active (matches the existing channel pattern).
    file_ids = [row[0] for row in rows]
    with get_search_db_read() as session:
        active_ids = {
            f.file_id
            for f in session.query(IndexedFile.file_id)
            .filter(
                IndexedFile.file_id.in_(file_ids),
                IndexedFile.active.is_(True),
            )
            .all()
        }

    max_relevance = max(row[2] for row in rows) if rows else 1.0
    normalizer = max_relevance if max_relevance > 0 else 1.0

    return [
        _RetrievalKeywordMatch(
            file_id=row[0],
            score=min(row[2] / normalizer, 1.0),
            # The whole keywords field is what the FTS matched against;
            # we surface it as-is so the UI can later show which term
            # surfaced this hit. Truncate to keep wire size predictable.
            matched_keyword=(row[1] or "")[:200],
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
    with get_search_db_read() as session:
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
        | list[_RetrievalKeywordMatch]
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
    retrieval_keyword_matches: list[_RetrievalKeywordMatch] | None = None,
    *,
    k: int,
    recall_params: _RecallParams | None = None,
    include_clip: bool = True,
) -> dict[str, _FileScore]:
    """Combine search results using Reciprocal Rank Fusion (RRF).

    Each retrieval system produces a file-level ranking. RRF merges them:
        score = Σ weight / (k + rank + 1)
    Files appearing in multiple systems are naturally boosted.

    Args:
        text_matches: Text vector search results.
        clip_matches: CLIP vector search results.
        keyword_matches: Keyword search results (metadata fields).
        transcript_keyword_matches: Keyword search results (transcript text).
        text_content_keyword_matches: Keyword search results (text content).
        k: RRF smoothing constant (typically 60).
        recall_params: Optional recall-mode channel weight overrides.
            When None, the legacy precision-mode RRF weights are used
            (all channels 1.0 except CLIP which uses
            ``settings.search.rrf_weight_clip``). When provided, the
            RAG-tuned weights from ``_RecallParams`` are applied.
        include_clip: When False, the CLIP channel is excluded entirely
            (weight set to 0). Used in recall mode when BLIP is disabled:
            without BLIP captions there is nothing the LLM can read from
            an image, so CLIP hits would only waste candidate slots.

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
            page=m.page,
        ))

    for m in clip_matches:
        match_label = m.embedding_type or "clip"
        file_match_types.setdefault(m.file_id, set()).add(match_label)
        file_matches.setdefault(m.file_id, []).append(MatchInfo(
            match_type=match_label,
            text=m.content_preview,
            score=m.score,
            timestamp_start=m.timestamp_start,
            timestamp_end=m.timestamp_end,
            page=m.page,
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

    # SIRA retrieval-keywords hits: contribute a chip-only entry. The
    # MatchInfo carries the matched keyword string as ``text`` so the
    # UI can surface it later, but no time_range / page is set — the
    # expansion does not point at a body location, so search_merge
    # treats it as a chip-only badge rather than a jump target.
    for m in (retrieval_keyword_matches or []):
        file_match_types.setdefault(m.file_id, set()).add("retrieval_keywords")
        file_matches.setdefault(m.file_id, []).append(MatchInfo(
            match_type="retrieval_keywords",
            text=m.matched_keyword,
            score=m.score,
        ))

    # --- Build per-system file rankings ---
    text_ranking, _ = _file_ranking_from_vector(text_matches)
    clip_ranking, _ = _file_ranking_from_vector(clip_matches)
    keyword_ranking = _file_ranking_from_keywords(keyword_matches)
    transcript_kw_ranking = _file_ranking_from_keywords(transcript_keyword_matches)
    text_content_kw_ranking = _file_ranking_from_keywords(
        text_content_keyword_matches or []
    )
    retrieval_kw_ranking = _file_ranking_from_keywords(
        retrieval_keyword_matches or []
    )

    # Weighted RRF. Precision mode uses the legacy weights (text/keyword
    # at 1.0, CLIP at the configured rrf_weight_clip default 0.5).
    # Recall mode uses the _RecallParams weights, which upweight
    # transcript/text_content to 1.5 (so the LLM gets quotable sources
    # ranked higher) and push CLIP down to 0.2 (BLIP captions are thin
    # information; raw pixels never reach the LLM). When include_clip
    # is False, the CLIP channel weight is forced to 0.
    if recall_params is not None:
        clip_weight = (
            recall_params.rrf_weight_clip if include_clip else 0.0
        )
        weighted_rankings: list[tuple[dict[str, int], float]] = [
            (text_ranking, recall_params.rrf_weight_text),
            (clip_ranking, clip_weight),
            (keyword_ranking, recall_params.rrf_weight_keyword),
            (transcript_kw_ranking, recall_params.rrf_weight_transcript_keyword),
            (text_content_kw_ranking, recall_params.rrf_weight_text_content_keyword),
            (retrieval_kw_ranking, recall_params.rrf_weight_retrieval_keywords),
        ]
    else:
        clip_weight = (
            settings.search.rrf_weight_clip if include_clip else 0.0
        )
        weighted_rankings = [
            (text_ranking, 1.0),
            (clip_ranking, clip_weight),
            (keyword_ranking, 1.0),
            (transcript_kw_ranking, 1.0),
            (text_content_kw_ranking, 1.0),
            # Precision mode mirrors the other keyword channels at
            # weight 1.0; the small-than-keyword bias lives in
            # _RecallParams.rrf_weight_retrieval_keywords (recall-mode
            # only) where the cost of an LLM-only hit is more visible.
            (retrieval_kw_ranking, 1.0),
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
    mode: SearchMode = "precision",
    file_id_scope: list[str] | None = None,
) -> list[SearchResult]:
    """Build final search results from aggregated scores.

    Applies filters, groups matches into segments, and sorts by score.

    Args:
        file_scores: Aggregated file scores.
        file_type: Optional file type filter.
        drive: Optional drive filter.
        limit: Maximum results.
        skip_cutoff: If True, skip the dynamic score cutoff.
        mode: precision (default) uses ``settings.search.score_cutoff_ratio``;
            recall uses the much more permissive ``_RECALL_PARAMS``
            ratio so borderline RAG candidates are not stripped before
            they reach the LLM.
        file_id_scope: Optional allow-list. ``None`` disables; ``[]``
            short-circuits to zero results. Filter runs BEFORE the
            cutoff so cutoff thresholds reflect the in-scope cohort.

    Returns:
        Sorted list of SearchResult objects.
    """
    if not file_scores:
        return []

    # Hierarchical RAG scope filter (Phase 2). Empty list short-circuits
    # to zero results — the caller asked for "nothing in scope" and we
    # honour that rather than silently falling back to unscoped search.
    if file_id_scope is not None:
        scope_set = set(file_id_scope)
        if not scope_set:
            return []
        file_scores = {
            fid: fs for fid, fs in file_scores.items() if fid in scope_set
        }
        if not file_scores:
            return []

    file_ids = list(file_scores.keys())

    with get_search_db_read() as session:
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
        if mode == "recall":
            cutoff_ratio = _RECALL_PARAMS.score_cutoff_ratio
        else:
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
# Similar files search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimilarFileResult:
    """A file similar to a given source file."""

    file_id: str
    drive: str
    filename: str
    file_type: str
    mime_type: str
    score: float
    match_type: str  # e.g. "clip", "tfidf", "clip+tfidf"
    primary_score: float | None = None
    secondary_score: float | None = None
    shared_keywords: tuple[dict, ...] = ()


@dataclass(frozen=True)
class SimilarSearchResult:
    """Complete result of a similar files search."""

    results: list[SimilarFileResult]
    source_keywords: tuple[dict, ...] = ()


# In-memory LRU cache for similar-files results.
# Previously persisted to a `similar_cache` table on disk, but virtiofs
# fsync semantics on Docker Desktop for Mac caused corruption under the
# wholesale-invalidate + per-request-rewrite churn pattern. Since these
# results are pure derivations of indexed data and cheap to rebuild,
# memory-only is the right durability tier.
_SIMILAR_CACHE_MAX = 2048
_similar_cache: "OrderedDict[str, SimilarSearchResult]" = OrderedDict()
_similar_cache_lock = threading.Lock()


def _build_similar_cache_key(
    file_id: str, limit: int, drive: str | None,
) -> str:
    """Build a cache key for similar files results."""
    return f"{file_id}:{limit}:{drive or '_'}"


def _similar_cache_get(key: str) -> "SimilarSearchResult | None":
    with _similar_cache_lock:
        result = _similar_cache.get(key)
        if result is not None:
            _similar_cache.move_to_end(key)
        return result


def _similar_cache_set(key: str, result: "SimilarSearchResult") -> None:
    with _similar_cache_lock:
        _similar_cache[key] = result
        _similar_cache.move_to_end(key)
        while len(_similar_cache) > _SIMILAR_CACHE_MAX:
            _similar_cache.popitem(last=False)


def invalidate_similar_cache() -> int:
    """Drop all similar files cache entries.

    Called on index updates (scan-complete, files-deleted, etc.).
    Returns number of evicted entries.
    """
    with _similar_cache_lock:
        count = len(_similar_cache)
        _similar_cache.clear()
    logger.info("Invalidated %d similar cache entries", count)
    return count


def find_similar(
    file_id: str,
    limit: int = 6,
    drive: str | None = None,
) -> SimilarSearchResult:
    """Find files similar to the given file using its existing embeddings.

    Selects the primary embedding type based on the file's type:
    - image/video: CLIP embeddings
    - audio: Whisper (text) embeddings
    - document/text: text_content embeddings
    - other: metadata embeddings

    For video files, TF-IDF on transcript text is used as the secondary
    signal instead of whisper embeddings (which lose topic information
    when averaged).

    Args:
        file_id: The source file ID.
        limit: Maximum number of similar files to return.
        drive: Optional drive filter (restricts results to this drive).

    Returns:
        List of similar files sorted by similarity score.
    """
    # Check cache first
    cache_key = _build_similar_cache_key(file_id, limit, drive)
    cached = _similar_cache_get(cache_key)
    if cached is not None:
        return cached

    with get_search_db_read() as session:
        source = (
            session.query(IndexedFile)
            .filter(
                IndexedFile.file_id == file_id,
                IndexedFile.active.is_(True),
            )
            .first()
        )
        if source is None:
            return SimilarSearchResult(results=[])

        file_type = source.file_type

    # Determine which embedding types to use based on file type
    primary_type, secondary_type = _select_embedding_types(file_type)

    primary_results = _find_similar_by_embedding(
        file_id, primary_type, limit * 2, drive,
    )

    # For video files, also search by pre-computed TF-IDF keyword embeddings
    secondary_results: list[dict] = []
    source_keywords: list[str] = []
    if secondary_type and secondary_type != primary_type:
        secondary_results = _find_similar_by_embedding(
            file_id, secondary_type, limit * 2, drive,
        )
        merged = _merge_similar_results(
            primary_results, secondary_results, limit,
        )
    else:
        merged = primary_results[:limit]

    # For tfidf_keywords secondary, read source keywords from the stored
    # content_preview (keyword string saved at index time) — O(1) DB lookup,
    # no Janome or IDF rebuild needed at query time.
    if secondary_type == "tfidf_keywords" and not source_keywords:
        with get_search_db_read() as _kw_session:
            _kw_emb = (
                _kw_session.query(Embedding.content_preview)
                .filter_by(file_id=file_id, embedding_type="tfidf_keywords")
                .first()
            )
        if _kw_emb and _kw_emb.content_preview:
            _words = _kw_emb.content_preview.split()
            source_keywords = [{"word": w, "score": 1.0} for w in _words]

    # Build lookup maps for score breakdown and keywords
    primary_by_id = {r["file_id"]: r["score"] for r in primary_results}
    secondary_by_id = {r["file_id"]: r for r in secondary_results}

    result = SimilarSearchResult(
        results=[
            SimilarFileResult(
                file_id=r["file_id"],
                drive=r["drive"],
                filename=r["filename"],
                file_type=r["file_type"],
                mime_type=r["mime_type"],
                score=r["score"],
                match_type=(
                    f"{primary_type}+{secondary_type}"
                    if r["file_id"] in primary_by_id and r["file_id"] in secondary_by_id
                    else primary_type if r["file_id"] in primary_by_id
                    else secondary_type or primary_type
                ),
                primary_score=primary_by_id.get(r["file_id"]),
                secondary_score=(
                    secondary_by_id[r["file_id"]]["score"]
                    if r["file_id"] in secondary_by_id
                    else None
                ),
                shared_keywords=tuple(
                    secondary_by_id[r["file_id"]].get("shared_keywords", [])
                    if r["file_id"] in secondary_by_id
                    else []
                ),
            )
            for r in merged
        ],
        source_keywords=tuple(source_keywords),
    )

    _similar_cache_set(cache_key, result)
    return result


def _merge_similar_results(
    primary: list[dict],
    secondary: list[dict],
    limit: int,
) -> list[dict]:
    """Merge results from primary and secondary sources.

    Both signals are normalized to [0, 1] and combined with equal
    weight.  This treats visual similarity (CLIP) and topic similarity
    (TF-IDF) as equally valuable signals:

      both:           0.5 * norm_primary + 0.5 * norm_secondary
      primary only:   norm_primary * single_signal_weight
      secondary only: norm_secondary * single_signal_weight

    Args:
        primary: Results from primary source (e.g. CLIP).
        secondary: Results from secondary source (e.g. TF-IDF).
        limit: Maximum results to return.

    Returns:
        Merged and sorted list of results.
    """
    primary_by_id = {r["file_id"]: r for r in primary}
    secondary_by_id = {r["file_id"]: r for r in secondary}

    # Normalize primary scores to [0, 1]
    if primary:
        pri_scores = [r["score"] for r in primary]
        pri_max = max(pri_scores)
        pri_min = min(pri_scores)
        pri_range = pri_max - pri_min
    else:
        pri_range = 0.0
        pri_max = 0.0
        pri_min = 0.0

    # Normalize secondary scores to [0, 1]
    if secondary:
        sec_scores = [r["score"] for r in secondary]
        sec_max = max(sec_scores)
        sec_min = min(sec_scores)
        sec_range = sec_max - sec_min
    else:
        sec_range = 0.0
        sec_max = 0.0
        sec_min = 0.0

    # Weight for results that have only one signal (mild penalty)
    single_signal_weight = 0.7

    merged: list[dict] = []
    all_file_ids = set(primary_by_id) | set(secondary_by_id)

    for fid in all_file_ids:
        p = primary_by_id.get(fid)
        s = secondary_by_id.get(fid)

        norm_pri = (
            (p["score"] - pri_min) / pri_range
            if p and pri_range > 0
            else 1.0 if p else None
        )
        norm_sec = (
            (s["score"] - sec_min) / sec_range
            if s and sec_range > 0
            else 1.0 if s else None
        )

        if norm_pri is not None and norm_sec is not None:
            score = 0.5 * norm_pri + 0.5 * norm_sec
        elif norm_pri is not None:
            score = norm_pri * single_signal_weight
        else:
            score = norm_sec * single_signal_weight

        # Use whichever source dict has the file metadata
        base = p if p else s
        merged.append({**base, "score": score})

    merged.sort(key=lambda r: r["score"], reverse=True)
    return merged[:limit]


def _select_embedding_types(
    file_type: str,
) -> tuple[str, str | None]:
    """Select primary and fallback embedding types based on file type.

    No metadata fallback: filename-based embeddings produce false positives
    for files with non-descriptive names (e.g. "IMG_1234.jpg", UUIDs).

    Returns:
        Tuple of (primary_type, fallback_type). fallback_type may be None.
    """
    # Spec 2026-05-02-thumbnail-clip-default-shallow-search.md:
    # the "find similar files" action uses the 1-frame route for
    # both images and videos (visual similarity of the gestalt /
    # main subject), not scene CLIP — which would surface "videos
    # that contain a similar scene" rather than "videos about the
    # same subject".
    type_map: dict[str, tuple[str, str | None]] = {
        "image": ("clip_thumbnail", None),
        "video": ("clip_thumbnail", "tfidf_keywords"),
        "audio": ("whisper", None),
        "document": ("text_content", None),
    }
    return type_map.get(file_type, ("metadata", None))


def _find_similar_by_embedding(
    file_id: str,
    embedding_type: str,
    limit: int,
    drive: str | None,
) -> list[dict]:
    """Find similar files using a specific embedding type.

    Averages all embeddings of the given type for the source file,
    then queries the appropriate vector table for nearest neighbors.

    Args:
        file_id: Source file ID.
        embedding_type: The embedding type to use.
        limit: Max results.
        drive: Optional drive filter.

    Returns:
        List of dicts with file info and similarity score.
    """
    # Determine which vector table to query. ``clip`` and
    # ``clip_thumbnail`` share ``vec_clip`` (same dim/model); other
    # types live in ``vec_text``.
    vec_table = validate_vector_table(
        "vec_clip"
        if embedding_type in ("clip", "clip_thumbnail")
        else "vec_text"
    )

    # For whisper embeddings, skip if transcript is too short to be meaningful.
    # BGM-only files often produce a single spurious word ("you", "the", etc.)
    # whose embedding matches everything.
    if embedding_type == "whisper":
        with get_search_db_read() as session:
            from app.models import TranscriptChunk
            chunks = (
                session.query(TranscriptChunk)
                .filter(TranscriptChunk.file_id == file_id)
                .all()
            )
            total_text = " ".join(c.text for c in chunks).strip()
            if len(total_text) < 20:
                logger.debug(
                    "Skipping whisper similar for %s: transcript too short (%d chars)",
                    file_id, len(total_text),
                )
                return []

    # Get the source file's embedding IDs
    with get_search_db_read() as session:
        source_embeddings = (
            session.query(Embedding.id)
            .filter(
                Embedding.file_id == file_id,
                Embedding.embedding_type == embedding_type,
            )
            .all()
        )

    if not source_embeddings:
        return []

    embedding_ids = [e.id for e in source_embeddings]

    # Retrieve vectors and compute average
    engine = get_search_engine()
    vectors: list[np.ndarray] = []

    with engine.connect() as conn:
        for eid in embedding_ids:
            row = conn.execute(
                sql_text(
                    f"SELECT vector FROM {vec_table} "
                    f"WHERE embedding_id = :eid"
                ),
                {"eid": eid},
            ).fetchone()
            if row and row[0]:
                vec = np.frombuffer(row[0], dtype=np.float32)
                vectors.append(vec)

    if not vectors:
        return []

    # Average and normalize
    avg_vector = np.mean(vectors, axis=0)
    norm = np.linalg.norm(avg_vector)
    if norm > 0:
        avg_vector = avg_vector / norm

    # Query for nearest neighbors.
    # Must fetch enough to look past the source file's own embeddings
    # (e.g. a video with 89 CLIP frames will occupy the top ~89 slots).
    source_count = len(embedding_ids)
    fetch_limit = source_count + limit + 20
    with engine.connect() as conn:
        rows = conn.execute(
            sql_text(
                f"SELECT embedding_id, distance "
                f"FROM {vec_table} "
                f"WHERE vector MATCH :vec "
                f"ORDER BY distance "
                f"LIMIT :limit"
            ),
            {"vec": avg_vector.tobytes(), "limit": fetch_limit},
        ).fetchall()

    if not rows:
        return []

    neighbor_ids = [row[0] for row in rows]
    distances = {row[0]: row[1] for row in rows}

    # Look up file info for neighbor embeddings
    with get_search_db_read() as session:
        neighbor_embeddings = (
            session.query(Embedding)
            .filter(Embedding.id.in_(neighbor_ids))
            .all()
        )

        # Map embedding -> file_id with best score per file
        file_best_score: dict[str, float] = {}
        for emb in neighbor_embeddings:
            if emb.file_id == file_id:
                continue  # exclude self
            score = _l2_to_cosine_similarity(distances.get(emb.id, 2.0))
            if score > file_best_score.get(emb.file_id, 0.0):
                file_best_score[emb.file_id] = score

        if not file_best_score:
            return []

        # --- Quality filters ---

        # 1. Remove near-identical scores (score >= 0.999) — these are
        #    duplicate embeddings from empty/trivial content (e.g. BGM
        #    files where Whisper outputs the same single word)
        file_best_score = {
            fid: s for fid, s in file_best_score.items() if s < 0.999
        }

        if not file_best_score:
            return []

        # 2. Absolute minimum score: below this, similarity is not
        #    meaningful regardless of relative ranking
        min_similar_score = 0.70
        file_best_score = {
            fid: s for fid, s in file_best_score.items()
            if s >= min_similar_score
        }

        if not file_best_score:
            return []

        # 3. Score gap analysis: if many candidates all have flat scores,
        #    the embedding doesn't meaningfully distinguish files.
        #    Only apply when there are enough candidates to compute
        #    a meaningful gap (>= 5); with few candidates, the absolute
        #    min_score threshold is sufficient.
        #    Skip when top_score is very high — uniform-high scores then
        #    mean "everything is genuinely similar" (e.g. long-running
        #    anime series with many episodes), not a non-discriminating
        #    embedding.
        scores = list(file_best_score.values())
        top_score = max(scores)
        uniform_high_threshold = 0.85

        if len(scores) >= 5 and top_score < uniform_high_threshold:
            mean_score = sum(scores) / len(scores)
            gap = top_score - mean_score

            if gap < 0.01:
                logger.debug(
                    "Similar search gap too small (%.4f) for %s via %s, "
                    "discarding %d candidates",
                    gap, file_id, embedding_type, len(scores),
                )
                return []

        # 4. Margin cutoff: keep only results within margin of top score
        margin = 0.05
        file_best_score = {
            fid: s for fid, s in file_best_score.items()
            if s >= top_score - margin
        }

        if not file_best_score:
            return []

        # 5. Spread check: if many candidates remain and all are
        #    bunched within a narrow band, the embedding doesn't
        #    meaningfully distinguish them. Use coefficient of
        #    variation (std/mean) which captures spread relative
        #    to the score level. Same uniform-high carve-out as the
        #    gap check above.
        scores = list(file_best_score.values())
        if len(scores) >= 5 and top_score < uniform_high_threshold:
            s_mean = sum(scores) / len(scores)
            s_std = (sum((s - s_mean) ** 2 for s in scores) / len(scores)) ** 0.5
            cv = s_std / s_mean if s_mean > 0 else 0.0
            if cv < 0.01:
                logger.debug(
                    "Similar search CV too small (%.4f) for %s via %s, "
                    "discarding %d candidates",
                    cv, file_id, embedding_type, len(scores),
                )
                return []

        # Get file metadata for matched files
        query = session.query(IndexedFile).filter(
            IndexedFile.file_id.in_(list(file_best_score.keys())),
            IndexedFile.active.is_(True),
        )
        if drive:
            query = query.filter(IndexedFile.drive == drive)

        files = {f.file_id: f for f in query.all()}

    # Build results sorted by score
    results: list[dict] = []
    for fid, score in sorted(
        file_best_score.items(), key=lambda x: x[1], reverse=True
    ):
        f = files.get(fid)
        if f is None:
            continue
        results.append({
            "file_id": f.file_id,
            "drive": f.drive,
            "filename": f.filename,
            "file_type": f.file_type,
            "mime_type": f.mime_type,
            "score": score,
        })
        if len(results) >= limit:
            break

    return results


# ---------------------------------------------------------------------------
# Cosine-similarity-based scoring (alternative to RRF)
# ---------------------------------------------------------------------------

_TYPE_WEIGHTS: dict[str, str] = {
    "metadata": "type_weight_metadata",
    "transcript": "type_weight_transcript",
    "text_content": "type_weight_text_content",
    "clip": "type_weight_clip",
    # ``clip_thumbnail`` ranks at parity with text-class signals because
    # it represents a deliberately-chosen single frame (no per-file
    # bias from frame count). Spec
    # 2026-05-02-thumbnail-clip-default-shallow-search.md.
    "clip_thumbnail": "type_weight_clip_thumbnail",
}


def _combine_scores_cosine(
    text_matches: list[_VectorMatch],
    clip_matches: list[_VectorMatch],
    keyword_matches: list[_KeywordMatch],
    transcript_keyword_matches: list[_TranscriptKeywordMatch],
    text_content_keyword_matches: list[_TextContentKeywordMatch] | None = None,
    retrieval_keyword_matches: list[_RetrievalKeywordMatch] | None = None,
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
        "retrieval_keywords": {},
    }

    for m in text_matches:
        file_match_types.setdefault(m.file_id, set()).add(m.embedding_type)
        file_matches.setdefault(m.file_id, []).append(MatchInfo(
            match_type=m.embedding_type,
            text=m.content_preview,
            score=m.score,
            timestamp_start=m.timestamp_start,
            timestamp_end=m.timestamp_end,
            page=m.page,
        ))
        weight = getattr(search_config, _TYPE_WEIGHTS.get(m.embedding_type, ""), 1.0)
        weighted = m.score * weight
        src = file_source_best["text_vector"]
        src[m.file_id] = max(src.get(m.file_id, 0.0), weighted)

    for m in clip_matches:
        # ``embedding_type`` is now opaque ("clip" or "clip_thumbnail");
        # surface it as the match_type so the UI can label hits and
        # apply the correct ``type_weight_*`` knob.
        match_label = m.embedding_type or "clip"
        file_match_types.setdefault(m.file_id, set()).add(match_label)
        file_matches.setdefault(m.file_id, []).append(MatchInfo(
            match_type=match_label,
            text=m.content_preview,
            score=m.score,
            timestamp_start=m.timestamp_start,
            timestamp_end=m.timestamp_end,
            page=m.page,
        ))
        weight = getattr(
            search_config,
            _TYPE_WEIGHTS.get(match_label, "type_weight_clip"),
            search_config.type_weight_clip,
        )
        weighted = m.score * weight
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

    # SIRA retrieval-keywords hits: chip-only MatchInfo (no time_range
    # / page) so search_merge.ts treats it as a badge rather than a
    # jump target. The score participates in best_keyword max() so a
    # file the LLM expanded to a relevant query still surfaces.
    for m in (retrieval_keyword_matches or []):
        file_match_types.setdefault(m.file_id, set()).add("retrieval_keywords")
        file_matches.setdefault(m.file_id, []).append(MatchInfo(
            match_type="retrieval_keywords",
            text=m.matched_keyword,
            score=m.score,
        ))
        src = file_source_best["retrieval_keywords"]
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
            file_source_best["retrieval_keywords"].get(fid, 0.0),
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

    with get_search_db_read() as session:
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


# --- Required-keyword hard filter (Phase 2 of structured retriever) ------
#
# A small layer on top of the existing FTS5 trigram tables that enforces
# "the user's required proper nouns must appear in this file" before
# any vector / RRF ranking happens. The structured query transform
# (``app.rag.query_transform.transform_query_structured``) decides which
# terms are required and what their kana / case / diacritic variants
# are; this module just runs the resulting OR-of-aliases clauses across
# the existing FTS tables and intersects the per-term file_id sets.
#
# Spec: docs/superpowers/specs/2026-04-30-required-semantic-hybrid-retrieval.md


def _build_required_or_clause(term: "RequiredTerm") -> str:
    """Build the FTS5 boolean clause for a single required-term group.

    All non-empty aliases are deduplicated, stripped of double-quotes
    (which cannot appear inside a phrase token), and joined with ``OR``.
    A single alias returns the bare phrase; multiple aliases are
    wrapped in parentheses so an outer ``AND`` join (across multiple
    required groups) keeps the precedence unambiguous.

    Returns the empty string when the term contributes nothing usable
    after sanitisation. Callers must skip empty clauses rather than
    issuing them — FTS5 raises on an empty MATCH.
    """
    seen: set[str] = set()
    aliases: list[str] = []
    for alias in term.aliases:
        if not isinstance(alias, str):
            continue
        sanitized = alias.replace('"', "").strip()
        if not sanitized or sanitized in seen:
            continue
        seen.add(sanitized)
        aliases.append(sanitized)
    if not aliases:
        return ""
    if len(aliases) == 1:
        return f'"{aliases[0]}"'
    return "(" + " OR ".join(f'"{a}"' for a in aliases) + ")"


def _fts_lookup_required(or_clause: str) -> set[str]:
    """Return file_ids matching the OR clause in any required-filter FTS table.

    The hard filter unions hits across the dual FTS5 surface: the
    legacy trigram tables (strong on CJK substring matching, but
    shatter Latin/Cyrillic/Hangul tokens) and the word-tokenized
    parallel tables introduced in Phase 3 (``unicode61 remove_diacritics 2``,
    strong on word-boundary languages and case/diacritic folding).
    Any single hit in any of the six tables passes the required term —
    callers benefit from substring-recall for CJK and word-precision
    for Latin without having to dispatch by language at the call site.

    Active-only filtering is applied at the end so soft-deleted /
    missing files do not show up. An empty ``or_clause`` returns the
    empty set (callers must already skip empties, this is a defence-
    in-depth check).
    """
    if not or_clause:
        return set()

    engine = get_search_engine()
    file_ids: set[str] = set()
    with engine.connect() as conn:
        for table_name in (
            "fts_files",
            "fts_transcripts",
            "fts_text_content",
            "fts_files_word",
            "fts_transcripts_word",
            "fts_text_content_word",
        ):
            sql = (
                f"SELECT DISTINCT file_id FROM {table_name} "
                f"WHERE {table_name} MATCH :q"
            )
            try:
                rows = conn.execute(
                    sql_text(sql), {"q": or_clause}
                ).fetchall()
            except Exception as e:  # pragma: no cover - FTS syntax safety net
                logger.warning(
                    "Required-filter FTS lookup failed on %s: %s",
                    table_name, e,
                )
                continue
            file_ids.update(row[0] for row in rows if row[0])

    if not file_ids:
        return set()

    with get_search_db_read() as session:
        active_ids = {
            row.file_id
            for row in session.query(IndexedFile.file_id)
            .filter(
                IndexedFile.file_id.in_(list(file_ids)),
                IndexedFile.active.is_(True),
            )
            .all()
        }
    return active_ids


def _required_keyword_filter(
    required: "tuple[RequiredTerm, ...]",
) -> set[str] | None:
    """Compute the AND-intersection of per-required-term FTS lookups.

    Semantics:

    * Empty ``required`` → returns ``None``. Callers should leave the
      effective ``file_id_scope`` unchanged (no hard filter).
    * Non-empty ``required`` with all terms degenerating to empty FTS
      clauses → also returns ``None`` (treated identically to "no
      hard filter" so an LLM emitting nothing usable does not nuke
      the entire pipeline).
    * Non-empty ``required`` with at least one usable term → returns
      the intersection ``set[str]`` of file_ids matching every usable
      term. Empty set means the hard filter dropped everything; the
      caller may then trigger Tier 3 fallback (demote required to
      semantic) per the spec §3.5.

    Per-term semantics within a group is OR across aliases; per-group
    semantics across multiple required terms is AND. The function
    short-circuits as soon as the running intersection is empty so
    pathological queries do not waste FTS lookups.
    """
    if not required:
        return None

    survivors: set[str] | None = None
    used_any_clause = False

    for term in required:
        clause = _build_required_or_clause(term)
        if not clause:
            continue
        used_any_clause = True
        ids = _fts_lookup_required(clause)
        if survivors is None:
            survivors = ids
        else:
            survivors = survivors & ids
        if not survivors:
            # AND with empty intersection ⇒ empty for the rest of the
            # chain; bail out before issuing further FTS queries.
            return set()

    if not used_any_clause:
        return None
    return survivors
