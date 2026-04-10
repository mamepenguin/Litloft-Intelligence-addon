"""Debug search endpoint for analyzing raw scores from each search system.

Provides detailed per-system score breakdowns and per-step timing
to diagnose false positives, tune thresholds, and profile latency.
Only intended for development/debugging use.
"""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

import numpy as np
from pydantic import BaseModel
from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_search_db, get_search_engine
from app.models import Embedding, IndexedFile, TranscriptChunk
from app.search import (
    _build_fts_query,
    _combine_scores_rrf,
    _keyword_search,
    _keyword_search_text_content,
    _keyword_search_transcripts,
    _l2_to_cosine_similarity,
    _vector_search_clip,
    _vector_search_text,
)
from app.workers.clip import embed_text_clip
from app.workers.embedder import embed_query

logger = logging.getLogger(__name__)


_T = TypeVar("_T")


def _time_it(timings: dict[str, float], label: str, fn: Callable[[], _T]) -> _T:
    """Run fn() and record its wall-clock duration in milliseconds."""
    start = time.perf_counter()
    result = fn()
    timings[label] = round((time.perf_counter() - start) * 1000, 2)
    return result


# --- Response models ---


class DebugVectorMatch(BaseModel):
    file_id: str
    filename: str
    score: float
    embedding_type: str
    content_preview: str
    timestamp_start: float | None = None
    timestamp_end: float | None = None


class DebugKeywordMatch(BaseModel):
    file_id: str
    filename: str
    score: float
    matched_field: str


class DebugTranscriptKeywordMatch(BaseModel):
    file_id: str
    filename: str
    score: float
    text: str
    timestamp_start: float | None = None
    timestamp_end: float | None = None


class DebugTextContentKeywordMatch(BaseModel):
    file_id: str
    filename: str
    score: float
    text: str
    page: int | None = None


class DebugCombinedResult(BaseModel):
    file_id: str
    filename: str
    rrf_score: float
    match_types: list[str]
    source_scores: dict[str, float]


class DebugSearchResponse(BaseModel):
    query: str
    fts_query: str
    config: dict
    text_vector: list[DebugVectorMatch]
    clip_vector: list[DebugVectorMatch]
    keyword: list[DebugKeywordMatch]
    transcript_keyword: list[DebugTranscriptKeywordMatch]
    text_content_keyword: list[DebugTextContentKeywordMatch]
    combined: list[DebugCombinedResult]
    score_stats: dict
    timings_ms: dict[str, float]  # wall-clock ms per step


def _filenames_map(file_ids: list[str]) -> dict[str, str]:
    """Look up filenames for a list of file_ids."""
    if not file_ids:
        return {}
    with get_search_db() as session:
        files = (
            session.query(IndexedFile.file_id, IndexedFile.filename)
            .filter(IndexedFile.file_id.in_(file_ids))
            .all()
        )
        return {f.file_id: f.filename for f in files}


def debug_search(query: str) -> DebugSearchResponse:
    """Run search with full diagnostic output from each system."""
    search_config = settings.search
    candidates = search_config.rrf_candidates
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    # Generate embeddings
    text_vector = _time_it(
        timings, "embed_query_text", lambda: embed_query(query),
    )

    def _clip_embed() -> np.ndarray | None:
        try:
            return embed_text_clip(query)
        except Exception:
            return None

    clip_vector = _time_it(timings, "embed_query_clip", _clip_embed)

    # Run each system independently
    text_matches = _time_it(
        timings,
        "vector_search_text",
        lambda: _vector_search_text(text_vector, candidates),
    )
    clip_matches = _time_it(
        timings,
        "vector_search_clip",
        lambda: (
            _vector_search_clip(clip_vector, candidates)
            if clip_vector is not None
            else []
        ),
    )
    keyword_matches = _time_it(
        timings,
        "keyword_search_metadata",
        lambda: _keyword_search(query, candidates),
    )
    transcript_kw_matches = _time_it(
        timings,
        "keyword_search_transcripts",
        lambda: _keyword_search_transcripts(query, candidates),
    )
    text_content_kw_matches = _time_it(
        timings,
        "keyword_search_text_content",
        lambda: _keyword_search_text_content(query, candidates),
    )

    # Collect all file_ids for filename lookup
    all_file_ids = list({
        *[m.file_id for m in text_matches],
        *[m.file_id for m in clip_matches],
        *[m.file_id for m in keyword_matches],
        *[m.file_id for m in transcript_kw_matches],
        *[m.file_id for m in text_content_kw_matches],
    })
    names = _time_it(
        timings,
        "filename_lookup",
        lambda: _filenames_map(all_file_ids),
    )

    # Build debug output for each system
    debug_text = sorted(
        [
            DebugVectorMatch(
                file_id=m.file_id,
                filename=names.get(m.file_id, "?"),
                score=round(m.score, 4),
                embedding_type=m.embedding_type,
                content_preview=m.content_preview[:100],
                timestamp_start=m.timestamp_start,
                timestamp_end=m.timestamp_end,
            )
            for m in text_matches
        ],
        key=lambda x: x.score,
        reverse=True,
    )

    debug_clip = sorted(
        [
            DebugVectorMatch(
                file_id=m.file_id,
                filename=names.get(m.file_id, "?"),
                score=round(m.score, 4),
                embedding_type=m.embedding_type,
                content_preview=m.content_preview[:100],
                timestamp_start=m.timestamp_start,
                timestamp_end=m.timestamp_end,
            )
            for m in clip_matches
        ],
        key=lambda x: x.score,
        reverse=True,
    )

    debug_kw = sorted(
        [
            DebugKeywordMatch(
                file_id=m.file_id,
                filename=names.get(m.file_id, "?"),
                score=round(m.score, 4),
                matched_field=m.matched_field,
            )
            for m in keyword_matches
        ],
        key=lambda x: x.score,
        reverse=True,
    )

    debug_transcript_kw = sorted(
        [
            DebugTranscriptKeywordMatch(
                file_id=m.file_id,
                filename=names.get(m.file_id, "?"),
                score=round(m.score, 4),
                text=m.text[:100],
                timestamp_start=m.timestamp_start,
                timestamp_end=m.timestamp_end,
            )
            for m in transcript_kw_matches
        ],
        key=lambda x: x.score,
        reverse=True,
    )

    debug_text_content_kw = sorted(
        [
            DebugTextContentKeywordMatch(
                file_id=m.file_id,
                filename=names.get(m.file_id, "?"),
                score=round(m.score, 4),
                text=m.text[:100],
                page=m.page,
            )
            for m in text_content_kw_matches
        ],
        key=lambda x: x.score,
        reverse=True,
    )

    # Combined RRF
    file_scores = _time_it(
        timings,
        "combine_scores_rrf",
        lambda: _combine_scores_rrf(
            text_matches=text_matches,
            clip_matches=clip_matches,
            keyword_matches=keyword_matches,
            transcript_keyword_matches=transcript_kw_matches,
            text_content_keyword_matches=text_content_kw_matches,
            k=search_config.rrf_k,
        ),
    )

    # Build per-file source score map
    # Track best text vector score per file
    text_best: dict[str, float] = {}
    for m in text_matches:
        text_best[m.file_id] = max(text_best.get(m.file_id, 0.0), m.score)
    clip_best: dict[str, float] = {}
    for m in clip_matches:
        clip_best[m.file_id] = max(clip_best.get(m.file_id, 0.0), m.score)
    kw_best: dict[str, float] = {}
    for m in keyword_matches:
        kw_best[m.file_id] = max(kw_best.get(m.file_id, 0.0), m.score)
    tkw_best: dict[str, float] = {}
    for m in transcript_kw_matches:
        tkw_best[m.file_id] = max(tkw_best.get(m.file_id, 0.0), m.score)
    tckw_best: dict[str, float] = {}
    for m in text_content_kw_matches:
        tckw_best[m.file_id] = max(tckw_best.get(m.file_id, 0.0), m.score)

    combined_sorted = sorted(
        file_scores.values(), key=lambda fs: fs.combined_score, reverse=True
    )

    debug_combined = [
        DebugCombinedResult(
            file_id=fs.file_id,
            filename=names.get(fs.file_id, "?"),
            rrf_score=round(fs.combined_score, 6),
            match_types=sorted(fs.match_types),
            source_scores={
                k: round(v, 4)
                for k, v in {
                    "text_vector": text_best.get(fs.file_id, 0.0),
                    "clip_vector": clip_best.get(fs.file_id, 0.0),
                    "keyword": kw_best.get(fs.file_id, 0.0),
                    "transcript_keyword": tkw_best.get(fs.file_id, 0.0),
                    "text_content_keyword": tckw_best.get(fs.file_id, 0.0),
                }.items()
                if v > 0
            },
        )
        for fs in combined_sorted
    ]

    # Score statistics
    text_scores = [m.score for m in text_matches]
    clip_scores = [m.score for m in clip_matches]

    score_stats = {
        "text_vector": {
            "count": len(text_scores),
            "min": round(min(text_scores), 4) if text_scores else None,
            "max": round(max(text_scores), 4) if text_scores else None,
            "mean": round(sum(text_scores) / len(text_scores), 4) if text_scores else None,
            "threshold": search_config.min_score_text,
        },
        "clip_vector": {
            "count": len(clip_scores),
            "min": round(min(clip_scores), 4) if clip_scores else None,
            "max": round(max(clip_scores), 4) if clip_scores else None,
            "mean": round(sum(clip_scores) / len(clip_scores), 4) if clip_scores else None,
            "threshold": search_config.min_score_clip,
        },
        "keyword": {"count": len(keyword_matches)},
        "transcript_keyword": {"count": len(transcript_kw_matches)},
        "text_content_keyword": {"count": len(text_content_kw_matches)},
    }

    timings["total"] = round((time.perf_counter() - total_start) * 1000, 2)

    return DebugSearchResponse(
        query=query,
        fts_query=_build_fts_query(query),
        config={
            "min_score_text": search_config.min_score_text,
            "min_score_clip": search_config.min_score_clip,
            "score_gap_threshold": search_config.score_gap_threshold,
            "score_cutoff_margin": search_config.score_cutoff_margin,
            "rrf_k": search_config.rrf_k,
            "rrf_candidates": search_config.rrf_candidates,
            "rrf_weight_clip": search_config.rrf_weight_clip,
            "score_cutoff_ratio": search_config.score_cutoff_ratio,
            "text_model": settings.models.text_embedding,
            "clip_model": settings.models.clip,
        },
        text_vector=debug_text,
        clip_vector=debug_clip,
        keyword=debug_kw,
        transcript_keyword=debug_transcript_kw,
        text_content_keyword=debug_text_content_kw,
        combined=debug_combined,
        score_stats=score_stats,
        timings_ms=timings,
    )
