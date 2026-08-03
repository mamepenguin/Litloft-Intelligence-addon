"""Stage 1/2/3 runners + per-case metric computation.

Each stage function is async because the underlying RAG pipeline is
async; the runner awaits them sequentially per case (parallelism would
just oversubscribe the LLM endpoint and obscure timing variance).
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from app.evals.agentic_metrics import (
    route_correctness as _route_correctness,
)
from app.evals.loader import Case, GroundTruthFile, SegmentHint
from app.evals.resolver import ResolvedGroundTruth
from app.evals.text_match import (
    coverage as _coverage,
    global_exclude_terms,
    violation_count as _violation_count,
)
from app.rag.agentic_types import AgenticTelemetry
from app.rag.query_transform import (
    RequiredTerm,
    iter_required_fallback_subsets,
    transform_query_structured,
)
from app.rag.service import answer_question
from app.search import SearchResult, search

logger = logging.getLogger(__name__)

EVAL_TEMPERATURE = 0.0


def _iou_seconds(
    a: tuple[float, float], b: tuple[float, float]
) -> float:
    """Intersection-over-Union for two time ranges, guarded against
    inverted/zero spans (returns 0.0 in those cases)."""
    a0, a1 = min(a), max(a)
    b0, b1 = min(b), max(b)
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def _segment_matches_hint(result: SearchResult, hint: SegmentHint) -> bool:
    """True if any segment in ``result`` satisfies the ground-truth hint.

    For time hints we accept either:
    * ``IoU >= 0.3`` (tight alignment between segment and hint), or
    * ``hint_coverage >= 0.5`` (the segment covers at least half of the
      hint window, even if the segment itself is much wider).

    The coverage clause matters for long videos where the indexer
    aggregates many transcript chunks into a single wide ``SegmentGroup``
    (e.g. [95s, 478s] — 6+ minutes wide). With IoU alone, a 40-second
    hint inside such a group scores ~0.10 and fails, even though the
    retriever clearly surfaced the correct moment. Coverage rewards
    that "right span surfaced" outcome without dropping the IoU-based
    discrimination on tighter matches.
    """
    for group in result.segments:
        if hint.time_range is not None and group.time_range is not None:
            iou = _iou_seconds(group.time_range, hint.time_range)
            if iou >= 0.3:
                return True
            seg_lo, seg_hi = min(group.time_range), max(group.time_range)
            hint_lo, hint_hi = min(hint.time_range), max(hint.time_range)
            hint_size = hint_hi - hint_lo
            if hint_size > 0:
                intersection = max(0.0, min(seg_hi, hint_hi) - max(seg_lo, hint_lo))
                if intersection / hint_size >= 0.5:
                    return True
        # page hint: exact match against any matchInfo.page
        if hint.page is not None:
            for m in group.matches:
                if m.page is not None and m.page == hint.page:
                    return True
    return False


# --------------------------------------------------------------------------- #
# Stage 1: query_transform
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Stage1Result:
    keywords: str
    must_include_coverage: float
    must_exclude_violations: int
    # Phase 2: structured-transform output is forwarded to Stage 2 so
    # the hard filter has the same shape the production retriever uses.
    required: tuple[RequiredTerm, ...] = ()


async def run_stage1(case: Case) -> Stage1Result:
    structured = await transform_query_structured(
        case.query, temperature=EVAL_TEMPERATURE
    )
    # The flat keywords string used by the eval text-match metrics is
    # the structured form's raw projection (canonicals + semantic
    # joined by spaces). Falls back to the raw query on full
    # passthrough so coverage / violation counts remain meaningful.
    keywords = structured.raw_keywords or case.query
    # must_exclude = case-local set ∪ global blocklists (question + file-type)
    exclude_terms = tuple(
        set(case.expected_keywords.must_exclude) | set(global_exclude_terms())
    )
    return Stage1Result(
        keywords=keywords,
        must_include_coverage=_coverage(
            case.expected_keywords.must_include, keywords
        ),
        must_exclude_violations=_violation_count(exclude_terms, keywords),
        required=structured.required,
    )


# --------------------------------------------------------------------------- #
# Stage 2: retrieve
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Stage2Result:
    keywords: str
    top_file_ids: tuple[str, ...]  # ordered
    file_recall_at_5: float
    file_recall_at_10: float
    segment_recall_at_5: float
    segment_recall_at_10: float
    mrr: float
    precision_at_5: float
    # Diagnostic: which ground-truth file_ids landed in top-10 (for failures)
    retrieved_gt_file_ids: tuple[str, ...]


def _file_recall_at_k(
    results: list[SearchResult], gt_ids: set[str], k: int
) -> float:
    if not gt_ids:
        return 1.0
    top = {r.file_id for r in results[:k]}
    return len(gt_ids & top) / len(gt_ids)


def _segment_recall_at_k(
    results: list[SearchResult],
    gts_with_hint: list[ResolvedGroundTruth],
    k: int,
) -> float:
    if not gts_with_hint:
        return 1.0
    top = results[:k]
    by_id: dict[str, SearchResult] = {r.file_id: r for r in top}
    hits = 0
    for gt in gts_with_hint:
        if gt.file_id is None or gt.file_id not in by_id:
            continue
        if _segment_matches_hint(by_id[gt.file_id], gt.segment_hint):  # type: ignore[arg-type]
            hits += 1
    return hits / len(gts_with_hint)


def _mrr(results: list[SearchResult], gt_ids: set[str]) -> float:
    for rank, r in enumerate(results, start=1):
        if r.file_id in gt_ids:
            return 1.0 / rank
    return 0.0


async def run_stage2(
    case: Case,
    keywords: str,
    resolved: list[ResolvedGroundTruth],
    drive: str,
    top_k: int = 10,
    *,
    required: tuple[RequiredTerm, ...] = (),
) -> Stage2Result:
    response = await asyncio.to_thread(
        search,
        keywords,
        limit=top_k,
        drive=drive,
        mode="recall",
        semantic_query=case.query,
        required=required or None,
    )

    # Tier 2/3 fallback ladder: mirror the production retriever
    # (retrieve_with_keywords) so eval numbers reflect what users
    # actually experience. Each step drops the most-aliased required
    # term; the terminal step demotes everything to semantic.
    if required and not response.results:
        for subset in iter_required_fallback_subsets(required):
            response = await asyncio.to_thread(
                search,
                keywords,
                limit=top_k,
                drive=drive,
                mode="recall",
                semantic_query=case.query,
                required=subset or None,
            )
            if response.results:
                break
    results = list(response.results)
    gt_ids = {gt.file_id for gt in resolved if gt.file_id is not None}
    gts_with_hint = [
        gt for gt in resolved if gt.file_id is not None and gt.segment_hint is not None
    ]

    top_5 = results[:5]
    return Stage2Result(
        keywords=keywords,
        top_file_ids=tuple(r.file_id for r in results),
        file_recall_at_5=_file_recall_at_k(results, gt_ids, 5),
        file_recall_at_10=_file_recall_at_k(results, gt_ids, 10),
        segment_recall_at_5=_segment_recall_at_k(results, gts_with_hint, 5),
        segment_recall_at_10=_segment_recall_at_k(results, gts_with_hint, 10),
        mrr=_mrr(results, gt_ids),
        precision_at_5=(
            sum(1 for r in top_5 if r.file_id in gt_ids) / 5 if top_5 else 0.0
        ),
        retrieved_gt_file_ids=tuple(
            r.file_id for r in results[:10] if r.file_id in gt_ids
        ),
    )


# --------------------------------------------------------------------------- #
# Stage 3: generate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Stage3SingleRun:
    answer: str | None
    citation_file_ids: tuple[str, ...]
    retrieved_file_ids: tuple[str, ...]
    must_mention_coverage: float
    citation_in_ground_truth: float
    citation_segment_match: float | None
    citation_in_retrieved: float
    took_ms: int
    # Agentic-loop telemetry (Phase 1.A). ``None`` for legacy runs.
    agentic_telemetry: AgenticTelemetry | None = None
    # Cached per-run agentic metrics so the report layer does not
    # recompute them. ``None`` means "not applicable for this run".
    tool_call_count: int | None = None
    route_correctness: float | None = None
    max_context_tokens_used: int | None = None
    forced_answer: bool | None = None


async def run_stage3_once(
    case: Case,
    resolved: list[ResolvedGroundTruth],
    drive: str,
    top_k: int,
    *,
    force_legacy_rag: bool = False,
) -> Stage3SingleRun:
    start = time.monotonic()
    response = await answer_question(
        query=case.query,
        credential=None,
        top_k=top_k,
        drive=drive,
        temperature=EVAL_TEMPERATURE,
        force_legacy_rag=force_legacy_rag,
    )

    answer = response.answer or ""
    citation_ids = tuple(c.get("file_id") for c in response.citations if c.get("file_id"))
    retrieved_ids = tuple(s.get("file_id") for s in response.sources if s.get("file_id"))

    gt_ids = {gt.file_id for gt in resolved if gt.file_id is not None}
    gt_with_hint_by_id: dict[str, SegmentHint] = {
        gt.file_id: gt.segment_hint  # type: ignore[misc]
        for gt in resolved
        if gt.file_id is not None and gt.segment_hint is not None
    }

    if citation_ids:
        in_gt = sum(1 for fid in citation_ids if fid in gt_ids) / len(citation_ids)
        in_retrieved = sum(1 for fid in citation_ids if fid in retrieved_ids) / len(
            citation_ids
        )
    else:
        in_gt = 0.0
        in_retrieved = 0.0

    # citation_segment_match: only count citations whose target file has a hint.
    seg_hits = 0
    seg_total = 0
    for c in response.citations:
        fid = c.get("file_id")
        hint = gt_with_hint_by_id.get(fid) if fid else None
        if hint is None:
            continue
        seg_total += 1
        loc = (c.get("segment_location") or "").strip()
        if not loc:
            continue
        # segment_location is "m:ss" for time hints, "page N" for page hints.
        if hint.time_range is not None and ":" in loc:
            try:
                mm, ss = loc.split(":", 1)
                seconds = int(mm) * 60 + int(ss)
            except ValueError:
                continue
            lo, hi = hint.time_range
            # Allow a 30s window (matches transcript_window_seconds default).
            if lo - 30 <= seconds <= hi + 30:
                seg_hits += 1
        elif hint.page is not None and loc.lower().startswith("page"):
            try:
                page_num = int(loc.split()[-1])
            except (ValueError, IndexError):
                continue
            if page_num == hint.page:
                seg_hits += 1
    seg_match = (seg_hits / seg_total) if seg_total > 0 else None

    telemetry = response.agentic_telemetry
    return Stage3SingleRun(
        answer=response.answer,
        citation_file_ids=tuple(fid for fid in citation_ids if fid is not None),
        retrieved_file_ids=tuple(fid for fid in retrieved_ids if fid is not None),
        must_mention_coverage=_coverage(case.must_mention, answer),
        citation_in_ground_truth=in_gt,
        citation_segment_match=seg_match,
        citation_in_retrieved=in_retrieved,
        took_ms=int((time.monotonic() - start) * 1000),
        agentic_telemetry=telemetry,
        tool_call_count=(
            len(telemetry.tool_calls) if telemetry is not None else None
        ),
        route_correctness=_route_correctness(case.expected_tool_sequence, telemetry),
        max_context_tokens_used=(
            telemetry.max_context_tokens_used if telemetry is not None else None
        ),
        forced_answer=(telemetry.forced_answer if telemetry is not None else None),
    )


@dataclass(frozen=True)
class AggregatedMetric:
    """Median + min + max + unstable flag for a Stage 3 metric over N runs."""

    values: tuple[float, ...]
    median: float
    min: float
    max: float
    unstable: bool

    @classmethod
    def from_values(cls, values: list[float], epsilon: float) -> "AggregatedMetric":
        if not values:
            return cls(values=(), median=0.0, min=0.0, max=0.0, unstable=False)
        vmin = min(values)
        vmax = max(values)
        return cls(
            values=tuple(values),
            median=statistics.median(values),
            min=vmin,
            max=vmax,
            unstable=(vmax - vmin) > epsilon,
        )


@dataclass(frozen=True)
class Stage3Aggregated:
    runs: tuple[Stage3SingleRun, ...]
    must_mention_coverage: AggregatedMetric
    citation_in_ground_truth: AggregatedMetric
    citation_segment_match: AggregatedMetric  # values may be empty
    citation_in_retrieved: AggregatedMetric


def aggregate_stage3(
    runs: list[Stage3SingleRun], epsilon: float
) -> Stage3Aggregated:
    seg_values = [r.citation_segment_match for r in runs if r.citation_segment_match is not None]
    return Stage3Aggregated(
        runs=tuple(runs),
        must_mention_coverage=AggregatedMetric.from_values(
            [r.must_mention_coverage for r in runs], epsilon
        ),
        citation_in_ground_truth=AggregatedMetric.from_values(
            [r.citation_in_ground_truth for r in runs], epsilon
        ),
        citation_segment_match=AggregatedMetric.from_values(seg_values, epsilon),
        citation_in_retrieved=AggregatedMetric.from_values(
            [r.citation_in_retrieved for r in runs], epsilon
        ),
    )


# --------------------------------------------------------------------------- #
# Per-case driver
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaseReport:
    case: Case
    resolved: tuple[ResolvedGroundTruth, ...]
    stage1: Stage1Result
    stage2: Stage2Result
    stage3: Stage3Aggregated


async def run_case(
    case: Case,
    resolved: list[ResolvedGroundTruth],
    drive: str,
    runs_stage3: int,
    epsilon: float,
    top_k: int,
    *,
    force_legacy_rag: bool = False,
) -> CaseReport:
    logger.info("Running case %s", case.id)
    stage1 = await run_stage1(case)
    stage2 = await run_stage2(
        case,
        stage1.keywords,
        resolved,
        drive,
        top_k=10,
        required=stage1.required,
    )

    runs: list[Stage3SingleRun] = []
    for i in range(max(1, runs_stage3)):
        logger.info("  stage3 run %d/%d", i + 1, runs_stage3)
        run = await run_stage3_once(
            case, resolved, drive, top_k=top_k, force_legacy_rag=force_legacy_rag
        )
        runs.append(run)
    stage3 = aggregate_stage3(runs, epsilon)

    return CaseReport(
        case=case,
        resolved=tuple(resolved),
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
    )
