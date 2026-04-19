"""Score a citation case against ground truth.

Given a ``CitationCase`` and the file's stored ``detailed_summary``, run
``app.citations.compute_citations`` and score the output segment-by-
segment. Metrics are simple and interpretable:

* **top1_hit** — for each expected segment, is the top-ranked chunk an
  accepted match (chunk id exact OR hint-range overlap)?
* **recall_at_3** — is any of the top-3 chunks an accepted match?
* **has_citation_precision** — of the segments the linker marked
  ``has_citation = True``, what fraction are correct (top1_hit)?
* **missing_citations** — segments the case declared
  ``must_have_citation: true`` that the linker left at ⚠.

Aggregated per case and per segment_type (paragraph / bullet / table
row) for the "3-種別 どこで外しやすいか" report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text as sql_text

from app.database import get_search_db
from app.evals_citations.loader import CitationCase, SegmentExpectation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentScore:
    """Per-segment scoring result."""

    section_path: str
    segment_type: str
    top1_hit: bool
    recall_at_3: bool
    has_citation: bool
    must_have_citation: bool | None
    top_score: float
    top_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class CaseReport:
    """One case's scoring summary."""

    case_id: str
    file_path: str
    file_id: str | None
    segments: tuple[SegmentScore, ...]
    error: str | None = None


def load_detailed_summary(file_id: str) -> str | None:
    """Fetch the stored ``detailed_summary`` text for ``file_id``.

    Returns ``None`` when the row is absent or the column is empty —
    the runner reports this as "no detailed_summary available" rather
    than attempting to generate one on the fly. Generation cost
    (LLM call) would make the harness non-deterministic.
    """
    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT detailed_summary FROM file_summaries "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()
    if not row:
        return None
    md = row[0]
    if not md or not md.strip():
        return None
    return str(md)


def _hint_matches_chunk(
    hint_time: tuple[float, float] | None,
    hint_page: int | None,
    file_id: str,
    chunk_id: str,
) -> bool:
    """Return True when ``chunk_id`` overlaps the case's hint range.

    ``chunk_id`` has the UI form ``transcript:{idx}`` or
    ``document:{idx}``. For transcripts we resolve the timestamp from
    ``transcript_chunks``; for documents we read ``page`` from
    ``fts_text_content`` (the chunk_index there is a string which may
    or may not parse to int; we reuse the same tolerance).
    """
    if hint_time is None and hint_page is None:
        return False
    if ":" not in chunk_id:
        return False
    kind, _, raw_idx = chunk_id.partition(":")
    try:
        idx = int(raw_idx)
    except ValueError:
        return False
    with get_search_db() as session:
        if kind == "transcript" and hint_time is not None:
            row = session.execute(
                sql_text(
                    "SELECT timestamp_start, timestamp_end "
                    "FROM transcript_chunks "
                    "WHERE file_id = :fid AND chunk_index = :idx"
                ),
                {"fid": file_id, "idx": idx},
            ).fetchone()
            if not row:
                return False
            cs, ce = float(row[0] or 0), float(row[1] or 0)
            hs, he = hint_time
            # Overlap (inclusive on both ends) counts as a hit.
            return not (ce < hs or cs > he)
        if kind == "document" and hint_page is not None:
            row = session.execute(
                sql_text(
                    "SELECT page FROM fts_text_content "
                    "WHERE file_id = :fid AND chunk_index = :idx "
                    "LIMIT 1"
                ),
                {"fid": file_id, "idx": str(idx)},
            ).fetchone()
            if not row:
                return False
            try:
                page_num = int(row[0])
            except (TypeError, ValueError):
                return False
            return page_num == hint_page
    return False


def _chunk_matches_expectation(
    chunk_id: str,
    expectation: SegmentExpectation,
    file_id: str,
) -> bool:
    """Exact-id match OR hint-range overlap — either counts."""
    if chunk_id in expectation.chunk_ids:
        return True
    if expectation.hint is None:
        return False
    return _hint_matches_chunk(
        expectation.hint.time_range,
        expectation.hint.page,
        file_id,
        chunk_id,
    )


def resolve_file_id(file_path: str, drive: str) -> str | None:
    """Resolve a relative ``file_path`` under ``drive`` to its file_id.

    Matches the Ask harness convention: the snapshot's ``indexed_files``
    table has ``drive`` and ``relative_path`` columns, both used as a
    compound key.
    """
    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT file_id FROM indexed_files "
                "WHERE drive = :drive AND relative_path = :path"
            ),
            {"drive": drive, "path": file_path},
        ).fetchone()
    return str(row[0]) if row else None


def run_case(case: CitationCase, drive: str) -> CaseReport:
    """Score a single citation case end-to-end."""
    from app.citations import compute_citations  # lazy: heavy import chain

    file_id = resolve_file_id(case.file_path, drive)
    if file_id is None:
        return CaseReport(
            case_id=case.id,
            file_path=case.file_path,
            file_id=None,
            segments=(),
            error=f"file not found in snapshot for drive={drive!r}",
        )

    md = load_detailed_summary(file_id)
    if md is None:
        return CaseReport(
            case_id=case.id,
            file_path=case.file_path,
            file_id=file_id,
            segments=(),
            error="no detailed_summary stored for this file",
        )

    computed = compute_citations(file_id, md)
    by_path: dict[str, dict[str, Any]] = {c["section_path"]: c for c in computed}

    segment_scores: list[SegmentScore] = []
    for exp in case.expectations:
        produced = by_path.get(exp.section_path)
        if produced is None:
            # Linker didn't emit this section_path — treat as miss.
            segment_scores.append(
                SegmentScore(
                    section_path=exp.section_path,
                    segment_type="missing",
                    top1_hit=False,
                    recall_at_3=False,
                    has_citation=False,
                    must_have_citation=exp.must_have_citation,
                    top_score=0.0,
                    top_chunk_ids=(),
                )
            )
            continue
        chunk_ids: list[str] = list(produced.get("citation_chunk_ids") or [])
        top1 = (
            _chunk_matches_expectation(chunk_ids[0], exp, file_id)
            if chunk_ids
            else False
        )
        recall3 = any(
            _chunk_matches_expectation(cid, exp, file_id)
            for cid in chunk_ids[:3]
        )
        segment_scores.append(
            SegmentScore(
                section_path=exp.section_path,
                segment_type=str(produced.get("segment_type") or "?"),
                top1_hit=top1,
                recall_at_3=recall3,
                has_citation=bool(produced.get("has_citation")),
                must_have_citation=exp.must_have_citation,
                top_score=float(produced.get("top_score") or 0.0),
                top_chunk_ids=tuple(chunk_ids),
            )
        )

    return CaseReport(
        case_id=case.id,
        file_path=case.file_path,
        file_id=file_id,
        segments=tuple(segment_scores),
        error=None,
    )


@dataclass(frozen=True)
class AggregateMetrics:
    """Summary numbers across one or many cases."""

    total_segments: int
    top1_accuracy: float
    recall_at_3: float
    has_citation_precision: float
    missing_required_citations: int
    # segment_type → (top1_accuracy, recall_at_3, count)
    by_segment_type: dict[str, tuple[float, float, int]]


def aggregate(reports: list[CaseReport]) -> AggregateMetrics:
    """Roll per-segment scores into overall + per-segment_type metrics.

    ``has_citation_precision`` is computed only over segments the
    linker flagged ``has_citation = True``; it measures "when we said
    we had a source, how often was it right?" Returning 0.0 when the
    denominator is empty keeps the output stable across empty runs.
    """
    total = 0
    top1 = 0
    recall3 = 0
    has_cit = 0
    has_cit_correct = 0
    missing_required = 0
    by_type: dict[str, dict[str, int]] = {}

    for r in reports:
        for s in r.segments:
            total += 1
            stype = s.segment_type
            by_type.setdefault(stype, {"top1": 0, "recall3": 0, "n": 0})
            by_type[stype]["n"] += 1
            if s.top1_hit:
                top1 += 1
                by_type[stype]["top1"] += 1
            if s.recall_at_3:
                recall3 += 1
                by_type[stype]["recall3"] += 1
            if s.has_citation:
                has_cit += 1
                if s.top1_hit:
                    has_cit_correct += 1
            if s.must_have_citation and not s.has_citation:
                missing_required += 1

    def _ratio(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    by_segment_type = {
        k: (
            _ratio(v["top1"], v["n"]),
            _ratio(v["recall3"], v["n"]),
            v["n"],
        )
        for k, v in by_type.items()
    }

    return AggregateMetrics(
        total_segments=total,
        top1_accuracy=_ratio(top1, total),
        recall_at_3=_ratio(recall3, total),
        has_citation_precision=_ratio(has_cit_correct, has_cit),
        missing_required_citations=missing_required,
        by_segment_type=by_segment_type,
    )
