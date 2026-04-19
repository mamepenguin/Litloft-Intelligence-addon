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
    """Per-segment scoring result.

    ``offset_at_top1`` is the chunk-index distance between the system's
    top-1 chunk and the nearest ground-truth chunk (0 = exact hit,
    1-2 = adjacent / near-miss, 5+ = different part of the file).
    ``None`` when no ground truth is available or the top-1 chunk kind
    doesn't match the GT kind (e.g. top-1 is ``transcript:*`` but GT
    is ``document:*``). This is the primary location-precision signal;
    ``top1_hit`` is just the ``offset == 0`` special case.
    """

    section_path: str
    segment_type: str
    top1_hit: bool
    recall_at_3: bool
    has_citation: bool
    must_have_citation: bool | None
    top_score: float
    top_chunk_ids: tuple[str, ...]
    offset_at_top1: int | None = None
    gt_chunk_count: int = 0


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


def _parse_chunk_id(chunk_id: str) -> tuple[str, int] | None:
    """Return ``(kind, idx)`` for a ``transcript:N`` / ``document:N`` id."""
    if ":" not in chunk_id:
        return None
    kind, _, raw = chunk_id.partition(":")
    try:
        return (kind, int(raw))
    except ValueError:
        return None


def _ground_truth_chunks(
    expectation: SegmentExpectation, file_id: str
) -> list[tuple[str, int]]:
    """Return the GT chunk set for offset computation.

    Priority:

    1. Explicit ``chunk_ids`` — parsed as ``(kind, idx)`` pairs.
    2. ``segment_hint.time_range`` — expand to every transcript chunk
       whose interval overlaps the range. This lets a case author pin
       a section-sized range without knowing the exact chunk numbers
       (and the eval still reports offset from the nearest member).
    3. ``segment_hint.page`` — expand to every ``document`` chunk at
       that page (fts_text_content stores chunk_index as string).

    Returned list may contain duplicates from multiple sources; callers
    should only care about the min offset anyway.
    """
    out: list[tuple[str, int]] = []
    for cid in expectation.chunk_ids:
        p = _parse_chunk_id(cid)
        if p is not None:
            out.append(p)
    if out:
        return out
    if expectation.hint is None:
        return out
    with get_search_db() as session:
        if expectation.hint.time_range is not None:
            hs, he = expectation.hint.time_range
            rows = session.execute(
                sql_text(
                    "SELECT chunk_index FROM transcript_chunks "
                    "WHERE file_id = :fid "
                    "  AND NOT (timestamp_end < :hs OR timestamp_start > :he)"
                ),
                {"fid": file_id, "hs": hs, "he": he},
            ).fetchall()
            out.extend(("transcript", int(r[0])) for r in rows)
        if expectation.hint.page is not None:
            rows = session.execute(
                sql_text(
                    "SELECT DISTINCT chunk_index FROM fts_text_content "
                    "WHERE file_id = :fid AND page = :pg"
                ),
                {"fid": file_id, "pg": str(expectation.hint.page)},
            ).fetchall()
            for r in rows:
                try:
                    out.append(("document", int(r[0])))
                except (TypeError, ValueError):
                    continue
    return out


def _offset_at_top1(
    top_chunk_ids: list[str], gt_set: list[tuple[str, int]]
) -> int | None:
    """Compute chunk-index distance from top-1 to nearest GT chunk.

    ``None`` when there is no top-1 chunk, or when the top-1's kind
    (``transcript``/``document``) has no entries in the GT set. The
    eval aggregate drops None offsets so mixing chunk kinds doesn't
    silently pollute the statistics.
    """
    if not top_chunk_ids:
        return None
    top1_parsed = _parse_chunk_id(top_chunk_ids[0])
    if top1_parsed is None:
        return None
    kind, idx = top1_parsed
    same_kind = [i for (k, i) in gt_set if k == kind]
    if not same_kind:
        return None
    return min(abs(idx - g) for g in same_kind)


def resolve_file_id(file_path: str, drive: str) -> str | None:
    """Resolve a case-relative ``file_path`` under ``drive`` to a file_id.

    ``indexed_files.file_path`` stores absolute paths (e.g.
    ``/drives/default/YouTube/foo.mp4``). The mount prefix is env-only
    (``DRIVE_MOUNTS`` in the intelligence container), so the drive name
    and the prefix don't always coincide. We try, in order:

    1. Exact match against ``file_path`` (caller passes the absolute path).
    2. Match against ``/drives/{drive}/{file_path}`` (the conventional
       ``/drives/{drive}/`` prefix when drive name = mount leaf).
    3. Path-boundary suffix match (``%/{file_path}``) — tolerates any
       mount prefix as long as the tail uniquely identifies the file.
       Ambiguous suffix matches return ``None`` so the case fails
       loudly rather than silently picking one.
    """
    rel = file_path.lstrip("/")
    with get_search_db() as session:
        for cand in (file_path, f"/drives/{drive}/{rel}"):
            row = session.execute(
                sql_text(
                    "SELECT file_id FROM indexed_files "
                    "WHERE drive = :drive AND file_path = :path"
                ),
                {"drive": drive, "path": cand},
            ).fetchone()
            if row:
                return str(row[0])
        rows = session.execute(
            sql_text(
                "SELECT file_id FROM indexed_files "
                "WHERE drive = :drive AND file_path LIKE :suffix"
            ),
            {"drive": drive, "suffix": f"%/{rel}"},
        ).fetchall()
    if len(rows) == 1:
        return str(rows[0][0])
    if len(rows) > 1:
        logger.warning(
            "resolve_file_id: ambiguous suffix %r in drive %r (%d matches)",
            rel,
            drive,
            len(rows),
        )
    return None


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
        gt_set = _ground_truth_chunks(exp, file_id)
        offset = _offset_at_top1(chunk_ids, gt_set)
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
                offset_at_top1=offset,
                gt_chunk_count=len(gt_set),
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
    """Summary numbers across one or many cases.

    ``top1_accuracy`` / ``recall_at_3`` are the legacy binary metrics.
    The location-precision picture lives in ``offset_*`` and
    ``hit_at_offset_*`` — a continuous view that shows whether the
    system's top-1 lands on the exact chunk, the neighbour, or the
    wrong side of the file. ``calibration_by_score_band`` lets a
    reader check whether higher ``top_score`` actually corresponds to
    lower offset — if not, the 2-state citation UI (citation vs ⚠) is
    discarding information we should be using.
    """

    total_segments: int
    top1_accuracy: float
    recall_at_3: float
    has_citation_precision: float
    missing_required_citations: int
    # segment_type → (top1_accuracy, recall_at_3, count)
    by_segment_type: dict[str, tuple[float, float, int]]
    # Offset distribution (only over segments with a computable offset)
    n_with_offset: int
    p50_offset: float
    p95_offset: float
    max_offset: int
    mean_offset: float
    hit_at_offset: dict[int, float]  # {0, 1, 2, 5} → ratio in ≤ N
    # score band → {n, mean_offset, hit_at_0, median_offset}
    calibration_by_score_band: dict[str, dict[str, float]]


_OFFSET_HIT_THRESHOLDS = (0, 1, 2, 5)
_SCORE_BANDS = (
    (0.00, 0.70, "<0.70"),
    (0.70, 0.80, "[0.70-0.80)"),
    (0.80, 0.85, "[0.80-0.85)"),
    (0.85, 0.90, "[0.85-0.90)"),
    (0.90, 1.01, "≥0.90"),
)


def _percentile(xs: list[int], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 100]). 0.0 on empty."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (q / 100.0) * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def aggregate(reports: list[CaseReport]) -> AggregateMetrics:
    """Roll per-segment scores into overall + offset + calibration metrics.

    ``has_citation_precision`` counts only ``has_citation=True`` segments
    and asks "when we said we had a source, was it the right one?"
    Offset-based metrics are the primary measure — they capture "how
    wrong" rather than just hit/miss. The score-band calibration table
    shows whether the system's own confidence signal is usable for
    thresholding / multi-state UI.
    """
    total = 0
    top1 = 0
    recall3 = 0
    has_cit = 0
    has_cit_correct = 0
    missing_required = 0
    by_type: dict[str, dict[str, int]] = {}
    offsets: list[int] = []
    band_accum: dict[str, dict[str, object]] = {
        label: {"offsets": [], "hit0": 0} for _, _, label in _SCORE_BANDS
    }

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
            if s.offset_at_top1 is not None:
                offsets.append(s.offset_at_top1)
                # Score-band calibration uses the system's top_score.
                for lo, hi, label in _SCORE_BANDS:
                    if lo <= s.top_score < hi:
                        band_accum[label]["offsets"].append(s.offset_at_top1)
                        if s.offset_at_top1 == 0:
                            band_accum[label]["hit0"] = (
                                int(band_accum[label]["hit0"]) + 1
                            )
                        break

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

    hit_at_offset = {
        n: _ratio(sum(1 for o in offsets if o <= n), len(offsets))
        for n in _OFFSET_HIT_THRESHOLDS
    }
    calibration: dict[str, dict[str, float]] = {}
    for label, entry in band_accum.items():
        band_offsets: list[int] = entry["offsets"]  # type: ignore[assignment]
        n = len(band_offsets)
        if n == 0:
            calibration[label] = {
                "n": 0,
                "mean_offset": 0.0,
                "median_offset": 0.0,
                "hit_at_0": 0.0,
            }
            continue
        calibration[label] = {
            "n": n,
            "mean_offset": sum(band_offsets) / n,
            "median_offset": _percentile(band_offsets, 50),
            "hit_at_0": _ratio(int(entry["hit0"]), n),
        }

    return AggregateMetrics(
        total_segments=total,
        top1_accuracy=_ratio(top1, total),
        recall_at_3=_ratio(recall3, total),
        has_citation_precision=_ratio(has_cit_correct, has_cit),
        missing_required_citations=missing_required,
        by_segment_type=by_segment_type,
        n_with_offset=len(offsets),
        p50_offset=_percentile(offsets, 50),
        p95_offset=_percentile(offsets, 95),
        max_offset=max(offsets) if offsets else 0,
        mean_offset=(sum(offsets) / len(offsets)) if offsets else 0.0,
        hit_at_offset=hit_at_offset,
        calibration_by_score_band=calibration,
    )
