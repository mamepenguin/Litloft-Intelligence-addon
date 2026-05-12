"""Markdown report writer for eval runs.

The report is meant to be git-committed and diffed across baseline /
candidate runs (Phase E will add the side-by-side comparator). Aim
for a layout that reads top-down: meta → aggregate → per-case →
failures → raw appendix.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.evals.agentic_metrics import forced_answer_rate as _forced_answer_rate
from app.evals.stages import CaseReport, Stage3SingleRun


@dataclass(frozen=True)
class ReportMeta:
    label: str
    git_commit: str
    llm_model: str
    llm_base_url: str
    rag_top_k: int
    rag_max_tokens: int
    snapshot_path: str
    snapshot_sha256: str
    indexed_with: dict[str, str]
    runs_stage1: int
    runs_stage2: int
    runs_stage3: int
    epsilon: float
    drive: str
    blocklists: dict[str, str] = field(default_factory=dict)


def _fmt_pct(x: float) -> str:
    return f"{x:.2f}"


def _fmt_optional_pct(x: float | None) -> str:
    return "N/A" if x is None else f"{x:.2f}"


def _fmt_optional_int(x: int | None) -> str:
    return "N/A" if x is None else str(x)


def _all_runs(case_reports: list[CaseReport]) -> list[Stage3SingleRun]:
    return [r for c in case_reports for r in c.stage3.runs]


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile_or_none(values: list[float], pct: float) -> float | None:
    """Naive percentile (linear interpolation). Returns None when empty."""
    if not values:
        return None
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def _tool_call_counts(case_reports: list[CaseReport]) -> list[float]:
    return [
        float(r.tool_call_count)
        for r in _all_runs(case_reports)
        if r.tool_call_count is not None
    ]


def _route_correctness_values(case_reports: list[CaseReport]) -> list[float]:
    return [
        r.route_correctness
        for r in _all_runs(case_reports)
        if r.route_correctness is not None
    ]


def _max_context_tokens(case_reports: list[CaseReport]) -> list[float]:
    return [
        float(r.max_context_tokens_used)
        for r in _all_runs(case_reports)
        if r.max_context_tokens_used is not None
    ]


def _agg_value(case_reports: list[CaseReport], pick) -> float:
    vals = [pick(c) for c in case_reports]
    if not vals:
        return 0.0
    return statistics.median(vals)


def _stage3_median_of_medians(
    case_reports: list[CaseReport], attr: str
) -> float:
    # Skip cases whose metric has no observed values (e.g. citation_segment_match
    # is N/A when the LLM did not cite any segment-hinted GT file). Treating
    # those as 0.0 dragged the outer median down, so a perfect run on hinted
    # cases looked like 0.50 when half the cases were N/A.
    vals = [
        getattr(c.stage3, attr).median
        for c in case_reports
        if getattr(c.stage3, attr).values
    ]
    if not vals:
        return 0.0
    return statistics.median(vals)


def _flags(c: CaseReport) -> str:
    flags: list[str] = []
    if (
        c.stage3.must_mention_coverage.unstable
        or c.stage3.citation_in_ground_truth.unstable
        or c.stage3.citation_segment_match.unstable
        or c.stage3.citation_in_retrieved.unstable
    ):
        flags.append("unstable")
    if c.stage2.file_recall_at_5 < 1.0:
        flags.append("recall<1")
    if c.stage2.segment_recall_at_5 < 1.0:
        flags.append("seg-recall<1")
    return ",".join(flags)


def _is_failure(c: CaseReport) -> bool:
    return (
        c.stage2.file_recall_at_5 < 1.0
        or c.stage2.segment_recall_at_5 < 1.0
        or c.stage3.must_mention_coverage.unstable
        or c.stage3.citation_in_ground_truth.unstable
    )


def _runs_to_yaml(c: CaseReport) -> str:
    payload: dict[str, Any] = {
        "case": c.case.id,
        "query": c.case.query,
        "stage1": {
            "keywords": c.stage1.keywords,
            "must_include_coverage": c.stage1.must_include_coverage,
            "must_exclude_violations": c.stage1.must_exclude_violations,
        },
        "stage2": {
            "top_file_ids": list(c.stage2.top_file_ids),
            "file_recall@5": c.stage2.file_recall_at_5,
            "file_recall@10": c.stage2.file_recall_at_10,
            "segment_recall@5": c.stage2.segment_recall_at_5,
            "segment_recall@10": c.stage2.segment_recall_at_10,
            "mrr": c.stage2.mrr,
            "precision@5": c.stage2.precision_at_5,
        },
        "stage3_runs": [
            {
                "answer": (r.answer[:280] + "…")
                if r.answer and len(r.answer) > 280
                else r.answer,
                "citation_file_ids": list(r.citation_file_ids),
                "must_mention_coverage": r.must_mention_coverage,
                "citation_in_ground_truth": r.citation_in_ground_truth,
                "citation_segment_match": r.citation_segment_match,
                "citation_in_retrieved": r.citation_in_retrieved,
                "took_ms": r.took_ms,
            }
            for r in c.stage3.runs
        ],
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def render(case_reports: list[CaseReport], meta: ReportMeta) -> str:
    """Render a full markdown report string."""
    lines: list[str] = []
    title = f"Eval Report: {meta.label}" if meta.label else "Eval Report"
    lines.append(f"# {title}\n")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(f"- date: {now}")
    lines.append(f"- git_commit: {meta.git_commit}")
    lines.append(f"- llm_model: {meta.llm_model}")
    lines.append(f"- llm_base_url: {meta.llm_base_url}")
    lines.append("- llm_temperature: 0 (forced by runner)")
    lines.append("- search_config:")
    lines.append(f"    rag.top_k: {meta.rag_top_k}")
    lines.append(f"    rag.max_tokens: {meta.rag_max_tokens}")
    lines.append("    search.mode: recall")
    lines.append("- index_snapshot:")
    lines.append(f"    file: {meta.snapshot_path}")
    lines.append(f"    sha256: {meta.snapshot_sha256}")
    lines.append("    indexed_with:")
    for k, v in meta.indexed_with.items():
        lines.append(f"      {k}: {v!r}")
    if meta.blocklists:
        lines.append("- blocklists:")
        for k, v in meta.blocklists.items():
            lines.append(f"    {k}: {v}")
    else:
        lines.append("- blocklists: (none loaded)")
    lines.append(
        f"- runs: stage1={meta.runs_stage1} "
        f"stage2={meta.runs_stage2} stage3={meta.runs_stage3}"
    )
    lines.append(f"- epsilon: {meta.epsilon}")
    lines.append(f"- drive: {meta.drive}")
    lines.append(f"- total_cases: {len(case_reports)}")
    lines.append("")

    # Aggregate ----------------------------------------------------------
    lines.append("## Aggregate\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(
        "| Stage 1: must_include_coverage (median) | "
        f"{_fmt_pct(_agg_value(case_reports, lambda c: c.stage1.must_include_coverage))} |"
    )
    total_violations = sum(c.stage1.must_exclude_violations for c in case_reports)
    lines.append(
        f"| Stage 1: must_exclude_violations (sum) | {total_violations} |"
    )
    lines.append(
        "| Stage 2: file recall@5 (median) | "
        f"{_fmt_pct(_agg_value(case_reports, lambda c: c.stage2.file_recall_at_5))} |"
    )
    lines.append(
        "| Stage 2: file recall@10 (median) | "
        f"{_fmt_pct(_agg_value(case_reports, lambda c: c.stage2.file_recall_at_10))} |"
    )
    lines.append(
        "| Stage 2: segment recall@5 (median) | "
        f"{_fmt_pct(_agg_value(case_reports, lambda c: c.stage2.segment_recall_at_5))} |"
    )
    lines.append(
        "| Stage 2: MRR (median) | "
        f"{_fmt_pct(_agg_value(case_reports, lambda c: c.stage2.mrr))} |"
    )
    lines.append(
        "| Stage 3: must_mention_coverage (median) | "
        f"{_fmt_pct(_stage3_median_of_medians(case_reports, 'must_mention_coverage'))} |"
    )
    lines.append(
        "| Stage 3: citation_in_ground_truth (median) | "
        f"{_fmt_pct(_stage3_median_of_medians(case_reports, 'citation_in_ground_truth'))} |"
    )
    lines.append(
        "| Stage 3: citation_segment_match (median) | "
        f"{_fmt_pct(_stage3_median_of_medians(case_reports, 'citation_segment_match'))} |"
    )
    lines.append(
        "| Stage 3: citation_in_retrieved (median) | "
        f"{_fmt_pct(_stage3_median_of_medians(case_reports, 'citation_in_retrieved'))} |"
    )

    # Agentic-loop axes (Phase 1.A). Each row reports N/A unless at
    # least one run produced telemetry — legacy A/B reports show N/A,
    # agentic A/B reports show real numbers.
    tcc = _tool_call_counts(case_reports)
    lines.append(
        "| Agentic: tool_call_count (median) | "
        f"{_fmt_optional_pct(_median_or_none(tcc))} |"
    )
    lines.append(
        "| Agentic: tool_call_count (95p) | "
        f"{_fmt_optional_pct(_percentile_or_none(tcc, 0.95))} |"
    )
    rc = _route_correctness_values(case_reports)
    lines.append(
        "| Agentic: route_correctness (median) | "
        f"{_fmt_optional_pct(_median_or_none(rc))} |"
    )
    mct = _max_context_tokens(case_reports)
    lines.append(
        "| Agentic: max_context_tokens_used (median) | "
        f"{_fmt_optional_pct(_median_or_none(mct))} |"
    )
    lines.append(
        "| Agentic: max_context_tokens_used (95p) | "
        f"{_fmt_optional_pct(_percentile_or_none(mct, 0.95))} |"
    )
    telemetries = [r.agentic_telemetry for r in _all_runs(case_reports)]
    lines.append(
        "| Agentic: forced_answer_rate | "
        f"{_fmt_optional_pct(_forced_answer_rate(telemetries))} |"
    )
    lines.append("")

    # Per-case -----------------------------------------------------------
    lines.append("## Per-case summary\n")
    lines.append(
        "| id | recall@5 | seg-recall@5 | MRR | must_mention | citations "
        "| tools | route | flags |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in case_reports:
        per_case_tcc = [
            float(r.tool_call_count)
            for r in c.stage3.runs
            if r.tool_call_count is not None
        ]
        per_case_route = [
            r.route_correctness
            for r in c.stage3.runs
            if r.route_correctness is not None
        ]
        lines.append(
            "| {id} | {r5} | {sr5} | {mrr} | {mm} | {cit} | {tools} | "
            "{route} | {flags} |".format(
                id=c.case.id,
                r5=_fmt_pct(c.stage2.file_recall_at_5),
                sr5=_fmt_pct(c.stage2.segment_recall_at_5),
                mrr=_fmt_pct(c.stage2.mrr),
                mm=_fmt_pct(c.stage3.must_mention_coverage.median),
                cit=_fmt_pct(c.stage3.citation_in_ground_truth.median),
                tools=_fmt_optional_pct(_median_or_none(per_case_tcc)),
                route=_fmt_optional_pct(_median_or_none(per_case_route)),
                flags=_flags(c) or "-",
            )
        )
    lines.append("")

    # Failures -----------------------------------------------------------
    failures = [c for c in case_reports if _is_failure(c)]
    lines.append("## Failures\n")
    if not failures:
        lines.append("(no failures)\n")
    else:
        for c in failures:
            lines.append(f"### {c.case.id}\n")
            lines.append(f"- query: {c.case.query!r}")
            lines.append(f"- keywords: {c.stage1.keywords!r}")
            gt_ids = {gt.file_id for gt in c.resolved if gt.file_id}
            top10 = list(c.stage2.top_file_ids[:10])
            marked = [
                ("✓" if fid in gt_ids else "✗") + fid for fid in top10
            ]
            lines.append(f"- top_10: {marked}")
            missing = [
                gt.path for gt in c.resolved
                if gt.file_id and gt.file_id not in c.stage2.top_file_ids[:10]
            ]
            if missing:
                lines.append(f"- ground_truth not retrieved: {missing}")
            lines.append("- Stage 3 must_mention values: "
                         f"{list(c.stage3.must_mention_coverage.values)}")
            lines.append("")

    # Appendix -----------------------------------------------------------
    lines.append("<details>")
    lines.append("<summary>Full appendix (raw runs)</summary>\n")
    lines.append("```yaml")
    for c in case_reports:
        lines.append(_runs_to_yaml(c))
        lines.append("---")
    lines.append("```")
    lines.append("</details>")
    lines.append("")

    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# --------------------------------------------------------------------------- #
# JSON sidecar (Phase E)
# --------------------------------------------------------------------------- #


def build_sidecar(case_reports: list[CaseReport], meta: ReportMeta) -> dict:
    """Parse-free structured payload for diff / --baseline comparison.

    Schema is flat per-case with scalar metrics so ``compare.py`` does not
    need to walk nested aggregates. Stage 3 aggregates expose ``*_median``
    and the full ``runs[]`` list for debugging.
    """
    cases_out: list[dict] = []
    for c in case_reports:
        s3 = c.stage3
        cases_out.append({
            "id": c.case.id,
            "query": c.case.query,
            "stage1": {
                "keywords": c.stage1.keywords,
                "must_include_coverage": c.stage1.must_include_coverage,
                "must_exclude_violations": c.stage1.must_exclude_violations,
            },
            "stage2": {
                "top_file_ids": list(c.stage2.top_file_ids),
                "file_recall_at_5": c.stage2.file_recall_at_5,
                "file_recall_at_10": c.stage2.file_recall_at_10,
                "segment_recall_at_5": c.stage2.segment_recall_at_5,
                "segment_recall_at_10": c.stage2.segment_recall_at_10,
                "mrr": c.stage2.mrr,
                "precision_at_5": c.stage2.precision_at_5,
                "retrieved_gt_file_ids": list(c.stage2.retrieved_gt_file_ids),
            },
            "stage3": {
                "runs": [
                    {
                        "answer": r.answer,
                        "citation_file_ids": list(r.citation_file_ids),
                        "retrieved_file_ids": list(r.retrieved_file_ids),
                        "must_mention_coverage": r.must_mention_coverage,
                        "citation_in_ground_truth": r.citation_in_ground_truth,
                        "citation_segment_match": r.citation_segment_match,
                        "citation_in_retrieved": r.citation_in_retrieved,
                        "took_ms": r.took_ms,
                        "agentic": (
                            None
                            if r.agentic_telemetry is None
                            else {
                                "tool_calls": list(r.agentic_telemetry.tool_calls),
                                "tool_call_count": r.tool_call_count,
                                "route_correctness": r.route_correctness,
                                "max_context_tokens_used": (
                                    r.max_context_tokens_used
                                ),
                                "forced_answer": r.forced_answer,
                                "citation_retries": (
                                    r.agentic_telemetry.citation_retries
                                ),
                            }
                        ),
                    }
                    for r in s3.runs
                ],
                "must_mention_coverage_median": s3.must_mention_coverage.median,
                "must_mention_coverage_unstable": s3.must_mention_coverage.unstable,
                "citation_in_ground_truth_median": s3.citation_in_ground_truth.median,
                "citation_in_ground_truth_unstable": s3.citation_in_ground_truth.unstable,
                "citation_segment_match_median": s3.citation_segment_match.median,
                "citation_segment_match_unstable": s3.citation_segment_match.unstable,
                "citation_in_retrieved_median": s3.citation_in_retrieved.median,
                "citation_in_retrieved_unstable": s3.citation_in_retrieved.unstable,
            },
        })

    return {
        "schema_version": 1,
        "meta": {
            "label": meta.label,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": meta.git_commit,
            "llm_model": meta.llm_model,
            "llm_base_url": meta.llm_base_url,
            "rag_top_k": meta.rag_top_k,
            "rag_max_tokens": meta.rag_max_tokens,
            "snapshot_path": meta.snapshot_path,
            "snapshot_sha256": meta.snapshot_sha256,
            "indexed_with": meta.indexed_with,
            "runs_stage1": meta.runs_stage1,
            "runs_stage2": meta.runs_stage2,
            "runs_stage3": meta.runs_stage3,
            "epsilon": meta.epsilon,
            "drive": meta.drive,
            "blocklists": meta.blocklists,
        },
        "cases": cases_out,
    }


def write_json_sidecar(path: Path, sidecar: dict) -> None:
    """Write the sidecar JSON. ``path`` is the md report path; we swap suffix."""
    json_path = path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
