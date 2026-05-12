"""Pair comparison: baseline sidecar JSON vs current run (Phase E).

Consumes the JSON sidecar emitted alongside the md report. For each
case × metric we classify the delta as ``improved`` / ``regressed`` /
``tied`` using the run's epsilon threshold (spec §"ペア比較サマリー").

Only cases that appear in both sidecars are compared; cases added or
removed are reported separately.

Phase 1.D extension: aggregate-level rows for the agentic axes
(tool_call_count distribution, route_correctness, max context tokens,
forced answer rate) so an A/B report between legacy and agentic runs
can be read top-down.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

# Metric paths compared per case. Each entry is (label, stage_key, metric_key).
# stage_key indexes into the sidecar's case dict; metric_key picks a scalar
# (median for Stage 3 aggregates, direct for Stage 1/2).
_METRICS: tuple[tuple[str, str, str], ...] = (
    ("Stage 1: must_include_coverage", "stage1", "must_include_coverage"),
    ("Stage 1: must_exclude_violations", "stage1", "must_exclude_violations"),
    ("Stage 2: recall@5 (file)", "stage2", "file_recall_at_5"),
    ("Stage 2: recall@10 (file)", "stage2", "file_recall_at_10"),
    ("Stage 2: segment recall@5", "stage2", "segment_recall_at_5"),
    ("Stage 2: MRR", "stage2", "mrr"),
    ("Stage 3: must_mention (median)", "stage3", "must_mention_coverage_median"),
    ("Stage 3: citation_in_ground_truth (median)", "stage3", "citation_in_ground_truth_median"),
    ("Stage 3: citation_segment_match (median)", "stage3", "citation_segment_match_median"),
    ("Stage 3: citation_in_retrieved (median)", "stage3", "citation_in_retrieved_median"),
)


@dataclass(frozen=True)
class MetricDelta:
    label: str
    improved: int
    regressed: int
    tied: int


@dataclass(frozen=True)
class AgenticAxisSummary:
    """Aggregate-level summary for an agentic-only axis on one side.

    Values are ``None`` when no run on that side produced the axis
    (e.g. a legacy-only run will report ``None`` for tool_call_count).
    """

    label: str
    baseline_median: float | None
    current_median: float | None
    baseline_p95: float | None = None
    current_p95: float | None = None


@dataclass(frozen=True)
class ComparisonResult:
    baseline_label: str
    baseline_path: str
    common_case_ids: tuple[str, ...]
    only_in_baseline: tuple[str, ...]
    only_in_current: tuple[str, ...]
    deltas: tuple[MetricDelta, ...]
    agentic_axes: tuple[AgenticAxisSummary, ...] = ()


def _load_sidecar(path: Path) -> dict:
    """Load a sidecar JSON; if a .md path is passed, try the .json twin."""
    if path.suffix == ".md":
        twin = path.with_suffix(".json")
        if twin.exists():
            path = twin
        else:
            raise FileNotFoundError(
                f"Baseline comparison requires a JSON sidecar. "
                f"Neither {path} nor {twin} was loadable."
            )
    return json.loads(path.read_text(encoding="utf-8"))


def _pick(case: dict, stage_key: str, metric_key: str) -> float | None:
    stage = case.get(stage_key)
    if not isinstance(stage, dict):
        return None
    v = stage.get(metric_key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _classify(baseline: float, current: float, epsilon: float, lower_is_better: bool) -> str:
    delta = current - baseline
    if abs(delta) <= epsilon:
        return "tied"
    improved = delta < 0 if lower_is_better else delta > 0
    return "improved" if improved else "regressed"


def _collect_agentic_axis(
    cases: list[dict], axis_key: str
) -> list[float]:
    """Pull the agentic ``axis_key`` from every run across all cases.

    Each case's ``stage3.runs`` may contain an ``agentic`` dict written
    by ``report.build_sidecar`` (Phase 1.A). Non-agentic runs are
    skipped so legacy reports collapse to an empty list.
    """
    values: list[float] = []
    for case in cases:
        s3 = case.get("stage3")
        if not isinstance(s3, dict):
            continue
        for run in s3.get("runs", []) or []:
            agentic = (run or {}).get("agentic")
            if not isinstance(agentic, dict):
                continue
            v = agentic.get(axis_key)
            if v is None:
                continue
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                continue
    return values


def _percentile(values: list[float], pct: float) -> float | None:
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


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _agentic_axes_summary(
    baseline: dict, current: dict
) -> tuple[AgenticAxisSummary, ...]:
    """Build aggregate-level summaries for the agentic axes."""
    base_cases = baseline.get("cases", [])
    cur_cases = current.get("cases", [])

    tcc_base = _collect_agentic_axis(base_cases, "tool_call_count")
    tcc_cur = _collect_agentic_axis(cur_cases, "tool_call_count")
    rc_base = _collect_agentic_axis(base_cases, "route_correctness")
    rc_cur = _collect_agentic_axis(cur_cases, "route_correctness")
    ctx_base = _collect_agentic_axis(base_cases, "max_context_tokens_used")
    ctx_cur = _collect_agentic_axis(cur_cases, "max_context_tokens_used")
    forced_base = _collect_agentic_axis(base_cases, "forced_answer")
    forced_cur = _collect_agentic_axis(cur_cases, "forced_answer")

    return (
        AgenticAxisSummary(
            label="Agentic: tool_call_count",
            baseline_median=_median(tcc_base),
            current_median=_median(tcc_cur),
            baseline_p95=_percentile(tcc_base, 0.95),
            current_p95=_percentile(tcc_cur, 0.95),
        ),
        AgenticAxisSummary(
            label="Agentic: route_correctness",
            baseline_median=_median(rc_base),
            current_median=_median(rc_cur),
        ),
        AgenticAxisSummary(
            label="Agentic: max_context_tokens_used",
            baseline_median=_median(ctx_base),
            current_median=_median(ctx_cur),
            baseline_p95=_percentile(ctx_base, 0.95),
            current_p95=_percentile(ctx_cur, 0.95),
        ),
        AgenticAxisSummary(
            label="Agentic: forced_answer_rate",
            baseline_median=(
                sum(forced_base) / len(forced_base) if forced_base else None
            ),
            current_median=(
                sum(forced_cur) / len(forced_cur) if forced_cur else None
            ),
        ),
    )


def compare_sidecars(
    baseline_path: Path, current_sidecar: dict, epsilon: float
) -> ComparisonResult:
    """Compare two sidecar JSON payloads. ``current_sidecar`` is in-memory."""
    baseline = _load_sidecar(baseline_path)
    base_cases = {c["id"]: c for c in baseline.get("cases", [])}
    cur_cases = {c["id"]: c for c in current_sidecar.get("cases", [])}

    common = sorted(set(base_cases) & set(cur_cases))
    only_base = sorted(set(base_cases) - set(cur_cases))
    only_cur = sorted(set(cur_cases) - set(base_cases))

    deltas: list[MetricDelta] = []
    for label, stage_key, metric_key in _METRICS:
        lower_is_better = "violations" in metric_key
        imp = reg = tie = 0
        for cid in common:
            b = _pick(base_cases[cid], stage_key, metric_key)
            c = _pick(cur_cases[cid], stage_key, metric_key)
            if b is None or c is None:
                continue  # skip null segment_match etc.
            result = _classify(b, c, epsilon, lower_is_better)
            if result == "improved":
                imp += 1
            elif result == "regressed":
                reg += 1
            else:
                tie += 1
        deltas.append(MetricDelta(label=label, improved=imp, regressed=reg, tied=tie))

    return ComparisonResult(
        baseline_label=str(baseline.get("meta", {}).get("label", "")) or baseline_path.name,
        baseline_path=str(baseline_path),
        common_case_ids=tuple(common),
        only_in_baseline=tuple(only_base),
        only_in_current=tuple(only_cur),
        deltas=tuple(deltas),
        agentic_axes=_agentic_axes_summary(baseline, current_sidecar),
    )


def _fmt_optional(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def render_comparison_md(result: ComparisonResult) -> str:
    """Markdown section to append to the current report."""
    lines: list[str] = []
    label = result.baseline_label or result.baseline_path
    lines.append(f"## Pair comparison vs {label}\n")
    lines.append(f"- baseline: `{result.baseline_path}`")
    lines.append(f"- common cases: {len(result.common_case_ids)}")
    if result.only_in_baseline:
        lines.append(f"- only in baseline: {list(result.only_in_baseline)}")
    if result.only_in_current:
        lines.append(f"- only in current: {list(result.only_in_current)}")
    lines.append("")
    lines.append("| stage / metric | improved | regressed | tied |")
    lines.append("|---|---|---|---|")
    for d in result.deltas:
        lines.append(f"| {d.label} | {d.improved} | {d.regressed} | {d.tied} |")
    lines.append("")

    if result.agentic_axes:
        lines.append("### Agentic-loop axes (aggregate)\n")
        lines.append(
            "| axis | baseline (median / 95p) | current (median / 95p) |"
        )
        lines.append("|---|---|---|")
        for axis in result.agentic_axes:
            base = (
                f"{_fmt_optional(axis.baseline_median)} / "
                f"{_fmt_optional(axis.baseline_p95)}"
                if axis.baseline_p95 is not None
                else _fmt_optional(axis.baseline_median)
            )
            cur = (
                f"{_fmt_optional(axis.current_median)} / "
                f"{_fmt_optional(axis.current_p95)}"
                if axis.current_p95 is not None
                else _fmt_optional(axis.current_median)
            )
            lines.append(f"| {axis.label} | {base} | {cur} |")
        lines.append("")

    return "\n".join(lines)
