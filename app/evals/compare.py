"""Pair comparison: baseline sidecar JSON vs current run (Phase E).

Consumes the JSON sidecar emitted alongside the md report. For each
case × metric we classify the delta as ``improved`` / ``regressed`` /
``tied`` using the run's epsilon threshold (spec §"ペア比較サマリー").

Only cases that appear in both sidecars are compared; cases added or
removed are reported separately.
"""

from __future__ import annotations

import json
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
class ComparisonResult:
    baseline_label: str
    baseline_path: str
    common_case_ids: tuple[str, ...]
    only_in_baseline: tuple[str, ...]
    only_in_current: tuple[str, ...]
    deltas: tuple[MetricDelta, ...]


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
    )


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
    return "\n".join(lines)
