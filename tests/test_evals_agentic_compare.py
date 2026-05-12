"""Tests for the Phase 1.D agentic-axis aggregations in compare.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest  # noqa: E402

from app.evals.compare import (  # noqa: E402
    compare_sidecars,
    render_comparison_md,
)


def _case_with_agentic(
    case_id: str,
    *,
    tool_call_count: int | None = 2,
    route_correctness: float | None = 1.0,
    max_context_tokens_used: int | None = 5000,
    forced_answer: bool | None = False,
) -> dict:
    agentic = (
        None
        if tool_call_count is None
        else {
            "tool_calls": ["search_files", "get_file_chunks"][:tool_call_count],
            "tool_call_count": tool_call_count,
            "route_correctness": route_correctness,
            "max_context_tokens_used": max_context_tokens_used,
            "forced_answer": forced_answer,
        }
    )
    return {
        "id": case_id,
        "query": "q",
        "stage1": {
            "keywords": "k",
            "must_include_coverage": 1.0,
            "must_exclude_violations": 0,
        },
        "stage2": {
            "top_file_ids": [],
            "file_recall_at_5": 1.0,
            "file_recall_at_10": 1.0,
            "segment_recall_at_5": 1.0,
            "segment_recall_at_10": 1.0,
            "mrr": 1.0,
            "precision_at_5": 0.5,
            "retrieved_gt_file_ids": [],
        },
        "stage3": {
            "runs": [
                {
                    "answer": "a",
                    "citation_file_ids": [],
                    "retrieved_file_ids": [],
                    "must_mention_coverage": 1.0,
                    "citation_in_ground_truth": 1.0,
                    "citation_segment_match": None,
                    "citation_in_retrieved": 1.0,
                    "took_ms": 100,
                    "agentic": agentic,
                }
            ],
            "must_mention_coverage_median": 1.0,
            "must_mention_coverage_unstable": False,
            "citation_in_ground_truth_median": 1.0,
            "citation_in_ground_truth_unstable": False,
            "citation_segment_match_median": 0.0,
            "citation_segment_match_unstable": False,
            "citation_in_retrieved_median": 1.0,
            "citation_in_retrieved_unstable": False,
        },
    }


def _sidecar(cases: list[dict], label: str = "x") -> dict:
    return {"schema_version": 1, "meta": {"label": label}, "cases": cases}


def test_compare_aggregates_agentic_axes(tmp_path: Path) -> None:
    """Agentic axes summarise medians + p95 for both sides of the A/B."""
    baseline_cases = [
        _case_with_agentic(
            "c1", tool_call_count=None  # legacy run — no agentic data
        )
    ]
    current_cases = [
        _case_with_agentic(
            "c1",
            tool_call_count=3,
            route_correctness=0.8,
            max_context_tokens_used=12_000,
            forced_answer=False,
        )
    ]
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_sidecar(baseline_cases, "legacy")))

    result = compare_sidecars(
        baseline_path,
        _sidecar(current_cases, "agentic"),
        epsilon=0.05,
    )

    by_label = {a.label: a for a in result.agentic_axes}
    assert "Agentic: tool_call_count" in by_label
    assert by_label["Agentic: tool_call_count"].baseline_median is None
    assert by_label["Agentic: tool_call_count"].current_median == 3
    assert by_label["Agentic: route_correctness"].current_median == 0.8


def test_compare_handles_both_sides_legacy(tmp_path: Path) -> None:
    """When neither side has agentic data, axes show N/A everywhere."""
    cases = [_case_with_agentic("c1", tool_call_count=None)]
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_sidecar(cases, "a")))

    result = compare_sidecars(
        baseline_path,
        _sidecar([_case_with_agentic("c1", tool_call_count=None)], "b"),
        epsilon=0.05,
    )
    for axis in result.agentic_axes:
        assert axis.baseline_median is None
        assert axis.current_median is None


def test_render_includes_agentic_axes_section(tmp_path: Path) -> None:
    baseline_cases = [_case_with_agentic("c1", tool_call_count=None)]
    current_cases = [
        _case_with_agentic(
            "c1", tool_call_count=2, route_correctness=1.0
        )
    ]
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_sidecar(baseline_cases, "legacy")))

    result = compare_sidecars(
        baseline_path,
        _sidecar(current_cases, "agentic"),
        epsilon=0.05,
    )
    md = render_comparison_md(result)
    assert "Agentic-loop axes" in md
    assert "Agentic: tool_call_count" in md
    assert "N/A" in md  # baseline side has no agentic data


def test_render_omits_section_when_no_agentic_axes() -> None:
    """If agentic_axes is empty (e.g. an older sidecar), the renderer
    must not blow up; the section is simply omitted."""
    from app.evals.compare import ComparisonResult

    result = ComparisonResult(
        baseline_label="x",
        baseline_path="x.json",
        common_case_ids=(),
        only_in_baseline=(),
        only_in_current=(),
        deltas=(),
        agentic_axes=(),
    )
    md = render_comparison_md(result)
    assert "Agentic-loop axes" not in md
