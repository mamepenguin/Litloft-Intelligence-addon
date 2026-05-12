"""Unit tests for the Phase 1.A agentic metric helpers.

The functions are pure (no DB, no LLM), so the tests stay in this file
without snapshot fixtures. They lock in the "legacy run → None"
contract: aggregate reports must show ``N/A`` for cases that did not
exercise the agentic loop, not a misleading ``0.00``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Same heavyweight-dep stubs other eval tests use — keeps module
# imports cheap without pulling torch / sentence_transformers.
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

from app.evals.agentic_metrics import (  # noqa: E402
    forced_answer,
    forced_answer_rate,
    max_context_tokens_used,
    route_correctness,
    tool_call_count,
)
from app.evals.loader import load_case  # noqa: E402
from app.rag.agentic_types import AgenticTelemetry  # noqa: E402


# ---------------------------------------------------------------------------
# tool_call_count
# ---------------------------------------------------------------------------


def test_tool_call_count_returns_none_for_legacy() -> None:
    assert tool_call_count(None) is None


def test_tool_call_count_counts_tool_calls() -> None:
    t = AgenticTelemetry(tool_calls=("search_files", "get_file_chunks"))
    assert tool_call_count(t) == 2


def test_tool_call_count_zero_when_agentic_did_not_call_tools() -> None:
    t = AgenticTelemetry(tool_calls=())
    assert tool_call_count(t) == 0


# ---------------------------------------------------------------------------
# route_correctness
# ---------------------------------------------------------------------------


def test_route_correctness_none_when_expected_empty() -> None:
    t = AgenticTelemetry(tool_calls=("search_files",))
    assert route_correctness((), t) is None


def test_route_correctness_none_for_legacy_run() -> None:
    assert route_correctness(("search_files",), None) is None


def test_route_correctness_full_match() -> None:
    t = AgenticTelemetry(tool_calls=("search_files", "get_file_chunks"))
    assert route_correctness(("search_files", "get_file_chunks"), t) == 1.0


def test_route_correctness_tolerates_extra_trailing_calls() -> None:
    # The metric is prefix-match: the agentic loop may make extra calls
    # after the expected route without being penalised.
    t = AgenticTelemetry(
        tool_calls=("search_files", "get_file_chunks", "get_related_files")
    )
    assert route_correctness(("search_files", "get_file_chunks"), t) == 1.0


def test_route_correctness_partial_prefix() -> None:
    t = AgenticTelemetry(tool_calls=("search_files",))
    assert (
        route_correctness(("search_files", "get_file_chunks"), t) == 0.5
    )


def test_route_correctness_diverges_at_first_call() -> None:
    t = AgenticTelemetry(tool_calls=("get_file_chunks",))
    assert route_correctness(("search_files", "get_file_chunks"), t) == 0.0


def test_route_correctness_shorter_actual_than_expected() -> None:
    t = AgenticTelemetry(tool_calls=("search_files", "get_file_chunks"))
    assert route_correctness(
        ("search_files", "get_file_chunks", "get_related_files"), t
    ) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# max_context_tokens_used / forced_answer
# ---------------------------------------------------------------------------


def test_max_context_tokens_returns_none_for_legacy() -> None:
    assert max_context_tokens_used(None) is None


def test_max_context_tokens_returns_telemetry_value() -> None:
    t = AgenticTelemetry(max_context_tokens_used=12345)
    assert max_context_tokens_used(t) == 12345


def test_forced_answer_returns_none_for_legacy() -> None:
    assert forced_answer(None) is None


def test_forced_answer_returns_telemetry_value() -> None:
    assert forced_answer(AgenticTelemetry(forced_answer=True)) is True
    assert forced_answer(AgenticTelemetry(forced_answer=False)) is False


# ---------------------------------------------------------------------------
# forced_answer_rate
# ---------------------------------------------------------------------------


def test_forced_answer_rate_excludes_legacy_runs() -> None:
    runs = [
        None,
        AgenticTelemetry(forced_answer=True),
        AgenticTelemetry(forced_answer=False),
        AgenticTelemetry(forced_answer=False),
    ]
    assert forced_answer_rate(runs) == pytest.approx(1 / 3)


def test_forced_answer_rate_all_none_returns_none() -> None:
    assert forced_answer_rate([None, None, None]) is None


def test_forced_answer_rate_empty_returns_none() -> None:
    assert forced_answer_rate([]) is None


def test_forced_answer_rate_all_forced() -> None:
    runs = [
        AgenticTelemetry(forced_answer=True),
        AgenticTelemetry(forced_answer=True),
    ]
    assert forced_answer_rate(runs) == 1.0


# ---------------------------------------------------------------------------
# Loader: expected_tool_sequence parsing
# ---------------------------------------------------------------------------


def test_loader_parses_expected_tool_sequence(tmp_path: Path) -> None:
    p = tmp_path / "case.yml"
    p.write_text(
        """
id: x
query: q
expected_keywords:
  must_include: []
ground_truth_files: []
expected_tool_sequence:
  - search_files
  - get_file_chunks
""".strip(),
        encoding="utf-8",
    )
    case = load_case(p)
    assert case.expected_tool_sequence == ("search_files", "get_file_chunks")


def test_loader_defaults_expected_tool_sequence_to_empty(tmp_path: Path) -> None:
    p = tmp_path / "case.yml"
    p.write_text(
        """
id: x
query: q
expected_keywords:
  must_include: []
ground_truth_files: []
""".strip(),
        encoding="utf-8",
    )
    case = load_case(p)
    assert case.expected_tool_sequence == ()


def test_loader_rejects_non_string_tool_entry(tmp_path: Path) -> None:
    p = tmp_path / "case.yml"
    p.write_text(
        """
id: x
query: q
expected_keywords:
  must_include: []
ground_truth_files: []
expected_tool_sequence:
  - 42
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected_tool_sequence"):
        load_case(p)
