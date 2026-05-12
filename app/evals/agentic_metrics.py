"""Metrics specific to the agentic Ask loop (Phase 1.A).

These metrics observe ``AgenticTelemetry`` recorded by the RAG service
when the agentic loop runs. Legacy single-turn answers have no
telemetry; the helpers below treat ``None`` as "not applicable" so the
aggregate row excludes those cases rather than averaging them as zero.

Why a separate module: the eval-side aggregation logic does not belong
in ``rag/agentic.py`` (production loop) or ``rag/service.py`` (already
sprawling). The harness owns metric definitions; the service owns
telemetry emission.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag.agentic_types import AgenticTelemetry


def tool_call_count(telemetry: "AgenticTelemetry | None") -> int | None:
    """Number of tool calls the agentic loop issued for this answer.

    Returns ``None`` when the legacy path ran (no telemetry). Returning
    ``None`` instead of ``0`` lets aggregate reports exclude legacy
    runs from the "tool calls per query" distribution.
    """
    if telemetry is None:
        return None
    return len(telemetry.tool_calls)


def route_correctness(
    expected: Sequence[str], actual: "AgenticTelemetry | None"
) -> float | None:
    """Prefix-match score between the expected and actual tool call order.

    Semantics:
    * Empty ``expected`` → ``None`` (case has no route expectation).
    * No telemetry → ``None`` (legacy path; nothing to compare).
    * Actual starts with ``expected`` (in order) → ``1.0``.
    * Otherwise the longest common prefix length divided by
      ``len(expected)``. ``0.0`` means the very first call already
      diverged.

    Prefix match (rather than set match) was chosen because the
    "correct" path for a question is meaningful as a sequence — e.g.
    ``search_files`` should always precede ``get_file_chunks`` for the
    same target file. Tool calls after the expected prefix do not
    penalise the score; the route is right, it just kept going.
    """
    if not expected:
        return None
    if actual is None:
        return None
    actual_calls = actual.tool_calls
    match = 0
    for i, name in enumerate(expected):
        if i >= len(actual_calls):
            break
        if actual_calls[i] != name:
            break
        match += 1
    return match / len(expected)


def max_context_tokens_used(telemetry: "AgenticTelemetry | None") -> int | None:
    """Peak cumulative-token usage during the agentic loop.

    ``None`` for legacy runs (single LLM call; the harness does not
    instrument it here).
    """
    if telemetry is None:
        return None
    return telemetry.max_context_tokens_used


def forced_answer(telemetry: "AgenticTelemetry | None") -> bool | None:
    """Whether the loop hit the budget/max-tool-call ceiling and was
    forced to answer with the data it already had.

    ``None`` for legacy runs.
    """
    if telemetry is None:
        return None
    return telemetry.forced_answer


def forced_answer_rate(
    telemetries: Sequence["AgenticTelemetry | None"],
) -> float | None:
    """Fraction of agentic runs that ended in a forced answer.

    Legacy runs (``None``) are excluded from both numerator and
    denominator. Returns ``None`` if no agentic runs are present (so
    the aggregate row can show ``N/A`` instead of a misleading ``0.00``).
    """
    agentic = [t for t in telemetries if t is not None]
    if not agentic:
        return None
    return sum(1 for t in agentic if t.forced_answer) / len(agentic)


__all__ = [
    "tool_call_count",
    "route_correctness",
    "max_context_tokens_used",
    "forced_answer",
    "forced_answer_rate",
]
