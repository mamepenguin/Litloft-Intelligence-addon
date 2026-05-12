"""Shared types for the agentic Ask loop.

Lives in its own module so the eval harness can import the telemetry
shape without pulling the full ``rag.service`` (which transitively
imports the LLM client and retriever stack). Phase 1.A introduces the
shape; Phase 1.C populates it from the real loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgenticTelemetry:
    """Per-answer telemetry from the agentic loop.

    Legacy (non-agentic) responses leave ``agentic_telemetry`` as
    ``None``. The eval harness distinguishes "ran legacy" (telemetry
    absent) from "ran agentic with zero tool calls" (telemetry present
    with empty ``tool_calls``).
    """

    tool_calls: tuple[str, ...] = ()
    max_context_tokens_used: int = 0
    forced_answer: bool = False
    citation_retries: int = 0


__all__ = ["AgenticTelemetry"]
