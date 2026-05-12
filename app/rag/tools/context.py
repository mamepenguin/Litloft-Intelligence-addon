"""Per-answer agentic loop context (Phase 1.B).

The ``ToolContext`` is a mutable scratch area threaded through every
tool call within a single ``run_agentic_loop`` invocation. It carries
the inputs the loop needs to enforce invariants the LLM cannot:

* ``drive`` / ``viewer_id`` / ``lit_token`` — request identity used by
  ``search_files`` and by access-control filtering.
* ``tool_returned_file_ids`` — the running allow-list of file_ids the
  LLM is permitted to cite. Each tool appends to this set; the loop's
  citation-strict verifier rejects any ``citations[].file_id`` not in
  it.
* ``cumulative_tokens`` — running sum of tool-result tokens so the
  loop can short-circuit before another tool call blows the context
  window.
* ``tool_calls`` — ordered list of tool names invoked. Phase 1.A's
  ``route_correctness`` metric reads this from the eventual
  ``AgenticTelemetry``.

Why a class instead of plain kwargs: each tool needs to mutate the
allow-list and token counter atomically. A frozen dataclass would
force the caller to thread the new state back through; a mutable
object keeps the wrappers focused on their actual work.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from app.rag.tools._access import ALLOW_LIST_CAP


@dataclass
class ToolContext:
    """Mutable per-answer state shared by every tool invocation.

    ``cumulative_tokens`` reflects how much tool output the LLM has
    already seen this answer; the agentic loop must call
    ``register_result_tokens`` after each tool returns so the next
    pre-call budget check is accurate.
    """

    drive: str | None = None
    viewer_id: str | None = None
    lit_token: str | None = None
    tool_returned_file_ids: set[str] = field(default_factory=set)
    tool_calls: list[str] = field(default_factory=list)
    cumulative_tokens: int = 0

    def register_tool_call(self, name: str) -> None:
        self.tool_calls.append(name)

    def register_file_ids(self, file_ids: Iterable[str]) -> None:
        # ``ALLOW_LIST_CAP`` keeps the citation allow-list bounded so a
        # tool-spam LLM cannot grow it to thousands of entries (which
        # would weaken citation-strict and the budget accounting).
        for fid in file_ids:
            if isinstance(fid, str) and fid:
                if len(self.tool_returned_file_ids) >= ALLOW_LIST_CAP:
                    break
                self.tool_returned_file_ids.add(fid)

    def register_result_tokens(self, tokens: int) -> None:
        # ``int(tokens)`` so a downstream that returns ``float`` (rare,
        # but possible if a token model exposes fractional cost) still
        # behaves arithmetically.
        if tokens > 0:
            self.cumulative_tokens += int(tokens)

    def is_allowed_citation(self, file_id: str) -> bool:
        return file_id in self.tool_returned_file_ids


@dataclass(frozen=True)
class ToolResultEnvelope:
    """Wrapper around a tool's payload + accounting metadata.

    Tools return this so the agentic loop can:

    * Inspect ``payload`` to forward as the OpenAI ``tool`` message.
    * Read ``token_estimate`` to update cumulative budget without
      re-encoding the result.
    * Read ``truncated`` to know when to add a warning to the LLM
      message ("results were truncated; ask narrower next time").
    """

    payload: Any
    token_estimate: int
    truncated: bool = False
    warning: str | None = None


__all__ = ["ToolContext", "ToolResultEnvelope"]
