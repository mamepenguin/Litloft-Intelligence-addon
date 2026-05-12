"""Unit tests for the Phase 1.C agentic loop.

Focus: state-machine correctness of ``run_agentic_loop`` and the
capability/budget helpers. The LLM client is replaced by an in-memory
fake that yields a pre-scripted sequence of ``ChatTurnResult``
instances; the tools are mocked too so the loop can be exercised
without any DB / HTTP traffic.
"""

from __future__ import annotations

import json
import sys
from typing import Any
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

from app.config import AgenticModelEntry, LLMConfig  # noqa: E402
from app.llm import ChatToolCall, ChatTurnResult  # noqa: E402
from app.rag.agentic import (  # noqa: E402
    DEFAULT_MAX_TOOL_CALLS,
    FAIL_LOUD_ANSWER,
    agentic_capability_supported,
    compute_token_budget,
    get_agentic_model_entry,
    run_agentic_loop,
)
from app.rag.tools.context import ToolResultEnvelope  # noqa: E402


class _ScriptedLLM:
    """In-memory LLM stub that replays a fixed list of turns."""

    enabled = True

    def __init__(self, turns: list[ChatTurnResult | None]) -> None:
        self._turns = list(turns)
        self.calls: list[list[dict]] = []

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        temperature: float | None = None,
        max_tokens_override: int | None = None,
        tool_choice: str | dict = "auto",
        response_format: dict | None = None,
    ) -> ChatTurnResult | None:
        # Capture for assertions about message growth & tool_choice.
        self.calls.append(list(messages))
        if not self._turns:
            return None
        return self._turns.pop(0)


def _final_json_turn(payload: dict[str, Any]) -> ChatTurnResult:
    return ChatTurnResult(
        content=json.dumps(payload),
        tool_calls=(),
        finish_reason="stop",
    )


def _tool_call_turn(
    name: str, arguments: dict, call_id: str = "tc1"
) -> ChatTurnResult:
    return ChatTurnResult(
        content=None,
        tool_calls=(
            ChatToolCall(
                id=call_id,
                name=name,
                arguments_raw=json.dumps(arguments),
            ),
        ),
        finish_reason="tool_calls",
    )


# ---------------------------------------------------------------------------
# Capability / budget helpers
# ---------------------------------------------------------------------------


def test_agentic_capability_supported_off_kill_switch() -> None:
    cfg = LLMConfig(
        provider="openai_compatible",
        base_url="http://x",
        model="gpt-4o",
        agentic_mode="off",
        agentic_models=(AgenticModelEntry(name="gpt-4o", context_window=8192),),
    )
    assert agentic_capability_supported("gpt-4o", cfg) is False


def test_agentic_capability_supported_unknown_model() -> None:
    cfg = LLMConfig(
        model="qwen-7b",
        agentic_models=(AgenticModelEntry(name="gpt-4o"),),
    )
    assert agentic_capability_supported("qwen-7b", cfg) is False


def test_agentic_capability_supported_allowlisted() -> None:
    cfg = LLMConfig(
        model="qwen2.5:14b",
        agentic_models=(
            AgenticModelEntry(name="qwen2.5:14b", context_window=32768),
            AgenticModelEntry(name="gpt-4o", context_window=128000),
        ),
    )
    assert agentic_capability_supported("qwen2.5:14b", cfg) is True


def test_get_agentic_model_entry_returns_matching_entry() -> None:
    cfg = LLMConfig(
        agentic_models=(
            AgenticModelEntry(name="qwen2.5:14b", context_window=32768),
        ),
    )
    entry = get_agentic_model_entry("qwen2.5:14b", cfg)
    assert entry is not None
    assert entry.context_window == 32768


def test_compute_token_budget_seventy_percent() -> None:
    # context_window=10000 * 0.7 = 7000
    assert compute_token_budget(10000) == 7000


def test_compute_token_budget_floor() -> None:
    # Tiny context windows clamp to a sane minimum.
    assert compute_token_budget(100) == 1024


# ---------------------------------------------------------------------------
# run_agentic_loop: happy path
# ---------------------------------------------------------------------------


@pytest.fixture
def _fake_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(*, context, query, top_k=10, lit_token=None):
        context.register_tool_call("search_files")
        context.register_file_ids(["FID1"])
        return ToolResultEnvelope(
            payload={"files": [{"file_id": "FID1", "title": "Hit"}]},
            token_estimate=100,
        )

    async def fake_detail(*, context, file_id):
        context.register_tool_call("get_file_detail")
        return ToolResultEnvelope(
            payload={"file_id": file_id, "title": "Hit"},
            token_estimate=50,
        )

    async def fake_chunks(*, context, file_id, type, mode="summary", range=None):
        context.register_tool_call("get_file_chunks")
        return ToolResultEnvelope(
            payload={"chunks": [{"chunk_id": 0, "text": "body"}]},
            token_estimate=150,
        )

    async def fake_related(*, context, file_id, kind=None):
        context.register_tool_call("get_related_files")
        return ToolResultEnvelope(
            payload={"file_id": file_id, "relations": []},
            token_estimate=20,
        )

    monkeypatch.setattr("app.rag.agentic.search_files", fake_search)
    monkeypatch.setattr("app.rag.agentic.get_file_detail", fake_detail)
    monkeypatch.setattr("app.rag.agentic.get_file_chunks", fake_chunks)
    monkeypatch.setattr("app.rag.agentic.get_related_files", fake_related)


@pytest.mark.asyncio
async def test_loop_single_turn_returns_final_answer(_fake_tools) -> None:
    """LLM answers immediately with no tool calls."""
    llm = _ScriptedLLM(
        [_final_json_turn({"answer": "hi", "citations": []})]
    )
    result = await run_agentic_loop(
        query="hello",
        llm_client=llm,  # type: ignore[arg-type]
        max_total_tokens=10_000,
    )
    assert result.answer == "hi"
    assert result.citations == []
    assert result.telemetry.tool_calls == ()
    assert result.telemetry.forced_answer is False


@pytest.mark.asyncio
async def test_loop_search_then_answer(_fake_tools) -> None:
    """Typical sequence: search → final answer citing the returned file."""
    llm = _ScriptedLLM(
        [
            _tool_call_turn("search_files", {"query": "x"}),
            _final_json_turn(
                {
                    "answer": "found",
                    "citations": [{"file_id": "FID1", "location": "0:00"}],
                }
            ),
        ]
    )
    result = await run_agentic_loop(
        query="x",
        llm_client=llm,  # type: ignore[arg-type]
        max_total_tokens=10_000,
    )
    assert result.telemetry.tool_calls == ("search_files",)
    assert result.citations == [{"file_id": "FID1", "location": "0:00"}]
    assert result.answer == "found"


# ---------------------------------------------------------------------------
# Citation-strict / fail-loud paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_citation_strict_retry_succeeds(_fake_tools) -> None:
    """First answer cites an unknown file_id; the retry must strip it."""
    llm = _ScriptedLLM(
        [
            _tool_call_turn("search_files", {"query": "x"}),
            _final_json_turn(
                {
                    "answer": "borrowed",
                    "citations": [{"file_id": "GHOST", "location": ""}],
                }
            ),
            _final_json_turn(
                {
                    "answer": "borrowed",
                    "citations": [{"file_id": "FID1", "location": ""}],
                }
            ),
        ]
    )
    result = await run_agentic_loop(
        query="x",
        llm_client=llm,  # type: ignore[arg-type]
        max_total_tokens=10_000,
    )
    assert result.telemetry.citation_retries == 1
    assert result.citations == [{"file_id": "FID1", "location": ""}]


@pytest.mark.asyncio
async def test_loop_citation_strict_retry_exhausted(_fake_tools) -> None:
    """If the LLM keeps citing ghosts, the loop fails loud."""
    llm = _ScriptedLLM(
        [
            _tool_call_turn("search_files", {"query": "x"}),
            _final_json_turn(
                {
                    "answer": "first try",
                    "citations": [{"file_id": "GHOST1", "location": ""}],
                }
            ),
            _final_json_turn(
                {
                    "answer": "second try",
                    "citations": [{"file_id": "GHOST2", "location": ""}],
                }
            ),
        ]
    )
    result = await run_agentic_loop(
        query="x",
        llm_client=llm,  # type: ignore[arg-type]
        max_total_tokens=10_000,
    )
    assert result.answer == FAIL_LOUD_ANSWER
    assert result.citations == []
    assert result.telemetry.citation_retries == 1


@pytest.mark.asyncio
async def test_loop_llm_failure_fail_loud(_fake_tools) -> None:
    """LLM client returns None on transient failure → fail-loud."""
    llm = _ScriptedLLM([None])
    result = await run_agentic_loop(
        query="x",
        llm_client=llm,  # type: ignore[arg-type]
        max_total_tokens=10_000,
    )
    assert result.answer == FAIL_LOUD_ANSWER


@pytest.mark.asyncio
async def test_loop_malformed_tool_arguments_fail_loud(_fake_tools) -> None:
    """A tool_call whose ``arguments`` is not valid JSON → fail-loud."""
    llm = _ScriptedLLM(
        [
            ChatTurnResult(
                content=None,
                tool_calls=(
                    ChatToolCall(
                        id="x",
                        name="search_files",
                        arguments_raw="{not json}",
                    ),
                ),
                finish_reason="tool_calls",
            ),
        ]
    )
    result = await run_agentic_loop(
        query="x",
        llm_client=llm,  # type: ignore[arg-type]
        max_total_tokens=10_000,
    )
    assert result.answer == FAIL_LOUD_ANSWER


# ---------------------------------------------------------------------------
# Forced answer / budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_max_tool_calls_triggers_forced_answer(
    _fake_tools,
) -> None:
    """After hitting max_tool_calls the loop injects the forced-answer
    system message and accepts the next non-tool reply as final."""
    turns: list[ChatTurnResult] = [
        _tool_call_turn("search_files", {"query": "x"}, call_id=f"tc{i}")
        for i in range(3)
    ]
    turns.append(
        _final_json_turn(
            {"answer": "best-effort", "citations": []}
        )
    )
    llm = _ScriptedLLM(turns)
    result = await run_agentic_loop(
        query="x",
        llm_client=llm,  # type: ignore[arg-type]
        max_tool_calls=3,
        max_total_tokens=10_000,
    )
    assert result.telemetry.forced_answer is True
    assert result.answer == "best-effort"


@pytest.mark.asyncio
async def test_loop_budget_exhausted_triggers_forced_answer(
    monkeypatch: pytest.MonkeyPatch, _fake_tools
) -> None:
    """Cumulative-token budget triggers forced answer before
    max_tool_calls."""

    async def big_search(*, context, query, top_k=10, lit_token=None):
        context.register_tool_call("search_files")
        context.register_file_ids(["FID1"])
        # Crash through the budget on the first call.
        context.register_result_tokens(50_000)
        return ToolResultEnvelope(
            payload={"files": [{"file_id": "FID1"}]},
            token_estimate=50_000,
        )

    monkeypatch.setattr("app.rag.agentic.search_files", big_search)

    llm = _ScriptedLLM(
        [
            _tool_call_turn("search_files", {"query": "x"}),
            _final_json_turn(
                {"answer": "forced", "citations": []}
            ),
        ]
    )
    result = await run_agentic_loop(
        query="x",
        llm_client=llm,  # type: ignore[arg-type]
        max_tool_calls=10,
        max_total_tokens=5_000,
    )
    assert result.telemetry.forced_answer is True
    assert result.answer == "forced"


# ---------------------------------------------------------------------------
# Tool dispatch dispatch (unknown tool, bad args)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_unknown_tool_name_does_not_crash(
    monkeypatch: pytest.MonkeyPatch, _fake_tools
) -> None:
    """An unknown tool name comes back as an error envelope. The loop
    still records the call and progresses to the next iteration."""
    llm = _ScriptedLLM(
        [
            ChatTurnResult(
                content=None,
                tool_calls=(
                    ChatToolCall(
                        id="x",
                        name="totally_unknown_tool",
                        arguments_raw="{}",
                    ),
                ),
                finish_reason="tool_calls",
            ),
            _final_json_turn({"answer": "ok", "citations": []}),
        ]
    )
    result = await run_agentic_loop(
        query="x",
        llm_client=llm,  # type: ignore[arg-type]
        max_total_tokens=10_000,
    )
    assert result.answer == "ok"


# ---------------------------------------------------------------------------
# Message-list growth (assistant + tool entries appended correctly)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_appends_tool_messages_with_call_id(_fake_tools) -> None:
    llm = _ScriptedLLM(
        [
            _tool_call_turn("search_files", {"query": "x"}, call_id="abc"),
            _final_json_turn({"answer": "y", "citations": []}),
        ]
    )
    await run_agentic_loop(
        query="x",
        llm_client=llm,  # type: ignore[arg-type]
        max_total_tokens=10_000,
    )
    # Second call sees the assistant tool_calls + a tool reply.
    second_call_messages = llm.calls[1]
    assistant = next(
        m for m in second_call_messages if m.get("role") == "assistant"
    )
    assert assistant["tool_calls"][0]["id"] == "abc"
    tool_msg = next(m for m in second_call_messages if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "abc"
    assert tool_msg["name"] == "search_files"
