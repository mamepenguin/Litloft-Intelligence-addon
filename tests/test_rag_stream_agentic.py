"""Verify ``stream_answer`` activates the agentic loop and yields the
expected SSE-vocabulary events when the loop wins the gate.

Companion to ``test_rag_agentic.py`` (which exercises the loop in
isolation). This file pins the streaming surface so the HTTP
endpoint stays consistent with the non-streaming path.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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

from app.config import AgenticModelEntry, LLMConfig, Settings  # noqa: E402
from app.rag.agentic import AgenticAnswer  # noqa: E402
from app.rag.agentic_types import AgenticTelemetry  # noqa: E402
from app.rag.service import AnswerEvent, stream_answer  # noqa: E402


@pytest.fixture
def _agentic_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Plant a settings object whose LLM is on the agentic allowlist."""
    from pathlib import Path

    settings = Settings(
        intelligence_data_dir=Path("/tmp/x"),
        litloft_db_path=Path("/tmp/x.db"),
        model_cache_dir=Path("/tmp/models"),
        search_db_path=Path("/tmp/search.db"),
        llm=LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="qwen2.5:14b",
            agentic_mode="auto",
            agentic_models=(
                AgenticModelEntry(name="qwen2.5:14b", context_window=32768),
            ),
        ),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.rag.service.settings", settings)
    # A bare MagicMock with the duck-typed attributes the gate checks.
    llm_client = MagicMock()
    llm_client.enabled = True
    llm_client.chat_with_tools = AsyncMock()  # presence only
    monkeypatch.setattr(
        "app.rag.service.get_llm_client", lambda: llm_client
    )
    return settings


async def _collect(gen) -> list[AnswerEvent]:
    out: list[AnswerEvent] = []
    async for event in gen:
        out.append(event)
    return out


@pytest.mark.asyncio
async def test_stream_answer_uses_agentic_when_gate_open(
    monkeypatch: pytest.MonkeyPatch, _agentic_settings: Settings
) -> None:
    """When the LLM is on the allowlist, stream_answer must call into
    the agentic loop instead of the legacy retriever."""
    agentic_answer = AgenticAnswer(
        answer="agentic reply",
        citations=[
            {"file_id": "FID1", "location": "0:00"},
        ],
        telemetry=AgenticTelemetry(
            tool_calls=("search_files", "get_file_chunks"),
            max_context_tokens_used=1234,
            forced_answer=False,
        ),
    )
    monkeypatch.setattr(
        "app.rag.service.run_agentic_loop",
        AsyncMock(return_value=agentic_answer),
    )

    events = await _collect(
        stream_answer(query="hello", lit_token=None, drive="d")
    )

    kinds = [e.kind for e in events]
    # The agentic SSE adapter emits exactly this order:
    # keywords → sources (empty) → answer_chunk → citation × N
    # → citations (terminal list) → done
    assert kinds == [
        "keywords",
        "sources",
        "answer_chunk",
        "citation",
        "citations",
        "done",
    ]
    answer_chunk = events[2]
    assert answer_chunk.data == {"delta": "agentic reply"}
    citation = events[3]
    assert citation.data["citation"]["file_id"] == "FID1"
    assert citation.data["index"] == 1
    done = events[-1]
    assert done.data["retrieved_count"] == 2
    assert done.data["agentic"]["tool_calls"] == [
        "search_files",
        "get_file_chunks",
    ]


@pytest.mark.asyncio
async def test_stream_answer_skips_agentic_when_model_not_allowlisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the active model is NOT in agentic_models, the gate fails
    closed and the legacy path runs (we don't unit-test legacy here,
    only assert the agentic loop never gets called)."""
    from pathlib import Path

    settings = Settings(
        intelligence_data_dir=Path("/tmp/x"),
        litloft_db_path=Path("/tmp/x.db"),
        model_cache_dir=Path("/tmp/models"),
        search_db_path=Path("/tmp/search.db"),
        llm=LLMConfig(
            provider="openai_compatible",
            base_url="http://x",
            model="something-else",
            agentic_mode="auto",
            agentic_models=(),  # empty allowlist
        ),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.rag.service.settings", settings)
    llm_client = MagicMock()
    llm_client.enabled = True
    llm_client.chat_with_tools = AsyncMock()
    monkeypatch.setattr(
        "app.rag.service.get_llm_client", lambda: llm_client
    )

    loop_spy = AsyncMock()
    monkeypatch.setattr("app.rag.service.run_agentic_loop", loop_spy)
    # Force legacy to bail early so we don't need the full retrieval
    # plumbing in this test (we only care that agentic was skipped).
    monkeypatch.setattr(
        "app.rag.service._resolve_personal_history",
        AsyncMock(
            return_value=MagicMock(
                short_circuit=True, decomposed=None, file_ids=None
            )
        ),
    )

    await _collect(stream_answer(query="x", lit_token=None))
    loop_spy.assert_not_called()


@pytest.mark.asyncio
async def test_stream_answer_skips_agentic_when_mode_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kill switch agentic_mode='off' must short-circuit even with
    the active model on the list."""
    from pathlib import Path

    settings = Settings(
        intelligence_data_dir=Path("/tmp/x"),
        litloft_db_path=Path("/tmp/x.db"),
        model_cache_dir=Path("/tmp/models"),
        search_db_path=Path("/tmp/search.db"),
        llm=LLMConfig(
            provider="openai_compatible",
            base_url="http://x",
            model="qwen2.5:14b",
            agentic_mode="off",
            agentic_models=(
                AgenticModelEntry(name="qwen2.5:14b", context_window=32768),
            ),
        ),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.rag.service.settings", settings)
    llm_client = MagicMock()
    llm_client.enabled = True
    llm_client.chat_with_tools = AsyncMock()
    monkeypatch.setattr(
        "app.rag.service.get_llm_client", lambda: llm_client
    )
    loop_spy = AsyncMock()
    monkeypatch.setattr("app.rag.service.run_agentic_loop", loop_spy)
    monkeypatch.setattr(
        "app.rag.service._resolve_personal_history",
        AsyncMock(
            return_value=MagicMock(
                short_circuit=True, decomposed=None, file_ids=None
            )
        ),
    )

    await _collect(stream_answer(query="x", lit_token=None))
    loop_spy.assert_not_called()


@pytest.mark.asyncio
async def test_stream_answer_skips_agentic_without_chat_with_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama-native client lacks chat_with_tools → gate must fail closed."""
    from pathlib import Path

    settings = Settings(
        intelligence_data_dir=Path("/tmp/x"),
        litloft_db_path=Path("/tmp/x.db"),
        model_cache_dir=Path("/tmp/models"),
        search_db_path=Path("/tmp/search.db"),
        llm=LLMConfig(
            provider="ollama",
            base_url="http://x",
            model="qwen2.5:14b",
            agentic_mode="auto",
            agentic_models=(
                AgenticModelEntry(name="qwen2.5:14b", context_window=32768),
            ),
        ),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.rag.service.settings", settings)

    class _NoToolsLLM:
        enabled = True
        # No chat_with_tools attribute.

    monkeypatch.setattr(
        "app.rag.service.get_llm_client", lambda: _NoToolsLLM()
    )
    loop_spy = AsyncMock()
    monkeypatch.setattr("app.rag.service.run_agentic_loop", loop_spy)
    monkeypatch.setattr(
        "app.rag.service._resolve_personal_history",
        AsyncMock(
            return_value=MagicMock(
                short_circuit=True, decomposed=None, file_ids=None
            )
        ),
    )

    await _collect(stream_answer(query="x", lit_token=None))
    loop_spy.assert_not_called()


@pytest.mark.asyncio
async def test_stream_answer_agentic_empty_citations(
    monkeypatch: pytest.MonkeyPatch, _agentic_settings: Settings
) -> None:
    """A "no answer" agentic response still emits terminal citations + done."""
    agentic_answer = AgenticAnswer(
        answer="情報を取得できませんでした。",
        citations=[],
        telemetry=AgenticTelemetry(
            tool_calls=("search_files",),
            forced_answer=False,
        ),
    )
    monkeypatch.setattr(
        "app.rag.service.run_agentic_loop",
        AsyncMock(return_value=agentic_answer),
    )

    events = await _collect(
        stream_answer(query="hello", lit_token=None)
    )
    kinds = [e.kind for e in events]
    # No ``citation`` events when the list is empty, but the
    # terminal ``citations`` + ``done`` still fire.
    assert "citation" not in kinds
    assert kinds[-2:] == ["citations", "done"]
    citations_event = next(e for e in events if e.kind == "citations")
    assert citations_event.data["citations"] == []
