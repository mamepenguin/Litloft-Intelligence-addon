"""SPEC-ADDON-054 I2: the agentic branch ranks, reads and cites inside E.

``ToolContext`` carries the effective set. ``search_files`` hands it to the
retriever, the id-taking tools answer an out-of-scope id like an inaccessible
one without any Internal API or database call, related files outside E are
dropped, and the citation allow-list refuses them.
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
from app.llm import ChatToolCall, ChatTurnResult  # noqa: E402
from app.rag.agentic import AgenticAnswer, run_agentic_loop  # noqa: E402
from app.rag.agentic_types import AgenticTelemetry  # noqa: E402
from app.rag.service import _stream_answer  # noqa: E402
from app.rag.tools.context import (  # noqa: E402
    ToolContext,
    ToolResultEnvelope,
)
from app.rag.tools.get_file_chunks import get_file_chunks  # noqa: E402
from app.rag.tools.get_file_detail import get_file_detail  # noqa: E402
from app.rag.tools.get_related_files import get_related_files  # noqa: E402
from app.rag.tools.search_files import search_files  # noqa: E402
from tests.llm_helpers import bind_llm  # noqa: E402

IN_SCOPE = "s1"
OUT_OF_SCOPE = "x9"


def _scoped_context() -> ToolContext:
    return ToolContext(file_id_constraint=frozenset({IN_SCOPE}))


def test_register_file_ids_drops_ids_outside_the_constraint():
    context = _scoped_context()

    context.register_file_ids([IN_SCOPE, OUT_OF_SCOPE])

    assert context.tool_returned_file_ids == {IN_SCOPE}
    assert context.is_allowed_citation(IN_SCOPE) is True
    assert context.is_allowed_citation(OUT_OF_SCOPE) is False


@pytest.mark.asyncio
async def test_search_files_forwards_the_constraint_to_the_retriever(monkeypatch):
    retrieve = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "app.rag.tools.search_files.retrieve_with_keywords", retrieve
    )
    monkeypatch.setattr(
        "app.rag.tools.search_files._transcript_bearing_file_ids",
        lambda ids: set(),
    )
    context = _scoped_context()

    await search_files(context=context, query="budget", top_k=3)

    assert retrieve.await_args.kwargs["file_id_constraint"] == frozenset(
        {IN_SCOPE}
    )


@pytest.mark.asyncio
async def test_get_file_detail_refuses_without_reaching_the_internal_api(monkeypatch):
    access = AsyncMock(side_effect=AssertionError("access filter was called"))
    monkeypatch.setattr(
        "app.rag.tools.get_file_detail.ensure_access", access
    )
    basic = AsyncMock(side_effect=AssertionError("metadata was fetched"))
    hydrate = AsyncMock(side_effect=AssertionError("files were hydrated"))
    relations = AsyncMock(side_effect=AssertionError("relations were fetched"))
    chunks = MagicMock(side_effect=AssertionError("chunks were counted"))
    monkeypatch.setattr(
        "app.rag.tools.get_file_detail._fetch_basic_metadata", basic
    )
    monkeypatch.setattr("app.rag.tools.get_file_detail.hydrate_files", hydrate)
    monkeypatch.setattr(
        "app.rag.tools.get_file_detail._fetch_relations_summary", relations
    )
    monkeypatch.setattr(
        "app.rag.tools.get_file_detail._count_transcript_chunks", chunks
    )

    envelope = await get_file_detail(
        context=_scoped_context(), file_id=OUT_OF_SCOPE
    )

    assert envelope.payload == {
        "file_id": OUT_OF_SCOPE,
        "error": "not_found",
    }
    assert envelope.warning == "file not found"
    assert not access.called
    assert not basic.called and not hydrate.called
    assert not relations.called and not chunks.called


@pytest.mark.asyncio
async def test_get_file_chunks_refuses_without_reading_the_database(monkeypatch):
    access = AsyncMock(side_effect=AssertionError("access filter was called"))
    monkeypatch.setattr(
        "app.rag.tools.get_file_chunks.ensure_access", access
    )
    load = MagicMock(side_effect=AssertionError("chunks were loaded"))
    monkeypatch.setattr("app.rag.tools.get_file_chunks._load_transcript_chunks", load)

    envelope = await get_file_chunks(
        context=_scoped_context(),
        file_id=OUT_OF_SCOPE,
        type="transcript",
    )

    assert envelope.payload == {
        "file_id": OUT_OF_SCOPE,
        "error": "not_found",
    }
    assert envelope.warning == "file not found"
    assert not access.called and not load.called


@pytest.mark.asyncio
async def test_get_related_files_refuses_an_out_of_scope_input(monkeypatch):
    access = AsyncMock(side_effect=AssertionError("access filter was called"))
    monkeypatch.setattr(
        "app.rag.tools.get_related_files.ensure_access", access
    )
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.get = AsyncMock(side_effect=AssertionError("relations were fetched"))

    with patch(
        "app.rag.tools.get_related_files.httpx.AsyncClient",
        return_value=client,
    ):
        envelope = await get_related_files(
            context=_scoped_context(), file_id=OUT_OF_SCOPE
        )

    assert envelope.payload == {
        "file_id": OUT_OF_SCOPE,
        "relations": [],
        "error": "not_found",
    }
    assert envelope.warning == "file not found"
    assert not access.called and not client.get.called


@pytest.mark.asyncio
async def test_get_related_files_drops_related_files_outside_the_constraint(
    monkeypatch,
):
    async def _allow(file_ids, credential=None, *, drive):
        return {fid for fid in file_ids if isinstance(fid, str)}

    monkeypatch.setattr(
        "app.rag.tools.get_related_files.ensure_access", _allow
    )
    raw: list[dict[str, Any]] = [
        {"id": 1, "file_id_a": IN_SCOPE, "file_id_b": "s2", "kind": "cite"},
        {"id": 2, "file_id_a": OUT_OF_SCOPE, "file_id_b": IN_SCOPE, "kind": "see"},
    ]
    response = MagicMock()
    response.status_code = 200
    response.json = MagicMock(return_value=raw)
    response.raise_for_status = MagicMock()
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.get = AsyncMock(return_value=response)

    with patch(
        "app.rag.tools.get_related_files.httpx.AsyncClient",
        return_value=client,
    ):
        context = ToolContext(file_id_constraint=frozenset({IN_SCOPE, "s2"}))
        envelope = await get_related_files(context=context, file_id=IN_SCOPE)

    assert envelope.payload["relations"] == [
        {"file_id": "s2", "kind": "cite", "direction": "outgoing"}
    ]
    assert context.tool_returned_file_ids == {"s2"}


@pytest.fixture()
def agentic_settings(monkeypatch):
    """An agentic-capable profile so ``_stream_answer`` takes the loop."""
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
    llm_client = MagicMock()
    llm_client.enabled = True
    llm_client.generate_json = AsyncMock(return_value=None)
    llm_client.chat_with_tools = AsyncMock()
    bind_llm(monkeypatch, lambda: llm_client, config=settings.llm)
    return settings


@pytest.mark.asyncio
async def test_scoped_stream_gives_the_agentic_loop_the_effective_set(
    monkeypatch, agentic_settings
):
    """SPEC-ADDON-054 I1/I2: the agentic branch's E is S, not the drive."""
    loop = AsyncMock(
        return_value=AgenticAnswer(
            answer="",
            citations=[],
            telemetry=AgenticTelemetry(
                tool_calls=(), max_context_tokens_used=0, forced_answer=False
            ),
        )
    )
    monkeypatch.setattr("app.rag.service.run_agentic_loop", loop)

    events = [
        event
        async for event in _stream_answer(
            query="hello",
            credential=None,
            drive="family",
            scope={
                "folders": ["Work"],
                "collections": [],
                "tags": [],
                "kinds": [],
            },
            scope_file_ids=["s1", "s2"],
        )
    ]

    assert events[0].kind == "scope"
    assert loop.await_args.kwargs["file_id_constraint"] == frozenset(
        {"s1", "s2"}
    )


class _ScriptedLLM:
    enabled = True

    def __init__(self, turns: list[ChatTurnResult | None]) -> None:
        self._turns = list(turns)

    async def chat_with_tools(self, messages, tools=None, **kwargs):
        return self._turns.pop(0)


def _tool_call_turn(name: str, arguments: dict) -> ChatTurnResult:
    return ChatTurnResult(
        content=None,
        tool_calls=(
            ChatToolCall(
                id="tc1",
                name=name,
                arguments_raw=json.dumps(arguments),
            ),
        ),
        finish_reason="tool_calls",
    )


def _final_turn(payload: dict[str, Any]) -> ChatTurnResult:
    return ChatTurnResult(
        content=json.dumps(payload), tool_calls=(), finish_reason="stop"
    )


@pytest.mark.asyncio
async def test_agentic_loop_context_carries_the_effective_set(monkeypatch):
    """SPEC-ADDON-054 I2: the tools and citation allow-list see E."""
    contexts: list[ToolContext] = []

    async def _dispatch(name, arguments, *, context):
        contexts.append(context)
        return ToolResultEnvelope(payload={"files": []}, token_estimate=0)

    monkeypatch.setattr("app.rag.agentic._dispatch_tool", _dispatch)
    llm = _ScriptedLLM(
        [
            _tool_call_turn("search_files", {"query": "x"}),
            _final_turn({"answer": "ok", "citations": []}),
        ]
    )

    await run_agentic_loop(
        query="x",
        llm_client=llm,  # type: ignore[arg-type]
        max_total_tokens=10_000,
        file_id_constraint=frozenset({IN_SCOPE}),
    )

    assert contexts
    assert contexts[0].file_id_constraint == frozenset({IN_SCOPE})
