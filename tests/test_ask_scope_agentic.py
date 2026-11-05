"""SPEC-ADDON-054 I2: the agentic branch ranks, reads and cites inside E.

``ToolContext`` carries the effective set. ``search_files`` hands it to the
retriever, the id-taking tools answer an out-of-scope id like an inaccessible
one without any Internal API or database call, related files outside E are
dropped, and the citation allow-list refuses them.
"""

from __future__ import annotations

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

from app.rag.tools.context import ToolContext  # noqa: E402
from app.rag.tools.get_file_chunks import get_file_chunks  # noqa: E402
from app.rag.tools.get_file_detail import get_file_detail  # noqa: E402
from app.rag.tools.get_related_files import get_related_files  # noqa: E402
from app.rag.tools.search_files import search_files  # noqa: E402

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
