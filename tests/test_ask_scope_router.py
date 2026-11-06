"""SPEC-ADDON-054 I3 and I5: resolution, the ``scope`` event, the empty scope.

The route resolves the scope through ``POST /files/matching`` after the body
gates and before the concurrency slot, answers ``scope_unavailable`` without
opening a stream, and hands the resolved set to the service. A scoped stream
opens with exactly one ``scope`` event; an empty resolved set answers "no
files" without calling an LLM.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from tests.llm_helpers import bind_llm
from tests.matching_core_double import MatchingCore, install

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

from app.config import (  # noqa: E402
    HierarchicalRagConfig,
    LLMConfig,
    PersonalHistoryConfig,
    RagConfig,
)
from app.file_matching import FilterUnavailable  # noqa: E402
from app.rag.coarse_retriever import ShortlistResult  # noqa: E402
from app.rag.query_decomposer import DecomposedQuery, TimeRange  # noqa: E402
from app.rag.query_transform import StructuredQuery  # noqa: E402
from app.rag.service import _stream_answer  # noqa: E402
from app.routers import rag as rag_module  # noqa: E402
from app.routers.rag import ask_endpoint  # noqa: E402
from app.schemas import AskRequest  # noqa: E402

HOME = "family"
QUESTION = "what did we decide in the meeting"


def _make_settings(make_settings, *, fallback_when_empty="graceful"):
    return make_settings(
        features=type(make_settings().features)(rag=True),
        llm=LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        ),
        rag=RagConfig(
            hierarchical=HierarchicalRagConfig(enabled=False),
            personal_history=PersonalHistoryConfig(
                enabled=True, fallback_when_empty=fallback_when_empty
            ),
        ),
    )


def _bind(monkeypatch, settings):
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.rag.settings", settings)
    monkeypatch.setattr("app.rag.service.settings", settings)
    llm_stub = MagicMock()
    llm_stub.enabled = True
    bind_llm(monkeypatch, lambda: llm_stub)
    return llm_stub


def _reset_semaphore():
    rag_module._ask_semaphore = None
    rag_module._ask_semaphore_loop = None
    return rag_module._get_ask_semaphore()


@pytest.fixture()
def ask_env(monkeypatch, make_settings):
    return _bind(monkeypatch, _make_settings(make_settings))


@pytest.fixture()
def ask_env_strict(monkeypatch, make_settings):
    return _bind(
        monkeypatch,
        _make_settings(make_settings, fallback_when_empty="strict"),
    )


async def _collect_sse(response: StreamingResponse) -> list[tuple[str, dict]]:
    buffer = ""
    async for chunk in response.body_iterator:
        buffer += chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
    parsed: list[tuple[str, dict]] = []
    for frame in buffer.split("\n\n"):
        if not frame.strip():
            continue
        name = ""
        data = ""
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        parsed.append((name, json.loads(data) if data else {}))
    return parsed


def _scope_payload():
    return {
        "folders": ["/Work/2024/"],
        "collections": ["読書メモ"],
        "tags": [" 会議 "],
        "kinds": ["markdown"],
    }


def _no_history():
    from app.rag.service import _PersonalHistoryResolution

    return _PersonalHistoryResolution(decomposed=None, file_ids=None)


def _stub_legacy_retrieval(monkeypatch, *, candidates=None):
    monkeypatch.setattr(
        "app.rag.service._resolve_personal_history",
        AsyncMock(return_value=_no_history()),
    )
    monkeypatch.setattr(
        "app.rag.service.transform_query_structured",
        AsyncMock(
            return_value=StructuredQuery(
                required=(), semantic=("kw",), raw_keywords="kw"
            )
        ),
    )
    retrieve = AsyncMock(return_value=candidates or [])
    monkeypatch.setattr("app.rag.service.retrieve_with_keywords", retrieve)
    monkeypatch.setattr(
        "app.rag.service.assemble_contexts", lambda *a, **k: []
    )
    return retrieve


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scope_is_resolved_before_the_semaphore_as_the_matching_filter(
    monkeypatch, ask_env
):
    semaphore = _reset_semaphore()
    seen: dict = {}

    async def _resolve(*args, **kwargs):
        seen["drive"] = args[0] if args else kwargs.get("drive")
        if len(args) > 1:
            seen["filter"] = args[1]
        else:
            seen["filter"] = kwargs.get("search_filter", kwargs.get("filter"))
        seen["free_slots"] = semaphore._value
        return ["s1"]

    monkeypatch.setattr(
        rag_module, "fetch_matching_file_ids", _resolve, raising=False
    )
    _stub_legacy_retrieval(monkeypatch)

    result = await ask_endpoint(
        body=AskRequest(query=QUESTION, scope=_scope_payload()),
        access_token="tok",
        drive=HOME,
    )
    await _collect_sse(result)

    assert seen["drive"] == HOME
    assert seen["free_slots"] == rag_module._MAX_CONCURRENT_ASK
    search_filter = seen["filter"]
    assert search_filter.folders == ["Work/2024"]
    assert search_filter.collections == ["読書メモ"]
    assert search_filter.tags == ["会議"]
    assert search_filter.kinds == ["text"]
    assert search_filter.favorite is None
    assert search_filter.liked is None


@pytest.mark.asyncio
async def test_failed_resolution_answers_503_scope_unavailable_without_a_stream(
    monkeypatch, ask_env
):
    semaphore = _reset_semaphore()

    async def _fail(*args, **kwargs):
        raise FilterUnavailable("ConnectError")

    monkeypatch.setattr(
        rag_module, "fetch_matching_file_ids", _fail, raising=False
    )

    with pytest.raises(HTTPException) as exc_info:
        await ask_endpoint(
            body=AskRequest(query=QUESTION, scope=_scope_payload()),
            access_token="tok",
            drive=HOME,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {
        "code": "scope_unavailable",
        "message": "Scope unavailable",
    }
    assert semaphore._value == rag_module._MAX_CONCURRENT_ASK


@pytest.mark.asyncio
async def test_resolution_failure_logs_the_drive_and_no_scope_value(
    monkeypatch, ask_env, caplog
):
    install(monkeypatch, MatchingCore(raises=httpx.ConnectError("boom")))

    with caplog.at_level(logging.WARNING):
        with pytest.raises(HTTPException) as exc_info:
            await ask_endpoint(
                body=AskRequest(query=QUESTION, scope=_scope_payload()),
                access_token="tok",
                drive=HOME,
            )

    assert exc_info.value.status_code == 503
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name.startswith(("app.file_matching", "app.routers.rag"))
    ]
    assert any(HOME in message for message in messages)
    for secret in (QUESTION, "Work/2024", "読書メモ", "会議"):
        assert all(secret not in message for message in messages)


# ---------------------------------------------------------------------------
# The scope event and the empty set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scoped_stream_opens_with_one_normalised_scope_event(
    monkeypatch, ask_env
):
    async def _resolve(*args, **kwargs):
        return ["s1", "s2"]

    monkeypatch.setattr(
        rag_module, "fetch_matching_file_ids", _resolve, raising=False
    )
    retrieve = _stub_legacy_retrieval(monkeypatch)

    result = await ask_endpoint(
        body=AskRequest(query=QUESTION, scope=_scope_payload()),
        access_token="tok",
        drive=HOME,
    )
    events = await _collect_sse(result)

    scope_events = [data for name, data in events if name == "scope"]
    assert len(scope_events) == 1
    assert events[0][0] == "scope"
    assert scope_events[0] == {
        "scope": {
            "folders": ["Work/2024"],
            "collections": ["読書メモ"],
            "tags": ["会議"],
            "kinds": ["text"],
        },
        "matched_file_count": 2,
    }
    names = [name for name, _ in events]
    assert "keywords" in names
    done = next(data for name, data in events if name == "done")
    assert "scope_empty" not in done
    assert "error" not in done
    # The scope event is a report, not the narrowing: the retriever itself
    # must rank inside S and must not receive the old post-filter scope.
    assert set(retrieve.await_args.kwargs["file_id_constraint"]) == {"s1", "s2"}
    assert retrieve.await_args.kwargs.get("file_id_scope") is None


@pytest.mark.asyncio
async def test_empty_resolved_set_answers_no_files_without_calling_an_llm(
    monkeypatch, ask_env
):
    llm_stub = ask_env

    async def _resolve(*args, **kwargs):
        return []

    monkeypatch.setattr(
        rag_module, "fetch_matching_file_ids", _resolve, raising=False
    )
    decompose = AsyncMock(side_effect=AssertionError("decomposition ran"))
    transform = AsyncMock(side_effect=AssertionError("keyword transform ran"))
    monkeypatch.setattr("app.rag.service.decompose_query", decompose)
    monkeypatch.setattr("app.rag.service.transform_query_structured", transform)

    result = await ask_endpoint(
        body=AskRequest(query=QUESTION, scope=_scope_payload()),
        access_token="tok",
        drive=HOME,
        viewer_id="viewer-1",
    )
    events = await _collect_sse(result)

    assert [name for name, _ in events] == ["scope", "citations", "done"]
    assert events[0][1]["matched_file_count"] == 0
    assert events[1][1] == {"citations": []}
    assert events[2][1]["retrieved_count"] == 0
    assert events[2][1]["scope_empty"] is True
    assert isinstance(events[2][1]["took_ms"], int)
    assert not decompose.called
    assert not transform.called
    assert not llm_stub.generate_stream.called


# ---------------------------------------------------------------------------
# Personal history (H4)
# ---------------------------------------------------------------------------


def _decomposed() -> DecomposedQuery:
    return DecomposedQuery(
        raw_query=QUESTION,
        time_range=TimeRange(label="last_week", after=datetime(2026, 1, 1), before=None),
        personal_scope="viewed",
        file_type_hint="none",
        semantic_query="meeting",
    )


def _stub_history(monkeypatch, history_ids: list[str]):
    monkeypatch.setattr(
        "app.rag.service.decompose_query", AsyncMock(return_value=_decomposed())
    )
    monkeypatch.setattr(
        "app.rag.service.fetch_viewer_history",
        AsyncMock(return_value=list(history_ids)),
    )


@pytest.mark.asyncio
async def test_personal_history_narrows_to_the_intersection_with_the_scope(
    monkeypatch, ask_env
):
    async def _resolve(*args, **kwargs):
        return ["s1", "s2"]

    monkeypatch.setattr(
        rag_module, "fetch_matching_file_ids", _resolve, raising=False
    )
    _stub_history(monkeypatch, ["s2", "h9"])
    monkeypatch.setattr(
        "app.rag.service.transform_query_structured",
        AsyncMock(
            return_value=StructuredQuery(
                required=(), semantic=("kw",), raw_keywords="kw"
            )
        ),
    )
    retrieve = AsyncMock(return_value=[])
    monkeypatch.setattr("app.rag.service.retrieve_with_keywords", retrieve)
    monkeypatch.setattr("app.rag.service.assemble_contexts", lambda *a, **k: [])

    result = await ask_endpoint(
        body=AskRequest(query=QUESTION, scope=_scope_payload()),
        access_token="tok",
        drive=HOME,
        viewer_id="viewer-1",
    )
    events = await _collect_sse(result)

    history = next(data for name, data in events if name == "history_filter")
    assert history["matched_file_count"] == 1
    call = retrieve.await_args
    assert set(call.kwargs["file_id_constraint"]) == {"s2"}
    assert call.kwargs.get("file_id_scope") is None


@pytest.mark.asyncio
async def test_history_empty_intersection_strict_short_circuits_with_zero(
    monkeypatch, ask_env_strict
):
    async def _resolve(*args, **kwargs):
        return ["s1", "s2"]

    monkeypatch.setattr(
        rag_module, "fetch_matching_file_ids", _resolve, raising=False
    )
    _stub_history(monkeypatch, ["h9"])
    retrieve = AsyncMock(return_value=[])
    monkeypatch.setattr("app.rag.service.retrieve_with_keywords", retrieve)

    result = await ask_endpoint(
        body=AskRequest(query=QUESTION, scope=_scope_payload()),
        access_token="tok",
        drive=HOME,
        viewer_id="viewer-1",
    )
    events = await _collect_sse(result)

    names = [name for name, _ in events]
    assert names[0] == "scope"
    history = next(data for name, data in events if name == "history_filter")
    assert history["matched_file_count"] == 0
    assert "keywords" not in names
    assert names[-2:] == ["citations", "done"]
    done = events[-1][1]
    assert "scope_empty" not in done
    assert not retrieve.called


@pytest.mark.asyncio
async def test_history_empty_intersection_graceful_falls_back_to_the_scope(
    monkeypatch, ask_env
):
    async def _resolve(*args, **kwargs):
        return ["s1", "s2"]

    monkeypatch.setattr(
        rag_module, "fetch_matching_file_ids", _resolve, raising=False
    )
    _stub_history(monkeypatch, ["h9"])
    monkeypatch.setattr(
        "app.rag.service.transform_query_structured",
        AsyncMock(
            return_value=StructuredQuery(
                required=(), semantic=("kw",), raw_keywords="kw"
            )
        ),
    )
    retrieve = AsyncMock(return_value=[])
    monkeypatch.setattr("app.rag.service.retrieve_with_keywords", retrieve)
    monkeypatch.setattr("app.rag.service.assemble_contexts", lambda *a, **k: [])

    result = await ask_endpoint(
        body=AskRequest(query=QUESTION, scope=_scope_payload()),
        access_token="tok",
        drive=HOME,
        viewer_id="viewer-1",
    )
    events = await _collect_sse(result)

    assert "history_filter" not in [name for name, _ in events]
    assert set(retrieve.await_args.kwargs["file_id_constraint"]) == {"s1", "s2"}


@pytest.mark.asyncio
async def test_unscoped_request_stays_unscoped_through_the_stream(
    monkeypatch, ask_env
):
    """SPEC-ADDON-054 I4: no scope in, no scope event and no constraint."""
    retrieve = _stub_legacy_retrieval(monkeypatch)

    result = await ask_endpoint(
        body=AskRequest(query=QUESTION),
        access_token="tok",
        drive=HOME,
    )
    events = await _collect_sse(result)

    names = [name for name, _ in events]
    assert "scope" not in names
    assert "keywords" in names
    assert retrieve.await_args.kwargs.get("file_id_constraint") is None
    done = next(data for name, data in events if name == "done")
    assert "scope_empty" not in done
    assert "error" not in done


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("history_ids", "expected"),
    [
        (None, ["s1", "s2", "s3"]),
        (["s2", "h9"], ["s2"]),
    ],
    ids=["no-history", "history-intersection"],
)
async def test_scoped_hierarchical_keeps_the_effective_set_everywhere(
    monkeypatch, make_settings, history_ids, expected
):
    """SPEC-ADDON-054 I1: the shortlist, exact match and clue retrievals all
    rank inside E, and the scoped clue retrieval carries no post-filter."""
    settings = make_settings(
        features=type(make_settings().features)(rag=True),
        llm=LLMConfig(provider="openai_compatible", model="llama3"),
        rag=RagConfig(
            hierarchical=HierarchicalRagConfig(
                enabled=True,
                coarse_top_k=5,
                coarse_score_threshold=0.3,
                min_drive_files_for_shortlist=1,
                exact_match_max_files=10,
                fallback_full_search=False,
            ),
            personal_history=PersonalHistoryConfig(
                enabled=True, fallback_when_empty="graceful"
            ),
        ),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.rag.service.settings", settings)

    if history_ids is None:
        monkeypatch.setattr(
            "app.rag.service._resolve_personal_history",
            AsyncMock(return_value=_no_history()),
        )
    else:
        _stub_history(monkeypatch, history_ids)
    monkeypatch.setattr(
        "app.rag.service.transform_query_structured",
        AsyncMock(
            return_value=StructuredQuery(
                required=(), semantic=("kw",), raw_keywords="kw"
            )
        ),
    )
    monkeypatch.setattr(
        "app.rag.service._filter_file_ids_via_internal_api",
        AsyncMock(side_effect=lambda ids, credential=None, **kw: set(ids)),
    )
    monkeypatch.setattr(
        "app.rag.service.fetch_long_summaries", lambda ids: {}
    )
    monkeypatch.setattr(
        "app.rag.service.generate_clues",
        AsyncMock(return_value=["clue a", "clue b"]),
    )

    coarse_calls: list[list[str]] = []

    async def _coarse(query, drive, *, top_k, file_id_constraint=None):
        constraint = list(file_id_constraint or [])
        coarse_calls.append(constraint)
        ids = tuple(constraint[:2])
        scores = tuple([0.9] * len(ids))
        return ShortlistResult(
            file_ids=ids,
            scores=scores,
            top_score=scores[0] if scores else 0.0,
            drive_file_count=len(constraint),
        )

    exact_calls: list[list[str]] = []

    async def _exact(keywords, drive, *, max_files, file_ids=None):
        exact_calls.append(list(file_ids or []))
        return ()

    monkeypatch.setattr("app.rag.service.coarse_retrieve", _coarse)
    monkeypatch.setattr("app.rag.service.lookup_exact_match_files", _exact)
    retrieve = AsyncMock(return_value=[])
    monkeypatch.setattr("app.rag.service.retrieve_with_keywords", retrieve)

    events = [
        event
        async for event in _stream_answer(
            query=QUESTION,
            credential=None,
            drive=HOME,
            viewer_id="viewer-1",
            scope={
                "folders": ["Work/2024"],
                "collections": ["読書メモ"],
                "tags": ["会議"],
                "kinds": ["text"],
            },
            scope_file_ids=["s1", "s2", "s3"],
        )
    ]

    assert coarse_calls == [expected]
    assert exact_calls == [expected]
    assert retrieve.await_args_list
    for call in retrieve.await_args_list:
        assert set(call.kwargs["file_id_constraint"]) == set(expected)
        assert call.kwargs.get("file_id_scope") is None

    shortlist_event = next(e for e in events if e.kind == "shortlist")
    assert set(shortlist_event.data["file_ids"]) <= set(expected)
    done = next(e for e in events if e.kind == "done")
    assert "error" not in done.data
