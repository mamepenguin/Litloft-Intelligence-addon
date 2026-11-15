"""SPEC-ADDON-110 step 6, I8; SPEC-ADDON-054 I3 and I5 as revised.

``AskScope.file_ids`` carries a Smart Collection's members. The scope set is
``B ∪ (A ∩ file_ids)``, where ``A`` is ``files/matching`` without collections
and ``B`` is the same call with them; ``file_ids`` only ever keeps what ``A``
returns for the request's drive.
"""

from __future__ import annotations

import json
import logging
import sys
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

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

import app.routers.rag as rag_router  # noqa: E402
import app.schemas as schemas  # noqa: E402
from app.config import (  # noqa: E402
    HierarchicalRagConfig,
    LLMConfig,
    PersonalHistoryConfig,
    RagConfig,
)
from app.rag.query_transform import StructuredQuery  # noqa: E402
from app.routers import rag as rag_module  # noqa: E402
from app.routers.rag import ask_endpoint  # noqa: E402
from app.schemas import AskRequest  # noqa: E402

HOME = "family"
QUESTION = "what did we decide in the meeting"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected_ids"),
    [
        ({"file_ids": []}, []),
        ({"file_ids": ["a", "b"]}, ["a", "b"]),
        ({"file_ids": [f"f{i}" for i in range(10_000)]}, [f"f{i}" for i in range(10_000)]),
        ({"folders": ["Work"], "file_ids": ["a"]}, ["a"]),
    ],
    ids=["empty-list-is-a-condition", "two-ids", "at-the-cap", "beside-a-folder"],
)
def test_spec_addon_110_scope_accepts_file_ids(payload, expected_ids):
    body = AskRequest(query=QUESTION, scope=payload)

    assert body.scope.model_dump()["file_ids"] == expected_ids


def test_spec_addon_110_file_ids_are_absent_by_default():
    body = AskRequest(query=QUESTION, scope={"folders": ["Work"]})

    assert body.scope.file_ids is None


def test_spec_addon_110_file_ids_are_capped_at_ten_thousand():
    schemas.AskScope.model_validate({"file_ids": [f"f{i}" for i in range(10_000)]})

    with pytest.raises(ValidationError):
        schemas.AskScope.model_validate({"file_ids": [f"f{i}" for i in range(10_001)]})


def test_spec_addon_054_i3_fastapi_answers_422_for_too_many_file_ids():
    app = FastAPI()
    app.include_router(rag_router.router)

    response = TestClient(app).post(
        "/ask",
        headers={"X-Lit-Drive": HOME},
        json={
            "query": QUESTION,
            "scope": {"file_ids": [f"f{i}" for i in range(10_001)]},
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert isinstance(body["detail"], list)
    assert body["detail"][0]["loc"][:2] == ["body", "scope"]
    assert all(error["type"] != "extra_forbidden" for error in body["detail"])


# ---------------------------------------------------------------------------
# Resolution: S = B ∪ (A ∩ file_ids)
# ---------------------------------------------------------------------------


class SplitMatchingCore(MatchingCore):
    """Answers ``A`` for a body without collections and ``B`` for one with them."""

    def __init__(self, *, without_collections, with_collections=(), **kwargs):
        super().__init__(**kwargs)
        self.without_collections = list(without_collections)
        self.with_collections = list(with_collections)

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/internal/files/matching"):
            sent = json.loads(request.content or b"{}")
            self.file_ids = (
                self.with_collections if sent.get("collections") else self.without_collections
            )
        return super().handle(request)


def _bodies(core: MatchingCore) -> list[dict]:
    return [json.loads(request.content) for request in core.matching_requests]


@pytest.fixture()
def ask_env(monkeypatch, make_settings):
    settings = make_settings(
        features=type(make_settings().features)(rag=True),
        llm=LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        ),
        rag=RagConfig(
            hierarchical=HierarchicalRagConfig(enabled=False),
            personal_history=PersonalHistoryConfig(
                enabled=True, fallback_when_empty="graceful"
            ),
        ),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.rag.settings", settings)
    monkeypatch.setattr("app.rag.service.settings", settings)
    llm_stub = MagicMock()
    llm_stub.enabled = True
    bind_llm(monkeypatch, lambda: llm_stub)
    rag_module._ask_semaphore = None
    rag_module._ask_semaphore_loop = None
    return llm_stub


def _stub_retrieval(monkeypatch):
    from app.rag.service import _PersonalHistoryResolution

    monkeypatch.setattr(
        "app.rag.service._resolve_personal_history",
        AsyncMock(return_value=_PersonalHistoryResolution(decomposed=None, file_ids=None)),
    )
    monkeypatch.setattr(
        "app.rag.service.transform_query_structured",
        AsyncMock(
            return_value=StructuredQuery(required=(), semantic=("kw",), raw_keywords="kw")
        ),
    )
    retrieve = AsyncMock(return_value=[])
    monkeypatch.setattr("app.rag.service.retrieve_with_keywords", retrieve)
    monkeypatch.setattr("app.rag.service.assemble_contexts", lambda *a, **k: [])
    return retrieve


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


async def _ask(scope: dict) -> list[tuple[str, dict]]:
    result = await ask_endpoint(
        body=AskRequest(query=QUESTION, scope=scope),
        access_token="tok",
        drive=HOME,
    )
    return await _collect_sse(result)


def _scope_event(events):
    scope_events = [data for name, data in events if name == "scope"]
    assert len(scope_events) == 1
    assert events[0][0] == "scope"
    return scope_events[0]


@pytest.mark.asyncio
async def test_spec_addon_110_file_ids_alone_are_intersected_with_the_drives_active_files(
    monkeypatch, ask_env
):
    core = install(monkeypatch, SplitMatchingCore(without_collections=["a", "b", "c"]))
    retrieve = _stub_retrieval(monkeypatch)

    events = await _ask({"file_ids": ["b", "elsewhere", "c"]})

    scope = _scope_event(events)
    assert scope == {
        "scope": {"folders": [], "collections": [], "tags": [], "kinds": []},
        "matched_file_count": 2,
    }
    assert set(retrieve.await_args.kwargs["file_id_constraint"]) == {"b", "c"}
    bodies = _bodies(core)
    assert bodies
    assert all(body["drive"] == HOME for body in bodies)
    assert all(not body.get("collections") for body in bodies)


@pytest.mark.asyncio
async def test_spec_addon_110_file_ids_beside_a_folder_and_a_tag_keep_only_ids_inside_them(
    monkeypatch, ask_env
):
    core = install(monkeypatch, SplitMatchingCore(without_collections=["a", "b"]))
    retrieve = _stub_retrieval(monkeypatch)

    events = await _ask({"folders": ["/Work/"], "tags": [" 会議 "], "file_ids": ["b", "z"]})

    scope = _scope_event(events)
    assert scope["scope"] == {"folders": ["Work"], "collections": [], "tags": ["会議"], "kinds": []}
    assert scope["matched_file_count"] == 1
    assert set(retrieve.await_args.kwargs["file_id_constraint"]) == {"b"}
    a_body = next(body for body in _bodies(core) if not body.get("collections"))
    assert a_body["folders"] == ["Work"]
    assert a_body["tags"] == ["会議"]


@pytest.mark.asyncio
async def test_spec_addon_110_collections_and_file_ids_give_the_union(monkeypatch, ask_env):
    core = install(
        monkeypatch,
        SplitMatchingCore(without_collections=["a", "b", "c"], with_collections=["x", "y"]),
    )
    retrieve = _stub_retrieval(monkeypatch)

    events = await _ask({"collections": ["読書メモ"], "file_ids": ["b", "q", "x"]})

    scope = _scope_event(events)
    assert scope["scope"] == {"folders": [], "collections": ["読書メモ"], "tags": [], "kinds": []}
    assert scope["matched_file_count"] == 3
    assert set(retrieve.await_args.kwargs["file_id_constraint"]) == {"x", "y", "b"}
    collections_sent = sorted(tuple(body.get("collections") or []) for body in _bodies(core))
    assert collections_sent == [(), ("読書メモ",)]


@pytest.mark.asyncio
async def test_spec_addon_054_i5_an_empty_file_ids_list_is_an_empty_scope(monkeypatch, ask_env):
    llm_stub = ask_env
    install(monkeypatch, SplitMatchingCore(without_collections=["a", "b"]))
    transform = AsyncMock(side_effect=AssertionError("keyword transform ran"))
    monkeypatch.setattr("app.rag.service.transform_query_structured", transform)
    monkeypatch.setattr(
        "app.rag.service.decompose_query",
        AsyncMock(side_effect=AssertionError("decomposition ran")),
    )

    events = await _ask({"file_ids": []})

    assert [name for name, _ in events] == ["scope", "citations", "done"]
    assert events[0][1]["matched_file_count"] == 0
    assert events[1][1] == {"citations": []}
    assert events[2][1]["scope_empty"] is True
    assert not transform.called
    assert not llm_stub.generate_stream.called


@pytest.mark.asyncio
async def test_spec_addon_054_i5_ids_outside_the_drive_give_an_empty_scope(monkeypatch, ask_env):
    install(monkeypatch, SplitMatchingCore(without_collections=["a", "b"]))
    _stub_retrieval(monkeypatch)

    events = await _ask({"file_ids": ["other-drive-1", "other-drive-2"]})

    assert events[0][1]["matched_file_count"] == 0
    done = next(data for name, data in events if name == "done")
    assert done["scope_empty"] is True


@pytest.mark.asyncio
async def test_spec_addon_054_i5_a_non_empty_scope_has_no_scope_empty_flag(monkeypatch, ask_env):
    install(monkeypatch, SplitMatchingCore(without_collections=["a"]))
    _stub_retrieval(monkeypatch)

    events = await _ask({"file_ids": ["a"]})

    done = next(data for name, data in events if name == "done")
    assert "scope_empty" not in done


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [{"file_ids": ["a"]}, {"collections": ["読書メモ"], "file_ids": ["a"]}],
    ids=["file-ids-only", "with-collections"],
)
async def test_spec_addon_054_i3_a_failed_matching_call_answers_scope_unavailable(
    monkeypatch, ask_env, scope
):
    install(monkeypatch, MatchingCore(raises=httpx.ConnectError("boom")))
    semaphore = rag_module._get_ask_semaphore()

    with pytest.raises(HTTPException) as exc_info:
        await ask_endpoint(
            body=AskRequest(query=QUESTION, scope=scope),
            access_token="tok",
            drive=HOME,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {"code": "scope_unavailable", "message": "Scope unavailable"}
    assert semaphore._value == rag_module._MAX_CONCURRENT_ASK


@pytest.mark.asyncio
async def test_spec_addon_054_i3_a_failing_second_call_also_answers_scope_unavailable(
    monkeypatch, ask_env
):
    class FailsWithCollections(SplitMatchingCore):
        def handle(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/api/internal/files/matching"):
                sent = json.loads(request.content or b"{}")
                if sent.get("collections"):
                    raise httpx.ConnectError("boom")
            return super().handle(request)

    install(monkeypatch, FailsWithCollections(without_collections=["a"]))

    with pytest.raises(HTTPException) as exc_info:
        await ask_endpoint(
            body=AskRequest(query=QUESTION, scope={"collections": ["読書メモ"], "file_ids": ["a"]}),
            access_token="tok",
            drive=HOME,
        )

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_spec_addon_054_i5_no_log_line_carries_a_file_id(monkeypatch, ask_env, caplog):
    install(monkeypatch, MatchingCore(raises=httpx.ConnectError("boom")))

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(HTTPException):
            await ask_endpoint(
                body=AskRequest(query=QUESTION, scope={"file_ids": ["fileid-secret-1", "fileid-secret-2"]}),
                access_token="tok",
                drive=HOME,
            )

    messages = [record.getMessage() for record in caplog.records]
    for secret in ("fileid-secret-1", "fileid-secret-2", QUESTION):
        assert all(secret not in message for message in messages)
