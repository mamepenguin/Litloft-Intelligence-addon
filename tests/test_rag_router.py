"""Tests for app.routers.rag module.

Tests the POST /ask endpoint: feature gating, LLM gating, query
validation, normal path, header forwarding, and error handling.

Handlers are invoked directly as async functions (mirroring the
existing test_webhook.py style). No TestClient required.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

# Stub out heavy dependencies before importing router / service.
for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "numpy",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.config import LLMConfig, RagConfig  # noqa: E402
from app.rag.service import AnswerResponse  # noqa: E402
from app.routers.rag import ask_endpoint  # noqa: E402
from app.schemas import AskRequest  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _answer_response(
    query: str = "test query",
    answer: str | None = "A generated answer.",
    citations: list[dict] | None = None,
    sources: list[dict] | None = None,
    retrieved_count: int = 2,
    took_ms: int = 123,
) -> AnswerResponse:
    # Use `is None` sentinel check instead of `or` so that an explicit
    # empty list [] from a caller is respected (needed for the
    # "null answer / empty citations" test case).
    default_citations = [
        {
            "file_id": "f1",
            "drive": "Videos",
            "filename": "a.mp4",
            "file_type": "video",
            "quote": "quoted text",
            "relevance": 0.9,
            "segment_location": "0:45",
        }
    ]
    default_sources = [
        {
            "file_id": "f1",
            "drive": "Videos",
            "filename": "a.mp4",
            "file_type": "video",
            "score": 0.9,
            "match_types": ["transcript"],
        },
        {
            "file_id": "f2",
            "drive": "Videos",
            "filename": "b.mp4",
            "file_type": "video",
            "score": 0.8,
            "match_types": ["transcript"],
        },
    ]
    return AnswerResponse(
        query=query,
        answer=answer,
        citations=default_citations if citations is None else citations,
        sources=default_sources if sources is None else sources,
        retrieved_count=retrieved_count,
        took_ms=took_ms,
    )


@pytest.fixture()
def rag_enabled(monkeypatch, make_settings):
    """Settings with features.rag=True and LLM enabled."""
    settings = make_settings(
        features=type(make_settings().features)(rag=True),
        llm=LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        ),
        rag=RagConfig(),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.rag.settings", settings)

    # Mock LLM client as enabled.
    llm_stub = MagicMock()
    llm_stub.enabled = True
    monkeypatch.setattr(
        "app.routers.rag.get_llm_client", lambda: llm_stub
    )
    return settings


@pytest.fixture()
def rag_disabled(monkeypatch, make_settings):
    """Settings with features.rag=False."""
    settings = make_settings(
        features=type(make_settings().features)(rag=False),
        llm=LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        ),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.rag.settings", settings)

    llm_stub = MagicMock()
    llm_stub.enabled = True
    monkeypatch.setattr(
        "app.routers.rag.get_llm_client", lambda: llm_stub
    )
    return settings


@pytest.fixture()
def llm_disabled(monkeypatch, make_settings):
    """Settings with features.rag=True but LLM disabled."""
    settings = make_settings(
        features=type(make_settings().features)(rag=True),
        llm=LLMConfig(provider="disabled"),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.routers.rag.settings", settings)

    llm_stub = MagicMock()
    llm_stub.enabled = False
    monkeypatch.setattr(
        "app.routers.rag.get_llm_client", lambda: llm_stub
    )
    return settings


# ---------------------------------------------------------------------------
# Feature / LLM gating
# ---------------------------------------------------------------------------


class TestAskEndpointFeatureGating:
    """T1: features.rag=False -> 400."""

    @pytest.mark.asyncio
    async def test_returns_400_when_rag_disabled(
        self, monkeypatch, rag_disabled
    ):
        body = AskRequest(query="a valid question")

        with pytest.raises(HTTPException) as exc_info:
            await ask_endpoint(body=body, access_token="tok")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_returns_400_when_llm_disabled(
        self, monkeypatch, llm_disabled
    ):
        """T2: LLM disabled -> 400 regardless of features.rag."""
        body = AskRequest(query="a valid question")

        with pytest.raises(HTTPException) as exc_info:
            await ask_endpoint(body=body, access_token="tok")

        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------


class TestAskEndpointQueryValidation:
    """Query length bounds enforced at the router layer."""

    @pytest.mark.asyncio
    async def test_returns_400_for_short_query(
        self, monkeypatch, rag_enabled
    ):
        """T3: query under 3 characters -> 400."""
        # Pydantic's min_length=1 allows single chars at the model level,
        # but the router enforces a stricter >= 3 character minimum.
        body = AskRequest(query="ab")

        with pytest.raises(HTTPException) as exc_info:
            await ask_endpoint(body=body, access_token="tok")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_returns_400_for_whitespace_only_query(
        self, monkeypatch, rag_enabled
    ):
        body = AskRequest(query="   ")

        with pytest.raises(HTTPException) as exc_info:
            await ask_endpoint(body=body, access_token="tok")

        assert exc_info.value.status_code == 400

    def test_query_over_1000_chars_rejected_by_pydantic(self):
        """T7: query > 1000 chars is rejected at Pydantic validation."""
        with pytest.raises(Exception):
            AskRequest(query="x" * 1001)

    def test_query_exactly_1000_chars_accepted(self):
        body = AskRequest(query="y" * 1000)
        assert len(body.query) == 1000


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestAskEndpointHappyPath:
    """T4: normal response shape."""

    @pytest.mark.asyncio
    async def test_returns_valid_response_shape(
        self, monkeypatch, rag_enabled
    ):
        monkeypatch.setattr(
            "app.routers.rag.answer_question",
            AsyncMock(return_value=_answer_response()),
        )

        body = AskRequest(query="What is the topic?")
        result = await ask_endpoint(body=body, access_token="tok")

        # The router returns a Pydantic model. Convert to dict for
        # shape assertions while being agnostic to the exact class.
        data = result.model_dump() if hasattr(result, "model_dump") else dict(result)

        assert "query" in data
        assert "answer" in data
        assert "citations" in data
        assert "sources" in data
        assert "retrieved_count" in data
        assert "took_ms" in data

        assert data["answer"] == "A generated answer."
        assert data["retrieved_count"] == 2
        assert len(data["sources"]) == 2
        assert len(data["citations"]) == 1
        assert data["citations"][0]["file_id"] == "f1"

    @pytest.mark.asyncio
    async def test_echoes_query_back(self, monkeypatch, rag_enabled):
        monkeypatch.setattr(
            "app.routers.rag.answer_question",
            AsyncMock(return_value=_answer_response(query="echoed")),
        )

        body = AskRequest(query="echoed back to caller")
        result = await ask_endpoint(body=body, access_token="tok")

        data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        assert data["query"] == "echoed"

    @pytest.mark.asyncio
    async def test_null_answer_serializes(
        self, monkeypatch, rag_enabled
    ):
        """When retrieval produced nothing, answer may be None."""
        monkeypatch.setattr(
            "app.routers.rag.answer_question",
            AsyncMock(return_value=_answer_response(
                answer=None,
                citations=[],
                sources=[],
                retrieved_count=0,
            )),
        )

        body = AskRequest(query="nothing to see here")
        result = await ask_endpoint(body=body, access_token="tok")

        data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        assert data["answer"] is None
        assert data["citations"] == []
        assert data["sources"] == []
        assert data["retrieved_count"] == 0


# ---------------------------------------------------------------------------
# Header / parameter forwarding
# ---------------------------------------------------------------------------


class TestAskEndpointHeaderForwarding:
    """T5: access_token cookie is passed into the service layer."""

    @pytest.mark.asyncio
    async def test_forwards_hv_token(self, monkeypatch, rag_enabled):
        spy = AsyncMock(return_value=_answer_response())
        monkeypatch.setattr("app.routers.rag.answer_question", spy)

        body = AskRequest(query="a valid query")
        await ask_endpoint(body=body, access_token="my-token-value")

        kwargs = spy.call_args.kwargs
        assert kwargs.get("hv_token") == "my-token-value"

    @pytest.mark.asyncio
    async def test_missing_hv_token_passes_none(
        self, monkeypatch, rag_enabled
    ):
        spy = AsyncMock(return_value=_answer_response())
        monkeypatch.setattr("app.routers.rag.answer_question", spy)

        body = AskRequest(query="a valid query")
        await ask_endpoint(body=body, access_token=None)

        kwargs = spy.call_args.kwargs
        assert kwargs.get("hv_token") is None


class TestAskEndpointParameterForwarding:
    """T6: top_k and filters are forwarded."""

    @pytest.mark.asyncio
    async def test_forwards_top_k(self, monkeypatch, rag_enabled):
        spy = AsyncMock(return_value=_answer_response())
        monkeypatch.setattr("app.routers.rag.answer_question", spy)

        body = AskRequest(query="a valid query", top_k=7)
        await ask_endpoint(body=body, access_token="t")

        kwargs = spy.call_args.kwargs
        assert kwargs.get("top_k") == 7

    @pytest.mark.asyncio
    async def test_forwards_file_type_and_drive(
        self, monkeypatch, rag_enabled
    ):
        spy = AsyncMock(return_value=_answer_response())
        monkeypatch.setattr("app.routers.rag.answer_question", spy)

        body = AskRequest(
            query="a valid query",
            file_type="video",
            drive="Videos",
        )
        await ask_endpoint(body=body, access_token="t")

        kwargs = spy.call_args.kwargs
        assert kwargs.get("file_type") == "video"
        assert kwargs.get("drive") == "Videos"


# ---------------------------------------------------------------------------
# Internal errors
# ---------------------------------------------------------------------------


class TestAskEndpointInternalErrors:
    """T8: unexpected exceptions surface as 500."""

    @pytest.mark.asyncio
    async def test_service_exception_returns_500(
        self, monkeypatch, rag_enabled
    ):
        async def _raise(*args, **kwargs):
            raise RuntimeError("internal boom")

        monkeypatch.setattr("app.routers.rag.answer_question", _raise)

        body = AskRequest(query="a valid query")

        with pytest.raises(HTTPException) as exc_info:
            await ask_endpoint(body=body, access_token="tok")

        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# AskRequest schema
# ---------------------------------------------------------------------------


class TestAskRequestSchema:
    """Sanity checks for the AskRequest Pydantic model."""

    def test_defaults(self):
        body = AskRequest(query="valid query")
        assert body.query == "valid query"
        assert body.top_k is None
        assert body.file_type is None
        assert body.drive is None

    def test_top_k_must_be_in_range(self):
        with pytest.raises(Exception):
            AskRequest(query="q", top_k=0)
        with pytest.raises(Exception):
            AskRequest(query="q", top_k=21)

    def test_top_k_valid_range(self):
        assert AskRequest(query="q", top_k=1).top_k == 1
        assert AskRequest(query="q", top_k=20).top_k == 20

    def test_empty_query_rejected(self):
        with pytest.raises(Exception):
            AskRequest(query="")
