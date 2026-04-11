"""Tests for app.rag.query_transform.

The transform is a narrow wrapper around a single LLM call with
strict graceful-degradation semantics: on *any* failure mode the raw
query is returned unchanged so the RAG pipeline can always attempt
retrieval. These tests cover each failure mode individually plus the
happy path.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

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

from app.rag.query_transform import transform_query  # noqa: E402


def _llm_stub(
    *,
    enabled: bool = True,
    response: dict | list | None = None,
    raises: type[Exception] | None = None,
) -> MagicMock:
    """Build a MagicMock LLM client with a stubbed generate_json.

    ``enabled`` toggles the ``enabled`` property. ``response`` is what
    ``generate_json`` should return on success. ``raises`` causes
    ``generate_json`` to raise the given exception type (useful for
    simulating transport failures that slipped past the client's
    internal retry loop).
    """
    client = MagicMock()
    client.enabled = enabled
    if raises is not None:
        client.generate_json = AsyncMock(side_effect=raises("boom"))
    else:
        client.generate_json = AsyncMock(return_value=response)
    return client


class TestTransformQueryHappyPath:
    @pytest.mark.asyncio
    async def test_returns_keywords_field_from_llm(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response={"keywords": "おでかけ子ザメ"}),
        )

        result = await transform_query("おでかけ子ザメの共通点は？")

        assert result == "おでかけ子ザメ"

    @pytest.mark.asyncio
    async def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response={"keywords": "  京都 紅葉  "}),
        )

        result = await transform_query("京都の紅葉について教えて")

        assert result == "京都 紅葉"


class TestTransformQueryFallbacks:
    """Every failure mode must fall back to the original query."""

    @pytest.mark.asyncio
    async def test_falls_back_when_llm_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(enabled=False),
        )

        result = await transform_query("raw query")

        assert result == "raw query"

    @pytest.mark.asyncio
    async def test_falls_back_when_client_unavailable(self, monkeypatch):
        def _raise():
            raise RuntimeError("not initialized")

        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client", _raise
        )

        result = await transform_query("raw query")

        assert result == "raw query"

    @pytest.mark.asyncio
    async def test_falls_back_when_llm_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response=None),
        )

        result = await transform_query("raw query")

        assert result == "raw query"

    @pytest.mark.asyncio
    async def test_falls_back_when_llm_returns_list(self, monkeypatch):
        # The prompt instructs the LLM to return an object; a list is
        # a shape mismatch that should be treated as failure.
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response=["kw1", "kw2"]),
        )

        result = await transform_query("raw query")

        assert result == "raw query"

    @pytest.mark.asyncio
    async def test_falls_back_when_keywords_missing(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response={"other_field": "value"}),
        )

        result = await transform_query("raw query")

        assert result == "raw query"

    @pytest.mark.asyncio
    async def test_falls_back_when_keywords_empty(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response={"keywords": "   "}),
        )

        result = await transform_query("raw query")

        assert result == "raw query"

    @pytest.mark.asyncio
    async def test_falls_back_when_keywords_wrong_type(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response={"keywords": 123}),
        )

        result = await transform_query("raw query")

        assert result == "raw query"

    @pytest.mark.asyncio
    async def test_handles_empty_input(self, monkeypatch):
        # Whitespace-only input short-circuits without calling the LLM.
        spy_called = False

        def _get_client():
            nonlocal spy_called
            spy_called = True
            return _llm_stub(response={"keywords": "whatever"})

        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client", _get_client
        )

        result = await transform_query("   ")

        # Returns the input unchanged and did not waste an LLM call.
        assert result == "   "
        assert spy_called is False
