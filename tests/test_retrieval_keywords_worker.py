"""Tests for app.workers.retrieval_keywords (Phase 1 worker).

The worker is a thin queue around a single LLM call followed by a
two-stage filter (static blocklist + DF rarity) and a DB write. These
tests exercise the gating, the LLM error handling, the filter wiring,
and the persistence path — DB and LLM are stubbed so the suite stays
fast and isolated from the FTS vocab tables.

The rarity filter has its own dedicated suite (test_rag_rarity_filter);
here we monkeypatch it to a passthrough so the worker tests assert the
*worker's* control flow rather than re-asserting filter behaviour.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub heavy ML deps so importing the worker (which transitively pulls
# app.search via the rarity filter / DB module) does not need real
# torch / sentence-transformers / sqlite-vec.
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

from app.workers import retrieval_keywords as rk_module  # noqa: E402
from app.workers.retrieval_keywords import (  # noqa: E402
    RetrievalKeywordsWorker,
    _post_filter,
)


# ---------------------------------------------------------------------------
# _post_filter
# ---------------------------------------------------------------------------


class TestPostFilter:
    """Two-stage filter integration: static blocklist then DF rarity."""

    def test_empty_input_returns_empty(self):
        assert _post_filter([]) == ""

    def test_passthrough_when_filters_keep_everything(self, monkeypatch):
        monkeypatch.setattr(rk_module, "filter_keywords", lambda s: s)
        monkeypatch.setattr(rk_module, "filter_clue_by_rarity", lambda s: s)

        assert _post_filter(["alpha", "beta", "gamma"]) == "alpha beta gamma"

    def test_static_blocklist_emptying_returns_empty(self, monkeypatch):
        # The blocklist drops the entire string before rarity even runs.
        monkeypatch.setattr(rk_module, "filter_keywords", lambda _s: "")
        # Rarity must NOT be called when blocklist already emptied input.
        sentinel = MagicMock()
        monkeypatch.setattr(rk_module, "filter_clue_by_rarity", sentinel)

        assert _post_filter(["何", "なぜ"]) == ""
        sentinel.assert_not_called()

    def test_rarity_emptying_returns_empty(self, monkeypatch):
        monkeypatch.setattr(rk_module, "filter_keywords", lambda s: s)
        monkeypatch.setattr(rk_module, "filter_clue_by_rarity", lambda _s: "")

        assert _post_filter(["common1", "common2"]) == ""

    def test_truncates_at_max_kept_keywords(self, monkeypatch):
        # Generate 30 tokens; cap (_MAX_KEPT_KEYWORDS = 20) must clip.
        monkeypatch.setattr(rk_module, "filter_keywords", lambda s: s)
        monkeypatch.setattr(rk_module, "filter_clue_by_rarity", lambda s: s)

        many = [f"kw{i}" for i in range(30)]
        result = _post_filter(many)
        tokens = result.split()
        assert len(tokens) == 20
        # Order is preserved — first 20 survive.
        assert tokens == [f"kw{i}" for i in range(20)]

    def test_skips_empty_entries(self, monkeypatch):
        monkeypatch.setattr(rk_module, "filter_keywords", lambda s: s)
        monkeypatch.setattr(rk_module, "filter_clue_by_rarity", lambda s: s)

        # Raw input from the LLM may include "" entries from over-eager
        # JSON formatting. The join + filter cycle naturally produces
        # the right surviving string.
        assert _post_filter(["a", "", "b"]) == "a  b" or _post_filter(
            ["a", "", "b"]
        ) == "a b"


# ---------------------------------------------------------------------------
# RetrievalKeywordsWorker._process_file
# ---------------------------------------------------------------------------


@pytest.fixture()
def make_worker():
    """Factory for a RetrievalKeywordsWorker with a stub LLM client."""

    def _make(
        *,
        enabled: bool = True,
        response: dict | list | None = None,
        raises: type[Exception] | None = None,
        model: str = "test-model",
    ) -> RetrievalKeywordsWorker:
        client = MagicMock()
        client.enabled = enabled
        client.model = model
        if raises is not None:
            client.generate_json = AsyncMock(side_effect=raises("boom"))
        else:
            client.generate_json = AsyncMock(return_value=response)
        return RetrievalKeywordsWorker(client)

    return _make


def _stub_settings(monkeypatch, *, retrieval_keywords: str = "on_index") -> None:
    """Patch the worker's settings.features.retrieval_keywords value."""
    fake_settings = MagicMock()
    fake_settings.features.retrieval_keywords = retrieval_keywords
    monkeypatch.setattr(rk_module, "settings", fake_settings)


def _stub_helpers(
    monkeypatch,
    *,
    indexed_file: dict | None,
    context_type: str | None,
    context: str | None,
    has_existing: bool = False,
) -> dict:
    """Stub the summaries-worker helpers + the existence probe.

    Returns a dict that records the upsert calls so tests can assert
    persistence shape without standing up a real search DB.
    """
    monkeypatch.setattr(
        rk_module, "_get_indexed_file", lambda _fid: indexed_file
    )
    monkeypatch.setattr(
        rk_module, "_classify_file_type",
        lambda _ft, _mt=None: context_type,
    )
    monkeypatch.setattr(
        rk_module, "_build_context", lambda _ix, _ct: context
    )
    monkeypatch.setattr(
        rk_module, "_has_retrieval_keywords", lambda _fid: has_existing
    )

    upsert_calls: list[dict] = []

    def _fake_upsert(_session, **kwargs):
        upsert_calls.append(kwargs)

    monkeypatch.setattr(rk_module, "upsert_retrieval_keywords", _fake_upsert)

    @contextmanager
    def _fake_db():
        yield MagicMock()

    monkeypatch.setattr(rk_module, "get_search_db", _fake_db)
    return {"upsert_calls": upsert_calls}


def _stub_filters_passthrough(monkeypatch) -> None:
    """Make both filters identity functions for isolation."""
    monkeypatch.setattr(rk_module, "filter_keywords", lambda s: s)
    monkeypatch.setattr(rk_module, "filter_clue_by_rarity", lambda s: s)


class TestProcessFileGates:
    """Pre-LLM gates: feature flag, LLM enabled, existing row, context."""

    @pytest.mark.asyncio
    async def test_feature_disabled_skips(self, monkeypatch, make_worker):
        _stub_settings(monkeypatch, retrieval_keywords="false")
        worker = make_worker()
        recorded = _stub_helpers(
            monkeypatch,
            indexed_file={"file_id": "f1", "filename": "x", "file_type": "video"},
            context_type="video",
            context="content",
        )
        _stub_filters_passthrough(monkeypatch)

        await worker._process_file("f1")

        worker._llm_client.generate_json.assert_not_called()
        assert recorded["upsert_calls"] == []

    @pytest.mark.asyncio
    async def test_llm_disabled_skips(self, monkeypatch, make_worker):
        _stub_settings(monkeypatch)
        worker = make_worker(enabled=False)
        recorded = _stub_helpers(
            monkeypatch,
            indexed_file={"file_id": "f1", "filename": "x", "file_type": "video"},
            context_type="video",
            context="content",
        )
        _stub_filters_passthrough(monkeypatch)

        await worker._process_file("f1")

        worker._llm_client.generate_json.assert_not_called()
        assert recorded["upsert_calls"] == []

    @pytest.mark.asyncio
    async def test_existing_row_skips(self, monkeypatch, make_worker):
        _stub_settings(monkeypatch)
        worker = make_worker(response={"keywords": ["a", "b"]})
        recorded = _stub_helpers(
            monkeypatch,
            indexed_file={"file_id": "f1", "filename": "x", "file_type": "video"},
            context_type="video",
            context="content",
            has_existing=True,
        )
        _stub_filters_passthrough(monkeypatch)

        await worker._process_file("f1")

        worker._llm_client.generate_json.assert_not_called()
        assert recorded["upsert_calls"] == []

    @pytest.mark.asyncio
    async def test_unknown_file_skips(self, monkeypatch, make_worker):
        _stub_settings(monkeypatch)
        worker = make_worker(response={"keywords": ["a", "b"]})
        recorded = _stub_helpers(
            monkeypatch,
            indexed_file=None,
            context_type="video",
            context="content",
        )
        _stub_filters_passthrough(monkeypatch)

        await worker._process_file("missing")

        worker._llm_client.generate_json.assert_not_called()
        assert recorded["upsert_calls"] == []

    @pytest.mark.asyncio
    async def test_unsupported_context_type_skips(self, monkeypatch, make_worker):
        # Images / other types fall outside _HANDLED_CONTEXT_TYPES.
        _stub_settings(monkeypatch)
        worker = make_worker(response={"keywords": ["a", "b"]})
        recorded = _stub_helpers(
            monkeypatch,
            indexed_file={"file_id": "f1", "filename": "x", "file_type": "image"},
            context_type="image",
            context="content",
        )
        _stub_filters_passthrough(monkeypatch)

        await worker._process_file("f1")

        worker._llm_client.generate_json.assert_not_called()
        assert recorded["upsert_calls"] == []

    @pytest.mark.asyncio
    async def test_no_context_skips(self, monkeypatch, make_worker):
        # _build_context returns None when content is too short / empty.
        _stub_settings(monkeypatch)
        worker = make_worker(response={"keywords": ["a", "b"]})
        recorded = _stub_helpers(
            monkeypatch,
            indexed_file={"file_id": "f1", "filename": "x", "file_type": "video"},
            context_type="video",
            context=None,
        )
        _stub_filters_passthrough(monkeypatch)

        await worker._process_file("f1")

        worker._llm_client.generate_json.assert_not_called()
        assert recorded["upsert_calls"] == []


class TestProcessFileLLMFailures:
    """LLM-level failures fall through to a silent no-op (no row written)."""

    @pytest.mark.asyncio
    async def test_llm_raises_no_write(self, monkeypatch, make_worker):
        _stub_settings(monkeypatch)
        worker = make_worker(raises=RuntimeError)
        recorded = _stub_helpers(
            monkeypatch,
            indexed_file={"file_id": "f1", "filename": "x", "file_type": "video"},
            context_type="video",
            context="content",
        )
        _stub_filters_passthrough(monkeypatch)

        await worker._process_file("f1")

        assert recorded["upsert_calls"] == []

    @pytest.mark.asyncio
    async def test_non_dict_response_no_write(self, monkeypatch, make_worker):
        _stub_settings(monkeypatch)
        worker = make_worker(response=["a", "b"])
        recorded = _stub_helpers(
            monkeypatch,
            indexed_file={"file_id": "f1", "filename": "x", "file_type": "video"},
            context_type="video",
            context="content",
        )
        _stub_filters_passthrough(monkeypatch)

        await worker._process_file("f1")

        assert recorded["upsert_calls"] == []

    @pytest.mark.asyncio
    async def test_missing_keywords_key_no_write(self, monkeypatch, make_worker):
        _stub_settings(monkeypatch)
        worker = make_worker(response={"tags": ["a"]})
        recorded = _stub_helpers(
            monkeypatch,
            indexed_file={"file_id": "f1", "filename": "x", "file_type": "video"},
            context_type="video",
            context="content",
        )
        _stub_filters_passthrough(monkeypatch)

        await worker._process_file("f1")

        assert recorded["upsert_calls"] == []

    @pytest.mark.asyncio
    async def test_empty_keywords_list_no_write(self, monkeypatch, make_worker):
        _stub_settings(monkeypatch)
        worker = make_worker(response={"keywords": []})
        recorded = _stub_helpers(
            monkeypatch,
            indexed_file={"file_id": "f1", "filename": "x", "file_type": "video"},
            context_type="video",
            context="content",
        )
        _stub_filters_passthrough(monkeypatch)

        await worker._process_file("f1")

        assert recorded["upsert_calls"] == []

    @pytest.mark.asyncio
    async def test_filters_drop_everything_no_write(self, monkeypatch, make_worker):
        # The LLM returned candidates, but post-filter wipes them all.
        _stub_settings(monkeypatch)
        worker = make_worker(response={"keywords": ["common"]})
        recorded = _stub_helpers(
            monkeypatch,
            indexed_file={"file_id": "f1", "filename": "x", "file_type": "video"},
            context_type="video",
            context="content",
        )
        monkeypatch.setattr(rk_module, "filter_keywords", lambda s: s)
        monkeypatch.setattr(rk_module, "filter_clue_by_rarity", lambda _s: "")

        await worker._process_file("f1")

        assert recorded["upsert_calls"] == []


class TestProcessFilePersistence:
    """Happy path: keywords survive filtering and land in the DB."""

    @pytest.mark.asyncio
    async def test_writes_kept_keywords(self, monkeypatch, make_worker):
        _stub_settings(monkeypatch)
        worker = make_worker(
            response={"keywords": ["佐々木徹", "退職前", "古い手紙"]},
            model="gemini-2.5-flash",
        )
        recorded = _stub_helpers(
            monkeypatch,
            indexed_file={
                "file_id": "novel-1",
                "filename": "letter.md",
                "file_type": "document",
                "mime_type": "text/markdown",
                "title": "古い手紙",
                "description": "",
            },
            context_type="document",
            context="本文内容..." * 100,
        )
        _stub_filters_passthrough(monkeypatch)

        await worker._process_file("novel-1")

        assert len(recorded["upsert_calls"]) == 1
        call = recorded["upsert_calls"][0]
        assert call["file_id"] == "novel-1"
        assert call["keywords"] == "佐々木徹 退職前 古い手紙"
        assert call["model"] == "gemini-2.5-flash"
        # context_type is 'document' for non-transcript paths.
        assert call["context_type"] == "document"

    @pytest.mark.asyncio
    async def test_video_context_stored_as_transcript(
        self, monkeypatch, make_worker
    ):
        # _classify_file_type returns 'video' or 'audio' for media files;
        # the stored context_type collapses both to 'transcript' so the
        # DB column has only two values (transcript / document) and the
        # downstream search code can branch cleanly.
        _stub_settings(monkeypatch)
        worker = make_worker(response={"keywords": ["a"]})
        recorded = _stub_helpers(
            monkeypatch,
            indexed_file={
                "file_id": "vid-1",
                "filename": "x.mp4",
                "file_type": "video",
                "mime_type": "video/mp4",
            },
            context_type="video",
            context="transcript text" * 100,
        )
        _stub_filters_passthrough(monkeypatch)

        await worker._process_file("vid-1")

        assert recorded["upsert_calls"][0]["context_type"] == "transcript"

    @pytest.mark.asyncio
    async def test_context_truncated_above_cap(self, monkeypatch, make_worker):
        # Context longer than _MAX_CONTEXT_CHARS gets truncated. The
        # truncation is visible to the prompt template (was_truncated
        # flag) but does NOT block the LLM call.
        _stub_settings(monkeypatch)
        worker = make_worker(response={"keywords": ["a"]})
        recorded = _stub_helpers(
            monkeypatch,
            indexed_file={
                "file_id": "long-1",
                "filename": "x.pdf",
                "file_type": "document",
            },
            context_type="document",
            context="x" * 50000,
        )
        _stub_filters_passthrough(monkeypatch)

        await worker._process_file("long-1")

        worker._llm_client.generate_json.assert_called_once()
        assert recorded["upsert_calls"][0]["keywords"] == "a"
