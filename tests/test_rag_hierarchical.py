"""Tests for hierarchical RAG flow in app.rag.service.

Phase 1 (shadow) is verified via the absence of side effects on
retrieval when the master switch is off; Phase 2 verifies the actual
scope flag propagation, bypass conditions, fallback merge, and the
new ``shortlist`` SSE event.
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
    "numpy",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.config import HierarchicalRagConfig, LLMConfig, RagConfig  # noqa: E402
from app.rag.coarse_retriever import ShortlistResult  # noqa: E402
from app.rag.context import ContextSnippet, FileContext  # noqa: E402
from app.rag.retriever import RetrievedFile  # noqa: E402
from app.rag.service import AnswerEvent, stream_answer  # noqa: E402
from app.search import MatchInfo, SegmentGroup  # noqa: E402


def _retrieved(file_id: str, score: float = 0.9) -> RetrievedFile:
    match = MatchInfo(
        match_type="transcript",
        text="snippet",
        score=score,
        timestamp_start=0.0,
        timestamp_end=10.0,
    )
    return RetrievedFile(
        file_id=file_id,
        drive="Videos",
        filename=f"{file_id}.mp4",
        file_type="video",
        title=f"Title {file_id}",
        description=f"Description {file_id}",
        score=score,
        match_types=("transcript",),
        segments=(SegmentGroup(time_range=(0.0, 10.0), matches=(match,)),),
    )


def _context(file_id: str) -> FileContext:
    return FileContext(
        file_id=file_id,
        filename=f"{file_id}.mp4",
        drive="Videos",
        file_type="video",
        title=f"Title {file_id}",
        description=f"Description {file_id}",
        snippets=(
            ContextSnippet(source="transcript", text="t", location="0:01"),
        ),
        total_chars=1,
    )


def _make_stream_llm():
    client = MagicMock()
    client.enabled = True

    async def _stream(*args, **kwargs):
        yield '{"answer": "ans", "citations": []}'

    client.generate_stream = _stream
    return client


def _settings_with_hierarchical(make_settings, *, enabled, **overrides):
    hier = HierarchicalRagConfig(enabled=enabled, **overrides)
    return make_settings(
        features=type(make_settings().features)(rag=True),
        llm=LLMConfig(provider="openai_compatible", model="m"),
        rag=RagConfig(hierarchical=hier),
    )


@pytest.fixture()
def common_patches(monkeypatch):
    monkeypatch.setattr(
        "app.rag.service.transform_query",
        AsyncMock(return_value="kw"),
    )
    monkeypatch.setattr(
        "app.rag.service.assemble_contexts",
        lambda cands, cfg, **_kw: [_context(c.file_id) for c in cands],
    )
    monkeypatch.setattr(
        "app.rag.service.get_llm_client", lambda: _make_stream_llm()
    )
    # Default: access filter is a no-op pass-through. Tests that need
    # to exercise the gate override this with their own AsyncMock.
    monkeypatch.setattr(
        "app.rag.service._filter_file_ids_via_internal_api",
        AsyncMock(side_effect=lambda ids, _token: set(ids)),
    )


async def _collect(gen) -> list[AnswerEvent]:
    events: list[AnswerEvent] = []
    async for event in gen:
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Hierarchical disabled
# ---------------------------------------------------------------------------


class TestHierarchicalDisabled:
    @pytest.mark.asyncio
    async def test_legacy_path_no_coarse_call(
        self, monkeypatch, make_settings, common_patches
    ):
        s = _settings_with_hierarchical(make_settings, enabled=False)
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        coarse_spy = AsyncMock()
        monkeypatch.setattr("app.rag.service.coarse_retrieve", coarse_spy)
        retrieve_spy = AsyncMock(return_value=[_retrieved("f1")])
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(query="q", lit_token="t", drive="Videos")
        )

        coarse_spy.assert_not_awaited()
        # legacy retrieve called with file_id_scope=None
        retrieve_spy.assert_awaited_once()
        kwargs = retrieve_spy.await_args.kwargs
        assert kwargs.get("file_id_scope") is None
        # No shortlist event emitted.
        assert "shortlist" not in [e.kind for e in events]


# ---------------------------------------------------------------------------
# Drive missing — bypass even when enabled
# ---------------------------------------------------------------------------


class TestHierarchicalNoDrive:
    @pytest.mark.asyncio
    async def test_no_drive_means_no_coarse(
        self, monkeypatch, make_settings, common_patches
    ):
        s = _settings_with_hierarchical(make_settings, enabled=True)
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        coarse_spy = AsyncMock()
        monkeypatch.setattr("app.rag.service.coarse_retrieve", coarse_spy)
        retrieve_spy = AsyncMock(return_value=[_retrieved("f1")])
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(query="q", lit_token="t", drive=None)
        )

        coarse_spy.assert_not_awaited()
        kwargs = retrieve_spy.await_args.kwargs
        assert kwargs.get("file_id_scope") is None
        assert "shortlist" not in [e.kind for e in events]


# ---------------------------------------------------------------------------
# Hierarchical enabled — confident shortlist
# ---------------------------------------------------------------------------


class TestHierarchicalScoped:
    @pytest.mark.asyncio
    async def test_scoped_retrieval_passes_shortlist_ids(
        self, monkeypatch, make_settings, common_patches
    ):
        s = _settings_with_hierarchical(
            make_settings,
            enabled=True,
            coarse_top_k=5,
            coarse_score_threshold=0.3,
            min_drive_files_for_shortlist=10,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        shortlist = ShortlistResult(
            file_ids=("a", "b", "c"),
            scores=(0.9, 0.8, 0.7),
            top_score=0.9,
            drive_file_count=100,
        )
        monkeypatch.setattr(
            "app.rag.service.coarse_retrieve",
            AsyncMock(return_value=shortlist),
        )
        retrieve_spy = AsyncMock(
            return_value=[_retrieved("a"), _retrieved("b")]
        )
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(query="q", lit_token="t", drive="Videos")
        )

        retrieve_spy.assert_awaited_once()
        kwargs = retrieve_spy.await_args.kwargs
        assert kwargs.get("file_id_scope") == ["a", "b", "c"]

        # Shortlist event emitted between keywords and sources.
        kinds = [e.kind for e in events]
        assert "shortlist" in kinds
        keywords_idx = kinds.index("keywords")
        shortlist_idx = kinds.index("shortlist")
        sources_idx = kinds.index("sources")
        assert keywords_idx < shortlist_idx < sources_idx

        shortlist_event = events[shortlist_idx]
        assert shortlist_event.data["file_ids"] == ["a", "b", "c"]
        assert shortlist_event.data["drive_file_count"] == 100
        assert shortlist_event.data["top_score"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_access_filter_drops_inaccessible_shortlist_ids(
        self, monkeypatch, make_settings, common_patches
    ):
        # Coarse returns three files; the host's Internal API access
        # filter only returns "a" and "c" — "b" is in a locked
        # protected drive the caller cannot unlock. The SSE shortlist
        # event must NOT include "b", and the scoped retrieval must be
        # called with the access-filtered ids only (not the raw
        # shortlist).
        s = _settings_with_hierarchical(
            make_settings,
            enabled=True,
            coarse_top_k=5,
            coarse_score_threshold=0.3,
            min_drive_files_for_shortlist=10,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        shortlist = ShortlistResult(
            file_ids=("a", "b", "c"),
            scores=(0.9, 0.8, 0.7),
            top_score=0.9,
            drive_file_count=100,
        )
        monkeypatch.setattr(
            "app.rag.service.coarse_retrieve",
            AsyncMock(return_value=shortlist),
        )
        monkeypatch.setattr(
            "app.rag.service._filter_file_ids_via_internal_api",
            AsyncMock(return_value={"a", "c"}),
        )
        retrieve_spy = AsyncMock(
            return_value=[_retrieved("a"), _retrieved("c")]
        )
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(query="q", lit_token="t", drive="Videos")
        )

        # Scoped call carries only the accessible ids (order preserved).
        kwargs = retrieve_spy.await_args.kwargs
        assert kwargs.get("file_id_scope") == ["a", "c"]

        # The SSE shortlist event reflects only accessible ids — the
        # caller must never see "b" exists.
        shortlist_event = next(e for e in events if e.kind == "shortlist")
        assert shortlist_event.data["file_ids"] == ["a", "c"]
        # top_score follows the surviving top-rank file ("a", score 0.9).
        assert shortlist_event.data["top_score"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_access_filter_drops_all_then_no_shortlist_event(
        self, monkeypatch, make_settings, common_patches
    ):
        # Caller has no access to any of the shortlist's files. To match
        # the design rule "保護ドライブが locked の場合は API 応答から
        # 完全除外する", the SSE shortlist event must NOT be emitted at
        # all (even with file_ids=[]). The retrieval falls back to the
        # unscoped path so the user still gets whatever the host's
        # access filter does allow them to see (typically: nothing,
        # but the response shape matches the bypass case rather than
        # leaking a "we ran hierarchical" signal).
        s = _settings_with_hierarchical(
            make_settings,
            enabled=True,
            coarse_top_k=5,
            coarse_score_threshold=0.3,
            min_drive_files_for_shortlist=10,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        shortlist = ShortlistResult(
            file_ids=("a", "b"),
            scores=(0.9, 0.8),
            top_score=0.9,
            drive_file_count=100,
        )
        monkeypatch.setattr(
            "app.rag.service.coarse_retrieve",
            AsyncMock(return_value=shortlist),
        )
        # Access filter returns empty set — every shortlist file is
        # behind a lock the caller hasn't unlocked.
        monkeypatch.setattr(
            "app.rag.service._filter_file_ids_via_internal_api",
            AsyncMock(return_value=set()),
        )
        retrieve_spy = AsyncMock(return_value=[])
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(query="q", lit_token="t", drive="Videos")
        )

        # Fallback to unscoped retrieval (no file_id_scope kwarg).
        kwargs = retrieve_spy.await_args.kwargs
        assert kwargs.get("file_id_scope") is None
        # No shortlist event emitted — design says "完全除外".
        assert "shortlist" not in [e.kind for e in events]


# ---------------------------------------------------------------------------
# Bypass conditions
# ---------------------------------------------------------------------------


class TestHierarchicalBypass:
    @pytest.mark.asyncio
    async def test_small_drive_bypass(
        self, monkeypatch, make_settings, common_patches
    ):
        s = _settings_with_hierarchical(
            make_settings,
            enabled=True,
            min_drive_files_for_shortlist=50,
            coarse_score_threshold=0.0,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        # drive_file_count < threshold -> bypass
        shortlist = ShortlistResult(
            file_ids=("a",), scores=(0.9,), top_score=0.9,
            drive_file_count=10,
        )
        monkeypatch.setattr(
            "app.rag.service.coarse_retrieve",
            AsyncMock(return_value=shortlist),
        )
        retrieve_spy = AsyncMock(return_value=[_retrieved("x")])
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(query="q", lit_token="t", drive="Tiny")
        )

        kwargs = retrieve_spy.await_args.kwargs
        assert kwargs.get("file_id_scope") is None
        assert "shortlist" not in [e.kind for e in events]

    @pytest.mark.asyncio
    async def test_low_top_score_bypass(
        self, monkeypatch, make_settings, common_patches
    ):
        s = _settings_with_hierarchical(
            make_settings,
            enabled=True,
            min_drive_files_for_shortlist=10,
            coarse_score_threshold=0.5,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        shortlist = ShortlistResult(
            file_ids=("a",), scores=(0.2,), top_score=0.2,
            drive_file_count=100,
        )
        monkeypatch.setattr(
            "app.rag.service.coarse_retrieve",
            AsyncMock(return_value=shortlist),
        )
        retrieve_spy = AsyncMock(return_value=[_retrieved("x")])
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(query="q", lit_token="t", drive="Videos")
        )

        kwargs = retrieve_spy.await_args.kwargs
        assert kwargs.get("file_id_scope") is None
        assert "shortlist" not in [e.kind for e in events]

    @pytest.mark.asyncio
    async def test_empty_shortlist_bypass(
        self, monkeypatch, make_settings, common_patches
    ):
        s = _settings_with_hierarchical(
            make_settings,
            enabled=True,
            min_drive_files_for_shortlist=10,
            coarse_score_threshold=0.0,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        shortlist = ShortlistResult(
            file_ids=(), scores=(), top_score=0.0,
            drive_file_count=100,
        )
        monkeypatch.setattr(
            "app.rag.service.coarse_retrieve",
            AsyncMock(return_value=shortlist),
        )
        retrieve_spy = AsyncMock(return_value=[_retrieved("x")])
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(query="q", lit_token="t", drive="Videos")
        )

        kwargs = retrieve_spy.await_args.kwargs
        assert kwargs.get("file_id_scope") is None
        assert "shortlist" not in [e.kind for e in events]


# ---------------------------------------------------------------------------
# Fallback merge: scoped pass returns < 2 candidates
# ---------------------------------------------------------------------------


class TestHierarchicalFallbackMerge:
    @pytest.mark.asyncio
    async def test_unscoped_pass_runs_when_scoped_returns_few(
        self, monkeypatch, make_settings, common_patches
    ):
        s = _settings_with_hierarchical(
            make_settings,
            enabled=True,
            min_drive_files_for_shortlist=10,
            coarse_score_threshold=0.0,
            fallback_full_search=True,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        shortlist = ShortlistResult(
            file_ids=("a",), scores=(0.9,), top_score=0.9,
            drive_file_count=100,
        )
        monkeypatch.setattr(
            "app.rag.service.coarse_retrieve",
            AsyncMock(return_value=shortlist),
        )

        # First call (scoped) returns 1 candidate; second (unscoped)
        # returns more, and "a" is deduped from the merge.
        scoped_result = [_retrieved("a")]
        unscoped_result = [_retrieved("a"), _retrieved("b"), _retrieved("c")]
        retrieve_spy = AsyncMock(side_effect=[scoped_result, unscoped_result])
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(query="q", lit_token="t", drive="Videos")
        )

        assert retrieve_spy.await_count == 2
        # First call carries scope, second does not.
        first = retrieve_spy.await_args_list[0].kwargs
        second = retrieve_spy.await_args_list[1].kwargs
        assert first.get("file_id_scope") == ["a"]
        assert second.get("file_id_scope") is None

        # sources event reflects merged candidates (a, b, c) — scoped first.
        sources_event = next(e for e in events if e.kind == "sources")
        ids = [s["file_id"] for s in sources_event.data["sources"]]
        assert ids == ["a", "b", "c"]
        # Shortlist event still emitted (hierarchical path executed).
        assert "shortlist" in [e.kind for e in events]
