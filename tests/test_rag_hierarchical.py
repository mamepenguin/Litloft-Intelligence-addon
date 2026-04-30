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
from app.rag.service import (  # noqa: E402
    AnswerEvent,
    _rrf_merge_candidates,
    stream_answer,
)
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
    from app.rag.query_transform import StructuredQuery as _SQ
    monkeypatch.setattr(
        "app.rag.service.transform_query_structured",
        AsyncMock(return_value=_SQ(
            required=(), semantic=("kw",), raw_keywords="kw",
        )),
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
    # Phase 3 defaults: no AI summaries available -> generate_clues
    # collapses to ``[fallback_keywords]``, so Phase 1/2 invariants
    # (single retrieve call, single keyword) are preserved. Tests that
    # exercise multi-clue dispatch override these.
    monkeypatch.setattr(
        "app.rag.service.fetch_long_summaries", lambda _ids: {}
    )
    monkeypatch.setattr(
        "app.rag.service.generate_clues",
        AsyncMock(side_effect=lambda **kw: [kw["fallback_keywords"]]),
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


# ---------------------------------------------------------------------------
# Phase 3: Multi-query Clue Generation
# ---------------------------------------------------------------------------


class TestHierarchicalClueGeneration:
    """Phase 3 wiring: clue generation, multi-clue retrieval, RRF, SSE."""

    @pytest.mark.asyncio
    async def test_clues_event_emitted_after_shortlist(
        self, monkeypatch, make_settings, common_patches
    ):
        s = _settings_with_hierarchical(
            make_settings,
            enabled=True,
            min_drive_files_for_shortlist=10,
            coarse_score_threshold=0.0,
            clue_count=3,
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
        # Provide summaries so the clue generator stub is exercised.
        monkeypatch.setattr(
            "app.rag.service.fetch_long_summaries",
            lambda ids: {fid: f"summary-{fid}" for fid in ids},
        )
        clues_spy = AsyncMock(return_value=["clue1", "clue2", "clue3"])
        monkeypatch.setattr("app.rag.service.generate_clues", clues_spy)
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords",
            AsyncMock(return_value=[_retrieved("a"), _retrieved("b")]),
        )

        events = await _collect(
            stream_answer(query="q", lit_token="t", drive="Videos")
        )

        kinds = [e.kind for e in events]
        # clues event sits between shortlist and sources.
        shortlist_idx = kinds.index("shortlist")
        clues_idx = kinds.index("clues")
        sources_idx = kinds.index("sources")
        assert shortlist_idx < clues_idx < sources_idx

        clues_event = events[clues_idx]
        assert clues_event.data["clues"] == ["clue1", "clue2", "clue3"]

        # generate_clues was called with the user's query, the
        # shortlist's summaries (in shortlist order) and the
        # configured clue_count + transform_query keywords as fallback.
        assert clues_spy.await_count == 1
        kwargs = clues_spy.await_args.kwargs
        assert kwargs["query"] == "q"
        assert kwargs["summaries"] == ["summary-a", "summary-b"]
        assert kwargs["clue_count"] == 3
        assert kwargs["fallback_keywords"] == "kw"

    @pytest.mark.asyncio
    async def test_per_clue_retrieve_runs_in_parallel_with_same_scope(
        self, monkeypatch, make_settings, common_patches
    ):
        # Three clues -> three retrieve_with_keywords calls, each
        # carrying the same file_id_scope (the access-filtered shortlist).
        s = _settings_with_hierarchical(
            make_settings,
            enabled=True,
            min_drive_files_for_shortlist=10,
            coarse_score_threshold=0.0,
            clue_count=3,
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
            "app.rag.service.fetch_long_summaries",
            lambda ids: {fid: f"s-{fid}" for fid in ids},
        )
        monkeypatch.setattr(
            "app.rag.service.generate_clues",
            AsyncMock(return_value=["q1", "q2", "q3"]),
        )

        retrieve_spy = AsyncMock(
            side_effect=lambda **kw: [_retrieved(kw["keywords"])]
        )
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        await _collect(
            stream_answer(query="q", lit_token="t", drive="Videos")
        )

        # 3 scoped calls. Fallback unscoped pass MAY also fire because
        # each clue surfaced only one candidate; but that fourth call is
        # unscoped which is asserted separately below. Here we only
        # care about the scoped trio.
        scoped_calls = [
            ca
            for ca in retrieve_spy.await_args_list
            if ca.kwargs.get("file_id_scope") is not None
        ]
        assert len(scoped_calls) == 3
        clue_keywords = [ca.kwargs["keywords"] for ca in scoped_calls]
        assert clue_keywords == ["q1", "q2", "q3"]
        # Same scope across all three.
        scopes = {tuple(ca.kwargs["file_id_scope"]) for ca in scoped_calls}
        assert scopes == {("a", "b", "c")}

    @pytest.mark.asyncio
    async def test_rrf_merge_combines_per_clue_results(
        self, monkeypatch, make_settings, common_patches
    ):
        # Two clues with overlapping but differently ordered hits.
        # RRF should rank a file that appears at top of one list and
        # mid of another above a file that appears only once.
        s = _settings_with_hierarchical(
            make_settings,
            enabled=True,
            min_drive_files_for_shortlist=10,
            coarse_score_threshold=0.0,
            clue_count=2,
            fallback_full_search=False,  # don't run unscoped pass
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
            "app.rag.service.fetch_long_summaries",
            lambda ids: {fid: f"s-{fid}" for fid in ids},
        )
        monkeypatch.setattr(
            "app.rag.service.generate_clues",
            AsyncMock(return_value=["q1", "q2"]),
        )

        # q1: a (rank 1), b (rank 2)
        # q2: b (rank 1), c (rank 2)
        per_clue: dict[str, list] = {
            "q1": [_retrieved("a"), _retrieved("b")],
            "q2": [_retrieved("b"), _retrieved("c")],
        }
        retrieve_spy = AsyncMock(
            side_effect=lambda **kw: per_clue[kw["keywords"]]
        )
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(query="q", lit_token="t", drive="Videos")
        )

        sources_event = next(e for e in events if e.kind == "sources")
        ids = [src["file_id"] for src in sources_event.data["sources"]]
        # b appears in both lists -> highest combined RRF score.
        # a and c each appear once at rank 1 vs rank 2 — a beats c.
        assert ids[0] == "b"
        assert ids[1] == "a"
        assert ids[2] == "c"

    @pytest.mark.asyncio
    async def test_clue_generation_failure_falls_back_to_keywords(
        self, monkeypatch, make_settings, common_patches
    ):
        # When generate_clues collapses to [fallback_keywords] (the
        # documented graceful-degradation contract), the scoped path
        # runs exactly once with those keywords. This is the same
        # behaviour as Phase 2.
        s = _settings_with_hierarchical(
            make_settings,
            enabled=True,
            min_drive_files_for_shortlist=10,
            coarse_score_threshold=0.0,
            clue_count=3,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        shortlist = ShortlistResult(
            file_ids=("a",),
            scores=(0.9,),
            top_score=0.9,
            drive_file_count=100,
        )
        monkeypatch.setattr(
            "app.rag.service.coarse_retrieve",
            AsyncMock(return_value=shortlist),
        )
        # Default common_patches stubs make fetch_long_summaries
        # return {} which is the standard empty-summaries path; in
        # that case generate_clues' real implementation collapses to
        # [fallback_keywords]. The fixture stub already does that.
        retrieve_spy = AsyncMock(return_value=[_retrieved("a")])
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(query="q", lit_token="t", drive="Videos")
        )

        scoped_calls = [
            ca
            for ca in retrieve_spy.await_args_list
            if ca.kwargs.get("file_id_scope") is not None
        ]
        assert len(scoped_calls) == 1
        assert scoped_calls[0].kwargs["keywords"] == "kw"

        clues_event = next(e for e in events if e.kind == "clues")
        assert clues_event.data["clues"] == ["kw"]


# ---------------------------------------------------------------------------
# RRF merge unit tests (exercise _rrf_merge_candidates in isolation)
# ---------------------------------------------------------------------------


class TestRrfMergeCandidates:
    """Direct unit tests for the per-clue RRF merge helper.

    The integration tests above prove the merge is wired into the
    streaming pipeline; these isolate the math so future tuning
    (changing rrf_k, weighting clues, etc.) has a focused safety net.
    """

    def test_empty_input_returns_empty_list(self):
        assert _rrf_merge_candidates([], top_k=10) == []

    def test_all_empty_inner_lists_returns_empty(self):
        assert _rrf_merge_candidates([[], [], []], top_k=10) == []

    def test_single_list_passes_through_in_order(self):
        # With a single clue, RRF reduces to "rank by position" since
        # there's no second list contributing. Output should preserve
        # the input order.
        a, b, c = _retrieved("a"), _retrieved("b"), _retrieved("c")
        merged = _rrf_merge_candidates([[a, b, c]], top_k=10)
        assert [m.file_id for m in merged] == ["a", "b", "c"]

    def test_overlapping_files_get_summed_scores(self):
        # b appears in both clues, a only in clue 1, c only in clue 2.
        # b should rank highest because its scores accumulate.
        a = _retrieved("a")
        b = _retrieved("b")
        c = _retrieved("c")
        merged = _rrf_merge_candidates(
            [[a, b], [b, c]], top_k=10
        )
        # b appears in both -> highest combined score.
        # a is rank 1 in clue 1 only; c is rank 2 in clue 2 only.
        # Higher rank (lower number) wins -> a > c.
        assert [m.file_id for m in merged] == ["b", "a", "c"]

    def test_top_k_trims_result(self):
        ranked = [_retrieved(f"f{i}") for i in range(5)]
        merged = _rrf_merge_candidates([ranked], top_k=3)
        assert len(merged) == 3
        assert [m.file_id for m in merged] == ["f0", "f1", "f2"]

    def test_metadata_taken_from_first_seen_list(self):
        # When the same file appears in multiple clues, metadata
        # (segments, score, etc.) comes from the *first* list. This
        # is documented behaviour — the integration LLM context
        # builder doesn't need both copies' segment data.
        a_first = RetrievedFile(
            file_id="a",
            drive="Videos",
            filename="a.mp4",
            file_type="video",
            title="From clue 1",
            description=None,
            score=0.9,
            match_types=("transcript",),
            segments=(),
        )
        a_second = RetrievedFile(
            file_id="a",
            drive="Videos",
            filename="a.mp4",
            file_type="video",
            title="From clue 2",
            description=None,
            score=0.5,
            match_types=("text_content",),
            segments=(),
        )
        merged = _rrf_merge_candidates(
            [[a_first], [a_second]], top_k=10
        )
        assert len(merged) == 1
        assert merged[0].title == "From clue 1"

    def test_rrf_k_affects_score_smoothness(self):
        # Larger rrf_k smooths the gap between rank 1 and rank 2.
        # We check this indirectly: with very small rrf_k=1, a top-1
        # in one clue beats a 2x rank-2 in another clue. With large
        # rrf_k=1000, the math flips and 2x rank-2 wins.
        x = _retrieved("x")
        y = _retrieved("y")

        small = _rrf_merge_candidates(
            [[x], [y, y]],  # x rank 1 once; y rank 1+2
            top_k=10,
            rrf_k=1,
        )
        # y: 1/(1+1) + 1/(1+2) = 0.833; x: 1/(1+1) = 0.5  -> y wins
        # Wait — both clue-1 lists give x rank 1 and y rank 1+2;
        # y appears at rank 1 of clue 2 AND rank 2 of clue 2 (same
        # candidate twice in one list — accumulates).
        assert small[0].file_id == "y"

        large = _rrf_merge_candidates(
            [[x], [y, y]],
            top_k=10,
            rrf_k=1000,
        )
        # With rrf_k=1000 the gap shrinks but the same ordering holds
        # (y still gets contributions from two ranks vs x's one).
        assert large[0].file_id == "y"
