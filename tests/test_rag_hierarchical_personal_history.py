"""Tests for hierarchical retrieval × personal_history composition.

Spec: ``2026-04-29-intelligence-ask-hierarchical-personal-history-composition.md``

When both ``hierarchical.enabled`` and ``personal_history.enabled`` fire
on the same Ask request, the service must run hierarchical retrieval
*within* the personal-history file_id scope (Stage B file_ids ∩ shortlist).
These tests pin down:

1. The composition activates only when both features fire.
2. The intersection preserves rank order from the coarse retriever.
3. Bypass paths (empty intersection, low confidence, access-filter empty)
   fall back to ``retrieve_with_keywords`` scoped to the personal-history
   file_ids — never unscoped, so the user's personal scope is preserved.
4. ``min_drive_files_for_shortlist`` is skipped when ``pre_scope_file_ids``
   is set (the scope is already much smaller than the drive).
5. Stage C ``category_expansion`` is NOT run on the composed path —
   ``clue_generation`` supersedes it for semantic intent expansion.
6. SSE event order: query_decomposed → history_filter → keywords →
   shortlist → clues → sources.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
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

from app.config import (  # noqa: E402
    CategoryExpansionConfig,
    HierarchicalRagConfig,
    LLMConfig,
    PersonalHistoryConfig,
    RagConfig,
)
from app.rag import service as service_mod  # noqa: E402
from app.rag.coarse_retriever import ShortlistResult  # noqa: E402
from app.rag.context import ContextSnippet, FileContext  # noqa: E402
from app.rag.query_decomposer import DecomposedQuery, TimeRange  # noqa: E402
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
        drive="movies",
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
        drive="movies",
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


def _decomposed(
    *,
    label: str = "last_week",
    after: datetime | None = datetime(2026, 4, 22, tzinfo=UTC),
    before: datetime | None = datetime(2026, 4, 29, tzinfo=UTC),
    personal_scope: str = "viewed",
    semantic: str = "SF",
    file_type: str = "video",
) -> DecomposedQuery:
    return DecomposedQuery(
        raw_query="先週観た映画の中で SF っぽいのどれ？",
        time_range=TimeRange(label=label, after=after, before=before),
        personal_scope=personal_scope,
        file_type_hint=file_type,
        semantic_query=semantic,
    )


def _settings_with_both(
    make_settings,
    *,
    hier_enabled: bool = True,
    history_enabled: bool = True,
    fallback_when_empty: str = "graceful",
    **hier_overrides,
):
    """Settings with both hierarchical and personal_history toggles.

    The ``hier_overrides`` mirror :class:`HierarchicalRagConfig` fields
    so individual tests can lower thresholds without re-stating the
    shared scaffolding.
    """
    hier = HierarchicalRagConfig(enabled=hier_enabled, **hier_overrides)
    history = PersonalHistoryConfig(
        enabled=history_enabled, fallback_when_empty=fallback_when_empty
    )
    return make_settings(
        features=type(make_settings().features)(rag=True),
        llm=LLMConfig(provider="openai_compatible", model="m"),
        rag=RagConfig(
            hierarchical=hier,
            personal_history=history,
            category_expansion=CategoryExpansionConfig(),
        ),
    )


@pytest.fixture()
def common_patches(monkeypatch):
    """Default stubs shared by every composed-path test.

    ``transform_query`` collapses to a literal ``"kw"``, contexts are
    built 1:1 with the candidates, and the LLM stream returns a fixed
    JSON envelope. Access filter is a pass-through; tests that need to
    drop ids override it locally.
    """
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
    monkeypatch.setattr(
        "app.rag.service._filter_file_ids_via_internal_api",
        AsyncMock(side_effect=lambda ids, _token: set(ids)),
    )
    monkeypatch.setattr(
        "app.rag.service.fetch_long_summaries", lambda _ids: {}
    )
    monkeypatch.setattr(
        "app.rag.service.generate_clues",
        AsyncMock(side_effect=lambda **kw: [kw["fallback_keywords"]]),
    )
    # Personal-history Stage A + B stubs: every test below overrides
    # fetch_viewer_history's return value but the decompose stays fixed.
    monkeypatch.setattr(
        "app.rag.service.decompose_query",
        AsyncMock(return_value=_decomposed()),
    )
    # Default category-expansion: empty list (Stage C disabled). The
    # composed path tests assert the expander is NOT called; tests that
    # need to verify the personal-history-only branch override this.
    monkeypatch.setattr(
        "app.rag.service.expand_category",
        AsyncMock(side_effect=AssertionError("must not be called")),
    )


async def _collect(gen) -> list[AnswerEvent]:
    events: list[AnswerEvent] = []
    async for event in gen:
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Composition activation
# ---------------------------------------------------------------------------


class TestCompositionActivation:
    """Composition fires only when both features fire on the request."""

    @pytest.mark.asyncio
    async def test_both_features_compose(
        self, monkeypatch, make_settings, common_patches
    ):
        """Both enabled + drive + history.file_ids → composed path runs.

        The composed path is detected by:
        * coarse_retrieve being awaited (hierarchical Stage 1 ran)
        * retrieve_with_keywords being called with file_id_scope set to
          the *intersection* of shortlist and history.file_ids
        * the SSE shortlist event firing
        """
        s = _settings_with_both(
            make_settings,
            coarse_top_k=5,
            coarse_score_threshold=0.3,
            min_drive_files_for_shortlist=10,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        monkeypatch.setattr(
            "app.rag.service.fetch_viewer_history",
            AsyncMock(return_value=["b", "d", "e"]),
        )
        shortlist = ShortlistResult(
            file_ids=("a", "b", "c", "d"),
            scores=(0.9, 0.8, 0.7, 0.6),
            top_score=0.9,
            drive_file_count=100,
        )
        coarse_spy = AsyncMock(return_value=shortlist)
        monkeypatch.setattr("app.rag.service.coarse_retrieve", coarse_spy)

        retrieve_spy = AsyncMock(
            return_value=[_retrieved("b"), _retrieved("d")]
        )
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(
                query="先週観た映画の中で SF っぽいのどれ？",
                lit_token="t",
                drive="movies",
                viewer_id="x" * 16,
            )
        )

        coarse_spy.assert_awaited_once()
        # Stage 3 ran with the intersected shortlist (b, d) — preserved
        # rank order. ``fallback_full_search`` may add a second unscoped
        # call, so probe ``await_args_list`` instead of ``await_args``.
        scoped_calls = [
            c for c in retrieve_spy.await_args_list
            if c.kwargs.get("file_id_scope") == ["b", "d"]
        ]
        assert scoped_calls, retrieve_spy.await_args_list

        kinds = [e.kind for e in events]
        assert "shortlist" in kinds
        # Shortlist event reflects the *intersected* set, not the raw
        # coarse output. "a" and "c" were not viewed last week — they
        # must not leak into the SSE stream.
        shortlist_evt = events[kinds.index("shortlist")]
        assert shortlist_evt.data["file_ids"] == ["b", "d"]

    @pytest.mark.asyncio
    async def test_only_history_skips_hierarchical(
        self, monkeypatch, make_settings, common_patches
    ):
        """history enabled, hierarchical disabled → no coarse_retrieve."""
        s = _settings_with_both(
            make_settings, hier_enabled=False, history_enabled=True
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        monkeypatch.setattr(
            "app.rag.service.fetch_viewer_history",
            AsyncMock(return_value=["fid1", "fid2"]),
        )
        coarse_spy = AsyncMock()
        monkeypatch.setattr("app.rag.service.coarse_retrieve", coarse_spy)
        retrieve_spy = AsyncMock(return_value=[_retrieved("fid1")])
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        await _collect(
            stream_answer(
                query="q",
                lit_token="t",
                drive="movies",
                viewer_id="x" * 16,
            )
        )

        coarse_spy.assert_not_awaited()
        # History scope still applied via the existing path.
        scoped_calls = [
            c for c in retrieve_spy.await_args_list
            if c.kwargs.get("file_id_scope") == ["fid1", "fid2"]
        ]
        assert scoped_calls, retrieve_spy.await_args_list

    @pytest.mark.asyncio
    async def test_only_hierarchical_no_history(
        self, monkeypatch, make_settings, common_patches
    ):
        """hierarchical enabled, history disabled → no Stage A/B."""
        s = _settings_with_both(
            make_settings,
            hier_enabled=True,
            history_enabled=False,
            coarse_top_k=5,
            coarse_score_threshold=0.3,
            min_drive_files_for_shortlist=10,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        decompose_spy = AsyncMock(side_effect=AssertionError("must not be called"))
        monkeypatch.setattr("app.rag.service.decompose_query", decompose_spy)
        history_spy = AsyncMock(side_effect=AssertionError("must not be called"))
        monkeypatch.setattr(
            "app.rag.service.fetch_viewer_history", history_spy
        )

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
        retrieve_spy = AsyncMock(
            return_value=[_retrieved("a"), _retrieved("b")]
        )
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        await _collect(
            stream_answer(
                query="q",
                lit_token="t",
                drive="movies",
                viewer_id="x" * 16,  # provided but feature off
            )
        )

        # No personal_history machinery should have fired. Hierarchical-
        # only scope. ``fallback_full_search`` may add an unscoped call;
        # check via the call list rather than ``await_args``.
        scoped_calls = [
            c for c in retrieve_spy.await_args_list
            if c.kwargs.get("file_id_scope") == ["a", "b"]
        ]
        assert scoped_calls, retrieve_spy.await_args_list


# ---------------------------------------------------------------------------
# Intersection ordering
# ---------------------------------------------------------------------------


class TestIntersectionOrdering:
    @pytest.mark.asyncio
    async def test_intersection_preserves_coarse_rank(
        self, monkeypatch, make_settings, common_patches
    ):
        """Intersection rank order = coarse rank order (not pre_scope's)."""
        s = _settings_with_both(
            make_settings,
            coarse_top_k=10,
            coarse_score_threshold=0.3,
            min_drive_files_for_shortlist=10,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        # pre_scope gives a different order than coarse — we must
        # follow coarse, since it carries the score signal.
        monkeypatch.setattr(
            "app.rag.service.fetch_viewer_history",
            AsyncMock(return_value=["d", "a", "b"]),
        )
        shortlist = ShortlistResult(
            file_ids=("a", "b", "c", "d"),
            scores=(0.9, 0.8, 0.7, 0.6),
            top_score=0.9,
            drive_file_count=100,
        )
        monkeypatch.setattr(
            "app.rag.service.coarse_retrieve",
            AsyncMock(return_value=shortlist),
        )
        retrieve_spy = AsyncMock(
            return_value=[_retrieved("a"), _retrieved("b"), _retrieved("d")]
        )
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        await _collect(
            stream_answer(
                query="q",
                lit_token="t",
                drive="movies",
                viewer_id="x" * 16,
            )
        )

        # coarse order: a > b > c > d. pre_scope: {a, b, d}. Result:
        # [a, b, d] — c filtered out, original coarse rank preserved.
        scoped_calls = [
            c for c in retrieve_spy.await_args_list
            if c.kwargs.get("file_id_scope") == ["a", "b", "d"]
        ]
        assert scoped_calls, retrieve_spy.await_args_list


# ---------------------------------------------------------------------------
# Bypass fallback policy
# ---------------------------------------------------------------------------


class TestBypassFallback:
    """Bypass paths fall back to pre_scope-only legacy retrieve.

    Per design (Stage B file_ids ∩ shortlist):
    * empty intersection → fall back to pre_scope_file_ids retrieve
    * coarse top score below threshold → fall back to pre_scope retrieve
    * access filter drops everything → fall back to pre_scope retrieve

    Crucially, NEVER fall back to unscoped retrieval — the user's
    personal scope must not be leaked away.
    """

    @pytest.mark.asyncio
    async def test_empty_intersection_falls_back_to_pre_scope(
        self, monkeypatch, make_settings, common_patches
    ):
        s = _settings_with_both(
            make_settings,
            coarse_top_k=5,
            coarse_score_threshold=0.3,
            min_drive_files_for_shortlist=10,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        # No overlap between shortlist {a, b} and pre_scope {p, q, r}.
        monkeypatch.setattr(
            "app.rag.service.fetch_viewer_history",
            AsyncMock(return_value=["p", "q", "r"]),
        )
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
        retrieve_spy = AsyncMock(return_value=[_retrieved("p")])
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(
                query="q",
                lit_token="t",
                drive="movies",
                viewer_id="x" * 16,
            )
        )

        retrieve_spy.assert_awaited_once()
        # Crucial invariant: scope is the personal-history file_ids,
        # NOT None. The user's "先週観た" filter must survive bypass.
        assert retrieve_spy.await_args.kwargs["file_id_scope"] == [
            "p",
            "q",
            "r",
        ]
        # Bypassed → no shortlist event.
        assert "shortlist" not in [e.kind for e in events]

    @pytest.mark.asyncio
    async def test_low_coarse_score_falls_back_to_pre_scope(
        self, monkeypatch, make_settings, common_patches
    ):
        s = _settings_with_both(
            make_settings,
            coarse_top_k=5,
            coarse_score_threshold=0.5,  # threshold above shortlist top
            min_drive_files_for_shortlist=10,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        monkeypatch.setattr(
            "app.rag.service.fetch_viewer_history",
            AsyncMock(return_value=["p", "q"]),
        )
        shortlist = ShortlistResult(
            file_ids=("a", "b"),
            scores=(0.2, 0.1),
            top_score=0.2,  # < threshold 0.5
            drive_file_count=100,
        )
        monkeypatch.setattr(
            "app.rag.service.coarse_retrieve",
            AsyncMock(return_value=shortlist),
        )
        retrieve_spy = AsyncMock(return_value=[])
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        await _collect(
            stream_answer(
                query="q",
                lit_token="t",
                drive="movies",
                viewer_id="x" * 16,
            )
        )

        retrieve_spy.assert_awaited_once()
        assert retrieve_spy.await_args.kwargs["file_id_scope"] == ["p", "q"]

    @pytest.mark.asyncio
    async def test_access_filter_empty_falls_back_to_pre_scope(
        self, monkeypatch, make_settings, common_patches
    ):
        """Locked drive drops the whole intersected shortlist.

        retrieve_with_keywords still gets called *with the personal-
        history scope intact* — the access gate downstream then decides
        which of those are accessible. The composed path must not leak
        the user out of their personal scope just because the locked
        drive happened to dominate the shortlist.
        """
        s = _settings_with_both(
            make_settings,
            coarse_top_k=5,
            coarse_score_threshold=0.3,
            min_drive_files_for_shortlist=10,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        monkeypatch.setattr(
            "app.rag.service.fetch_viewer_history",
            AsyncMock(return_value=["a", "b"]),
        )
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
        # Access filter drops the entire intersected shortlist.
        monkeypatch.setattr(
            "app.rag.service._filter_file_ids_via_internal_api",
            AsyncMock(return_value=set()),
        )
        retrieve_spy = AsyncMock(return_value=[])
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(
                query="q",
                lit_token="t",
                drive="movies",
                viewer_id="x" * 16,
            )
        )

        retrieve_spy.assert_awaited_once()
        assert retrieve_spy.await_args.kwargs["file_id_scope"] == ["a", "b"]
        # No shortlist SSE event when bypass happens.
        assert "shortlist" not in [e.kind for e in events]


# ---------------------------------------------------------------------------
# Small-drive guard
# ---------------------------------------------------------------------------


class TestMinDriveFilesGuard:
    @pytest.mark.asyncio
    async def test_min_drive_files_check_skipped_when_pre_scope_set(
        self, monkeypatch, make_settings, common_patches
    ):
        """``pre_scope`` is already narrow → the small-drive guard
        (which protects against shortlist on a too-small corpus) is
        irrelevant. Composition must still try Stage 1 → 2 → 3.
        """
        s = _settings_with_both(
            make_settings,
            coarse_top_k=5,
            coarse_score_threshold=0.1,
            min_drive_files_for_shortlist=200,  # nominal drive too small
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        monkeypatch.setattr(
            "app.rag.service.fetch_viewer_history",
            AsyncMock(return_value=["a"]),
        )
        shortlist = ShortlistResult(
            file_ids=("a", "b"),
            scores=(0.9, 0.8),
            top_score=0.9,
            drive_file_count=20,  # below the 200 threshold above
        )
        monkeypatch.setattr(
            "app.rag.service.coarse_retrieve",
            AsyncMock(return_value=shortlist),
        )
        retrieve_spy = AsyncMock(return_value=[_retrieved("a")])
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        events = await _collect(
            stream_answer(
                query="q",
                lit_token="t",
                drive="movies",
                viewer_id="x" * 16,
            )
        )

        # Composition went through (shortlist event fired).
        assert "shortlist" in [e.kind for e in events]
        # Stage 3 scope = intersection.
        assert retrieve_spy.await_args.kwargs["file_id_scope"] == ["a"]


# ---------------------------------------------------------------------------
# Stage C interaction
# ---------------------------------------------------------------------------


class TestStageCNotInvokedOnComposedPath:
    """``category_expansion`` is suppressed when hierarchical composes."""

    @pytest.mark.asyncio
    async def test_expand_category_not_called(
        self, monkeypatch, make_settings, common_patches
    ):
        # category_expansion enabled but should not run because the
        # composed path's clue generation supersedes it.
        s = _settings_with_both(
            make_settings,
            coarse_top_k=5,
            coarse_score_threshold=0.3,
            min_drive_files_for_shortlist=10,
        )
        # Override category_expansion to enabled. The fixture's
        # expand_category mock raises on call — that's the assertion.
        s = type(s)(
            **{
                **{f.name: getattr(s, f.name) for f in s.__dataclass_fields__.values()},
                "rag": RagConfig(
                    hierarchical=s.rag.hierarchical,
                    personal_history=s.rag.personal_history,
                    category_expansion=CategoryExpansionConfig(enabled=True),
                ),
            }
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        monkeypatch.setattr(
            "app.rag.service.fetch_viewer_history",
            AsyncMock(return_value=["a", "b"]),
        )
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
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords",
            AsyncMock(return_value=[_retrieved("a")]),
        )

        # The fixture's expand_category raises on call. If composition
        # invokes Stage C, this propagates and fails the test.
        await _collect(
            stream_answer(
                query="q",
                lit_token="t",
                drive="movies",
                viewer_id="x" * 16,
            )
        )

        # No category_expanded SSE event either (the helper guards
        # double-emission).
        # Tolerated: the test's success is "no exception raised by the
        # AssertionError-side-effect in expand_category".


# ---------------------------------------------------------------------------
# SSE event ordering
# ---------------------------------------------------------------------------


class TestComposedPathSSEOrder:
    @pytest.mark.asyncio
    async def test_event_order_query_decomposed_history_filter_keywords_shortlist(
        self, monkeypatch, make_settings, common_patches
    ):
        """SSE order: query_decomposed → history_filter → keywords →
        shortlist → clues → sources → answer_chunk* → citations → done.
        """
        s = _settings_with_both(
            make_settings,
            coarse_top_k=5,
            coarse_score_threshold=0.3,
            min_drive_files_for_shortlist=10,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        monkeypatch.setattr(
            "app.rag.service.fetch_viewer_history",
            AsyncMock(return_value=["a", "b"]),
        )
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
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords",
            AsyncMock(return_value=[_retrieved("a")]),
        )
        # Force clue gen to produce 2 clues so the ``clues`` event fires
        # with non-trivial content.
        monkeypatch.setattr(
            "app.rag.service.fetch_long_summaries",
            lambda ids: {fid: f"summary {fid}" for fid in ids},
        )
        monkeypatch.setattr(
            "app.rag.service.generate_clues",
            AsyncMock(return_value=["clue1", "clue2"]),
        )

        events = await _collect(
            stream_answer(
                query="先週観た映画",
                lit_token="t",
                drive="movies",
                viewer_id="x" * 16,
            )
        )

        kinds = [e.kind for e in events]
        # Required prefix order.
        order = [
            "query_decomposed",
            "history_filter",
            "keywords",
            "shortlist",
            "clues",
            "sources",
        ]
        positions = [kinds.index(k) for k in order]
        assert positions == sorted(positions), (
            f"composed-path event order broken: {kinds!r}"
        )
        # Stage C suppression: ``category_expanded`` must never fire on
        # the composed path even when category_expansion is enabled in
        # config (clue generation supersedes it).
        assert "category_expanded" not in kinds


# ---------------------------------------------------------------------------
# fallback_full_search keeps personal scope
# ---------------------------------------------------------------------------


class TestFallbackFullSearchPreservesPreScope:
    @pytest.mark.asyncio
    async def test_widening_pass_uses_pre_scope_not_unscoped(
        self, monkeypatch, make_settings, common_patches
    ):
        """When the scoped pass returns < 2 candidates and
        ``fallback_full_search`` is on, the widening pass must keep
        ``file_id_scope=pre_scope_file_ids``. Letting it widen to
        unscoped (drive-wide) would leak the user out of their
        "先週観た" personal scope.
        """
        s = _settings_with_both(
            make_settings,
            coarse_top_k=5,
            coarse_score_threshold=0.3,
            min_drive_files_for_shortlist=10,
            fallback_full_search=True,
        )
        monkeypatch.setattr("app.config.settings", s)
        monkeypatch.setattr("app.rag.service.settings", s)

        monkeypatch.setattr(
            "app.rag.service.fetch_viewer_history",
            AsyncMock(return_value=["a", "b", "c"]),
        )
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

        # Scoped pass returns 1 result → triggers fallback_full_search.
        # The widening call MUST receive file_id_scope = pre_scope, not
        # None.
        retrieve_spy = AsyncMock(return_value=[_retrieved("a")])
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords", retrieve_spy
        )

        await _collect(
            stream_answer(
                query="q",
                lit_token="t",
                drive="movies",
                viewer_id="x" * 16,
            )
        )

        # Two scopes that show up across calls: the intersected
        # shortlist (a, b) for the clue pass, and pre_scope (a, b, c)
        # for the widening pass. Critically: NO call should have
        # file_id_scope=None.
        scopes = [
            c.kwargs.get("file_id_scope") for c in retrieve_spy.await_args_list
        ]
        assert None not in scopes, (
            f"composed-path widening must not run unscoped, got {scopes!r}"
        )
        assert ["a", "b", "c"] in scopes, (
            f"widening pass must use pre_scope, got {scopes!r}"
        )
