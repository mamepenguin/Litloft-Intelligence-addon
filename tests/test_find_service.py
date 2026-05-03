"""Tests for app.rag.service.find_files (Find mode service layer).

Spec: ``docs/superpowers/specs/2026-04-30-intelligence-find-mode.md``.

Find mode is the file-listing sibling of Ask mode. It reuses Stages
A-D (decompose / history filter / category expansion / scoped retrieve)
and stops short of Stage E (LLM answer generation). The service
function under test is ``find_files``: a single-shot entry point that
returns the structured response shape from spec §3.2 directly, without
any SSE multiplexing.

The contract these tests pin down:

1. ``find_files`` calls ``decompose_query`` UNLESS ``overrides`` is
   supplied. Overrides win ⇒ no LLM call for Stage A.
2. ``fetch_viewer_history`` is called only when personal_scope ≠ "none"
   AND viewer_id is non-None. Missing viewer_id ⇒ silently skip Stage B.
3. ``expand_category`` is called only when semantic_query is non-empty.
   Failure ⇒ falls back to using the raw semantic_query (mirror
   query_transform's existing graceful degradation).
4. ``retriever.retrieve`` is called with the file_id_scope from Stage B
   (when present), the expanded keywords, and the file_type filter.
5. The response includes ``decomposed`` (with all five spec keys),
   ``results`` (≤ limit), ``total``, and ``limit``.
6. ``find_files`` does NOT call ``stream_answer`` or any other
   LLM-driven answer-text generator. Verified via mock spy.
7. Each result has the schema from spec §3.2:
   ``file_id``, ``score``, ``hit.{kind, location, text}``,
   ``file.{name, file_type, thumbnail_url, viewed_at}``.
8. ``find_files`` honours the ``limit`` parameter — at most ``limit``
   results returned.

All tests RED until Phase 4 ships:
* ``app.rag.service.find_files`` — the new service function.
* ``app.schemas.FindResponse`` — the new pydantic response model
  (or whatever shape the service returns; tests accept dict OR
  pydantic dump).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

# Heavy ML deps are stubbed before importing service.py.
for _mod in (
    "PIL",
    "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome",
    "janome.tokenizer",
    "numpy",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.config import (  # noqa: E402
    CategoryExpansionConfig,
    PersonalHistoryConfig,
    RagConfig,
)
from app.rag import service as service_mod  # noqa: E402
from app.rag.query_decomposer import DecomposedQuery, TimeRange  # noqa: E402

# RED gate: this import must fail until Phase 4 lands the new function.
from app.rag.service import find_files  # noqa: E402


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _decomposed(
    *,
    label: str = "last_week",
    after: datetime | None = datetime(2026, 4, 13, tzinfo=UTC),
    before: datetime | None = datetime(2026, 4, 20, tzinfo=UTC),
    personal_scope: str = "viewed",
    semantic: str = "SF",
    file_type: str = "video",
    raw: str = "先週観た映画の中で SF っぽいのどれ？",
) -> DecomposedQuery:
    return DecomposedQuery(
        raw_query=raw,
        time_range=TimeRange(label=label, after=after, before=before),
        personal_scope=personal_scope,
        file_type_hint=file_type,
        semantic_query=semantic,
    )


def _retrieved_file(
    *,
    file_id: str = "f-abc123",
    drive: str = "movies",
    filename: str = "Interstellar.mp4",
    file_type: str = "video",
    score: float = 0.82,
    text: str = "...宇宙船が時空を超えて...",
    start: float = 415.2,
    end: float = 460.0,
):
    """Build a stand-in for a retriever.RetrievedFile result.

    The service may consume RetrievedFile-like objects or SearchResult-
    like objects internally. Tests assert on the *output* shape only,
    so we use a permissive duck-typed MagicMock that satisfies any
    attribute access the service may perform.
    """
    obj = MagicMock()
    obj.file_id = file_id
    obj.drive = drive
    obj.filename = filename
    obj.file_type = file_type
    obj.score = score
    obj.match_types = ("transcript",)
    # Segment-like accessor used by the response builder to extract
    # hit.text / hit.location.
    seg = MagicMock()
    seg.text = text
    seg.start_seconds = start
    seg.end_seconds = end
    seg.kind = "transcript"
    obj.segments = (seg,)
    obj.title = filename
    obj.description = None
    obj.mime_type = "video/mp4"
    return obj


def _enable_find(monkeypatch, *, fallback: str = "graceful") -> None:
    """Patch settings.rag with personal-history enabled.

    Find mode shares ``RagConfig`` with Ask. We don't add a separate
    flag in MVP — features.rag governs both.
    """
    rag_cfg = RagConfig(
        personal_history=PersonalHistoryConfig(
            enabled=True, fallback_when_empty=fallback
        ),
        # Stage C is now driven by ``_resolve_category_expansion`` which
        # gates on ``cfg.enabled`` (default False). Find tests need it
        # on to exercise the multi-query path.
        category_expansion=CategoryExpansionConfig(enabled=True),
    )
    settings_mock = MagicMock()
    settings_mock.rag = rag_cfg
    monkeypatch.setattr(service_mod, "settings", settings_mock)


# ---------------------------------------------------------------------------
# Stage A: decompose vs. overrides
# ---------------------------------------------------------------------------


class TestFindFilesStageA:
    @pytest.mark.asyncio
    async def test_calls_decompose_when_overrides_absent(
        self, monkeypatch
    ):
        _enable_find(monkeypatch)
        decomposed = _decomposed()

        decompose_spy = AsyncMock(return_value=decomposed)
        monkeypatch.setattr(service_mod, "decompose_query", decompose_spy)
        monkeypatch.setattr(
            service_mod,
            "fetch_viewer_history",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(
            service_mod,
            "expand_category",
            AsyncMock(return_value=["SF"]),
        )
        # The service may call retrieve_with_keywords or a new dedicated
        # retriever. We stub both so the test passes regardless.
        monkeypatch.setattr(
            service_mod,
            "retrieve_with_keywords",
            AsyncMock(return_value=[]),
        )

        await find_files(
            question="先週観た映画でSF",
            drive="movies",
            viewer_id="x" * 16,
            overrides=None,
            limit=20,
        )

        decompose_spy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_decompose_when_overrides_supplied(
        self, monkeypatch
    ):
        """``overrides`` short-circuits Stage A entirely.

        chip × clicks rebuild the structured query in the frontend; the
        service must NOT re-run the LLM-driven decomposer when the
        client has already done that work.
        """
        _enable_find(monkeypatch)

        decompose_spy = AsyncMock(side_effect=AssertionError("must not run"))
        monkeypatch.setattr(service_mod, "decompose_query", decompose_spy)
        monkeypatch.setattr(
            service_mod,
            "fetch_viewer_history",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(
            service_mod,
            "expand_category",
            AsyncMock(return_value=["SF"]),
        )
        monkeypatch.setattr(
            service_mod,
            "retrieve_with_keywords",
            AsyncMock(return_value=[]),
        )

        overrides = {
            "time_range": "none",
            "personal_scope": "viewed",
            "file_type_hint": "video",
            "semantic_query": "SF",
        }

        await find_files(
            question="先週観た映画でSF",
            drive="movies",
            viewer_id="x" * 16,
            overrides=overrides,
            limit=20,
        )

        decompose_spy.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_overrides_propagate_to_downstream_stages(
        self, monkeypatch
    ):
        """The overridden personal_scope / file_type_hint must reach
        Stages B, C, and D.

        If the service silently kept LLM-decomposed values around the
        chip × interaction would feel broken — the user clicked × but
        the same filter still applied.
        """
        _enable_find(monkeypatch)
        # No decompose call — overrides supersede.
        monkeypatch.setattr(
            service_mod,
            "decompose_query",
            AsyncMock(side_effect=AssertionError("must not run")),
        )
        history_spy = AsyncMock(return_value=["fid1"])
        monkeypatch.setattr(service_mod, "fetch_viewer_history", history_spy)
        expand_spy = AsyncMock(return_value=["SF"])
        monkeypatch.setattr(service_mod, "expand_category", expand_spy)
        retrieve_spy = AsyncMock(return_value=[])
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", retrieve_spy
        )

        overrides = {
            "time_range": "none",
            "personal_scope": "viewed",
            "file_type_hint": "video",
            "semantic_query": "SF",
        }

        await find_files(
            question="any question that is at least three chars long",
            drive="movies",
            viewer_id="x" * 16,
            overrides=overrides,
            limit=20,
        )

        # Stage B saw the overridden scope.
        history_spy.assert_awaited()
        history_kwargs = history_spy.await_args.kwargs
        assert history_kwargs.get("kind") == "viewed"

        # Stage C saw the overridden semantic_query.
        expand_spy.assert_awaited()
        expand_args = expand_spy.await_args
        # First positional arg or ``semantic_query`` kwarg.
        first_arg = (
            expand_args.args[0]
            if expand_args.args
            else expand_args.kwargs.get("semantic_query")
        )
        assert first_arg == "SF"

        # Stage D saw the overridden file_type filter.
        retrieve_spy.assert_awaited()
        retrieve_kwargs = retrieve_spy.await_args.kwargs
        assert retrieve_kwargs.get("file_type") == "video"


# ---------------------------------------------------------------------------
# Stage B: viewer history filter
# ---------------------------------------------------------------------------


class TestFindFilesStageB:
    @pytest.mark.asyncio
    async def test_history_called_when_personal_scope_and_viewer_id(
        self, monkeypatch
    ):
        _enable_find(monkeypatch)
        decomposed = _decomposed(personal_scope="viewed")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        history_spy = AsyncMock(return_value=["fid1", "fid2"])
        monkeypatch.setattr(service_mod, "fetch_viewer_history", history_spy)
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=["SF"])
        )
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", AsyncMock(return_value=[])
        )

        await find_files(
            question="先週観た映画でSF",
            drive="movies",
            viewer_id="x" * 16,
            overrides=None,
            limit=20,
        )

        history_spy.assert_awaited_once()
        kwargs = history_spy.await_args.kwargs
        assert kwargs["viewer_id"] == "x" * 16
        assert kwargs["drive"] == "movies"
        assert kwargs["after"] == decomposed.time_range.after
        assert kwargs["before"] == decomposed.time_range.before
        assert kwargs["kind"] == "viewed"

    @pytest.mark.asyncio
    async def test_history_skipped_when_personal_scope_none(
        self, monkeypatch
    ):
        _enable_find(monkeypatch)
        decomposed = _decomposed(personal_scope="none", semantic="ベイズ")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        history_spy = AsyncMock(side_effect=AssertionError("must not run"))
        monkeypatch.setattr(service_mod, "fetch_viewer_history", history_spy)
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=["ベイズ"])
        )
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", AsyncMock(return_value=[])
        )

        await find_files(
            question="ベイズについて教えて",
            drive="movies",
            viewer_id="x" * 16,
            overrides=None,
            limit=20,
        )

        history_spy.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_history_skipped_when_viewer_id_missing(
        self, monkeypatch
    ):
        """Personal-scope query without viewer_id -> graceful skip.

        Spec §13.A: viewer_id is opt-in. The service must NOT 4xx —
        instead Stage B is silently bypassed and the rest of the
        pipeline runs.
        """
        _enable_find(monkeypatch)
        decomposed = _decomposed(personal_scope="viewed")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        history_spy = AsyncMock(side_effect=AssertionError("must not run"))
        monkeypatch.setattr(service_mod, "fetch_viewer_history", history_spy)
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=["SF"])
        )
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", AsyncMock(return_value=[])
        )

        result = await find_files(
            question="先週観た映画でSF",
            drive="movies",
            viewer_id=None,  # <-- key
            overrides=None,
            limit=20,
        )

        history_spy.assert_not_awaited()
        # Service still returned a valid response shape.
        payload = result if isinstance(result, dict) else result.model_dump()
        assert "results" in payload

    @pytest.mark.asyncio
    async def test_history_file_ids_passed_as_scope_to_retriever(
        self, monkeypatch
    ):
        """Stage B's file_ids become Stage D's file_id_scope.

        This is the whole point of the personal-history pipeline: the
        retriever runs ONLY against the viewer's filtered file set.
        """
        _enable_find(monkeypatch)
        decomposed = _decomposed(personal_scope="viewed")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        monkeypatch.setattr(
            service_mod,
            "fetch_viewer_history",
            AsyncMock(return_value=["fid1", "fid2", "fid3"]),
        )
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=["SF"])
        )
        retrieve_spy = AsyncMock(return_value=[])
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", retrieve_spy
        )

        await find_files(
            question="先週観た映画でSF",
            drive="movies",
            viewer_id="x" * 16,
            overrides=None,
            limit=20,
        )

        retrieve_spy.assert_awaited()
        kwargs = retrieve_spy.await_args.kwargs
        assert kwargs.get("file_id_scope") == ["fid1", "fid2", "fid3"]


# ---------------------------------------------------------------------------
# Stage C: category expansion
# ---------------------------------------------------------------------------


class TestFindFilesStageC:
    @pytest.mark.asyncio
    async def test_expand_called_when_semantic_query_nonempty(
        self, monkeypatch
    ):
        _enable_find(monkeypatch)
        decomposed = _decomposed(personal_scope="none", semantic="SF")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        expand_spy = AsyncMock(
            return_value=["SF", "science fiction", "宇宙船"]
        )
        monkeypatch.setattr(service_mod, "expand_category", expand_spy)
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", AsyncMock(return_value=[])
        )

        await find_files(
            question="SF っぽいやつ",
            drive="movies",
            viewer_id=None,
            overrides=None,
            limit=20,
        )

        expand_spy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_expand_skipped_when_semantic_query_empty(
        self, monkeypatch
    ):
        """Empty semantic_query -> retrieve runs on structural axes only.

        Spec §8 open question: "空 query ⇒ 構造軸のみで列挙". Stage C
        has nothing to expand and must not be called.
        """
        _enable_find(monkeypatch)
        decomposed = _decomposed(personal_scope="not_viewed", semantic="")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        expand_spy = AsyncMock(side_effect=AssertionError("must not run"))
        monkeypatch.setattr(service_mod, "expand_category", expand_spy)
        monkeypatch.setattr(
            service_mod, "fetch_viewer_history", AsyncMock(return_value=[])
        )
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", AsyncMock(return_value=[])
        )

        await find_files(
            question="今月観てない動画でおすすめ",
            drive="movies",
            viewer_id="x" * 16,
            overrides=None,
            limit=20,
        )

        expand_spy.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_expand_failure_falls_back_to_raw_semantic(
        self, monkeypatch
    ):
        """Mirror query_transform's pattern: expansion failure must NOT
        hard-fail the pipeline. The service falls back to using the raw
        ``semantic_query`` as the keyword query.
        """
        _enable_find(monkeypatch)
        decomposed = _decomposed(personal_scope="none", semantic="SF")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        # expand_category raises -> service must catch it and continue.
        monkeypatch.setattr(
            service_mod,
            "expand_category",
            AsyncMock(side_effect=RuntimeError("LLM blew up")),
        )
        retrieve_spy = AsyncMock(return_value=[])
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", retrieve_spy
        )

        # No exception bubbles out — the user sees an empty list, not
        # a 500.
        result = await find_files(
            question="SF っぽい",
            drive="movies",
            viewer_id=None,
            overrides=None,
            limit=20,
        )

        retrieve_spy.assert_awaited()
        payload = result if isinstance(result, dict) else result.model_dump()
        assert "results" in payload


# ---------------------------------------------------------------------------
# Stage D: scoped retrieval
# ---------------------------------------------------------------------------


class TestFindFilesStageD:
    @pytest.mark.asyncio
    async def test_retrieve_called_with_file_type_filter(
        self, monkeypatch
    ):
        _enable_find(monkeypatch)
        decomposed = _decomposed(file_type="video", personal_scope="none")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=["SF"])
        )
        retrieve_spy = AsyncMock(return_value=[])
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", retrieve_spy
        )

        await find_files(
            question="SF 動画",
            drive="movies",
            viewer_id=None,
            overrides=None,
            limit=20,
        )

        retrieve_spy.assert_awaited()
        kwargs = retrieve_spy.await_args.kwargs
        assert kwargs.get("file_type") == "video"
        assert kwargs.get("drive") == "movies"

    @pytest.mark.asyncio
    async def test_retrieve_called_without_file_type_when_none(
        self, monkeypatch
    ):
        _enable_find(monkeypatch)
        decomposed = _decomposed(file_type="none", personal_scope="none")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=["x"])
        )
        retrieve_spy = AsyncMock(return_value=[])
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", retrieve_spy
        )

        await find_files(
            question="any old question goes here",
            drive="movies",
            viewer_id=None,
            overrides=None,
            limit=20,
        )

        kwargs = retrieve_spy.await_args.kwargs
        # file_type=none from the decomposer should NOT become the
        # literal string "none" filter — that would never match. Either
        # absent or None is acceptable.
        assert kwargs.get("file_type") in (None, "")

    @pytest.mark.asyncio
    async def test_multi_term_expansion_runs_per_term_and_rrf_merges(
        self, monkeypatch
    ):
        """Stage C with 2+ terms must fan out to per-term retrievals.

        Regression: previously the service only used
        ``category_expansion[0]`` as the keyword, which made expansion
        cosmetic — single-term retrieval kept hitting embedding noise
        (e.g. "料理" alone surfaced "スマホ料金プラン"). Mirror the Ask
        path: run ``retrieve_with_keywords`` once per expansion term
        and RRF-merge so the file that matches multiple expansions wins
        over the file that only matches one weakly.
        """
        _enable_find(monkeypatch)
        decomposed = _decomposed(personal_scope="none", semantic="料理")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        terms = ["料理", "レシピ", "調理", "食材"]
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=terms)
        )

        # Per-term retriever returns: only the noisy file appears for
        # the head term, while the cooking file appears across multiple
        # expansions. RRF must rank cooking above noise.
        noise = _retrieved_file(file_id="noise", filename="phone-plan.mp4")
        cooking = _retrieved_file(file_id="cooking", filename="recipe.mp4")

        async def fake_retrieve(*, keywords, **_):
            if keywords == "料理":
                return [noise, cooking]
            return [cooking]

        retrieve_spy = AsyncMock(side_effect=fake_retrieve)
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", retrieve_spy
        )

        result = await find_files(
            question="料理に関する動画",
            drive="movies",
            viewer_id=None,
            overrides=None,
            limit=20,
        )

        # Called exactly once per expansion term.
        assert retrieve_spy.await_count == len(terms)
        called_keywords = sorted(
            call.kwargs["keywords"] for call in retrieve_spy.await_args_list
        )
        assert called_keywords == sorted(terms)

        # RRF merge: cooking (4 hits) ranks above noise (1 hit).
        payload = result if isinstance(result, dict) else result.model_dump()
        ordered_ids = [r["file_id"] for r in payload["results"]]
        assert ordered_ids[0] == "cooking"
        assert "noise" in ordered_ids

    @pytest.mark.asyncio
    async def test_required_overlapping_with_semantic_query_dropped_when_expanding(
        self, monkeypatch
    ):
        """When category expansion runs, ``required`` terms whose
        canonical overlaps with ``semantic_query`` must be dropped from
        per-term retrieves.

        Regression: small local LLMs (gemma e4b etc.) sometimes
        misclassify the broad concept ("料理") as ``required`` even
        though the prompt forbids it. Combined with multi-query
        expansion, every per-term retrieve still hard-filters on
        "料理", excluding actual recipe videos that say "弁当" / "副菜
        レシピ" but never the literal "料理". Proper-noun anchors that
        are NOT in semantic_query (e.g. "東福寺") must survive.
        """
        from app.rag.query_transform import RequiredTerm

        _enable_find(monkeypatch)
        decomposed = _decomposed(personal_scope="none", semantic="料理")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        monkeypatch.setattr(
            service_mod,
            "expand_category",
            AsyncMock(return_value=["料理", "レシピ", "調理"]),
        )

        # LLM mis-classifies "料理" as required AND also extracts a
        # legitimate proper noun "東福寺" — only the overlapping term
        # should be dropped.
        structured_mock = MagicMock()
        structured_mock.required = (
            RequiredTerm(canonical="料理", script="han", aliases=("料理",)),
            RequiredTerm(
                canonical="東福寺", script="han", aliases=("東福寺",)
            ),
        )
        structured_mock.raw_keywords = "料理 東福寺"
        monkeypatch.setattr(
            service_mod,
            "transform_query_structured",
            AsyncMock(return_value=structured_mock),
        )

        retrieve_spy = AsyncMock(return_value=[])
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", retrieve_spy
        )

        await find_files(
            question="東福寺の料理",
            drive="movies",
            viewer_id=None,
            overrides=None,
            limit=20,
        )

        # Every per-term retrieve should drop "料理" but keep "東福寺".
        for call in retrieve_spy.await_args_list:
            req = call.kwargs.get("required")
            assert req is not None, "東福寺 should survive as anchor"
            canonicals = [t.canonical for t in req]
            assert "料理" not in canonicals
            assert "東福寺" in canonicals

    @pytest.mark.asyncio
    async def test_expansion_head_terms_outweigh_tail_terms_in_rrf(
        self, monkeypatch
    ):
        """Linear-decay weighting biases the merge toward head expansion
        terms.

        Regression: with equal-weight RRF, a noise file that ranks
        moderately high in tail terms ("ディナー" / "盛り付け") could
        accumulate enough total score to outvote a true cooking file
        that ranks #1 in only the head terms ("料理" / "レシピ").
        Linear decay (head=1.0, tail=0.5) preserves the head signal
        while still benefiting from tail consensus when present.
        """
        _enable_find(monkeypatch)
        decomposed = _decomposed(personal_scope="none", semantic="料理")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        # 4-term expansion: head 2 = real cooking concepts, tail 2 =
        # broad words that weakly attract noise files.
        terms = ["料理", "レシピ", "ディナー", "盛り付け"]
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=terms)
        )

        cooking = _retrieved_file(file_id="cooking", filename="recipe.mp4")
        noise = _retrieved_file(file_id="noise", filename="podcast.mp4")

        async def fake_retrieve(*, keywords, **_):
            # Realistic split: cooking files match the head terms
            # ("料理"/"レシピ") strongly but not the tail; the tail
            # words ("ディナー"/"盛り付け") happen to surface unrelated
            # videos that mention them in passing. Both at rank 1 of
            # their respective lists, so unweighted RRF would tie.
            # Linear-decay weighting tilts toward head — cooking wins.
            head_terms = {"料理", "レシピ"}
            if keywords in head_terms:
                return [cooking]
            return [noise]

        retrieve_spy = AsyncMock(side_effect=fake_retrieve)
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", retrieve_spy
        )

        result = await find_files(
            question="料理に関する動画",
            drive="movies",
            viewer_id=None,
            overrides=None,
            limit=20,
        )

        payload = result if isinstance(result, dict) else result.model_dump()
        ordered_ids = [r["file_id"] for r in payload["results"]]
        # cooking should rank above noise. Under unweighted RRF
        # they would tie (both rank #1 in two lists each); the
        # linear-decay weighting [1.0, 0.83, 0.67, 0.5] gives cooking
        # the head's full weight while noise rides the discounted
        # tail.
        assert ordered_ids.index("cooking") < ordered_ids.index("noise")

    @pytest.mark.asyncio
    async def test_required_kept_intact_on_single_term_expansion_path(
        self, monkeypatch
    ):
        """Single-keyword (non-expansion) path must NOT drop required.

        The overlap-drop heuristic is justified only when the multi-
        query expansion is actually replacing the discriminating role
        of the keyword. When expansion collapses (single element →
        ``[]`` per ``_resolve_category_expansion``), required keeps
        its full hard-filter role.
        """
        from app.rag.query_transform import RequiredTerm

        _enable_find(monkeypatch)
        decomposed = _decomposed(personal_scope="none", semantic="料理")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        # Single-element expansion → helper returns [] → fallback path.
        monkeypatch.setattr(
            service_mod,
            "expand_category",
            AsyncMock(return_value=["料理"]),
        )

        structured_mock = MagicMock()
        structured_mock.required = (
            RequiredTerm(canonical="料理", script="han", aliases=("料理",)),
        )
        structured_mock.raw_keywords = "料理"
        monkeypatch.setattr(
            service_mod,
            "transform_query_structured",
            AsyncMock(return_value=structured_mock),
        )

        retrieve_spy = AsyncMock(return_value=[])
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", retrieve_spy
        )

        await find_files(
            question="料理",
            drive="movies",
            viewer_id=None,
            overrides=None,
            limit=20,
        )

        # Single retrieve, required intact.
        assert retrieve_spy.await_count == 1
        req = retrieve_spy.await_args.kwargs.get("required")
        assert req is not None
        assert [t.canonical for t in req] == ["料理"]


# ---------------------------------------------------------------------------
# LLM answer-generation must NOT run
# ---------------------------------------------------------------------------


class TestFindFilesNoLLMAnswerCall:
    """Spec §3.3: Stage E_find calls the LLM zero times.

    The whole pitch of Find mode is "no hallucination, no answer text".
    These tests ensure no future refactor accidentally pipes the
    Find pipeline through ``stream_answer`` or ``answer_question``.
    """

    @pytest.mark.asyncio
    async def test_does_not_call_stream_answer(self, monkeypatch):
        _enable_find(monkeypatch)
        monkeypatch.setattr(
            service_mod,
            "decompose_query",
            AsyncMock(return_value=_decomposed(personal_scope="none")),
        )
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=["x"])
        )
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", AsyncMock(return_value=[])
        )

        # Spy on stream_answer so any accidental invocation explodes.
        stream_spy = MagicMock(side_effect=AssertionError("must not run"))
        monkeypatch.setattr(service_mod, "stream_answer", stream_spy)

        await find_files(
            question="anything plausible",
            drive="movies",
            viewer_id=None,
            overrides=None,
            limit=20,
        )

        stream_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_call_answer_question(self, monkeypatch):
        _enable_find(monkeypatch)
        monkeypatch.setattr(
            service_mod,
            "decompose_query",
            AsyncMock(return_value=_decomposed(personal_scope="none")),
        )
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=["x"])
        )
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", AsyncMock(return_value=[])
        )

        answer_spy = AsyncMock(side_effect=AssertionError("must not run"))
        monkeypatch.setattr(service_mod, "answer_question", answer_spy)

        await find_files(
            question="anything plausible",
            drive="movies",
            viewer_id=None,
            overrides=None,
            limit=20,
        )

        answer_spy.assert_not_awaited()


# ---------------------------------------------------------------------------
# Response shape (spec §3.2)
# ---------------------------------------------------------------------------


class TestFindFilesResponseShape:
    @pytest.mark.asyncio
    async def test_returns_top_level_keys(self, monkeypatch):
        _enable_find(monkeypatch)
        monkeypatch.setattr(
            service_mod,
            "decompose_query",
            AsyncMock(return_value=_decomposed()),
        )
        monkeypatch.setattr(
            service_mod,
            "fetch_viewer_history",
            AsyncMock(return_value=["fid1"]),
        )
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=["SF"])
        )
        monkeypatch.setattr(
            service_mod,
            "retrieve_with_keywords",
            AsyncMock(return_value=[_retrieved_file()]),
        )

        result = await find_files(
            question="先週観た映画でSF",
            drive="movies",
            viewer_id="x" * 16,
            overrides=None,
            limit=20,
        )

        payload = result if isinstance(result, dict) else result.model_dump()
        assert set(payload.keys()) >= {
            "decomposed",
            "results",
            "total",
            "limit",
        }
        assert payload["limit"] == 20
        assert isinstance(payload["total"], int)

    @pytest.mark.asyncio
    async def test_decomposed_block_has_all_five_keys(self, monkeypatch):
        _enable_find(monkeypatch)
        decomposed = _decomposed(
            label="last_week", personal_scope="viewed", file_type="video"
        )
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        monkeypatch.setattr(
            service_mod,
            "fetch_viewer_history",
            AsyncMock(return_value=["fid1"]),
        )
        monkeypatch.setattr(
            service_mod,
            "expand_category",
            AsyncMock(return_value=["SF", "science fiction", "宇宙船"]),
        )
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", AsyncMock(return_value=[])
        )

        result = await find_files(
            question="先週観た映画でSF",
            drive="movies",
            viewer_id="x" * 16,
            overrides=None,
            limit=20,
        )

        payload = result if isinstance(result, dict) else result.model_dump()
        for key in (
            "time_range",
            "personal_scope",
            "file_type_hint",
            "semantic_query",
            "category_expansion",
        ):
            assert key in payload["decomposed"], (
                f"decomposed missing required spec key: {key}"
            )
        # Category expansion is the LLM-emitted vocabulary.
        assert payload["decomposed"]["category_expansion"] == [
            "SF",
            "science fiction",
            "宇宙船",
        ]

    @pytest.mark.asyncio
    async def test_each_result_has_spec_schema(self, monkeypatch):
        """Spec §3.2: results[].file_id, score, hit.{kind,location,text},
        file.{name, file_type, thumbnail_url, viewed_at}."""
        _enable_find(monkeypatch)
        monkeypatch.setattr(
            service_mod,
            "decompose_query",
            AsyncMock(return_value=_decomposed(personal_scope="none")),
        )
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=["SF"])
        )
        monkeypatch.setattr(
            service_mod,
            "retrieve_with_keywords",
            AsyncMock(return_value=[_retrieved_file()]),
        )

        result = await find_files(
            question="anything plausible",
            drive="movies",
            viewer_id=None,
            overrides=None,
            limit=20,
        )

        payload = result if isinstance(result, dict) else result.model_dump()
        assert len(payload["results"]) == 1
        item = payload["results"][0]

        # Top-level fields
        assert "file_id" in item
        assert "score" in item
        assert isinstance(item["score"], (int, float))

        # hit.* fields
        assert "hit" in item
        assert "kind" in item["hit"]
        assert "location" in item["hit"]
        assert "text" in item["hit"]

        # file.* fields
        assert "file" in item
        assert "name" in item["file"]
        assert "file_type" in item["file"]
        assert "thumbnail_url" in item["file"]
        # viewed_at present (may be None when no viewer_id was supplied
        # to history filter, but the key must exist for stable shape).
        assert "viewed_at" in item["file"]

    @pytest.mark.asyncio
    async def test_hit_text_is_retrieve_segment_not_llm_output(
        self, monkeypatch
    ):
        """Tier 1: hit.text comes from the retriever segment verbatim.

        If a future refactor accidentally piped the text through an
        LLM rewrite step (e.g. to "summarise the hit") this test would
        fail — the marker string we plant in the retrieved segment
        must round-trip unchanged.
        """
        _enable_find(monkeypatch)
        marker = "VERBATIM-MARKER-12345 ...宇宙船が時空を超えて..."
        monkeypatch.setattr(
            service_mod,
            "decompose_query",
            AsyncMock(return_value=_decomposed(personal_scope="none")),
        )
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=["SF"])
        )
        monkeypatch.setattr(
            service_mod,
            "retrieve_with_keywords",
            AsyncMock(return_value=[_retrieved_file(text=marker)]),
        )

        result = await find_files(
            question="anything plausible",
            drive="movies",
            viewer_id=None,
            overrides=None,
            limit=20,
        )

        payload = result if isinstance(result, dict) else result.model_dump()
        assert payload["results"][0]["hit"]["text"] == marker


# ---------------------------------------------------------------------------
# Limit handling
# ---------------------------------------------------------------------------


class TestFindFilesLimit:
    @pytest.mark.asyncio
    async def test_results_capped_at_limit(self, monkeypatch):
        """When the retriever returns more than ``limit``, the response
        must be truncated to ``limit`` entries.

        ``total`` may exceed ``limit`` to communicate "more available";
        the spec example shows ``total: 8, limit: 20`` so total is the
        true count and limit is the page size.
        """
        _enable_find(monkeypatch)
        monkeypatch.setattr(
            service_mod,
            "decompose_query",
            AsyncMock(return_value=_decomposed(personal_scope="none")),
        )
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=["x"])
        )
        # Retriever returns 25 files; limit is 5.
        many = [_retrieved_file(file_id=f"f-{i}") for i in range(25)]
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", AsyncMock(return_value=many)
        )

        result = await find_files(
            question="any plausible question",
            drive="movies",
            viewer_id=None,
            overrides=None,
            limit=5,
        )

        payload = result if isinstance(result, dict) else result.model_dump()
        assert len(payload["results"]) == 5
        assert payload["limit"] == 5
        # total reflects the post-retrieval count, which the service
        # may report as the unfiltered length OR the limited length —
        # both interpretations are spec-compatible. We assert the
        # weaker invariant: total >= len(results).
        assert payload["total"] >= len(payload["results"])

    @pytest.mark.asyncio
    async def test_default_limit_is_20(self, monkeypatch):
        """Spec §3.2 example uses limit=20 by default."""
        _enable_find(monkeypatch)
        monkeypatch.setattr(
            service_mod,
            "decompose_query",
            AsyncMock(return_value=_decomposed(personal_scope="none")),
        )
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=["x"])
        )
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", AsyncMock(return_value=[])
        )

        result = await find_files(
            question="any plausible question",
            drive="movies",
            viewer_id=None,
            overrides=None,
            # limit omitted -> service default
        )

        payload = result if isinstance(result, dict) else result.model_dump()
        assert payload["limit"] == 20


# ---------------------------------------------------------------------------
# Empty retrieve -> empty results, no error
# ---------------------------------------------------------------------------


class TestFindFilesEmptyRetrieve:
    @pytest.mark.asyncio
    async def test_empty_retrieve_returns_empty_results_not_error(
        self, monkeypatch
    ):
        _enable_find(monkeypatch)
        monkeypatch.setattr(
            service_mod,
            "decompose_query",
            AsyncMock(return_value=_decomposed(personal_scope="none")),
        )
        monkeypatch.setattr(
            service_mod, "expand_category", AsyncMock(return_value=["zzz"])
        )
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", AsyncMock(return_value=[])
        )

        result = await find_files(
            question="something matching nothing",
            drive="movies",
            viewer_id=None,
            overrides=None,
            limit=20,
        )

        payload = result if isinstance(result, dict) else result.model_dump()
        assert payload["results"] == []
        assert payload["total"] == 0
        # decomposed block still populated so frontend can render the
        # chip layout even on zero hits.
        assert "decomposed" in payload
