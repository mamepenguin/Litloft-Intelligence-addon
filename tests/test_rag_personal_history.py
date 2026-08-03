"""Tests for the personal-history scoping in app.rag.service.

Two pieces under test:

1. ``_resolve_personal_history`` — the helper that runs Stages A + B
   and returns the file_id_scope for the retriever, including the
   ``fallback_when_empty`` strict/graceful branching.
2. ``stream_answer`` SSE event ordering when personal-history is
   active — ``query_decomposed`` and ``history_filter`` events fire
   before ``keywords`` so the UI can show "filtering by N viewed
   files" before chunk retrieval kicks off.

The tests stub ``decompose_query`` and ``fetch_viewer_history`` so
the assertions stay focused on the wiring, not on the underlying
LLM/HTTP behaviour (those are covered in the dedicated test files).
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
    HierarchicalRagConfig,
    PersonalHistoryConfig,
    RagConfig,
)
from app.rag import service as service_mod  # noqa: E402
from app.rag.query_decomposer import DecomposedQuery, TimeRange  # noqa: E402


def _decomposed(
    *,
    label: str = "last_week",
    after: datetime | None = datetime(2026, 4, 13, tzinfo=UTC),
    before: datetime | None = datetime(2026, 4, 20, tzinfo=UTC),
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


def _enable_personal_history(
    monkeypatch,
    *,
    fallback: str = "graceful",
    enabled: bool = True,
) -> None:
    """Patch settings.rag with a config that enables personal-history.

    The dataclass is frozen, so we replace the whole RagConfig (mirrors
    how production loads it from YAML).
    """
    rag_cfg = RagConfig(
        hierarchical=HierarchicalRagConfig(enabled=False),
        personal_history=PersonalHistoryConfig(
            enabled=enabled, fallback_when_empty=fallback
        ),
        category_expansion=CategoryExpansionConfig(),
    )
    settings_mock = MagicMock()
    settings_mock.rag = rag_cfg
    monkeypatch.setattr(service_mod, "settings", settings_mock)


# ---------------------------------------------------------------------------
# _resolve_personal_history
# ---------------------------------------------------------------------------


class TestResolvePersonalHistoryBypass:
    """Bypass paths return ``decomposed=None`` / ``file_ids=None``."""

    @pytest.mark.asyncio
    async def test_disabled_short_circuits(self, monkeypatch):
        _enable_personal_history(monkeypatch, enabled=False)
        # decompose_query and fetch_viewer_history must NOT be called.
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(side_effect=AssertionError)
        )
        monkeypatch.setattr(
            service_mod,
            "fetch_viewer_history",
            AsyncMock(side_effect=AssertionError),
        )
        result = await service_mod._resolve_personal_history(
            query="先週観た映画", viewer_id="x" * 16, drive="movies"
        )
        assert result.decomposed is None
        assert result.file_ids is None
        assert result.short_circuit is False

    @pytest.mark.asyncio
    async def test_no_viewer_short_circuits(self, monkeypatch):
        _enable_personal_history(monkeypatch)
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(side_effect=AssertionError)
        )
        result = await service_mod._resolve_personal_history(
            query="先週観た映画", viewer_id=None, drive="movies"
        )
        assert result.decomposed is None
        assert result.file_ids is None

    @pytest.mark.asyncio
    async def test_no_drive_short_circuits(self, monkeypatch):
        _enable_personal_history(monkeypatch)
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(side_effect=AssertionError)
        )
        result = await service_mod._resolve_personal_history(
            query="先週観た映画", viewer_id="x" * 16, drive=None
        )
        assert result.decomposed is None
        assert result.file_ids is None


class TestResolvePersonalHistorySignals:
    @pytest.mark.asyncio
    async def test_no_personal_signal_skips_history(self, monkeypatch):
        _enable_personal_history(monkeypatch)
        decomposed = _decomposed(personal_scope="none", semantic="ベイズ統計")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        history_mock = AsyncMock(side_effect=AssertionError)
        monkeypatch.setattr(service_mod, "fetch_viewer_history", history_mock)

        result = await service_mod._resolve_personal_history(
            query="ベイズ統計", viewer_id="x" * 16, drive="movies"
        )
        # Decomposition is surfaced for SSE transparency, but Stage B
        # is skipped entirely.
        assert result.decomposed is decomposed
        assert result.file_ids is None
        history_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_personal_signal_calls_history(self, monkeypatch):
        _enable_personal_history(monkeypatch)
        decomposed = _decomposed(personal_scope="viewed")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        history_mock = AsyncMock(return_value=["fid1", "fid2"])
        monkeypatch.setattr(service_mod, "fetch_viewer_history", history_mock)

        result = await service_mod._resolve_personal_history(
            query="先週観た映画", viewer_id="x" * 16, drive="movies"
        )
        assert result.file_ids == ["fid1", "fid2"]
        # Stage B was called with the decomposed window + scope.
        history_mock.assert_awaited_once()
        kwargs = history_mock.await_args.kwargs
        assert kwargs["viewer_id"] == "x" * 16
        assert kwargs["drive"] == "movies"
        assert kwargs["after"] == decomposed.time_range.after
        assert kwargs["before"] == decomposed.time_range.before
        assert kwargs["kind"] == "viewed"


class TestResolvePersonalHistoryEmptyResult:
    """``fallback_when_empty`` decides what to do when Stage B is empty."""

    @pytest.mark.asyncio
    async def test_graceful_drops_filter(self, monkeypatch):
        _enable_personal_history(monkeypatch, fallback="graceful")
        decomposed = _decomposed(personal_scope="viewed")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        monkeypatch.setattr(
            service_mod,
            "fetch_viewer_history",
            AsyncMock(return_value=[]),
        )
        result = await service_mod._resolve_personal_history(
            query="先週観た映画", viewer_id="x" * 16, drive="movies"
        )
        # decomposed is still surfaced (UI hint), file_ids drops to
        # None so the caller runs the legacy retrieval path.
        assert result.decomposed is decomposed
        assert result.file_ids is None
        assert result.short_circuit is False

    @pytest.mark.asyncio
    async def test_strict_short_circuits(self, monkeypatch):
        _enable_personal_history(monkeypatch, fallback="strict")
        decomposed = _decomposed(personal_scope="viewed")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        monkeypatch.setattr(
            service_mod,
            "fetch_viewer_history",
            AsyncMock(return_value=[]),
        )
        result = await service_mod._resolve_personal_history(
            query="先週観た映画", viewer_id="x" * 16, drive="movies"
        )
        # Strict mode: short_circuit set so the streaming caller can
        # emit "該当なし" without further retrieval.
        assert result.short_circuit is True
        assert result.file_ids == []


# ---------------------------------------------------------------------------
# stream_answer SSE wiring
# ---------------------------------------------------------------------------


class TestStreamAnswerPersonalHistoryEvents:
    """``query_decomposed`` and ``history_filter`` fire before retrieval."""

    @pytest.mark.asyncio
    async def test_strict_short_circuit_emits_done_only(self, monkeypatch):
        # Verify the strict-fallback path doesn't run keywords / sources.
        _enable_personal_history(monkeypatch, fallback="strict")
        decomposed = _decomposed(personal_scope="viewed")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        monkeypatch.setattr(
            service_mod,
            "fetch_viewer_history",
            AsyncMock(return_value=[]),
        )
        # transform_query and retrieval helpers must NOT be called.
        monkeypatch.setattr(
            service_mod,
            "transform_query",
            AsyncMock(side_effect=AssertionError("must not be called")),
        )
        monkeypatch.setattr(
            service_mod,
            "_run_hierarchical_retrieval",
            AsyncMock(side_effect=AssertionError("must not be called")),
        )
        monkeypatch.setattr(
            service_mod,
            "retrieve_with_keywords",
            AsyncMock(side_effect=AssertionError("must not be called")),
        )

        events = []
        async for evt in service_mod.stream_answer(
            query="先週観た映画",
            credential=None,
            viewer_id="x" * 16,
            drive="movies",
        ):
            events.append(evt)

        kinds = [e.kind for e in events]
        # query_decomposed → history_filter → citations(empty) → done
        assert kinds == ["query_decomposed", "history_filter", "citations", "done"]
        # history_filter advertises the empty match.
        assert events[1].data["matched_file_count"] == 0

    @pytest.mark.asyncio
    async def test_graceful_path_falls_through_to_legacy(self, monkeypatch):
        # Empty Stage B + graceful → file_ids None → service runs the
        # hierarchical helper. We mock that helper to return [] so the
        # pipeline collapses cleanly without touching the LLM.
        _enable_personal_history(monkeypatch, fallback="graceful")
        decomposed = _decomposed(personal_scope="viewed")
        monkeypatch.setattr(
            service_mod, "decompose_query", AsyncMock(return_value=decomposed)
        )
        monkeypatch.setattr(
            service_mod,
            "fetch_viewer_history",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(
            service_mod,
            "transform_query",
            AsyncMock(return_value="SF 映画"),
        )
        monkeypatch.setattr(
            service_mod,
            "_run_hierarchical_retrieval",
            AsyncMock(return_value=([], None, None)),
        )

        events = []
        async for evt in service_mod.stream_answer(
            query="先週観た映画",
            credential=None,
            viewer_id="x" * 16,
            drive="movies",
        ):
            events.append(evt)

        kinds = [e.kind for e in events]
        # Graceful: history_filter is NOT emitted because file_ids
        # collapsed to None (no scope was applied). query_decomposed
        # is still emitted so the UI can hint that the personal
        # narrowing was *attempted*.
        assert "query_decomposed" in kinds
        assert "history_filter" not in kinds
        assert "keywords" in kinds  # legacy path ran
        # Empty retrieval → empty sources, citations, done.
        assert kinds[-1] == "done"

    @pytest.mark.asyncio
    async def test_personal_history_scope_used_for_retrieval(self, monkeypatch):
        """Stage B file_ids feed retrieve_with_keywords as file_id_scope."""
        _enable_personal_history(monkeypatch)
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
            service_mod,
            "transform_query",
            AsyncMock(return_value="SF 映画"),
        )
        retrieve_mock = AsyncMock(return_value=[])
        monkeypatch.setattr(
            service_mod, "retrieve_with_keywords", retrieve_mock
        )
        # Hierarchical helper must NOT be called when personal_history
        # gives us a scope.
        monkeypatch.setattr(
            service_mod,
            "_run_hierarchical_retrieval",
            AsyncMock(side_effect=AssertionError("must not be called")),
        )

        events = []
        async for evt in service_mod.stream_answer(
            query="先週観た映画",
            credential=None,
            viewer_id="x" * 16,
            drive="movies",
        ):
            events.append(evt)

        retrieve_mock.assert_awaited_once()
        kwargs = retrieve_mock.await_args.kwargs
        assert kwargs["file_id_scope"] == ["fid1", "fid2", "fid3"]
        # SSE: history_filter event reports the matched count.
        history_evt = next(e for e in events if e.kind == "history_filter")
        assert history_evt.data["matched_file_count"] == 3
