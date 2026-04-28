"""Tests for app.rag.query_decomposer (Stage A query decomposition).

The decomposer wraps a single LLM call with strict graceful-degradation
semantics: every failure path returns
``DecomposedQuery.passthrough(natural_query)`` so the caller never has
to handle ``None``. These tests exercise the happy path plus every
documented failure mode plus the deterministic time-range arithmetic.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest


# Heavy ML deps are stubbed so importing app.rag.query_decomposer (which
# transitively pulls app.dependencies → app.llm) does not need real
# torch / sentence-transformers / sqlite-vec at import time.
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

from app.rag.query_decomposer import (  # noqa: E402
    DecomposedQuery,
    TimeRange,
    _resolve_time_range,
    decompose_query,
)


# A fixed "current time" used for every time-range assertion. Pinned to
# a Wednesday so week-boundary arithmetic has obvious right-and-wrong
# answers (Monday is two days before, Sunday is four days after).
NOW = datetime(2026, 4, 22, 14, 30, 0, tzinfo=UTC)


def _llm_stub(
    *,
    enabled: bool = True,
    response: dict | list | None = None,
    raises: type[Exception] | None = None,
) -> MagicMock:
    """Build a MagicMock LLM client whose generate_json is stubbed."""
    client = MagicMock()
    client.enabled = enabled
    if raises is not None:
        client.generate_json = AsyncMock(side_effect=raises("boom"))
    else:
        client.generate_json = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# _resolve_time_range — symbolic label → concrete window
# ---------------------------------------------------------------------------


class TestResolveTimeRange:
    """Pure-function tests for the symbolic-label resolver."""

    def test_none_label_returns_empty(self):
        tr = _resolve_time_range("none", now=NOW, max_lookback_days=365)
        assert tr.label == "none"
        assert tr.after is None
        assert tr.before is None

    def test_unknown_label_returns_empty(self):
        # Defence in depth: even though decompose_query normalises the
        # label, _resolve_time_range must not blow up on garbage input.
        tr = _resolve_time_range("yesterweek", now=NOW, max_lookback_days=365)
        assert tr.label == "none"

    def test_today_window(self):
        tr = _resolve_time_range("today", now=NOW, max_lookback_days=365)
        assert tr.label == "today"
        assert tr.after == datetime(2026, 4, 22, 0, 0, 0, tzinfo=UTC)
        assert tr.before == datetime(2026, 4, 23, 0, 0, 0, tzinfo=UTC)

    def test_yesterday_window(self):
        tr = _resolve_time_range("yesterday", now=NOW, max_lookback_days=365)
        assert tr.after == datetime(2026, 4, 21, 0, 0, 0, tzinfo=UTC)
        assert tr.before == datetime(2026, 4, 22, 0, 0, 0, tzinfo=UTC)

    def test_this_week_starts_monday(self):
        # NOW is Wednesday; Monday is 2026-04-20.
        tr = _resolve_time_range("this_week", now=NOW, max_lookback_days=365)
        assert tr.after == datetime(2026, 4, 20, 0, 0, 0, tzinfo=UTC)
        assert tr.before == datetime(2026, 4, 27, 0, 0, 0, tzinfo=UTC)

    def test_last_week_is_seven_days_before_this_week(self):
        tr = _resolve_time_range("last_week", now=NOW, max_lookback_days=365)
        assert tr.after == datetime(2026, 4, 13, 0, 0, 0, tzinfo=UTC)
        assert tr.before == datetime(2026, 4, 20, 0, 0, 0, tzinfo=UTC)

    def test_this_month_runs_to_now(self):
        tr = _resolve_time_range("this_month", now=NOW, max_lookback_days=365)
        assert tr.after == datetime(2026, 4, 1, 0, 0, 0, tzinfo=UTC)
        # this_month uses ``now`` (not next-month-start) as upper bound
        # so the window matches the watch_history rows the user actually
        # has — see docstring on ``_resolve_time_range``.
        assert tr.before == NOW

    def test_last_month_is_calendar_previous_month(self):
        tr = _resolve_time_range("last_month", now=NOW, max_lookback_days=365)
        assert tr.after == datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)
        assert tr.before == datetime(2026, 4, 1, 0, 0, 0, tzinfo=UTC)

    def test_last_month_january_wraps_to_december(self):
        # January wraparound is the only branch in _previous_month_start
        # that does not fall out of plain (year, month - 1) arithmetic.
        jan = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        tr = _resolve_time_range("last_month", now=jan, max_lookback_days=365)
        assert tr.after == datetime(2025, 12, 1, 0, 0, 0, tzinfo=UTC)
        assert tr.before == datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_this_year(self):
        tr = _resolve_time_range("this_year", now=NOW, max_lookback_days=365)
        assert tr.after == datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert tr.before == NOW

    def test_last_year_unclipped(self):
        # With a generous lookback ceiling (3 years) the calendar
        # boundaries are kept verbatim. The 365-day default *does*
        # clip "last_year" — that path is covered by the dedicated
        # clipping test below.
        tr = _resolve_time_range(
            "last_year", now=NOW, max_lookback_days=365 * 3
        )
        assert tr.after == datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert tr.before == datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_last_year_default_lookback_clips_lower_bound(self):
        # NOW = 2026-04-22; last_year normally starts 2025-01-01 (~477
        # days ago) but max_lookback_days=365 floors the lower bound at
        # NOW.midnight - 365 days = 2025-04-22 00:00. The upper bound
        # (2026-01-01) is unchanged. This is the documented behaviour
        # for the spec's default ceiling — the rule is "never look
        # further back than max_lookback_days, no matter what label".
        tr = _resolve_time_range("last_year", now=NOW, max_lookback_days=365)
        assert tr.after == datetime(2025, 4, 22, 0, 0, 0, tzinfo=UTC)
        assert tr.before == datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_recent_uses_default_14_days(self):
        tr = _resolve_time_range("recent", now=NOW, max_lookback_days=365)
        assert tr.after == datetime(2026, 4, 8, 0, 0, 0, tzinfo=UTC)
        assert tr.before == NOW

    def test_recent_honours_recent_days_override(self):
        tr = _resolve_time_range(
            "recent", now=NOW, max_lookback_days=365, recent_days=3
        )
        assert tr.after == datetime(2026, 4, 19, 0, 0, 0, tzinfo=UTC)

    def test_max_lookback_clips_lower_bound(self):
        # this_month would normally start 2026-04-01, but a 10-day floor
        # pulls the lower bound up to NOW.midnight - 10 days = 2026-04-12.
        # ``this_month`` is the right label to test the non-collapse
        # branch: the upper bound (NOW) sits well above any reasonable
        # floor, so clipping shifts ``after`` without emptying the window.
        tr = _resolve_time_range("this_month", now=NOW, max_lookback_days=10)
        assert tr.after == datetime(2026, 4, 12, 0, 0, 0, tzinfo=UTC)
        assert tr.before == NOW

    def test_floor_past_upper_bound_returns_empty(self):
        # Pathological: max_lookback_days=0 makes the floor today, but
        # last_year's upper bound is the start of this year — already in
        # the past. The window collapses, so we return empty().
        tr = _resolve_time_range("last_year", now=NOW, max_lookback_days=0)
        assert tr.label == "none"
        assert tr.after is None
        assert tr.before is None


# ---------------------------------------------------------------------------
# decompose_query — full LLM happy-path
# ---------------------------------------------------------------------------


class TestDecomposeQueryHappyPath:
    @pytest.mark.asyncio
    async def test_full_decomposition(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_decomposer.get_llm_client",
            lambda: _llm_stub(
                response={
                    "time_range": "last_week",
                    "personal_scope": "viewed",
                    "file_type_hint": "video",
                    "semantic_query": "SF",
                }
            ),
        )

        result = await decompose_query(
            "先週観た映画の中で SF っぽいのどれ？",
            now=NOW,
        )

        assert result.raw_query == "先週観た映画の中で SF っぽいのどれ？"
        assert result.personal_scope == "viewed"
        assert result.file_type_hint == "video"
        assert result.semantic_query == "SF"
        assert result.time_range.label == "last_week"
        assert result.time_range.after == datetime(
            2026, 4, 13, 0, 0, 0, tzinfo=UTC
        )
        assert result.time_range.before == datetime(
            2026, 4, 20, 0, 0, 0, tzinfo=UTC
        )
        assert result.has_personal_signal is True

    @pytest.mark.asyncio
    async def test_not_viewed_scope(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_decomposer.get_llm_client",
            lambda: _llm_stub(
                response={
                    "time_range": "this_month",
                    "personal_scope": "not_viewed",
                    "file_type_hint": "video",
                    "semantic_query": "",
                }
            ),
        )

        result = await decompose_query("今月観てない動画でおすすめ", now=NOW)
        assert result.personal_scope == "not_viewed"
        assert result.semantic_query == ""  # legitimate "no concept" signal
        assert result.has_personal_signal is True

    @pytest.mark.asyncio
    async def test_no_personal_signal(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_decomposer.get_llm_client",
            lambda: _llm_stub(
                response={
                    "time_range": "none",
                    "personal_scope": "none",
                    "file_type_hint": "none",
                    "semantic_query": "ベイズ統計",
                }
            ),
        )

        result = await decompose_query("ベイズ統計について", now=NOW)
        assert result.has_personal_signal is False
        assert result.time_range.label == "none"
        assert result.semantic_query == "ベイズ統計"


# ---------------------------------------------------------------------------
# decompose_query — graceful-degradation paths
# ---------------------------------------------------------------------------


class TestDecomposeQueryFallback:
    @pytest.mark.asyncio
    async def test_empty_query_returns_passthrough(self, monkeypatch):
        # No LLM call should happen — the empty string is rejected
        # before the dependency lookup.
        called = False

        def _fake_get_llm_client():
            nonlocal called
            called = True
            return _llm_stub()

        monkeypatch.setattr(
            "app.rag.query_decomposer.get_llm_client", _fake_get_llm_client
        )

        result = await decompose_query("   ", now=NOW)
        assert result == DecomposedQuery.passthrough("   ")
        assert called is False

    @pytest.mark.asyncio
    async def test_llm_disabled_returns_passthrough(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_decomposer.get_llm_client",
            lambda: _llm_stub(enabled=False),
        )
        result = await decompose_query("先週観た映画", now=NOW)
        assert result == DecomposedQuery.passthrough("先週観た映画")

    @pytest.mark.asyncio
    async def test_llm_runtime_error_returns_passthrough(self, monkeypatch):
        # Dependency container not initialised yet — startup race.
        def _raises():
            raise RuntimeError("not ready")

        monkeypatch.setattr(
            "app.rag.query_decomposer.get_llm_client", _raises
        )
        result = await decompose_query("先週観た映画", now=NOW)
        assert result == DecomposedQuery.passthrough("先週観た映画")

    @pytest.mark.asyncio
    async def test_non_dict_response_returns_passthrough(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_decomposer.get_llm_client",
            lambda: _llm_stub(response=["not", "a", "dict"]),
        )
        result = await decompose_query("先週観た映画", now=NOW)
        assert result == DecomposedQuery.passthrough("先週観た映画")

    @pytest.mark.asyncio
    async def test_none_response_returns_passthrough(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_decomposer.get_llm_client",
            lambda: _llm_stub(response=None),
        )
        result = await decompose_query("先週観た映画", now=NOW)
        assert result == DecomposedQuery.passthrough("先週観た映画")

    @pytest.mark.asyncio
    async def test_unknown_labels_normalise_to_none(self, monkeypatch):
        # The LLM hallucinated ``last_quarter`` / ``maybe_viewed`` etc.
        # Each unknown label collapses to ``"none"`` so downstream stages
        # do not attempt to act on garbage.
        monkeypatch.setattr(
            "app.rag.query_decomposer.get_llm_client",
            lambda: _llm_stub(
                response={
                    "time_range": "last_quarter",
                    "personal_scope": "maybe_viewed",
                    "file_type_hint": "spreadsheet",
                    "semantic_query": "ベイズ統計",
                }
            ),
        )
        result = await decompose_query("ベイズ統計", now=NOW)
        assert result.time_range.label == "none"
        assert result.personal_scope == "none"
        assert result.file_type_hint == "none"
        # semantic_query is preserved even when other fields collapse.
        assert result.semantic_query == "ベイズ統計"

    @pytest.mark.asyncio
    async def test_collapsed_decomposition_falls_back_to_raw(self, monkeypatch):
        # All structured fields ``"none"`` *and* semantic_query empty —
        # the LLM did not actually understand the query. We restore
        # the raw text so retrieval still has something to embed.
        monkeypatch.setattr(
            "app.rag.query_decomposer.get_llm_client",
            lambda: _llm_stub(
                response={
                    "time_range": "none",
                    "personal_scope": "none",
                    "file_type_hint": "none",
                    "semantic_query": "",
                }
            ),
        )
        result = await decompose_query("曖昧な質問", now=NOW)
        assert result.semantic_query == "曖昧な質問"

    @pytest.mark.asyncio
    async def test_max_lookback_clips_resolution(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_decomposer.get_llm_client",
            lambda: _llm_stub(
                response={
                    "time_range": "this_month",
                    "personal_scope": "viewed",
                    "file_type_hint": "none",
                    "semantic_query": "アニメ",
                }
            ),
        )

        result = await decompose_query(
            "今月観たアニメ",
            now=NOW,
            max_lookback_days=10,
        )
        # this_month would normally start 2026-04-01 but the 10-day
        # ceiling pulls the lower bound up to NOW - 10 days = 2026-04-12.
        # The window stays non-empty because the upper bound is NOW.
        assert result.time_range.label == "this_month"
        assert result.time_range.after == datetime(
            2026, 4, 12, 0, 0, 0, tzinfo=UTC
        )

    @pytest.mark.asyncio
    async def test_default_now_uses_clock(self, monkeypatch):
        # When ``now`` is omitted, the resolver must still return a
        # populated TimeRange for non-``none`` labels — proving the
        # default datetime.now() path works without an explicit override.
        monkeypatch.setattr(
            "app.rag.query_decomposer.get_llm_client",
            lambda: _llm_stub(
                response={
                    "time_range": "today",
                    "personal_scope": "viewed",
                    "file_type_hint": "none",
                    "semantic_query": "メモ",
                }
            ),
        )
        result = await decompose_query("今日観たメモ")
        assert result.time_range.label == "today"
        assert result.time_range.after is not None
        assert result.time_range.before is not None
