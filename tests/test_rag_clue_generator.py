"""Tests for app.rag.clue_generator (Stage 2 multi-query expansion).

The generator wraps a single LLM call with strict graceful-degradation
semantics: on any failure the result is the single-element list
``[fallback_keywords]`` so the downstream retriever can iterate
without special-casing the empty-list path. These tests exercise the
happy path plus every documented failure mode.

``fetch_long_summaries`` is covered in TestFetchLongSummaries — it's
the only DB touch point in this module so isolating it here keeps
``app.rag.service`` tests free of file_summaries fixtures.
"""

import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

# Heavy ML deps are stubbed so importing app.rag.clue_generator (which
# transitively pulls app.search via the keyword filter / DB module)
# does not need real torch / sentence-transformers / sqlite-vec.
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

from app.rag.clue_generator import fetch_long_summaries, generate_clues  # noqa: E402
from app.rag import clue_generator as cg_mod  # noqa: E402


# ---------------------------------------------------------------------------
# generate_clues
# ---------------------------------------------------------------------------


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


class TestGenerateCluesHappyPath:
    @pytest.mark.asyncio
    async def test_returns_clues_list_from_llm(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.clue_generator.get_llm_client",
            lambda: _llm_stub(
                response={"clues": ["京都 紅葉", "嵐山", "東福寺"]}
            ),
        )

        result = await generate_clues(
            "京都の紅葉について教えて",
            ["京都の紅葉名所巡り", "嵐山の秋の散策"],
            clue_count=3,
            fallback_keywords="京都 紅葉",
        )

        assert result == ["京都 紅葉", "嵐山", "東福寺"]

    @pytest.mark.asyncio
    async def test_trims_to_clue_count_when_llm_returns_more(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.clue_generator.get_llm_client",
            lambda: _llm_stub(
                response={"clues": ["a", "b", "c", "d", "e"]}
            ),
        )

        result = await generate_clues(
            "q",
            ["s1"],
            clue_count=2,
            fallback_keywords="kw",
        )

        assert result == ["a", "b"]

    @pytest.mark.asyncio
    async def test_keeps_fewer_clues_when_llm_returns_fewer(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.clue_generator.get_llm_client",
            lambda: _llm_stub(response={"clues": ["only_one"]}),
        )

        result = await generate_clues(
            "q",
            ["s1"],
            clue_count=3,
            fallback_keywords="kw",
        )

        assert result == ["only_one"]

    @pytest.mark.asyncio
    async def test_passes_summaries_to_prompt(self, monkeypatch):
        client = _llm_stub(response={"clues": ["x"]})
        monkeypatch.setattr(
            "app.rag.clue_generator.get_llm_client", lambda: client
        )

        await generate_clues(
            "京都の紅葉",
            ["秋の嵐山", "東福寺の紅葉"],
            clue_count=2,
            fallback_keywords="kw",
        )

        # The user prompt must wrap the question in <user_question> and
        # the summaries in <candidate_summaries> with index markers.
        kwargs = client.generate_json.await_args.kwargs
        # The user prompt is the second positional argument by signature.
        args = client.generate_json.await_args.args
        user_prompt = args[1] if len(args) >= 2 else kwargs.get("user_prompt", "")
        assert "<user_question>" in user_prompt
        assert "京都の紅葉" in user_prompt
        assert "<candidate_summaries>" in user_prompt
        assert "[1]" in user_prompt and "秋の嵐山" in user_prompt
        assert "[2]" in user_prompt and "東福寺の紅葉" in user_prompt


class TestGenerateCluesFallbacks:
    """Every documented failure mode collapses to [fallback_keywords]."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_fallback(self, monkeypatch):
        result = await generate_clues(
            "   ", ["s1"], clue_count=3, fallback_keywords="kw"
        )
        assert result == ["kw"]

    @pytest.mark.asyncio
    async def test_zero_clue_count_returns_fallback(self, monkeypatch):
        result = await generate_clues(
            "q", ["s1"], clue_count=0, fallback_keywords="kw"
        )
        assert result == ["kw"]

    @pytest.mark.asyncio
    async def test_no_summaries_returns_fallback(self, monkeypatch):
        # No LLM call is needed — short-circuits on empty summaries.
        spy = MagicMock()
        monkeypatch.setattr(
            "app.rag.clue_generator.get_llm_client", lambda: spy
        )

        result = await generate_clues(
            "q", [], clue_count=3, fallback_keywords="kw"
        )

        assert result == ["kw"]
        spy.generate_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_blank_summaries_returns_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.clue_generator.get_llm_client",
            lambda: _llm_stub(response={"clues": ["x"]}),
        )

        result = await generate_clues(
            "q", ["", "   "], clue_count=3, fallback_keywords="kw"
        )

        assert result == ["kw"]

    @pytest.mark.asyncio
    async def test_llm_disabled_returns_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.clue_generator.get_llm_client",
            lambda: _llm_stub(enabled=False),
        )

        result = await generate_clues(
            "q", ["s1"], clue_count=3, fallback_keywords="kw"
        )

        assert result == ["kw"]

    @pytest.mark.asyncio
    async def test_llm_client_unavailable_returns_fallback(self, monkeypatch):
        def _raise():
            raise RuntimeError("not configured")

        monkeypatch.setattr(
            "app.rag.clue_generator.get_llm_client", _raise
        )

        result = await generate_clues(
            "q", ["s1"], clue_count=3, fallback_keywords="kw"
        )

        assert result == ["kw"]

    @pytest.mark.asyncio
    async def test_non_dict_response_returns_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.clue_generator.get_llm_client",
            lambda: _llm_stub(response=["a", "b"]),
        )

        result = await generate_clues(
            "q", ["s1"], clue_count=3, fallback_keywords="kw"
        )

        assert result == ["kw"]

    @pytest.mark.asyncio
    async def test_missing_clues_key_returns_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.clue_generator.get_llm_client",
            lambda: _llm_stub(response={"keywords": "x"}),
        )

        result = await generate_clues(
            "q", ["s1"], clue_count=3, fallback_keywords="kw"
        )

        assert result == ["kw"]

    @pytest.mark.asyncio
    async def test_clues_not_a_list_returns_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.clue_generator.get_llm_client",
            lambda: _llm_stub(response={"clues": "single string"}),
        )

        result = await generate_clues(
            "q", ["s1"], clue_count=3, fallback_keywords="kw"
        )

        assert result == ["kw"]

    @pytest.mark.asyncio
    async def test_empty_clues_array_returns_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.clue_generator.get_llm_client",
            lambda: _llm_stub(response={"clues": []}),
        )

        result = await generate_clues(
            "q", ["s1"], clue_count=3, fallback_keywords="kw"
        )

        assert result == ["kw"]

    @pytest.mark.asyncio
    async def test_all_clues_filtered_to_empty_returns_fallback(
        self, monkeypatch
    ):
        # filter_keywords drops the whole string for noise-only entries.
        # Stub it so this test does not depend on the exact blocklist.
        monkeypatch.setattr(
            "app.rag.clue_generator.filter_keywords", lambda _s: ""
        )
        monkeypatch.setattr(
            "app.rag.clue_generator.get_llm_client",
            lambda: _llm_stub(response={"clues": ["何", "なぜ"]}),
        )

        result = await generate_clues(
            "q", ["s1"], clue_count=3, fallback_keywords="kw"
        )

        assert result == ["kw"]

    @pytest.mark.asyncio
    async def test_partial_filtering_keeps_surviving_clues(self, monkeypatch):
        # filter_keywords drops the second entry entirely; the others
        # survive. The fallback path should NOT trigger.
        def _filter(s: str) -> str:
            if "noise" in s:
                return ""
            return s

        monkeypatch.setattr(
            "app.rag.clue_generator.filter_keywords", _filter
        )
        monkeypatch.setattr(
            "app.rag.clue_generator.get_llm_client",
            lambda: _llm_stub(
                response={"clues": ["good_a", "noise here", "good_b"]}
            ),
        )

        result = await generate_clues(
            "q", ["s1"], clue_count=3, fallback_keywords="kw"
        )

        assert result == ["good_a", "good_b"]


# ---------------------------------------------------------------------------
# fetch_long_summaries
# ---------------------------------------------------------------------------


def _patch_db(monkeypatch, *, rows):
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = rows

    @contextmanager
    def _get_search_db():
        yield session

    monkeypatch.setattr(cg_mod, "get_search_db", _get_search_db)
    return session


class TestFetchLongSummaries:
    def test_returns_mapping_for_present_files(self, monkeypatch):
        # Three-column row shape: (file_id, long_summary, visual_description).
        # File ``a`` has a text summary; ``b`` has a visual description;
        # both should land in the unified output map.
        rows = [
            ("a", "summary A", None),
            ("b", None, "visual desc B"),
        ]
        _patch_db(monkeypatch, rows=rows)

        result = fetch_long_summaries(["a", "b", "c"])

        assert result == {"a": "summary A", "b": "visual desc B"}

    def test_prefers_long_summary_when_both_present(self, monkeypatch):
        # If both columns are populated for the same file (a future
        # schema drift could allow this), the text summary wins because
        # it's richer source material for clue-generation prompting.
        rows = [("a", "long text summary", "fallback visual")]
        _patch_db(monkeypatch, rows=rows)

        result = fetch_long_summaries(["a"])

        assert result == {"a": "long text summary"}

    def test_empty_input_returns_empty_dict(self, monkeypatch):
        # No DB call — fast path.
        @contextmanager
        def _should_not_be_called():
            raise AssertionError("get_search_db must not be invoked")
            yield  # pragma: no cover

        monkeypatch.setattr(cg_mod, "get_search_db", _should_not_be_called)

        result = fetch_long_summaries([])

        assert result == {}

    def test_drops_blank_summary_rows(self, monkeypatch):
        rows = [("a", "  ", None), ("b", "real", None)]
        _patch_db(monkeypatch, rows=rows)

        result = fetch_long_summaries(["a", "b"])

        # Whitespace-only summaries are skipped — they would only feed
        # the LLM noise and trigger the fallback path anyway.
        assert "a" not in result
        assert result["b"] == "real"

    def test_uses_named_placeholders_for_in_clause(self, monkeypatch):
        rows = [("a", "s1", None)]
        session = _patch_db(monkeypatch, rows=rows)

        fetch_long_summaries(["a", "b", "c"])

        # The bound parameters must include one per id so the IN list
        # cannot turn into a SQL injection vector via raw f-string.
        call = session.execute.call_args
        params = call.args[1] if len(call.args) >= 2 else call.kwargs
        assert params == {"id0": "a", "id1": "b", "id2": "c"}
