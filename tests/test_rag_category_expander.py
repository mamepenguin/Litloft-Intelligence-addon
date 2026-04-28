"""Tests for app.rag.category_expander (Stage C semantic expansion).

Same graceful-degradation contract as ``query_transform`` and
``clue_generator``: any failure path returns the single-element list
``[semantic_query]`` (or ``[]`` for empty input) so the caller's
multi-query loop can iterate without branching on None.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub heavy ML deps before importing the module under test.
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

from app.rag import category_expander  # noqa: E402


def _llm_stub(
    *,
    enabled: bool = True,
    response: dict | list | None = None,
    raises: type[Exception] | None = None,
) -> MagicMock:
    client = MagicMock()
    client.enabled = enabled
    if raises is not None:
        client.generate_json = AsyncMock(side_effect=raises("boom"))
    else:
        client.generate_json = AsyncMock(return_value=response)
    return client


class TestExpandCategoryHappy:
    @pytest.mark.asyncio
    async def test_returns_terms_from_llm(self, monkeypatch):
        monkeypatch.setattr(
            category_expander,
            "get_llm_client",
            lambda: _llm_stub(
                response={
                    "terms": [
                        "SF",
                        "science fiction",
                        "宇宙船",
                        "ロボット",
                        "ディストピア",
                    ]
                }
            ),
        )
        result = await category_expander.expand_category("SF", max_terms=8)
        # Original is preserved when present in the LLM list, plus
        # the bilingual surface forms.
        assert "SF" in result
        assert "science fiction" in result
        assert "宇宙船" in result
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_caps_at_max_terms(self, monkeypatch):
        monkeypatch.setattr(
            category_expander,
            "get_llm_client",
            lambda: _llm_stub(
                response={
                    "terms": ["a", "b", "c", "d", "e", "f", "g"]
                }
            ),
        )
        result = await category_expander.expand_category("category", max_terms=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_dedupes_case_insensitive(self, monkeypatch):
        monkeypatch.setattr(
            category_expander,
            "get_llm_client",
            lambda: _llm_stub(
                response={"terms": ["SF", "sf", "Sf", "宇宙船"]}
            ),
        )
        result = await category_expander.expand_category("SF", max_terms=8)
        # First-seen wins; "SF" / "sf" / "Sf" collapse to one entry.
        lower_set = {t.lower() for t in result}
        assert lower_set == {"sf", "宇宙船"}

    @pytest.mark.asyncio
    async def test_prepends_raw_when_llm_omits_it(self, monkeypatch):
        # LLM returned only related terms but not the raw query itself.
        # The expander prepends it because per-term retrieval would
        # otherwise lose hits that match the original word verbatim.
        monkeypatch.setattr(
            category_expander,
            "get_llm_client",
            lambda: _llm_stub(
                response={"terms": ["science fiction", "宇宙船", "ロボット"]}
            ),
        )
        result = await category_expander.expand_category("SF", max_terms=8)
        assert result[0] == "SF"
        assert "science fiction" in result


class TestExpandCategoryFallback:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self, monkeypatch):
        # Don't even call the LLM for an empty concept — caller skips
        # multi-query entirely.
        monkeypatch.setattr(
            category_expander,
            "get_llm_client",
            lambda: _llm_stub(side_effect=AssertionError),
        )
        result = await category_expander.expand_category("   ", max_terms=8)
        assert result == []

    @pytest.mark.asyncio
    async def test_max_terms_zero_returns_raw(self, monkeypatch):
        monkeypatch.setattr(
            category_expander,
            "get_llm_client",
            lambda: _llm_stub(side_effect=AssertionError),
        )
        result = await category_expander.expand_category("SF", max_terms=0)
        assert result == ["SF"]

    @pytest.mark.asyncio
    async def test_llm_disabled_returns_raw(self, monkeypatch):
        monkeypatch.setattr(
            category_expander,
            "get_llm_client",
            lambda: _llm_stub(enabled=False),
        )
        result = await category_expander.expand_category("SF", max_terms=8)
        assert result == ["SF"]

    @pytest.mark.asyncio
    async def test_llm_runtime_error_returns_raw(self, monkeypatch):
        def _raises():
            raise RuntimeError("not ready")

        monkeypatch.setattr(category_expander, "get_llm_client", _raises)
        result = await category_expander.expand_category("SF", max_terms=8)
        assert result == ["SF"]

    @pytest.mark.asyncio
    async def test_non_dict_response_returns_raw(self, monkeypatch):
        monkeypatch.setattr(
            category_expander,
            "get_llm_client",
            lambda: _llm_stub(response=["not", "a", "dict"]),
        )
        result = await category_expander.expand_category("SF", max_terms=8)
        assert result == ["SF"]

    @pytest.mark.asyncio
    async def test_missing_terms_key_returns_raw(self, monkeypatch):
        monkeypatch.setattr(
            category_expander,
            "get_llm_client",
            lambda: _llm_stub(response={"other": "shape"}),
        )
        result = await category_expander.expand_category("SF", max_terms=8)
        assert result == ["SF"]

    @pytest.mark.asyncio
    async def test_all_terms_filtered_returns_raw(self, monkeypatch):
        # Every term collapses under filter_keywords (e.g. all blanks
        # or all on the noise-words blocklist).
        monkeypatch.setattr(
            category_expander,
            "get_llm_client",
            lambda: _llm_stub(response={"terms": ["", "  ", None]}),
        )
        result = await category_expander.expand_category("SF", max_terms=8)
        assert result == ["SF"]
