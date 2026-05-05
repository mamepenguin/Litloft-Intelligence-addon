"""Snapshot regression tests for the Jinja2 prompt migration.

These tests assert byte-identical output between the migrated prompt
builders and golden files captured pre-migration (from the original
f-string implementation). If you intentionally change a prompt, the
golden files must be regenerated alongside the change so the diff is
visible on the PR.

The loader-level tests (basic render, StrictUndefined) live here too
because there is no other home for them and they are tiny.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jinja2 import UndefinedError

from app.prompt_loader import render


GOLDEN_DIR = Path(__file__).resolve().parent / "golden_prompts"


def _golden(name: str) -> str:
    return (GOLDEN_DIR / name).read_bytes().decode("utf-8")


# ---------------------------------------------------------------------------
# Loader-level tests
# ---------------------------------------------------------------------------


def test_render_returns_string() -> None:
    out = render("auto_tags/system.jinja2", language_instruction="")
    assert isinstance(out, str)
    assert "tagging assistant" in out


def test_render_strict_undefined_raises_on_missing_var() -> None:
    with pytest.raises(UndefinedError):
        # auto_tags/system.jinja2 expects ``language_instruction``.
        render("auto_tags/system.jinja2")


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


@pytest.fixture
def _set_lang(monkeypatch):
    """Helper to flip ``settings.llm.output_language`` for one test."""

    def _apply(lang: str) -> None:
        from app.config import settings as real_settings

        # Settings dataclasses are frozen; bypass via object.__setattr__.
        object.__setattr__(real_settings.llm, "output_language", lang)

    return _apply


@pytest.mark.parametrize("lang", ["ja", "en", "auto"])
def test_summaries_short_long_system_matches_golden(_set_lang, lang) -> None:
    from app.workers import summaries

    _set_lang(lang)
    assert summaries._build_system_prompt() == _golden(
        f"summaries_short_long_system_{lang}.txt"
    )


@pytest.mark.parametrize("lang", ["ja", "en", "auto"])
def test_summaries_detailed_system_matches_golden(_set_lang, lang) -> None:
    from app.workers import summaries

    _set_lang(lang)
    assert summaries._build_detailed_system_prompt() == _golden(
        f"summaries_detailed_system_{lang}.txt"
    )


def test_summaries_short_long_user_minimal_matches_golden() -> None:
    from app.workers import summaries

    indexed = {"filename": "test.mp4", "title": "", "description": ""}
    out = summaries._build_user_prompt(indexed, "video", "ここに本文。", was_truncated=False)
    assert out == _golden("summaries_short_long_user_minimal.txt")


def test_summaries_short_long_user_full_matches_golden() -> None:
    from app.workers import summaries

    indexed = {
        "filename": "test.mp4",
        "title": "別のタイトル",
        "description": "テスト動画の説明文。",
    }
    out = summaries._build_user_prompt(indexed, "video", "本文ここ。", was_truncated=True)
    assert out == _golden("summaries_short_long_user_full.txt")


def test_summaries_short_long_user_title_eq_filename_matches_golden() -> None:
    from app.workers import summaries

    indexed = {"filename": "same.mp4", "title": "same.mp4", "description": ""}
    out = summaries._build_user_prompt(indexed, "video", "ctx", was_truncated=False)
    assert out == _golden("summaries_short_long_user_title_eq_filename.txt")


def test_summaries_detailed_user_minimal_matches_golden() -> None:
    from app.workers import summaries

    indexed = {"filename": "test.mp4", "title": "", "description": ""}
    out = summaries._build_detailed_user_prompt(indexed, "video", "本文。", was_truncated=False)
    assert out == _golden("summaries_detailed_user_minimal.txt")


def test_summaries_detailed_user_full_matches_golden() -> None:
    from app.workers import summaries

    indexed = {
        "filename": "test.mp4",
        "title": "別のタイトル",
        "description": "テスト動画の説明文。",
    }
    out = summaries._build_detailed_user_prompt(indexed, "video", "本文。", was_truncated=True)
    assert out == _golden("summaries_detailed_user_full.txt")


# ---------------------------------------------------------------------------
# Auto tags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lang", ["ja", "en", "auto"])
def test_auto_tags_system_matches_golden(_set_lang, lang) -> None:
    from app.workers import auto_tags

    _set_lang(lang)
    assert auto_tags._build_system_prompt() == _golden(f"auto_tags_system_{lang}.txt")


def test_auto_tags_user_minimal_matches_golden() -> None:
    from app.workers import auto_tags

    indexed = {
        "filename": "f.mp4",
        "title": "",
        "description": "",
        "tags_text": "",
    }
    out = auto_tags._build_user_prompt(indexed, "video", "", [])
    assert out == _golden("auto_tags_user_minimal.txt")


def test_auto_tags_user_rich_matches_golden() -> None:
    from app.workers import auto_tags

    indexed = {
        "filename": "f.mp4",
        "title": "別タイトル",
        "description": "説明文。",
        "tags_text": "tag1, tag2",
    }
    out = auto_tags._build_user_prompt(
        indexed, "video", "本文ここ。", ["existing1", "existing2"]
    )
    assert out == _golden("auto_tags_user_rich.txt")


def test_auto_tags_user_with_candidates_matches_golden() -> None:
    from app.workers import auto_tags

    indexed = {
        "filename": "f.mp4",
        "title": "別タイトル",
        "description": "説明文。",
        "tags_text": "tag1, tag2",
    }
    cands = auto_tags.TagCandidates(
        clip=["clip1", "clip2"], tfidf=["tfidf1"], knn=["knn1"]
    )
    out = auto_tags._build_user_prompt(
        indexed, "video", "本文。", ["existing"], cands
    )
    assert out == _golden("auto_tags_user_with_candidates.txt")


# ---------------------------------------------------------------------------
# Refine
# ---------------------------------------------------------------------------


def test_refine_system_matches_golden() -> None:
    from app.workers import refine

    assert refine._build_system_prompt() == _golden("refine_system.txt")


def test_refine_user_matches_golden() -> None:
    from app.workers import refine

    class _Chunk:
        def __init__(self, cid: int, text: str) -> None:
            self.id = cid
            self.text = text

    out = refine._build_user_prompt([_Chunk(1, "hello"), _Chunk(2, "world")])
    assert out == _golden("refine_user.txt")
    # Sanity: it's still parseable as JSON.
    json.loads(out)


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lang,golden_name",
    [
        ("ja", "vision_system_ja.txt"),
        ("en", "vision_system_en.txt"),
        ("auto", "vision_system_auto.txt"),
        ("", "vision_system_empty.txt"),
    ],
)
def test_vision_system_matches_golden(lang, golden_name) -> None:
    from app import llm

    assert llm._build_vision_system_prompt(lang) == _golden(golden_name)


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lang", ["ja", "en", "auto"])
def test_rag_answer_system_matches_golden(lang) -> None:
    from app.rag import prompt as rag_prompt

    assert rag_prompt.build_system_prompt(lang) == _golden(
        f"rag_answer_system_{lang}.txt"
    )


def test_rag_answer_user_empty_matches_golden() -> None:
    from app.rag import prompt as rag_prompt

    out = rag_prompt.build_user_prompt("クエリだ", [])
    assert out == _golden("rag_answer_user_empty.txt")


def test_rag_answer_user_one_ctx_matches_golden() -> None:
    from app.rag import prompt as rag_prompt
    from app.rag.context import ContextSnippet, FileContext

    ctx = FileContext(
        file_id="abc123",
        filename="video.mp4",
        drive="movies",
        file_type="video",
        title="title",
        description="description",
        snippets=(
            ContextSnippet(source="transcript", location="0:45", text="snippet text"),
            ContextSnippet(source="metadata", location=None, text="meta snippet"),
        ),
        total_chars=42,
    )
    out = rag_prompt.build_user_prompt("クエリ", [ctx])
    assert out == _golden("rag_answer_user_one_ctx.txt")


def test_rag_query_decomposer_system_matches_golden() -> None:
    from app.rag import query_decomposer

    assert query_decomposer._SYSTEM_PROMPT == _golden(
        "rag_query_decomposer_system.txt"
    )


def test_rag_query_transform_system_matches_golden() -> None:
    from app.rag import query_transform

    assert query_transform._SYSTEM_PROMPT == _golden(
        "rag_query_transform_system.txt"
    )


def test_rag_clue_generator_system_matches_golden() -> None:
    out = render("rag/clue_generator_system.jinja2", clue_count=4)
    assert out == _golden("rag_clue_generator_system_4.txt")


def test_rag_category_expander_system_matches_golden() -> None:
    out = render("rag/category_expander_system.jinja2", max_terms=5)
    assert out == _golden("rag_category_expander_system_5.txt")
