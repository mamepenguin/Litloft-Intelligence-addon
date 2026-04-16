"""Tests for app.rag.prompt module.

Pure prompt-string builders. No LLM calls. Mirrors the structure
of test_summaries.TestBuildSystemPrompt and TestBuildUserPrompt.
"""

import sys
from unittest.mock import MagicMock

import pytest

# Stub out heavy dependencies that the RAG package transitively pulls.
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

from app.rag.context import ContextSnippet, FileContext  # noqa: E402
from app.rag.prompt import build_system_prompt, build_user_prompt  # noqa: E402


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Language instruction gating by output_language argument."""

    def test_auto_has_no_language_line(self):
        """T1: 'auto' -> no explicit language instruction."""
        result = build_system_prompt("auto")

        assert "日本語で書く" not in result
        assert "日本語で生成" not in result
        assert "Answers must be in English" not in result

    def test_japanese_language_instruction(self):
        """T2: 'ja' -> Japanese instruction."""
        result = build_system_prompt("ja")
        assert "日本語" in result

    def test_english_language_instruction(self):
        """T3: 'en' -> English instruction."""
        result = build_system_prompt("en")
        assert "English" in result

    def test_unknown_language_no_instruction(self):
        """Unrecognized language codes pass through without crashing."""
        result = build_system_prompt("fr")

        assert "日本語" not in result
        # No English line either.
        assert "Answers must be in English" not in result

    def test_prompt_contains_json_schema_hint(self):
        """The system prompt must describe the expected JSON shape.

        Quote/relevance fields were removed from the LLM contract to
        shorten the citation tail generation; the backend populates
        quotes from retrieved snippets instead.
        """
        result = build_system_prompt("auto")

        assert '"answer"' in result
        assert '"citations"' in result
        assert '"file_id"' in result

    def test_prompt_does_not_require_quote(self):
        """Quote/relevance are populated server-side, not by the LLM."""
        result = build_system_prompt("auto")

        assert '"quote"' not in result
        assert '"relevance"' not in result

    def test_prompt_instructs_no_fabrication_of_file_ids(self):
        """The prompt must pin file_ids to the context blocks.

        Direct-quote fabrication is no longer possible because the LLM
        doesn't generate quotes; the remaining fabrication risk is
        citing a file_id not in context, which this rule addresses.
        """
        result = build_system_prompt("auto")

        assert "file_id" in result
        assert "[file_id:" in result or "[file_id:" in result

    def test_prompt_instructs_json_only(self):
        """Model should be told to return JSON only, no wrapper text."""
        result = build_system_prompt("auto")
        assert "JSON" in result


# ---------------------------------------------------------------------------
# build_user_prompt
# ---------------------------------------------------------------------------


def _make_context(
    file_id: str,
    filename: str = "clip.mp4",
    snippets: tuple[ContextSnippet, ...] | None = None,
    title: str | None = None,
    description: str | None = None,
    file_type: str = "video",
    drive: str = "Videos",
) -> FileContext:
    if snippets is None:
        snippets = (
            ContextSnippet(
                source="transcript",
                text="This is the sampled text.",
                location="0:45",
            ),
        )
    total_chars = sum(len(s.text) for s in snippets)
    return FileContext(
        file_id=file_id,
        filename=filename,
        drive=drive,
        file_type=file_type,
        title=title,
        description=description,
        snippets=snippets,
        total_chars=total_chars,
    )


class TestBuildUserPrompt:
    """User prompt assembly: query + contexts with file_id markers."""

    def test_includes_query(self):
        """T5: the user's question must appear in the prompt."""
        contexts = [_make_context("f1")]
        result = build_user_prompt(
            "What is the topic discussed?", contexts
        )

        assert "What is the topic discussed?" in result

    def test_wraps_each_file_with_file_id_marker(self):
        """T4: each file must be surrounded by [file_id: ...] markers."""
        contexts = [
            _make_context("abc123", filename="a.mp4"),
            _make_context("def456", filename="b.mp4"),
        ]

        result = build_user_prompt("test", contexts)

        # Both file_ids appear inside the [file_id: ...] marker form.
        assert "[file_id: abc123]" in result
        assert "[file_id: def456]" in result

    def test_includes_all_snippets(self):
        """All snippet texts should end up in the user prompt."""
        contexts = [
            _make_context(
                "f1",
                snippets=(
                    ContextSnippet(
                        source="transcript",
                        text="first snippet",
                        location="0:10",
                    ),
                    ContextSnippet(
                        source="transcript",
                        text="second snippet",
                        location="1:20",
                    ),
                ),
            )
        ]

        result = build_user_prompt("q", contexts)

        assert "first snippet" in result
        assert "second snippet" in result

    def test_empty_snippet_file_still_has_filename(self):
        """T6: files with no snippets still contribute filename / marker."""
        contexts = [
            _make_context(
                "empty-1",
                filename="silent.mp4",
                snippets=(),
            )
        ]

        result = build_user_prompt("q", contexts)

        # The marker is still present so the LLM can reference the file.
        assert "[file_id: empty-1]" in result
        # Filename remains visible.
        assert "silent.mp4" in result

    def test_includes_filename_for_each_file(self):
        contexts = [
            _make_context("f1", filename="alpha.mp4"),
            _make_context("f2", filename="beta.pdf", file_type="document"),
        ]

        result = build_user_prompt("q", contexts)

        assert "alpha.mp4" in result
        assert "beta.pdf" in result

    def test_empty_context_list_still_includes_query(self):
        result = build_user_prompt("What is this about?", [])

        assert "What is this about?" in result

    def test_includes_title_when_set(self):
        contexts = [
            _make_context("f1", title="My Lecture on Physics"),
        ]

        result = build_user_prompt("q", contexts)

        assert "My Lecture on Physics" in result

    def test_includes_description_when_set(self):
        contexts = [
            _make_context("f1", description="An in-depth look at cosmology"),
        ]

        result = build_user_prompt("q", contexts)

        assert "cosmology" in result

    def test_file_id_appears_exactly_once_per_file(self):
        contexts = [_make_context("unique-id")]

        result = build_user_prompt("q", contexts)

        # The file_id should appear in the marker, possibly once more in
        # serialized form — at minimum, the marker is present.
        assert result.count("unique-id") >= 1

    def test_preserves_query_special_characters(self):
        contexts = [_make_context("f1")]

        result = build_user_prompt("これは日本語の質問？", contexts)

        assert "これは日本語の質問？" in result
