"""Tests for app.workers.auto_tags module.

Covers pure functions: file type classification, system prompt
construction with language variants, user prompt building, and
tag filtering logic.
"""

import pytest

from app.config import LLMConfig, Settings
from app.workers.auto_tags import (
    _build_system_prompt,
    _build_user_prompt,
    _classify_file_type,
)


# ---------------------------------------------------------------------------
# _classify_file_type
# ---------------------------------------------------------------------------


class TestClassifyFileType:
    """Tests for _classify_file_type: maps file_type string to category."""

    def test_video(self):
        assert _classify_file_type("video") == "video"

    def test_audio(self):
        assert _classify_file_type("audio") == "audio"

    def test_image(self):
        assert _classify_file_type("image") == "image"

    def test_document(self):
        assert _classify_file_type("document") == "document"

    def test_text_maps_to_document(self):
        assert _classify_file_type("text") == "document"

    def test_archive_maps_to_other(self):
        assert _classify_file_type("archive") == "other"

    def test_unknown_type_maps_to_other(self):
        assert _classify_file_type("spreadsheet") == "other"

    def test_empty_string_maps_to_other(self):
        assert _classify_file_type("") == "other"


# ---------------------------------------------------------------------------
# _build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Tests for _build_system_prompt: language-specific instructions."""

    def test_japanese_language_instruction(self, monkeypatch, make_settings):
        settings = make_settings(
            llm=LLMConfig(output_language="ja")
        )
        monkeypatch.setattr("app.config.settings", settings)
        # Also patch the module-level reference in auto_tags
        monkeypatch.setattr("app.workers.auto_tags.settings", settings)

        result = _build_system_prompt()

        assert "日本語で生成" in result

    def test_english_language_instruction(self, monkeypatch, make_settings):
        settings = make_settings(
            llm=LLMConfig(output_language="en")
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.auto_tags.settings", settings)

        result = _build_system_prompt()

        assert "English" in result

    def test_auto_language_no_specific_instruction(self, monkeypatch, make_settings):
        settings = make_settings(
            llm=LLMConfig(output_language="auto")
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.auto_tags.settings", settings)

        result = _build_system_prompt()

        assert "日本語で生成" not in result
        assert "English" not in result

    def test_unknown_language_no_specific_instruction(self, monkeypatch, make_settings):
        settings = make_settings(
            llm=LLMConfig(output_language="fr")
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.auto_tags.settings", settings)

        result = _build_system_prompt()

        # "fr" is not in _LANGUAGE_INSTRUCTIONS, so no language line
        assert "日本語で生成" not in result
        assert "English" not in result

    def test_prompt_contains_json_instruction(self, monkeypatch, make_settings):
        settings = make_settings(llm=LLMConfig(output_language="auto"))
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.auto_tags.settings", settings)

        result = _build_system_prompt()

        assert "JSON" in result


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    """Tests for _build_user_prompt: assembles file info into prompt."""

    def test_basic_prompt_with_filename_and_type(self):
        indexed_file = {
            "file_id": "abc123",
            "filename": "video.mp4",
            "file_type": "video",
            "mime_type": "video/mp4",
            "title": None,
            "description": None,
            "tags_text": None,
        }

        result = _build_user_prompt(
            indexed_file, "video", "", ["existing-tag"]
        )

        assert "ファイル名: video.mp4" in result
        assert "タイプ: video" in result
        assert "既存タグ: existing-tag" in result

    def test_includes_title_when_different_from_filename(self):
        indexed_file = {
            "file_id": "abc123",
            "filename": "video.mp4",
            "file_type": "video",
            "mime_type": "video/mp4",
            "title": "My Great Video",
            "description": None,
            "tags_text": None,
        }

        result = _build_user_prompt(indexed_file, "video", "", [])

        assert "タイトル: My Great Video" in result

    def test_excludes_title_when_same_as_filename(self):
        indexed_file = {
            "file_id": "abc123",
            "filename": "video.mp4",
            "file_type": "video",
            "mime_type": "video/mp4",
            "title": "video.mp4",
            "description": None,
            "tags_text": None,
        }

        result = _build_user_prompt(indexed_file, "video", "", [])

        assert "タイトル:" not in result

    def test_includes_description(self):
        indexed_file = {
            "file_id": "abc123",
            "filename": "video.mp4",
            "file_type": "video",
            "mime_type": "video/mp4",
            "title": None,
            "description": "A test description",
            "tags_text": None,
        }

        result = _build_user_prompt(indexed_file, "video", "", [])

        assert "説明: A test description" in result

    def test_includes_tags_text(self):
        indexed_file = {
            "file_id": "abc123",
            "filename": "photo.jpg",
            "file_type": "image",
            "mime_type": "image/jpeg",
            "title": None,
            "description": None,
            "tags_text": "sunset, beach",
        }

        result = _build_user_prompt(indexed_file, "image", "", [])

        assert "メタデータタグ: sunset, beach" in result

    def test_includes_context_string(self):
        indexed_file = {
            "file_id": "abc123",
            "filename": "lecture.mp4",
            "file_type": "video",
            "mime_type": "video/mp4",
            "title": None,
            "description": None,
            "tags_text": None,
        }
        context = "Transcript:\nHello and welcome to today's lecture."

        result = _build_user_prompt(indexed_file, "video", context, [])

        assert "Transcript:" in result
        assert "today's lecture" in result

    def test_no_existing_tags_shows_nashi(self):
        indexed_file = {
            "file_id": "abc123",
            "filename": "file.txt",
            "file_type": "text",
            "mime_type": "text/plain",
            "title": None,
            "description": None,
            "tags_text": None,
        }

        result = _build_user_prompt(indexed_file, "document", "", [])

        assert "既存タグ: なし" in result

    def test_multiple_existing_tags_comma_separated(self):
        indexed_file = {
            "file_id": "abc123",
            "filename": "file.txt",
            "file_type": "text",
            "mime_type": "text/plain",
            "title": None,
            "description": None,
            "tags_text": None,
        }

        result = _build_user_prompt(
            indexed_file, "document", "", ["tag1", "tag2", "tag3"]
        )

        assert "既存タグ: tag1, tag2, tag3" in result


# ---------------------------------------------------------------------------
# Tag filtering logic (from _process_file)
# ---------------------------------------------------------------------------


class TestTagFilteringLogic:
    """Tests for the tag filtering comprehension used in _process_file.

    Extracted logic:
        existing_lower = {t.lower() for t in existing_tags}
        filtered = [
            t for t in tags
            if isinstance(t, str) and t.strip() and t.lower() not in existing_lower
        ]
    """

    @staticmethod
    def _filter_tags(tags: list, existing_tags: list[str]) -> list[str]:
        """Replicate the filtering logic from _process_file."""
        existing_lower = {t.lower() for t in existing_tags}
        return [
            t for t in tags
            if isinstance(t, str) and t.strip() and t.lower() not in existing_lower
        ]

    def test_removes_case_insensitive_duplicates(self):
        tags = ["Music", "rock", "Jazz"]
        existing = ["music", "JAZZ"]

        result = self._filter_tags(tags, existing)

        assert result == ["rock"]

    def test_removes_non_string_values(self):
        tags = ["valid", 42, None, True, ["nested"], "also-valid"]
        existing: list[str] = []

        result = self._filter_tags(tags, existing)

        assert result == ["valid", "also-valid"]

    def test_removes_empty_strings(self):
        tags = ["valid", "", "  ", "\t", "also-valid"]
        existing: list[str] = []

        result = self._filter_tags(tags, existing)

        assert result == ["valid", "also-valid"]

    def test_keeps_all_when_no_existing(self):
        tags = ["tag1", "tag2", "tag3"]
        existing: list[str] = []

        result = self._filter_tags(tags, existing)

        assert result == ["tag1", "tag2", "tag3"]

    def test_empty_tags_list(self):
        tags: list = []
        existing = ["something"]

        result = self._filter_tags(tags, existing)

        assert result == []

    def test_all_filtered_out(self):
        tags = ["Music", "ROCK"]
        existing = ["music", "rock"]

        result = self._filter_tags(tags, existing)

        assert result == []

    def test_preserves_original_casing(self):
        tags = ["Machine Learning", "deep-learning"]
        existing: list[str] = []

        result = self._filter_tags(tags, existing)

        assert result == ["Machine Learning", "deep-learning"]
