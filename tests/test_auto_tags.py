"""Tests for app.workers.auto_tags module.

Covers pure functions: file type classification, system prompt
construction with language variants, user prompt building, and
tag filtering logic.
"""

from unittest.mock import MagicMock

import pytest

from app.config import LLMConfig, Settings
from app.workers import auto_tags as at_module
from app.workers.auto_tags import (
    TagCandidates,
    _build_context,
    _build_system_prompt,
    _build_user_prompt,
    _classify_file_type,
    _filter_tags,
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

        assert "Japanese" in result

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

        assert "Filename: video.mp4" in result
        assert "Type: video" in result
        assert "Existing tags: existing-tag" in result

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

        assert "Title: My Great Video" in result

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

        assert "Title:" not in result

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

        assert "Description: A test description" in result

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

        assert "Metadata tags: sunset, beach" in result

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

        assert "Existing tags: none" in result

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

        assert "Existing tags: tag1, tag2, tag3" in result


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


# ---------------------------------------------------------------------------
# TagCandidates.merged: CLIP + TF-IDF dedupe & cap
# ---------------------------------------------------------------------------


class TestTagCandidatesMerged:
    """Tests for the local-only (no LLM) candidate merge logic."""

    def test_empty_returns_empty(self):
        c = TagCandidates(clip=[], tfidf=[])
        assert c.merged(10) == []

    def test_clip_before_tfidf(self):
        c = TagCandidates(clip=["料理"], tfidf=["パスタ"])
        assert c.merged(10) == ["料理", "パスタ"]

    def test_dedupe_case_insensitive(self):
        c = TagCandidates(clip=["Music"], tfidf=["MUSIC", "Rock"])
        assert c.merged(10) == ["Music", "Rock"]

    def test_limit_caps_output(self):
        c = TagCandidates(
            clip=["a", "b", "c"],
            tfidf=["d", "e", "f"],
        )
        assert c.merged(4) == ["a", "b", "c", "d"]

    def test_has_any_false_when_both_empty(self):
        assert TagCandidates(clip=[], tfidf=[]).has_any() is False

    def test_has_any_true_when_clip_only(self):
        assert TagCandidates(clip=["x"], tfidf=[]).has_any() is True

    def test_has_any_true_when_tfidf_only(self):
        assert TagCandidates(clip=[], tfidf=["x"]).has_any() is True

    def test_has_any_true_when_knn_only(self):
        assert TagCandidates(clip=[], tfidf=[], knn=["x"]).has_any() is True

    def test_knn_priority_over_clip_and_tfidf(self):
        c = TagCandidates(
            clip=["clip_a"], tfidf=["tfidf_a"], knn=["knn_a"]
        )
        assert c.merged(3) == ["knn_a", "clip_a", "tfidf_a"]

    def test_knn_dedupe_removes_duplicates_from_other_sources(self):
        c = TagCandidates(
            clip=["料理"], tfidf=["料理"], knn=["料理", "和食"]
        )
        assert c.merged(10) == ["料理", "和食"]


# ---------------------------------------------------------------------------
# _filter_tags (module function, covers the dedupe + trim + case path)
# ---------------------------------------------------------------------------


class TestFilterTagsModule:
    """Tests for the module-level _filter_tags (not the test class variant)."""

    def test_deduplicates_within_list(self):
        result = _filter_tags(["料理", "RYORI", "料理"], [])
        # Case-insensitive dedupe preserves first occurrence
        assert result == ["料理", "RYORI"]

    def test_trims_whitespace(self):
        result = _filter_tags(["  料理  ", "\tパスタ\n"], [])
        assert result == ["料理", "パスタ"]

    def test_filters_against_existing(self):
        result = _filter_tags(["料理", "パスタ"], ["料理"])
        assert result == ["パスタ"]

    def test_skips_non_strings(self):
        result = _filter_tags(["料理", 42, None, "パスタ"], [])
        assert result == ["料理", "パスタ"]


# ---------------------------------------------------------------------------
# _build_user_prompt with candidates (LLM grounding mode)
# ---------------------------------------------------------------------------


class TestBuildUserPromptWithCandidates:
    """Tests for the LLM grounding hint injection in the user prompt."""

    _BASE_FILE = {
        "file_id": "abc",
        "filename": "video.mp4",
        "file_type": "video",
        "mime_type": "video/mp4",
        "title": None,
        "description": None,
        "tags_text": None,
    }

    def test_omits_candidate_section_when_both_empty(self):
        candidates = TagCandidates(clip=[], tfidf=[])
        result = _build_user_prompt(
            self._BASE_FILE, "video", "", [], candidates
        )
        assert "Reference candidates" not in result

    def test_candidate_section_shown_when_clip_present(self):
        candidates = TagCandidates(clip=["料理", "屋内"], tfidf=[])
        result = _build_user_prompt(
            self._BASE_FILE, "video", "", [], candidates
        )
        assert "Reference candidates" in result
        assert "Image analysis candidates: 料理, 屋内" in result
        assert "Keyword extraction candidates" not in result

    def test_candidate_section_shown_when_tfidf_present(self):
        candidates = TagCandidates(clip=[], tfidf=["パスタ", "トマト"])
        result = _build_user_prompt(
            self._BASE_FILE, "video", "", [], candidates
        )
        assert "Keyword extraction candidates: パスタ, トマト" in result
        assert "Image analysis candidates" not in result

    def test_both_candidate_types_shown(self):
        candidates = TagCandidates(clip=["料理"], tfidf=["パスタ"])
        result = _build_user_prompt(
            self._BASE_FILE, "video", "", [], candidates
        )
        assert "Image analysis candidates: 料理" in result
        assert "Keyword extraction candidates: パスタ" in result

    def test_knn_candidates_shown_when_present(self):
        candidates = TagCandidates(
            clip=[], tfidf=[], knn=["和食", "レシピ"]
        )
        result = _build_user_prompt(
            self._BASE_FILE, "video", "", [], candidates
        )
        assert "Tags from similar files: 和食, レシピ" in result

    def test_override_hint_present(self):
        """The LLM must be told candidates are advisory, not required."""
        candidates = TagCandidates(clip=["x"], tfidf=[])
        result = _build_user_prompt(
            self._BASE_FILE, "video", "", [], candidates
        )
        assert "reference" in result.lower() or "hint" in result.lower()

    def test_legacy_call_without_candidates_still_works(self):
        """Backwards compatibility: the original 4-arg call still builds."""
        result = _build_user_prompt(
            self._BASE_FILE, "video", "", ["existing-tag"]
        )
        assert "Reference candidates" not in result
        assert "Existing tags: existing-tag" in result


# ---------------------------------------------------------------------------
# AutoTagsWorker._process_file branching: LLM on/off
# ---------------------------------------------------------------------------


class TestProcessFileBranches:
    """Tests for the LLM-on vs LLM-off paths of _process_file.

    Uses monkeypatching to replace the local-candidate generators and
    DB calls with deterministic stubs. The worker's control flow is
    the unit under test — not the candidate pipelines themselves.
    """

    def _install_common_stubs(self, monkeypatch, *, candidates, saved):
        """Install stubs for DB lookups, candidate generation, and save.

        ``saved`` is mutated in place to capture the arguments of
        _save_suggested_tags so the test can assert on them.
        """
        from app.workers import auto_tags as at

        monkeypatch.setattr(at, "_has_suggested_tags", lambda fid: False)
        monkeypatch.setattr(
            at,
            "_get_indexed_file",
            lambda fid: {
                "file_id": fid,
                "drive": "main",
                "filename": "sample.mp4",
                "file_type": "video",
                "mime_type": "video/mp4",
                "title": None,
                "description": None,
                "tags_text": None,
            },
        )
        monkeypatch.setattr(at, "_get_existing_tags", lambda fid: [])
        monkeypatch.setattr(at, "_generate_candidates", lambda *a, **k: candidates)
        monkeypatch.setattr(at, "_build_context", lambda *a, **k: "")

        def fake_save(**kwargs):
            saved.update(kwargs)

        monkeypatch.setattr(at, "_save_suggested_tags", fake_save)

    @pytest.mark.asyncio
    async def test_llm_disabled_uses_local_candidates(
        self, monkeypatch, make_settings
    ):
        from app.config import FeaturesConfig
        from app.workers import auto_tags as at

        settings = make_settings(
            features=FeaturesConfig(auto_tags="on_index"),
        )
        monkeypatch.setattr(at, "settings", settings)

        llm = MagicMock(spec=[])
        llm.enabled = False

        saved: dict = {}
        candidates = TagCandidates(clip=["料理"], tfidf=["パスタ"])
        self._install_common_stubs(monkeypatch, candidates=candidates, saved=saved)

        worker = at.AutoTagsWorker(llm)
        await worker._process_file("file-1")

        assert saved["tags"] == ["料理", "パスタ"]
        assert saved["model"] == "clip+tfidf"

    @pytest.mark.asyncio
    async def test_llm_enabled_uses_llm_output(
        self, monkeypatch, make_settings
    ):
        from app.config import FeaturesConfig
        from app.workers import auto_tags as at

        settings = make_settings(
            llm=LLMConfig(provider="openai_compatible", model="test-llm"),
            features=FeaturesConfig(auto_tags="on_index"),
        )
        monkeypatch.setattr(at, "settings", settings)

        llm = MagicMock()
        llm.enabled = True

        async def fake_generate_json(system, user):
            # Ensure candidates were passed to the LLM
            assert "Reference candidates" in user
            assert "料理" in user
            return ["料理失敗談", "パスタ料理"]

        llm.generate_json = fake_generate_json

        saved: dict = {}
        candidates = TagCandidates(clip=["料理"], tfidf=["パスタ"])
        self._install_common_stubs(monkeypatch, candidates=candidates, saved=saved)

        worker = at.AutoTagsWorker(llm)
        await worker._process_file("file-1")

        assert saved["tags"] == ["料理失敗談", "パスタ料理"]
        assert saved["model"] == "clip+tfidf+test-llm"

    @pytest.mark.asyncio
    async def test_llm_non_list_response_falls_back_to_local(
        self, monkeypatch, make_settings
    ):
        from app.config import FeaturesConfig
        from app.workers import auto_tags as at

        settings = make_settings(
            llm=LLMConfig(provider="openai_compatible", model="bad-llm"),
            features=FeaturesConfig(auto_tags="on_index"),
        )
        monkeypatch.setattr(at, "settings", settings)

        llm = MagicMock()
        llm.enabled = True

        async def fake_generate_json(system, user):
            return {"oops": "not a list"}

        llm.generate_json = fake_generate_json

        saved: dict = {}
        candidates = TagCandidates(clip=["料理"], tfidf=["パスタ"])
        self._install_common_stubs(monkeypatch, candidates=candidates, saved=saved)

        worker = at.AutoTagsWorker(llm)
        await worker._process_file("file-1")

        # Fallback uses local candidates but keeps the llm-tagged model label
        assert saved["tags"] == ["料理", "パスタ"]
        assert "bad-llm" in saved["model"]

    @pytest.mark.asyncio
    async def test_llm_dict_with_tags_key_is_unwrapped(
        self, monkeypatch, make_settings
    ):
        """When the LLM returns {"tags": [...]} (json_object mode),
        the worker should unwrap the tags list rather than fall back."""
        from app.config import FeaturesConfig
        from app.workers import auto_tags as at

        settings = make_settings(
            llm=LLMConfig(provider="openai_compatible", model="test-llm"),
            features=FeaturesConfig(auto_tags="on_index"),
        )
        monkeypatch.setattr(at, "settings", settings)

        llm = MagicMock()
        llm.enabled = True

        async def fake_generate_json(system, user):
            return {"tags": ["料理失敗談", "パスタ料理"]}

        llm.generate_json = fake_generate_json

        saved: dict = {}
        candidates = TagCandidates(clip=["料理"], tfidf=["パスタ"])
        self._install_common_stubs(monkeypatch, candidates=candidates, saved=saved)

        worker = at.AutoTagsWorker(llm)
        await worker._process_file("file-1")

        assert saved["tags"] == ["料理失敗談", "パスタ料理"]
        assert saved["model"] == "clip+tfidf+test-llm"

    @pytest.mark.asyncio
    async def test_no_candidates_no_save(
        self, monkeypatch, make_settings
    ):
        from app.config import FeaturesConfig
        from app.workers import auto_tags as at

        settings = make_settings(
            features=FeaturesConfig(auto_tags="on_index"),
        )
        monkeypatch.setattr(at, "settings", settings)

        llm = MagicMock(spec=[])
        llm.enabled = False

        saved: dict = {}
        self._install_common_stubs(
            monkeypatch,
            candidates=TagCandidates(clip=[], tfidf=[]),
            saved=saved,
        )

        worker = at.AutoTagsWorker(llm)
        await worker._process_file("file-1")

        # No local candidates → no save attempted
        assert saved == {}

    @pytest.mark.asyncio
    async def test_feature_disabled_skips(
        self, monkeypatch, make_settings
    ):
        from app.config import FeaturesConfig
        from app.workers import auto_tags as at

        settings = make_settings(
            features=FeaturesConfig(auto_tags="false"),
        )
        monkeypatch.setattr(at, "settings", settings)

        llm = MagicMock(spec=[])
        llm.enabled = False

        saved: dict = {}
        candidates = TagCandidates(clip=["料理"], tfidf=[])
        self._install_common_stubs(monkeypatch, candidates=candidates, saved=saved)

        worker = at.AutoTagsWorker(llm)
        await worker._process_file("file-1")

        assert saved == {}


# ---------------------------------------------------------------------------
# _build_context — vision_description integration (spec 2026-04-23)
# ---------------------------------------------------------------------------


class TestBuildContextVisionDescription:
    """Image files should get BLIP captions AND vision_description.

    Phase 1.5 follow-up to the vision_describe spec: feeding the richer
    vision LLM output alongside the terse BLIP caption gives auto_tags
    a much stronger context signal for image files. Verifies the
    wiring without touching the real DB.
    """

    def test_image_includes_vision_description_when_present(self, monkeypatch):
        monkeypatch.setattr(
            at_module, "_get_blip_captions", lambda fid: "a cat on a mat"
        )
        monkeypatch.setattr(
            at_module,
            "_get_visual_description",
            lambda fid: "An orange tabby cat resting on a red patterned mat.",
        )
        ctx = _build_context({"file_id": "img-1"}, "image")
        assert "Image captions:\na cat on a mat" in ctx
        assert "Visual description:\nAn orange tabby cat" in ctx

    def test_image_omits_vision_section_when_absent(self, monkeypatch):
        monkeypatch.setattr(
            at_module, "_get_blip_captions", lambda fid: "a cat on a mat"
        )
        monkeypatch.setattr(
            at_module, "_get_visual_description", lambda fid: ""
        )
        ctx = _build_context({"file_id": "img-1"}, "image")
        assert "Image captions:\na cat on a mat" in ctx
        assert "Visual description:" not in ctx

    def test_image_with_only_vision_description(self, monkeypatch):
        """BLIP missing but vision_description present — still usable."""
        monkeypatch.setattr(at_module, "_get_blip_captions", lambda fid: "")
        monkeypatch.setattr(
            at_module,
            "_get_visual_description",
            lambda fid: "A bowl of yellow rice with grilled meat.",
        )
        ctx = _build_context({"file_id": "img-1"}, "image")
        assert "Image captions:" not in ctx
        assert "Visual description:\nA bowl of yellow rice" in ctx

    def test_video_does_not_pull_vision_description(self, monkeypatch):
        """Non-image types must not touch the vision helpers (Phase 1 scope)."""
        monkeypatch.setattr(
            at_module, "_get_transcript_text", lambda fid: "some narration"
        )
        # If the wiring is wrong these would raise; stub them to fail loudly.
        monkeypatch.setattr(
            at_module,
            "_get_visual_description",
            lambda fid: pytest.fail(
                "vision_description should not be fetched for video"
            ),
        )
        ctx = _build_context({"file_id": "vid-1"}, "video")
        assert "Transcript:\nsome narration" in ctx
        assert "Visual description:" not in ctx


class TestGetVisualDescription:
    """Unit tests for the `_get_visual_description` DB helper.

    Uses monkeypatched SQLAlchemy session rather than a real sqlite
    file — the logic under test is just a SELECT + truncate.
    """

    def _patch_session(self, monkeypatch, row):
        """Install a stub `get_search_db` that yields a fake session."""
        fake_result = MagicMock()
        fake_result.fetchone.return_value = row
        fake_session = MagicMock()
        fake_session.execute.return_value = fake_result

        class _Ctx:
            def __enter__(self_inner):
                return fake_session

            def __exit__(self_inner, *a):
                return False

        monkeypatch.setattr(at_module, "get_search_db", lambda: _Ctx())
        return fake_session

    def test_returns_text_for_success_row(self, monkeypatch):
        from app.workers.auto_tags import _get_visual_description

        self._patch_session(monkeypatch, ("A beautiful mountain landscape.",))
        assert _get_visual_description("img-1") == "A beautiful mountain landscape."

    def test_returns_empty_when_no_row(self, monkeypatch):
        from app.workers.auto_tags import _get_visual_description

        self._patch_session(monkeypatch, None)
        assert _get_visual_description("img-1") == ""

    def test_returns_empty_when_text_is_null(self, monkeypatch):
        from app.workers.auto_tags import _get_visual_description

        self._patch_session(monkeypatch, (None,))
        assert _get_visual_description("img-1") == ""

    def test_truncates_to_max_context_chars(self, monkeypatch):
        from app.workers.auto_tags import _MAX_CONTEXT_CHARS, _get_visual_description

        long_text = "x" * (_MAX_CONTEXT_CHARS + 500)
        self._patch_session(monkeypatch, (long_text,))
        result = _get_visual_description("img-1")
        assert len(result) == _MAX_CONTEXT_CHARS

    def test_filters_on_success_status(self, monkeypatch):
        """The SELECT must only match `visual_description_status = 'success'`.

        Failed / pending / unsupported rows contain no useful text
        (or stale text from an earlier run); auto_tags should treat them
        as "no context" rather than injecting noise into the prompt.
        """
        from app.workers.auto_tags import _get_visual_description

        fake_session = self._patch_session(monkeypatch, ("ok",))
        _get_visual_description("img-1")
        call_args = fake_session.execute.call_args
        sql_arg = call_args[0][0]
        # sqlalchemy.text() wraps the string; stringify to match.
        assert "visual_description_status = 'success'" in str(sql_arg)
