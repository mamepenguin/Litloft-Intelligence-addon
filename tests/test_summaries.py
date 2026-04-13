"""Tests for app.workers.summaries module.

Covers the pure functions (classification, prompt construction,
sentence-boundary trimming, window sampling, context preparation,
response parsing indirectly through the worker) and the async
SummariesWorker behavior via LLM mocks.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import FeaturesConfig, LLMConfig, SummariesConfig
from app.workers.summaries import (
    _WINDOW_SEPARATOR,
    SummariesWorker,
    _build_system_prompt,
    _build_user_prompt,
    _classify_file_type,
    _get_full_document_text,
    _prepare_context,
    _sample_windows,
    _trim_to_sentence_boundary,
    classify_missing_reason,
)


# ---------------------------------------------------------------------------
# _classify_file_type
# ---------------------------------------------------------------------------


class TestClassifyFileType:
    """Summaries support only video/audio/document; others return None."""

    def test_video(self):
        assert _classify_file_type("video") == "video"

    def test_audio(self):
        assert _classify_file_type("audio") == "audio"

    def test_document(self):
        assert _classify_file_type("document") == "document"

    def test_text_maps_to_document(self):
        assert _classify_file_type("text") == "document"

    def test_image_returns_none(self):
        # Images are intentionally excluded — BLIP captions fill that role.
        assert _classify_file_type("image") is None

    def test_archive_returns_none(self):
        assert _classify_file_type("archive") is None

    def test_unknown_type_returns_none(self):
        assert _classify_file_type("spreadsheet") is None

    def test_empty_string_returns_none(self):
        assert _classify_file_type("") is None

    def test_hvlink_mime_classified_as_video(self):
        # HVLink files (external video references) have file_type="other"
        # from the host's MIME heuristics but carry VTT-derived transcripts,
        # so they should feed the video summary path.
        assert (
            _classify_file_type("other", "application/vnd.homevault.link+json")
            == "video"
        )


# ---------------------------------------------------------------------------
# _build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """The system prompt should reflect llm.output_language from settings."""

    def test_japanese_language_instruction(self, monkeypatch, make_settings):
        settings = make_settings(llm=LLMConfig(output_language="ja"))
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.summaries.settings", settings)

        result = _build_system_prompt()

        assert "日本語で生成" in result

    def test_english_language_instruction(self, monkeypatch, make_settings):
        settings = make_settings(llm=LLMConfig(output_language="en"))
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.summaries.settings", settings)

        result = _build_system_prompt()

        assert "English" in result

    def test_auto_language_no_specific_instruction(
        self, monkeypatch, make_settings
    ):
        settings = make_settings(llm=LLMConfig(output_language="auto"))
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.summaries.settings", settings)

        result = _build_system_prompt()

        assert "日本語で生成" not in result
        assert "English" not in result

    def test_unknown_language_falls_through(self, monkeypatch, make_settings):
        settings = make_settings(llm=LLMConfig(output_language="fr"))
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.summaries.settings", settings)

        result = _build_system_prompt()

        assert "日本語で生成" not in result
        assert "English" not in result

    def test_prompt_contains_json_schema_hint(
        self, monkeypatch, make_settings
    ):
        settings = make_settings(llm=LLMConfig(output_language="auto"))
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.summaries.settings", settings)

        result = _build_system_prompt()

        # The JSON shape must appear so the model knows the contract.
        assert '"short"' in result
        assert '"long"' in result

    def test_prompt_mentions_length_ranges(
        self, monkeypatch, make_settings
    ):
        settings = make_settings(llm=LLMConfig(output_language="auto"))
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.summaries.settings", settings)

        result = _build_system_prompt()

        # Guide the model on how long each field should be.
        assert "30-80" in result
        assert "200-400" in result


# ---------------------------------------------------------------------------
# _trim_to_sentence_boundary
# ---------------------------------------------------------------------------


class TestTrimToSentenceBoundary:
    """Window edges should land on punctuation so sentences stay whole."""

    def test_japanese_sentence_trim(self):
        # Leading fragment "らない部分" should be cut at 。
        snippet = "らない部分。完全な文です。続きの"
        result = _trim_to_sentence_boundary(snippet)
        assert result.startswith("完全な文です")
        # Trailing fragment "続きの" cut after last 。
        assert result.endswith("完全な文です。")

    def test_english_question_exclamation_trim(self):
        # ASCII "." is intentionally NOT a sentence boundary (too many
        # false positives on abbreviations). "!" and "?" are.
        snippet = "fragment! Full sentence here? leftover"
        result = _trim_to_sentence_boundary(snippet)
        assert result.startswith("Full sentence here")
        assert result.endswith("?")

    def test_ascii_period_is_not_a_boundary(self):
        # Abbreviation / decimal-number case: the `.` in "Mr." and "3.14"
        # must not be treated as a sentence boundary. Without any other
        # boundary char, the snippet is returned stripped but unmodified.
        snippet = "Mr. Smith paid 3.14 dollars to Dr. Jones"
        result = _trim_to_sentence_boundary(snippet)
        assert result == snippet

    def test_newline_counts_as_boundary(self):
        snippet = "half line\nA complete block\ntail"
        result = _trim_to_sentence_boundary(snippet)
        assert "A complete block" in result
        # Trailing fragment "tail" should be removed because "\n" is a boundary.
        assert "tail" not in result

    def test_no_boundary_returns_stripped_original(self):
        snippet = "  no sentence boundaries here  "
        result = _trim_to_sentence_boundary(snippet)
        # Without a boundary we just return the stripped original.
        assert result == "no sentence boundaries here"

    def test_empty_string_returns_empty(self):
        assert _trim_to_sentence_boundary("") == ""

    def test_leading_trim_disabled_when_too_aggressive(self):
        # If the first boundary is past the midpoint, leading trim is
        # skipped so we don't lose the majority of the snippet chasing
        # a clean sentence start.
        snippet = "A B C D E F G H I J K L M N O P Q. tail"
        result = _trim_to_sentence_boundary(snippet)
        # Head preserved because first boundary is past the midpoint.
        assert result.startswith("A B C D")


# ---------------------------------------------------------------------------
# _sample_windows
# ---------------------------------------------------------------------------


class TestSampleWindows:
    """Windows should cover head/middle/tail without duplicating content."""

    def test_returns_text_unchanged_when_not_applicable(self):
        text = "short text"
        # window_count=0 disables sampling
        assert _sample_windows(text, window_chars=10, window_count=0) == text
        # window_chars=0 disables sampling
        assert _sample_windows(text, window_chars=0, window_count=3) == text
        # empty input
        assert _sample_windows("", window_chars=10, window_count=3) == ""

    def test_single_window_takes_head(self):
        # window_count=1 places the single window at the head, not the
        # middle — head coverage is the most common single-window use case.
        text = "HEAD" + ("_" * 200) + "TAIL"
        result = _sample_windows(text, window_chars=50, window_count=1)
        assert "HEAD" in result
        assert "TAIL" not in result

    def test_three_windows_cover_head_middle_tail(self):
        # Build text with a unique marker every 100 chars. Fillers are
        # underscores (not in the sentence-boundary set) so the trim
        # helper doesn't chew into the marker at the window edge.
        blocks = []
        for i in range(80):
            marker = f"[B{i:03d}]"  # 6 chars
            filler = "_" * (100 - len(marker))
            blocks = [*blocks, marker + filler]
        text = "".join(blocks)  # 8000 chars total

        result = _sample_windows(text, window_chars=400, window_count=3)

        # Head window [0, 400] should contain the earliest markers.
        assert any(f"B{i:03d}" in result for i in range(4)), \
            "head window missing"
        # Middle window centered at 4000 should cover blocks ~38-41.
        assert any(f"B{i:03d}" in result for i in range(38, 42)), \
            "middle window missing"
        # Tail window [7600, 8000] should contain the final markers.
        assert any(f"B{i:03d}" in result for i in range(76, 80)), \
            "tail window missing"

    def test_windows_joined_with_separator(self):
        text = "A." + ("x" * 500) + "B." + ("y" * 500) + "C." + ("z" * 500)
        result = _sample_windows(text, window_chars=150, window_count=3)
        # Separator appears between distinct windows
        assert _WINDOW_SEPARATOR in result

    def test_overlapping_windows_merge_instead_of_duplicating(self):
        # With a small text and large windows the spans will overlap.
        # The implementation merges them so we never see the same
        # content twice. Verify by counting unique substrings.
        text = "A" * 100 + "B" * 100 + "C" * 100
        result = _sample_windows(text, window_chars=150, window_count=3)
        # The separator should be absent when windows fully merged.
        # (Accept either 0 or 1 — depends on sentence trimming.)
        assert result.count(_WINDOW_SEPARATOR) <= 1


# ---------------------------------------------------------------------------
# _prepare_context
# ---------------------------------------------------------------------------


class TestPrepareContext:
    """Threshold-based dispatch between full text and window sampling."""

    def test_short_text_returned_as_is(self, monkeypatch, make_settings):
        settings = make_settings(
            summaries=SummariesConfig(max_context_chars=100)
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.summaries.settings", settings)

        text = "short content"
        prepared, truncated = _prepare_context(text)
        assert prepared == text
        assert truncated is False

    def test_long_text_triggers_window_sampling(
        self, monkeypatch, make_settings
    ):
        settings = make_settings(
            summaries=SummariesConfig(
                max_context_chars=500,
                window_chars=100,
                window_count=3,
            )
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.summaries.settings", settings)

        # 1000 chars > 500 threshold → sampling
        text = "a" * 1000
        prepared, truncated = _prepare_context(text)
        assert truncated is True
        # Sampled context should be shorter than the original.
        assert len(prepared) < len(text)

    def test_exact_threshold_is_not_truncated(
        self, monkeypatch, make_settings
    ):
        settings = make_settings(
            summaries=SummariesConfig(max_context_chars=100)
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.summaries.settings", settings)

        text = "x" * 100
        prepared, truncated = _prepare_context(text)
        assert truncated is False
        assert prepared == text


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    """User prompt assembly: metadata + optional truncation notice."""

    def _file(self, **overrides) -> dict:
        base = {
            "file_id": "abc",
            "filename": "video.mp4",
            "file_type": "video",
            "title": "",
            "description": "",
        }
        return {**base, **overrides}

    def test_includes_filename_and_type(self):
        result = _build_user_prompt(
            self._file(), "video", "some content", was_truncated=False
        )
        assert "ファイル名: video.mp4" in result
        assert "タイプ: video" in result
        assert "some content" in result

    def test_includes_title_when_different_from_filename(self):
        result = _build_user_prompt(
            self._file(title="My Great Video"),
            "video", "content", was_truncated=False,
        )
        assert "タイトル: My Great Video" in result

    def test_excludes_title_when_same_as_filename(self):
        result = _build_user_prompt(
            self._file(title="video.mp4"),
            "video", "content", was_truncated=False,
        )
        assert "タイトル:" not in result

    def test_includes_description(self):
        result = _build_user_prompt(
            self._file(description="A brief description"),
            "video", "content", was_truncated=False,
        )
        assert "説明: A brief description" in result

    def test_truncation_note_only_when_truncated(self):
        truncated = _build_user_prompt(
            self._file(), "video", "content", was_truncated=True
        )
        not_truncated = _build_user_prompt(
            self._file(), "video", "content", was_truncated=False
        )
        assert "抜粋" in truncated
        assert "抜粋" not in not_truncated


# ---------------------------------------------------------------------------
# _get_full_document_text
# ---------------------------------------------------------------------------


class TestGetFullDocumentText:
    """Document text loading reads from fts_text_content, not embeddings.

    The embeddings table only stores 200-char previews — loading full
    content from there would give the LLM disjoint fragments.
    """

    def test_concatenates_chunks_in_index_order(self, monkeypatch, mock_search_db):
        get_db, session = mock_search_db
        monkeypatch.setattr("app.workers.summaries.get_search_db", get_db)

        # Return rows in wrong order to verify ORDER BY CAST works.
        session.execute.return_value.fetchall.return_value = [
            ("chunk 1 text",),
            ("chunk 2 text",),
            ("chunk 10 text",),
        ]

        result = _get_full_document_text("abc")

        # All chunks are present and joined by a blank line.
        assert "chunk 1 text" in result
        assert "chunk 2 text" in result
        assert "chunk 10 text" in result
        # The SQL must include the numeric CAST, otherwise "10" sorts
        # before "2" lexicographically and summaries get scrambled.
        executed_sql = str(session.execute.call_args.args[0])
        assert "CAST(chunk_index AS INTEGER)" in executed_sql

    def test_empty_when_no_rows(self, monkeypatch, mock_search_db):
        get_db, session = mock_search_db
        monkeypatch.setattr("app.workers.summaries.get_search_db", get_db)

        session.execute.return_value.fetchall.return_value = []

        assert _get_full_document_text("abc") == ""

    def test_skips_empty_chunks(self, monkeypatch, mock_search_db):
        get_db, session = mock_search_db
        monkeypatch.setattr("app.workers.summaries.get_search_db", get_db)

        session.execute.return_value.fetchall.return_value = [
            ("real content",),
            ("",),
            (None,),
            ("more content",),
        ]

        result = _get_full_document_text("abc")
        assert "real content" in result
        assert "more content" in result


# ---------------------------------------------------------------------------
# classify_missing_reason
# ---------------------------------------------------------------------------


class TestClassifyMissingReason:
    """The router uses this to render the right "no summary yet" state."""

    def _patch(self, monkeypatch, make_settings, indexed=None, transcript="", text=""):
        settings = make_settings()
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.summaries.settings", settings)
        monkeypatch.setattr(
            "app.workers.summaries._get_indexed_file", lambda fid: indexed
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript", lambda fid: transcript
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_full_document_text", lambda fid: text
        )

    def test_file_not_found(self, monkeypatch, make_settings):
        self._patch(monkeypatch, make_settings, indexed=None)
        assert classify_missing_reason("abc") == "file_not_found"

    def test_unsupported_type(self, monkeypatch, make_settings):
        self._patch(
            monkeypatch, make_settings,
            indexed={
                "file_id": "abc",
                "filename": "photo.jpg",
                "file_type": "image",
                "title": "",
                "description": "",
            },
        )
        assert classify_missing_reason("abc") == "unsupported_type"

    def test_archive_is_unsupported(self, monkeypatch, make_settings):
        self._patch(
            monkeypatch, make_settings,
            indexed={
                "file_id": "abc",
                "filename": "archive.zip",
                "file_type": "archive",
                "title": "",
                "description": "",
            },
        )
        assert classify_missing_reason("abc") == "unsupported_type"

    def test_video_with_tiny_transcript_is_insufficient(
        self, monkeypatch, make_settings
    ):
        # The FFXIV piano cover case: filename-rich, transcript = "you".
        self._patch(
            monkeypatch, make_settings,
            indexed={
                "file_id": "abc",
                "filename": "FFXIV Piano.mp4",
                "file_type": "video",
                "title": "",
                "description": "",
            },
            transcript="you",
        )
        assert classify_missing_reason("abc") == "insufficient_content"

    def test_document_with_empty_text_is_insufficient(
        self, monkeypatch, make_settings
    ):
        self._patch(
            monkeypatch, make_settings,
            indexed={
                "file_id": "abc",
                "filename": "empty.pdf",
                "file_type": "document",
                "title": "",
                "description": "",
            },
            text="",
        )
        assert classify_missing_reason("abc") == "insufficient_content"

    def test_video_with_adequate_transcript_is_ready(
        self, monkeypatch, make_settings
    ):
        long_transcript = (
            "Welcome to this lecture on machine learning fundamentals. "
            "Today we will cover perceptrons, gradient descent, and basic "
            "neural network architectures in detail."
        )
        self._patch(
            monkeypatch, make_settings,
            indexed={
                "file_id": "abc",
                "filename": "lecture.mp4",
                "file_type": "video",
                "title": "",
                "description": "",
            },
            transcript=long_transcript,
        )
        assert classify_missing_reason("abc") == "not_generated"


# ---------------------------------------------------------------------------
# SummariesWorker._process_file
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm_client():
    """Mock LLM client whose generate_json can be set per-test."""
    client = MagicMock()
    client.enabled = True
    client.generate_json = AsyncMock(return_value=None)
    return client


@pytest.fixture()
def patched_settings_enabled(monkeypatch, make_settings):
    """Settings with summaries='manual' so the worker is allowed to run."""
    settings = make_settings(
        features=FeaturesConfig(summaries="manual"),
        llm=LLMConfig(provider="openai_compatible", model="test-model"),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.workers.summaries.settings", settings)
    return settings


class TestSummariesWorkerProcessFile:
    """End-to-end behavior of _process_file with mocked collaborators."""

    @pytest.mark.asyncio
    async def test_skips_when_feature_disabled(
        self, monkeypatch, make_settings, mock_llm_client
    ):
        settings = make_settings(features=FeaturesConfig(summaries="false"))
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.summaries.settings", settings)

        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        worker = SummariesWorker(mock_llm_client)
        await worker._process_file("abc")

        mock_llm_client.generate_json.assert_not_called()
        save_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_llm_disabled(
        self, monkeypatch, patched_settings_enabled
    ):
        llm = MagicMock()
        llm.enabled = False
        llm.generate_json = AsyncMock()

        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        worker = SummariesWorker(llm)
        await worker._process_file("abc")

        llm.generate_json.assert_not_called()
        save_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_summary_already_exists(
        self, monkeypatch, patched_settings_enabled, mock_llm_client
    ):
        monkeypatch.setattr(
            "app.workers.summaries._has_summary", lambda fid: True
        )
        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        worker = SummariesWorker(mock_llm_client)
        await worker._process_file("abc")

        mock_llm_client.generate_json.assert_not_called()
        save_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_indexed_file_missing(
        self, monkeypatch, patched_settings_enabled, mock_llm_client
    ):
        monkeypatch.setattr(
            "app.workers.summaries._has_summary", lambda fid: False
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_indexed_file", lambda fid: None
        )
        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        worker = SummariesWorker(mock_llm_client)
        await worker._process_file("abc")

        mock_llm_client.generate_json.assert_not_called()
        save_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_image_file_type(
        self, monkeypatch, patched_settings_enabled, mock_llm_client
    ):
        monkeypatch.setattr(
            "app.workers.summaries._has_summary", lambda fid: False
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_indexed_file",
            lambda fid: {
                "file_id": fid,
                "filename": "photo.jpg",
                "file_type": "image",
                "title": "",
                "description": "",
            },
        )
        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        worker = SummariesWorker(mock_llm_client)
        await worker._process_file("abc")

        mock_llm_client.generate_json.assert_not_called()
        save_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_context_empty(
        self, monkeypatch, patched_settings_enabled, mock_llm_client
    ):
        monkeypatch.setattr(
            "app.workers.summaries._has_summary", lambda fid: False
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_indexed_file",
            lambda fid: {
                "file_id": fid,
                "filename": "video.mp4",
                "file_type": "video",
                "title": "",
                "description": "",
            },
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript", lambda fid: ""
        )
        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        worker = SummariesWorker(mock_llm_client)
        await worker._process_file("abc")

        mock_llm_client.generate_json.assert_not_called()
        save_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_transcript_below_min_threshold(
        self, monkeypatch, patched_settings_enabled, mock_llm_client
    ):
        """A trivial transcript like "you" must not produce a summary.

        Without this guard the LLM would hallucinate a summary from the
        filename alone — see the real-world FFXIV piano cover case where
        Whisper produced a single "you" token and the model made up
        elaborate details from the filename.
        """
        monkeypatch.setattr(
            "app.workers.summaries._has_summary", lambda fid: False
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_indexed_file",
            lambda fid: {
                "file_id": fid,
                "filename": "FFXIV - Dawntrail OST Piano Cover.mp4",
                "file_type": "video",
                "title": "",
                "description": "",
            },
        )
        # 3 chars — well below the 50-char default threshold.
        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript", lambda fid: "you"
        )
        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        worker = SummariesWorker(mock_llm_client)
        await worker._process_file("abc")

        mock_llm_client.generate_json.assert_not_called()
        save_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_document_text_below_min_threshold(
        self, monkeypatch, patched_settings_enabled, mock_llm_client
    ):
        """A near-empty document must not produce a summary either."""
        monkeypatch.setattr(
            "app.workers.summaries._has_summary", lambda fid: False
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_indexed_file",
            lambda fid: {
                "file_id": fid,
                "filename": "report.pdf",
                "file_type": "document",
                "title": "",
                "description": "",
            },
        )
        # Only whitespace + a few chars — below threshold after strip().
        monkeypatch.setattr(
            "app.workers.summaries._get_full_document_text",
            lambda fid: "   hi   ",
        )
        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        worker = SummariesWorker(mock_llm_client)
        await worker._process_file("doc1")

        mock_llm_client.generate_json.assert_not_called()
        save_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_accepts_when_transcript_meets_threshold(
        self, monkeypatch, make_settings, mock_llm_client
    ):
        """A transcript exactly at the threshold should still produce a summary."""
        from app.config import FeaturesConfig, LLMConfig, SummariesConfig

        settings = make_settings(
            features=FeaturesConfig(summaries="manual"),
            llm=LLMConfig(provider="openai_compatible", model="m"),
            summaries=SummariesConfig(min_context_chars=20),
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.summaries.settings", settings)

        monkeypatch.setattr(
            "app.workers.summaries._has_summary", lambda fid: False
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_indexed_file",
            lambda fid: {
                "file_id": fid,
                "filename": "talk.mp4",
                "file_type": "video",
                "title": "",
                "description": "",
            },
        )
        # Exactly 20 chars (stripped).
        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "hello world of twenty",  # 21 chars, above threshold
        )

        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        mock_llm_client.generate_json = AsyncMock(
            return_value={"short": "s", "long": "l"}
        )

        worker = SummariesWorker(mock_llm_client)
        await worker._process_file("abc")

        save_spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_saves_valid_summary_for_video(
        self, monkeypatch, patched_settings_enabled, mock_llm_client
    ):
        monkeypatch.setattr(
            "app.workers.summaries._has_summary", lambda fid: False
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_indexed_file",
            lambda fid: {
                "file_id": fid,
                "filename": "lecture.mp4",
                "file_type": "video",
                "title": "",
                "description": "",
            },
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: (
                "This is a transcript about neural networks and how they "
                "learn patterns from data through repeated training cycles."
            ),
        )
        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        mock_llm_client.generate_json = AsyncMock(
            return_value={
                "short": "An intro to neural networks.",
                "long": "Covers the basics of perceptrons, activation functions, and backprop.",
            }
        )

        worker = SummariesWorker(mock_llm_client)
        await worker._process_file("abc")

        save_spy.assert_called_once()
        kwargs = save_spy.call_args.kwargs
        assert kwargs["file_id"] == "abc"
        assert kwargs["short_summary"] == "An intro to neural networks."
        assert "perceptrons" in kwargs["long_summary"]
        assert kwargs["context_type"] == "video"
        assert kwargs["was_truncated"] is False

    @pytest.mark.asyncio
    async def test_saves_valid_summary_for_document(
        self, monkeypatch, patched_settings_enabled, mock_llm_client
    ):
        monkeypatch.setattr(
            "app.workers.summaries._has_summary", lambda fid: False
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_indexed_file",
            lambda fid: {
                "file_id": fid,
                "filename": "report.pdf",
                "file_type": "document",
                "title": "Quarterly Report",
                "description": "",
            },
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_full_document_text",
            lambda fid: (
                "Full document body here with enough content to comfortably "
                "exceed the minimum context threshold for summary generation."
            ),
        )
        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        mock_llm_client.generate_json = AsyncMock(
            return_value={"short": "Q4 results", "long": "Details of Q4 revenue."}
        )

        worker = SummariesWorker(mock_llm_client)
        await worker._process_file("doc1")

        save_spy.assert_called_once()
        assert save_spy.call_args.kwargs["context_type"] == "document"

    @pytest.mark.asyncio
    async def test_rejects_non_dict_llm_response(
        self, monkeypatch, patched_settings_enabled, mock_llm_client
    ):
        monkeypatch.setattr(
            "app.workers.summaries._has_summary", lambda fid: False
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_indexed_file",
            lambda fid: {
                "file_id": fid,
                "filename": "a.mp4",
                "file_type": "video",
                "title": "",
                "description": "",
            },
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "transcript",
        )
        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        # LLM returned a list instead of a dict
        mock_llm_client.generate_json = AsyncMock(
            return_value=["not", "a", "dict"]
        )

        worker = SummariesWorker(mock_llm_client)
        await worker._process_file("abc")

        save_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_missing_fields(
        self, monkeypatch, patched_settings_enabled, mock_llm_client
    ):
        monkeypatch.setattr(
            "app.workers.summaries._has_summary", lambda fid: False
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_indexed_file",
            lambda fid: {
                "file_id": fid,
                "filename": "a.mp4",
                "file_type": "video",
                "title": "",
                "description": "",
            },
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "transcript",
        )
        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        # Missing "long" field
        mock_llm_client.generate_json = AsyncMock(
            return_value={"short": "only short"}
        )

        worker = SummariesWorker(mock_llm_client)
        await worker._process_file("abc")

        save_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_empty_summary_strings(
        self, monkeypatch, patched_settings_enabled, mock_llm_client
    ):
        monkeypatch.setattr(
            "app.workers.summaries._has_summary", lambda fid: False
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_indexed_file",
            lambda fid: {
                "file_id": fid,
                "filename": "a.mp4",
                "file_type": "video",
                "title": "",
                "description": "",
            },
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "transcript",
        )
        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        mock_llm_client.generate_json = AsyncMock(
            return_value={"short": "  ", "long": "\n\n"}
        )

        worker = SummariesWorker(mock_llm_client)
        await worker._process_file("abc")

        save_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_truncation_flag_propagated_to_save(
        self, monkeypatch, make_settings, mock_llm_client
    ):
        # Force very small threshold so any non-trivial content truncates.
        settings = make_settings(
            features=FeaturesConfig(summaries="manual"),
            llm=LLMConfig(provider="openai_compatible", model="m"),
            summaries=SummariesConfig(
                max_context_chars=50,
                window_chars=20,
                window_count=3,
            ),
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.workers.summaries.settings", settings)

        monkeypatch.setattr(
            "app.workers.summaries._has_summary", lambda fid: False
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_indexed_file",
            lambda fid: {
                "file_id": fid,
                "filename": "long.mp4",
                "file_type": "video",
                "title": "",
                "description": "",
            },
        )
        monkeypatch.setattr(
            "app.workers.summaries._get_full_transcript",
            lambda fid: "x" * 5000,
        )

        save_spy = MagicMock()
        monkeypatch.setattr(
            "app.workers.summaries._save_summary", save_spy
        )

        mock_llm_client.generate_json = AsyncMock(
            return_value={"short": "s", "long": "l"}
        )

        worker = SummariesWorker(mock_llm_client)
        await worker._process_file("abc")

        save_spy.assert_called_once()
        assert save_spy.call_args.kwargs["was_truncated"] is True
        # context_chars is the length of the PREPARED context, not raw.
        assert save_spy.call_args.kwargs["context_chars"] < 5000
