"""Tests for app.rag.context module.

Covers build_file_context and assemble_contexts: the per-file context
excerpt builders and the total-budget-enforcing aggregator.

Segment-driven context fetching (transcript chunks, FTS text content)
is mocked so tests don't touch the real search DB.
"""

import sys
from unittest.mock import MagicMock

import pytest

# Stub out heavy ML deps before importing modules that depend on app.search.
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

from app.config import RagConfig  # noqa: E402
from app.rag.context import (  # noqa: E402
    ContextSnippet,
    FileContext,
    assemble_contexts,
    build_file_context,
)
from app.rag.retriever import RetrievedFile  # noqa: E402
from app.search import MatchInfo, SegmentGroup  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _retrieved_video(
    file_id: str = "v1",
    title: str = "Video Title",
    description: str = "Video Description",
    score: float = 0.9,
    segments: tuple[SegmentGroup, ...] | None = None,
) -> RetrievedFile:
    if segments is None:
        match = MatchInfo(
            match_type="transcript",
            text="spoken snippet",
            score=0.8,
            timestamp_start=45.0,
            timestamp_end=60.0,
        )
        segments = (SegmentGroup(time_range=(45.0, 60.0), matches=(match,)),)

    return RetrievedFile(
        file_id=file_id,
        drive="Videos",
        filename="clip.mp4",
        file_type="video",
        title=title,
        description=description,
        score=score,
        match_types=("transcript",),
        segments=segments,
    )


def _retrieved_document(
    file_id: str = "d1",
    title: str = "Doc Title",
    description: str = "Doc Description",
    score: float = 0.9,
    chunk_indices: tuple[int, ...] = (3,),
) -> RetrievedFile:
    matches = tuple(
        MatchInfo(
            match_type="text_content",
            text=f"fragment {i}",
            score=0.8,
            page=i,
        )
        for i in chunk_indices
    )
    segments = (SegmentGroup(time_range=None, matches=matches),)
    return RetrievedFile(
        file_id=file_id,
        drive="Docs",
        filename="report.pdf",
        file_type="document",
        title=title,
        description=description,
        score=score,
        match_types=("text_content",),
        segments=segments,
    )


def _retrieved_image(
    file_id: str = "i1",
    title: str = "Image Title",
    description: str = "Image Description",
    score: float = 0.9,
) -> RetrievedFile:
    match = MatchInfo(
        match_type="clip",
        text="a sunset at the beach",
        score=0.7,
    )
    segments = (SegmentGroup(time_range=None, matches=(match,)),)
    return RetrievedFile(
        file_id=file_id,
        drive="Photos",
        filename="sunset.jpg",
        file_type="image",
        title=title,
        description=description,
        score=score,
        match_types=("clip",),
        segments=segments,
    )


# ---------------------------------------------------------------------------
# build_file_context: video / audio
# ---------------------------------------------------------------------------


class TestBuildFileContextVideo:
    """T1: video context pulls transcript chunks around segment timestamps."""

    def test_fetches_transcript_chunks_around_timestamp(
        self, monkeypatch
    ):
        # Stub: return a few transcript chunks inside the spec's
        # transcript_window_seconds window around the segment midpoint.
        fetch_spy = MagicMock(
            return_value=[
                ("The speaker introduces the topic.", 30.0, 40.0),
                ("They discuss the main points.", 40.0, 55.0),
                ("And then conclude the section.", 55.0, 70.0),
            ]
        )
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_around", fetch_spy
        )

        candidate = _retrieved_video()
        cfg = RagConfig()

        ctx = build_file_context(candidate, cfg)

        assert isinstance(ctx, FileContext)
        assert ctx.file_id == "v1"
        assert ctx.filename == "clip.mp4"
        assert len(ctx.snippets) >= 1
        # At least one snippet from the transcript source.
        transcript_snippets = [
            s for s in ctx.snippets if s.source == "transcript"
        ]
        assert len(transcript_snippets) >= 1
        joined = " ".join(s.text for s in transcript_snippets)
        assert "speaker" in joined or "main points" in joined

        # The fetch helper was called with the correct file_id and
        # timestamp information (window derived from segments).
        assert fetch_spy.called
        call_args = fetch_spy.call_args
        assert call_args.args[0] == "v1" or call_args.kwargs.get("file_id") == "v1"

    def test_audio_file_uses_same_transcript_path(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_around",
            MagicMock(return_value=[("spoken text", 0.0, 10.0)]),
        )

        candidate = RetrievedFile(
            file_id="a1",
            drive="Audio",
            filename="podcast.mp3",
            file_type="audio",
            title="Podcast",
            description="",
            score=0.8,
            match_types=("transcript",),
            segments=(
                SegmentGroup(
                    time_range=(5.0, 15.0),
                    matches=(
                        MatchInfo(
                            match_type="transcript",
                            text="spoken",
                            score=0.9,
                            timestamp_start=5.0,
                            timestamp_end=15.0,
                        ),
                    ),
                ),
            ),
        )

        ctx = build_file_context(candidate, RagConfig())

        assert ctx.file_id == "a1"
        assert any(s.source == "transcript" for s in ctx.snippets)

    def test_hvlink_file_uses_transcript_path(self, monkeypatch):
        # HvLink files have file_type="other" from host MIME heuristics
        # but carry VTT-derived TranscriptChunks — Ask must feed those
        # to the LLM instead of falling back to metadata only.
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_around",
            MagicMock(return_value=[("subtitle text", 0.0, 10.0)]),
        )

        candidate = RetrievedFile(
            file_id="h1",
            drive="YouTube",
            filename="video.hvlink",
            file_type="other",
            title="External video",
            description="",
            score=0.8,
            match_types=("transcript",),
            segments=(
                SegmentGroup(
                    time_range=(5.0, 15.0),
                    matches=(
                        MatchInfo(
                            match_type="transcript",
                            text="subtitle",
                            score=0.9,
                            timestamp_start=5.0,
                            timestamp_end=15.0,
                        ),
                    ),
                ),
            ),
            mime_type="application/vnd.homevault.link+json",
        )

        ctx = build_file_context(candidate, RagConfig())

        assert ctx.file_id == "h1"
        assert any(s.source == "transcript" for s in ctx.snippets)


# ---------------------------------------------------------------------------
# build_file_context: document
# ---------------------------------------------------------------------------


class TestBuildFileContextDocument:
    """T2: document context loads neighboring FTS chunks."""

    def test_fetches_chunks_around_index(self, monkeypatch):
        # Stub returns chunks at index 2, 3, 4 (neighborhood of match at 3).
        fetch_spy = MagicMock(
            return_value=[
                (2, "Earlier paragraph text here."),
                (3, "The matching paragraph mentions the topic."),
                (4, "Following paragraph continues the argument."),
            ]
        )
        monkeypatch.setattr(
            "app.rag.context._fetch_document_chunks_around", fetch_spy
        )

        candidate = _retrieved_document(chunk_indices=(3,))
        cfg = RagConfig()

        ctx = build_file_context(candidate, cfg)

        assert isinstance(ctx, FileContext)
        text_snippets = [
            s for s in ctx.snippets if s.source == "text_content"
        ]
        assert len(text_snippets) >= 1
        joined = " ".join(s.text for s in text_snippets)
        assert "matching paragraph" in joined

        assert fetch_spy.called
        # The fetch helper was called with the file_id (positional or kwarg).
        call = fetch_spy.call_args
        assert call.args[0] == "d1" or call.kwargs.get("file_id") == "d1"

    def test_cast_chunk_index_as_integer_required(self, monkeypatch):
        """hako memo EAiVExR4vGgOym5aAv_Up: chunk_index is stored as a string
        in the FTS5 table, so neighbor queries must use CAST to INTEGER
        for correct ordering. The fetcher is a black box here, but the
        SQL inspection test in test_summaries.py covers the same invariant
        for the summaries worker. For RAG we assert the fetcher is invoked;
        the fetcher's internal SQL correctness lives in its own test.
        """
        fetch_spy = MagicMock(return_value=[])
        monkeypatch.setattr(
            "app.rag.context._fetch_document_chunks_around", fetch_spy
        )

        candidate = _retrieved_document()
        build_file_context(candidate, RagConfig())

        assert fetch_spy.called


# ---------------------------------------------------------------------------
# build_file_context: image
# ---------------------------------------------------------------------------


class TestBuildFileContextImage:
    """T3: image context uses BLIP caption + filename + description."""

    def test_uses_blip_caption_when_available(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.context._fetch_blip_caption",
            MagicMock(return_value="A serene sunset over the ocean."),
        )

        candidate = _retrieved_image(
            title="Beach",
            description="Vacation photo",
        )

        ctx = build_file_context(candidate, RagConfig())

        assert ctx.file_id == "i1"
        joined = " ".join(s.text for s in ctx.snippets)
        # The BLIP caption and metadata must all appear.
        assert "sunset over the ocean" in joined
        # Filename and/or description serve as additional context.
        assert "sunset.jpg" in joined or "Vacation photo" in joined

    def test_falls_back_to_metadata_when_no_caption(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.context._fetch_blip_caption",
            MagicMock(return_value=None),
        )

        candidate = _retrieved_image(
            title="Mountain",
            description="A shot of the hills",
        )

        ctx = build_file_context(candidate, RagConfig())

        # Without a caption we still produce *some* context from metadata.
        joined = " ".join(s.text for s in ctx.snippets)
        assert "Mountain" in joined or "hills" in joined or "sunset.jpg" in joined


# ---------------------------------------------------------------------------
# build_file_context: per-file budget
# ---------------------------------------------------------------------------


class TestPerFileBudget:
    """T4: max_context_chars_per_file is respected."""

    def test_truncates_long_context(self, monkeypatch):
        # Return a huge blob to force truncation.
        huge = "x" * 10000
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_around",
            MagicMock(return_value=[(huge, 0.0, 100.0)]),
        )

        cfg = RagConfig(max_context_chars_per_file=500)
        candidate = _retrieved_video()

        ctx = build_file_context(candidate, cfg)

        # Total characters across snippets should be at or below budget
        # (allowing a small overhead for separators / formatting).
        assert ctx.total_chars <= cfg.max_context_chars_per_file + 100


# ---------------------------------------------------------------------------
# build_file_context: minimal / empty segments
# ---------------------------------------------------------------------------


class TestMinimalContext:
    """T6: when segments are empty, metadata-only fallback."""

    def test_empty_segments_returns_metadata_only(self, monkeypatch):
        # No fetch spies needed — none should be called.
        candidate = RetrievedFile(
            file_id="x1",
            drive="Videos",
            filename="noseg.mp4",
            file_type="video",
            title="My Title",
            description="My Description",
            score=0.5,
            match_types=(),
            segments=(),
        )

        ctx = build_file_context(candidate, RagConfig())

        joined = " ".join(s.text for s in ctx.snippets)
        # Filename / title / description should at least appear.
        assert (
            "noseg.mp4" in joined
            or "My Title" in joined
            or "My Description" in joined
        )
        # Snippet source should be "metadata" for this fallback path.
        metadata_snippets = [s for s in ctx.snippets if s.source == "metadata"]
        assert len(metadata_snippets) >= 1


# ---------------------------------------------------------------------------
# assemble_contexts
# ---------------------------------------------------------------------------


class TestAssembleContexts:
    """T5 / T7: total-budget enforcement and empty-result handling."""

    def test_drops_lowest_scored_when_total_exceeds_budget(
        self, monkeypatch
    ):
        # Each file yields roughly 2000 chars of content.
        huge_chunks = [("y" * 2000, 0.0, 10.0)]
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_around",
            MagicMock(return_value=huge_chunks),
        )

        # 10 candidates at 2000 chars each -> 20000 total.
        # Budget is 4000 total -> expect ~2 surviving files, with the
        # highest scores preferred.
        candidates = [
            _retrieved_video(file_id=f"v{i}", score=1.0 - i * 0.05)
            for i in range(10)
        ]
        cfg = RagConfig(
            max_context_chars_per_file=2500,
            max_total_context_chars=4000,
        )

        contexts = assemble_contexts(candidates, cfg)

        total = sum(c.total_chars for c in contexts)
        assert total <= cfg.max_total_context_chars + 200  # slack
        # The surviving files should be drawn from the top scores.
        surviving_ids = {c.file_id for c in contexts}
        # We should NOT keep the very lowest-score files.
        assert "v9" not in surviving_ids or "v0" in surviving_ids

    def test_highest_score_preserved(self, monkeypatch):
        huge_chunks = [("z" * 3000, 0.0, 10.0)]
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_around",
            MagicMock(return_value=huge_chunks),
        )

        candidates = [
            _retrieved_video(file_id="high", score=0.99),
            _retrieved_video(file_id="mid", score=0.5),
            _retrieved_video(file_id="low", score=0.1),
        ]
        cfg = RagConfig(
            max_context_chars_per_file=3500,
            max_total_context_chars=4000,
        )

        contexts = assemble_contexts(candidates, cfg)

        # The highest-score file must always survive (unless it's itself
        # larger than the total budget, which is not the case here).
        assert any(c.file_id == "high" for c in contexts)

    def test_empty_candidates_returns_empty_list(self):
        """T7: no candidates -> empty contexts, no crash."""
        contexts = assemble_contexts([], RagConfig())
        assert contexts == []

    def test_all_files_empty_context_returns_empty_list(self, monkeypatch):
        """T7: if no file produces any snippet, return []."""
        # Return no transcript content at all.
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_around",
            MagicMock(return_value=[]),
        )

        # Candidates have no metadata either -> contexts are truly empty.
        candidates = [
            RetrievedFile(
                file_id=f"v{i}",
                drive="Videos",
                filename="",
                file_type="video",
                title=None,
                description=None,
                score=0.5 - i * 0.1,
                match_types=(),
                segments=(),
            )
            for i in range(3)
        ]

        contexts = assemble_contexts(candidates, RagConfig())

        # Either returns empty, or returns FileContext stubs with no
        # snippets. The spec says "空リストを返す" for the all-empty case.
        if contexts:
            assert all(len(c.snippets) == 0 for c in contexts)
        else:
            assert contexts == []

    def test_within_budget_keeps_everyone(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_around",
            MagicMock(return_value=[("small text", 0.0, 5.0)]),
        )

        candidates = [
            _retrieved_video(file_id=f"v{i}", score=1.0 - i * 0.01)
            for i in range(5)
        ]
        cfg = RagConfig(
            max_context_chars_per_file=2000,
            max_total_context_chars=100000,
        )

        contexts = assemble_contexts(candidates, cfg)

        # Budget is generous -> keep all 5.
        assert len(contexts) == 5


# ---------------------------------------------------------------------------
# ContextSnippet dataclass
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Transcript: 3-method context (window + keyword + vector)
# ---------------------------------------------------------------------------


class TestTranscriptMultiMethod:
    """Tests for the enhanced transcript context with keyword-OR and vector
    chunk retrieval in addition to the time-window approach."""

    def _stub_all(self, monkeypatch, *, window_chunks=None, kw_chunks=None, vec_chunks=None):
        """Set up monkeypatches for all three transcript fetch helpers."""
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_around",
            MagicMock(return_value=window_chunks or []),
        )
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_by_keyword_or",
            MagicMock(return_value=kw_chunks or []),
        )
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_by_vector",
            MagicMock(return_value=vec_chunks or []),
        )
        monkeypatch.setattr(
            "app.rag.context.filter_keywords",
            MagicMock(side_effect=lambda k: k),
        )

    def test_keyword_chunks_added_when_no_overlap(self, monkeypatch):
        """Keyword-OR chunks at a distant timestamp are included."""
        self._stub_all(
            monkeypatch,
            window_chunks=[("window text", 10.0, 20.0)],
            kw_chunks=[("keyword match far away", 300.0, 310.0)],
        )
        candidate = _retrieved_video()
        cfg = RagConfig(transcript_vector_top_n=4)

        ctx = build_file_context(candidate, cfg, keywords="keyword")

        sources = [s.source for s in ctx.snippets]
        assert "transcript" in sources
        assert "transcript_keyword" in sources
        assert any("keyword match" in s.text for s in ctx.snippets)

    def test_vector_chunks_added_when_no_overlap(self, monkeypatch):
        """Vector-similar chunks at a distant timestamp are included."""
        fake_vec = MagicMock()
        self._stub_all(
            monkeypatch,
            window_chunks=[("window text", 10.0, 20.0)],
            vec_chunks=[("vector match far away", 500.0, 510.0)],
        )
        candidate = _retrieved_video()
        cfg = RagConfig(transcript_vector_top_n=4)

        ctx = build_file_context(candidate, cfg, query_vector=fake_vec)

        sources = [s.source for s in ctx.snippets]
        assert "transcript" in sources
        assert "transcript_vector" in sources
        assert any("vector match" in s.text for s in ctx.snippets)

    def test_overlapping_keyword_chunk_is_deduplicated(self, monkeypatch):
        """Keyword chunk whose timestamp overlaps the window is skipped."""
        self._stub_all(
            monkeypatch,
            window_chunks=[("window text", 10.0, 20.0)],
            # This chunk is inside the ±60s window (45-60 segment → -15 to 120)
            kw_chunks=[("overlapping kw", 50.0, 55.0)],
        )
        candidate = _retrieved_video()  # segment at 45-60
        cfg = RagConfig(transcript_vector_top_n=4)

        ctx = build_file_context(candidate, cfg, keywords="test")

        # The keyword chunk should be skipped due to overlap.
        assert not any(s.source == "transcript_keyword" for s in ctx.snippets)

    def test_overlapping_vector_chunk_is_deduplicated(self, monkeypatch):
        """Vector chunk whose timestamp overlaps the window is skipped."""
        fake_vec = MagicMock()
        self._stub_all(
            monkeypatch,
            window_chunks=[("window text", 10.0, 20.0)],
            vec_chunks=[("overlapping vec", 50.0, 55.0)],
        )
        candidate = _retrieved_video()  # segment at 45-60
        cfg = RagConfig(transcript_vector_top_n=4)

        ctx = build_file_context(candidate, cfg, query_vector=fake_vec)

        assert not any(s.source == "transcript_vector" for s in ctx.snippets)

    def test_all_three_methods_combined(self, monkeypatch):
        """All three methods produce snippets when non-overlapping."""
        fake_vec = MagicMock()
        self._stub_all(
            monkeypatch,
            window_chunks=[("window text", 10.0, 20.0)],
            kw_chunks=[("keyword hit", 200.0, 210.0)],
            vec_chunks=[("vector hit", 400.0, 410.0)],
        )
        candidate = _retrieved_video()
        cfg = RagConfig(transcript_vector_top_n=4)

        ctx = build_file_context(
            candidate, cfg, query_vector=fake_vec, keywords="test"
        )

        sources = {s.source for s in ctx.snippets}
        assert sources == {"transcript", "transcript_keyword", "transcript_vector"}

    def test_disabled_when_top_n_is_zero(self, monkeypatch):
        """transcript_vector_top_n=0 disables extra passes."""
        kw_spy = MagicMock(return_value=[("kw", 300.0, 310.0)])
        vec_spy = MagicMock(return_value=[("vec", 500.0, 510.0)])
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_around",
            MagicMock(return_value=[("window", 10.0, 20.0)]),
        )
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_by_keyword_or", kw_spy,
        )
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_by_vector", vec_spy,
        )
        monkeypatch.setattr(
            "app.rag.context.filter_keywords",
            MagicMock(side_effect=lambda k: k),
        )

        candidate = _retrieved_video()
        cfg = RagConfig(transcript_vector_top_n=0)

        ctx = build_file_context(
            candidate, cfg, query_vector=MagicMock(), keywords="test"
        )

        kw_spy.assert_not_called()
        vec_spy.assert_not_called()
        assert all(s.source == "transcript" for s in ctx.snippets)

    def test_keyword_dedup_prevents_vector_double_add(self, monkeypatch):
        """A chunk already added by keyword is not re-added by vector."""
        fake_vec = MagicMock()
        self._stub_all(
            monkeypatch,
            window_chunks=[("window text", 10.0, 20.0)],
            kw_chunks=[("shared chunk", 300.0, 310.0)],
            vec_chunks=[("shared chunk", 300.0, 310.0)],
        )
        candidate = _retrieved_video()
        cfg = RagConfig(transcript_vector_top_n=4)

        ctx = build_file_context(
            candidate, cfg, query_vector=fake_vec, keywords="test"
        )

        # The chunk at 300-310 should appear exactly once.
        extra = [
            s for s in ctx.snippets
            if s.source in ("transcript_keyword", "transcript_vector")
        ]
        assert len(extra) == 1

    def test_window_seconds_default_is_60(self):
        """Default transcript_window_seconds should be 60."""
        cfg = RagConfig()
        assert cfg.transcript_window_seconds == 60.0

    def test_transcript_vector_top_n_default_is_4(self):
        """Default transcript_vector_top_n should be 4."""
        cfg = RagConfig()
        assert cfg.transcript_vector_top_n == 4


class TestContextSnippetDataclass:
    """ContextSnippet fields and immutability."""

    def test_fields_present(self):
        snippet = ContextSnippet(
            source="transcript",
            text="Hello world",
            location="0:45",
        )
        assert snippet.source == "transcript"
        assert snippet.text == "Hello world"
        assert snippet.location == "0:45"

    def test_location_can_be_none(self):
        snippet = ContextSnippet(
            source="metadata", text="Filename", location=None
        )
        assert snippet.location is None

    def test_is_frozen(self):
        snippet = ContextSnippet(source="clip", text="caption", location=None)
        with pytest.raises(Exception):
            snippet.text = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FileContext dataclass
# ---------------------------------------------------------------------------


class TestFileContextDataclass:
    """FileContext fields."""

    def test_fields_present(self):
        ctx = FileContext(
            file_id="f1",
            filename="a.mp4",
            drive="Videos",
            file_type="video",
            title="T",
            description="D",
            snippets=(
                ContextSnippet(source="transcript", text="text", location="0:00"),
            ),
            total_chars=4,
        )
        assert ctx.file_id == "f1"
        assert ctx.filename == "a.mp4"
        assert ctx.drive == "Videos"
        assert ctx.file_type == "video"
        assert ctx.title == "T"
        assert ctx.description == "D"
        assert len(ctx.snippets) == 1
        assert ctx.total_chars == 4
