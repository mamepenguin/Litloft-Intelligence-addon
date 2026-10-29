"""SPEC-ADDON-037: an Ask citation quotes the passage at its location, not the
context before it.

Each test names the spec item it checks (step N of Required items, or an
invariant I1-I10).
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.llm_helpers import bind_llm

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

from app.config import HierarchicalRagConfig, LLMConfig, RagConfig  # noqa: E402
from app.rag.context import (  # noqa: E402
    ContextSnippet,
    FileContext,
    _cap_snippets,
    build_file_context,
)
from app.rag.prompt import build_user_prompt  # noqa: E402
from app.rag.retriever import RetrievedFile  # noqa: E402
from app.rag.service import _answer_question as answer_question  # noqa: E402
from app.rag.service import _quote_from_contexts  # noqa: E402
from app.search import MatchInfo, SegmentGroup  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rule(text: str, limit: int = 200) -> str:
    """Today's quote rule: strip, then cap with a trailing ellipsis."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _sentences(tag: str, count: int) -> str:
    """`count` sentences of 40 characters each, every one ending in a boundary."""
    return "".join(f"{tag}{i:02d}" + "x" * 35 + "。" for i in range(count))


def _ctx(snippets: list[ContextSnippet], file_id: str = "f") -> FileContext:
    return FileContext(
        file_id=file_id, filename="f.mp4", drive="D", file_type="video",
        title=None, description=None, snippets=tuple(snippets), total_chars=1,
    )


def _video(start: float = 100.0, end: float = 150.0) -> RetrievedFile:
    match = MatchInfo(
        match_type="transcript", text="spoken", score=0.8,
        timestamp_start=start, timestamp_end=end,
    )
    return RetrievedFile(
        file_id="v1", drive="Videos", filename="clip.mp4", file_type="video",
        title=None, description=None, score=0.9, match_types=("transcript",),
        segments=(SegmentGroup(time_range=(start, end), matches=(match,)),),
    )


def _document(chunk_index: int = 3, mime: str = "application/pdf") -> RetrievedFile:
    match = MatchInfo(
        match_type="text_content", text="fragment", score=0.8,
        page=None, chunk_index=chunk_index,
    )
    return RetrievedFile(
        file_id="d1", drive="Docs", filename="report.pdf", file_type="document",
        mime_type=mime, title=None, description=None, score=0.9,
        match_types=("text_content",),
        segments=(SegmentGroup(time_range=None, matches=(match,)),),
    )


def _stub_transcript(monkeypatch, window_chunks) -> None:
    monkeypatch.setattr(
        "app.rag.context._fetch_transcript_chunks_around",
        MagicMock(return_value=window_chunks),
    )
    monkeypatch.setattr(
        "app.rag.context._fetch_transcript_chunks_by_keyword_or",
        MagicMock(return_value=[]),
    )
    monkeypatch.setattr(
        "app.rag.context._fetch_transcript_chunks_by_vector",
        MagicMock(return_value=[]),
    )


def _window_snippet(ctx: FileContext) -> ContextSnippet:
    found = [s for s in ctx.snippets if s.source == "transcript"]
    assert len(found) == 1, f"expected one time-window snippet, got {len(found)}"
    return found[0]


def _quote_of(ctx: FileContext, snippet: ContextSnippet) -> str:
    return _quote_from_contexts(ctx.file_id, [ctx], location=snippet.location)


def _build_window(monkeypatch, chunks, *, start=100.0, budget=100000):
    _stub_transcript(monkeypatch, chunks)
    cfg = RagConfig(max_context_chars_per_file=budget)
    return build_file_context(_video(start, start + 50.0), cfg)


# ---------------------------------------------------------------------------
# ContextSnippet.anchor (step 1)
# ---------------------------------------------------------------------------


class TestAnchorField:
    def test_default_is_zero(self) -> None:
        """SPEC-ADDON-037 step 1"""
        assert ContextSnippet(source="clip", text="x", location=None).anchor == 0

    def test_is_stored_and_frozen(self) -> None:
        """SPEC-ADDON-037 step 1"""
        s = ContextSnippet(source="transcript", text="abcdef", location="0:01", anchor=3)
        assert s.anchor == 3
        with pytest.raises(Exception):
            s.anchor = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Time-window snippet (step 2, 4, failure cases, I1)
# ---------------------------------------------------------------------------


class TestWindowAnchor:
    def test_quote_starts_at_first_chunk_ending_after_start(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 2, I1"""
        ctx = _build_window(monkeypatch, [
            ("lead one", 40.0, 70.0),
            ("lead two", 70.0, 99.0),
            ("cited passage", 99.5, 130.0),
            ("after", 130.0, 160.0),
        ])
        snippet = _window_snippet(ctx)

        assert snippet.text == "lead one lead two cited passage after"
        assert snippet.anchor == len("lead one lead two ")
        assert _quote_of(ctx, snippet) == "cited passage after"

    def test_chunk_ending_exactly_at_start_is_not_the_target(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 2 (strictly greater)"""
        ctx = _build_window(monkeypatch, [
            ("ends at start", 70.0, 100.0),
            ("begins at start", 100.0, 130.0),
        ])
        snippet = _window_snippet(ctx)

        assert _quote_of(ctx, snippet) == "begins at start"

    def test_identical_earlier_chunk_does_not_capture_the_anchor(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 4, I1"""
        ctx = _build_window(monkeypatch, [
            ("[音楽]", 40.0, 70.0),
            ("[音楽]", 70.0, 130.0),
            ("then words", 130.0, 160.0),
        ])
        snippet = _window_snippet(ctx)

        assert snippet.anchor == len("[音楽] ")
        assert _quote_of(ctx, snippet) == "[音楽] then words"

    def test_empty_chunks_before_the_target_add_no_separator(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 4"""
        ctx = _build_window(monkeypatch, [
            ("lead", 40.0, 60.0),
            ("", 60.0, 80.0),
            ("", 80.0, 90.0),
            ("target", 100.0, 130.0),
        ])
        snippet = _window_snippet(ctx)

        assert _quote_of(ctx, snippet) == "target"

    def test_no_chunk_ends_after_start_anchors_at_zero(self, monkeypatch) -> None:
        """SPEC-ADDON-037 section 2 failure case"""
        ctx = _build_window(monkeypatch, [
            ("early one", 10.0, 40.0),
            ("early two", 40.0, 90.0),
        ])
        snippet = _window_snippet(ctx)

        assert snippet.anchor == 0
        assert _quote_of(ctx, snippet) == _rule(snippet.text)

    def test_long_quote_is_capped_from_the_anchor(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 7, I9"""
        target = "t" * 500
        ctx = _build_window(monkeypatch, [
            ("lead-in " * 20, 40.0, 90.0),
            (target, 100.0, 130.0),
        ])
        quote = _quote_of(ctx, _window_snippet(ctx))

        assert quote == "t" * 200 + "…"


# ---------------------------------------------------------------------------
# Expanded neighbour document snippet (step 3, 4, failure cases, I2)
# ---------------------------------------------------------------------------


def _document_ctx(monkeypatch, around, chunk_index=3):
    monkeypatch.setattr(
        "app.rag.context._fetch_document_chunks_around",
        MagicMock(return_value=around),
    )
    return build_file_context(
        _document(chunk_index), RagConfig(max_context_chars_per_file=100000),
    )


def _neighbour_snippet(ctx: FileContext, needle: str) -> ContextSnippet:
    found = [s for s in ctx.snippets if s.source == "text_content" and needle in s.text]
    assert found, "no document snippet carries the neighbour text"
    return found[0]


class TestNeighbourAnchor:
    def test_quote_is_the_target_chunk_not_the_one_before(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 3, I2"""
        ctx = _document_ctx(monkeypatch, [
            (2, "Earlier paragraph.", 7),
            (3, "The cited paragraph.", 8),
            (4, "Following paragraph.", 8),
        ])
        snippet = _neighbour_snippet(ctx, "Earlier paragraph.")

        assert snippet.anchor == len("Earlier paragraph.\n\n")
        assert _quote_of(ctx, snippet) == "The cited paragraph.\n\nFollowing paragraph."

    def test_identical_previous_chunk_does_not_capture_the_anchor(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 4, I2"""
        ctx = _document_ctx(monkeypatch, [
            (2, "Boilerplate.", 7),
            (3, "Boilerplate.", 8),
            (4, "Real content.", 8),
        ])
        snippet = _neighbour_snippet(ctx, "Real content.")

        assert snippet.anchor == len("Boilerplate.\n\n")
        assert _quote_of(ctx, snippet) == "Boilerplate.\n\nReal content."

    def test_target_is_first_chunk_of_the_fetch(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 3 boundary: nothing before the target"""
        ctx = _document_ctx(monkeypatch, [
            (3, "The cited paragraph.", 8),
            (4, "Following.", 8),
        ])
        snippet = _neighbour_snippet(ctx, "Following.")

        assert snippet.anchor == 0
        assert _quote_of(ctx, snippet) == "The cited paragraph.\n\nFollowing."

    def test_empty_chunk_before_the_target_adds_no_separator(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 4, I2"""
        ctx = _document_ctx(monkeypatch, [
            (2, "", 7),
            (3, "Cited.", 8),
            (4, "Next.", 8),
        ])
        snippet = _neighbour_snippet(ctx, "Next.")

        assert snippet.anchor == 0
        assert _quote_of(ctx, snippet) == "Cited.\n\nNext."

    def test_vector_selected_chunk_is_not_anchored_past_its_start(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 5, I4"""
        monkeypatch.setattr(
            "app.rag.context._fetch_document_chunks_around",
            MagicMock(return_value=[
                (2, "Prev.", 1), (3, "Vector hit.", 1), (4, "Next.", 1),
            ]),
        )
        monkeypatch.setattr(
            "app.rag.context._fetch_document_chunks_by_vector",
            MagicMock(return_value=[3]),
        )
        ctx = build_file_context(
            _document(chunk_index=9),
            RagConfig(max_context_chars_per_file=100000, document_vector_top_n=4),
            query_vector=MagicMock(),
        )

        vector = [s for s in ctx.snippets if s.source == "text_content_vector"]
        assert [s.text for s in vector] == ["Vector hit."]
        assert vector[0].anchor == 0

    def test_missing_target_chunk_anchors_at_zero(self, monkeypatch) -> None:
        """SPEC-ADDON-037 section 2 failure case"""
        ctx = _document_ctx(monkeypatch, [
            (2, "Earlier paragraph.", 7),
            (4, "Following paragraph.", 8),
        ])
        snippet = _neighbour_snippet(ctx, "Earlier paragraph.")

        assert snippet.anchor == 0
        assert _quote_of(ctx, snippet) == _rule(snippet.text)


# ---------------------------------------------------------------------------
# Snippets without surrounding context (step 5, I4)
# ---------------------------------------------------------------------------


class TestOtherSnippetsKeepZero:
    def test_keyword_and_vector_transcript_snippets(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 5, I4"""
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_around",
            MagicMock(return_value=[("window", 10.0, 20.0)]),
        )
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_by_keyword_or",
            MagicMock(return_value=[("keyword hit far away", 300.0, 310.0)]),
        )
        monkeypatch.setattr(
            "app.rag.context._fetch_transcript_chunks_by_vector",
            MagicMock(return_value=[("vector hit far away", 500.0, 510.0)]),
        )
        monkeypatch.setattr(
            "app.rag.context.filter_keywords", MagicMock(side_effect=lambda k: k),
        )
        ctx = build_file_context(
            _video(10.0, 20.0), RagConfig(transcript_vector_top_n=4),
            keywords="keyword", query_vector=MagicMock(),
        )

        extra = [s for s in ctx.snippets if s.source != "transcript"
                 and ("hit far away" in s.text)]
        assert len(extra) == 2
        assert [s.anchor for s in extra] == [0, 0]

    def test_image_caption_and_metadata_snippets(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 5, I4"""
        match = MatchInfo(match_type="clip", text="a sunset", score=0.7)
        image = RetrievedFile(
            file_id="i1", drive="Photos", filename="s.jpg", file_type="image",
            title="Title", description="Desc", score=0.9, match_types=("clip",),
            segments=(SegmentGroup(time_range=None, matches=(match,)),),
        )
        monkeypatch.setattr(
            "app.rag.context._fetch_blip_caption",
            MagicMock(return_value="A serene sunset."),
        )
        ctx = build_file_context(image, RagConfig())

        assert ctx.snippets
        assert all(s.anchor == 0 for s in ctx.snippets)


# ---------------------------------------------------------------------------
# _quote_from_contexts (step 7, I4, I5, I9)
# ---------------------------------------------------------------------------


class TestQuoteFromAnchor:
    def test_quote_starts_at_anchor(self) -> None:
        """SPEC-ADDON-037 step 7"""
        text = "lead-in words. cited words"
        ctx = _ctx([ContextSnippet("transcript", text, "1:00", anchor=text.index("cited"))])
        assert _quote_from_contexts("f", [ctx]) == "cited words"

    def test_leading_whitespace_after_anchor_is_stripped(self) -> None:
        """SPEC-ADDON-037 step 7, I9"""
        ctx = _ctx([ContextSnippet("transcript", "lead\n\n  cited  ", "1:00", anchor=4)])
        assert _quote_from_contexts("f", [ctx]) == "cited"

    def test_anchor_at_end_means_lost_and_quote_is_empty(self) -> None:
        """SPEC-ADDON-037 step 7, I6"""
        text = "only lead-in text"
        ctx = _ctx([ContextSnippet("transcript", text, "1:00", anchor=len(text))])
        assert _quote_from_contexts("f", [ctx]) == ""

    def test_whitespace_only_remainder_is_empty(self) -> None:
        """SPEC-ADDON-037 section 2 failure case"""
        ctx = _ctx([ContextSnippet("transcript", "lead   \n ", "1:00", anchor=4)])
        assert _quote_from_contexts("f", [ctx]) == ""

    def test_cap_applies_to_the_text_after_the_anchor(self) -> None:
        """SPEC-ADDON-037 step 7, I9"""
        text = "L" * 50 + "c" * 250
        ctx = _ctx([ContextSnippet("transcript", text, "1:00", anchor=50)])
        assert _quote_from_contexts("f", [ctx]) == "c" * 200 + "…"

    def test_whitespace_at_the_cap_is_not_kept_before_the_ellipsis(self) -> None:
        """SPEC-ADDON-037 I9"""
        text = "L" * 5 + "c" * 199 + " " + "d" * 10
        ctx = _ctx([ContextSnippet("transcript", text, "1:00", anchor=5)])
        assert _quote_from_contexts("f", [ctx]) == "c" * 199 + "…"

    def test_exactly_the_cap_has_no_ellipsis(self) -> None:
        """SPEC-ADDON-037 I9 boundary"""
        text = "L" * 5 + "c" * 200
        ctx = _ctx([ContextSnippet("transcript", text, "1:00", anchor=5)])
        assert _quote_from_contexts("f", [ctx]) == "c" * 200

    def test_quote_never_contains_text_before_the_anchor(self) -> None:
        """SPEC-ADDON-037 I5"""
        text = "SECRETLEAD cited"
        ctx = _ctx([ContextSnippet("transcript", text, "1:00", anchor=11)])
        assert "SECRETLEAD" not in _quote_from_contexts("f", [ctx])

    def test_snippet_choice_is_unchanged_and_each_uses_its_own_anchor(self) -> None:
        """SPEC-ADDON-037 step 7, I7"""
        ctx = _ctx([
            ContextSnippet("transcript", "aaa first", "0:15", anchor=4),
            ContextSnippet("transcript", "bbb second", "1:30", anchor=4),
            ContextSnippet("transcript", "ccc third", "3:00", anchor=4),
        ])
        assert _quote_from_contexts("f", [ctx], location="1:30") == "second"
        assert _quote_from_contexts("f", [ctx], location="3:00") == "third"
        assert _quote_from_contexts("f", [ctx]) == "first"
        assert _quote_from_contexts("f", [ctx], location="99:99") == "first"

    def test_whole_token_location_match_still_wins(self) -> None:
        """SPEC-ADDON-037 step 7, I7"""
        ctx = _ctx([
            ContextSnippet("transcript", "xx later", "10:45", anchor=3),
            ContextSnippet("transcript", "yy embedded", "transcript @ 0:45", anchor=3),
        ])
        assert _quote_from_contexts("f", [ctx], location="0:45") == "embedded"


# ---------------------------------------------------------------------------
# Per-file budget (step 6, I6)
# ---------------------------------------------------------------------------


def _clip_chunks():
    lead = _sentences("A", 10)
    target = "TARGETSTART" + _sentences("T", 10)
    tail = _sentences("B", 10)
    return [
        (lead, 40.0, 90.0),
        (target, 100.0, 130.0),
        (tail, 130.0, 160.0),
    ]


class TestBudgetClip:
    @pytest.mark.parametrize("budget", [0, len("ab。cited。mid")])
    def test_a_snippet_that_fits_is_kept_whole_with_its_anchor(self, budget) -> None:
        """SPEC-ADDON-037 step 6 (kept whole): no budget, and an exact fit"""
        snippet = ContextSnippet("transcript", "ab。cited。mid", "1:00", anchor=3)

        kept, _total = _cap_snippets([snippet], budget)

        assert [(s.text, s.anchor) for s in kept] == [("ab。cited。mid", 3)]

    def test_whole_snippet_keeps_its_anchor(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 6 (kept whole)"""
        ctx = _build_window(monkeypatch, _clip_chunks(), budget=100000)
        snippet = _window_snippet(ctx)

        assert snippet.text[snippet.anchor:].startswith("TARGETSTART")
        assert snippet.anchor == len(_sentences("A", 10)) + 1

    def test_passage_that_survives_the_clip_is_still_the_quote_start(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 6, I6"""
        ctx = _build_window(monkeypatch, _clip_chunks(), budget=700)
        snippet = _window_snippet(ctx)

        assert len(snippet.text) < len(_sentences("A", 10)) * 3
        assert 0 < snippet.anchor < len(snippet.text)
        assert snippet.text[snippet.anchor:].startswith("TARGETSTART")
        assert _quote_of(ctx, snippet) == _rule(snippet.text[snippet.anchor:])
        assert _quote_of(ctx, snippet).startswith("TARGETSTART")

    def test_lost_passage_has_anchor_at_end_and_empty_quote(self, monkeypatch) -> None:
        """SPEC-ADDON-037 step 6, step 7, I5, I6"""
        ctx = _build_window(monkeypatch, _clip_chunks(), budget=200)
        snippet = _window_snippet(ctx)

        assert "TARGETSTART" not in snippet.text
        assert snippet.anchor == len(snippet.text)
        assert _quote_of(ctx, snippet) == ""

    @pytest.mark.parametrize("budget", [150, 200, 300, 450, 600, 700, 900, 1100])
    def test_anchor_is_always_within_the_clipped_text(self, monkeypatch, budget) -> None:
        """SPEC-ADDON-037 step 6 (0 <= anchor <= len)"""
        ctx = _build_window(monkeypatch, _clip_chunks(), budget=budget)
        snippet = _window_snippet(ctx)

        assert isinstance(snippet.anchor, int)
        assert 0 <= snippet.anchor <= len(snippet.text)
        if "TARGETSTART" in snippet.text:
            assert snippet.text[snippet.anchor:].startswith("TARGETSTART")
        else:
            assert snippet.anchor == len(snippet.text)

    def test_clip_that_starts_inside_the_passage_anchors_at_the_clipped_start(
        self, monkeypatch,
    ) -> None:
        """SPEC-ADDON-037 step 6 (removed > anchor), I6"""
        chunks = [
            ("cut。" + _sentences("T", 10), 100.0, 130.0),
            (_sentences("B", 10), 130.0, 160.0),
        ]
        ctx = _build_window(monkeypatch, chunks, budget=300)
        snippet = _window_snippet(ctx)

        assert snippet.anchor == 0
        assert _quote_of(ctx, snippet) == _rule(snippet.text)

    def test_leading_fragment_cut_shifts_the_anchor_by_what_was_removed(
        self, monkeypatch,
    ) -> None:
        """SPEC-ADDON-037 step 6 (anchor - removed): the clipped text starts
        after the leading fragment, and the anchor still points at the passage."""
        chunks = [
            ("fragment。" + _sentences("A", 9), 40.0, 90.0),
            ("TARGETSTART" + _sentences("T", 10), 100.0, 130.0),
        ]
        ctx = _build_window(monkeypatch, chunks, budget=700)
        snippet = _window_snippet(ctx)

        assert not snippet.text.startswith("fragment")
        assert snippet.text[snippet.anchor:].startswith("TARGETSTART")
        assert snippet.anchor == snippet.text.index("TARGETSTART")


# ---------------------------------------------------------------------------
# Prompt unchanged (I3, section 5)
# ---------------------------------------------------------------------------


class TestPromptUnchanged:
    def test_anchor_is_not_rendered(self) -> None:
        """SPEC-ADDON-037 I3"""
        plain = _ctx([ContextSnippet("transcript", "lead cited", "1:00")])
        anchored = _ctx([ContextSnippet("transcript", "lead cited", "1:00", anchor=5)])

        assert build_user_prompt("q", [anchored]) == build_user_prompt("q", [plain])


# ---------------------------------------------------------------------------
# Whole pipeline: _to_citation_dict (step 8, 9, I7, I8)
# ---------------------------------------------------------------------------


@pytest.fixture()
def rag_on(monkeypatch, make_settings):
    settings = make_settings(
        features=type(make_settings().features)(
            indexing=True, search=True, auto_tags="false", summaries="false", rag=True,
        ),
        llm=LLMConfig(
            provider="openai_compatible", base_url="http://localhost:11434/v1", model="llama3",
        ),
        rag=RagConfig(hierarchical=HierarchicalRagConfig(enabled=False)),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.rag.service.settings", settings)


def _two_match_video() -> RetrievedFile:
    futon = MatchInfo(
        match_type="transcript", text="futon dryer passage", score=0.8,
        timestamp_start=1624.0, timestamp_end=1680.0,
    )
    heater = MatchInfo(
        match_type="transcript", text="panel heater passage", score=0.8,
        timestamp_start=1682.5, timestamp_end=1738.3,
    )
    return RetrievedFile(
        file_id="f1", drive="Videos", filename="f1.mp4", file_type="video",
        title=None, description=None, score=0.9, match_types=("transcript",),
        segments=(
            SegmentGroup(time_range=(1624.0, 1680.0), matches=(futon,)),
            SegmentGroup(time_range=(1682.5, 1738.3), matches=(heater,)),
        ),
    )


async def _ask(monkeypatch, snippet: ContextSnippet, citation: dict) -> dict:
    ctx = _ctx([snippet], file_id="f1")
    monkeypatch.setattr(
        "app.rag.service.retrieve_candidates",
        AsyncMock(return_value=[_two_match_video()]),
    )
    monkeypatch.setattr(
        "app.rag.service.assemble_contexts", lambda cands, cfg, **_kw: [ctx],
    )
    llm = MagicMock()
    llm.enabled = True
    llm.generate_json = AsyncMock(return_value={
        "answer": "a [1]", "citations": [{"file_id": "f1", "relevance": 0.9, **citation}],
    })
    bind_llm(monkeypatch, lambda: llm)
    result = await answer_question(query="q", credential="t")
    assert len(result.citations) == 1
    return result.citations[0]


class TestCitationDict:
    @pytest.mark.asyncio
    async def test_accepted_location_keeps_seek_target_and_gets_anchored_quote(
        self, monkeypatch, rag_on,
    ) -> None:
        """SPEC-ADDON-037 step 9, I7"""
        text = "futon dryer passage panel heater passage"
        snippet = ContextSnippet(
            "transcript", text, "28:02", anchor=text.index("panel"),
        )
        cit = await _ask(monkeypatch, snippet, {"location": "28:02"})

        assert cit["quote"] == "panel heater passage"
        assert cit["segment_location"] == "28:02"

    @pytest.mark.asyncio
    async def test_unaccepted_location_is_chosen_from_the_cited_passage(
        self, monkeypatch, rag_on,
    ) -> None:
        """SPEC-ADDON-037 step 9: seek target chosen from the anchored quote"""
        text = "futon dryer passage panel heater passage"
        snippet = ContextSnippet(
            "transcript", text, "28:02", anchor=text.index("panel"),
        )
        cit = await _ask(monkeypatch, snippet, {"location": "the heater part"})

        assert cit["quote"] == "panel heater passage"
        assert cit["segment_location"] == "28:02"

    @pytest.mark.asyncio
    async def test_lost_passage_unaccepted_location_falls_back_to_first_match(
        self, monkeypatch, rag_on,
    ) -> None:
        """SPEC-ADDON-037 step 9, section 9"""
        text = "panel heater passage"
        snippet = ContextSnippet("transcript", text, "28:02", anchor=len(text))
        cit = await _ask(monkeypatch, snippet, {"location": "the heater part"})

        assert cit["quote"] == ""
        assert cit["segment_location"] == "27:04"

    @pytest.mark.asyncio
    async def test_lost_passage_accepted_location_keeps_seek_target(
        self, monkeypatch, rag_on,
    ) -> None:
        """SPEC-ADDON-037 step 9, section 9"""
        text = "panel heater passage"
        snippet = ContextSnippet("transcript", text, "28:02", anchor=len(text))
        cit = await _ask(monkeypatch, snippet, {"location": "28:02"})

        assert cit["quote"] == ""
        assert cit["segment_location"] == "28:02"

    @pytest.mark.asyncio
    async def test_llm_supplied_quote_is_kept(self, monkeypatch, rag_on) -> None:
        """SPEC-ADDON-037 step 8, I8"""
        text = "futon dryer passage panel heater passage"
        snippet = ContextSnippet(
            "transcript", text, "28:02", anchor=text.index("panel"),
        )
        cit = await _ask(
            monkeypatch, snippet, {"location": "28:02", "quote": "the model's own words"},
        )

        assert cit["quote"] == "the model's own words"
