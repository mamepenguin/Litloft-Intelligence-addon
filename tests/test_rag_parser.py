"""Tests for app.rag.parser module.

parse_answer validates LLM JSON output against the expected schema
and enforces the key security invariant: citations referencing
file_ids not in the retrieved set are dropped (anti-hallucination).
"""

import sys
from unittest.mock import MagicMock

import pytest

# Stub out heavy dependencies pulled via the RAG package.
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

from app.rag.parser import Citation, ParsedAnswer, parse_answer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _allowed(*ids: str) -> frozenset[str]:
    return frozenset(ids)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestParseAnswerHappyPath:
    """Well-formed LLM responses are parsed into ParsedAnswer."""

    def test_parses_valid_json(self):
        """T1: normal JSON with answer + citations."""
        raw = {
            "answer": "The file discusses neural networks [1].",
            "citations": [
                {
                    "file_id": "f1",
                    "quote": "Neural networks learn from data.",
                    "relevance": 0.9,
                }
            ],
        }

        parsed = parse_answer(raw, _allowed("f1"))

        assert parsed is not None
        assert isinstance(parsed, ParsedAnswer)
        assert parsed.answer == "The file discusses neural networks [1]."
        assert len(parsed.citations) == 1
        cit = parsed.citations[0]
        assert isinstance(cit, Citation)
        assert cit.file_id == "f1"
        assert cit.quote == "Neural networks learn from data."
        assert cit.relevance == pytest.approx(0.9)

    def test_multiple_citations(self):
        raw = {
            "answer": "See [1] and [2].",
            "citations": [
                {"file_id": "f1", "quote": "Quote 1", "relevance": 0.9},
                {"file_id": "f2", "quote": "Quote 2", "relevance": 0.7},
            ],
        }

        parsed = parse_answer(raw, _allowed("f1", "f2"))

        assert parsed is not None
        assert len(parsed.citations) == 2
        ids = [c.file_id for c in parsed.citations]
        assert ids == ["f1", "f2"]


# ---------------------------------------------------------------------------
# Missing / malformed fields
# ---------------------------------------------------------------------------


class TestParseAnswerMissingFields:
    """Missing fields fall back to documented defaults."""

    def test_missing_answer_field_returns_none(self):
        """T2: no 'answer' key -> whole response is invalid -> None."""
        raw = {
            "citations": [
                {"file_id": "f1", "quote": "q", "relevance": 0.5}
            ]
        }

        parsed = parse_answer(raw, _allowed("f1"))

        assert parsed is None

    def test_missing_citations_field_empty_list(self):
        """T3: no 'citations' key -> ParsedAnswer with empty citations."""
        raw = {"answer": "Just an answer with no citations."}

        parsed = parse_answer(raw, _allowed("f1"))

        assert parsed is not None
        assert parsed.answer == "Just an answer with no citations."
        assert parsed.citations == ()

    def test_missing_quote_field_defaults_to_empty(self):
        """T8: no 'quote' on a citation -> empty string fallback."""
        raw = {
            "answer": "Answer text",
            "citations": [
                {"file_id": "f1", "relevance": 0.5}
            ],
        }

        parsed = parse_answer(raw, _allowed("f1"))

        assert parsed is not None
        assert len(parsed.citations) == 1
        assert parsed.citations[0].quote == ""

    def test_missing_relevance_defaults_to_zero(self):
        raw = {
            "answer": "Answer",
            "citations": [
                {"file_id": "f1", "quote": "Q"}
            ],
        }

        parsed = parse_answer(raw, _allowed("f1"))

        assert parsed is not None
        assert len(parsed.citations) == 1
        # Missing relevance should default to a known value (0.0 or similar).
        assert parsed.citations[0].relevance == pytest.approx(0.0) or \
            parsed.citations[0].relevance == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# file_id fabrication security
# ---------------------------------------------------------------------------


class TestParseAnswerFileIdFabrication:
    """hako RftwcVMgA0pWVBWMbN6An: drop citations for IDs not in the allowed set."""

    def test_drops_unknown_file_id_citations(self):
        """T4: citations referencing file_ids outside allowed_file_ids are dropped."""
        raw = {
            "answer": "The file talks about stuff.",
            "citations": [
                {"file_id": "real-1", "quote": "actual quote", "relevance": 0.8},
                {"file_id": "fake-1", "quote": "fabricated", "relevance": 0.9},
            ],
        }

        parsed = parse_answer(raw, _allowed("real-1"))

        assert parsed is not None
        assert parsed.answer == "The file talks about stuff."
        assert len(parsed.citations) == 1
        assert parsed.citations[0].file_id == "real-1"

    def test_all_citations_unknown_keeps_answer(self):
        """T5: if every citation is fabricated, keep answer but empty citations."""
        raw = {
            "answer": "Some answer",
            "citations": [
                {"file_id": "fake-1", "quote": "x", "relevance": 0.5},
                {"file_id": "fake-2", "quote": "y", "relevance": 0.5},
            ],
        }

        parsed = parse_answer(raw, _allowed("real-1", "real-2"))

        assert parsed is not None
        assert parsed.answer == "Some answer"
        assert parsed.citations == ()

    def test_empty_allowed_set_drops_all_citations(self):
        raw = {
            "answer": "Answer",
            "citations": [
                {"file_id": "f1", "quote": "q", "relevance": 0.5}
            ],
        }

        parsed = parse_answer(raw, frozenset())

        assert parsed is not None
        assert parsed.citations == ()


# ---------------------------------------------------------------------------
# relevance validation
# ---------------------------------------------------------------------------


class TestParseAnswerRelevance:
    """T6: out-of-range relevance values are clamped or dropped."""

    def test_relevance_above_one_is_clamped_or_dropped(self):
        raw = {
            "answer": "A",
            "citations": [
                {"file_id": "f1", "quote": "q", "relevance": 1.5}
            ],
        }

        parsed = parse_answer(raw, _allowed("f1"))

        assert parsed is not None
        # Either the citation was kept with relevance clamped to 1.0,
        # or it was dropped entirely. Both are acceptable per spec.
        if parsed.citations:
            assert parsed.citations[0].relevance <= 1.0

    def test_relevance_below_zero_is_clamped_or_dropped(self):
        raw = {
            "answer": "A",
            "citations": [
                {"file_id": "f1", "quote": "q", "relevance": -0.3}
            ],
        }

        parsed = parse_answer(raw, _allowed("f1"))

        assert parsed is not None
        if parsed.citations:
            assert parsed.citations[0].relevance >= 0.0

    def test_relevance_non_numeric_string_is_handled(self):
        """A string like 'high' is invalid — clamp to default or drop."""
        raw = {
            "answer": "A",
            "citations": [
                {"file_id": "f1", "quote": "q", "relevance": "high"}
            ],
        }

        # Must not raise.
        parsed = parse_answer(raw, _allowed("f1"))

        assert parsed is not None
        # Either dropped, or clamped to 0.0/default.
        if parsed.citations:
            assert 0.0 <= parsed.citations[0].relevance <= 1.0

    def test_relevance_none_is_handled(self):
        raw = {
            "answer": "A",
            "citations": [
                {"file_id": "f1", "quote": "q", "relevance": None}
            ],
        }

        parsed = parse_answer(raw, _allowed("f1"))

        assert parsed is not None


# ---------------------------------------------------------------------------
# Unparseable input
# ---------------------------------------------------------------------------


class TestParseAnswerUnparseable:
    """T7 + additional null/malformed handling."""

    def test_none_input_returns_none(self):
        """T7: LLM returned None (parse failure upstream) -> None."""
        assert parse_answer(None, _allowed("f1")) is None

    def test_list_input_returns_none(self):
        """Root-level list instead of dict -> invalid shape -> None."""
        raw = [
            {"answer": "wrong shape", "citations": []}
        ]

        # The parser accepts dict or list | None per signature, but
        # a list at the root is not the expected answer shape.
        result = parse_answer(raw, _allowed("f1"))
        assert result is None

    def test_empty_dict_returns_none(self):
        """Empty dict has no 'answer' field -> None."""
        assert parse_answer({}, _allowed("f1")) is None

    def test_answer_is_not_string_returns_none(self):
        raw = {"answer": 42, "citations": []}

        parsed = parse_answer(raw, _allowed("f1"))

        # Non-string answer should be rejected or coerced — the safer
        # behaviour per spec is to reject the whole response.
        assert parsed is None or isinstance(parsed.answer, str)

    def test_citations_not_a_list_treated_as_empty(self):
        """If citations is the wrong type, treat it as empty list."""
        raw = {"answer": "Answer", "citations": "not a list"}

        parsed = parse_answer(raw, _allowed("f1"))

        # Spec behaviour: keep answer, drop malformed citations.
        assert parsed is not None
        assert parsed.citations == ()

    def test_citation_entry_missing_file_id_is_dropped(self):
        raw = {
            "answer": "A",
            "citations": [
                {"quote": "no id", "relevance": 0.5},
                {"file_id": "f1", "quote": "has id", "relevance": 0.8},
            ],
        }

        parsed = parse_answer(raw, _allowed("f1"))

        assert parsed is not None
        # Only the valid citation survives.
        assert len(parsed.citations) == 1
        assert parsed.citations[0].file_id == "f1"

    def test_citation_entry_not_a_dict_is_dropped(self):
        raw = {
            "answer": "A",
            "citations": [
                "just a string",
                {"file_id": "f1", "quote": "ok", "relevance": 0.5},
            ],
        }

        parsed = parse_answer(raw, _allowed("f1"))

        assert parsed is not None
        assert len(parsed.citations) == 1
        assert parsed.citations[0].file_id == "f1"


# ---------------------------------------------------------------------------
# Citation / ParsedAnswer dataclasses
# ---------------------------------------------------------------------------


class TestCitationDataclass:
    def test_fields(self):
        cit = Citation(file_id="f1", quote="hello", relevance=0.7)
        assert cit.file_id == "f1"
        assert cit.quote == "hello"
        assert cit.relevance == 0.7

    def test_is_frozen(self):
        cit = Citation(file_id="f1", quote="q", relevance=0.5)
        with pytest.raises(Exception):
            cit.file_id = "changed"  # type: ignore[misc]


class TestParsedAnswerDataclass:
    def test_fields(self):
        pa = ParsedAnswer(
            answer="text",
            citations=(Citation(file_id="f1", quote="q", relevance=0.5),),
        )
        assert pa.answer == "text"
        assert len(pa.citations) == 1

    def test_is_frozen(self):
        pa = ParsedAnswer(answer="t", citations=())
        with pytest.raises(Exception):
            pa.answer = "changed"  # type: ignore[misc]


class TestCitationDeduplication:
    """Local LLMs often repeat the same (file_id, location) when
    uncertain. The parser drops duplicates so the UI shows each
    source once.
    """

    def test_exact_duplicate_citations_collapse(self):
        raw = {
            "answer": "Here is the answer.",
            "citations": [
                {"file_id": "abc", "location": "0:15"},
                {"file_id": "abc", "location": "0:15"},
                {"file_id": "abc", "location": "0:15"},
            ],
        }

        result = parse_answer(raw, frozenset({"abc"}))

        assert result is not None
        assert len(result.citations) == 1
        assert result.citations[0].file_id == "abc"
        assert result.citations[0].location == "0:15"

    def test_distinct_locations_preserved(self):
        raw = {
            "answer": "Answer.",
            "citations": [
                {"file_id": "abc", "location": "0:15"},
                {"file_id": "abc", "location": "1:30"},
                {"file_id": "abc", "location": "1:30"},  # dup of #2
                {"file_id": "abc", "location": "3:00"},
            ],
        }

        result = parse_answer(raw, frozenset({"abc"}))

        assert result is not None
        assert len(result.citations) == 3
        locations = [c.location for c in result.citations]
        assert locations == ["0:15", "1:30", "3:00"]

    def test_first_occurrence_wins(self):
        """When duplicates differ in non-key fields, the first wins."""
        raw = {
            "answer": "Answer.",
            "citations": [
                {
                    "file_id": "abc",
                    "location": "0:15",
                    "quote": "first quote",
                    "relevance": 0.9,
                },
                {
                    "file_id": "abc",
                    "location": "0:15",
                    "quote": "second quote",
                    "relevance": 0.3,
                },
            ],
        }

        result = parse_answer(raw, frozenset({"abc"}))

        assert result is not None
        assert len(result.citations) == 1
        assert result.citations[0].quote == "first quote"
        assert result.citations[0].relevance == 0.9
