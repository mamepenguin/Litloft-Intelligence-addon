"""Unit tests for AnswerStreamExtractor.

The extractor is the piece that stops the UI from seeing raw JSON
syntax scroll past during RAG answer streaming. It parses the
incoming chunks character-by-character and emits only the decoded
``answer`` field value, with prose fallback when the model ignores
the JSON instruction entirely and code-fence stripping when the
model wraps its output in ``\\`\\`\\`json``.

These tests cover:

* The happy path (single-chunk JSON).
* Chunks that split at every "dangerous" boundary: mid-key,
  mid-escape, mid-\\uXXXX, mid-closing-quote.
* Escape decoding (``\\n``, ``\\"``, ``\\u3042``).
* Prose fallback when the first char is not ``{``.
* Code-fenced JSON handling (``\\`\\`\\`json`` preamble).
* Truncated answer strings flushed via ``finalize``.
"""

from __future__ import annotations

from app.rag.answer_stream import AnswerStreamExtractor, CitationStreamExtractor


def _feed_all(ex: AnswerStreamExtractor, chunks: list[str]) -> str:
    out = "".join(ex.feed(c) for c in chunks)
    out = out + ex.finalize()
    return out


class TestSingleChunk:
    def test_plain_answer(self):
        ex = AnswerStreamExtractor()
        result = _feed_all(
            ex,
            ['{"answer": "hello world", "citations": []}'],
        )
        assert result == "hello world"

    def test_unicode_answer(self):
        ex = AnswerStreamExtractor()
        result = _feed_all(
            ex,
            ['{"answer": "京都の紅葉", "citations": []}'],
        )
        assert result == "京都の紅葉"

    def test_citation_markers_in_answer(self):
        ex = AnswerStreamExtractor()
        result = _feed_all(
            ex,
            ['{"answer": "see [1] and [2]", "citations": []}'],
        )
        assert result == "see [1] and [2]"


class TestMultiChunkSplits:
    def test_split_inside_key(self):
        ex = AnswerStreamExtractor()
        result = _feed_all(
            ex,
            ['{"ans', 'wer": "text", "citations": []}'],
        )
        assert result == "text"

    def test_split_inside_value(self):
        ex = AnswerStreamExtractor()
        result = _feed_all(
            ex,
            ['{"answer": "hel', 'lo", "citations": []}'],
        )
        assert result == "hello"

    def test_split_at_every_character(self):
        src = '{"answer": "hi", "citations": []}'
        ex = AnswerStreamExtractor()
        result = _feed_all(ex, list(src))
        assert result == "hi"


class TestEscapeDecoding:
    def test_newline_escape(self):
        ex = AnswerStreamExtractor()
        result = _feed_all(
            ex,
            ['{"answer": "line1\\nline2", "citations": []}'],
        )
        assert result == "line1\nline2"

    def test_embedded_quote_escape(self):
        ex = AnswerStreamExtractor()
        result = _feed_all(
            ex,
            ['{"answer": "he said \\"hi\\"", "citations": []}'],
        )
        assert result == 'he said "hi"'

    def test_unicode_escape(self):
        ex = AnswerStreamExtractor()
        # \u3042 = あ
        result = _feed_all(
            ex,
            ['{"answer": "\\u3042", "citations": []}'],
        )
        assert result == "あ"

    def test_backslash_split_across_chunks(self):
        ex = AnswerStreamExtractor()
        # Split so the backslash is the last char of chunk 1 and the
        # 'n' is the first char of chunk 2. The extractor must hold
        # the backslash until the escape byte arrives.
        result = _feed_all(
            ex,
            ['{"answer": "a\\', 'nb", "citations": []}'],
        )
        assert result == "a\nb"

    def test_unicode_escape_split_across_chunks(self):
        ex = AnswerStreamExtractor()
        result = _feed_all(
            ex,
            ['{"answer": "\\u30', '42", "citations": []}'],
        )
        assert result == "あ"


class TestProseFallback:
    def test_bare_prose_is_forwarded(self):
        ex = AnswerStreamExtractor()
        result = _feed_all(ex, ["this is not json at all"])
        assert result == "this is not json at all"

    def test_leading_whitespace_then_json(self):
        ex = AnswerStreamExtractor()
        result = _feed_all(
            ex,
            ['  \n  {"answer": "ok", "citations": []}'],
        )
        assert result == "ok"

    def test_leading_whitespace_then_prose(self):
        ex = AnswerStreamExtractor()
        result = _feed_all(ex, ["  hello world"])
        assert result == "  hello world"

    def test_prose_preamble_before_json_falls_to_prose(self):
        """Prose we cannot safely delete triggers prose fallback.

        When the model says something like ``Here is: {...}``, we
        cannot silently strip the "Here is: " part without losing
        user-visible content. The extractor commits to prose mode
        instead so the UI still sees the whole response.
        """
        ex = AnswerStreamExtractor()
        src = 'Here is: {"answer": "hi", "citations": []}'
        result = _feed_all(ex, [src])
        assert result == src


class TestCodeFencedJson:
    """LLMs commonly wrap JSON output in ``\\`\\`\\`json ... \\`\\`\\``` —
    the extractor must treat the fence as a safe-to-skip preamble so
    the user sees the decoded answer text, not the raw JSON."""

    def test_bare_code_fence(self):
        ex = AnswerStreamExtractor()
        src = '```\n{"answer": "hi", "citations": []}\n```'
        result = _feed_all(ex, [src])
        assert result == "hi"

    def test_code_fence_with_language_tag(self):
        ex = AnswerStreamExtractor()
        src = '```json\n{"answer": "hi", "citations": []}\n```'
        result = _feed_all(ex, [src])
        assert result == "hi"

    def test_code_fence_with_leading_whitespace(self):
        ex = AnswerStreamExtractor()
        src = '  ```json\n{"answer": "hi", "citations": []}\n```'
        result = _feed_all(ex, [src])
        assert result == "hi"

    def test_code_fence_split_across_chunks(self):
        """Fence arrives in one chunk, opening brace in the next."""
        ex = AnswerStreamExtractor()
        result = _feed_all(
            ex,
            ["```json\n", '{"answer": "hi", "citations": []}\n```'],
        )
        assert result == "hi"

    def test_code_fence_and_answer_in_same_chunk(self):
        ex = AnswerStreamExtractor()
        src = '```json\n{"answer": "京都の紅葉 [1]", "citations": []}\n```'
        result = _feed_all(ex, [src])
        assert result == "京都の紅葉 [1]"


class TestFinalize:
    def test_truncated_answer_total_output(self):
        """Mid-string truncation: feed+finalize still reproduce the prefix."""
        ex = AnswerStreamExtractor()
        # No closing quote — the model hit max_tokens mid-sentence.
        emitted = ex.feed('{"answer": "partial')
        flushed = ex.finalize()
        assert emitted + flushed == "partial"

    def test_finalize_after_done_is_noop(self):
        ex = AnswerStreamExtractor()
        ex.feed('{"answer": "done"')
        assert ex.finalize() == ""

    def test_finalize_drops_incomplete_trailing_escape(self):
        ex = AnswerStreamExtractor()
        emitted = ex.feed('{"answer": "a\\')
        flushed = ex.finalize()
        # The backslash was never completed; drop it instead of
        # emitting a raw backslash that would look like malformed text.
        assert emitted + flushed == "a"

    def test_finalize_in_search_state_is_empty(self):
        """Model emitted `{` then truncated before the answer key."""
        ex = AnswerStreamExtractor()
        ex.feed("{")
        assert ex.finalize() == ""


class TestIdempotency:
    def test_empty_chunks_are_noops(self):
        ex = AnswerStreamExtractor()
        assert ex.feed("") == ""
        result = _feed_all(ex, ['{"answer": "x", "citations": []}'])
        assert result == "x"

    def test_post_done_chunks_are_discarded(self):
        """Citations array tokens arriving after the answer are ignored."""
        ex = AnswerStreamExtractor()
        assert ex.feed('{"answer": "x"') == "x"
        # The real stream continues with `, "citations": [...]}` — we
        # must not accidentally leak any of that into a delta.
        assert ex.feed(', "citations": [{"file_id": "f1"}]}') == ""
        assert ex.finalize() == ""


# ---------------------------------------------------------------------------
# CitationStreamExtractor
#
# This extractor is a sibling of ``AnswerStreamExtractor`` but focused on
# the ``citations`` array. It is fed the SAME raw chunks (so callers do
# not need to coordinate two state machines about where the answer
# string ends) and yields each completed citation object as a parsed
# dict, one at a time, as soon as its closing ``}`` arrives. This is
# what lets the UI render citation cards progressively instead of
# waiting for the whole JSON to close.
#
# Choosing "feed the extractor every chunk" (same data as the answer
# extractor) rather than "hand off chunks once the answer extractor
# reports done" keeps the two classes independent of each other's
# internal timing and makes unit testing trivial: we just feed a
# string and assert on the returned list. The extractor has its own
# small state machine that skips past the answer-field region
# deterministically.
# ---------------------------------------------------------------------------


def _feed_citations_all(
    ex: CitationStreamExtractor, chunks: list[str]
) -> list[dict]:
    """Feed all chunks then finalize, returning the concatenated output."""
    out: list[dict] = []
    for c in chunks:
        out = [*out, *ex.feed(c)]
    out = [*out, *ex.finalize()]
    return out


class TestCitationStreamExtractorHappyPath:
    def test_emits_each_citation_after_its_closing_brace(self):
        ex = CitationStreamExtractor()
        src = (
            '{"answer": "text", "citations": ['
            '{"file_id": "f1", "quote": "q1", "relevance": 0.9},'
            '{"file_id": "f2", "quote": "q2", "relevance": 0.8}'
            "]}"
        )
        citations = _feed_citations_all(ex, [src])
        assert [c["file_id"] for c in citations] == ["f1", "f2"]
        assert citations[0]["quote"] == "q1"
        assert citations[1]["relevance"] == 0.8

    def test_emits_citation_progressively_not_all_at_end(self):
        """Each citation should be returned as soon as its closing ``}`` arrives."""
        ex = CitationStreamExtractor()
        # Feed up to just before the first citation's closing brace.
        first = ex.feed(
            '{"answer": "text", "citations": ['
            '{"file_id": "f1", "quote": "q1", "relevance": 0.9'
        )
        # Object isn't closed yet — nothing to yield.
        assert first == []
        # Now feed the closing brace: the object completes and f1 emits.
        second = ex.feed("}")
        assert [c["file_id"] for c in second] == ["f1"]
        # Feed second citation; it emits as soon as its ``}`` arrives.
        third = ex.feed(
            ',{"file_id": "f2", "quote": "q2", "relevance": 0.8}'
        )
        assert [c["file_id"] for c in third] == ["f2"]
        # Closing ``]`` of the array — nothing more to yield.
        fourth = ex.feed("]}")
        assert fourth == []
        # Finalize returns nothing — we've already emitted everything.
        assert ex.finalize() == []

    def test_empty_citations_array(self):
        ex = CitationStreamExtractor()
        src = '{"answer": "text", "citations": []}'
        assert _feed_citations_all(ex, [src]) == []


class TestCitationStreamExtractorSplits:
    def test_split_inside_citation_object(self):
        ex = CitationStreamExtractor()
        chunks = [
            '{"answer": "a", "citations": [{"file_id": "f1",',
            ' "quote": "hello", "relevance": 0.5}]}',
        ]
        citations = _feed_citations_all(ex, chunks)
        assert [c["file_id"] for c in citations] == ["f1"]
        assert citations[0]["quote"] == "hello"

    def test_split_between_citations(self):
        ex = CitationStreamExtractor()
        chunks = [
            '{"answer": "a", "citations": [{"file_id": "f1", "quote": "q1", "relevance": 0.1}',
            ',{"file_id": "f2", "quote": "q2", "relevance": 0.2}]}',
        ]
        citations = _feed_citations_all(ex, chunks)
        assert [c["file_id"] for c in citations] == ["f1", "f2"]

    def test_split_at_every_character(self):
        """Character-by-character streaming still produces the right dicts."""
        ex = CitationStreamExtractor()
        src = (
            '{"answer": "x", "citations": ['
            '{"file_id": "f1", "quote": "q1", "relevance": 0.9},'
            '{"file_id": "f2", "quote": "q2", "relevance": 0.8}'
            "]}"
        )
        citations = _feed_citations_all(ex, list(src))
        assert [c["file_id"] for c in citations] == ["f1", "f2"]


class TestCitationStreamExtractorQuoteContent:
    def test_brace_inside_quote_is_not_object_boundary(self):
        """A ``{`` inside a string value must not be mistaken for an object start."""
        ex = CitationStreamExtractor()
        src = (
            '{"answer": "a", "citations": ['
            '{"file_id": "f1", "quote": "code {block} here", "relevance": 0.9}'
            "]}"
        )
        citations = _feed_citations_all(ex, [src])
        assert len(citations) == 1
        assert citations[0]["quote"] == "code {block} here"

    def test_escaped_quote_inside_string(self):
        """An escaped ``\\"`` must not end the string early."""
        ex = CitationStreamExtractor()
        src = (
            '{"answer": "a", "citations": ['
            '{"file_id": "f1", "quote": "he said \\"hi\\" loud", "relevance": 0.9}'
            "]}"
        )
        citations = _feed_citations_all(ex, [src])
        assert len(citations) == 1
        assert citations[0]["quote"] == 'he said "hi" loud'

    def test_escaped_backslash_then_quote(self):
        """``\\\\"`` (escaped backslash followed by real closing quote) terminates the string."""
        ex = CitationStreamExtractor()
        src = (
            '{"answer": "a", "citations": ['
            '{"file_id": "f1", "quote": "path\\\\", "relevance": 0.9}'
            "]}"
        )
        citations = _feed_citations_all(ex, [src])
        assert len(citations) == 1
        assert citations[0]["quote"] == "path\\"

    def test_nested_object_in_value(self):
        """Future-proofing: nested sub-object inside a citation is tracked correctly."""
        ex = CitationStreamExtractor()
        src = (
            '{"answer": "a", "citations": ['
            '{"file_id": "f1", "meta": {"k": "v"}, "relevance": 0.9}'
            "]}"
        )
        citations = _feed_citations_all(ex, [src])
        assert len(citations) == 1
        assert citations[0]["file_id"] == "f1"
        assert citations[0]["meta"] == {"k": "v"}


class TestCitationStreamExtractorEdgeCases:
    def test_non_array_citations_emits_nothing(self):
        ex = CitationStreamExtractor()
        src = '{"answer": "a", "citations": null}'
        assert _feed_citations_all(ex, [src]) == []

    def test_missing_citations_key_emits_nothing(self):
        ex = CitationStreamExtractor()
        src = '{"answer": "a"}'
        assert _feed_citations_all(ex, [src]) == []

    def test_prose_mode_emits_nothing(self):
        """Bare prose (no JSON at all) means no citations to extract."""
        ex = CitationStreamExtractor()
        assert _feed_citations_all(ex, ["this is not json at all"]) == []

    def test_incomplete_trailing_object_is_dropped(self):
        """A half-written citation object at stream end is NOT emitted."""
        ex = CitationStreamExtractor()
        chunks = [
            '{"answer": "a", "citations": [{"file_id": "f1", "quote": "q1", "relevance": 0.9},{"file_',
        ]
        # First citation completes; second is mid-write at stream end.
        citations = _feed_citations_all(ex, chunks)
        assert [c["file_id"] for c in citations] == ["f1"]

    def test_malformed_object_is_skipped_silently(self):
        """A ``{... }`` that fails json.loads must not crash the extractor."""
        ex = CitationStreamExtractor()
        src = (
            '{"answer": "a", "citations": ['
            '{not valid json},'
            '{"file_id": "f2", "quote": "q2", "relevance": 0.8}'
            "]}"
        )
        citations = _feed_citations_all(ex, [src])
        # The first garbage object is dropped silently; the second survives.
        assert [c["file_id"] for c in citations] == ["f2"]

    def test_code_fenced_json_works(self):
        """Code-fenced output still yields citations (fence is discarded)."""
        ex = CitationStreamExtractor()
        src = (
            "```json\n"
            '{"answer": "a", "citations": ['
            '{"file_id": "f1", "quote": "q", "relevance": 0.9}'
            "]}\n```"
        )
        citations = _feed_citations_all(ex, [src])
        assert [c["file_id"] for c in citations] == ["f1"]


class TestCitationStreamExtractorIdempotency:
    def test_empty_chunk_is_noop(self):
        ex = CitationStreamExtractor()
        assert ex.feed("") == []

    def test_post_done_chunks_are_noops(self):
        ex = CitationStreamExtractor()
        src = (
            '{"answer": "a", "citations": ['
            '{"file_id": "f1", "quote": "q", "relevance": 0.9}'
            "]}"
        )
        first_pass = ex.feed(src)
        assert [c["file_id"] for c in first_pass] == ["f1"]
        # Trailing whitespace / garbage after array close — ignored.
        assert ex.feed("  \n  garbage") == []
        assert ex.finalize() == []
