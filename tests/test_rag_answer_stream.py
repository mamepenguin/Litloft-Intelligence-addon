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

from app.rag.answer_stream import AnswerStreamExtractor


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
