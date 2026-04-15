"""Tests for the words-based chunk builder in the Whisper worker.

Exercises ``_build_chunks_from_words`` against the invariants the
search pipeline relies on: bounded duration, sentence-boundary cuts,
no empty chunks, and monotonic timestamps. Replaces the implicit
coverage the old ``_merge_segments`` / ``_split_long_segments`` pair
used to carry (ref hako qx19g-IBnLc7_C-WBo-rf).
"""

from app.workers.whisper import (
    _build_chunks_from_words,
    _flatten_words,
    _join_words,
)


def _word(text: str, start: float, end: float, language: str = "en") -> dict:
    return {"text": text, "start": start, "end": end, "language": language}


def test_empty_words_returns_empty():
    assert _build_chunks_from_words([], 10, 30) == []


def test_below_min_duration_produces_single_chunk():
    words = [_word("hello", 0.0, 0.5), _word("world.", 0.5, 1.0)]
    chunks = _build_chunks_from_words(words, min_duration=10, max_duration=30)
    assert len(chunks) == 1
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 1.0
    assert chunks[0]["text"] == "hello world."


def test_sentence_break_flushes_once_past_min():
    words = [
        _word("one", 0.0, 1.0),
        _word("two", 1.0, 2.0),
        _word("three.", 2.0, 11.0),  # terminal punctuation past min
        _word("four", 11.5, 12.5),
        _word("five.", 12.5, 13.5),
    ]
    chunks = _build_chunks_from_words(words, min_duration=10, max_duration=30)
    assert len(chunks) == 2
    assert chunks[0]["text"].endswith("three.")
    assert chunks[0]["end"] == 11.0
    assert chunks[1]["start"] == 11.5


def test_silence_gap_treated_as_break():
    words = [
        _word("alpha", 0.0, 1.0),
        _word("beta", 1.0, 10.5),
        # Gap 1.2 s > 0.4 s threshold.
        _word("gamma", 11.7, 12.7),
    ]
    chunks = _build_chunks_from_words(words, min_duration=10, max_duration=30)
    assert len(chunks) == 2
    assert chunks[0]["end"] == 10.5
    assert chunks[1]["start"] == 11.7


def test_max_duration_enforced_even_without_break():
    # No punctuation, no pauses — pure run-on speech.
    words = [_word(f"w{i}", i * 0.5, (i + 1) * 0.5) for i in range(80)]
    chunks = _build_chunks_from_words(words, min_duration=10, max_duration=15)
    assert chunks, "must produce at least one chunk"
    for c in chunks:
        assert c["end"] - c["start"] <= 15.5, (
            f"chunk exceeded max: {c['start']:.2f}→{c['end']:.2f}"
        )


def test_single_long_word_past_max_still_emits():
    # Regression: old _merge_segments dropped the max check when the
    # accumulator was empty, so a single 40-s segment went through whole.
    words = [_word("mega", 0.0, 40.0)]
    chunks = _build_chunks_from_words(words, min_duration=10, max_duration=20)
    assert len(chunks) == 1
    # A single >max word can't be sub-split — we emit it as-is but never
    # silently hide it.
    assert chunks[0]["text"] == "mega"


def test_timestamps_are_monotonic():
    words = [_word(f"w{i}.", i * 2.0, i * 2.0 + 1.9) for i in range(20)]
    chunks = _build_chunks_from_words(words, min_duration=5, max_duration=12)
    last_end = -1.0
    for c in chunks:
        assert c["start"] >= last_end, (
            f"chunk {c} starts before previous end {last_end}"
        )
        assert c["end"] >= c["start"]
        last_end = c["end"]


def test_no_empty_chunks_emitted():
    words = [_word("hi.", 0.0, 0.3), _word("there.", 0.3, 0.6)]
    chunks = _build_chunks_from_words(words, min_duration=10, max_duration=30)
    for c in chunks:
        assert c["text"].strip()


def test_join_words_japanese_uses_no_separator():
    assert _join_words(["今日", "は", "良い", "天気"], "ja") == "今日は良い天気"


def test_join_words_english_uses_space():
    assert _join_words(["hello", "world"], "en") == "hello world"


def test_flatten_words_skips_segments_without_words():
    segments = [
        {"language": "en", "words": [{"word": " hello", "start": 0.0, "end": 0.5}]},
        {"language": "en"},  # no words key
        {"language": "en", "words": []},
        {"language": "en", "words": [{"word": " world", "start": 1.0, "end": 1.5}]},
    ]
    flat = _flatten_words(segments)
    assert [w["text"] for w in flat] == ["hello", "world"]
    assert flat[0]["start"] == 0.0
    assert flat[1]["start"] == 1.0
