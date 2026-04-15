"""Tests for the WebVTT packer (words → cues).

Keeps the packer behaviour bounded: no cue exceeds ``max_duration`` or
``max_width``, cues stay monotonic, east-asian width is counted as 2,
and the serialised document starts with a valid WEBVTT header.
"""

from app.subtitle_builder import (
    CueConfig,
    _display_width,
    build_cues,
    build_vtt,
    to_webvtt,
)


def _word(text: str, start: float, end: float) -> dict:
    return {"text": text, "timestamp_start": start, "timestamp_end": end}


def test_empty_words_returns_no_cues():
    assert build_cues([]) == []
    assert build_vtt([]).startswith("WEBVTT")


def test_cue_bounded_by_max_duration():
    words = [_word(f"w{i}", i * 0.2, (i + 1) * 0.2) for i in range(100)]
    cues = build_cues(words, config=CueConfig(max_duration=3.0, max_width=200))
    assert cues
    for c in cues:
        assert c["end"] - c["start"] <= 3.05, (
            f"cue too long: {c['start']:.2f}→{c['end']:.2f}"
        )


def test_cue_bounded_by_max_width_ascii():
    words = [_word(f"word{i:02d}", i * 0.1, (i + 1) * 0.1) for i in range(60)]
    cues = build_cues(words, language="en", config=CueConfig(max_duration=100, max_width=20))
    for c in cues:
        for line in c["text"].split("\n"):
            assert _display_width(line) <= 20, f"line too wide: {line!r}"


def test_cue_bounded_by_max_width_cjk():
    # Japanese character is width 2 — 10 chars fills a width-20 cue.
    words = [_word(f"あ", i * 0.1, (i + 1) * 0.1) for i in range(30)]
    cues = build_cues(words, language="ja", config=CueConfig(max_duration=100, max_width=20))
    for c in cues:
        for line in c["text"].split("\n"):
            assert _display_width(line) <= 20, f"line too wide: {line!r}"


def test_sentence_break_flushes_cue():
    words = [
        _word("hello", 0.0, 0.5),
        _word("world.", 0.5, 1.0),
        _word("next", 1.2, 1.6),
    ]
    cues = build_cues(words, language="en")
    assert len(cues) >= 2
    assert cues[0]["text"].endswith("world.")


def test_silence_gap_flushes_cue():
    words = [
        _word("first", 0.0, 1.0),
        _word("part", 1.0, 1.5),
        _word("second", 3.0, 3.5),  # 1.5 s gap
    ]
    cues = build_cues(words, language="en")
    assert len(cues) == 2
    assert cues[0]["end"] == 1.5
    assert cues[1]["start"] == 3.0


def test_cues_are_monotonic():
    words = [_word(f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(40)]
    cues = build_cues(words, language="en", config=CueConfig(max_duration=2.0, max_width=30))
    last = -1.0
    for c in cues:
        assert c["start"] >= last
        assert c["end"] >= c["start"]
        last = c["end"]


def test_display_width_cjk_is_two():
    assert _display_width("あ") == 2
    assert _display_width("a") == 1
    assert _display_width("あa") == 3


def test_webvtt_header_and_cue_format():
    words = [_word("hello.", 0.0, 1.0), _word("world.", 2.0, 3.0)]
    doc = build_vtt(words, language="en")
    lines = doc.splitlines()
    assert lines[0] == "WEBVTT"
    assert "Language: en" in doc
    # Format: "00:00:00.000 --> 00:00:01.000"
    assert any("-->" in ln and ln.count(":") == 4 for ln in lines)


def test_sanitises_cue_breaking_characters():
    from app.subtitle_builder import _sanitise_cue_text

    # "-->" would be re-parsed as a timestamp separator.
    assert "-->" not in _sanitise_cue_text("before --> after")
    # Stray blank lines would terminate the cue early.
    assert "\n\n" not in _sanitise_cue_text("line1\n\n\nline2")


def test_japanese_joins_without_spaces():
    words = [
        _word("今日", 0.0, 0.5),
        _word("は", 0.5, 0.7),
        _word("良い", 0.7, 1.0),
        _word("天気。", 1.0, 1.5),
    ]
    cues = build_cues(words, language="ja")
    text = cues[0]["text"].replace("\n", "")
    assert text == "今日は良い天気。"
