"""SPEC-ADDON-013: align_segment keeps the rows of every sentence WhisperX returns."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from app.workers import aligner


def _install(monkeypatch, segments, language):
    aligner.release_all()
    fake = MagicMock()
    fake.align = MagicMock(return_value={"segments": segments})
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    aligner._align_models[language] = (MagicMock(), MagicMock())
    return fake


def _words(*rows):
    return {"words": [{"word": w, "start": s, "end": e} for w, s, e in rows]}


def _chars(*rows):
    return {"chars": [{"char": c, "start": s, "end": e} for c, s, e in rows]}


def _align(text, language, start=0.0, end=20.0):
    return aligner.align_segment(
        waveform=object(),
        chunk_start=start,
        chunk_end=end,
        text=text,
        language=language,
    )


def _triples(rows):
    return [(r["text"], r["timestamp_start"], r["timestamp_end"]) for r in rows]


def test_spec_addon_013_every_sentence_of_a_word_level_result_is_kept_in_order(monkeypatch):
    text = "We walked far. It rained all day. Then we went home."
    _install(
        monkeypatch,
        [
            _words(("We", 0.0, 0.4), ("walked", 0.4, 0.9), ("far.", 0.9, 1.5)),
            _words(
                ("It", 2.0, 2.3), ("rained", 2.3, 2.8), ("all", 2.8, 3.1),
                ("day.", 3.1, 3.6),
            ),
            _words(
                ("Then", 4.0, 4.3), ("we", 4.3, 4.5), ("went", 4.5, 4.9),
                ("home.", 4.9, 5.5),
            ),
        ],
        "en",
    )

    rows = _align(text, "en")

    assert rows is not None
    assert [r["text"] for r in rows] == text.split()
    assert len(rows) == len(text.split())
    assert _triples(rows)[3] == ("It", 2.0, 2.3)
    assert _triples(rows)[-1] == ("home.", 4.9, 5.5)


def test_spec_addon_013_every_sentence_of_a_character_level_result_is_kept_in_order(
    monkeypatch,
):
    _install(
        monkeypatch,
        [
            _chars(("歩", 0.0, 0.2), ("い", 0.2, 0.4), ("た", 0.4, 0.6), ("。", 0.6, 0.7)),
            _chars(("雨", 1.0, 1.2), ("だ", 1.2, 1.4), ("。", 1.4, 1.5)),
        ],
        "ja",
    )

    rows = _align("歩いた。雨だ。", "ja")

    assert rows is not None
    assert [r["text"] for r in rows] == list("歩いた。雨だ。")


def test_spec_addon_013_empty_first_sentence_still_yields_the_later_rows(monkeypatch):
    _install(
        monkeypatch,
        [
            {"words": []},
            _words(("It", 2.0, 2.3), ("rained.", 2.3, 2.8)),
            {"words": []},
            _words(("Then", 4.0, 4.3), ("home.", 4.3, 4.9)),
        ],
        "en",
    )

    rows = _align("It rained. Then home.", "en")

    assert rows is not None
    assert [r["text"] for r in rows] == ["It", "rained.", "Then", "home."]


@pytest.mark.parametrize(
    "segments",
    [
        [{"words": []}, {"words": []}],
        [{"words": []}],
    ],
    ids=["two-empty", "one-empty"],
)
def test_spec_addon_013_no_rows_in_any_sentence_yields_none(monkeypatch, segments):
    _install(monkeypatch, segments, "en")

    assert _align("It rained. Then home.", "en") is None


def test_spec_addon_013_rows_out_of_time_order_across_sentences_keep_segment_order(
    monkeypatch,
):
    _install(
        monkeypatch,
        [
            _words(("late", 6.0, 6.5), ("first.", 6.5, 7.0)),
            _words(("early", 2.0, 2.5), ("second.", 2.5, 3.0)),
        ],
        "en",
    )

    rows = _align("late first. early second.", "en", start=1.0, end=10.0)

    assert rows is not None
    assert [r["text"] for r in rows] == ["late", "first.", "early", "second."]


def test_spec_addon_013_rows_outside_the_window_are_dropped_from_every_sentence(
    monkeypatch,
):
    _install(
        monkeypatch,
        [
            _words(("before", 0.2, 0.8), ("inside.", 1.2, 1.8)),
            _words(("also", 3.0, 3.5), ("after.", 4.8, 5.4)),
        ],
        "en",
    )

    rows = _align("before inside. also after.", "en", start=1.0, end=5.0)

    assert rows is not None
    assert [r["text"] for r in rows] == ["inside.", "also"]
    assert all(1.0 <= r["timestamp_start"] <= r["timestamp_end"] <= 5.0 for r in rows)


def test_spec_addon_013_untimed_token_at_a_sentence_start_sits_in_the_gap_between_sentences(
    monkeypatch,
):
    _install(
        monkeypatch,
        [
            _words(("we", 1.0, 1.5), ("walked.", 1.5, 2.0)),
            {"words": [{"word": "3"}, {"word": "km", "start": 4.0, "end": 4.5}]},
        ],
        "en",
    )

    rows = _align("we walked. 3 km", "en", start=0.0, end=10.0)

    assert rows is not None
    assert [r["text"] for r in rows] == ["we", "walked.", "3", "km"]
    untimed = rows[2]
    assert 2.0 <= untimed["timestamp_start"] <= untimed["timestamp_end"] <= 4.0
