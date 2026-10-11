"""SPEC-ADDON-018: a cue ends no later than the next cue starts, when the next cue starts
later than it does; the word rows themselves keep their overlapping times."""

from __future__ import annotations

import copy
import random

import pytest

from app import subtitle_builder
from app.subtitle_builder import CueConfig, build_cues, build_vtt

# min_duration 0 so every word ending with a full stop closes its own cue.
SENTENCES = CueConfig(max_duration=100, max_width=200, min_duration=0.0, silence_gap=1.0)


@pytest.fixture(autouse=True)
def _no_janome(monkeypatch):
    monkeypatch.setattr(subtitle_builder, "_get_ja_tokenizer", lambda: None)


def _words(rows):
    return [{"text": t, "timestamp_start": s, "timestamp_end": e} for t, s, e in rows]


def _flat(text: str, language: str) -> str:
    """Cue text without its line layout (English loses every space, see SPEC-ADDON-017)."""
    return text.replace("\n", "") if language == "ja" else "".join(text.split())


def _cues(rows, language, config):
    cues = build_cues(_words(rows), language=language, config=config)
    return [(_flat(c["text"], language), c["start"], c["end"]) for c in cues]


@pytest.mark.parametrize(
    "language, rows, expected",
    [
        pytest.param(
            "ja",
            [("了解しました", 12.0, 13.0), ("。", 13.12, 14.88), ("え", 13.695, 14.0), ("以上。", 14.0, 14.5)],
            [("了解しました。", 12.0, 13.695), ("え以上。", 13.695, 14.5)],
            id="ja-measured-full-stop-row",
        ),
        pytest.param(
            "en",
            [("One.", 0.0, 2.0), ("Two.", 1.5, 3.0), ("Three.", 3.0, 4.0), ("Four.", 5.5, 7.0)],
            [("One.", 0.0, 1.5), ("Two.", 1.5, 3.0), ("Three.", 3.0, 4.0), ("Four.", 5.5, 7.0)],
            id="en-overlap-then-touching-then-gap-then-last",
        ),
        pytest.param(
            "en",
            [("One.", 0.0, 3.0), ("Last.", 1.0, 9.0)],
            [("One.", 0.0, 1.0), ("Last.", 1.0, 9.0)],
            id="en-second-to-last-against-last",
        ),
    ],
)
def test_spec_addon_018_overlapping_cue_ends_at_next_start_and_others_keep_their_end(
    language, rows, expected
):
    """I1, I2, I3, I4."""
    assert _cues(rows, language, SENTENCES) == expected


def test_spec_addon_018_each_cue_is_compared_with_the_next_cues_start():
    """I2: a cue clamped in turn does not move the end of the cue before it any further."""
    rows = [("One.", 0.0, 5.0), ("Two.", 1.0, 6.0), ("Three.", 2.0, 3.0)]

    assert _cues(rows, "en", SENTENCES) == [
        ("One.", 0.0, 1.0),
        ("Two.", 1.0, 2.0),
        ("Three.", 2.0, 3.0),
    ]


def test_spec_addon_018_cue_followed_by_one_starting_no_later_keeps_its_end():
    """I6, alongside a clamp in the same list so the list as a whole is checked."""
    rows = [
        ("One.", 0.0, 3.0),
        ("Two.", 1.0, 4.0),
        ("Three.", 1.0, 2.0),
        ("Four.", 0.5, 1.0),
        ("Five.", 2.0, 2.5),
    ]

    assert _cues(rows, "en", SENTENCES) == [
        ("One.", 0.0, 1.0),
        ("Two.", 1.0, 4.0),
        ("Three.", 1.0, 2.0),
        ("Four.", 0.5, 1.0),
        ("Five.", 2.0, 2.5),
    ]


def test_spec_addon_018_webvtt_carries_the_clamped_end():
    """Item 1.4: the WebVTT is serialised from the clamped cues."""
    words = _words(
        [("了解しました", 12.0, 13.0), ("。", 13.12, 14.88), ("え", 13.695, 14.0), ("以上。", 14.0, 14.5)]
    )

    timings = [line for line in build_vtt(words, language="ja").splitlines() if "-->" in line]

    assert timings == ["00:00:12.000 --> 00:00:13.695", "00:00:13.695 --> 00:00:14.500"]


EN_VOCAB = ["we", "meeting", "Acknowledged.", "today", "today.", "now,", "ok?", "well!", "x", "and"]
JA_VOCAB = ["会議の", "開始は", "本日", "の", "議題は、", "二つ", "です。", "。", "え", "了解しました。"]


def _random_rows(rng, vocabulary):
    """Rows that often overlap, sometimes start earlier than the row before, and sometimes
    start at the same instant."""
    rows = []
    t = 0.0
    for _ in range(rng.randint(0, 50)):
        t = max(t + rng.choice([0.0, 0.1, 0.6, 1.5, -0.3, -1.0]), 0.0)
        length = rng.choice([0.0, 0.1, 0.3, 0.8, 2.0, 4.0])
        rows.append((rng.choice(vocabulary), round(t, 3), round(t + length, 3)))
        t += rng.choice([length, length / 2, 0.0])
    return rows


def _inputs():
    for language, vocabulary in (("en", EN_VOCAB), ("ja", JA_VOCAB)):
        yield language, [], CueConfig()
        yield language, [(vocabulary[0], 1.0, 2.0)], CueConfig()
        for seed in range(150):
            rng = random.Random(seed)
            config = CueConfig(
                max_duration=rng.choice([1.0, 3.0, 5.0]),
                max_width=rng.choice([8, 16, 24, 42]),
                min_duration=rng.choice([0.0, 0.6, 1.0]),
                silence_gap=rng.choice([0.3, 0.6, 1.0]),
            )
            yield language, _random_rows(rng, vocabulary), config


def _word_spans(cues, rows, language):
    """For each cue, the first and last row it was built from, matched by text in order."""
    spans = []
    i = 0
    for c in cues:
        target = _flat(c["text"], language)
        first = i
        acc = ""
        while len(acc) < len(target):
            acc += _flat(rows[i][0], language)
            i += 1
        assert acc == target, (c, rows[first:i])
        spans.append((rows[first], rows[i - 1]))
    assert i == len(rows)
    return spans


def test_spec_addon_018_cue_ends_hold_on_overlapping_word_rows():
    """I1-I7 over rows that overlap, run out of order and share a start; includes the empty
    input and a single word.

    Before the clamp a cue starts at its first word's ``timestamp_start`` and ends at its last
    word's ``timestamp_end``, so those are the values the clamp is checked against. A cue whose
    last word ends before its first word starts is skipped: the spec does not say what its end
    should be.
    """
    failures = []
    clamped = 0
    for language, rows, config in _inputs():
        words = _words(rows)
        before = copy.deepcopy(words)

        cues = build_cues(words, language=language, config=config)

        if words != before:
            failures.append(("I7 words changed", language, rows))
            continue
        spans = _word_spans(cues, rows, language)
        for k, (cue, (first, last)) in enumerate(zip(cues, spans)):
            original_end = last[2]
            if original_end < first[1]:
                continue
            expected_end = original_end
            if k + 1 < len(cues):
                next_start = cues[k + 1]["start"]
                if original_end > next_start and next_start > cue["start"]:
                    expected_end = next_start
                    clamped += 1
                if next_start > cue["start"] and cue["end"] > next_start:
                    failures.append(("I1", language, cue, cues[k + 1]))
            if cue["start"] != first[1]:
                failures.append(("I4 start", language, cue, first))
            if cue["end"] != expected_end:
                failures.append(("I2/I3/I6 end", language, cue, expected_end))
            if cue["end"] < cue["start"]:
                failures.append(("I5", language, cue))

    assert clamped > 0
    assert failures == []
