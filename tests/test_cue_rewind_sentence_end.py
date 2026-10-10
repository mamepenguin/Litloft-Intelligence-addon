"""SPEC-ADDON-017: the tail a width or duration flush carries over is tested with the
hard- and soft-boundary rules, measured on the tail itself, and ended there when they hold."""

from __future__ import annotations

import random

import pytest

from app import subtitle_builder
from app.subtitle_builder import CueConfig, build_cues

WIDTH_24 = CueConfig(max_duration=100, max_width=24, min_duration=1.0, silence_gap=1.0)
DURATION_3 = CueConfig(max_duration=3.0, max_width=200, min_duration=1.0, silence_gap=1.0)


class _Token:
    def __init__(self, surface: str) -> None:
        self.surface = surface
        self.part_of_speech = "名詞,一般,*,*"
        self.infl_type = "*"
        self.infl_form = "*"
        self.base_form = surface
        self.reading = surface
        self.phonetic = surface
        self.node_type = "SYS_DICT"
        self.extra = None


class _VocabularyJanome:
    """Returns the given words as tokens, so regrouping leaves the words as they are."""

    def __init__(self, vocabulary) -> None:
        self._vocabulary = sorted(set(vocabulary), key=len, reverse=True)

    def tokenize(self, text, **kwargs):
        tokens = []
        i = 0
        while i < len(text):
            match = next((v for v in self._vocabulary if text.startswith(v, i)), text[i])
            tokens.append(_Token(match))
            i += len(match)
        return tokens


def _words(rows):
    return [{"text": t, "timestamp_start": s, "timestamp_end": e} for t, s, e in rows]


def _flat(text: str, language: str) -> str:
    """Cue text without its line layout.

    English drops every space: the two-line layout may break a single long word in the
    middle, so where a space was is not recoverable from the cue text.
    """
    return text.replace("\n", "") if language == "ja" else "".join(text.split())


def _cues(rows, language, config, janome, monkeypatch):
    tokenizer = _VocabularyJanome(t for t, _, _ in rows) if janome == "with-janome" else None
    monkeypatch.setattr(subtitle_builder, "_get_ja_tokenizer", lambda: tokenizer)
    cues = build_cues(_words(rows), language=language, config=config)
    return [(_flat(c["text"], language), c["start"], c["end"]) for c in cues]


def _expected(cues, language):
    return [(_flat(text, language), start, end) for text, start, end in cues]


LANGUAGES = [
    pytest.param("en", "without-janome", id="en"),
    pytest.param("ja", "without-janome", id="ja-without-janome"),
    pytest.param("ja", "with-janome", id="ja-with-janome"),
]

# The expected heads are the cues the rewind emits for these rows without this change
# (I7): after the full stop for a width flush, before the last word for a duration flush.
PREFIX = {
    "en": [("Acknowledged.", 0.0, 0.9), ("we", 1.0, 1.2), ("meet", 1.25, 1.6)],
    "ja": [("了解しました。", 0.0, 0.9), ("会議は", 1.0, 1.4)],
}
HEAD = {"en": PREFIX["en"][0], "ja": PREFIX["ja"][0]}
LAST_START = {"en": 1.65, "ja": 1.45}
# Short enough that tail + following stays under max_width and max_duration, so only
# the boundary tests on the tail can end it.
FOLLOWING = {
    "en": ([("so", 0.0, 0.35), ("long.", 0.35, 1.05)], "so long."),
    "ja": ([("以上。", 0.0, 1.05)], "以上。"),
}
# Short enough in time that tail + silence + following stays under max_duration.
SHORT_FOLLOWING = {"en": ([("bye.", 0.0, 0.2)], "bye."), "ja": ([("以上。", 0.0, 0.2)], "以上。")}
SOFT_FOLLOWING = {"en": ([("ok.", 0.0, 0.5)], "ok."), "ja": ([("ね", 0.0, 0.5)], "ね")}


def _shifted(rows, offset):
    return [(t, round(s + offset, 3), round(e + offset, 3)) for t, s, e in rows]


def _join(words, language):
    return (" " if language == "en" else "").join(words)


def _scenario(language, last, last_end, next_start, flush, following=FOLLOWING):
    """Rows of a cue that reaches a limit at ``last``, then a short following sentence,
    and the three cues SPEC-ADDON-017 expects from them."""
    rows_after, text_after = following[language]
    after = _shifted(rows_after, next_start)
    prefix = PREFIX[language]
    last_row = (last, LAST_START[language], last_end)
    if flush == "width":
        head = HEAD[language]
        tail = (_join([t for t, _, _ in prefix[1:]] + [last], language), prefix[1][1], last_end)
    else:
        head = (_join([t for t, _, _ in prefix], language), 0.0, prefix[-1][2])
        tail = last_row
    expected = [head, tail, (text_after, after[0][1], after[-1][2])]
    return prefix + [last_row] + after, _expected(expected, language)


class TestTailEndsAtSentenceEnd:
    """I1, I7, I10: a width flush whose tail ends with a full stop."""

    @pytest.mark.parametrize("language, janome", LANGUAGES)
    @pytest.mark.parametrize("tail_end", [2.2, 2.0], ids=["over-min-duration", "exactly-min-duration"])
    def test_spec_addon_017_width_flush_tail_ending_with_full_stop_is_its_own_cue(
        self, language, janome, tail_end, monkeypatch
    ):
        last = {"en": "today.", "ja": "正午。"}[language]
        rows, expected = _scenario(language, last, tail_end, tail_end + 0.05, "width")

        assert _cues(rows, language, WIDTH_24, janome, monkeypatch) == expected


class TestTailEndsAtSilence:
    """I2, I7, I10: a width flush whose tail has no punctuation and is followed by a silence."""

    @pytest.mark.parametrize("language, janome", LANGUAGES)
    @pytest.mark.parametrize("gap", [1.2, 1.0], ids=["over-silence-gap", "exactly-silence-gap"])
    def test_spec_addon_017_width_flush_tail_followed_by_silence_is_its_own_cue(
        self, language, janome, gap, monkeypatch
    ):
        last = {"en": "today", "ja": "正午に"}[language]
        rows, expected = _scenario(language, last, 2.5, 2.5 + gap, "width")

        assert _cues(rows, language, WIDTH_24, janome, monkeypatch) == expected


class TestDurationFlush:
    """I2, I7, I10: the cue reached max_duration rather than max_width."""

    @pytest.mark.parametrize("language, janome", LANGUAGES)
    def test_spec_addon_017_duration_flush_tail_ending_with_full_stop_is_its_own_cue(
        self, language, janome, monkeypatch
    ):
        last = {"en": "today.", "ja": "正午。"}[language]
        rows, expected = _scenario(language, last, 3.1, 3.15, "duration")

        assert _cues(rows, language, DURATION_3, janome, monkeypatch) == expected

    @pytest.mark.parametrize("language, janome", LANGUAGES)
    def test_spec_addon_017_duration_flush_tail_followed_by_silence_is_its_own_cue(
        self, language, janome, monkeypatch
    ):
        last = {"en": "today", "ja": "正午に"}[language]
        rows, expected = _scenario(
            language, last, 3.1, 4.2, "duration", following=SHORT_FOLLOWING
        )

        assert _cues(rows, language, DURATION_3, janome, monkeypatch) == expected


class TestSoftBoundary:
    """I3, I10: a tail ending with a comma is ended only when it is wide enough and long enough."""

    @pytest.mark.parametrize("language, janome", LANGUAGES)
    @pytest.mark.parametrize(
        "last_words",
        [
            # tail widths 19 (en) and 20 (ja) against max_width * 0.75 = 18
            {"en": "afterwards,", "ja": "十二時ですが、"},
            # tail width exactly 18
            {"en": "yesterday,", "ja": "正午ですが、"},
        ],
        ids=["over-three-quarters", "exactly-three-quarters"],
    )
    def test_spec_addon_017_wide_tail_ending_with_comma_is_its_own_cue(
        self, language, janome, last_words, monkeypatch
    ):
        rows, expected = _scenario(
            language, last_words[language], 2.2, 2.25, "width", following=SOFT_FOLLOWING
        )

        assert _cues(rows, language, WIDTH_24, janome, monkeypatch) == expected

    @pytest.mark.parametrize("language, janome", LANGUAGES)
    def test_spec_addon_017_narrow_tail_ending_with_comma_carries_forward(
        self, language, janome, monkeypatch
    ):
        last, after, joined = {
            "en": ("today,", "ok.", "we meet today, ok."),
            "ja": ("正午、", "以上。", "会議は正午、以上。"),
        }[language]
        rows = PREFIX[language] + [
            (last, 1.65 if language == "en" else 1.45, 2.2), (after, 2.25, 2.7)
        ]

        assert _cues(rows, language, WIDTH_24, janome, monkeypatch) == _expected(
            [HEAD[language], (joined, 1.0, 2.7)], language
        )

    @pytest.mark.parametrize("language, janome", LANGUAGES)
    def test_spec_addon_017_wide_tail_ending_with_comma_under_min_duration_carries_forward(
        self, language, janome, monkeypatch
    ):
        rows, joined = {
            "en": (
                [("we", 1.0, 1.1), ("meet", 1.1, 1.2), ("yesterday,", 1.2, 1.6), ("ok.", 1.65, 2.0)],
                "we meet yesterday, ok.",
            ),
            "ja": (
                [("会議は", 1.0, 1.2), ("正午ですが、", 1.2, 1.6), ("ね。", 1.65, 2.0)],
                "会議は正午ですが、ね。",
            ),
        }[language]

        assert _cues([HEAD[language]] + rows, language, WIDTH_24, janome, monkeypatch) == _expected(
            [HEAD[language], (joined, 1.0, 2.0)], language
        )


class TestCarriesForward:
    """I4, I5, I10: tails that do not pass the boundary tests carry into the next cue."""

    @pytest.mark.parametrize("language, janome", LANGUAGES)
    def test_spec_addon_017_short_tail_ending_with_full_stop_carries_forward(
        self, language, janome, monkeypatch
    ):
        rows, joined = {
            "en": (
                [("we", 1.0, 1.1), ("meet", 1.1, 1.2), ("today.", 1.2, 1.6), ("ok.", 1.65, 2.0)],
                "we meet today. ok.",
            ),
            "ja": (
                [("会議は", 1.0, 1.2), ("正午。", 1.2, 1.6), ("以上。", 1.65, 2.0)],
                "会議は正午。以上。",
            ),
        }[language]

        assert _cues([HEAD[language]] + rows, language, WIDTH_24, janome, monkeypatch) == _expected(
            [HEAD[language], (joined, 1.0, 2.0)], language
        )

    @pytest.mark.parametrize("language, janome", LANGUAGES)
    def test_spec_addon_017_tail_ending_with_digit_separator_carries_forward(
        self, language, janome, monkeypatch
    ):
        rows, joined = {
            "en": (
                [("we", 1.0, 1.2), ("meet", 1.25, 1.6), ("12:", 1.65, 2.2), ("30", 2.25, 2.6)],
                "we meet 12: 30",
            ),
            "ja": (
                [("会議は", 1.0, 1.4), ("午後12：", 1.45, 2.2), ("30", 2.25, 2.6)],
                "会議は午後12：30",
            ),
        }[language]

        assert _cues([HEAD[language]] + rows, language, WIDTH_24, janome, monkeypatch) == _expected(
            [HEAD[language], (joined, 1.0, 2.6)], language
        )


EN_VOCAB = ["a", "we", "meeting", "Acknowledged.", "today", "12:", "30",
            "3.", "5km", "today.", "now,", "ok?", "well!", "yesterday,", "x"]
JA_VOCAB = ["会議の", "開始時刻は", "12：", "30", "からです。", "本日", "の", "議題は、",
            "二つ", "です。", "ー", "」", "ゃ", "正午ですが、", "了解しました。"]


def _random_rows(rng, vocabulary):
    rows = []
    t = 0.0
    for _ in range(rng.randint(0, 60)):
        t += rng.choice([0.0, 0.0, 0.1, 0.6, 1.5])
        length = rng.choice([0.0, 0.1, 0.3, 0.8, 2.0])
        rows.append((rng.choice(vocabulary), round(t, 3), round(t + length, 3)))
        t += length
    return rows


@pytest.mark.parametrize("language, janome", LANGUAGES)
@pytest.mark.parametrize("seed", range(40))
def test_spec_addon_017_cues_keep_every_word_in_order(language, janome, seed, monkeypatch):
    """I6: no word is lost, duplicated or reordered, and no cue ends before it starts."""
    rng = random.Random(seed)
    rows = _random_rows(rng, EN_VOCAB if language == "en" else JA_VOCAB)
    config = CueConfig(
        max_duration=rng.choice([1.0, 3.0, 5.0]),
        max_width=rng.choice([8, 16, 24, 42]),
        min_duration=rng.choice([0.0, 0.6, 1.0]),
        silence_gap=rng.choice([0.3, 0.6, 1.0]),
    )

    cues = _cues(rows, language, config, janome, monkeypatch)

    assert "".join(text for text, _, _ in cues) == _flat(
        "".join(t for t, _, _ in rows), language
    )
    assert all(end >= start for _, start, end in cues)
