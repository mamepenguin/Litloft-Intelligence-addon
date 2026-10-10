"""SPEC-ADDON-014, SPEC-ADDON-015: a two-line cue never breaks its line inside a number
written in digits."""

from __future__ import annotations

import re

import pytest

from app import subtitle_builder
from app.digit_separator import splits_number
from app.subtitle_builder import _balance_two_lines

DIGITS = "0123456789"
SEPARATORS = ".,:："


def _inside_number(text: str, p: int) -> bool:
    if p <= 0 or p >= len(text):
        return False
    before = text[:p]
    if text[p] in DIGITS:
        if before[-1] in DIGITS:
            return True
        if len(before) >= 2 and before[-1] in SEPARATORS and before[-2] in DIGITS:
            return True
        if (
            len(before) >= 3
            and before[-1] == " "
            and before[-2] in SEPARATORS
            and before[-3] in DIGITS
        ):
            return True
        return False
    if text[p] in SEPARATORS:
        return before[-1] in DIGITS and p + 1 < len(text) and text[p + 1] in DIGITS
    return False


def _split_position(original: str, out: str) -> int | None:
    if "\n" not in out:
        assert out == original
        return None
    first, second = out.split("\n")
    if f"{first} {second}" == original:
        return len(first) + 1
    assert first + second == original, out
    return len(first)


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


class _FakeJanome:
    """Splits digits and separators one character each, as janome does for numerals."""

    def tokenize(self, text, **kwargs):
        return [
            _Token(m.group(0))
            for m in re.finditer(r"[0-9]|[.,]|[ァ-ヴー]+|[一-龥]+|[ぁ-ん]{1,2}|.", text)
        ]


@pytest.fixture(params=["without-janome", "with-janome"])
def janome(request, monkeypatch):
    tokenizer = None if request.param == "without-janome" else _FakeJanome()
    monkeypatch.setattr(subtitle_builder, "_get_ja_tokenizer", lambda: tokenizer)
    return request.param


def _balance(text: str, soft_width: int) -> str:
    cue = {"start": 0.0, "end": 2.0, "text": text}
    out = _balance_two_lines(cue, soft_width)
    assert out["start"] == 0.0 and out["end"] == 2.0
    return out["text"]


SWEEP_TEXTS = [
    "we walked 3. 3km today and more",
    "12345678901234567890",
    "1234567890, 1234567890",
    "だいたい3.3キロくらい",
    "今日はだいたい3.3キロくらい歩いた",
    "あいうえおかき1,500円くけこ、さしすせそ",
    "あいうえおかき1,500円くけこさしすせそ",
    "あいうえおか2025年だった",
    "そこまで12.5キロある",
    "1234567890あいう",
    "1234567890っあいう",
    "合計は3,000,000円で2.75倍です",
]


@pytest.mark.parametrize("text", SWEEP_TEXTS)
def test_spec_addon_014_line_break_never_falls_inside_a_number(janome, text):
    for soft_width in range(4, 40):
        out = _balance(text, soft_width)
        p = _split_position(text, out)
        assert p is None or not _inside_number(text, p), (soft_width, out)


@pytest.mark.parametrize(
    "text, soft_width, expected",
    [
        ("だいたい3.3キロくらい", 16, "だいたい\n3.3キロくらい"),
        ("そこまで12.5キロある", 10, "そこまで\n12.5キロある"),
        ("あいうえおか2025年だった", 14, "あいうえおか\n2025年だった"),
        (
            "あいうえおかき1,500円くけこ、さしすせそ",
            20,
            "あいうえおかき1,500円くけこ、\nさしすせそ",
        ),
        ("1234567890あいう", 10, "1234567890\nあいう"),
        ("1234567890っあいう", 10, "1234567890っあいう"),
        ("12345678901234567890", 10, "12345678901234567890"),
    ],
    ids=[
        "point-moves-before-number",
        "janome-point-moves-before-number",
        "year-moves-before-number",
        "comma-in-number-skipped-for-later-touten",
        "number-at-start-moves-to-its-end",
        "number-at-start-before-no-break-char-stays-one-line",
        "whole-text-is-a-number-stays-one-line",
    ],
)
def test_spec_addon_014_cjk_break_inside_a_number_moves_to_its_edge(
    janome, text, soft_width, expected
):
    assert _balance(text, soft_width) == expected


@pytest.mark.parametrize(
    "text, soft_width, expected",
    [
        ("we walked 3. 3km today and more", 16, "we walked\n3. 3km today and more"),
        ("1234567890, 1234567890", 10, "1234567890, 1234567890"),
    ],
    ids=["space-inside-number-is-not-a-candidate", "only-candidate-inside-number"],
)
def test_spec_addon_014_space_inside_a_number_is_not_a_split(text, soft_width, expected):
    assert _balance(text, soft_width) == expected


@pytest.mark.parametrize(
    "text, soft_width, expected",
    [
        ("alpha bravo charlie delta echo", 20, "alpha bravo\ncharlie delta echo"),
        ("Yes, 3 people came", 8, "Yes, 3\npeople came"),
        ("Yes, 3 people came here today", 14, "Yes, 3 people\ncame here today"),
        ("we walked about 1,500 meters", 14, "we walked\nabout 1,500 meters"),
        ("in 2020, 50 people came here", 12, "in 2020, 50\npeople came here"),
        ("あいう、えおかきくけこさしすせそ", 14, "あいう、\nえおかきくけこさしすせそ"),
        ("だいたい3.3キロくらい歩きました", 16, "だいたい3.3キロ\nくらい歩きました"),
        ("we walked 3. 3km", 40, "we walked 3. 3km"),
        ("だいたい3.3キロ", 40, "だいたい3.3キロ"),
    ],
)
def test_spec_addon_014_split_outside_a_number_is_unchanged(janome, text, soft_width, expected):
    assert _balance(text, soft_width) == expected


@pytest.mark.parametrize(
    "text, pos, inside",
    [
        ("3.3", 1, True),
        ("3.3", 2, True),
        ("2025", 2, True),
        ("1,500", 1, True),
        ("3. 3km", 3, True),
        ("a3.3", 1, False),
        ("3.3", 0, False),
        ("3.3", 3, False),
        ("3.a", 1, False),
        ("a.3", 2, False),
        ("Yes, 3", 5, False),
        ("2020 50", 5, False),
        ("3.  3", 4, False),
        ("3.3キロ", 3, False),
    ],
)
def test_spec_addon_014_splits_number(text, pos, inside):
    assert splits_number(text, pos) is inside


COLON_SWEEP_TEXTS = [
    "朝12：30から出発します",
    "朝は12:30から出発します",
    "だいたい12:30キロくらい",
    "だいたい12：30キロくらい",
    "12：30からみんなで出発",
    "12:30からみんなで出発",
    "あいうえおかき12：30円くけこ、さしすせそ",
    "今日は1：0で勝った試合でした",
    "その1：3種類あります",
    "we meet at 12: 30 today and more",
    "we meet at 12： 30 today and more",
]


@pytest.mark.parametrize("text", COLON_SWEEP_TEXTS)
def test_spec_addon_015_line_break_never_falls_inside_a_time(janome, text):
    for soft_width in range(4, 40):
        out = _balance(text, soft_width)
        p = _split_position(text, out)
        assert p is None or not _inside_number(text, p), (soft_width, out)


@pytest.mark.parametrize(
    "text, soft_width, expected",
    [
        ("だいたい12:30キロくらい", 16, "だいたい\n12:30キロくらい"),
        ("だいたい12：30キロくらい", 16, "だいたい\n12：30キロくらい"),
        ("朝は12:30から出発します", 10, "朝は12:30\nから出発します"),
    ],
    ids=[
        "ascii-colon-moves-before-number",
        "fullwidth-colon-moves-before-number",
        "ascii-colon-moves-after-number",
    ],
)
def test_spec_addon_015_cjk_break_inside_a_time_moves_to_its_edge(
    janome, text, soft_width, expected
):
    assert _balance(text, soft_width) == expected


@pytest.mark.parametrize("colon", [":", "："], ids=["ascii", "fullwidth"])
def test_spec_addon_015_soft_punctuation_scan_skips_a_colon_inside_a_time(janome, colon):
    timed = f"あいうえおかき12{colon}30円くけこ、さしすせそ"
    worded = f"時刻{colon}今日はいい天気ですね"

    assert (_balance(timed, 20), _balance(worded, 10)) == (
        f"あいうえおかき12{colon}30円くけこ、\nさしすせそ",
        f"時刻{colon}\n今日はいい天気ですね",
    )


def test_spec_addon_015_space_after_a_time_colon_is_not_a_split():
    assert _balance("we meet at 12: 30 today and more", 16) == (
        "we meet at\n12: 30 today and more"
    )


def test_spec_addon_015_splits_number_inside_a_time():
    texts = ["朝12：30から", "at 12:30 today", "12:30", "12：30", "1：0", "時刻：12時", "Note: 3", "a:3", "3:a"]
    expected = {
        "朝12：30から": [False, False, True, True, True, True, False, False, False],
        "at 12:30 today": [False] * 4 + [True] * 4 + [False] * 7,
        "12:30": [False, True, True, True, True, False],
        "12：30": [False, True, True, True, True, False],
        "1：0": [False, True, True, False],
        "時刻：12時": [False, False, False, False, True, False, False],
        "Note: 3": [False] * 8,
        "a:3": [False] * 4,
        "3:a": [False] * 4,
    }

    assert {t: [splits_number(t, p) for p in range(len(t) + 1)] for t in texts} == expected


def test_spec_addon_015_splits_number_with_a_space_after_the_colon():
    text = "12: 30"

    assert [splits_number(text, p) for p in (0, 1, 3, 4, 5, 6)] == [
        False, True, False, True, True, False,
    ]
