"""SPEC-ADDON-014: a two-line cue never breaks its line inside a number written in digits."""

from __future__ import annotations

import re

import pytest

from app import subtitle_builder
from app.subtitle_builder import _balance_two_lines

DIGITS = "0123456789"


def _inside_number(text: str, p: int) -> bool:
    if p <= 0 or p >= len(text):
        return False
    before = text[:p]
    if text[p] in DIGITS:
        if before[-1] in DIGITS:
            return True
        if len(before) >= 2 and before[-1] in ".," and before[-2] in DIGITS:
            return True
        if (
            len(before) >= 3
            and before[-1] == " "
            and before[-2] in ".,"
            and before[-3] in DIGITS
        ):
            return True
        return False
    if text[p] in ".,":
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
