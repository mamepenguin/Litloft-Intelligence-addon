"""SPEC-ADDON-010: refine asks for numbers in digits and stores the answer as returned."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import settings as real_settings
from app.workers import refine

NUMBERS_LINE = (
    "- Numbers: write every number that states a quantity, count, date, time, amount of "
    "money, measurement or ordinal in Arabic digits (0-9), with no digit-group separators "
    "and a decimal point only where the speaker said one. When the speaker used a "
    "large-number unit word (a word for ten thousand, a million and the like), keep that "
    "word and write its multiplier in digits. Leave a numeral that is part of a fixed word "
    "or idiom as it is. Rewriting a number this way is allowed even though it changes the "
    "character count."
)

NO_WORDS_LINE = "Do not add words other than punctuation."

_HEAD = [
    "You are an assistant that corrects ASR (automatic speech recognition) mis-recognitions.",
    "Do not change the meaning of each segment; keep the character count similar to "
    "preserve timing information.",
    "Infer and unify proper nouns, place names, and technical terms from context.",
    "Return the output in the same language as the input (do not translate).",
]
_TAIL = [
    NO_WORDS_LINE,
    'Output JSON only: {"items": [{"id": <int>, "text_refined": <str>}, ...]}; '
    "include no other text.",
]
_PUNCTUATION = {
    "ja": "- Japanese: insert 。at sentence ends and 、at natural clause boundaries; "
    "do not add other words.",
    "en": "- English: add standard punctuation (periods, commas) at natural clause "
    "boundaries; do not add other words.",
    "fr": "- Add punctuation at natural clause boundaries following the language's "
    "conventions; do not add other words.",
}


@pytest.fixture()
def output_language():
    original = real_settings.llm.output_language

    def _set(lang: str) -> None:
        object.__setattr__(real_settings.llm, "output_language", lang)

    yield _set
    object.__setattr__(real_settings.llm, "output_language", original)


@pytest.mark.parametrize("lang", ["ja", "en", "fr"])
def test_spec_addon_010_numbers_line_sits_between_punctuation_and_no_words_lines(
    output_language, lang
):
    if lang == "fr":
        assert lang not in refine._PUNCTUATION_INSTRUCTIONS
    output_language(lang)

    lines = refine._build_system_prompt().splitlines()

    assert lines == [*_HEAD, _PUNCTUATION[lang], NUMBERS_LINE, *_TAIL]


def _chunk(cid: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=cid,
        file_id="fileabc",
        chunk_index=cid - 1,
        text=text,
        text_refined_at=None,
        refined_model=None,
        timestamp_start=float(cid),
        timestamp_end=float(cid) + 1.0,
        language="ja",
    )


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


@pytest.mark.asyncio
async def test_spec_addon_010_refine_sends_the_numbers_line_and_stores_the_answer_verbatim(
    output_language,
):
    output_language("ja")
    returned = [
        "二千二十五年に3.3kmを歩いた。",
        "四十四分かかった、1人で。",
        "一石二鳥だった, 1,500円 で 1万5000円。",
    ]
    chunks = [_chunk(i + 1, f"original {i}") for i in range(len(returned))]
    llm = MagicMock()
    llm.enabled = True
    llm.generate_json = AsyncMock(
        return_value=[{"id": c.id, "text_refined": t} for c, t in zip(chunks, returned)]
    )

    await refine.refine_chunks(MagicMock(), llm, chunks, model="m")

    sent = [
        s
        for call in llm.generate_json.call_args_list
        for s in _strings(list(call.args) + list(call.kwargs.values()))
    ]
    assert any(NUMBERS_LINE in s.splitlines() for s in sent)
    assert [c.text.encode("utf-8") for c in chunks] == [t.encode("utf-8") for t in returned]
