"""SPEC-ADDON-010: a `.` or `,` between two digits is not punctuation for chunks or cues."""

from __future__ import annotations

import pytest

from app.subtitle_builder import CueConfig, build_cues
from app.workers.whisper import _build_chunks_from_words


# (text, start, end). Expected groups are separated where the spec says a break
# still happens: after `today.` (not a separator), after `a.` (letter before
# the point), after `2024.` (next row starts with a letter), and at `5.` where a
# 2 s silence follows.
TOKEN_ROWS = [
    ("we", 0.0, 1.0), ("walked", 1.0, 2.0), ("3.", 2.0, 3.0), ("3km", 3.0, 4.0),
    ("today.", 4.0, 5.0),
    ("it", 5.0, 6.0), ("was", 6.0, 7.0), ("a.", 7.0, 8.0),
    ("3", 8.0, 9.0), ("people", 9.0, 10.0), ("came", 10.0, 11.0), ("in", 11.0, 12.0),
    ("2024.", 12.0, 13.0),
    ("next", 13.0, 14.0), ("year", 14.0, 15.0), ("5.", 15.0, 16.0),
    ("5", 18.0, 19.0), ("more", 19.0, 20.0), ("came.", 20.0, 21.0),
]
TOKEN_GROUPS = [
    ("we walked 3. 3km today.", 0.0, 5.0),
    ("it was a.", 5.0, 8.0),
    ("3 people came in 2024.", 8.0, 13.0),
    ("next year 5.", 13.0, 16.0),
    ("5 more came.", 18.0, 21.0),
]

# The aligner's character rows: the separator is a row of its own.
CHAR_ROWS = [
    ("距", 0.0, 1.0), ("離", 1.0, 2.0), ("は", 2.0, 3.0),
    ("3", 3.0, 4.0), (".", 4.0, 4.1), ("3", 4.1, 5.0), ("k", 5.0, 5.5), ("m", 5.5, 6.0),
    ("と", 6.0, 6.1), ("1", 6.1, 6.5), (",", 6.5, 6.6), ("5", 6.6, 7.0), ("0", 7.0, 7.2),
    ("0", 7.2, 7.4), ("円", 7.4, 8.0), ("で", 8.0, 9.0), ("す", 9.0, 10.0), ("。", 10.0, 10.2),
]
CHAR_TEXT = "距離は3.3kmと1,500円です。"


def _chunk_words(rows, language):
    return [{"text": t, "start": s, "end": e, "language": language} for t, s, e in rows]


def _cue_words(rows):
    return [{"text": t, "timestamp_start": s, "timestamp_end": e} for t, s, e in rows]


def _is_digit_separator_boundary(left: str, right: str) -> bool:
    left, right = left.strip(), right.strip()
    return (
        len(left) >= 2
        and left[-1] in ".,"
        and left[-2].isdigit()
        and right[:1].isdigit()
    )


class TestChunks:
    def test_spec_addon_010_token_row_separator_does_not_end_a_chunk(self):
        chunks = _build_chunks_from_words(
            _chunk_words(TOKEN_ROWS, "en"), min_duration=2, max_duration=30
        )

        assert [(c["text"], c["start"], c["end"]) for c in chunks] == TOKEN_GROUPS

    def test_spec_addon_010_character_row_separator_does_not_end_a_chunk(self):
        chunks = _build_chunks_from_words(
            _chunk_words(CHAR_ROWS, "ja"), min_duration=2, max_duration=30
        )

        assert [c["text"] for c in chunks] == [CHAR_TEXT]

    def test_spec_addon_010_comma_separator_is_not_a_preferred_chunk_break(self):
        rows = [(f"w{i}", float(i), i + 1.0) for i in range(5)]
        rows += [("1,", 5.0, 6.0), ("500", 6.0, 7.0)]
        rows += [(f"v{i}", 7.0 + i, 8.0 + i) for i in range(12)]

        chunks = _build_chunks_from_words(
            _chunk_words(rows, "en"), min_duration=3, max_duration=8
        )

        assert not any(c["text"].endswith("1,") for c in chunks)

    def test_spec_addon_010_max_duration_still_bounds_a_run_of_separators(self):
        rows = []
        t = 0.0
        for _ in range(40):
            for ch in ("1", ".", "5"):
                rows.append((ch, t, t + 0.2))
                t += 0.2

        chunks = _build_chunks_from_words(
            _chunk_words(rows, "ja"), min_duration=2, max_duration=6
        )

        assert chunks
        assert "".join(c["text"] for c in chunks) == "1.5" * 40
        assert all(c["end"] - c["start"] <= 6.5 for c in chunks)


def _flat(text: str, language: str) -> str:
    return text.replace("\n", "") if language == "ja" else " ".join(text.split())


WIDE = CueConfig(max_duration=100, max_width=200)


class TestCues:
    def test_spec_addon_010_token_row_separator_does_not_end_a_cue(self):
        cues = build_cues(_cue_words(TOKEN_ROWS), language="en", config=WIDE)

        assert [(_flat(c["text"], "en"), c["start"], c["end"]) for c in cues] == TOKEN_GROUPS

    def test_spec_addon_010_character_row_separator_does_not_end_a_japanese_cue(self):
        cues = build_cues(_cue_words(CHAR_ROWS), language="ja", config=WIDE)

        assert [_flat(c["text"], "ja") for c in cues] == [CHAR_TEXT]

    @pytest.mark.parametrize(
        "language, rows, max_width",
        [
            (
                "en",
                [(w, i * 0.3, i * 0.3 + 0.3) for i, w in enumerate(
                    ["alpha", "bravo", "char", "1,", "500", "delta", "echo",
                     "foxtrot", "golf", "hotel", "india", "juliet", "kilo"]
                )],
                20,
            ),
            (
                "en",
                [(w, i * 0.3, i * 0.3 + 0.3) for i, w in enumerate(
                    ["alpha", "bravo", "char", "3.", "5km", "delta", "echo",
                     "foxtrot", "golf", "hotel", "india", "juliet", "kilo"]
                )],
                20,
            ),
            (
                "ja",
                [(ch, i * 0.2, i * 0.2 + 0.2) for i, ch in enumerate(
                    list("あいうえ") + ["3", ".", "3"] + list("かきくけこさしすせそたちつてと")
                )],
                14,
            ),
            (
                "ja",
                [(ch, i * 0.2, i * 0.2 + 0.2) for i, ch in enumerate(
                    list("あいうえ") + ["1", ",", "5", "0", "0"] + list("かきくけこさしすせそたちつてと")
                )],
                14,
            ),
        ],
        ids=["en-comma", "en-point", "ja-point", "ja-comma"],
    )
    def test_spec_addon_010_separator_is_not_a_safe_break_for_width(
        self, language, rows, max_width
    ):
        cues = build_cues(
            _cue_words(rows),
            language=language,
            config=CueConfig(max_duration=100, max_width=max_width),
        )

        assert len(cues) >= 2
        texts = [_flat(c["text"], language) for c in cues]
        for left, right in zip(texts, texts[1:]):
            assert not _is_digit_separator_boundary(left, right), (left, right)
