"""SPEC-ADDON-061: ``GET /files/{file_id}/find`` over HTTP, against a real search DB."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from app.extractors.base import ContentExtractor
from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    EPUB,
    OTHER,
    api,
    search_index,
    seed,
    statements,
    visual,
)

MANIFEST = Path(__file__).resolve().parent.parent / "manifest.json"
PDF = "application/pdf"
VIDEO = "video/mp4"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
ROW_CAP = 2000


def _path(file_id: str = "doc") -> str:
    return f"/files/{file_id}/find"


def _ok(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


def _find(api, q, file_id="doc", drive=DRIVE) -> dict:
    return _ok(api(_path(file_id), {"q": q}, drive=drive))


def _ascii_lower(text: str) -> str:
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in text)


def _marked(row) -> str:
    start = row["match_start"]
    return row["snippet"][start:start + row["match_length"]]


def _flag(seed, file_id: str, **flags) -> None:
    for column, value in flags.items():
        seed.index.sql(
            f"UPDATE indexed_files SET {column} = :v WHERE file_id = :f",
            v=1 if value else 0, f=file_id,
        )


def _doc(seed, file_id="doc", mime="text/markdown", name=None, drive=DRIVE, active=True):
    seed.file(file_id, name, mime=mime, drive=drive, active=active)
    _flag(seed, file_id, text_indexed=True)


def _video(seed, file_id="vid", drive=DRIVE, active=True):
    seed.file(file_id, f"{file_id}.mp4", mime=VIDEO, drive=drive, active=active)
    _flag(seed, file_id, whisper_indexed=True)


def _distinct(n: int, salt: str = "") -> str:
    """Text of length ``n`` with no repeated 20-character window, so no
    suffix of it equals a prefix of an unrelated chunk by chance."""
    out = []
    i = 0
    while len("".join(out)) < n:
        out.append(f"<{salt}{i:05d}>")
        i += 1
    return "".join(out)[:n]


def _chunked(unit: str, size: int, replay: int) -> list[str]:
    """Cuts ``unit`` into chunks of ``size`` whose heads replay the last
    ``replay`` characters of the previous chunk, as the extractors do."""
    chunks = []
    start = 0
    while True:
        chunks.append(unit[start:start + size])
        if start + size >= len(unit):
            return chunks
        start += size - replay


class TestTextUnits:
    @pytest.mark.parametrize(
        "replay,mime",
        [
            pytest.param(20, "text/markdown", id="text-20"),
            pytest.param(80, "text/markdown", id="text-80"),
            pytest.param(200, DOCX, id="office-200"),
        ],
    )
    def test_spec_addon_061_an_occurrence_in_a_replayed_overlap_counts_once(
        self, api, seed, visual, replay, mime,
    ):
        visual(enabled=True)
        _doc(seed, mime=mime)
        unit = _distinct(500, "a") + " needle " + _distinct(500, "b")
        at = unit.index("needle")
        chunks = _chunked(unit, at + replay - 2, replay)
        assert "needle" in chunks[0] and "needle" in chunks[1]
        for i, chunk in enumerate(chunks):
            seed.text_chunk("doc", i, chunk)

        body = _find(api, "needle")

        assert body["total"] == 1
        assert [r["ordinal"] for r in body["rows"]] == [1]

    def test_spec_addon_061_a_replay_longer_than_200_is_not_merged_so_its_occurrence_counts_twice(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _doc(seed)
        unit = _distinct(500, "a") + " needle " + _distinct(500, "b")
        at = unit.index("needle")
        chunks = _chunked(unit, at + 201 - 2, 201)
        assert "needle" in chunks[0] and "needle" in chunks[1]
        for i, chunk in enumerate(chunks):
            seed.text_chunk("doc", i, chunk)

        body = _find(api, "needle")

        assert body["total"] == 2

    def test_spec_addon_061_a_suffix_prefix_match_under_20_is_a_coincidence_and_both_count(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _doc(seed)
        seed.text_chunk("doc", 0, _distinct(100, "a") + "x needle y")
        seed.text_chunk("doc", 1, "x needle y" + _distinct(100, "b"))

        body = _find(api, "needle")

        assert body["total"] == 2
        assert [r["ordinal"] for r in body["rows"]] == [1, 2]

    def test_spec_addon_061_a_19_character_overlap_is_not_merged_and_20_is(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        nineteen = "ab needle cd efghij"
        twenty = "ab needle cd efghijk"
        assert (len(nineteen), len(twenty)) == (19, 20)
        for file_id, shared in (("short", nineteen), ("exact", twenty)):
            _doc(seed, file_id)
            seed.text_chunk(file_id, 0, _distinct(60, file_id) + shared)
            seed.text_chunk(file_id, 1, shared + _distinct(60, file_id + "z"))

        assert _find(api, "needle", "short")["total"] == 2
        assert _find(api, "needle", "exact")["total"] == 1

    def test_spec_addon_061_a_chunk_wholly_inside_the_previous_suffix_adds_nothing(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _doc(seed)
        tail = "the needle is in the last thirty"
        seed.text_chunk("doc", 0, _distinct(100, "a") + tail)
        seed.text_chunk("doc", 1, tail)

        body = _find(api, "needle")

        assert body["total"] == 1

    @pytest.mark.parametrize(
        "mime,unit,term,count",
        [
            pytest.param(
                "text/markdown", "- [ ] check the lighthouse lamp\n" * 120,
                "lamp", 120, id="repeated-checklist-line",
            ),
            pytest.param(
                "text/plain", "we will sing the chorus line tonight\n" * 40,
                "chorus", 40, id="repeated-chorus-line",
            ),
            pytest.param(
                "text/csv", "id,name,note\n" + "1,lighthouse,lamp\n" * 100,
                "lamp", 100, id="csv-of-identical-rows",
            ),
        ],
    )
    def test_spec_addon_061_every_repeated_line_across_a_chunk_boundary_is_counted(
        self, api, seed, visual, mime, unit, term, count,
    ):
        visual(enabled=True)
        _doc(seed, mime=mime)
        chunks = ContentExtractor.chunk_text(unit, overlap=80)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            seed.text_chunk("doc", i, chunk)

        body = _find(api, term)

        assert body["total"] == count
        assert len(body["rows"]) == count

    @pytest.mark.parametrize(
        "mime,flags",
        [
            pytest.param(DOCX, {"text_indexed": True}, id="office"),
            pytest.param(
                "image/png", {"text_indexed": False, "ocr_indexed": True},
                id="picture-ocr",
            ),
        ],
    )
    def test_spec_addon_061_office_and_picture_ocr_merge_the_chunk_text_default_replay(
        self, api, seed, visual, mime, flags,
    ):
        visual(enabled=True)
        seed.file("doc", mime=mime)
        _flag(seed, "doc", **flags)
        unit = "- [ ] check the lighthouse lamp\n" * 120
        chunks = ContentExtractor.chunk_text(unit)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            seed.text_chunk("doc", i, chunk)

        body = _find(api, "lamp")

        assert body["total"] == 120

    def test_spec_addon_061_chunks_are_merged_in_chunk_index_order_not_insertion_order(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _doc(seed)
        unit = _distinct(150, "a") + " first needle " + _distinct(150, "b") + " second needle"
        chunks = _chunked(unit, 120, 80)
        for i in reversed(range(len(chunks))):
            seed.text_chunk("doc", i, chunks[i])

        body = _find(api, "needle")

        assert body["total"] == 2
        assert [r["snippet"].endswith("needle") for r in body["rows"]] == [False, True]

    def test_spec_addon_061_twelve_or_more_chunks_merge_in_numeric_chunk_index_order(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _doc(seed)
        unit = _distinct(100, "a") + " needle " + _distinct(700, "b")
        chunks = _chunked(unit, 120, 80)
        assert len(chunks) >= 12
        for i, chunk in enumerate(chunks):
            seed.text_chunk("doc", i, chunk)

        body = _find(api, "needle")

        assert body["total"] == 1
        assert [r["ordinal"] for r in body["rows"]] == [1]

    def test_spec_addon_061_an_occurrence_across_a_newline_join_is_not_found(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _doc(seed)
        seed.text_chunk("doc", 0, _distinct(60, "a") + " nee")
        seed.text_chunk("doc", 1, "dle " + _distinct(60, "b"))

        body = _find(api, "needle")

        assert body["total"] == 0
        assert body["rows"] == []

    def test_spec_addon_061_the_snippet_reaches_into_the_next_chunk_of_the_same_unit(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _doc(seed)
        unit = _distinct(200, "a") + "needle" + _distinct(200, "b")
        chunks = _chunked(unit, 210, 80)
        for i, chunk in enumerate(chunks):
            seed.text_chunk("doc", i, chunk)
        at = unit.index("needle")

        (row,) = _find(api, "needle")["rows"]

        assert row["snippet"] == "…" + unit[at - 80:at + 6 + 80] + "…"
        assert (row["match_start"], row["match_length"]) == (81, 6)


class TestTextKinds:
    @pytest.mark.parametrize(
        "mime,stored_page,kind,location",
        [
            pytest.param(PDF, "3", "page", {"page": 3}, id="pdf-page"),
            pytest.param(EPUB, "3", "section",
                         {"section": 3, "section_title": "Chapter Three"}, id="epub-with-title"),
            pytest.param(EPUB, "4", "section",
                         {"section": 4, "section_title": None}, id="epub-without-title"),
            pytest.param(EPUB, "", "text", {}, id="epub-without-page"),
            pytest.param(PDF, "", "text", {}, id="pdf-without-page"),
            pytest.param("image/png", "", "picture_text", {}, id="picture"),
            pytest.param("text/markdown", "", "text", {}, id="markdown"),
            pytest.param(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "", "text", {}, id="office",
            ),
        ],
    )
    def test_spec_addon_061_a_text_row_is_classified_as_the_passage_view_classifies_it(
        self, api, seed, visual, mime, stored_page, kind, location,
    ):
        visual(enabled=True)
        _doc(seed, mime=mime)
        if mime == "image/png":
            _flag(seed, "doc", text_indexed=False, ocr_indexed=True)
        seed.section("doc", 3, "Chapter Three")
        seed.text_chunk("doc", 0, "a needle", page=stored_page)

        (row,) = _find(api, "needle")["rows"]

        assert row["kind"] == kind
        for key, value in location.items():
            assert row[key] == value

    def test_spec_addon_061_each_page_is_its_own_unit_numbered_and_ordered_by_page(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _doc(seed, mime=PDF, name="paper.pdf")
        seed.text_chunk("doc", 0, "needle on ten and needle again", page="10")
        seed.text_chunk("doc", 1, "page two needle", page="2")
        seed.text_chunk("doc", 2, "page three has none", page="3")

        body = _find(api, "needle")

        assert [(r["kind"], r["page"], r["ordinal"]) for r in body["rows"]] == [
            ("page", 2, 1), ("page", 10, 1), ("page", 10, 2),
        ]
        assert body["total"] == 3

    def test_spec_addon_061_an_occurrence_across_two_pages_is_not_found(self, api, seed, visual):
        visual(enabled=True)
        _doc(seed, mime=PDF, name="paper.pdf")
        seed.text_chunk("doc", 0, _distinct(40, "a") + " nee", page="1")
        seed.text_chunk("doc", 1, "dle " + _distinct(40, "b"), page="2")

        assert _find(api, "needle")["total"] == 0

    def test_spec_addon_061_sections_are_ordered_by_section_then_ordinal(self, api, seed, visual):
        visual(enabled=True)
        _doc(seed, mime=EPUB, name="book.epub")
        seed.section("doc", 5, "Five")
        seed.text_chunk("doc", 0, "needle and needle", page="5")
        seed.text_chunk("doc", 1, "a needle", page="1")

        body = _find(api, "needle")

        assert [(r["section"], r["section_title"], r["ordinal"]) for r in body["rows"]] == [
            (1, None, 1), (5, "Five", 1), (5, "Five", 2),
        ]

    def test_spec_addon_061_a_section_title_comes_from_the_files_own_sections(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        for file_id, drive, name, title in (
            ("first", DRIVE, "first.epub", "First Chapter Three"),
            ("second", OTHER, "second.epub", "Second Chapter Three"),
        ):
            _doc(seed, file_id, mime=EPUB, name=name, drive=drive)
            seed.section(file_id, 3, title)
            seed.text_chunk(file_id, 0, "needle", page="3")

        first = _find(api, "needle", "first", drive=DRIVE)
        second = _find(api, "needle", "second", drive=OTHER)

        assert [r["section_title"] for r in first["rows"]] == ["First Chapter Three"]
        assert [r["section_title"] for r in second["rows"]] == ["Second Chapter Three"]


class TestTranscript:
    def test_spec_addon_061_each_occurrence_in_a_chunk_is_a_row_with_the_chunk_times(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _video(seed)
        seed.transcript("vid", 0, "a needle and a needle", times=(1.5, 4.0))
        seed.transcript("vid", 1, "nothing", times=(4.0, 8.0))
        seed.transcript("vid", 2, "needle", times=(8.0, 12.0))

        body = _find(api, "needle", "vid")

        assert [
            (r["kind"], r["start"], r["end"], r["ordinal"]) for r in body["rows"]
        ] == [
            ("utterance", 1.5, 4.0, 1),
            ("utterance", 1.5, 4.0, 2),
            ("utterance", 8.0, 12.0, 1),
        ]
        assert body["total"] == 3

    def test_spec_addon_061_a_transcript_row_with_no_transcript_chunk_is_skipped_and_not_counted(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _video(seed)
        seed.transcript("vid", 0, "needle with no chunk row")
        seed.transcript("vid", 2, "needle kept", times=(9.0, 10.0))

        body = _find(api, "needle", "vid")

        assert [r["start"] for r in body["rows"]] == [9.0]
        assert body["total"] == 1

    def test_spec_addon_061_an_occurrence_across_two_transcript_chunks_is_not_found(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _video(seed)
        seed.transcript("vid", 0, "the nee", times=(0.0, 1.0))
        seed.transcript("vid", 1, "dle", times=(1.0, 2.0))

        assert _find(api, "needle", "vid")["total"] == 0


class TestScenes:
    def test_spec_addon_061_screen_text_is_searched_and_the_scene_label_never(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _video(seed)
        run = seed.run("vid")
        seed.scene(run, 5.0, 9.0, label="needle on a desk", visible="no text")
        seed.scene(run, 9.0, None, label="a sign", visible="Needle Street needle")

        body = _find(api, "needle", "vid")

        assert [
            (r["kind"], r["start"], r["end"], r["ordinal"]) for r in body["rows"]
        ] == [("scene", 9.0, None, 1), ("scene", 9.0, None, 2)]

    @pytest.mark.parametrize(
        "enabled,policy",
        [
            pytest.param(False, None, id="feature-off"),
            pytest.param(True, {DRIVE: False}, id="drive-policy-off"),
            pytest.param(True, {DRIVE: RuntimeError("policy unavailable")},
                         id="drive-policy-unreadable"),
        ],
    )
    def test_spec_addon_061_scenes_add_nothing_while_the_visual_index_is_closed(
        self, api, seed, visual, enabled, policy,
    ):
        _video(seed)
        seed.transcript("vid", 0, "needle spoken", times=(1.0, 2.0))
        seed.scene(seed.run("vid"), 5.0, 9.0, visible="needle")
        visual(enabled=enabled, policy=policy)

        body = _find(api, "needle", "vid")

        assert [r["kind"] for r in body["rows"]] == ["utterance"]
        assert body["total"] == 1

    def test_spec_addon_061_scenes_of_an_inactive_run_are_not_searched(self, api, seed, visual):
        visual(enabled=True)
        _video(seed)
        seed.scene(seed.run("vid", "vvr_old", active=False), 1.0, 2.0, visible="needle old")
        seed.scene(seed.run("vid", "vvr_new"), 3.0, 4.0, visible="needle new")

        body = _find(api, "needle", "vid")

        assert [r["start"] for r in body["rows"]] == [3.0]

    def test_spec_addon_061_utterances_and_scenes_interleave_by_start_utterance_first_on_a_tie(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _video(seed)
        seed.transcript("vid", 0, "needle late", times=(40.0, 50.0))
        seed.transcript("vid", 1, "needle and needle", times=(5.0, 10.0))
        run = seed.run("vid")
        seed.scene(run, 5.0, 8.0, visible="needle needle")
        seed.scene(run, 20.0, 30.0, visible="needle")

        body = _find(api, "needle", "vid")

        assert [(r["kind"], r["start"], r["ordinal"]) for r in body["rows"]] == [
            ("utterance", 5.0, 1),
            ("utterance", 5.0, 2),
            ("scene", 5.0, 1),
            ("scene", 5.0, 2),
            ("scene", 20.0, 1),
            ("utterance", 40.0, 1),
        ]


class TestMatching:
    @pytest.mark.parametrize(
        "stored,q,count",
        [
            pytest.param("abc ABC aBc", "ABC", 3, id="ascii-folds-both-ways"),
            pytest.param("Ärger", "Ärger", 1, id="non-ascii-same-form"),
            pytest.param("ärger", "Ärger", 0, id="non-ascii-upper-q-not-folded"),
            pytest.param("Ärger", "ärger", 0, id="non-ascii-upper-text-not-folded"),
            pytest.param("ＡＢＣ", "ａｂｃ", 0, id="fullwidth-not-folded"),
            pytest.param("Straße", "strasse", 0, id="no-unicode-expansion"),
            pytest.param("Äb äb ÄB", "äb", 1, id="mixed-ascii-and-non-ascii"),
            pytest.param("aaaa", "aa", 2, id="non-overlapping-even"),
            pytest.param("aaa", "aa", 1, id="non-overlapping-odd"),
            pytest.param("100% a_b", "%", 1, id="percent-is-literal"),
            pytest.param("100% a_b", "_", 1, id="underscore-is-literal"),
            pytest.param("猫が猫を見た", "猫", 2, id="cjk"),
        ],
    )
    def test_spec_addon_061_ascii_letters_fold_and_every_other_character_is_exact(
        self, api, seed, visual, stored, q, count,
    ):
        visual(enabled=True)
        _doc(seed)
        seed.text_chunk("doc", 0, stored)

        body = _find(api, q)

        assert body["total"] == count
        assert len(body["rows"]) == count
        assert [r["ordinal"] for r in body["rows"]] == list(range(1, count + 1))
        for row in body["rows"]:
            assert _ascii_lower(_marked(row)) == _ascii_lower(q.strip())

    def test_spec_addon_061_q_is_stripped_before_matching_and_echoed_stripped(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _doc(seed)
        seed.text_chunk("doc", 0, "a needle, a needle")

        plain = _find(api, "needle")
        padded = _find(api, "  needle \t")

        assert padded["query"] == "needle"
        assert padded["rows"] == plain["rows"]
        assert padded["total"] == 2


def _filler(n: int, ch: str) -> str:
    return ch * n


class TestSnippet:
    @pytest.mark.parametrize(
        "before,after,lead,trail",
        [
            pytest.param(10, 10, False, False, id="short-both-sides"),
            pytest.param(80, 80, False, False, id="exactly-80-each-side"),
            pytest.param(81, 81, True, True, id="81-each-side"),
            pytest.param(0, 300, False, True, id="at-the-start"),
            pytest.param(300, 0, True, False, id="at-the-end"),
        ],
    )
    def test_spec_addon_061_snippet_keeps_80_characters_each_side_and_marks_where_it_is_cut(
        self, api, seed, visual, before, after, lead, trail,
    ):
        visual(enabled=True)
        _doc(seed)
        seed.text_chunk("doc", 0, _filler(before, "a") + "Needle" + _filler(after, "b"))

        (row,) = _find(api, "needle")["rows"]

        assert row["snippet"] == (
            ("…" if lead else "")
            + _filler(min(before, 80), "a") + "Needle" + _filler(min(after, 80), "b")
            + ("…" if trail else "")
        )
        assert row["match_start"] == min(before, 80) + (1 if lead else 0)
        assert row["match_length"] == 6
        assert _marked(row) == "Needle"

    def test_spec_addon_061_each_occurrence_gets_a_snippet_around_itself(self, api, seed, visual):
        visual(enabled=True)
        _doc(seed)
        stored = _filler(100, "a") + "NEEDLE" + _filler(100, "b") + "needle" + "ccc"
        seed.text_chunk("doc", 0, stored)

        first, second = _find(api, "needle")["rows"]

        assert _marked(first) == "NEEDLE"
        assert _marked(second) == "needle"
        second_at = stored.index("needle")
        assert second["snippet"] == "…" + stored[second_at - 80:]
        assert second["match_start"] == 81


class TestTotalsAndCap:
    @pytest.mark.parametrize(
        "occurrences,truncated",
        [
            pytest.param(1, False, id="one"),
            pytest.param(ROW_CAP, False, id="at-the-cap"),
            pytest.param(ROW_CAP + 1, True, id="one-past-the-cap"),
        ],
    )
    def test_spec_addon_061_rows_are_capped_at_2000_and_total_counts_every_occurrence(
        self, api, seed, visual, occurrences, truncated,
    ):
        visual(enabled=True)
        _doc(seed)
        seed.text_chunk("doc", 0, "needle " * occurrences)

        body = _find(api, "needle")

        assert body["total"] == occurrences
        assert body["truncated"] is truncated
        assert len(body["rows"]) == min(occurrences, ROW_CAP)
        assert [r["ordinal"] for r in body["rows"]] == list(
            range(1, min(occurrences, ROW_CAP) + 1)
        )

    def test_spec_addon_061_the_cap_keeps_the_first_rows_in_position_order(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _doc(seed, mime=PDF, name="paper.pdf")
        seed.text_chunk("doc", 0, "needle " * 5, page="9")
        seed.text_chunk("doc", 1, "needle " * ROW_CAP, page="1")

        body = _find(api, "needle")

        assert body["total"] == ROW_CAP + 5
        assert body["truncated"] is True
        assert {r["page"] for r in body["rows"]} == {1}

    def test_spec_addon_061_no_hit_on_an_indexed_file_is_200_with_zero_total(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        _doc(seed)
        seed.text_chunk("doc", 0, "only hay")

        body = _find(api, "needle")

        assert body == {
            "query": "needle", "indexed": True, "total": 0, "truncated": False, "rows": [],
        }


class TestIndexed:
    @pytest.mark.parametrize(
        "mime,flags,indexed",
        [
            pytest.param("text/markdown", {}, False, id="no-flag"),
            pytest.param("text/markdown", {"text_indexed": True}, True, id="text"),
            pytest.param(VIDEO, {"whisper_indexed": True}, True, id="whisper"),
            pytest.param("image/png", {"ocr_indexed": True}, True, id="ocr"),
        ],
    )
    def test_spec_addon_061_indexed_follows_the_files_index_flags(
        self, api, seed, visual, mime, flags, indexed,
    ):
        visual(enabled=True)
        seed.file("doc", mime=mime)
        _flag(seed, "doc", **flags)

        assert _find(api, "needle")["indexed"] is indexed

    @pytest.mark.parametrize(
        "mime,flags,indexed",
        [
            pytest.param(PDF, {"text_indexed": True}, True, id="pdf-own-stage-done"),
            pytest.param(
                PDF,
                {"text_indexed": False, "whisper_indexed": True, "ocr_indexed": True},
                False,
                id="pdf-own-stage-not-done",
            ),
            pytest.param("image/png", {"ocr_indexed": True}, True, id="image-own-stage-done"),
            pytest.param(
                "image/png",
                {"ocr_indexed": False, "text_indexed": True},
                False,
                id="image-own-stage-not-done",
            ),
            pytest.param(VIDEO, {"whisper_indexed": True}, True, id="video-own-stage-done"),
            pytest.param(
                VIDEO,
                {"whisper_indexed": False, "text_indexed": True, "ocr_indexed": True},
                False,
                id="video-own-stage-not-done",
            ),
            pytest.param("audio/mpeg", {"whisper_indexed": True}, True, id="audio-own-stage-done"),
            pytest.param(
                "audio/mpeg",
                {"whisper_indexed": False, "text_indexed": True, "ocr_indexed": True},
                False,
                id="audio-own-stage-not-done",
            ),
        ],
    )
    def test_spec_addon_061_indexed_follows_the_stage_of_the_files_kind(
        self, api, seed, visual, mime, flags, indexed,
    ):
        visual(enabled=True)
        seed.file("doc", mime=mime)
        _flag(seed, "doc", **flags)

        assert _find(api, "needle")["indexed"] is indexed

    @pytest.mark.parametrize(
        "enabled,indexed",
        [
            pytest.param(True, True, id="open"),
            pytest.param(False, False, id="closed"),
        ],
    )
    def test_spec_addon_061_a_video_whose_whisper_stage_is_unfinished_is_indexed_by_its_active_visual_run(
        self, api, seed, visual, enabled, indexed,
    ):
        visual(enabled=enabled)
        seed.file("vid", "vid.mp4", mime=VIDEO)
        _flag(seed, "vid", whisper_indexed=False, text_indexed=True, ocr_indexed=True)
        seed.run("vid")

        assert _find(api, "needle", "vid")["indexed"] is indexed

    @pytest.mark.parametrize(
        "enabled,active,indexed",
        [
            pytest.param(True, True, True, id="open-and-active"),
            pytest.param(False, True, False, id="closed"),
            pytest.param(True, False, False, id="inactive-run"),
        ],
    )
    def test_spec_addon_061_an_active_visual_index_run_counts_only_while_the_index_is_open(
        self, api, seed, visual, enabled, active, indexed,
    ):
        visual(enabled=enabled)
        seed.file("vid", "vid.mp4", mime=VIDEO)
        seed.run("vid", active=active)

        assert _find(api, "needle", "vid")["indexed"] is indexed

    def test_spec_addon_061_a_file_with_no_indexed_files_row_is_not_indexed_and_has_no_rows(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        seed.text_chunk("ghost", 0, "a needle")

        body = _find(api, "needle", "ghost")

        assert body["indexed"] is False
        assert body["rows"] == []
        assert body["total"] == 0


class TestScope:
    def test_spec_addon_061_only_the_requested_files_rows_are_returned(self, api, seed, visual):
        visual(enabled=True)
        _video(seed, "mine")
        seed.transcript("mine", 0, "needle", times=(0.0, 1.0))
        seed.scene(seed.run("mine"), 1.0, 2.0, visible="needle")
        _video(seed, "next")
        seed.transcript("next", 0, "needle needle needle", times=(7.0, 8.0))
        seed.scene(seed.run("next"), 9.0, 10.0, visible="needle needle")

        mine = _find(api, "needle", "mine")
        other = _find(api, "needle", "next")

        assert [(r["kind"], r["start"]) for r in mine["rows"]] == [
            ("utterance", 0.0), ("scene", 1.0),
        ]
        assert mine["total"] == 2
        assert other["total"] == 5

    @pytest.mark.parametrize(
        "drive,active",
        [
            pytest.param(OTHER, True, id="file-on-another-drive"),
            pytest.param(DRIVE, False, id="inactive-file"),
        ],
    )
    def test_spec_addon_061_a_file_outside_the_drive_or_inactive_has_no_rows(
        self, api, seed, visual, drive, active,
    ):
        visual(enabled=True)
        _video(seed, "vid", drive=drive, active=active)
        seed.transcript("vid", 0, "needle", times=(0.0, 1.0))
        seed.scene(seed.run("vid"), 1.0, 2.0, visible="needle")

        body = _find(api, "needle", "vid", drive=DRIVE)

        assert body["rows"] == []
        assert body["total"] == 0


class TestRejections:
    @pytest.mark.parametrize(
        "params",
        [
            pytest.param({}, id="no-q"),
            pytest.param({"q": ""}, id="empty-q"),
            pytest.param({"q": "   "}, id="blank-q"),
            pytest.param({"q": "x" * 201}, id="q-201"),
        ],
    )
    def test_spec_addon_061_an_empty_or_too_long_q_is_422_and_not_logged(
        self, api, seed, visual, caplog, params,
    ):
        visual(enabled=True)
        _doc(seed)
        caplog.set_level(logging.DEBUG)

        response = api(_path(), params)

        assert response.status_code == 422, response.text
        assert not [
            r for r in caplog.records
            if r.name.startswith("app") and r.levelno >= logging.WARNING
        ]

    @pytest.mark.parametrize(
        "q",
        [
            pytest.param("n", id="q-1"),
            pytest.param("n" * 200, id="q-200"),
            pytest.param("  " + "n" * 200 + "  ", id="q-200-after-strip"),
        ],
    )
    def test_spec_addon_061_q_of_1_to_200_characters_after_stripping_is_accepted(
        self, api, seed, visual, q,
    ):
        visual(enabled=True)
        _doc(seed)
        seed.text_chunk("doc", 0, "n" * 210)

        body = _find(api, q)

        assert body["query"] == q.strip()
        assert body["total"] == 210 // len(q.strip())


class TestSideEffects:
    def test_spec_addon_061_a_find_request_writes_nothing(self, api, seed, visual, statements):
        visual(enabled=True)
        _video(seed)
        seed.transcript("vid", 0, "needle", times=(0.0, 1.0))
        seed.scene(seed.run("vid"), 1.0, 2.0, visible="needle")

        statements.reset()
        body = _find(api, "needle", "vid")

        assert body["total"] == 2
        assert statements.statements, "the route must read the search DB"
        assert statements.writes() == []
        assert statements.on_loop == []


class TestRouteDeclaration:
    def test_spec_addon_061_manifest_declares_the_route_file_scoped_behind_the_search_feature(self):
        manifest = json.loads(MANIFEST.read_text())
        entries = [
            r for r in manifest["proxy"]["routes"] if r.get("path") == "/files/{file_id}/find"
        ]

        assert len(entries) == 1
        entry = entries[0]
        assert entry["methods"] == ["GET"]
        assert entry["pre_check"] == {"type": "file_access", "param": "file_id"}
        assert entry["addon_feature"] == "search"
        assert "response_filter" in entry
        assert entry["response_filter"] is None
        assert entry["timeout"] == 30
        assert "drive_optional" not in entry
