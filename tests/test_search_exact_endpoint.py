"""SPEC-ADDON-032: ``GET /search/exact`` over HTTP, against a real search DB."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

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

PATH = "/search/exact"
MANIFEST = Path(__file__).resolve().parent.parent / "manifest.json"
SEGMENT_KEYS = {
    "file_id", "filename", "source", "chunk_index", "page", "section",
    "section_title", "start_time", "end_time", "field", "match_count", "snippet",
}


def _ok(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


def _where(body) -> list[tuple]:
    return [
        (s["file_id"], s["source"], s["chunk_index"], s["start_time"], s["field"])
        for s in body["segments"]
    ]


@pytest.fixture()
def corpus(seed, visual):
    visual(enabled=True)
    seed.file("doc", "notes.md")
    seed.text_chunk("doc", 0, "a needle here")
    seed.text_chunk("doc", 1, "nothing to see")
    seed.text_chunk("doc", 2, "NEEDLE in capitals")
    seed.file("vid", "talk.mp4", mime="video/mp4")
    seed.transcript("vid", 0, "the needle was spoken", times=(1.0, 2.0))
    seed.transcript("vid", 1, "silence", times=(2.0, 3.0))
    run = seed.run("vid")
    seed.scene(run, 5.0, 9.0, label="needle on a desk", visible="no text")
    seed.scene(run, 9.0, 12.0, label="a sign", visible="Needle Street")
    seed.scene(run, 12.0, None, label="a cat", visible=None, excerpt="needle")
    return seed


class TestComparison:
    def test_spec_addon_032_lists_every_matching_segment_of_the_three_sources(
        self, api, corpus,
    ):
        body = _ok(api(PATH, {"q": "needle"}))

        assert body["total"] == 5
        assert _where(body) == [
            ("doc", "text", 0, None, None),
            ("doc", "text", 2, None, None),
            ("vid", "transcript", 0, 1.0, None),
            ("vid", "scene", None, 5.0, "scene_label"),
            ("vid", "scene", None, 9.0, "visible_text"),
        ]

    @pytest.mark.parametrize(
        "stored,q,count",
        [
            pytest.param("abc ABC aBc", "ABC", 3, id="ascii-folds-both-ways"),
            pytest.param("Ärger", "Ärger", 1, id="non-ascii-same-form"),
            pytest.param("ärger", "Ärger", None, id="non-ascii-upper-q-not-folded"),
            pytest.param("Ärger", "ärger", None, id="non-ascii-upper-text-not-folded"),
            pytest.param("ＡＢＣ", "ａｂｃ", None, id="fullwidth-not-folded"),
            pytest.param("Straße", "strasse", None, id="no-unicode-expansion"),
            pytest.param("Äb äb ÄB", "äb", 1, id="mixed-ascii-and-non-ascii"),
            pytest.param("aaaa", "aa", 2, id="non-overlapping-even"),
            pytest.param("aaa", "aa", 1, id="non-overlapping-odd"),
        ],
    )
    def test_spec_addon_032_ascii_letters_fold_and_every_other_character_is_exact(
        self, api, seed, visual, stored, q, count,
    ):
        visual(enabled=True)
        seed.file("doc")
        seed.text_chunk("doc", 0, stored)

        body = _ok(api(PATH, {"q": q}))

        if count is None:
            assert body["total"] == 0
            assert body["segments"] == []
        else:
            assert body["total"] == 1
            assert [s["match_count"] for s in body["segments"]] == [count]

    @pytest.mark.parametrize(
        "q,expected",
        [
            pytest.param("%", ["pct"], id="percent"),
            pytest.param("_", ["under"], id="underscore"),
            pytest.param("\\", ["slash"], id="backslash"),
            pytest.param("0%", ["pct"], id="percent-after-text"),
            pytest.param("a_b", ["under"], id="underscore-inside"),
            pytest.param("*", [], id="star-matches-nothing-absent"),
        ],
    )
    def test_spec_addon_032_no_character_of_q_is_a_wildcard(self, api, seed, visual, q, expected):
        visual(enabled=True)
        for file_id, body in (
            ("pct", "100% sure"), ("under", "a_b c"), ("slash", "back\\slash"),
            ("plain", "a plain a-b 10 b line"),
        ):
            seed.file(file_id)
            seed.text_chunk(file_id, 0, body)

        result = _ok(api(PATH, {"q": q}))

        assert [s["file_id"] for s in result["segments"]] == expected
        assert result["total"] == len(expected)

    @pytest.mark.parametrize(
        "q",
        [
            pytest.param("z", id="one-ascii-char"),
            pytest.param("猫", id="one-cjk-char"),
            pytest.param("猫が", id="two-cjk-chars"),
            pytest.param("qz", id="two-ascii-chars"),
            pytest.param("x" * 199 + "q", id="two-hundred-chars"),
        ],
    )
    def test_spec_addon_032_finds_q_of_every_length(self, api, seed, visual, q):
        visual(enabled=True)
        seed.file("hit")
        seed.text_chunk("hit", 0, f"before {q} after")
        seed.file("miss")
        seed.text_chunk("miss", 0, "before after")

        body = _ok(api(PATH, {"q": q}))

        assert [s["file_id"] for s in body["segments"]] == ["hit"]
        assert body["segments"][0]["match_count"] == 1

    def test_spec_addon_032_q_is_stripped_before_matching_and_echoed_stripped(
        self, api, corpus,
    ):
        plain = _ok(api(PATH, {"q": "needle"}))
        padded = _ok(api(PATH, {"q": "  needle \t"}))

        assert padded["query"] == "needle"
        assert padded["segments"] == plain["segments"]
        assert padded["total"] == plain["total"] == 5

    def test_spec_addon_032_an_occurrence_in_a_chunk_overlap_is_listed_in_both_chunks(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        seed.file("doc")
        seed.text_chunk("doc", 0, "first chunk ends with overlap needle text")
        seed.text_chunk("doc", 1, "overlap needle text then the second chunk")

        body = _ok(api(PATH, {"q": "needle"}))

        assert [s["chunk_index"] for s in body["segments"]] == [0, 1]
        assert body["total"] == 2


class TestScope:
    def test_spec_addon_032_other_drives_and_inactive_files_are_never_listed(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        for file_id, drive, active in (
            ("mine", DRIVE, True), ("theirs", OTHER, True), ("gone", DRIVE, False),
        ):
            seed.file(file_id, f"{file_id}.mp4", drive=drive, active=active, mime="video/mp4")
            seed.text_chunk(file_id, 0, "needle")
            seed.transcript(file_id, 0, "needle", times=(0.0, 1.0))
            seed.scene(seed.run(file_id), 1.0, 2.0, label="needle", visible="needle")

        body = _ok(api(PATH, {"q": "needle"}))

        assert {s["file_id"] for s in body["segments"]} == {"mine"}
        assert body["total"] == 4

        theirs = _ok(api(PATH, {"q": "needle"}, drive=OTHER))
        assert {s["file_id"] for s in theirs["segments"]} == {"theirs"}

    def test_spec_addon_032_scenes_of_an_inactive_run_are_not_listed(self, api, seed, visual):
        visual(enabled=True)
        seed.file("vid", "v.mp4", mime="video/mp4")
        old = seed.run("vid", "vvr_old", active=False)
        seed.scene(old, 1.0, 2.0, label="needle old", visible="needle old")
        new = seed.run("vid", "vvr_new", active=True)
        seed.scene(new, 3.0, 4.0, label="needle new", visible=None)

        body = _ok(api(PATH, {"q": "needle", "sources": "scene"}))

        assert [(s["start_time"], s["field"]) for s in body["segments"]] == [
            (3.0, "scene_label"),
        ]

    def test_spec_addon_032_other_stored_text_is_not_searched(self, api, seed, visual):
        visual(enabled=True)
        seed.file("vid", "v.mp4", mime="video/mp4")
        seed.file_metadata("vid", description="a needle description", tags="needle")
        seed.scene(
            seed.run("vid"), 1.0, 2.0, label="a cat", visible="a mat",
            excerpt="needle in the excerpt", description="needle in the v1 description",
        )

        body = _ok(api(PATH, {"q": "needle"}))

        assert body["total"] == 0
        assert body["segments"] == []


class TestSources:
    @pytest.mark.parametrize(
        "sources,echo,listed",
        [
            pytest.param(None, ["text", "transcript", "scene"],
                         ["text", "text", "transcript", "scene", "scene"], id="default"),
            pytest.param("text", ["text"], ["text", "text"], id="text"),
            pytest.param("transcript", ["transcript"], ["transcript"], id="transcript"),
            pytest.param("scene", ["scene"], ["scene", "scene"], id="scene"),
            pytest.param("scene,text", ["text", "scene"],
                         ["text", "text", "scene", "scene"], id="canonical-order"),
            pytest.param("scene,transcript,scene,scene", ["transcript", "scene"],
                         ["transcript", "scene", "scene"], id="duplicates"),
        ],
    )
    def test_spec_addon_032_sources_select_and_are_echoed_deduplicated_in_canonical_order(
        self, api, corpus, sources, echo, listed,
    ):
        params = {"q": "needle"}
        if sources is not None:
            params["sources"] = sources

        body = _ok(api(PATH, params))

        assert body["sources"] == echo
        assert [s["source"] for s in body["segments"]] == listed
        assert body["total"] == len(listed)


class TestVisualIndexGate:
    @pytest.mark.parametrize(
        "enabled,policy",
        [
            pytest.param(False, None, id="feature-off"),
            pytest.param(True, {DRIVE: False}, id="drive-policy-off"),
            pytest.param(True, {DRIVE: RuntimeError("policy unavailable")},
                         id="drive-policy-unreadable"),
        ],
    )
    def test_spec_addon_032_scene_source_contributes_nothing_and_the_rest_is_still_searched(
        self, api, seed, visual, enabled, policy,
    ):
        seed.file("doc", "notes.md")
        seed.text_chunk("doc", 0, "a needle here")
        seed.file("vid", "talk.mp4", mime="video/mp4")
        seed.transcript("vid", 0, "needle spoken", times=(1.0, 2.0))
        seed.scene(seed.run("vid"), 5.0, 9.0, label="needle", visible="needle")
        visual(enabled=enabled, policy=policy)

        everything = _ok(api(PATH, {"q": "needle"}))
        scenes_only = _ok(api(PATH, {"q": "needle", "sources": "scene"}))

        assert [s["source"] for s in everything["segments"]] == ["text", "transcript"]
        assert everything["total"] == 2
        assert scenes_only["segments"] == []
        assert scenes_only["total"] == 0

    def test_spec_addon_032_another_drives_policy_does_not_close_this_drives_scenes(
        self, api, seed, visual,
    ):
        seed.file("vid", "talk.mp4", mime="video/mp4")
        seed.scene(seed.run("vid"), 5.0, 9.0, label="needle", visible=None)
        visual(enabled=True, policy={OTHER: False})

        body = _ok(api(PATH, {"q": "needle"}))

        assert [s["source"] for s in body["segments"]] == ["scene"]


class TestOrderAndPaging:
    @pytest.fixture()
    def ordered(self, seed, visual):
        visual(enabled=True)
        seed.file("f-b", "same.md")
        seed.text_chunk("f-b", 0, "needle")
        seed.file("f-a", "same.md")
        seed.text_chunk("f-a", 0, "needle")
        seed.file("lower", "alpha.md")
        seed.text_chunk("lower", 0, "needle")
        seed.file("upper", "Zeta.mp4", mime="video/mp4")
        seed.text_chunk("upper", 10, "needle ten")
        seed.text_chunk("upper", 2, "needle two")
        seed.transcript("upper", 11, "needle eleven", times=(30.0, 31.0))
        seed.transcript("upper", 3, "needle three", times=(10.0, 11.0))
        run = seed.run("upper")
        seed.scene(run, 9.0, None, label="needle c", visible="needle c", ordering=0)
        seed.scene(run, 5.0, 9.0, label="needle a", visible="needle a", ordering=2)
        seed.scene(run, 5.0, 9.0, label="needle b", visible="needle b", ordering=1)
        return [
            ("upper", "text", 2, None, None),
            ("upper", "text", 10, None, None),
            ("upper", "transcript", 3, 10.0, None),
            ("upper", "transcript", 11, 30.0, None),
            ("upper", "scene", None, 5.0, "scene_label"),
            ("upper", "scene", None, 5.0, "visible_text"),
            ("upper", "scene", None, 5.0, "scene_label"),
            ("upper", "scene", None, 5.0, "visible_text"),
            ("upper", "scene", None, 9.0, "scene_label"),
            ("upper", "scene", None, 9.0, "visible_text"),
            ("lower", "text", 0, None, None),
            ("f-a", "text", 0, None, None),
            ("f-b", "text", 0, None, None),
        ]

    def test_spec_addon_032_order_is_filename_file_id_source_then_position(self, api, ordered):
        body = _ok(api(PATH, {"q": "needle"}))

        assert _where(body) == ordered
        assert [s["snippet"] for s in body["segments"][4:8]] == [
            "needle b", "needle b", "needle a", "needle a",
        ]

    @pytest.mark.parametrize("limit", [1, 2, 3, 5, 13])
    def test_spec_addon_032_paging_by_limit_visits_each_segment_once_in_order(
        self, api, ordered, limit,
    ):
        seen = []
        for offset in range(0, len(ordered), limit):
            body = _ok(api(PATH, {"q": "needle", "limit": limit, "offset": offset}))
            assert body["total"] == len(ordered)
            assert body["offset"] == offset
            assert len(body["segments"]) == min(limit, len(ordered) - offset)
            seen.extend(_where(body))

        assert seen == ordered

    @pytest.mark.parametrize("offset", [13, 14, 1000])
    def test_spec_addon_032_offset_at_or_past_total_is_empty_with_the_real_total(
        self, api, ordered, offset,
    ):
        body = _ok(api(PATH, {"q": "needle", "offset": offset}))

        assert body["segments"] == []
        assert body["total"] == len(ordered)
        assert body["offset"] == offset

    def test_spec_addon_032_default_limit_is_50_and_max_is_200(self, api, seed, visual):
        visual(enabled=True)
        seed.file("doc")
        for i in range(201):
            seed.text_chunk("doc", i, f"needle {i}")

        default = _ok(api(PATH, {"q": "needle"}))
        maximum = _ok(api(PATH, {"q": "needle", "limit": 200}))

        assert len(default["segments"]) == 50
        assert default["offset"] == 0
        assert len(maximum["segments"]) == 200
        assert [s["chunk_index"] for s in maximum["segments"]] == list(range(200))
        assert default["total"] == maximum["total"] == 201

    def test_spec_addon_032_no_match_is_200_with_zero_total(self, api, corpus):
        body = _ok(api(PATH, {"q": "haystack-only"}))

        assert body == {
            "query": "haystack-only", "sources": ["text", "transcript", "scene"],
            "total": 0, "offset": 0, "segments": [],
        }


class TestSegmentShape:
    def test_spec_addon_032_every_segment_has_every_key_with_null_for_another_sources(
        self, api, corpus,
    ):
        body = _ok(api(PATH, {"q": "needle"}))

        assert set(body) == {"query", "sources", "total", "offset", "segments"}
        for segment in body["segments"]:
            assert set(segment) == SEGMENT_KEYS
        text, _, transcript, label, visible = body["segments"]
        assert text == {
            "file_id": "doc", "filename": "notes.md", "source": "text",
            "chunk_index": 0, "page": None, "section": None, "section_title": None,
            "start_time": None, "end_time": None, "field": None,
            "match_count": 1, "snippet": "a needle here",
        }
        assert transcript == {
            "file_id": "vid", "filename": "talk.mp4", "source": "transcript",
            "chunk_index": 0, "page": None, "section": None, "section_title": None,
            "start_time": 1.0, "end_time": 2.0, "field": None,
            "match_count": 1, "snippet": "the needle was spoken",
        }
        assert label == {
            "file_id": "vid", "filename": "talk.mp4", "source": "scene",
            "chunk_index": None, "page": None, "section": None, "section_title": None,
            "start_time": 5.0, "end_time": 9.0, "field": "scene_label",
            "match_count": 1, "snippet": "needle on a desk",
        }
        assert visible["field"] == "visible_text"
        assert visible["snippet"] == "Needle Street"
        assert (visible["start_time"], visible["end_time"]) == (9.0, 12.0)

    def test_spec_addon_032_a_scene_without_an_end_has_null_end_time(self, api, seed, visual):
        visual(enabled=True)
        seed.file("vid", "v.mp4", mime="video/mp4")
        seed.scene(seed.run("vid"), 12.0, None, label="needle", visible=None)

        (segment,) = _ok(api(PATH, {"q": "needle"}))["segments"]

        assert (segment["start_time"], segment["end_time"]) == (12.0, None)

    @pytest.mark.parametrize(
        "mime,stored_page,location",
        [
            pytest.param(EPUB, "3", (None, 3, "Chapter Three"), id="epub-with-title"),
            pytest.param(EPUB, "4", (None, 4, None), id="epub-without-title"),
            pytest.param(EPUB, "", (None, None, None), id="epub-empty-page"),
            pytest.param("application/pdf", "3", (3, None, None), id="pdf-page-never-a-section"),
            pytest.param("application/pdf", "", (None, None, None), id="pdf-empty-page"),
            pytest.param("text/markdown", "", (None, None, None), id="markdown"),
        ],
    )
    def test_spec_addon_032_text_location_splits_page_and_section_as_search_does(
        self, api, seed, visual, mime, stored_page, location,
    ):
        visual(enabled=True)
        seed.file("doc", "doc.bin", mime=mime)
        seed.section("doc", 3, "Chapter Three")
        seed.text_chunk("doc", 7, "a needle", page=stored_page)

        (segment,) = _ok(api(PATH, {"q": "needle"}))["segments"]

        assert (segment["page"], segment["section"], segment["section_title"]) == location
        assert segment["chunk_index"] == 7

    def test_spec_addon_032_transcript_times_come_from_the_chunk_with_the_same_index(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        seed.file("vid", "v.mp4", mime="video/mp4")
        seed.transcript("vid", 0, "needle zero", times=(1.5, 4.0))
        seed.transcript("vid", 1, "needle one")
        seed.transcript("vid", 2, "unrelated", times=(9.0, 12.0))

        body = _ok(api(PATH, {"q": "needle", "sources": "transcript"}))

        assert [
            (s["chunk_index"], s["start_time"], s["end_time"]) for s in body["segments"]
        ] == [(0, 1.5, 4.0), (1, None, None)]


def _filler(n: int, ch: str = "x") -> str:
    return ch * n


class TestSnippetAndCount:
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
    def test_spec_addon_032_snippet_keeps_80_characters_each_side_and_marks_cut_ends(
        self, api, seed, visual, before, after, lead, trail,
    ):
        visual(enabled=True)
        seed.file("doc")
        stored = _filler(before, "a") + "needle" + _filler(after, "b")
        seed.text_chunk("doc", 0, stored)

        (segment,) = _ok(api(PATH, {"q": "needle"}))["segments"]

        expected = (
            ("…" if lead else "")
            + _filler(min(before, 80), "a") + "needle" + _filler(min(after, 80), "b")
            + ("…" if trail else "")
        )
        assert segment["snippet"] == expected

    def test_spec_addon_032_snippet_is_around_the_first_occurrence_under_the_comparison(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        seed.file("doc")
        stored = _filler(100, "a") + "NEEDLE" + _filler(100, "b") + "needle" + "ccc"
        seed.text_chunk("doc", 0, stored)

        (segment,) = _ok(api(PATH, {"q": "needle"}))["segments"]

        assert segment["snippet"] == "…" + stored[20:186] + "…"
        assert segment["match_count"] == 2

    def test_spec_addon_032_non_ascii_case_variants_do_not_move_the_snippet_or_add_to_the_count(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        seed.file("doc")
        stored = _filler(90, "y") + "Äb" + _filler(5, "y") + "äb" + _filler(5, "y")
        seed.text_chunk("doc", 0, stored)

        (segment,) = _ok(api(PATH, {"q": "äb"}))["segments"]

        first = stored.index("äb")
        assert segment["snippet"] == "…" + stored[first - 80:]
        assert segment["match_count"] == 1

    def test_spec_addon_032_scene_and_transcript_segments_get_the_same_snippet_and_count(
        self, api, seed, visual,
    ):
        visual(enabled=True)
        seed.file("vid", "v.mp4", mime="video/mp4")
        long = _filler(100, "a") + "needle" + _filler(100, "b") + "Needle"
        seed.transcript("vid", 0, long, times=(0.0, 1.0))
        seed.scene(seed.run("vid"), 1.0, 2.0, label=None, visible=long)

        body = _ok(api(PATH, {"q": "needle"}))

        expected = "…" + long[20:186] + "…"
        assert [(s["source"], s["snippet"], s["match_count"]) for s in body["segments"]] == [
            ("transcript", expected, 2), ("scene", expected, 2),
        ]


class TestRejections:
    @pytest.mark.parametrize(
        "params",
        [
            pytest.param({}, id="no-q"),
            pytest.param({"q": ""}, id="empty-q"),
            pytest.param({"q": "   "}, id="blank-q"),
            pytest.param({"q": "x" * 201}, id="q-201"),
            pytest.param({"q": "needle", "sources": ""}, id="empty-sources"),
            pytest.param({"q": "needle", "sources": "video"}, id="unknown-source"),
            pytest.param({"q": "needle", "sources": "text,video"}, id="one-unknown-source"),
            pytest.param({"q": "needle", "sources": "Text"}, id="source-case"),
            pytest.param({"q": "needle", "limit": 0}, id="limit-0"),
            pytest.param({"q": "needle", "limit": 201}, id="limit-201"),
            pytest.param({"q": "needle", "limit": "ten"}, id="limit-not-int"),
            pytest.param({"q": "needle", "limit": 1.5}, id="limit-fraction"),
            pytest.param({"q": "needle", "offset": -1}, id="offset-negative"),
            pytest.param({"q": "needle", "offset": "x"}, id="offset-not-int"),
        ],
    )
    def test_spec_addon_032_invalid_parameters_are_422_and_not_logged(
        self, api, corpus, caplog, params,
    ):
        caplog.set_level(logging.DEBUG)
        response = api(PATH, params)

        assert response.status_code == 422, response.text
        assert not [
            r for r in caplog.records
            if r.name.startswith("app") and r.levelno >= logging.WARNING
        ]

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param({"q": "n"}, id="q-1"),
            pytest.param({"q": "n" * 200}, id="q-200"),
            pytest.param({"q": "  " + "n" * 200 + "  "}, id="q-200-after-strip"),
            pytest.param({"q": "needle", "limit": 1}, id="limit-1"),
            pytest.param({"q": "needle", "limit": 200}, id="limit-200"),
            pytest.param({"q": "needle", "offset": 0}, id="offset-0"),
        ],
    )
    def test_spec_addon_032_boundary_parameters_are_accepted(self, api, corpus, params):
        assert api(PATH, params).status_code == 200

    def test_spec_addon_032_missing_drive_header_is_400(self, api, corpus):
        response = api(PATH, {"q": "needle"}, drive=None)

        assert response.status_code == 400


class TestFailures:
    def test_spec_addon_032_a_search_db_read_failure_is_500_search_failed_and_logged(
        self, api, corpus, caplog,
    ):
        corpus.index.sql("DROP TABLE fts_text_content")
        caplog.set_level(logging.DEBUG)

        response = api(PATH, {"q": "needle"})

        assert response.status_code == 500
        assert response.json()["detail"] == "Search failed"
        assert [r for r in caplog.records if r.levelno >= logging.ERROR]


class TestSideEffects:
    def test_spec_addon_032_writes_nothing_and_reads_nothing_on_the_event_loop(
        self, api, corpus, statements,
    ):
        statements.reset()
        body = _ok(api(PATH, {"q": "needle"}))

        assert body["total"] == 5
        assert statements.statements, "the endpoint must read the search DB"
        assert statements.writes() == []
        assert statements.on_loop == []


class TestRouteDeclaration:
    @pytest.mark.parametrize("path", ["/search/exact", "/search/scenes"])
    def test_spec_addon_032_and_033_manifest_gates_on_search_policy_without_a_response_filter(
        self, path,
    ):
        manifest = json.loads(MANIFEST.read_text())
        entries = [r for r in manifest["proxy"]["routes"] if r.get("path") == path]

        assert len(entries) == 1
        entry = entries[0]
        assert entry["methods"] == ["GET"]
        assert entry["pre_check"] == {"type": "addon_feature", "feature": "search"}
        assert entry["timeout"] == 30
        assert "drive_optional" not in entry
        assert "response_filter" in entry
        assert entry["response_filter"] is None
