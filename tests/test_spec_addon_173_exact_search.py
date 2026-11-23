"""Acceptance tests for SPEC-ADDON-173: exact search leaves out a frame-text hit
whose matching lines are all repeated, and keeps every other hit as today.
"""

from __future__ import annotations

import pytest

from tests.frame_repeat_doubles import search_frame_chunk, table_counts
from tests.search_endpoint_db import (  # noqa: F401 - fixtures
    api,
    search_index,
    seed,
    statements,
    visual,
)

PATH = "/search/exact"
VIDEO = "vid000000173"
IDENTITY = ("file_id", "filename")


def _video(seed, file_id=VIDEO):
    seed.file(file_id, f"{file_id}.mp4", mime="video/mp4")
    return file_id


def _segment(seed, file_id, first_index, start, end, texts):
    for offset, body in enumerate(texts):
        search_frame_chunk(seed, file_id, first_index + offset, start, end, body)


def _exact(api, q, **params):
    response = api(PATH, {"q": q, "sources": "text", **params})
    assert response.status_code == 200, response.text
    return response.json()


def _located(body):
    return [(segment["start_time"], segment["snippet"]) for segment in body["segments"]]


def _starts(body):
    return [segment["start_time"] for segment in body["segments"]]


@pytest.mark.parametrize(
    ("gap", "starts"),
    [
        pytest.param(60.0, [0.0], id="exactly-sixty"),
        pytest.param(60.001, [0.0, 70.001], id="sixty-and-a-millisecond"),
    ],
)
def test_spec_addon_173_i6_a_repeated_frame_hit_is_left_out_after_sixty_seconds(
    gap, starts, api, seed,
):
    """I6/I1: the 60 s window decides whether the later chunk is a hit."""
    _video(seed)
    _segment(seed, VIDEO, 0, 0.0, 10.0, ["HEADER\nfirst"])
    _segment(seed, VIDEO, 1, 10.0 + gap, 20.0 + gap, ["HEADER\nsecond"])

    body = _exact(api, "HEADER")

    assert body["total"] == len(starts)
    assert [segment["start_time"] for segment in body["segments"]] == starts


def test_spec_addon_173_i6_a_chain_stays_repeated_past_sixty_seconds(api, seed):
    """I6/I1: U is the latest earlier end_time of the line."""
    _video(seed)
    _segment(seed, VIDEO, 0, 0.0, 10.0, ["STAY"])
    _segment(seed, VIDEO, 1, 70.0, 80.0, ["STAY"])
    _segment(seed, VIDEO, 2, 140.0, 150.0, ["STAY"])

    body = _exact(api, "STAY")

    assert body["total"] == 1
    assert _located(body) == [(0.0, "STAY")]


def test_spec_addon_173_i6_a_line_absent_for_more_than_sixty_seconds_is_a_hit(
    api, seed,
):
    """I6/I1: a line hidden for 70 s is new when it comes back."""
    _video(seed)
    _segment(seed, VIDEO, 0, 0.0, 10.0, ["SLIDE"])
    _segment(seed, VIDEO, 1, 40.0, 50.0, ["OTHER"])
    _segment(seed, VIDEO, 2, 80.0, 90.0, ["SLIDE"])

    body = _exact(api, "SLIDE")

    assert body["total"] == 2
    assert [segment["start_time"] for segment in body["segments"]] == [0.0, 80.0]


def test_spec_addon_173_i6_identical_lines_inside_one_segment_are_both_hits(api, seed):
    """I6/I1: equal lines in one segment are not repeated against each other."""
    _video(seed)
    _segment(seed, VIDEO, 0, 10.0, 20.0, ["SAME", "SAME"])

    body = _exact(api, "SAME")

    assert body["total"] == 2
    assert [segment["chunk_index"] for segment in body["segments"]] == [0, 1]


def test_spec_addon_173_i6_one_repeated_and_one_new_matching_line_keeps_todays_hit(
    api, seed,
):
    """I6: a chunk is dropped only when every matching line is repeated."""
    _video(seed, "filtered")
    _segment(seed, "filtered", 0, 0.0, 10.0, ["NEEDLE stay"])
    _segment(
        seed, "filtered", 1, 70.0, 80.0, ["NEEDLE stay\nNEEDLE new"],
    )
    _video(seed, "control")
    _segment(seed, "control", 1, 70.0, 80.0, ["NEEDLE stay\nNEEDLE new"])

    filtered = _exact(api, "needle")
    kept = next(
        segment for segment in filtered["segments"]
        if segment["file_id"] == "filtered" and segment["start_time"] == 70.0
    )
    control = next(
        segment for segment in filtered["segments"] if segment["file_id"] == "control"
    )

    assert filtered["total"] == 3
    assert kept["match_count"] == 2
    assert {k: v for k, v in kept.items() if k not in IDENTITY} == {
        k: v for k, v in control.items() if k not in IDENTITY
    }


def test_spec_addon_173_i6_a_hit_whose_term_crosses_a_newline_is_kept(api, seed):
    """I6: no matching line means the hit is not a repeated line's hit."""
    _video(seed)
    _segment(seed, VIDEO, 0, 0.0, 10.0, ["alpha\nbeta"])
    _segment(seed, VIDEO, 1, 70.0, 80.0, ["alpha\nbeta"])

    body = _exact(api, "alpha\nbeta")

    assert body["total"] == 2
    assert _starts(body) == [0.0, 70.0]


def test_spec_addon_173_i6_total_and_paging_are_over_the_kept_hits(api, seed):
    """I6: a dropped hit is not counted and is not visited by a page."""
    _video(seed)
    _segment(seed, VIDEO, 0, 0.0, 10.0, ["HIT"])
    _segment(seed, VIDEO, 1, 70.0, 80.0, ["HIT"])
    _segment(seed, VIDEO, 2, 200.0, 210.0, ["HIT"])

    first = _exact(api, "HIT", limit=1, offset=0)
    second = _exact(api, "HIT", limit=1, offset=1)
    past = _exact(api, "HIT", limit=1, offset=2)

    assert [first["total"], second["total"], past["total"]] == [2, 2, 2]
    assert _starts(first) == [0.0]
    assert _starts(second) == [200.0]
    assert _starts(past) == []


def test_spec_addon_173_i7_transcript_and_scene_hits_are_unchanged(api, seed, visual):
    """I7: only a video's frame-text hits are filtered."""
    visual(enabled=True)
    _video(seed, "withspeech")
    seed.transcript("withspeech", 0, "needle spoken", times=(1.0, 2.0))
    _segment(seed, "withspeech", 0, 0.0, 10.0, ["needle screen"])
    _segment(seed, "withspeech", 1, 70.0, 80.0, ["needle again"])
    seed.scene(seed.run("withspeech"), 5.0, 9.0, label=None, visible="needle scene")

    body = _exact(api, "needle", sources="text,transcript,scene")

    transcript = [
        segment for segment in body["segments"] if segment["source"] == "transcript"
    ]
    scenes = [
        segment for segment in body["segments"]
        if segment["source"] == "scene" and segment["snippet"] == "needle scene"
    ]
    assert [
        (segment["start_time"], segment["end_time"], segment["snippet"])
        for segment in transcript
    ] == [(1.0, 2.0, "needle spoken")]
    assert [
        (segment["start_time"], segment["end_time"], segment["field"])
        for segment in scenes
    ] == [(5.0, 9.0, "visible_text")]


def test_spec_addon_173_i6_a_video_chunk_without_a_frame_row_is_unchanged(api, seed):
    """A failure case: a row outside every segment is neither joined nor filtered."""
    _video(seed)
    _segment(seed, VIDEO, 0, 0.0, 10.0, ["HEADER"])
    _segment(seed, VIDEO, 1, 70.0, 80.0, ["HEADER"])
    seed.text_chunk(VIDEO, 9, "HEADER ordinary")

    body = _exact(api, "HEADER")

    assert body["total"] == 2
    assert {(segment["start_time"], segment["snippet"]) for segment in body["segments"]} == {
        (0.0, "HEADER"),
        (None, "HEADER ordinary"),
    }


def test_spec_addon_173_i8_exact_search_writes_nothing(api, seed, statements):
    """I8: the exact-search request changes no row of the search DB."""
    _video(seed)
    _segment(seed, VIDEO, 0, 0.0, 10.0, ["STAY"])
    _segment(seed, VIDEO, 1, 70.0, 80.0, ["STAY"])
    before = table_counts(seed.index)
    statements.reset()

    _exact(api, "STAY")

    assert statements.writes() == []
    assert table_counts(seed.index) == before
