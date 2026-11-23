"""Acceptance tests for SPEC-ADDON-172: find leaves out an occurrence in a
repeated frame-text line, and keeps every other row as today.
"""

from __future__ import annotations

import json

import pytest

from tests.frame_repeat_doubles import (
    bulk_frame_segments,
    search_frame_chunk,
    table_counts,
)
from tests.search_endpoint_db import (  # noqa: F401 - fixtures
    DRIVE,
    api,
    search_index,
    seed,
    statements,
    visual,
)

VIDEO = "vid000000172"
Q0 = [0.1, 0.1, 0.4, 0.1, 0.4, 0.15, 0.1, 0.15]
Q1 = [0.1, 0.4, 0.7, 0.4, 0.7, 0.46, 0.1, 0.46]


def _video(seed, file_id=VIDEO):
    seed.file(file_id, f"{file_id}.mp4", mime="video/mp4")
    seed.index.sql(
        "UPDATE indexed_files SET whisper_indexed = 0, frame_ocr_indexed = 1 "
        "WHERE file_id = :f",
        f=file_id,
    )
    return file_id


def _segment(
    seed, file_id, first_index, start, end, texts, *,
    lines=None, size=None, segment_index=0,
):
    for offset, body in enumerate(texts):
        search_frame_chunk(seed, file_id, first_index + offset, start, end, body)
    if lines is not None:
        stored = json.dumps(lines)
        if size is None:
            seed.index.sql(
                "INSERT INTO video_frame_lines"
                "(file_id, segment_index, start_time, end_time, lines_json) "
                "VALUES(:f, :i, :s, :e, :j)",
                f=file_id, i=segment_index, s=start, e=end, j=stored,
            )
        else:
            seed.index.sql(
                "INSERT INTO video_frame_lines"
                "(file_id, segment_index, start_time, end_time, lines_json, "
                "frame_width, frame_height) "
                "VALUES(:f, :i, :s, :e, :j, :w, :h)",
                f=file_id, i=segment_index, s=start, e=end, j=stored,
                w=size[0], h=size[1],
            )


def _find(api, q, file_id=VIDEO, **params):  # noqa: F811
    response = api(f"/files/{file_id}/find", {"q": q, **params}, drive=DRIVE)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize(
    ("gap", "starts"),
    [
        pytest.param(60.0, [0.0], id="exactly-sixty"),
        pytest.param(60.001, [0.0, 70.001], id="sixty-and-a-millisecond"),
    ],
)
def test_spec_addon_172_i5_a_repeated_occurrence_has_no_row_and_a_late_one_keeps_it(
    gap, starts, api, seed,
):
    """I5/I1: the 60 s window decides whether the second occurrence is a row."""
    _video(seed)
    _segment(seed, VIDEO, 0, 0.0, 10.0, ["HEADER\nfirst"])
    _segment(seed, VIDEO, 1, 10.0 + gap, 20.0 + gap, ["HEADER\nsecond"])

    body = _find(api, "HEADER")

    assert body["total"] == len(starts)
    assert [row["start"] for row in body["rows"]] == starts


def test_spec_addon_172_i5_a_chain_stays_repeated_past_sixty_seconds(api, seed):
    """I5/I1: U is the latest earlier end_time of the line."""
    _video(seed)
    _segment(seed, VIDEO, 0, 0.0, 10.0, ["STAY"])
    _segment(seed, VIDEO, 1, 70.0, 80.0, ["STAY"])
    _segment(seed, VIDEO, 2, 140.0, 150.0, ["STAY"])

    body = _find(api, "STAY")

    assert body["total"] == 1
    assert [row["start"] for row in body["rows"]] == [0.0]


def test_spec_addon_172_i5_a_line_absent_for_more_than_sixty_seconds_returns_a_row(
    api, seed,
):
    """I5/I1: a line hidden for 70 s is new when it comes back."""
    _video(seed)
    _segment(seed, VIDEO, 0, 0.0, 10.0, ["SLIDE"])
    _segment(seed, VIDEO, 1, 40.0, 50.0, ["OTHER"])
    _segment(seed, VIDEO, 2, 80.0, 90.0, ["SLIDE"])

    body = _find(api, "SLIDE")

    assert body["total"] == 2
    assert [row["start"] for row in body["rows"]] == [0.0, 80.0]


def test_spec_addon_172_i5_identical_lines_inside_one_segment_each_have_a_row(
    api, seed,
):
    """I5/I1: equal lines in one segment are not repeated against each other."""
    _video(seed)
    _segment(seed, VIDEO, 0, 10.0, 20.0, ["SAME", "SAME"])

    body = _find(api, "SAME")

    assert body["total"] == 2
    assert [(row["start"], row["ordinal"]) for row in body["rows"]] == [
        (10.0, 1), (10.0, 2),
    ]


def test_spec_addon_172_i5_a_kept_row_is_todays_row_field_for_field(api, seed):
    """I5: a row beside a repeated line keeps its snippet, times and boxes."""
    lines = [
        {"text": "STAY", "quad": Q0},
        {"text": "warehouse moved", "quad": Q1},
    ]
    _video(seed, "filtered")
    _segment(seed, "filtered", 0, 0.0, 10.0, ["STAY"])
    _segment(
        seed, "filtered", 1, 70.0, 80.0, ["STAY\nwarehouse moved"],
        lines=lines, size=(1280, 720),
    )
    _video(seed, "control")
    _segment(
        seed, "control", 1, 70.0, 80.0, ["STAY\nwarehouse moved"],
        lines=lines, size=(1280, 720),
    )

    filtered = _find(api, "warehouse", "filtered")
    control = _find(api, "warehouse", "control")

    assert filtered["total"] == 1
    assert filtered["rows"] == control["rows"]


def test_spec_addon_172_i5_total_truncated_and_order_are_over_the_kept_rows(api, seed):
    """I5: a dropped occurrence does not count and does not shift the order."""
    _video(seed)
    _segment(seed, VIDEO, 0, 0.0, 10.0, ["STAY"])
    _segment(seed, VIDEO, 1, 70.0, 80.0, ["STAY"])
    _segment(seed, VIDEO, 2, 200.0, 210.0, ["STAY"])

    body = _find(api, "STAY")

    assert body["total"] == 2
    assert body["truncated"] is False
    assert [row["start"] for row in body["rows"]] == [0.0, 200.0]


def test_spec_addon_172_i5_the_row_cap_is_over_the_kept_rows(api, seed):
    """I5: 2,500 segments of one line are one kept row, so nothing is truncated."""
    _video(seed)
    bulk_frame_segments(seed.index, VIDEO, 2500, "STAY")

    body = _find(api, "STAY")

    assert body["total"] == 1
    assert body["truncated"] is False
    assert [row["start"] for row in body["rows"]] == [0.0]


def test_spec_addon_172_i7_transcript_document_picture_and_scene_rows_are_unchanged(
    api, seed, visual,
):
    """I7: only frame-text occurrences of a video are filtered."""
    visual(enabled=True)
    _video(seed, "withspeech")
    seed.transcript("withspeech", 0, "needle spoken", times=(10.0, 11.0))
    _segment(seed, "withspeech", 0, 0.0, 10.0, ["needle screen"])
    _segment(seed, "withspeech", 1, 70.0, 80.0, ["needle again"])
    seed.file("doc", "notes.md", mime="text/markdown")
    seed.text_chunk("doc", 0, "needle document")
    seed.file("pic", "picture.png", mime="image/png")
    seed.text_chunk("pic", 0, "needle picture")
    run = seed.run("withspeech")
    seed.scene(run, 5.0, 9.0, label=None, visible="needle scene")

    body = _find(api, "needle", "withspeech")

    rows = {
        (row["kind"], row["snippet"], row["start"], row["end"])
        for row in body["rows"]
    }
    assert ("utterance", "needle spoken", 10.0, 11.0) in rows
    assert ("scene", "needle scene", 5.0, 9.0) in rows

    documents = _find(api, "needle", "doc")
    pictures = _find(api, "needle", "pic")
    assert [
        (row["kind"], row["snippet"], row["start"], row["end"])
        for row in documents["rows"]
    ] == [("text", "needle document", None, None)]
    assert [
        (row["kind"], row["snippet"], row["start"], row["end"])
        for row in pictures["rows"]
    ] == [("picture_text", "needle picture", None, None)]


def test_spec_addon_172_i8_find_writes_nothing(api, seed, statements):
    """I8: the find request changes no row of the search DB."""
    _video(seed)
    _segment(seed, VIDEO, 0, 0.0, 10.0, ["STAY"])
    _segment(seed, VIDEO, 1, 70.0, 80.0, ["STAY"])
    before = table_counts(seed.index)
    statements.reset()

    _find(api, "STAY")

    assert statements.writes() == []
    assert table_counts(seed.index) == before
