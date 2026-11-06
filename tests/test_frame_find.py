"""Find in a video lists its frame-text segments as ``scene`` rows.

SPEC-ADDON-066 A7/I9: each segment is one unit, its chunks are merged with the
200-character replay limit, every occurrence is a row of kind ``scene`` with
the segment's start and end, the answer is ``indexed`` when
``frame_ocr_indexed`` is true, and no Visual Index run is needed.
"""

from __future__ import annotations

import pytest

from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    api,
    search_index,
    seed,
)

VIDEO = "video/mp4"


def _ok(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


def _flag(seed, file_id: str, **flags) -> None:
    for column, value in flags.items():
        seed.index.sql(
            f"UPDATE indexed_files SET {column} = :v WHERE file_id = :f",
            v=1 if value else 0, f=file_id,
        )


def _video(seed, file_id: str = "vid", **flags) -> None:
    seed.file(file_id, f"{file_id}.mp4", mime=VIDEO)
    defaults = dict(whisper_indexed=False, frame_ocr_indexed=True)
    defaults.update(flags)
    _flag(seed, file_id, **defaults)


def _frame_chunk(seed, file_id: str, chunk_index: int, body: str, start: float, end: float) -> None:
    seed.text_chunk(file_id, chunk_index, body)
    seed.index.sql(
        "INSERT INTO video_frame_texts"
        "(file_id, chunk_index, start_time, end_time, text) "
        "VALUES(:f, :c, :s, :e, :t)",
        f=file_id, c=chunk_index, s=start, e=end, t=body,
    )


def _find(api, q: str, file_id: str = "vid") -> dict:
    return _ok(api(f"/files/{file_id}/find", {"q": q}, drive=DRIVE))


def test_spec_addon_066_i9_frame_text_is_a_scene_row_with_its_segment_times(api, seed):
    """SPEC-ADDON-066 A7/I9: the row is a scene row carrying start and end."""
    _video(seed)
    _frame_chunk(seed, "vid", 0, "the warehouse moved", 10.0, 20.0)
    _frame_chunk(seed, "vid", 1, "warehouse again", 30.0, 40.0)

    body = _find(api, "warehouse")

    assert body["indexed"] is True
    assert body["total"] == 2
    assert [row["kind"] for row in body["rows"]] == ["scene", "scene"]
    assert [(row["start"], row["end"]) for row in body["rows"]] == [
        (10.0, 20.0), (30.0, 40.0),
    ]
    for row in body["rows"]:
        start = row["match_start"]
        assert row["snippet"][start:start + row["match_length"]].lower() == "warehouse"


def test_spec_addon_066_i9_frame_text_needs_no_visual_index(api, seed):
    """SPEC-ADDON-066 A7: the drive's Visual Index is not a condition."""
    _video(seed)
    _frame_chunk(seed, "vid", 0, "the warehouse moved", 10.0, 20.0)

    body = _ok(api(
        "/files/vid/find", {"q": "warehouse", "include_scenes": "false"},
        drive=DRIVE,
    ))

    assert [row["kind"] for row in body["rows"]] == ["scene"]


def test_spec_addon_066_i9_indexed_is_true_on_frame_ocr_indexed_alone(api, seed):
    """SPEC-ADDON-066 A7/I9: no transcript is needed to say indexed."""
    _video(seed, whisper_indexed=False, frame_ocr_indexed=True)
    _frame_chunk(seed, "vid", 0, "the warehouse moved", 10.0, 20.0)

    body = _find(api, "warehouse")

    assert body["indexed"] is True


def test_spec_addon_066_i9_a_chunk_overlap_inside_a_segment_counts_once(api, seed):
    """SPEC-ADDON-066 A7 / SPEC-ADDON-061 I1: one unit, one occurrence."""
    _video(seed)
    first = "a" * 780 + "warehouse" + "b" * 150
    second = first[-200:] + "c" * 800
    _frame_chunk(seed, "vid", 0, first, 10.0, 20.0)
    _frame_chunk(seed, "vid", 1, second, 10.0, 20.0)

    body = _find(api, "warehouse")

    assert body["total"] == 1


def test_spec_addon_066_i9_at_an_equal_start_utterance_comes_before_the_scene(api, seed):
    """SPEC-ADDON-066 A7/I9: start, then utterance, then scene."""
    _video(seed)
    seed.transcript("vid", 0, "warehouse speech", times=(10.0, 20.0))
    _flag(seed, "vid", whisper_indexed=True)
    _frame_chunk(seed, "vid", 0, "warehouse on screen", 10.0, 20.0)

    body = _find(api, "warehouse")

    assert [row["kind"] for row in body["rows"]] == ["utterance", "scene"]
