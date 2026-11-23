"""Acceptance tests for SPEC-ADDON-171: file Ask drops a repeated frame-text line.

A line is repeated for a segment when an earlier segment has the same line and
the latest end time among those earlier segments is at most 60 seconds before
the segment's start.
"""

from __future__ import annotations

import pytest

from app.rag.file_question import load_material
from tests.file_question_doubles import DRIVE, numbered_in_order, toward
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    addon,
    by_kind,
    call,
    db,
    llm,
    statements,
    vision_client,
)
from tests.frame_label_layout import assert_labelled
from tests.frame_repeat_doubles import ask_frame_chunk, table_counts
from tests.search_endpoint_db import patch_everywhere
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import FFmpeg, scenes

VIDEO = "vid000000171"
EVEN_80 = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
LOFT_MIME = "application/vnd.litloft.loft+json"


def _texts(material) -> list[str]:
    assert material is not None
    return [chunk.text for chunk in material.chunks]


def _video(db, file_id=VIDEO):
    db.file(file_id, mime="video/mp4")
    return file_id


def _answering(text: str):
    def script(_body):
        return 200, text

    return script


@pytest.mark.parametrize(
    ("gap", "second_row"),
    [
        pytest.param(60.0, "second caption", id="exactly-sixty"),
        pytest.param(60.001, "CHANNEL\nsecond caption", id="sixty-and-a-millisecond"),
    ],
)
def test_spec_addon_171_i1_a_line_is_repeated_at_sixty_seconds_and_not_after(
    gap, second_row, db, addon,
):
    """I1: S.start_time - U is compared with the 60 s window inclusively."""
    _video(db)
    ask_frame_chunk(db, VIDEO, 0, 0.0, 10.0, "CHANNEL\nfirst caption")
    ask_frame_chunk(db, VIDEO, 1, 10.0 + gap, 20.0 + gap, "CHANNEL\nsecond caption")

    material = load_material(VIDEO, DRIVE, "question", budget=1000)

    assert _texts(material) == ["CHANNEL\nfirst caption", second_row]


def test_spec_addon_171_i1_a_chain_keeps_the_line_repeated_past_sixty_seconds(
    db, addon,
):
    """I1: U is the latest earlier end, so a chain of segments stays repeated."""
    _video(db)
    ask_frame_chunk(db, VIDEO, 0, 0.0, 10.0, "STAY\nfirst")
    ask_frame_chunk(db, VIDEO, 1, 70.0, 80.0, "STAY\nsecond")
    ask_frame_chunk(db, VIDEO, 2, 140.0, 150.0, "STAY\nthird")

    material = load_material(VIDEO, DRIVE, "question", budget=1000)

    assert _texts(material) == ["STAY\nfirst", "second", "third"]


def test_spec_addon_171_i1_a_line_absent_for_more_than_sixty_seconds_is_new(
    db, addon,
):
    """I1: a line hidden for 70 s is not repeated when it comes back."""
    _video(db)
    ask_frame_chunk(db, VIDEO, 0, 0.0, 10.0, "SLIDE\nfirst")
    ask_frame_chunk(db, VIDEO, 1, 40.0, 50.0, "OTHER")
    ask_frame_chunk(db, VIDEO, 2, 80.0, 90.0, "SLIDE\nsecond")

    material = load_material(VIDEO, DRIVE, "question", budget=1000)

    assert _texts(material) == ["SLIDE\nfirst", "OTHER", "SLIDE\nsecond"]


def test_spec_addon_171_i1_identical_lines_inside_one_segment_are_never_repeated(
    db, addon,
):
    """I1: equality is between segments, so two rows of one segment both stay."""
    _video(db)
    ask_frame_chunk(db, VIDEO, 0, 10.0, 20.0, "SAME")
    ask_frame_chunk(db, VIDEO, 1, 10.0, 20.0, "SAME")

    material = load_material(VIDEO, DRIVE, "question", budget=1000)

    assert _texts(material) == ["SAME", "SAME"]


def test_spec_addon_171_i1_a_cut_fragment_is_not_removed_against_an_earlier_line(
    db, addon,
):
    """I1: only a line of the merged text repeats, never a cut chunk part."""
    _video(db)
    ask_frame_chunk(db, VIDEO, 0, 0.0, 10.0, "CHAN")
    first = "x" * 20 + "\nCHAN"
    second = first + "NEL\nsub"
    ask_frame_chunk(db, VIDEO, 1, 70.0, 80.0, first)
    ask_frame_chunk(db, VIDEO, 2, 70.0, 80.0, second)

    material = load_material(VIDEO, DRIVE, "question", budget=1000)

    assert _texts(material) == ["CHAN", first, second]


def test_spec_addon_171_i1_a_line_is_read_from_the_merged_text_of_a_segment(
    db, addon,
):
    """I1: a chunk-boundary fragment is not a line, the merged whole line is."""
    _video(db)
    ask_frame_chunk(db, VIDEO, 0, 0.0, 10.0, "CHANNEL\nCHAN")
    first = "x" * 20 + "\nCHAN"
    second = first + "NEL\nsub"
    ask_frame_chunk(db, VIDEO, 1, 70.0, 80.0, first)
    ask_frame_chunk(db, VIDEO, 2, 70.0, 80.0, second)

    material = load_material(VIDEO, DRIVE, "question", budget=1000)

    assert _texts(material) == [
        "CHANNEL\nCHAN",
        first,
        "x" * 20 + "\nsub",
    ]


def test_spec_addon_171_i2_a_row_with_nothing_removed_keeps_its_stored_text(
    db, addon,
):
    """I2: whitespace and a blank part survive byte for byte."""
    _video(db)
    stored = "  Alpha  \n\nBeta\t"
    ask_frame_chunk(db, VIDEO, 0, 0.0, 10.0, stored)

    material = load_material(VIDEO, DRIVE, "question", budget=1000)

    assert _texts(material) == [stored]


def test_spec_addon_171_i2_a_removed_part_leaves_the_other_parts_as_stored(
    db, addon,
):
    """I2: the remaining parts, blank ones included, are joined in their order."""
    _video(db)
    ask_frame_chunk(db, VIDEO, 0, 0.0, 10.0, "DROP")
    ask_frame_chunk(db, VIDEO, 1, 70.0, 80.0, "  Keep A  \nDROP\n\nKeep B\t")

    material = load_material(VIDEO, DRIVE, "question", budget=1000)

    assert _texts(material) == ["DROP", "  Keep A  \n\nKeep B\t"]


def test_spec_addon_171_i2_a_line_is_judged_against_segments_that_are_not_sent(
    db, addon, monkeypatch,
):
    """I2/I4: repeats are computed before ranking and fitting, not from rows sent."""
    _video(db)
    for index, (start, filler, cos) in enumerate((
        (0.0, "a", 0.1), (20.0, "b", 0.2), (40.0, "c", 0.3),
    )):
        ask_frame_chunk(
            db, VIDEO, index, start, start + 10.0,
            "REPEAT-LINE\n" + filler * 100, cos=cos,
        )
    ask_frame_chunk(
        db, VIDEO, 3, 60.0, 70.0, "REPEAT-LINE\nUNIQUE-KEPT", cos=0.9,
    )
    patch_everywhere(monkeypatch, "embed_query", lambda _text: toward(db.db.dim, 1.0))

    material = load_material(VIDEO, DRIVE, "question", budget=11)

    assert _texts(material) == ["UNIQUE-KEPT"]
    assert material.partial is True
    assert sum(len(text) for text in _texts(material)) <= 11


def test_spec_addon_171_i4_filtered_rows_that_fit_are_sent_whole_and_not_partial(
    db, addon,
):
    """I4: the budget applies to the filtered rows, not to the stored rows."""
    _video(db)
    banner = "B" * 90
    ask_frame_chunk(db, VIDEO, 0, 0.0, 10.0, banner + "\nu0")
    ask_frame_chunk(db, VIDEO, 1, 70.0, 80.0, banner + "\nu1")
    ask_frame_chunk(db, VIDEO, 2, 140.0, 150.0, banner + "\nu2")

    material = load_material(VIDEO, DRIVE, "question", budget=100)

    assert _texts(material) == [banner + "\nu0", "u1", "u2"]
    assert material.partial is False


def test_spec_addon_171_i4_filtered_rows_that_still_do_not_fit_stay_within_budget(
    db, addon, monkeypatch,
):
    """I4: partial is true and the sent text totals at most the budget."""
    _video(db)
    for index, start in enumerate((0.0, 20.0, 40.0)):
        ask_frame_chunk(db, VIDEO, index, start, start + 10.0, chr(97 + index) * 40)
    patch_everywhere(monkeypatch, "embed_query", lambda _text: toward(db.db.dim, 1.0))

    material = load_material(VIDEO, DRIVE, "question", budget=100)

    assert material.partial is True
    assert _texts(material) == ["a" * 40, "b" * 40]
    assert sum(len(text) for text in _texts(material)) <= 100


def test_spec_addon_171_i7_a_transcript_chunk_is_sent_unchanged(db, addon):
    """I7: transcript text and its position are not filtered."""
    _video(db)
    speech = "SPOKEN-PASSAGE"
    db.on(VIDEO).transcript(speech, start=0.0)
    ask_frame_chunk(db, VIDEO, 0, 10.0, 20.0, "ON SCREEN")

    material = load_material(VIDEO, DRIVE, "question", budget=1000)

    assert material is not None
    assert (material.chunks[0].text, material.chunks[0].position) == (
        speech, {"seconds": 0.0},
    )


@pytest.mark.parametrize(
    ("mime", "stored"),
    [
        pytest.param("text/markdown", "document text", id="document"),
        pytest.param("image/png", "picture text", id="picture"),
    ],
)
def test_spec_addon_171_i7_documents_and_pictures_are_sent_unchanged(
    mime, stored, db, addon,
):
    """I7: a non-video file has no frame-text segment to filter."""
    file_id = "doc000000171" if mime == "text/markdown" else "pic000000171"
    db.file(file_id, mime=mime)
    db.document(stored + "\nsecond", chunk_index=0, cos=None)

    material = load_material(file_id, DRIVE, "question", budget=1000)

    assert _texts(material) == [stored + "\nsecond"]


def test_spec_addon_171_a_video_row_without_a_frame_text_row_is_sent_unfiltered(
    db, addon,
):
    """A failure case: a row outside every segment adds no line and is not filtered."""
    _video(db)
    ask_frame_chunk(db, VIDEO, 0, 0.0, 10.0, "HEADER")
    db.on(VIDEO).document("HEADER\nordinary", chunk_index=9, cos=None)

    material = load_material(VIDEO, DRIVE, "question", budget=1000)

    assert _texts(material) == ["HEADER", "HEADER\nordinary"]


def test_spec_addon_171_a_youtube_loft_is_filtered_like_a_native_video(db, addon):
    """A ``.loft`` whose file_type is video follows the same rule."""
    file_id = "lft000000171"
    db.file(file_id, mime=LOFT_MIME)
    db.db.index.sql(
        "UPDATE indexed_files SET file_type = 'video' WHERE file_id = :f",
        f=file_id,
    )
    ask_frame_chunk(db, file_id, 0, 0.0, 10.0, "CHANNEL\nfirst")
    ask_frame_chunk(db, file_id, 1, 70.0, 80.0, "CHANNEL\nsecond")

    material = load_material(file_id, DRIVE, "question", budget=1000)

    assert _texts(material) == ["CHANNEL\nfirst", "second"]


def test_spec_addon_171_a_video_whose_only_text_is_frame_text_keeps_material(
    db, addon,
):
    """A failure case: the first segment of a line is always kept."""
    _video(db)
    ask_frame_chunk(db, VIDEO, 0, 0.0, 10.0, "ONLY")
    ask_frame_chunk(db, VIDEO, 1, 70.0, 80.0, "ONLY")
    ask_frame_chunk(db, VIDEO, 2, 140.0, 150.0, "ONLY")

    material = load_material(VIDEO, DRIVE, "question", budget=1000)

    assert material is not None
    assert _texts(material) == ["ONLY"]


def test_spec_addon_171_a_transcript_survives_when_the_frame_rows_repeat(
    db, addon,
):
    """A failure case: the transcript passages remain beside the filtered rows."""
    _video(db)
    db.on(VIDEO).transcript("SPOKEN", start=0.0)
    ask_frame_chunk(db, VIDEO, 0, 10.0, 20.0, "ONLY")
    ask_frame_chunk(db, VIDEO, 1, 80.0, 90.0, "ONLY")

    material = load_material(VIDEO, DRIVE, "question", budget=1000)

    assert _texts(material) == ["SPOKEN", "ONLY"]


async def test_spec_addon_171_i3_a_blank_row_takes_no_number_and_frames_follow_the_count(
    db, addon, llm, call, tmp_path, monkeypatch,
):
    """I3: P is the sent passage count; frame numbers and citations follow it."""
    from tests.video_frames_doubles import seed_text_less_video

    seed_text_less_video(
        db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=80.0,
    )
    ask_frame_chunk(db, VIDEO, 0, 0.0, 10.0, "first visible")
    ask_frame_chunk(db, VIDEO, 1, 10.0, 20.0, "STAY")
    ask_frame_chunk(db, VIDEO, 2, 70.0, 80.0, "STAY\n\n")
    scenes(monkeypatch, [], duration=80.0)
    FFmpeg().install(monkeypatch)
    _, wire = vision_client(llm, script=_answering("See [1], [2] and [3]."))

    response = await call(
        "POST", f"/files/{VIDEO}/ask", json={"question": "what is this"},
    )

    assert response.status_code == 200, response.text
    layout = assert_labelled(wire, wire.image_bodies[0], first=3, times=EVEN_80)
    assert numbered_in_order(layout.prompt, ["first visible", "STAY"])
    events = by_kind(response)
    assert events["meta"]["partial"] is False
    assert events["meta"]["image"] == "attached"
    assert {
        entry["n"]: entry["position"] for entry in events["citations"]
    } == {
        1: {"seconds": 0.0},
        2: {"seconds": 10.0},
        3: {"seconds": 5.0},
    }


def test_spec_addon_171_i8_ask_writes_nothing(db, addon, statements):
    """I8: ``load_material`` reads the search DB and changes no row."""
    _video(db)
    ask_frame_chunk(db, VIDEO, 0, 0.0, 10.0, "CHANNEL\nfirst")
    ask_frame_chunk(db, VIDEO, 1, 70.0, 80.0, "CHANNEL\nsecond")
    before = table_counts(db.db.index)
    statements.reset()

    load_material(VIDEO, DRIVE, "question", budget=1000)

    assert statements.writes() == []
    assert table_counts(db.db.index) == before
