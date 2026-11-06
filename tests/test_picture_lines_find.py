"""Find in this file carries the boxes of the lines it hits (SPEC-ADDON-075 I5-I7).

SPEC-ID: SPEC-ADDON-075

The pictures here are seeded rows, not OCR output, so the answer can be read
without the engine. ``find_in_file`` is called directly where the query must
keep a newline that the route strips.
"""

from __future__ import annotations

import json
import logging

import pytest

from app.extractors.base import ContentExtractor
from tests.extraction_fixtures import build_api
from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    api,
    search_index,
    seed,
    statements,
    visual,
)

LINES = [
    {"text": "alpha beta", "quad": [0, 0, 0.5, 0, 0.5, 0.5, 0, 0.5], "score": 0.9},
    {"text": "needle needle here", "quad": [0.5, 0, 1, 0, 1, 0.5, 0.5, 0.5], "score": 0.8},
    {"text": "gamma delta", "quad": [0, 0.5, 0.5, 0.5, 0.5, 1, 0, 1], "score": 0.7},
    {"text": "epsilon", "quad": [0.5, 0.5, 1, 0.5, 1, 1, 0.5, 1], "score": 0.6},
]
QUADS = [line["quad"] for line in LINES]
JOINED = "\n".join(line["text"] for line in LINES)


def _flag(seed, file_id: str, **flags) -> None:
    for column, value in flags.items():
        seed.index.sql(
            f"UPDATE indexed_files SET {column} = :v WHERE file_id = :f",
            v=1 if value else 0,
            f=file_id,
        )


def _seed_lines(seed, file_id: str, lines: list[dict]) -> None:
    seed.index.sql(
        "INSERT INTO picture_lines(file_id, lines_json, updated_at) "
        "VALUES(:f, :j, CURRENT_TIMESTAMP)",
        f=file_id,
        j=json.dumps(lines),
    )


def _seed_picture(
    seed,
    file_id: str,
    *,
    lines: list[dict] | None = None,
    text: str | None = None,
    chunks: list[str] | None = None,
) -> None:
    seed.file(file_id, f"{file_id}.png", mime="image/png")
    _flag(seed, file_id, text_indexed=False, ocr_indexed=True)
    body = text
    if body is None and lines is not None:
        body = "\n".join(line["text"] for line in lines)
    for index, chunk in enumerate(
        chunks if chunks is not None else ContentExtractor.chunk_text(body or "")
    ):
        seed.text_chunk(file_id, index, chunk)
    seed.index.sql(
        "INSERT INTO picture_texts(file_id, text, updated_at) "
        "VALUES(:f, :t, CURRENT_TIMESTAMP)",
        f=file_id,
        t=body or "",
    )
    if lines is not None:
        _seed_lines(seed, file_id, lines)


def _find(api, q: str, file_id: str = "pic1") -> dict:
    response = api(f"/files/{file_id}/find", {"q": q}, drive=DRIVE)
    assert response.status_code == 200, response.text
    return response.json()


def _boxes(body: dict) -> list[list[list[float]]]:
    return [row.get("boxes") for row in body["rows"]]


# ---------------------------------------------------------------------------
# I5 - which rows carry boxes, and which quads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "needle,line_index",
    [
        ("beta", 0),
        ("here", 1),
        ("gamma", 2),
        ("epsilon", 3),
    ],
)
def test_spec_addon_075_i5_a_hit_inside_one_line_marks_that_line(
    api, seed, visual, needle, line_index
):
    """SPEC-ADDON-075 I5: the row carries the quad of the line it hits."""
    visual(enabled=True)
    _seed_picture(seed, "pic1", lines=LINES)

    body = _find(api, needle)

    assert body["total"] == 1
    assert _boxes(body) == [[QUADS[line_index]]]


def test_spec_addon_075_i5_each_hit_on_the_same_line_carries_that_lines_quad(
    api, seed, visual
):
    """SPEC-ADDON-075 I5: two occurrences on one line, two rows, one quad each."""
    visual(enabled=True)
    _seed_picture(seed, "pic1", lines=LINES)

    body = _find(api, "needle")

    assert body["total"] == 2
    assert _boxes(body) == [[QUADS[1]], [QUADS[1]]]


@pytest.mark.parametrize(
    "needle,line_indices",
    [
        ("beta\nneedle", [0, 1]),
        ("here\ngamma", [1, 2]),
    ],
    ids=["first-two", "middle-two"],
)
def test_spec_addon_075_i5_a_needle_over_two_lines_marks_both_in_line_order(
    api, seed, visual, needle, line_indices
):
    """SPEC-ADDON-075 I5: a newline in the needle covers both lines."""
    visual(enabled=True)
    _seed_picture(seed, "pic1", lines=LINES)

    body = _find(api, needle)

    assert body["total"] == 1
    assert _boxes(body) == [[QUADS[index] for index in line_indices]]


def test_spec_addon_075_i5_a_newline_or_a_line_edge_overlaps_nothing(
    search_index, seed
):
    """SPEC-ADDON-075 I5: touching an edge or sitting on the newline marks no line."""
    from app import file_find

    _seed_picture(seed, "pic1", lines=LINES)

    on_newline = file_find.find_in_file(
        "pic1", DRIVE, "\n", include_scenes=False
    )
    at_edge = file_find.find_in_file(
        "pic1", DRIVE, "beta\n", include_scenes=False
    )

    assert on_newline["total"] == 3
    assert all("boxes" not in row for row in on_newline["rows"])
    assert at_edge["total"] == 1
    assert at_edge["rows"][0]["boxes"] == [QUADS[0]]


def test_spec_addon_075_i5_an_empty_kept_line_is_never_in_boxes(api, seed, visual):
    """SPEC-ADDON-075 I5: the empty line keeps the joined text equal, not a mark."""
    visual(enabled=True)
    lines = [
        {"text": "alpha beta", "quad": QUADS[0], "score": 0.9},
        {"text": "", "quad": QUADS[1], "score": None},
        {"text": "gamma delta", "quad": QUADS[2], "score": 0.7},
    ]
    _seed_picture(seed, "pic1", lines=lines)

    spanning = _find(api, "beta\n\ngamma")
    inside = _find(api, "gamma")

    assert spanning["total"] == 1
    assert _boxes(spanning) == [[QUADS[0], QUADS[2]]]
    assert _boxes(inside) == [[QUADS[2]]]
    assert all(QUADS[1] not in row["boxes"] for row in spanning["rows"])


def test_spec_addon_075_i5_boxes_is_never_an_empty_list(api, seed, visual):
    """SPEC-ADDON-075 I5: a row either carries the overlapped quads or no field."""
    visual(enabled=True)
    _seed_picture(seed, "pic1", lines=LINES)

    body = _find(api, "needle")

    for row in body["rows"]:
        assert row.get("boxes")
        assert isinstance(row["boxes"], list)


def test_spec_addon_075_i5_a_picture_with_no_lines_has_no_boxes(api, seed, visual):
    """SPEC-ADDON-075 I5: a picture read before the change has no marks."""
    visual(enabled=True)
    _seed_picture(seed, "pic1", text="alpha beta")

    body = _find(api, "alpha")

    assert body["total"] == 1
    assert all("boxes" not in row for row in body["rows"])


def test_spec_addon_075_i5_a_picture_whose_chunks_merge_differently_has_no_boxes(
    api, seed, visual
):
    """SPEC-ADDON-075 I5: usable needs the chunk merge to equal the joined lines."""
    visual(enabled=True)
    _seed_picture(
        seed,
        "pic1",
        lines=LINES,
        chunks=["alpha betaX\nneedle needle here\ngamma delta\nepsilon"],
    )

    body = _find(api, "beta")

    assert body["total"] == 1
    assert "boxes" not in body["rows"][0]


def test_spec_addon_075_i5_other_kinds_of_rows_never_have_boxes(api, seed, visual):
    """SPEC-ADDON-075 I5: a transcript and a text row are not pictures."""
    visual(enabled=True)
    seed.file("vid", "vid.mp4", mime="video/mp4")
    _flag(seed, "vid", whisper_indexed=True)
    seed.transcript("vid", 0, "alpha spoken", times=(0.0, 1.0))
    seed.file("doc", "doc.md", mime="text/markdown")
    _flag(seed, "doc", text_indexed=True)
    seed.text_chunk("doc", 0, "alpha written")
    seed.index.sql(
        "INSERT INTO picture_texts(file_id, text, updated_at) "
        "VALUES('doc', 'alpha written', CURRENT_TIMESTAMP)"
    )
    _seed_lines(seed, "doc", [
        {"text": "alpha written", "quad": [0, 0, 1, 0, 1, 1, 0, 1], "score": None},
    ])

    video = _find(api, "alpha", "vid")
    document = _find(api, "alpha", "doc")

    assert video["total"] == 1
    assert document["total"] == 1
    assert all("boxes" not in row for row in video["rows"] + document["rows"])


def test_spec_addon_075_i5_a_correction_removes_boxes_and_a_restore_returns_them(
    search_index, seed, monkeypatch
):
    """SPEC-ADDON-075 I5: boxes follow the stored text, which the lines never do."""
    from app import addon_events, file_find

    _seed_picture(seed, "pic1", lines=LINES)
    client, _ = build_api(monkeypatch)

    async def _no_event(*_args, **_kwargs):
        return True

    monkeypatch.setattr(addon_events, "post_addon_event", _no_event)
    assert client.get("/files/pic1/picture-text", headers={"X-Lit-Drive": DRIVE}).status_code == 200
    before = file_find.find_in_file("pic1", DRIVE, "alpha", include_scenes=False)
    assert before["rows"][0]["boxes"] == [QUADS[0]]

    corrected = client.put(
        "/files/pic1/picture-text",
        json={"text": "alpha fixed", "expected_text": JOINED},
        headers={"X-Lit-Drive": DRIVE},
    )
    assert corrected.status_code == 200, corrected.text
    after = file_find.find_in_file("pic1", DRIVE, "alpha", include_scenes=False)

    assert after["total"] == 1
    assert "boxes" not in after["rows"][0]

    restored = client.post(
        "/files/pic1/picture-text/restore", headers={"X-Lit-Drive": DRIVE}
    )
    assert restored.status_code == 200, restored.text
    again = file_find.find_in_file("pic1", DRIVE, "beta", include_scenes=False)

    assert again["total"] == 1
    assert again["rows"][0]["boxes"] == [QUADS[0]]


# ---------------------------------------------------------------------------
# I6 - the answer is otherwise unchanged
# ---------------------------------------------------------------------------


def test_spec_addon_075_i6_lines_present_answer_equals_lines_absent_except_boxes(
    api, seed, visual
):
    """SPEC-ADDON-075 I6: total, rows, order, snippets and offsets do not move."""
    visual(enabled=True)
    _seed_picture(seed, "pic1", lines=LINES)
    _seed_picture(seed, "pic2", text=JOINED)

    with_lines = _find(api, "needle", "pic1")
    without_lines = _find(api, "needle", "pic2")

    stripped = {
        **with_lines,
        "rows": [
            {key: value for key, value in row.items() if key != "boxes"}
            for row in with_lines["rows"]
        ],
    }
    assert stripped == without_lines
    assert all(row.get("boxes") for row in with_lines["rows"])


def test_spec_addon_075_i6_a_find_request_writes_nothing(api, seed, visual, statements):
    """SPEC-ADDON-075 I6: reading the lines is a read."""
    visual(enabled=True)
    _seed_picture(seed, "pic1", lines=LINES)

    statements.reset()
    body = _find(api, "needle")

    assert body["total"] == 2
    assert statements.statements, "the route must read the search DB"
    assert statements.writes() == []


def test_spec_addon_075_i6_a_failed_lines_read_returns_the_answer_without_boxes(
    api, seed, visual, caplog
):
    """SPEC-ADDON-075 I6: a broken lines read is a warning, not a failed find."""
    caplog.set_level(logging.WARNING)
    visual(enabled=True)
    _seed_picture(seed, "pic1", lines=LINES)
    expected = _find(api, "needle")
    seed.index.sql("ALTER TABLE picture_lines RENAME TO picture_lines_unreadable")

    response = api("/files/pic1/find", {"q": "needle"}, drive=DRIVE)

    assert response.status_code == 200, response.text
    body = response.json()
    assert {
        **expected,
        "rows": [
            {key: value for key, value in row.items() if key != "boxes"}
            for row in expected["rows"]
        ],
    } == body
    assert all("boxes" not in row for row in body["rows"])
    assert any(
        "pic1" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


# ---------------------------------------------------------------------------
# I7 - the route reads the requested file's lines only
# ---------------------------------------------------------------------------


def test_spec_addon_075_i7_each_picture_gets_its_own_quads(api, seed, visual):
    """SPEC-ADDON-075 I7: two pictures with different lines keep them apart."""
    visual(enabled=True)
    first = [{"text": "alpha beta", "quad": QUADS[0], "score": 0.9}]
    second = [{"text": "alpha beta", "quad": QUADS[3], "score": 0.5}]
    _seed_picture(seed, "pic1", lines=first)
    _seed_picture(seed, "pic2", lines=second)

    mine = _find(api, "alpha", "pic1")
    other = _find(api, "alpha", "pic2")

    assert _boxes(mine) == [[QUADS[0]]]
    assert _boxes(other) == [[QUADS[3]]]
