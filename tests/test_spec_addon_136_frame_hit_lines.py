"""SPEC-ADDON-136: a keyword frame-text passage carries the box of the line its
match falls on, and a frame-text find row carries the boxes of the lines its
occurrence overlaps, with the frame size those boxes were measured on.

The segments are seeded rows: ``video_frame_texts`` chunks rebuilt from the
lines' joined text, the same chunks in ``fts_text_content``, and one
``video_frame_lines`` row.
"""

from __future__ import annotations

import json
import logging

import pytest

from app.config import SearchConfig
from app.extractors.base import ContentExtractor
from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    Seed,
    api,
    patch_everywhere,
    search_index,
    statements,
)
from tests.test_search_by_image import (  # noqa: F401 - fixtures
    ask,
    client,
    configure,
    hydration,
    models,
    world,
)
from tests.test_search_screen_text import (  # noqa: F401 - fixtures
    _vector,
    embedded,
    no_hydration,
    ready,
    thresholds,
)
from tests.ocr_fixtures import pillow_real  # noqa: F401 - fixture

LOFT_MIME = "application/vnd.litloft.loft+json"
LINE_FIELDS = ("line_box", "line_match", "line_frame_url")

Q0 = [0.1, 0.1, 0.4, 0.1, 0.4, 0.15, 0.1, 0.15]
Q1 = [0.1, 0.4, 0.7, 0.4, 0.7, 0.46, 0.1, 0.46]
Q2 = [0.6, 0.85, 0.8, 0.85, 0.8, 0.9, 0.6, 0.9]

LINES = [
    {"text": "Quarterly report", "quad": Q0},
    {"text": "warehouse moved to Osaka", "quad": Q1},
    {"text": "page 3", "quad": Q2},
]


def _joined(lines: list[dict]) -> str:
    return "\n".join(line["text"] for line in lines)


def _file(file_id: str, *, mime: str = "video/mp4", filename: str | None = None) -> None:
    from app.models import IndexedFile
    import app.database as database

    name = filename or f"{file_id}.mp4"
    with database.get_search_db() as session:
        session.add(IndexedFile(
            file_id=file_id, drive=DRIVE, filename=name,
            file_path=f"/drives/{DRIVE}/{name}", file_type="video",
            mime_type=mime, file_size=1, active=True,
        ))


def _flag(seed: Seed, file_id: str, **flags) -> None:
    for column, value in flags.items():
        seed.index.sql(
            f"UPDATE indexed_files SET {column} = :v WHERE file_id = :f",
            v=1 if value else 0, f=file_id,
        )


def _segment(
    seed: Seed,
    file_id: str,
    segment_index: int,
    start: float,
    end: float,
    lines: list[dict] | None,
    *,
    chunk_base: int = 0,
    chunks: list[str] | None = None,
    lines_json: str | None = None,
    size: tuple[int, int] | None = None,
    keyword: bool = True,
) -> list[str]:
    """Writes the segment's chunks and, unless ``lines`` and ``lines_json`` are both
    None, its ``video_frame_lines`` row. Returns the chunk texts."""
    body = _joined(lines or [])
    texts = chunks if chunks is not None else ContentExtractor.chunk_text(body)
    for offset, text in enumerate(texts):
        if keyword:
            seed.text_chunk(file_id, chunk_base + offset, text, page="")
        seed.index.sql(
            "INSERT INTO video_frame_texts(file_id, chunk_index, start_time, end_time, text) "
            "VALUES(:f, :c, :s, :e, :t)",
            f=file_id, c=chunk_base + offset, s=start, e=end, t=text,
        )
    if lines is None and lines_json is None:
        return texts
    stored = lines_json if lines_json is not None else json.dumps(lines)
    if size is None:
        seed.index.sql(
            "INSERT INTO video_frame_lines(file_id, segment_index, start_time, end_time, lines_json) "
            "VALUES(:f, :i, :s, :e, :j)",
            f=file_id, i=segment_index, s=start, e=end, j=stored,
        )
    else:
        seed.index.sql(
            "INSERT INTO video_frame_lines"
            "(file_id, segment_index, start_time, end_time, lines_json, frame_width, frame_height) "
            "VALUES(:f, :i, :s, :e, :j, :w, :h)",
            f=file_id, i=segment_index, s=start, e=end, j=stored, w=size[0], h=size[1],
        )
    return texts


def _ok(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


def _search(api, q: str) -> dict:  # noqa: F811
    return _ok(api("/search", {"q": q, "passages": "true"}))


def _passages(body: dict, file_id: str) -> list[dict]:
    (hit,) = [r for r in body["results"] if r["file_id"] == file_id]
    return hit["passages"]


def _line_fields(passage: dict) -> tuple:
    return tuple(passage[name] for name in LINE_FIELDS)


def _frame_url(file_id: str, start_text: str) -> str:
    return f"/api/addons/intelligence/files/{file_id}/frame?t={start_text}&size=large"


# ---------------------------------------------------------------------------
# Steps 4-6, I1 - the passage's line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start, start_text",
    [
        pytest.param(725.0, "725.000", id="whole-second"),
        pytest.param(12.48, "12.480", id="two-decimals"),
        pytest.param(0.0, "0.000", id="zero"),
    ],
)
def test_spec_addon_136_i1_a_keyword_passage_carries_its_lines_box_match_and_large_frame(
    api, search_index, ready, no_hydration, statements, start, start_text,  # noqa: F811
):
    seed = Seed(search_index)
    _file("vid")
    _segment(seed, "vid", 0, start, start + 6.5, LINES)

    statements.reset()
    body = _search(api, "moved")

    (passage,) = _passages(body, "vid")
    assert passage["kind"] == "screen_text"
    assert passage["keyword"] is True
    assert passage["line_box"] == Q1
    assert passage["line_match"] == [0.4167, 0.625]
    assert passage["line_frame_url"] == _frame_url("vid", start_text)
    assert statements.writes() == []


@pytest.mark.parametrize(
    "lines, query, line_index, match",
    [
        pytest.param(
            [
                {"text": "warehouse north", "quad": Q0},
                {"text": "stock", "quad": Q1},
                {"text": "warehouse south", "quad": Q2},
            ],
            "warehouse", 0, [0.0, 0.6],
            id="the-first-of-several-lines",
        ),
        pytest.param(
            [
                {"text": "Stock list", "quad": Q0},
                {"text": "THE WAREHOUSE", "quad": Q1},
            ],
            "warehouse", 1, [0.3077, 1.0],
            id="ascii-folded",
        ),
        pytest.param(
            [
                {"text": "Osaka branch", "quad": Q0},
                {"text": "the warehouse", "quad": Q1},
            ],
            "warehouse osaka", 0, [0.0, 0.4167],
            id="the-occurrence-that-starts-first-not-the-first-term",
        ),
    ],
)
def test_spec_addon_136_i1_the_line_is_the_first_one_the_first_occurrence_overlaps(
    api, search_index, ready, no_hydration, lines, query, line_index, match,  # noqa: F811
):
    seed = Seed(search_index)
    _file("vid")
    _segment(seed, "vid", 0, 30.0, 40.0, lines)

    body = _search(api, query)

    (passage,) = _passages(body, "vid")
    assert passage["keyword"] is True
    folded_terms = {term.lower() for term in passage["highlight_terms"]}
    assert set(query.split()) <= folded_terms, passage["highlight_terms"]
    assert passage["line_box"] == lines[line_index]["quad"]
    assert passage["line_match"] == match
    assert passage["line_frame_url"] == _frame_url("vid", "30.000")


def test_spec_addon_136_i1_every_other_screen_text_passage_has_all_three_null(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    """A usable keyword passage beside each case that must stay null, in one answer."""
    seed = Seed(search_index)
    mismatched = _joined(LINES) + " extra"
    seven_numbers = [dict(LINES[0]), dict(LINES[1], quad=Q1[:7]), dict(LINES[2])]
    text_number = [dict(LINES[0]), dict(LINES[1], quad=[*Q1[:7], "0.46"]), dict(LINES[2])]

    _file("good")
    _segment(seed, "good", 0, 10.0, 20.0, LINES)
    _file("loft", mime=LOFT_MIME, filename="loft.loft")
    _segment(seed, "loft", 0, 10.0, 20.0, LINES)
    _file("nolines")
    _segment(seed, "nolines", 0, 10.0, 20.0, None, chunks=[_joined(LINES)])
    _file("mismatch")
    _segment(seed, "mismatch", 0, 10.0, 20.0, LINES, chunks=[mismatched])
    _file("sevennum")
    _segment(seed, "sevennum", 0, 10.0, 20.0, seven_numbers)
    _file("textnum")
    _segment(seed, "textnum", 0, 10.0, 20.0, text_number)
    _file("vector")
    _segment(seed, "vector", 0, 10.0, 20.0, LINES, keyword=False)
    _vector(search_index, "vector", chunk_index=0, preview=_joined(LINES))

    body = _search(api, "moved")

    keyword_null = {}
    for file_id in ("loft", "nolines", "mismatch", "sevennum", "textnum"):
        (passage,) = _passages(body, file_id)
        assert passage["kind"] == "screen_text", file_id
        assert passage["keyword"] is True, file_id
        keyword_null[file_id] = passage
    vector_passages = [
        p for p in _passages(body, "vector") if p["kind"] == "screen_text"
    ]
    assert vector_passages, "the vector-only file must still give a passage"
    assert all(p["keyword"] is False for p in vector_passages)

    (good,) = _passages(body, "good")
    assert _line_fields(good) == (Q1, [0.4167, 0.625], _frame_url("good", "10.000"))
    for file_id, passage in keyword_null.items():
        assert _line_fields(passage) == (None, None, None), file_id
    for passage in vector_passages:
        assert _line_fields(passage) == (None, None, None)


def test_spec_addon_136_step_6_the_answer_is_otherwise_what_it_is_without_the_lines(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    seed = Seed(search_index)
    _file("vid")
    _segment(seed, "vid", 0, 10.0, 20.0, LINES)
    _segment(seed, "vid", 1, 30.0, 40.0, [{"text": "moved again", "quad": Q2}], chunk_base=1)
    seed.file("note", "note.md", mime="text/markdown")
    seed.text_chunk("note", 0, "the office moved", page="")

    with_lines = _search(api, "moved")
    seed.index.sql("DELETE FROM video_frame_lines")
    without_lines = _search(api, "moved")

    def stripped(body: dict) -> dict:
        return {
            **body,
            "results": [
                {
                    **hit,
                    "passages": [
                        {k: v for k, v in p.items() if k not in LINE_FIELDS}
                        for p in hit.get("passages") or []
                    ],
                }
                for hit in body["results"]
            ],
        }

    def by_start(body: dict) -> dict:
        return {p["start"]: p["line_box"] for p in _passages(body, "vid")}

    assert by_start(with_lines) == {10.0: Q1, 30.0: Q2}
    assert by_start(without_lines) == {10.0: None, 30.0: None}
    assert stripped(with_lines) == stripped(without_lines)


# ---------------------------------------------------------------------------
# Failure 1 - the lines cannot be read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("breakage", ["malformed-json", "table-gone"])
def test_spec_addon_136_failure_1_unreadable_lines_give_nulls_and_one_warning_per_file(
    api, search_index, ready, no_hydration, caplog, breakage,  # noqa: F811
):
    seed = Seed(search_index)
    _file("broken")
    second = [{"text": "moved twice", "quad": Q2}]
    if breakage == "malformed-json":
        _segment(seed, "broken", 0, 10.0, 20.0, None, chunks=[_joined(LINES)], lines_json="{not json")
        _segment(seed, "broken", 1, 30.0, 40.0, None, chunks=[_joined(second)],
                 chunk_base=1, lines_json="[{]")
    else:
        _segment(seed, "broken", 0, 10.0, 20.0, LINES)
        _segment(seed, "broken", 1, 30.0, 40.0, second, chunk_base=1)
    _file("good")
    _segment(seed, "good", 0, 10.0, 20.0, LINES)
    expected_good = (Q1, [0.4167, 0.625], _frame_url("good", "10.000"))
    if breakage == "table-gone":
        expected_good = (None, None, None)
        seed.index.sql("ALTER TABLE video_frame_lines RENAME TO video_frame_lines_gone")

    caplog.set_level(logging.WARNING)
    body = _search(api, "moved")

    broken = _passages(body, "broken")
    assert len(broken) == 2
    assert [_line_fields(p) for p in broken] == [(None, None, None)] * 2
    (good,) = _passages(body, "good")
    assert _line_fields(good) == expected_good
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len([w for w in warnings if "broken" in w]) == 1, warnings
    good_warnings = [w for w in warnings if "good" in w]
    assert len(good_warnings) == (1 if breakage == "table-gone" else 0), warnings


OTHER_SHAPES = [
    pytest.param('{"text": "warehouse moved to Osaka", "quad": [0, 0, 1, 0, 1, 1, 0, 1]}', id="not-a-list"),
    pytest.param('[{"quad": [0.1, 0.4, 0.7, 0.4, 0.7, 0.46, 0.1, 0.46]}]', id="no-text"),
    pytest.param('[{"text": 7, "quad": [0.1, 0.4, 0.7, 0.4, 0.7, 0.46, 0.1, 0.46]}]', id="text-not-a-string"),
    pytest.param('[{"text": "warehouse moved to Osaka"}]', id="no-quad"),
]


@pytest.mark.parametrize("lines_json", OTHER_SHAPES)
def test_spec_addon_136_failure_2_valid_json_of_another_shape_is_no_lines_without_a_warning(
    api, search_index, ready, no_hydration, caplog, lines_json,  # noqa: F811
):
    seed = Seed(search_index)
    single = [{"text": "warehouse moved to Osaka", "quad": Q1}]
    _file("shaped")
    _segment(seed, "shaped", 0, 10.0, 20.0, None, chunks=[_joined(single)], lines_json=lines_json)
    _file("good")
    _segment(seed, "good", 0, 10.0, 20.0, LINES)

    caplog.set_level(logging.WARNING)
    body = _search(api, "moved")

    (shaped,) = _passages(body, "shaped")
    (good,) = _passages(body, "good")
    assert _line_fields(good) == (Q1, [0.4167, 0.625], _frame_url("good", "10.000"))
    assert _line_fields(shaped) == (None, None, None)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert [w for w in warnings if "shaped" in w] == []


# ---------------------------------------------------------------------------
# Step 4 - the image-query answer carries the three fields, null
# ---------------------------------------------------------------------------


def test_spec_addon_136_step_4_the_image_query_answer_carries_the_three_fields_as_null(
    client, world, models,  # noqa: F811
):
    models.set_lines("a screenshot of a page with enough text")
    world.chunk("pdf", 0.90, page=7, chunk_index=3, preview="the stored excerpt")

    (hit,) = _ok(ask(client, world))["documents"]

    (passage,) = hit["passages"]
    assert {name: passage[name] for name in LINE_FIELDS} == dict.fromkeys(LINE_FIELDS)


# ---------------------------------------------------------------------------
# Steps 7-9, I2 - find rows
# ---------------------------------------------------------------------------


FIND_ID = "fq7zr"


def _video(seed: Seed, file_id: str = FIND_ID) -> None:
    seed.file(file_id, f"{file_id}.mp4", mime="video/mp4")
    _flag(seed, file_id, whisper_indexed=False, frame_ocr_indexed=True)


def _find(api, q: str, file_id: str = FIND_ID) -> dict:  # noqa: F811
    return _ok(api(f"/files/{file_id}/find", {"q": q}, drive=DRIVE))


def test_spec_addon_136_i2_a_frame_row_carries_the_overlapped_quads_and_the_frame_size(
    api, search_index, statements,  # noqa: F811
):
    seed = Seed(search_index)
    _video(seed)
    _segment(seed, FIND_ID, 0, 10.0, 20.0, LINES, size=(1280, 720))

    statements.reset()
    body = _find(api, "moved")

    (row,) = body["rows"]
    assert row["kind"] == "scene"
    assert (row["start"], row["end"]) == (10.0, 20.0)
    assert row["boxes"] == [Q1]
    assert row["frame_size"] == [1280, 720]
    assert statements.writes() == []


def test_spec_addon_136_i2_an_occurrence_over_two_lines_carries_both_in_line_order(search_index):
    from app import file_find

    seed = Seed(search_index)
    _video(seed)
    _segment(seed, FIND_ID, 0, 10.0, 20.0, LINES)

    spanning = file_find.find_in_file(FIND_ID, DRIVE, "Osaka\npage", include_scenes=False)
    at_edge = file_find.find_in_file(FIND_ID, DRIVE, "Osaka\n", include_scenes=False)
    on_newline = file_find.find_in_file(FIND_ID, DRIVE, "\n", include_scenes=False)

    assert [row["boxes"] for row in spanning["rows"]] == [[Q1, Q2]]
    assert [row["boxes"] for row in at_edge["rows"]] == [[Q1]]
    assert on_newline["total"] == 2
    assert all("boxes" not in row and "frame_size" not in row for row in on_newline["rows"])


def test_spec_addon_136_i2_an_empty_line_is_never_in_boxes(search_index):
    from app import file_find

    seed = Seed(search_index)
    _video(seed)
    lines = [
        {"text": "alpha beta", "quad": Q0},
        {"text": "", "quad": Q1},
        {"text": "gamma delta", "quad": Q2},
    ]
    _segment(seed, FIND_ID, 0, 10.0, 20.0, lines)

    body = file_find.find_in_file(FIND_ID, DRIVE, "beta\n\ngamma", include_scenes=False)

    assert [row["boxes"] for row in body["rows"]] == [[Q0, Q2]]


def test_spec_addon_136_i2_frame_size_only_with_boxes_and_only_when_both_sizes_are_set(
    api, search_index,  # noqa: F811
):
    seed = Seed(search_index)
    _video(seed)
    sized = [{"text": "moved first", "quad": Q0}]
    unsized = [{"text": "moved second", "quad": Q1}]
    unusable = [{"text": "moved third", "quad": Q2}]
    _segment(seed, FIND_ID, 0, 10.0, 20.0, sized, size=(1280, 720))
    _segment(seed, FIND_ID, 1, 30.0, 40.0, unsized, chunk_base=1)
    _segment(seed, FIND_ID, 2, 50.0, 60.0, unusable, chunk_base=2,
             chunks=["moved third, misread"], size=(1280, 720))
    half = [{"text": "moved fourth", "quad": Q0}]
    _segment(seed, FIND_ID, 3, 70.0, 80.0, half, chunk_base=3)
    seed.index.sql(
        "UPDATE video_frame_lines SET frame_width = 1280 WHERE file_id = :f AND segment_index = 3",
        f=FIND_ID,
    )

    body = _find(api, "moved")

    rows = {row["start"]: row for row in body["rows"]}
    assert set(rows) == {10.0, 30.0, 50.0, 70.0}
    assert rows[10.0]["boxes"] == [Q0]
    assert rows[10.0]["frame_size"] == [1280, 720]
    assert rows[30.0]["boxes"] == [Q1]
    assert "frame_size" not in rows[30.0]
    assert "boxes" not in rows[50.0]
    assert "frame_size" not in rows[50.0]
    assert rows[70.0]["boxes"] == [Q0]
    assert "frame_size" not in rows[70.0]


def _strip_find(body: dict) -> dict:
    return {
        **body,
        "rows": [
            {k: v for k, v in row.items() if k not in ("boxes", "frame_size")}
            for row in body["rows"]
        ],
    }


def test_spec_addon_136_i2_the_find_answer_is_otherwise_what_it_is_without_the_lines(
    api, search_index,  # noqa: F811
):
    seed = Seed(search_index)
    _video(seed)
    _segment(seed, FIND_ID, 0, 10.0, 20.0, LINES)
    _segment(seed, FIND_ID, 1, 30.0, 40.0, [{"text": "moved moved", "quad": Q2}], chunk_base=1)
    seed.transcript(FIND_ID, 0, "it moved", times=(12.0, 14.0))
    _flag(seed, FIND_ID, whisper_indexed=True)

    with_lines = _find(api, "moved")
    seed.index.sql("DELETE FROM video_frame_lines")
    without_lines = _find(api, "moved")

    assert [row.get("boxes") for row in with_lines["rows"] if row["kind"] == "scene"] == [
        [Q1], [Q2], [Q2],
    ]
    assert _strip_find(with_lines) == without_lines


def test_spec_addon_136_failure_1_a_failed_lines_read_returns_find_without_boxes(
    api, search_index, caplog,  # noqa: F811
):
    seed = Seed(search_index)
    _video(seed)
    _segment(seed, FIND_ID, 0, 10.0, 20.0, LINES)
    expected = _find(api, "moved")
    assert expected["rows"][0]["boxes"] == [Q1]
    seed.index.sql("ALTER TABLE video_frame_lines RENAME TO video_frame_lines_gone")

    caplog.set_level(logging.WARNING)
    body = _find(api, "moved")

    assert body == _strip_find(expected)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len([w for w in warnings if FIND_ID in w]) == 1, warnings


# ---------------------------------------------------------------------------
# A segment longer than one chunk
# ---------------------------------------------------------------------------


LONG_LINE = " ".join(f"lorem{i:03d}" for i in range(140))
LONG_LINES = [
    {"text": LONG_LINE, "quad": Q0},
    {"text": "warehouse moved to Osaka", "quad": Q1},
]


def test_spec_addon_136_i1_a_segment_split_into_several_chunks_still_gives_its_line(
    api, search_index, ready, no_hydration,  # noqa: F811
):
    seed = Seed(search_index)
    _file("vid")
    chunks = _segment(seed, "vid", 0, 10.0, 20.0, LONG_LINES)
    assert len(chunks) > 1

    body = _search(api, "moved")

    passages = [p for p in _passages(body, "vid") if p["keyword"]]
    assert passages
    for passage in passages:
        assert _line_fields(passage) == (Q1, [0.4167, 0.625], _frame_url("vid", "10.000"))


def test_spec_addon_136_i2_a_segment_split_into_several_chunks_still_gives_boxes(
    api, search_index,  # noqa: F811
):
    seed = Seed(search_index)
    _video(seed)
    chunks = _segment(seed, FIND_ID, 0, 10.0, 20.0, LONG_LINES)
    assert len(chunks) > 1

    body = _find(api, "moved")

    assert body["total"] == 1
    assert [row["boxes"] for row in body["rows"]] == [[Q1]]


@pytest.mark.parametrize("lines_json", OTHER_SHAPES)
def test_spec_addon_136_failure_2_find_treats_valid_json_of_another_shape_as_no_lines(
    api, search_index, caplog, lines_json,  # noqa: F811
):
    seed = Seed(search_index)
    _video(seed)
    _segment(seed, FIND_ID, 0, 10.0, 20.0, LINES)
    single = [{"text": "warehouse moved to Osaka", "quad": Q1}]
    _segment(seed, FIND_ID, 1, 30.0, 40.0, None, chunk_base=1,
             chunks=[_joined(single)], lines_json=lines_json)

    caplog.set_level(logging.WARNING)
    body = _find(api, "moved")

    rows = {row["start"]: row for row in body["rows"]}
    assert rows[10.0]["boxes"] == [Q1]
    assert "boxes" not in rows[30.0]
    assert "frame_size" not in rows[30.0]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert [w for w in warnings if FIND_ID in w] == []
