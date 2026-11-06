"""Purges and the shared readers (SPEC-ADDON-075 I7-I8).

SPEC-ID: SPEC-ADDON-075

A purge removes the picture's lines with its other rows; the frame and
scanned-page readers share the engine but write no lines.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import text

from tests.ocr_fixtures import (  # noqa: F401
    pillow_real,
    seed_indexed_file,
)
from tests.search_endpoint_db import DRIVE, seed, search_index
from tests.test_frame_page_ocr_readers import (  # noqa: F401
    frame_reader,
    media,
    page_reader,
    search_db,
)

pytestmark = pytest.mark.usefixtures("pillow_real")


def _seed_lines(seed, file_id: str, lines: list[dict]) -> None:
    seed.index.sql(
        "INSERT INTO picture_lines(file_id, lines_json, updated_at) "
        "VALUES(:f, :j, CURRENT_TIMESTAMP)",
        f=file_id,
        j=json.dumps(lines),
    )


def _one(seed, sql: str, **params):
    with seed.index.engine.connect() as conn:
        return conn.execute(text(sql), params).fetchall()


def _has_lines(seed, file_id: str) -> bool:
    return bool(_one(
        seed, "SELECT 1 FROM picture_lines WHERE file_id = :f", f=file_id
    ))


def _seed_picture(seed, file_id: str, drive: str = DRIVE) -> None:
    seed.file(file_id, f"{file_id}.png", drive=drive, mime="image/png")
    seed.index.sql(
        "UPDATE indexed_files SET text_indexed = 0, ocr_indexed = 1 "
        "WHERE file_id = :f",
        f=file_id,
    )
    seed.text_chunk(file_id, 0, "alpha beta")
    seed.index.sql(
        "INSERT INTO picture_texts(file_id, text, updated_at) "
        "VALUES(:f, 'alpha beta', CURRENT_TIMESTAMP)",
        f=file_id,
    )
    _seed_lines(seed, file_id, [
        {"text": "alpha beta", "quad": [0, 0, 1, 0, 1, 1, 0, 1], "score": 0.9},
    ])


@pytest.mark.parametrize("entry", ["file", "bulk", "drive"])
def test_spec_addon_075_i7_every_purge_path_deletes_the_lines_row(
    search_index, seed, monkeypatch, entry
):
    """SPEC-ADDON-075 I7: the lines go with the picture_texts row."""
    _seed_picture(seed, "pic1")
    _seed_picture(seed, "pic2", drive="other-drive")
    assert _has_lines(seed, "pic1")

    if entry == "file":
        from app.indexer import _purge_file

        _purge_file("pic1")
    elif entry == "bulk":
        from app.indexer import _purge_files

        _purge_files(["pic1"])
    else:
        from app import purge

        monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: None)
        assert purge.purge_drive(DRIVE) == 1

    assert not _has_lines(seed, "pic1")
    assert _one(
        seed, "SELECT 1 FROM picture_texts WHERE file_id = :f", f="pic1"
    ) == []
    assert _has_lines(seed, "pic2")


def _with_boxes(engine):
    def call(image):
        answer = engine(image)
        width, height = image.size
        box = [[0, 0], [width, 0], [width, height], [0, height]]
        answer.boxes = [box for _ in answer.txts]
        answer.scores = [0.9 for _ in answer.txts]
        return answer

    return call


def _lines_count(search_db, file_id: str) -> int:
    return search_db.scalar(
        "SELECT COUNT(*) FROM picture_lines WHERE file_id = :f", f=file_id
    )


def test_spec_addon_075_i8_frame_ocr_writes_no_lines_and_keeps_its_text(
    frame_reader, search_db, media, monkeypatch
):
    """SPEC-ADDON-075 I8: a frame read is not a picture read."""
    module, engine = frame_reader
    monkeypatch.setattr(module, "_get_engine", lambda: _with_boxes(engine))
    seed_indexed_file(
        search_db, "vid1", "video/mp4", media["landscape"], file_type="video",
    )

    assert module.index_frame_ocr("vid1") is True

    assert _lines_count(search_db, "vid1") == 0
    assert search_db.rows(
        "SELECT chunk_index, start_time, end_time, text FROM video_frame_texts "
        "WHERE file_id = 'vid1' ORDER BY chunk_index"
    ) == [
        (0, 0.0, 10.0, "ALPHA"),
        (1, 10.0, 13.0, "BETA"),
    ]
    assert search_db.ocr_fts("vid1") == [(0, "", "ALPHA"), (1, "", "BETA")]


def test_spec_addon_075_i8_scanned_page_ocr_writes_no_lines_and_keeps_its_rows(
    page_reader, search_db, media, monkeypatch
):
    """SPEC-ADDON-075 I8: a page read is not a picture read."""
    module, engine = page_reader
    monkeypatch.setattr(module, "_get_engine", lambda: _with_boxes(engine))
    seed_indexed_file(
        search_db, "pdf1", "application/pdf", media["scanned"],
        file_type="document", text_indexed=True,
    )

    assert module.index_page_ocr("pdf1") is True

    assert _lines_count(search_db, "pdf1") == 0
    assert search_db.rows(
        "SELECT page, text FROM pdf_page_ocr_texts WHERE file_id = 'pdf1' "
        "ORDER BY page"
    ) == [(1, "PAGE ONE"), (2, "PAGE TWO")]
    assert search_db.flag("pdf1", "page_ocr_indexed") == 1
    assert search_db.flag("pdf1", "text_indexed") == 0
