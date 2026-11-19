"""SPEC-ADDON-117 §4: the two columns, and which ``.loft``s the ``vec_clip_change``
dimension reset and the text-model rebuild reopen.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import create_engine, text

import app.database as database
from app.models import IndexedFile
from tests.test_frame_page_ocr_schema import (  # noqa: F401 - fixture and helpers
    _GRANITE_97,
    _RURI_310,
    _columns,
    _patch_text_model,
    _run_text_migration,
    _skeleton,
    init_database,
)
from tests.vec_partition_db import search_index, start  # noqa: F401 - fixture

LOFT_MIME = "application/vnd.litloft.loft+json"
NEW_COLUMNS = ("loft_source", "loft_fetch_failures")


def test_spec_addon_117_the_migration_adds_the_two_columns_to_an_existing_database(
    init_database,
):
    begin, path = init_database
    engine = create_engine(f"sqlite:///{path}")
    from app.database import Base

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for column in NEW_COLUMNS:
            if column in _columns(engine, "indexed_files"):
                conn.execute(text(f"ALTER TABLE indexed_files DROP COLUMN {column}"))
        conn.execute(text(
            "INSERT INTO indexed_files(file_id, drive, filename, file_path, "
            "file_type, mime_type, file_size, active, title, description, "
            "tags_text, indexed_at, updated_at) "
            "VALUES('loft1', 'drive1', 'a.loft', '/drives/drive1/a.loft', "
            "'video', :mime, 10, 1, '', '', '', "
            "'2026-11-16 10:00:00', '2026-11-16 10:00:00')"
        ), {"mime": LOFT_MIME})
    engine.dispose()

    begin()
    begin()

    reader = create_engine(f"sqlite:///{path}")
    assert set(NEW_COLUMNS) <= set(_columns(reader, "indexed_files"))
    with reader.connect() as conn:
        assert tuple(conn.execute(text(
            "SELECT loft_source, loft_fetch_failures FROM indexed_files "
            "WHERE file_id = 'loft1'"
        )).fetchone()) == (None, 0)
    reader.dispose()


def test_spec_addon_117_a_text_model_change_rebuilds_the_frame_text_of_youtube_lofts_only(
    tmp_path, monkeypatch,
):
    engine = _skeleton(tmp_path, vec_text_dim=384, recorded_model=_GRANITE_97)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE indexed_files ADD COLUMN loft_source TEXT"))
        conn.execute(text(
            "ALTER TABLE indexed_files ADD COLUMN loft_fetch_failures INTEGER NOT NULL DEFAULT 0"
        ))
        conn.execute(text(
            "INSERT INTO indexed_files(file_id, drive, file_type, mime_type, "
            " frame_ocr_read, frame_ocr_indexed, loft_source) VALUES "
            "('native', 'd', 'video', 'video/mp4', 1, 1, NULL),"
            "('youtube', 'd', 'video', :m, 1, 1, 'youtube'),"
            "('other', 'd', 'video', :m, 1, 1, 'other'),"
            "('unclassified', 'd', 'video', :m, 1, 1, NULL)"
        ), {"m": LOFT_MIME})
    _patch_text_model(monkeypatch, _RURI_310)

    _run_text_migration(engine)

    with engine.connect() as conn:
        rows = dict(conn.execute(text(
            "SELECT file_id, frame_ocr_read || ',' || frame_ocr_indexed FROM indexed_files"
        )).fetchall())
    assert rows == {
        "native": "1,0", "youtube": "1,0", "other": "1,1", "unclassified": "1,1",
    }


def _add(file_id: str, mime: str) -> None:
    with database.get_search_db() as s:
        s.add(IndexedFile(
            file_id=file_id, drive="family", filename=file_id,
            file_path=f"/drives/family/{file_id}", file_type="video", mime_type=mime,
            file_size=1, active=True, frame_scan_indexed=True,
        ))


def test_spec_addon_117_a_change_table_dimension_reset_moves_no_flag(
    search_index,  # noqa: F811
):
    """SPEC-ADDON-145 item 5, revised SPEC-ADDON-111 I8: the rebuild leaves every flag."""
    for file_id, mime in (("native", "video/mp4"), ("youtube", LOFT_MIME),
                          ("capped", LOFT_MIME), ("other", LOFT_MIME),
                          ("unclassified", LOFT_MIME)):
        _add(file_id, mime)
    for file_id, source, failures in (("youtube", "youtube", 0), ("capped", "youtube", 3),
                                      ("other", "other", 0), ("unclassified", None, 0)):
        search_index.sql(
            "UPDATE indexed_files SET loft_source = :s, loft_fetch_failures = :n "
            "WHERE file_id = :f",
            s=source, n=failures, f=file_id,
        )
    search_index.sql("DROP TABLE vec_clip_change")
    search_index.sql(
        "CREATE VIRTUAL TABLE vec_clip_change USING vec0(embedding_id TEXT PRIMARY KEY, "
        "embedding_type text partition key, vector float[8])"
    )
    search_index.put_vector("vec_clip_change", "chg_youtube_3", np.ones(8),
                            partition="clip_change")

    start()

    with search_index.engine.connect() as conn:
        rows = dict(conn.execute(text(
            "SELECT file_id, frame_scan_indexed FROM indexed_files"
        )).fetchall())
        capped_failures = conn.execute(text(
            "SELECT loft_fetch_failures FROM indexed_files WHERE file_id = 'capped'"
        )).scalar()
    assert rows == {
        "native": 1, "youtube": 1, "capped": 1, "other": 1, "unclassified": 1,
    }
    assert capped_failures == 3
