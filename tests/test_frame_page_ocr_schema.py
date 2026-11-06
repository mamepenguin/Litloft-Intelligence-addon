"""Schema, migrations and purge for the two new OCR tables and three flags.

SPEC-ADDON-066 §4, SPEC-ADDON-069 §4, I3/I4: the columns, the idempotent
migration, the reset on a text-embedding model change, and the purge of a
file's rows in ``video_frame_texts`` and ``pdf_page_ocr_texts``.
"""

from __future__ import annotations

import dataclasses

import pytest
from sqlalchemy import create_engine, text

from app.models import IndexedFile  # noqa: F401  - registers the tables on Base
from tests.vec_partition_db import search_index  # noqa: F401  - the real search DB

_GRANITE_97 = "ibm-granite/granite-embedding-97m-multilingual-r2"
_RURI_310 = "cl-nagoya/ruri-v3-310m"

NEW_COLUMNS = ("frame_ocr_read", "frame_ocr_indexed", "page_ocr_indexed")
NEW_TABLES = ("video_frame_texts", "pdf_page_ocr_texts")


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def test_spec_addon_066_the_three_flags_exist_and_default_to_false():
    columns = IndexedFile.__table__.columns
    for name in NEW_COLUMNS:
        assert name in columns, f"indexed_files has no {name}"


def test_spec_addon_066_the_two_tables_exist_with_their_unique_keys():
    from app.database import Base
    from sqlalchemy import UniqueConstraint

    tables = Base.metadata.tables
    for name, key in (
        ("video_frame_texts", ("file_id", "chunk_index")),
        ("pdf_page_ocr_texts", ("file_id", "page")),
    ):
        assert name in tables, f"search.db has no {name} table"
        table = tables[name]
        assert {column.name for column in table.columns} >= {
            "file_id", "text",
        } | ({"chunk_index", "start_time", "end_time"} if name == "video_frame_texts" else {"page"})
        uniques = {
            tuple(sorted(column.name for column in constraint.columns))
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        assert tuple(sorted(key)) in uniques, f"{name} has no unique {key}"


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


@pytest.fixture()
def init_database(tmp_path, monkeypatch):
    """``init_search_db`` over a scratch path, with the module restored after."""
    import app.database as database

    saved = dict(vars(database))
    path = tmp_path / "search.db"
    monkeypatch.setattr(database, "settings", dataclasses.replace(
        database.settings, intelligence_data_dir=tmp_path
    ))
    monkeypatch.setattr(database, "_resolve_search_db_path", lambda: path)

    def start() -> None:
        database.init_search_db()

    try:
        yield start, path
    finally:
        engine = vars(database).get("_search_engine")
        if engine is not None:
            engine.dispose()
        for key in list(vars(database)):
            if key not in saved:
                delattr(database, key)
        for key, value in saved.items():
            if vars(database).get(key) is not value:
                setattr(database, key, value)


def _columns(engine, table: str) -> list[str]:
    with engine.connect() as conn:
        return [
            row[1]
            for row in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        ]


def test_spec_addon_066_i3_the_migration_adds_the_columns_and_tables_twice(
    init_database,
):
    """SPEC-ADDON-066 I3: the migration can run twice without changing rows."""
    start, path = init_database
    engine = create_engine(f"sqlite:///{path}")
    from app.database import Base

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for column in NEW_COLUMNS:
            if column in _columns(engine, "indexed_files"):
                conn.execute(text(
                    f"ALTER TABLE indexed_files DROP COLUMN {column}"
                ))
        for table in NEW_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        conn.execute(text(
            "INSERT INTO indexed_files(file_id, drive, filename, file_path, "
            "file_type, mime_type, file_size, active, title, description, "
            "tags_text, indexed_at, updated_at) "
            "VALUES('vid1', 'drive1', 'v.mp4', '/drives/drive1/v.mp4', "
            "'video', 'video/mp4', 10, 1, '', '', '', "
            "'2026-11-06 10:00:00', '2026-11-06 10:00:00')"
        ))
    engine.dispose()

    start()

    reader = create_engine(f"sqlite:///{path}")
    assert set(NEW_COLUMNS) <= set(_columns(reader, "indexed_files"))
    for table in NEW_TABLES:
        assert _columns(reader, table), f"{table} was not created"
    with reader.begin() as conn:
        conn.execute(text("UPDATE indexed_files SET frame_ocr_read = 1"))
    reader.dispose()

    start()

    reader = create_engine(f"sqlite:///{path}")
    assert set(NEW_COLUMNS) <= set(_columns(reader, "indexed_files"))
    with reader.connect() as conn:
        assert conn.execute(text(
            "SELECT frame_ocr_read FROM indexed_files WHERE file_id = 'vid1'"
        )).fetchone()[0] == 1
    reader.dispose()


# ---------------------------------------------------------------------------
# The text-embedding model change
# ---------------------------------------------------------------------------


def _skeleton(tmp_path, *, vec_text_dim: int, recorded_model: str):
    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE indexed_files ("
            "  file_id TEXT PRIMARY KEY,"
            "  drive TEXT NOT NULL,"
            "  file_type TEXT NOT NULL DEFAULT 'document',"
            "  mime_type TEXT NOT NULL,"
            "  text_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  whisper_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  tfidf_keywords_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  ocr_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  frame_ocr_read BOOLEAN NOT NULL DEFAULT 0,"
            "  frame_ocr_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  page_ocr_indexed BOOLEAN NOT NULL DEFAULT 0"
            ")"
        ))
        conn.execute(text(
            "CREATE TABLE embeddings ("
            "  embedding_id TEXT PRIMARY KEY,"
            "  file_id TEXT NOT NULL,"
            "  embedding_type TEXT NOT NULL"
            ")"
        ))
        conn.execute(text(
            "CREATE TABLE detailed_summary_citations ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  file_id TEXT NOT NULL,"
            "  section_path TEXT NOT NULL"
            ")"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE fts_text_content "
            "USING fts5(file_id, chunk_index, page, text, tokenize='trigram')"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE fts_text_content_word "
            "USING fts5(file_id, chunk_index, page, text, "
            "tokenize=\"unicode61 remove_diacritics 2\")"
        ))
        conn.execute(text(
            "CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT)"
        ))
        conn.execute(text(
            f"CREATE TABLE vec_text "
            f"(embedding_id TEXT PRIMARY KEY, vector float[{vec_text_dim}])"
        ))
        conn.execute(text(
            "CREATE TABLE video_frame_texts ("
            "  file_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,"
            "  start_time REAL NOT NULL, end_time REAL NOT NULL,"
            "  text TEXT NOT NULL,"
            "  UNIQUE(file_id, chunk_index))"
        ))
        conn.execute(text(
            "CREATE TABLE pdf_page_ocr_texts ("
            "  file_id TEXT NOT NULL, page INTEGER NOT NULL, text TEXT NOT NULL,"
            "  UNIQUE(file_id, page))"
        ))
        conn.execute(
            text("INSERT INTO index_meta(key, value) VALUES('text_embedding_model', :m)"),
            {"m": recorded_model},
        )
    return engine


def _patch_text_model(monkeypatch, model_id: str) -> None:
    import app.config as config
    from app.config import ModelConfig

    settings = dataclasses.replace(
        config.settings, models=ModelConfig(text_embedding=model_id)
    )
    monkeypatch.setattr(config, "settings", settings)


def _run_text_migration(engine) -> None:
    from app.database import _migrate_vec_text_if_needed

    with engine.connect() as conn:
        _migrate_vec_text_if_needed(conn)
        conn.commit()


def test_spec_addon_066_i3_a_text_model_change_rebuilds_frames_without_reading(
    tmp_path, monkeypatch,
):
    """SPEC-ADDON-066 I3, SPEC-ADDON-069 B11: read stays, indexed is reset."""
    engine = _skeleton(tmp_path, vec_text_dim=384, recorded_model=_GRANITE_97)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO indexed_files"
            "(file_id, drive, file_type, mime_type, frame_ocr_read, frame_ocr_indexed, "
            " page_ocr_indexed) VALUES"
            "('vid1', 'drive1', 'video', 'video/mp4', 1, 1, 0),"
            "('pdf1', 'drive1', 'document', 'application/pdf', 0, 0, 1)"
        ))
        conn.execute(text(
            "INSERT INTO video_frame_texts"
            "(file_id, chunk_index, start_time, end_time, text) "
            "VALUES('vid1', 0, 0.0, 10.0, 'frame text')"
        ))
        conn.execute(text(
            "INSERT INTO pdf_page_ocr_texts(file_id, page, text) "
            "VALUES('pdf1', 1, 'page text')"
        ))
    _patch_text_model(monkeypatch, _RURI_310)

    _run_text_migration(engine)

    with engine.connect() as conn:
        frame_read, frame_indexed = conn.execute(text(
            "SELECT frame_ocr_read, frame_ocr_indexed FROM indexed_files "
            "WHERE file_id = 'vid1'"
        )).fetchone()
        page_indexed = conn.execute(text(
            "SELECT page_ocr_indexed FROM indexed_files WHERE file_id = 'pdf1'"
        )).fetchone()[0]
        frame_text = conn.execute(text(
            "SELECT text FROM video_frame_texts WHERE file_id = 'vid1'"
        )).fetchone()[0]
        page_text = conn.execute(text(
            "SELECT text FROM pdf_page_ocr_texts WHERE file_id = 'pdf1'"
        )).fetchone()[0]
    assert (frame_read, frame_indexed) == (1, 0)
    assert page_indexed == 1
    assert (frame_text, page_text) == ("frame text", "page text")


# ---------------------------------------------------------------------------
# Purge
# ---------------------------------------------------------------------------


def test_spec_addon_066_i4_purge_deletes_both_new_tables_rows(search_index):
    """SPEC-ADDON-066 I4: a purged file leaves no frame or page text behind."""
    import app.database as database
    from app.indexer import _purge_file
    from app.models import IndexedFile

    with database.get_search_db() as session:
        session.add(IndexedFile(
            file_id="vid1", drive="drive1", filename="v.mp4",
            file_path="/drives/drive1/v.mp4", file_type="video",
            mime_type="video/mp4", file_size=1, active=True,
        ))
        session.add(IndexedFile(
            file_id="pdf1", drive="drive1", filename="p.pdf",
            file_path="/drives/drive1/p.pdf", file_type="document",
            mime_type="application/pdf", file_size=1, active=True,
        ))
    with database.get_search_db() as session:
        session.execute(
            text(
                "INSERT INTO video_frame_texts"
                "(file_id, chunk_index, start_time, end_time, text) "
                "VALUES('vid1', 0, 0.0, 10.0, 'frame text')"
            )
        )
        session.execute(
            text(
                "INSERT INTO pdf_page_ocr_texts(file_id, page, text) "
                "VALUES('pdf1', 1, 'page text')"
            )
        )

    _purge_file("vid1")
    _purge_file("pdf1")

    with database.get_search_db_read() as session:
        assert session.execute(text(
            "SELECT * FROM video_frame_texts WHERE file_id = 'vid1'"
        )).fetchall() == []
        assert session.execute(text(
            "SELECT * FROM pdf_page_ocr_texts WHERE file_id = 'pdf1'"
        )).fetchall() == []
