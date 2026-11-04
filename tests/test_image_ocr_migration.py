"""The ocr_indexed column and its reset by a text-model change (SPEC-ADDON-050)."""

from __future__ import annotations

import dataclasses

import pytest
from sqlalchemy import create_engine, text

from app.config import ModelConfig
from app.models import IndexedFile  # noqa: F401  - registers the tables on Base

_GRANITE_97 = "ibm-granite/granite-embedding-97m-multilingual-r2"  # 384
_RURI_310 = "cl-nagoya/ruri-v3-310m"  # 1024


def _build_skeleton(tmp_path, *, vec_text_dim: int, recorded_model: str):
    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE indexed_files ("
            "  file_id TEXT PRIMARY KEY,"
            "  drive TEXT NOT NULL,"
            "  mime_type TEXT NOT NULL,"
            "  text_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  whisper_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  tfidf_keywords_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  ocr_indexed BOOLEAN NOT NULL DEFAULT 0"
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
        conn.execute(
            text("INSERT INTO index_meta(key, value) VALUES('text_embedding_model', :m)"),
            {"m": recorded_model},
        )
    return engine


def _patch_text_model(monkeypatch, model_id: str) -> None:
    import app.config as config

    settings = dataclasses.replace(
        config.settings, models=ModelConfig(text_embedding=model_id)
    )
    monkeypatch.setattr(config, "settings", settings)


def _run_text_migration(engine) -> None:
    from app.database import _migrate_vec_text_if_needed

    with engine.connect() as conn:
        _migrate_vec_text_if_needed(conn)
        conn.commit()


def test_a_text_model_change_resets_ocr_only_for_pictures(
    tmp_path, monkeypatch
) -> None:
    """SPEC-ADDON-050 I8: the picture's text is read again; a document's is not."""
    engine = _build_skeleton(tmp_path, vec_text_dim=384, recorded_model=_GRANITE_97)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO indexed_files(file_id, drive, mime_type, ocr_indexed) "
            "VALUES('pic1', 'drive1', 'image/jpeg', 1)"
        ))
        conn.execute(text(
            "INSERT INTO indexed_files(file_id, drive, mime_type, ocr_indexed) "
            "VALUES('doc1', 'drive1', 'application/pdf', 1)"
        ))
    _patch_text_model(monkeypatch, _RURI_310)

    _run_text_migration(engine)

    with engine.connect() as conn:
        rows = dict(conn.execute(
            text("SELECT file_id, ocr_indexed FROM indexed_files")
        ).fetchall())
    assert rows == {"pic1": 0, "doc1": 1}


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


def test_the_ocr_column_migration_can_run_twice(init_database) -> None:
    """SPEC-ADDON-050 I8: adding the column again leaves existing rows alone."""
    start, path = init_database
    engine = create_engine(f"sqlite:///{path}")

    from app.database import Base

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        if "ocr_indexed" in {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(indexed_files)")).fetchall()
        }:
            # Turns a database this build created back into one that predates
            # the column, which is the state the migration must upgrade.
            conn.execute(text("ALTER TABLE indexed_files DROP COLUMN ocr_indexed"))
        conn.execute(text(
            "INSERT INTO indexed_files(file_id, drive, filename, file_path, "
            "file_type, mime_type, file_size, active, title, description, "
            "tags_text, indexed_at, updated_at) "
            "VALUES('pic1', 'drive1', 'pic.png', '/drives/drive1/pic.png', "
            "'image', 'image/png', 10, 1, '', '', '', "
            "'2026-11-03 10:00:00', '2026-11-03 10:00:00')"
        ))
        assert "ocr_indexed" not in {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(indexed_files)")).fetchall()
        }
    engine.dispose()

    start()

    reader = create_engine(f"sqlite:///{path}")
    with reader.connect() as conn:
        columns = [
            row[1]
            for row in conn.execute(text("PRAGMA table_info(indexed_files)")).fetchall()
        ]
        assert columns.count("ocr_indexed") == 1
        assert conn.execute(
            text("SELECT ocr_indexed FROM indexed_files WHERE file_id = 'pic1'")
        ).fetchone()[0] == 0
    with reader.begin() as conn:
        conn.execute(text("UPDATE indexed_files SET ocr_indexed = 1"))
    reader.dispose()

    start()

    reader = create_engine(f"sqlite:///{path}")
    with reader.connect() as conn:
        columns = [
            row[1]
            for row in conn.execute(text("PRAGMA table_info(indexed_files)")).fetchall()
        ]
        assert columns.count("ocr_indexed") == 1
        assert conn.execute(
            text("SELECT ocr_indexed FROM indexed_files WHERE file_id = 'pic1'")
        ).fetchone()[0] == 1
    reader.dispose()
