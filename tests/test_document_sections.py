"""Schema and helpers for ``document_sections`` (EPUB section titles)."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from app.database import Base, _create_document_sections_table
from app.document_sections import (
    EPUB_MIME,
    load_section_titles,
    replace_document_sections,
    section_title,
)
from app.models import DocumentSection


def _enable_fks(dbapi_conn: sqlite3.Connection, _: object) -> None:
    dbapi_conn.execute("PRAGMA foreign_keys=ON")


@pytest.fixture()
def engine(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    event.listen(engine, "connect", _enable_fks)
    # The order ``init_search_db`` runs: the ORM creates the table, then the
    # migration's ``IF NOT EXISTS`` finds it already there.
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _create_document_sections_table(conn)
    return engine


@pytest.fixture()
def session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture()
def patched_db(session_factory, monkeypatch):
    @contextmanager
    def _get_search_db():
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("app.document_sections.get_search_db_read", _get_search_db)
    return _get_search_db


def _insert_indexed_file(engine, file_id: str, mime: str = EPUB_MIME) -> None:
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO indexed_files "
                "(file_id, drive, filename, file_path, file_type, mime_type, "
                " file_size, active, text_indexed, metadata_indexed, "
                " title, description, tags_text, indexed_at, updated_at) "
                "VALUES (:fid, 'd1', :fid, :fid, 'document', :mime, 100, "
                " 1, 1, 1, '', '', '', :ts, :ts)"
            ),
            {"fid": file_id, "mime": mime, "ts": now},
        )


def _rows(engine, file_id: str) -> list[tuple[int, str]]:
    with engine.connect() as conn:
        return [
            (int(r[0]), r[1])
            for r in conn.execute(
                text(
                    "SELECT page, title FROM document_sections "
                    "WHERE file_id = :fid ORDER BY page"
                ),
                {"fid": file_id},
            ).fetchall()
        ]


def test_fresh_schema_columns(engine) -> None:
    with engine.connect() as conn:
        cols = [
            (r[1], r[2].upper(), bool(r[3]))
            for r in conn.execute(text("PRAGMA table_info(document_sections)"))
        ]

    assert cols == [
        ("file_id", "VARCHAR(12)", True),
        ("page", "INTEGER", True),
        ("title", "TEXT", True),
    ]


def test_primary_key_is_file_id_page(engine) -> None:
    with engine.connect() as conn:
        pk = sorted(
            (r[5], r[1])
            for r in conn.execute(text("PRAGMA table_info(document_sections)"))
            if r[5]
        )

    assert pk == [(1, "file_id"), (2, "page")]


def test_foreign_key_cascades_to_indexed_files(engine) -> None:
    with engine.connect() as conn:
        fks = [
            tuple(r)
            for r in conn.execute(text("PRAGMA foreign_key_list(document_sections)"))
        ]

    assert len(fks) == 1
    assert (fks[0][2], fks[0][3], fks[0][4], fks[0][6]) == (
        "indexed_files", "file_id", "file_id", "CASCADE",
    )


def test_create_is_idempotent(engine) -> None:
    _insert_indexed_file(engine, "book1")
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO document_sections (file_id, page, title) "
            "VALUES ('book1', 1, 'Opening')"
        ))

    with engine.begin() as conn:
        _create_document_sections_table(conn)

    assert _rows(engine, "book1") == [(1, "Opening")]


def test_replace_deletes_previous_rows(engine, session_factory) -> None:
    _insert_indexed_file(engine, "book1")
    _insert_indexed_file(engine, "book2")
    with session_factory() as session:
        replace_document_sections(session, "book1", ((1, "Old one"), (4, "Old four")))
        replace_document_sections(session, "book2", ((2, "Other book"),))
        session.commit()

    with session_factory() as session:
        replace_document_sections(session, "book1", ((2, "New two"),))
        session.commit()

    assert _rows(engine, "book1") == [(2, "New two")]
    assert _rows(engine, "book2") == [(2, "Other book")]


def test_replace_with_no_titles_deletes_all(engine, session_factory) -> None:
    _insert_indexed_file(engine, "book1")
    with session_factory() as session:
        replace_document_sections(session, "book1", ((1, "One"), (2, "Two")))
        session.commit()

    with session_factory() as session:
        replace_document_sections(session, "book1", ())
        session.commit()

    assert _rows(engine, "book1") == []


def test_orm_model_reads_rows(engine, session_factory) -> None:
    _insert_indexed_file(engine, "book1")
    with session_factory() as session:
        replace_document_sections(session, "book1", ((3, "Three"),))
        session.commit()

    with session_factory() as session:
        row = session.get(DocumentSection, ("book1", 3))

    assert (row.file_id, row.page, row.title) == ("book1", 3, "Three")


def test_purge_file_cascades_sections(engine, session_factory, monkeypatch) -> None:
    with engine.begin() as conn:
        from app.database import (
            _create_detailed_summary_citations_table,
            _create_file_summaries_table,
            _create_suggested_chapters_table,
            _create_suggested_tags_table,
        )

        _create_file_summaries_table(conn)
        _create_detailed_summary_citations_table(conn)
        _create_suggested_tags_table(conn)
        _create_suggested_chapters_table(conn)
        for name, cols, tokenize in (
            ("fts_files", "file_id, filename, title, description, tags_text", "trigram"),
            ("fts_transcripts", "file_id, chunk_index, text", "trigram"),
            ("fts_text_content", "file_id, chunk_index, page, text", "trigram"),
            ("fts_files_word", "file_id, filename, title, description, tags_text",
             "unicode61 remove_diacritics 2"),
            ("fts_transcripts_word", "file_id, chunk_index, text",
             "unicode61 remove_diacritics 2"),
            ("fts_text_content_word", "file_id, chunk_index, page, text",
             "unicode61 remove_diacritics 2"),
        ):
            conn.execute(text(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} "
                f"USING fts5({cols}, tokenize=\"{tokenize}\")"
            ))
    _insert_indexed_file(engine, "book1")
    _insert_indexed_file(engine, "book2")
    with session_factory() as session:
        replace_document_sections(session, "book1", ((1, "One"), (2, "Two")))
        replace_document_sections(session, "book2", ((1, "Kept"),))
        session.commit()

    @contextmanager
    def _get_search_db():
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    monkeypatch.setattr("app.indexer.get_search_db", _get_search_db)
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)

    from app.indexer import _purge_file

    _purge_file("book1")

    assert _rows(engine, "book1") == []
    assert _rows(engine, "book2") == [(1, "Kept")]


def test_load_section_titles_filters_pages(engine, session_factory) -> None:
    _insert_indexed_file(engine, "book1")
    _insert_indexed_file(engine, "book2")
    with session_factory() as session:
        replace_document_sections(
            session, "book1", ((1, "One"), (2, "Two"), (3, "Three")),
        )
        replace_document_sections(session, "book2", ((2, "Other two"),))
        session.commit()

    with session_factory() as session:
        titles = load_section_titles(
            session, [("book1", 3), ("book1", 1), ("book2", 2), ("book2", 9)],
        )

    assert titles == {
        ("book1", 1): "One",
        ("book1", 3): "Three",
        ("book2", 2): "Other two",
    }


def test_load_section_titles_with_no_pairs_is_empty(session_factory) -> None:
    with session_factory() as session:
        assert load_section_titles(session, []) == {}


def test_section_title_reads_one_row(engine, session_factory, patched_db) -> None:
    _insert_indexed_file(engine, "book1")
    with session_factory() as session:
        replace_document_sections(session, "book1", ((2, "Two"),))
        session.commit()

    assert section_title("book1", 2) == "Two"
    assert section_title("book1", 3) is None
    assert section_title("book1", None) is None
