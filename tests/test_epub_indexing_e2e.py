"""EPUB through ``index_text_content`` into chunks, FTS and ``document_sections``."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

import app.config as config
from app.database import Base, _create_document_sections_table, _create_pdf_markdown_table
from app.models import IndexedFile
from app.workers import metadata as metadata_worker
from tests.epub_fixtures import Item, make_epub, nav_doc, xhtml


def _enable_fks(dbapi_conn: sqlite3.Connection, _: object) -> None:
    dbapi_conn.execute("PRAGMA foreign_keys=ON")


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    event.listen(engine, "connect", _enable_fks)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _create_pdf_markdown_table(conn)
        _create_document_sections_table(conn)
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS vec_text ("
            "  embedding_id TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content "
            "USING fts5(file_id, chunk_index, page, text, tokenize='trigram')"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content_word "
            "USING fts5(file_id, chunk_index, page, text, "
            "tokenize=\"unicode61 remove_diacritics 2\")"
        ))

    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _get_search_db():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("app.workers.metadata.get_search_db", _get_search_db)
    monkeypatch.setattr(config, "validate_file_path", lambda _path: True)
    return engine


@pytest.fixture(autouse=True)
def stub_embed(monkeypatch):
    import numpy as np

    monkeypatch.setattr(
        metadata_worker,
        "embed_passages",
        lambda passages: [np.zeros(4, dtype=np.float32) for _ in passages],
    )


def _seed(engine, file_id: str, path: Path, mime: str = "application/epub+zip") -> None:
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as session:
        session.add(IndexedFile(
            file_id=file_id,
            drive="drive1",
            filename=path.name,
            file_path=str(path),
            file_type="document",
            mime_type=mime,
            file_size=path.stat().st_size,
            active=True,
            text_indexed=False,
        ))
        session.commit()


def _book(path: Path, titles: list[tuple[str, str]], bodies: dict[str, str]) -> Path:
    return make_epub(
        path,
        [
            Item("nav", "nav.xhtml", nav_doc(titles), properties="nav"),
            *(Item(name, f"{name}.xhtml", xhtml(body)) for name, body in bodies.items()),
        ],
        spine=["ghost", *bodies],
    )


def _sections(engine, file_id: str) -> list[tuple[int, str]]:
    with engine.connect() as conn:
        return [
            (int(r[0]), r[1]) for r in conn.execute(
                text("SELECT page, title FROM document_sections "
                     "WHERE file_id = :f ORDER BY page"),
                {"f": file_id},
            )
        ]


def _fts(engine, file_id: str) -> list[tuple[int, int | None, str]]:
    with engine.connect() as conn:
        return [
            (int(r[0]), int(r[1]) if r[1] not in (None, "") else None, r[2])
            for r in conn.execute(
                text("SELECT chunk_index, page, text FROM fts_text_content "
                     "WHERE file_id = :f ORDER BY CAST(chunk_index AS INTEGER)"),
                {"f": file_id},
            )
        ]


def _embedding_pages(engine, file_id: str) -> list[tuple[int, int]]:
    with engine.connect() as conn:
        return [
            (int(r[0]), int(r[1])) for r in conn.execute(
                text("SELECT chunk_index, page FROM embeddings "
                     "WHERE file_id = :f AND embedding_type = 'text_content' "
                     "ORDER BY chunk_index"),
                {"f": file_id},
            )
        ]


def test_chunk_page_and_row_use_section_number(search_db, tmp_path) -> None:
    path = _book(
        tmp_path / "b.epub",
        [("c2.xhtml", "Second")],
        {"c1": "<p>first body</p>", "c2": "<p>second body</p>"},
    )
    _seed(search_db, "book", path)

    assert metadata_worker.index_text_content("book") is True

    assert _fts(search_db, "book") == [(0, 1, "first body"), (1, 2, "second body")]
    assert _embedding_pages(search_db, "book") == [(0, 1), (1, 2)]
    assert _sections(search_db, "book") == [(2, "Second")]


def test_indexed_chunks_contain_no_ruby_readings(search_db, tmp_path) -> None:
    path = _book(
        tmp_path / "b.epub",
        [],
        {"c1": "<p><ruby>吾輩<rp>(</rp><rt>わがはい</rt><rp>)</rp></ruby>は猫である</p>"},
    )
    _seed(search_db, "book", path)

    metadata_worker.index_text_content("book")

    assert _fts(search_db, "book") == [(0, 1, "吾輩は猫である")]


def test_reindex_replaces_rows(search_db, tmp_path) -> None:
    path = tmp_path / "b.epub"
    _book(path, [("c1.xhtml", "Old one"), ("c2.xhtml", "Old two")],
          {"c1": "<p>a</p>", "c2": "<p>b</p>"})
    _seed(search_db, "book", path)
    metadata_worker.index_text_content("book")

    _book(path, [("c2.xhtml", "New two")], {"c1": "<p>a</p>", "c2": "<p>b</p>"})
    metadata_worker.index_text_content("book")

    assert _sections(search_db, "book") == [(2, "New two")]


def test_reindex_to_zero_chunks_deletes_rows(search_db, tmp_path) -> None:
    path = tmp_path / "b.epub"
    _book(path, [("c1.xhtml", "One")], {"c1": "<p>a</p>"})
    _seed(search_db, "book", path)
    metadata_worker.index_text_content("book")
    assert _sections(search_db, "book") == [(1, "One")]

    path.write_bytes(b"not a zip any more")
    assert metadata_worker.index_text_content("book") is True

    assert _sections(search_db, "book") == []


def test_non_epub_reindex_writes_no_rows(search_db, tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Heading\n\nSome notes about chapters.", encoding="utf-8")
    _seed(search_db, "notes", path, mime="text/markdown")

    assert metadata_worker.index_text_content("notes") is True

    assert _fts(search_db, "notes") != []
    with search_db.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM document_sections")).scalar() == 0


def test_failed_epub_does_not_block_next_file(search_db, tmp_path) -> None:
    broken = tmp_path / "broken.epub"
    broken.write_bytes(b"PK\x03\x04 truncated")
    good = _book(tmp_path / "good.epub", [("c1.xhtml", "One")], {"c1": "<p>fine</p>"})
    _seed(search_db, "broken", broken)
    _seed(search_db, "good", good)

    assert metadata_worker.index_text_content("broken") is True
    assert metadata_worker.index_text_content("good") is True

    assert _fts(search_db, "broken") == []
    assert _fts(search_db, "good") == [(0, 1, "fine")]
    with search_db.connect() as conn:
        flags = dict(conn.execute(text("SELECT file_id, text_indexed FROM indexed_files")).all())
    assert flags == {"broken": 1, "good": 1}


def test_epub_mime_is_text_indexed() -> None:
    from app.indexer import TEXT_MIMES

    assert "application/epub+zip" in TEXT_MIMES
