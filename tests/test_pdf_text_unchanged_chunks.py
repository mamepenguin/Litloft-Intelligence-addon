"""SPEC-ADDON-060 I7 and I9: what indexing writes when the chunks did not or did change."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import app.config as config
from app.database import (
    Base,
    _create_detailed_summary_citations_table,
    _create_pdf_markdown_table,
)
from app.models import FileInsight, IndexedFile
from app.workers import metadata as metadata_worker
from tests.document_fixtures import make_docx, make_pdf_lines, make_pptx, make_xlsx
from tests.epub_fixtures import Item, make_epub, xhtml

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
HTML = "text/html"
MARKDOWN = "text/markdown"
EPUB = "application/epub+zip"
PNG = "image/png"

CITATIONS_READY = "intelligence.detailed_summary.citations_ready"


class TextIndex:
    def __init__(self, engine, session_factory):
        self.engine = engine
        self.Session = session_factory

    def add_file(self, file_id: str, path: Path, mime: str = PDF) -> None:
        with self.Session() as session:
            session.add(IndexedFile(
                file_id=file_id,
                drive="drive1",
                filename=path.name,
                file_path=str(path),
                file_type="document",
                mime_type=mime,
                file_size=path.stat().st_size if path.exists() else 1,
                active=True,
            ))
            session.commit()

    def sql(self, statement: str, **params) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(statement), params)

    def rows(self, statement: str, **params) -> list[tuple]:
        with self.engine.connect() as conn:
            return [tuple(row) for row in conn.execute(text(statement), params)]

    def scalar(self, statement: str, **params):
        rows = self.rows(statement, **params)
        return rows[0][0] if rows else None


def _contexts(engine):
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    from contextlib import contextmanager

    @contextmanager
    def write():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def read():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    return write, read


@pytest.fixture()
def index(tmp_path, monkeypatch) -> TextIndex:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _create_pdf_markdown_table(conn)
        _create_detailed_summary_citations_table(conn)
        conn.execute(text(
            "CREATE TABLE vec_text ("
            "  embedding_id TEXT PRIMARY KEY, embedding_type TEXT, "
            "  vector BLOB NOT NULL)"
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
    write, read = _contexts(engine)
    monkeypatch.setattr(metadata_worker, "get_search_db", write)
    monkeypatch.setattr(metadata_worker, "get_search_db_read", read, raising=False)
    monkeypatch.setattr(config, "validate_file_path", lambda _path: True)
    return TextIndex(engine, sessionmaker(bind=engine, expire_on_commit=False))


@pytest.fixture()
def embeddings(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def embed(passages: list[str]):
        calls.append(list(passages))
        return [np.zeros(4, dtype=np.float32) for _ in passages]

    monkeypatch.setattr(metadata_worker, "embed_passages", embed)
    return calls


@pytest.fixture()
def posted(monkeypatch) -> list[tuple[str, dict, str | None]]:
    events: list[tuple[str, dict, str | None]] = []

    async def async_post(event, data, *args, **kwargs):
        events.append((event, data, kwargs.get("drive") or (args[0] if args else None)))
        return True

    def sync_post(event, data, *args, **kwargs):
        events.append((event, data, kwargs.get("drive") or (args[0] if args else None)))
        return True

    monkeypatch.setattr("app.addon_events.post_addon_event", async_post, raising=False)
    monkeypatch.setattr(
        "app.addon_events.post_addon_event_sync", sync_post, raising=False
    )
    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("app."):
            continue
        if "post_addon_event" in vars(module):
            monkeypatch.setattr(module, "post_addon_event", async_post)
        if "post_addon_event_sync" in vars(module):
            monkeypatch.setattr(module, "post_addon_event_sync", sync_post)
    return events


class Recompute:
    def __init__(self, engine):
        self.engine = engine
        self.calls: list[tuple[str, str]] = []
        self.seen: list[list[str]] = []
        self.fail = False

    def __call__(self, file_id: str, summary: str) -> tuple[int, int]:
        self.calls.append((file_id, summary))
        with self.engine.connect() as conn:
            self.seen.append([
                row[0] for row in conn.execute(
                    text("SELECT text FROM fts_text_content WHERE file_id = :f"),
                    {"f": file_id},
                )
            ])
        if self.fail:
            raise RuntimeError("simulated citation failure")
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE detailed_summary_citations "
                    "SET segment_text = 'RECOMPUTED' WHERE file_id = :f"
                ),
                {"f": file_id},
            )
        return (1, 0)


def _install_recompute(monkeypatch, recompute: Recompute) -> None:
    monkeypatch.setattr("app.citations.calculate_and_store", recompute, raising=False)
    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("app."):
            continue
        if "calculate_and_store" in vars(module):
            monkeypatch.setattr(module, "calculate_and_store", recompute)


def _seed_summary(index: TextIndex, file_id: str, content: str) -> None:
    with index.Session() as session:
        session.add(FileInsight(
            id="insight1",
            file_id=file_id,
            kind="detailed_summary",
            content=content,
            status="active",
            created_by="intelligence",
        ))
        session.commit()
    index.sql(
        "INSERT INTO detailed_summary_citations "
        "(file_id, section_path, segment_type, segment_text, citation_chunk_ids, "
        " top_score, has_citation) "
        "VALUES (:f, 'section/0', 'paragraph', 'ORIGINAL', '[\"document:0\"]', "
        " 0.9, 1)",
        f=file_id,
    )


def _write_pdf(path: Path, marker: str, pages: int = 2) -> Path:
    return make_pdf_lines(path, [
        [(f"{marker} body of page {number}.", 72, 72)]
        for number in range(1, pages + 1)
    ])


def _embedding_rows(index: TextIndex, file_id: str) -> list[tuple]:
    return index.rows(
        "SELECT id, chunk_index, content_preview FROM embeddings "
        "WHERE file_id = :f AND embedding_type = 'text_content' "
        "ORDER BY chunk_index",
        f=file_id,
    )


def _fts_rows(index: TextIndex, file_id: str) -> list[tuple]:
    return index.rows(
        "SELECT CAST(chunk_index AS INTEGER), page, text FROM fts_text_content "
        "WHERE file_id = :f ORDER BY CAST(chunk_index AS INTEGER)",
        f=file_id,
    )


def _vector_ids(index: TextIndex) -> list[tuple]:
    return index.rows("SELECT embedding_id FROM vec_text ORDER BY embedding_id")


def test_spec_addon_060_i7_unchanged_chunks_keep_their_ids_and_skip_the_embedder(
    index, embeddings, tmp_path,
):
    path = _write_pdf(tmp_path / "doc.pdf", "SAME")
    index.add_file("f1", path)

    assert metadata_worker.index_text_content("f1") is True
    first_embeddings = _embedding_rows(index, "f1")
    first_fts = _fts_rows(index, "f1")
    first_vectors = _vector_ids(index)
    assert len(embeddings) == 1
    index.sql("UPDATE pdf_markdown SET markdown = 'STALE' WHERE file_id = 'f1'")

    assert metadata_worker.index_text_content("f1") is True

    assert len(embeddings) == 1
    assert _embedding_rows(index, "f1") == first_embeddings
    assert _fts_rows(index, "f1") == first_fts
    assert _vector_ids(index) == first_vectors
    assert index.scalar(
        "SELECT text_indexed FROM indexed_files WHERE file_id = 'f1'"
    ) == 1
    markdown = index.scalar(
        "SELECT markdown FROM pdf_markdown WHERE file_id = 'f1'"
    )
    assert markdown != "STALE" and "SAME body of page 1." in markdown


def test_spec_addon_060_i7_rewritten_chunks_recompute_citations_and_announce_them(
    index, embeddings, posted, monkeypatch, tmp_path,
):
    path = _write_pdf(tmp_path / "doc.pdf", "OLDBODY")
    index.add_file("f1", path)
    assert metadata_worker.index_text_content("f1") is True
    _seed_summary(index, "f1", "ACTIVE SUMMARY")
    recompute = Recompute(index.engine)
    _install_recompute(monkeypatch, recompute)
    _write_pdf(path, "NEWBODY")

    assert metadata_worker.index_text_content("f1") is True

    assert len(embeddings) == 2
    texts = [row[2] for row in _fts_rows(index, "f1")]
    assert any("NEWBODY" in text for text in texts)
    assert not any("OLDBODY" in text for text in texts)
    assert recompute.calls == [("f1", "ACTIVE SUMMARY")]
    assert any("NEWBODY" in seen for seen in recompute.seen)
    assert index.scalar(
        "SELECT segment_text FROM detailed_summary_citations WHERE file_id = 'f1'"
    ) == "RECOMPUTED"
    ready = [
        data for event, data, _drive in posted if event == CITATIONS_READY
    ]
    assert len(ready) == 1
    assert ready[0]["file_id"] == "f1"
    assert ready[0]["drive"] == "drive1"
    assert (ready[0]["citation_count"], ready[0]["no_citation_count"]) == (1, 0)


def test_spec_addon_060_i7_a_failed_recompute_keeps_the_new_chunks_and_old_citations(
    index, embeddings, posted, monkeypatch, tmp_path, caplog,
):
    path = _write_pdf(tmp_path / "doc.pdf", "OLDBODY")
    index.add_file("f1", path)
    assert metadata_worker.index_text_content("f1") is True
    _seed_summary(index, "f1", "ACTIVE SUMMARY")
    recompute = Recompute(index.engine)
    recompute.fail = True
    _install_recompute(monkeypatch, recompute)
    _write_pdf(path, "NEWBODY")

    with caplog.at_level("WARNING"):
        assert metadata_worker.index_text_content("f1") is True

    texts = [row[2] for row in _fts_rows(index, "f1")]
    assert any("NEWBODY" in text for text in texts)
    assert len(embeddings) == 2
    assert index.scalar(
        "SELECT segment_text FROM detailed_summary_citations WHERE file_id = 'f1'"
    ) == "ORIGINAL"
    assert not [event for event, _data, _drive in posted if event == CITATIONS_READY]
    assert any(
        record.levelno >= 30 and "f1" in record.getMessage()
        for record in caplog.records
    )


def _html(path: Path) -> Path:
    path.write_text(
        "<html><body><h1>Heading</h1><p>MARK body text.</p></body></html>",
        encoding="utf-8",
    )
    return path


def _text(path: Path) -> Path:
    path.write_text("MARK alpha beta\n", encoding="utf-8")
    return path


def _epub(path: Path) -> Path:
    return make_epub(path, [Item("c1", "c1.xhtml", xhtml("<p>MARK section one</p>"))])


@pytest.mark.parametrize(
    "suffix,mime,build,expected",
    [
        pytest.param(".md", MARKDOWN, _text, "MARK alpha beta", id="text"),
        pytest.param(
            ".docx", DOCX,
            lambda path: make_docx(
                path, heading="Chapter", paragraphs=("MARK body.",),
            ),
            "# Chapter\nMARK body.",
            id="docx",
        ),
        pytest.param(".html", HTML, _html, "MARK body text.", id="html"),
        pytest.param(".epub", EPUB, _epub, "MARK section one", id="epub"),
        pytest.param(
            ".xlsx", XLSX,
            lambda path: make_xlsx(path, [("Data", [["MARK alpha", "beta"]])]),
            "## Sheet: Data\n\nMARK alpha beta",
            id="xlsx",
        ),
        pytest.param(
            ".pptx", PPTX,
            lambda path: make_pptx(path, [("Intro", ["MARK first"])]),
            "## Slide 1: Intro\n\nIntro\nMARK first",
            id="pptx",
        ),
    ],
)
def test_spec_addon_060_i9_other_types_keep_their_chunks_and_are_rewritten(
    index, embeddings, posted, monkeypatch, tmp_path, suffix, mime, build, expected,
):
    path = build(tmp_path / f"doc{suffix}")
    index.add_file("f1", path, mime)
    _seed_summary(index, "f1", "ACTIVE SUMMARY")
    recompute = Recompute(index.engine)
    _install_recompute(monkeypatch, recompute)

    assert metadata_worker.index_text_content("f1") is True
    first = _fts_rows(index, "f1")
    assert metadata_worker.index_text_content("f1") is True

    assert [row[2] for row in first] == [expected]
    assert _fts_rows(index, "f1") == first
    assert len(embeddings) == 2
    assert index.scalar(
        "SELECT COUNT(*) FROM pdf_markdown WHERE file_id = 'f1'"
    ) == 0
    assert recompute.calls == []
    assert not [event for event, _data, _drive in posted if event == CITATIONS_READY]
    assert index.scalar(
        "SELECT segment_text FROM detailed_summary_citations WHERE file_id = 'f1'"
    ) == "ORIGINAL"


def test_spec_addon_060_i9_a_picture_keeps_its_ocr_rows(index, embeddings, tmp_path):
    path = tmp_path / "picture.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    index.add_file("pic", path, PNG)
    index.sql(
        "INSERT INTO embeddings "
        "(id, file_id, embedding_type, vector_table, content_preview, chunk_index, "
        " created_at) "
        "VALUES ('ocr1', 'pic', 'text_content', 'vec_text', 'OCR EARLIER', 0, "
        " CURRENT_TIMESTAMP)"
    )
    index.sql(
        "INSERT INTO fts_text_content(file_id, chunk_index, page, text) "
        "VALUES ('pic', '0', '1', 'OCR EARLIER')"
    )
    embeddings_before = _embedding_rows(index, "pic")
    fts_before = _fts_rows(index, "pic")

    assert metadata_worker.index_text_content("pic") is True

    assert _embedding_rows(index, "pic") == embeddings_before
    assert _fts_rows(index, "pic") == fts_before
    assert embeddings == []
