"""End-to-end indexing pipeline coverage for the PDF Markdown rollout.

Phase 3 of ``2026-04-27-intelligence-pdf-markdown-indexing.md`` wires
the PyMuPDF4LLM Markdown rendering through ``index_text_content`` into
the ``pdf_markdown`` table. These tests drive the worker against a real
SQLite database with the relevant search-DB tables in place, then
assert the row landed (or didn't, on the fallback / size-cap paths).

The vec0 / vec_clip virtual tables are heavy and unrelated to this
phase, so we substitute plain tables with the same names — every
production code path uses parameterised SQL through ``validate_vector_table``,
so a regular table satisfies the INSERT/DELETE statements without
pulling sqlite-vec into the test fixture.

``embed_passages`` and ``pymupdf4llm.to_markdown`` are mocked so no
actual model load or PDF parsing happens, matching the patterns in
``test_pdf_extractor.py`` and ``test_detailed_citations.py``.
"""

from __future__ import annotations

import sys
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Match the ML-deps stubbing other tests rely on. ``conftest.py`` does
# the same at collection time but a few modules need to be re-pinned
# here so this file is self-contained when run in isolation.
for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine, event, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import app.config as config  # noqa: E402
from app.database import (  # noqa: E402
    Base,
    _create_pdf_markdown_table,
)
from app.models import IndexedFile  # noqa: E402,F401
from app.workers import metadata as metadata_worker  # noqa: E402


def _enable_fks(dbapi_conn: sqlite3.Connection, _: object) -> None:
    """Match the runtime listener so FK CASCADE actually fires in tests."""
    dbapi_conn.execute("PRAGMA foreign_keys=ON")


def _create_aux_tables(conn: object) -> None:
    """Create the minimum set of aux tables ``index_text_content`` writes to.

    We sidestep sqlite-vec by creating a plain ``vec_text`` table whose
    schema accepts the (embedding_id, vector) tuples production code
    inserts. The fields aren't queried in these tests; the goal is to
    let every INSERT/DELETE in the production path complete cleanly.
    """
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS vec_text ("
        "  embedding_id TEXT PRIMARY KEY,"
        "  vector BLOB NOT NULL"
        ")"
    ))
    conn.execute(text(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content "
        "USING fts5(file_id, chunk_index, page, text, tokenize='trigram')"
    ))
    # Phase 3 dual-index parallel table — production upsert/delete
    # helpers write to both tokenizations, so any test that exercises
    # those helpers must create the word table too or DELETE statements
    # explode on a missing-table error.
    conn.execute(text(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content_word "
        "USING fts5(file_id, chunk_index, page, text, "
        "tokenize=\"unicode61 remove_diacritics 2\")"
    ))


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    """Real SQLite wired into ``app.workers.metadata.get_search_db``."""
    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    event.listen(engine, "connect", _enable_fks)

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _create_pdf_markdown_table(conn)
        _create_aux_tables(conn)

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

    monkeypatch.setattr(
        "app.workers.metadata.get_search_db", _get_search_db
    )
    monkeypatch.setattr(config, "validate_file_path", lambda _path: True)
    return engine


@pytest.fixture()
def fake_pdf(tmp_path: Path) -> Path:
    """Non-empty file with a ``.pdf`` extension; the extractor only stats it."""
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4\n%fake\n")
    return path


def _seed_indexed_pdf(engine, *, file_id: str, file_path: Path) -> None:
    """Insert the ``indexed_files`` row the worker reads up front."""
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()
    try:
        session.add(
            IndexedFile(
                file_id=file_id,
                drive="drive1",
                filename=file_path.name,
                file_path=str(file_path),
                file_type="document",
                mime_type="application/pdf",
                file_size=file_path.stat().st_size,
                active=True,
                text_indexed=False,
            )
        )
        session.commit()
    finally:
        session.close()


def _stub_pymupdf4llm(
    monkeypatch: pytest.MonkeyPatch,
    page_chunks: list[dict] | Exception,
) -> MagicMock:
    """Install a fake ``pymupdf4llm`` module returning the given pages.

    Pass an Exception instance to make ``to_markdown`` raise (exercises
    the fitz fallback).
    """
    if isinstance(page_chunks, Exception):
        def _raise(*_a, **_kw):
            raise page_chunks
        to_markdown = _raise
    else:
        to_markdown = MagicMock(return_value=page_chunks)

    fake = MagicMock()
    fake.to_markdown = to_markdown
    monkeypatch.setitem(sys.modules, "pymupdf4llm", fake)
    return fake


def _stub_fitz(monkeypatch: pytest.MonkeyPatch, pages: list[str]) -> MagicMock:
    """Install a fake ``fitz`` module so the fallback path can run."""
    fake_pages = []
    for page_text in pages:
        page = MagicMock()
        page.get_text.return_value = page_text
        fake_pages = [*fake_pages, page]

    doc = MagicMock()
    doc.page_count = len(fake_pages)
    doc.__getitem__.side_effect = lambda i: fake_pages[i]
    doc.close = MagicMock()

    fake = MagicMock()
    fake.open.return_value = doc
    monkeypatch.setitem(sys.modules, "fitz", fake)
    return fake


def _stub_embed_passages(
    monkeypatch: pytest.MonkeyPatch,
) -> list[list[str]]:
    """Replace ``embed_passages`` with a deterministic numpy stub.

    Returns the captured argument list so tests can assert how many
    chunks were embedded.
    """
    import numpy as np

    captured: list[list[str]] = []

    def _fake_embed(passages: list[str]):
        captured.append(list(passages))
        return [np.zeros(4, dtype=np.float32) for _ in passages]

    monkeypatch.setattr(metadata_worker, "embed_passages", _fake_embed)
    return captured


def _pdf_markdown_row(engine, file_id: str) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT file_id, markdown, page_count, extractor, "
                "       generated_at, updated_at "
                "FROM pdf_markdown WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()
    if row is None:
        return None
    return {
        "file_id": row[0],
        "markdown": row[1],
        "page_count": row[2],
        "extractor": row[3],
        "generated_at": row[4],
        "updated_at": row[5],
    }


# ---------------------------------------------------------------------------
# Happy path: PyMuPDF4LLM Markdown lands in ``pdf_markdown``.
# ---------------------------------------------------------------------------


def test_index_pdf_persists_markdown_row(
    search_db, fake_pdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_pymupdf4llm(
        monkeypatch,
        [
            {"text": "# Title\n\nFirst page body."},
            {"text": "## Section\n\nSecond page body."},
        ],
    )
    embed_calls = _stub_embed_passages(monkeypatch)
    _seed_indexed_pdf(search_db, file_id="pdf-1", file_path=fake_pdf)

    ok = metadata_worker.index_text_content("pdf-1")

    assert ok is True
    row = _pdf_markdown_row(search_db, "pdf-1")
    assert row is not None
    assert row["extractor"] == "pymupdf4llm"
    assert row["page_count"] == 2
    assert "First page body." in row["markdown"]
    assert "Second page body." in row["markdown"]
    # First write: generated_at == updated_at (UPSERT only diverges on
    # subsequent re-indexes — see test_reindex_updates_pdf_markdown_row_in_place).
    assert row["generated_at"] == row["updated_at"]
    # text_indexed flipped + embeddings actually computed
    with search_db.connect() as conn:
        ti = conn.execute(text(
            "SELECT text_indexed FROM indexed_files WHERE file_id = 'pdf-1'"
        )).scalar()
    assert ti == 1
    assert len(embed_calls) == 1 and embed_calls[0]


def test_index_pdf_chunks_land_in_vec_text_and_fts(
    search_db, fake_pdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_pymupdf4llm(
        monkeypatch,
        [{"text": "Alpha content on page one."}],
    )
    _stub_embed_passages(monkeypatch)
    _seed_indexed_pdf(search_db, file_id="pdf-vec", file_path=fake_pdf)

    metadata_worker.index_text_content("pdf-vec")

    with search_db.connect() as conn:
        vec_count = conn.execute(text(
            "SELECT COUNT(*) FROM vec_text"
        )).scalar()
        fts_count = conn.execute(text(
            "SELECT COUNT(*) FROM fts_text_content WHERE file_id = 'pdf-vec'"
        )).scalar()
        emb_count = conn.execute(text(
            "SELECT COUNT(*) FROM embeddings "
            "WHERE file_id = 'pdf-vec' AND embedding_type = 'text_content'"
        )).scalar()
    assert vec_count >= 1
    assert fts_count >= 1
    assert emb_count >= 1


# ---------------------------------------------------------------------------
# Re-index → UPSERT bumps ``updated_at`` while keeping ``generated_at``.
# ---------------------------------------------------------------------------


def test_reindex_updates_pdf_markdown_row_in_place(
    search_db, fake_pdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _stub_pymupdf4llm(
        monkeypatch,
        [{"text": "Original body."}],
    )
    _stub_embed_passages(monkeypatch)
    _seed_indexed_pdf(search_db, file_id="pdf-up", file_path=fake_pdf)

    metadata_worker.index_text_content("pdf-up")
    first = _pdf_markdown_row(search_db, "pdf-up")
    assert first is not None
    assert "Original" in first["markdown"]
    original_generated = first["generated_at"]

    # Force a measurable timestamp delta and swap the body.
    fake.to_markdown.return_value = [{"text": "Revised body."}]
    fixed_now = datetime(2099, 1, 1, 0, 0, 0, tzinfo=UTC)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return fixed_now if tz is None else fixed_now.astimezone(tz)

    monkeypatch.setattr(metadata_worker, "datetime", _FixedDatetime)

    metadata_worker.index_text_content("pdf-up")
    second = _pdf_markdown_row(search_db, "pdf-up")
    assert second is not None
    assert "Revised" in second["markdown"]
    assert second["generated_at"] == original_generated
    assert second["updated_at"] != first["updated_at"]
    # Single row → UPSERT, not INSERT.
    with search_db.connect() as conn:
        rows = conn.execute(text(
            "SELECT COUNT(*) FROM pdf_markdown WHERE file_id = 'pdf-up'"
        )).scalar()
    assert rows == 1


# ---------------------------------------------------------------------------
# CASCADE: deleting indexed_files removes pdf_markdown.
# ---------------------------------------------------------------------------


def test_delete_indexed_file_cascades_to_pdf_markdown(
    search_db, fake_pdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_pymupdf4llm(monkeypatch, [{"text": "Content."}])
    _stub_embed_passages(monkeypatch)
    _seed_indexed_pdf(search_db, file_id="pdf-cas", file_path=fake_pdf)

    metadata_worker.index_text_content("pdf-cas")
    assert _pdf_markdown_row(search_db, "pdf-cas") is not None

    with search_db.begin() as conn:
        conn.execute(text(
            "DELETE FROM indexed_files WHERE file_id = 'pdf-cas'"
        ))

    assert _pdf_markdown_row(search_db, "pdf-cas") is None


# ---------------------------------------------------------------------------
# Fallback: extractor is fitz_fallback → markdown=None → no row inserted.
# ---------------------------------------------------------------------------


def test_fallback_path_does_not_persist_markdown_but_indexes_chunks(
    search_db, fake_pdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_pymupdf4llm(monkeypatch, RuntimeError("simulated parse failure"))
    _stub_fitz(monkeypatch, ["fallback page text content"])
    _stub_embed_passages(monkeypatch)
    _seed_indexed_pdf(search_db, file_id="pdf-fb", file_path=fake_pdf)

    ok = metadata_worker.index_text_content("pdf-fb")

    assert ok is True
    assert _pdf_markdown_row(search_db, "pdf-fb") is None
    with search_db.connect() as conn:
        ti = conn.execute(text(
            "SELECT text_indexed FROM indexed_files WHERE file_id = 'pdf-fb'"
        )).scalar()
        fts_count = conn.execute(text(
            "SELECT COUNT(*) FROM fts_text_content WHERE file_id = 'pdf-fb'"
        )).scalar()
    assert ti == 1
    assert fts_count >= 1


# ---------------------------------------------------------------------------
# Size cap: oversized Markdown skips the row, keeps chunks/embeddings.
# ---------------------------------------------------------------------------


def test_oversize_markdown_skips_row_and_logs_warning(
    search_db,
    fake_pdf,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # One page whose Markdown exceeds the 5MB cap. Use a single chunk
    # so we don't generate millions of embedding rows.
    big_body = "x" * (metadata_worker.MAX_PDF_MARKDOWN_BYTES + 1)
    _stub_pymupdf4llm(monkeypatch, [{"text": big_body}])
    _stub_embed_passages(monkeypatch)
    _seed_indexed_pdf(search_db, file_id="pdf-big", file_path=fake_pdf)

    with caplog.at_level("WARNING", logger=metadata_worker.logger.name):
        ok = metadata_worker.index_text_content("pdf-big")

    assert ok is True
    assert _pdf_markdown_row(search_db, "pdf-big") is None
    with search_db.connect() as conn:
        emb_count = conn.execute(text(
            "SELECT COUNT(*) FROM embeddings "
            "WHERE file_id = 'pdf-big' AND embedding_type = 'text_content'"
        )).scalar()
        fts_count = conn.execute(text(
            "SELECT COUNT(*) FROM fts_text_content WHERE file_id = 'pdf-big'"
        )).scalar()
    assert emb_count >= 1
    assert fts_count >= 1
    assert any(
        "exceeds" in record.message and "pdf-big" in record.message
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Empty extraction: re-index with no chunks clears stale Markdown row.
# ---------------------------------------------------------------------------


def test_reindex_empty_extraction_clears_stale_pdf_markdown_row(
    search_db, fake_pdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-index that produces zero pages must drop the prior row.

    Otherwise the API would surface stale Markdown for a PDF whose
    extractable content is now gone (e.g. content swapped for a scan,
    or the file was replaced under the same path).
    """
    fake = _stub_pymupdf4llm(
        monkeypatch,
        [{"text": "Initial body."}],
    )
    _stub_embed_passages(monkeypatch)
    _seed_indexed_pdf(search_db, file_id="pdf-empty", file_path=fake_pdf)

    # First index lands a row.
    metadata_worker.index_text_content("pdf-empty")
    assert _pdf_markdown_row(search_db, "pdf-empty") is not None

    # Re-index with empty page list (PyMuPDF4LLM still succeeds, but
    # there's nothing to extract — markdown="" path).
    fake.to_markdown.return_value = []
    metadata_worker.index_text_content("pdf-empty")

    # Stale row is gone; DB reflects current FS-derived state.
    assert _pdf_markdown_row(search_db, "pdf-empty") is None


# ---------------------------------------------------------------------------
# Every text embedding must be joinable to the full chunk text it was
# built from — the display string and the matched string are one string.
# Spec ``2026-08-29-related-passages.md`` §3.1 / §5.2.
# ---------------------------------------------------------------------------


def test_text_embeddings_record_the_fts_chunk_index_they_came_from(
    search_db, fake_pdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_pymupdf4llm(
        monkeypatch,
        [
            {"text": "Alpha body, the first chunk."},
            {"text": "Beta body, the second chunk."},
        ],
    )
    _stub_embed_passages(monkeypatch)
    _seed_indexed_pdf(search_db, file_id="pdf-ci", file_path=fake_pdf)

    metadata_worker.index_text_content("pdf-ci")

    with search_db.connect() as conn:
        embeddings = conn.execute(text(
            "SELECT chunk_index, content_preview FROM embeddings "
            "WHERE file_id = 'pdf-ci' AND embedding_type = 'text_content' "
            "ORDER BY chunk_index"
        )).fetchall()
        # FTS5 stores every column as text, so the join casts.
        joined = conn.execute(text(
            "SELECT e.chunk_index, f.text FROM embeddings AS e "
            "JOIN fts_text_content AS f ON f.file_id = e.file_id "
            "  AND f.chunk_index = CAST(e.chunk_index AS TEXT) "
            "WHERE e.file_id = 'pdf-ci' "
            "  AND e.embedding_type = 'text_content' "
            "ORDER BY e.chunk_index"
        )).fetchall()

    assert len(embeddings) == 2
    assert [row[0] for row in embeddings] == [0, 1]

    # Every embedding resolves to exactly one chunk, and that chunk is
    # the one it was built from.
    assert len(joined) == len(embeddings)
    for (_, preview), (_, full_text) in zip(embeddings, joined):
        assert full_text.startswith(preview)
