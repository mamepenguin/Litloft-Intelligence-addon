"""Tests for GET /files/{file_id}/pdf-markdown.

Phase 4 of ``2026-04-27-intelligence-pdf-markdown-indexing.md`` adds
the read-only viewing API for the Markdown body produced during
text-content indexing. Coverage:

* Happy path: indexed PDF with a ``pdf_markdown`` row → 200 + body.
* PDF that was indexed via the fitz fallback (no ``pdf_markdown`` row)
  → 404.
* Non-PDF indexed file → 404 (no ``pdf_markdown`` row exists for it
  either, so the SELECT just misses).
* Cross-drive file_id → 404 via ``_get_indexed_file_or_404`` (must
  not leak existence of files in other drives — ``drive_boundary``
  rule).
* Soft-deleted file (``active=False``) → 404 because the indexed-file
  guard already excludes inactive rows.

Mirrors the fixture style of ``test_chunk_excerpt.py``: a real SQLite
engine with the ORM tables created via ``Base.metadata.create_all``,
plus seed rows for in-drive / cross-drive / soft-deleted files. No
sqlite-vec or FTS5 needed for this endpoint.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

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

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import Base  # noqa: E402
from app.models import IndexedFile, PdfMarkdown  # noqa: E402
from app.routers.files import get_pdf_markdown  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    """Real SQLite engine with the ORM tables this endpoint reads.

    The endpoint imports ``get_search_db`` lazily inside the handler
    body, so patching the module-level binding in ``app.database`` is
    sufficient — there is no module-level reference in
    ``app.routers.files`` to override.
    """
    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    Base.metadata.create_all(engine)

    seed = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        seed.add_all([
            IndexedFile(
                file_id="pdf_in_drive",
                drive="drive1",
                filename="paper.pdf",
                file_path="/drives/drive1/paper.pdf",
                file_type="document",
                mime_type="application/pdf",
                file_size=1234,
                active=True,
            ),
            IndexedFile(
                file_id="pdf_fallback",
                drive="drive1",
                filename="scanned.pdf",
                file_path="/drives/drive1/scanned.pdf",
                file_type="document",
                mime_type="application/pdf",
                file_size=2345,
                active=True,
            ),
            IndexedFile(
                file_id="non_pdf",
                drive="drive1",
                filename="notes.txt",
                file_path="/drives/drive1/notes.txt",
                file_type="text",
                mime_type="text/plain",
                file_size=512,
                active=True,
            ),
            IndexedFile(
                file_id="pdf_other_drive",
                drive="other",
                filename="secret.pdf",
                file_path="/drives/other/secret.pdf",
                file_type="document",
                mime_type="application/pdf",
                file_size=999,
                active=True,
            ),
            IndexedFile(
                file_id="pdf_deleted",
                drive="drive1",
                filename="gone.pdf",
                file_path="/drives/drive1/gone.pdf",
                file_type="document",
                mime_type="application/pdf",
                file_size=777,
                active=False,
            ),
        ])
        seed.commit()
    finally:
        seed.close()

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

    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    return engine, Session


def _seed_pdf_markdown(
    Session,
    file_id: str,
    *,
    markdown: str = "# Title\n\nbody",
    page_count: int = 3,
    extractor: str = "pymupdf4llm",
    generated_at: datetime | None = None,
) -> PdfMarkdown:
    """Insert one ``pdf_markdown`` row tied to ``file_id``."""
    session = Session()
    try:
        row = PdfMarkdown(
            file_id=file_id,
            markdown=markdown,
            page_count=page_count,
            extractor=extractor,
            generated_at=generated_at or datetime.now(UTC),
        )
        session.add(row)
        session.commit()
    finally:
        session.close()
    return row


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestPdfMarkdownHappyPath:
    """Indexed PDF with a stored Markdown body."""

    @pytest.mark.asyncio
    async def test_returns_persisted_markdown(self, search_db):
        _, Session = search_db
        _seed_pdf_markdown(
            Session,
            "pdf_in_drive",
            markdown="# Heading\n\n- bullet\n- bullet\n",
            page_count=7,
            extractor="pymupdf4llm",
        )

        result = await get_pdf_markdown("pdf_in_drive", "drive1")

        assert result.file_id == "pdf_in_drive"
        assert result.markdown == "# Heading\n\n- bullet\n- bullet\n"
        assert result.page_count == 7
        assert result.extractor == "pymupdf4llm"
        assert isinstance(result.generated_at, datetime)


# ---------------------------------------------------------------------------
# Negative paths
# ---------------------------------------------------------------------------


class TestPdfMarkdownNotFound:
    """Every non-happy path collapses to 404 (no 403, no 400 leakage)."""

    @pytest.mark.asyncio
    async def test_pdf_without_markdown_row_404(self, search_db):
        # ``pdf_fallback`` is an indexed PDF whose extraction took the
        # fitz fallback path, so the worker did not write a
        # ``pdf_markdown`` row. Endpoint must return 404.
        with pytest.raises(HTTPException) as exc:
            await get_pdf_markdown("pdf_fallback", "drive1")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_non_pdf_file_404(self, search_db):
        # A ``.txt`` file is indexed but never has a ``pdf_markdown``
        # row by design — the SELECT misses and we 404.
        with pytest.raises(HTTPException) as exc:
            await get_pdf_markdown("non_pdf", "drive1")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_cross_drive_file_404(self, search_db):
        # ``pdf_other_drive`` exists and even has a Markdown row, but
        # belongs to drive 'other' — must surface as 404 to drive1
        # callers, never as 403 (drive_boundary rule).
        _, Session = search_db
        _seed_pdf_markdown(Session, "pdf_other_drive")

        with pytest.raises(HTTPException) as exc:
            await get_pdf_markdown("pdf_other_drive", "drive1")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_file_id_404(self, search_db):
        with pytest.raises(HTTPException) as exc:
            await get_pdf_markdown("ghost", "drive1")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_soft_deleted_file_404(self, search_db):
        # ``pdf_deleted`` has ``active=False``. The
        # ``_get_indexed_file_or_404`` guard already filters those out
        # before any SELECT against ``pdf_markdown``.
        _, Session = search_db
        _seed_pdf_markdown(Session, "pdf_deleted")

        with pytest.raises(HTTPException) as exc:
            await get_pdf_markdown("pdf_deleted", "drive1")
        assert exc.value.status_code == 404
