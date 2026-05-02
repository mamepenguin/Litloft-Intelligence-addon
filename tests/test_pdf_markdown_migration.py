"""Schema tests for ``pdf_markdown`` (Phase 1 of the PyMuPDF4LLM rollout).

Covers:

- Fresh DB: ``_create_pdf_markdown_table`` produces the expected
  columns and the FK to ``indexed_files`` with ON DELETE CASCADE.
- ORM CRUD: ``PdfMarkdown`` rows can be inserted, selected, updated,
  and deleted through SQLAlchemy.
- CASCADE: deleting an ``indexed_files`` row removes the matching
  ``pdf_markdown`` row.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

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

import sqlite3  # noqa: E402

from sqlalchemy import create_engine, event, text  # noqa: E402

from app.database import Base  # noqa: E402
from app.models import IndexedFile, PdfMarkdown  # noqa: E402


def _enable_fks(dbapi_conn: sqlite3.Connection, _: object) -> None:
    """Match the runtime listener so FK CASCADE actually fires in tests."""
    dbapi_conn.execute("PRAGMA foreign_keys=ON")


@pytest.fixture()
def engine(tmp_path):
    db_path = tmp_path / "search.db"
    engine = create_engine(f"sqlite:///{db_path}")
    event.listen(engine, "connect", _enable_fks)
    Base.metadata.create_all(engine)

    # Apply the new-table migration the same way ``init_search_db`` does
    # so the test exercises the production code path.
    from app.database import _create_pdf_markdown_table

    with engine.begin() as conn:
        _create_pdf_markdown_table(conn)

    return engine


def _columns(engine_, table: str) -> set[str]:
    with engine_.connect() as conn:
        return {
            row[1]
            for row in conn.execute(
                text(f"PRAGMA table_info({table})")
            ).fetchall()
        }


def _foreign_keys(engine_, table: str) -> list[tuple]:
    with engine_.connect() as conn:
        return [
            tuple(row)
            for row in conn.execute(
                text(f"PRAGMA foreign_key_list({table})")
            ).fetchall()
        ]


def _insert_indexed_file(
    engine_,
    *,
    file_id: str,
    mime_type: str = "application/pdf",
    active: bool = True,
    text_indexed: bool = True,
    metadata_indexed: bool = True,
) -> None:
    now_iso = datetime.now(UTC).isoformat()
    with engine_.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO indexed_files "
                "(file_id, drive, filename, file_path, file_type, "
                " mime_type, file_size, active, text_indexed, "
                " metadata_indexed, "
                " title, description, tags_text, indexed_at, updated_at) "
                "VALUES (:fid, 'd1', :fid, :fid, 'document', :mime, 100, "
                " :active, :ti, :mi, '', '', '', :ts, :ts)"
            ),
            {
                "fid": file_id,
                "mime": mime_type,
                "active": 1 if active else 0,
                "ti": 1 if text_indexed else 0,
                "mi": 1 if metadata_indexed else 0,
                "ts": now_iso,
            },
        )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_fresh_schema_creates_all_columns(engine):
    """All PdfMarkdown columns exist on a clean DB."""
    cols = _columns(engine, "pdf_markdown")
    assert cols == {
        "file_id",
        "markdown",
        "page_count",
        "extractor",
        "generated_at",
        "updated_at",
    }


def test_create_is_idempotent(engine):
    """Running the create twice does not raise and preserves the schema."""
    from app.database import _create_pdf_markdown_table

    with engine.begin() as conn:
        _create_pdf_markdown_table(conn)

    cols = _columns(engine, "pdf_markdown")
    assert "markdown" in cols


def test_foreign_key_cascades_to_indexed_files(engine):
    fks = _foreign_keys(engine, "pdf_markdown")
    # PRAGMA foreign_key_list rows:
    # (id, seq, table, from, to, on_update, on_delete, match)
    assert len(fks) == 1
    fk = fks[0]
    assert fk[2] == "indexed_files"
    assert fk[3] == "file_id"
    assert fk[4] == "file_id"
    assert fk[6] == "CASCADE"


# ---------------------------------------------------------------------------
# ORM CRUD
# ---------------------------------------------------------------------------


def test_orm_crud_roundtrip(engine):
    """INSERT / SELECT / UPDATE / DELETE via the ORM all behave."""
    from sqlalchemy.orm import Session

    _insert_indexed_file(engine, file_id="pdf-1")

    now = datetime(2026, 4, 27, 10, 0, 0, tzinfo=UTC)
    later = now + timedelta(minutes=5)

    with Session(engine) as session:
        session.add(
            PdfMarkdown(
                file_id="pdf-1",
                markdown="# Title\n\nbody",
                page_count=12,
                extractor="pymupdf4llm",
                generated_at=now,
                updated_at=now,
            )
        )
        session.commit()

    with Session(engine) as session:
        row = session.get(PdfMarkdown, "pdf-1")
        assert row is not None
        assert row.markdown == "# Title\n\nbody"
        assert row.page_count == 12
        assert row.extractor == "pymupdf4llm"

        row.markdown = "# Title\n\nbody (revised)"
        row.updated_at = later
        session.commit()

    with Session(engine) as session:
        row = session.get(PdfMarkdown, "pdf-1")
        assert row.markdown.endswith("(revised)")
        assert row.updated_at.replace(tzinfo=UTC) == later

        session.delete(row)
        session.commit()

    with Session(engine) as session:
        assert session.get(PdfMarkdown, "pdf-1") is None


def test_cascade_delete_when_indexed_file_removed(engine):
    """Deleting an indexed_files row drops the matching pdf_markdown row."""
    from sqlalchemy.orm import Session

    _insert_indexed_file(engine, file_id="pdf-cascade")

    now = datetime(2026, 4, 27, 10, 0, 0, tzinfo=UTC)
    with Session(engine) as session:
        session.add(
            PdfMarkdown(
                file_id="pdf-cascade",
                markdown="cascade me",
                page_count=1,
                extractor="pymupdf4llm",
                generated_at=now,
                updated_at=now,
            )
        )
        session.commit()

    # Sanity: the row landed.
    with engine.connect() as conn:
        assert conn.execute(text(
            "SELECT COUNT(*) FROM pdf_markdown WHERE file_id = 'pdf-cascade'"
        )).scalar() == 1

    with engine.begin() as conn:
        conn.execute(text(
            "DELETE FROM indexed_files WHERE file_id = 'pdf-cascade'"
        ))

    with engine.connect() as conn:
        assert conn.execute(text(
            "SELECT COUNT(*) FROM pdf_markdown WHERE file_id = 'pdf-cascade'"
        )).scalar() == 0


