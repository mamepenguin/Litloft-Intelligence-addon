"""Schema creation test for the new ``detailed_summary_citations`` table.

Covers:

* Fresh-install DDL produces the expected columns, index, and
  ``UNIQUE (file_id, section_path)`` constraint.
* Running the creator twice is a no-op (idempotent).
* Existing data is untouched by a second run (e.g. upgrade path).
"""

from __future__ import annotations

import sys
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

from sqlalchemy import create_engine, text  # noqa: E402


@pytest.fixture()
def fresh_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    return engine


def _columns(engine, table: str) -> set[str]:
    with engine.connect() as conn:
        return {
            row[1]
            for row in conn.execute(
                text(f"PRAGMA table_info({table})")
            ).fetchall()
        }


def _indexes(engine, table: str) -> set[str]:
    with engine.connect() as conn:
        return {
            row[1]
            for row in conn.execute(
                text(f"PRAGMA index_list({table})")
            ).fetchall()
        }


def test_creates_citations_table_columns(fresh_engine):
    from app.database import _create_detailed_summary_citations_table

    with fresh_engine.begin() as conn:
        _create_detailed_summary_citations_table(conn)

    cols = _columns(fresh_engine, "detailed_summary_citations")
    expected = {
        "id",
        "file_id",
        "section_path",
        "segment_type",
        "segment_text",
        "citation_chunk_ids",
        "top_score",
        "has_citation",
        "created_at",
    }
    assert expected.issubset(cols)


def test_creates_index_on_file_id(fresh_engine):
    from app.database import _create_detailed_summary_citations_table

    with fresh_engine.begin() as conn:
        _create_detailed_summary_citations_table(conn)

    indexes = _indexes(fresh_engine, "detailed_summary_citations")
    # The named index from the DDL plus any auto-UNIQUE index SQLite
    # generates for the composite constraint.
    assert "idx_detailed_citations_file" in indexes


def test_unique_constraint_on_file_id_section_path(fresh_engine):
    from app.database import _create_detailed_summary_citations_table

    with fresh_engine.begin() as conn:
        _create_detailed_summary_citations_table(conn)

    # Insert the first row; duplicate (file_id, section_path) must raise.
    from sqlalchemy.exc import IntegrityError

    with fresh_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO detailed_summary_citations "
                "(file_id, section_path, segment_type, segment_text, "
                "citation_chunk_ids, top_score, has_citation) "
                "VALUES ('f1', 's/0', 'paragraph', 'x', '[]', 0.5, 1)"
            )
        )

    with pytest.raises(IntegrityError):
        with fresh_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO detailed_summary_citations "
                    "(file_id, section_path, segment_type, segment_text, "
                    "citation_chunk_ids, top_score, has_citation) "
                    "VALUES ('f1', 's/0', 'paragraph', 'y', '[]', 0.5, 1)"
                )
            )


def test_creator_is_idempotent(fresh_engine):
    from app.database import _create_detailed_summary_citations_table

    with fresh_engine.begin() as conn:
        _create_detailed_summary_citations_table(conn)
    # Running again must not raise, must not duplicate.
    with fresh_engine.begin() as conn:
        _create_detailed_summary_citations_table(conn)

    cols = _columns(fresh_engine, "detailed_summary_citations")
    assert "file_id" in cols
    assert "section_path" in cols


def test_existing_data_survives_second_creation(fresh_engine):
    """Re-running DDL after inserting rows must preserve the data."""
    from app.database import _create_detailed_summary_citations_table

    with fresh_engine.begin() as conn:
        _create_detailed_summary_citations_table(conn)
        conn.execute(
            text(
                "INSERT INTO detailed_summary_citations "
                "(file_id, section_path, segment_type, segment_text, "
                "citation_chunk_ids, top_score, has_citation) "
                "VALUES ('f1', 's/0', 'paragraph', 'keep me', "
                "'[\"transcript:1\"]', 0.7, 1)"
            )
        )

    with fresh_engine.begin() as conn:
        _create_detailed_summary_citations_table(conn)

    with fresh_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT segment_text, top_score, has_citation "
                "FROM detailed_summary_citations WHERE file_id = 'f1'"
            )
        ).fetchone()

    assert row is not None
    assert row[0] == "keep me"
    assert float(row[1]) == pytest.approx(0.7)
    assert bool(row[2]) is True
