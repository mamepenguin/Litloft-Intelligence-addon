"""Schema migration tests for ``file_summaries``.

``_migrate_file_summaries_if_needed`` must:

* add ``edited_at`` / ``short_original`` / ``long_original`` to
  pre-existing rows without disturbing stored data
* add ``detailed_summary`` et al for the Markdown long-form summary
* be idempotent (running twice on an up-to-date schema is a no-op)

Fresh installs get the columns from ``_create_file_summaries_table``;
this test exercises the upgrade path.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
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
def legacy_engine(tmp_path):
    """Build a DB with the pre-edit schema and a row already in it."""
    db_path = tmp_path / "search.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # Pre-edit-feature CREATE statement (note: no edited_at / *_original).
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE file_summaries ("
            "  file_id TEXT PRIMARY KEY,"
            "  short_summary TEXT NOT NULL,"
            "  long_summary TEXT NOT NULL,"
            "  model TEXT NOT NULL,"
            "  context_type TEXT NOT NULL,"
            "  context_chars INTEGER NOT NULL,"
            "  was_truncated INTEGER NOT NULL DEFAULT 0,"
            "  status TEXT NOT NULL DEFAULT 'generated',"
            "  created_at TEXT NOT NULL"
            ")"
        ))
        conn.execute(
            text(
                "INSERT INTO file_summaries "
                "(file_id, short_summary, long_summary, model, context_type, "
                "context_chars, was_truncated, status, created_at) "
                "VALUES (:fid, :s, :l, :m, :ct, :cc, :wt, :st, :ca)"
            ),
            {
                "fid": "legacy-1",
                "s": "short text",
                "l": "long text",
                "m": "gemma:e4b",
                "ct": "video",
                "cc": 500,
                "wt": 0,
                "st": "generated",
                "ca": datetime.now(UTC).isoformat(),
            },
        )
    return engine


def _columns(engine) -> set[str]:
    with engine.connect() as conn:
        return {
            row[1]
            for row in conn.execute(
                text("PRAGMA table_info(file_summaries)")
            ).fetchall()
        }


def test_migration_adds_edit_columns(legacy_engine):
    from app.database import _migrate_file_summaries_if_needed

    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)

    cols = _columns(legacy_engine)
    assert "edited_at" in cols
    assert "short_original" in cols
    assert "long_original" in cols


def test_migration_preserves_existing_row(legacy_engine):
    """Adding columns must not disturb pre-existing data."""
    from app.database import _migrate_file_summaries_if_needed

    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)

    with legacy_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT short_summary, long_summary, edited_at, "
                "short_original, long_original FROM file_summaries "
                "WHERE file_id = :fid"
            ),
            {"fid": "legacy-1"},
        ).fetchone()

    assert row is not None
    assert row[0] == "short text"
    assert row[1] == "long text"
    # New columns default to NULL for pre-existing rows.
    assert row[2] is None
    assert row[3] is None
    assert row[4] is None


def test_migration_is_idempotent(legacy_engine):
    from app.database import _migrate_file_summaries_if_needed

    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)
    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)

    cols = _columns(legacy_engine)
    # No duplicate columns, all edit columns present.
    assert "edited_at" in cols
    assert "short_original" in cols
    assert "long_original" in cols


def test_migration_adds_detailed_summary_columns(legacy_engine):
    """Markdown long-form summary columns appear after migration."""
    from app.database import _migrate_file_summaries_if_needed

    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)

    cols = _columns(legacy_engine)
    assert "detailed_summary" in cols
    assert "detailed_status" in cols
    assert "detailed_model" in cols
    assert "detailed_generated_at" in cols
    assert "detailed_context_chars" in cols
    assert "detailed_was_truncated" in cols
    assert "detailed_error" in cols


def test_migration_preserves_row_after_detailed_migration(legacy_engine):
    """Detailed-column migration must not disturb pre-existing short/long data."""
    from app.database import _migrate_file_summaries_if_needed

    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)

    with legacy_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT short_summary, long_summary, detailed_summary, "
                "detailed_status FROM file_summaries WHERE file_id = :fid"
            ),
            {"fid": "legacy-1"},
        ).fetchone()

    assert row is not None
    assert row[0] == "short text"
    assert row[1] == "long text"
    # Detailed columns default to NULL for pre-existing rows.
    assert row[2] is None
    assert row[3] is None


def test_fresh_schema_includes_edit_columns(tmp_path):
    """``_create_file_summaries_table`` on a clean DB produces all columns."""
    from app.database import _create_file_summaries_table

    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    with engine.begin() as conn:
        _create_file_summaries_table(conn)

    cols = _columns(engine)
    assert "edited_at" in cols
    assert "short_original" in cols
    assert "long_original" in cols
    # Original columns still present.
    assert "short_summary" in cols
    assert "long_summary" in cols
    # Detailed-summary columns are part of the fresh schema too.
    assert "detailed_summary" in cols
    assert "detailed_status" in cols
    assert "detailed_model" in cols
    assert "detailed_generated_at" in cols
    assert "detailed_context_chars" in cols
    assert "detailed_was_truncated" in cols
    assert "detailed_error" in cols
