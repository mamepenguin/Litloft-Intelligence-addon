"""Schema migration tests for the detailed-summary edit columns.

``_migrate_file_summaries_if_needed`` must:

* add ``detailed_original`` and ``detailed_edited_at`` when they are
  missing from an otherwise-migrated table;
* preserve pre-existing data through the upgrade;
* stay idempotent across repeated runs;
* and the fresh-install DDL in ``_create_file_summaries_table`` must
  include the new columns so a clean install skips the ALTER path.
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
    """Build a DB that has the post-detailed-column schema but no edit columns.

    Simulates the world after the detailed-summary feature shipped but
    before the Phase 2 edit columns landed.
    """
    db_path = tmp_path / "search.db"
    engine = create_engine(f"sqlite:///{db_path}")

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
            "  created_at TEXT NOT NULL,"
            "  edited_at TEXT,"
            "  short_original TEXT,"
            "  long_original TEXT,"
            "  detailed_summary TEXT,"
            "  detailed_status TEXT,"
            "  detailed_model TEXT,"
            "  detailed_generated_at TEXT,"
            "  detailed_context_chars INTEGER,"
            "  detailed_was_truncated INTEGER,"
            "  detailed_error TEXT"
            ")"
        ))
        now = datetime.now(UTC).isoformat()
        conn.execute(
            text(
                "INSERT INTO file_summaries "
                "(file_id, short_summary, long_summary, model, context_type, "
                "context_chars, was_truncated, status, created_at, "
                "detailed_summary, detailed_status) "
                "VALUES (:fid, 's', 'l', 'm', 'video', 100, 0, 'generated', "
                ":now, '## 全体像\nbody', 'generated')"
            ),
            {"fid": "file-1", "now": now},
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


def test_migration_adds_detailed_edit_columns(legacy_engine):
    from app.database import _migrate_file_summaries_if_needed

    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)

    cols = _columns(legacy_engine)
    assert "detailed_original" in cols
    assert "detailed_edited_at" in cols


def test_migration_preserves_existing_row(legacy_engine):
    """The ALTER must not perturb previously stored columns."""
    from app.database import _migrate_file_summaries_if_needed

    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)

    with legacy_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT detailed_summary, detailed_status, "
                "detailed_original, detailed_edited_at "
                "FROM file_summaries WHERE file_id = 'file-1'"
            )
        ).fetchone()

    assert row is not None
    assert row[0].startswith("## 全体像")
    assert row[1] == "generated"
    # Freshly-added columns default to NULL.
    assert row[2] is None
    assert row[3] is None


def test_migration_idempotent(legacy_engine):
    from app.database import _migrate_file_summaries_if_needed

    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)
    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)

    cols = _columns(legacy_engine)
    # Only one column of each name — SQLite would have raised otherwise.
    assert "detailed_original" in cols
    assert "detailed_edited_at" in cols


def test_fresh_schema_includes_detailed_edit_columns(tmp_path):
    from app.database import _create_file_summaries_table

    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    with engine.begin() as conn:
        _create_file_summaries_table(conn)

    cols = _columns(engine)
    assert "detailed_original" in cols
    assert "detailed_edited_at" in cols
