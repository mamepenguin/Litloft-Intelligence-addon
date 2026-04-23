"""Schema tests for ``file_insights`` (Step 1 of the FileInsight rollout).

Covers:
- Fresh DB: ``_create_file_insights_table`` produces all expected
  columns and indexes.
- Idempotence: running the create twice on an existing DB is a no-op
  and does not raise.
- Coexistence with ``file_summaries``: both tables can live in the
  same DB without schema conflict.

Backfill behaviour is covered in ``test_file_insights_backfill.py``.
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


@pytest.fixture()
def fresh_engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'search.db'}")


def test_fresh_schema_creates_all_columns(fresh_engine):
    """All FileInsight columns exist on a clean DB."""
    from app.database import _create_file_insights_table

    with fresh_engine.begin() as conn:
        _create_file_insights_table(conn)

    cols = _columns(fresh_engine, "file_insights")
    assert cols == {
        "id",
        "file_id",
        "kind",
        "content",
        "metadata_json",
        "status",
        "created_by",
        "created_at",
        "invalidated_at",
    }


def test_fresh_schema_creates_indexes(fresh_engine):
    """Both composite indexes exist."""
    from app.database import _create_file_insights_table

    with fresh_engine.begin() as conn:
        _create_file_insights_table(conn)

    idx = _indexes(fresh_engine, "file_insights")
    assert "idx_file_insights_file_kind_status" in idx
    assert "idx_file_insights_kind_status" in idx


def test_create_is_idempotent(fresh_engine):
    """Running the create twice does not raise."""
    from app.database import _create_file_insights_table

    with fresh_engine.begin() as conn:
        _create_file_insights_table(conn)
    # Second invocation must be a no-op.
    with fresh_engine.begin() as conn:
        _create_file_insights_table(conn)

    cols = _columns(fresh_engine, "file_insights")
    assert "id" in cols


def test_coexists_with_file_summaries(fresh_engine):
    """file_summaries and file_insights can share a DB."""
    from app.database import (
        _create_file_insights_table,
        _create_file_summaries_table,
    )

    with fresh_engine.begin() as conn:
        _create_file_summaries_table(conn)
        _create_file_insights_table(conn)

    fs_cols = _columns(fresh_engine, "file_summaries")
    fi_cols = _columns(fresh_engine, "file_insights")
    # Step 2b: body/model/etc. no longer exist on file_summaries; only
    # the workflow markers survive on the summaries side.
    assert "detailed_status" in fs_cols
    assert "detailed_error" in fs_cols
    assert "kind" in fi_cols


def test_status_default_is_active(fresh_engine):
    """Inserts without explicit status end up active (column default)."""
    from app.database import _create_file_insights_table

    with fresh_engine.begin() as conn:
        _create_file_insights_table(conn)

    with fresh_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO file_insights "
            "(id, file_id, kind, content, created_by, created_at) "
            "VALUES ('insight-1', 'file-1', 'detailed_summary', "
            "  'body', 'intelligence', '2026-04-23T00:00:00')"
        ))

    with fresh_engine.connect() as conn:
        row = conn.execute(text(
            "SELECT status FROM file_insights WHERE id = 'insight-1'"
        )).fetchone()
    assert row[0] == "active"
