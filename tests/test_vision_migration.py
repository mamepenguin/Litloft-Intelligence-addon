"""Schema migration tests for the vision_describe feature.

``_migrate_file_summaries_if_needed`` must add five nullable columns
used by the vision_describe feature:

* ``visual_description`` TEXT NULL — LLM-generated description body
* ``visual_description_generated_at`` TEXT NULL — ISO timestamp of last
  successful generation
* ``visual_description_model`` TEXT NULL — model name used (for
  re-generation heuristics when the vision model changes)
* ``visual_description_status`` TEXT NULL — one of NULL / "pending" /
  "success" / "failed" / "unsupported"
* ``visual_description_error`` TEXT NULL — why the last attempt produced
  no description, as one of the ``app.llm`` FAILURE_* values

Idempotent: running migrations twice must not raise. Existing data on
pre-existing rows must be preserved. Fresh install via
``_create_file_summaries_table`` must include the new columns so the
ALTER path is skipped on clean installs.
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


_VISION_COLUMNS = (
    "visual_description",
    "visual_description_generated_at",
    "visual_description_model",
    "visual_description_status",
    "visual_description_error",
)


@pytest.fixture()
def legacy_engine(tmp_path):
    """Build a pre-vision file_summaries DB with the detailed-edit schema.

    Represents the world after the detailed-summary edit columns shipped
    but before vision_describe landed. Seeds one real row so the test
    can verify that pre-existing data survives the ALTER path unchanged.
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
            "  detailed_error TEXT,"
            "  detailed_original TEXT,"
            "  detailed_edited_at TEXT"
            ")"
        ))
        now = datetime.now(UTC).isoformat()
        conn.execute(
            text(
                "INSERT INTO file_summaries "
                "(file_id, short_summary, long_summary, model, context_type, "
                "context_chars, was_truncated, status, created_at, "
                "detailed_summary, detailed_status) "
                "VALUES (:fid, 's', 'l', 'm', 'image', 200, 0, 'generated', "
                ":now, 'pre-existing detailed', 'generated')"
            ),
            {"fid": "img-1", "now": now},
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


def test_migration_adds_all_four_vision_columns(legacy_engine):
    from app.database import _migrate_file_summaries_if_needed

    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)

    cols = _columns(legacy_engine)
    for name in _VISION_COLUMNS:
        assert name in cols, f"migration must add {name}"


def test_migration_preserves_existing_rows(legacy_engine):
    """Pre-existing file_summaries rows must survive the ALTER."""
    from app.database import _migrate_file_summaries_if_needed

    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)

    with legacy_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT detailed_summary, detailed_status, "
                "visual_description, visual_description_status "
                "FROM file_summaries WHERE file_id = 'img-1'"
            )
        ).fetchone()

    assert row is not None
    assert row[0] == "pre-existing detailed"
    assert row[1] == "generated"
    # Newly added columns default to NULL.
    assert row[2] is None
    assert row[3] is None


def test_migration_is_idempotent(legacy_engine):
    """Running the migration twice must not raise (duplicate column would)."""
    from app.database import _migrate_file_summaries_if_needed

    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)
    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)

    cols = _columns(legacy_engine)
    for name in _VISION_COLUMNS:
        assert name in cols


def test_fresh_schema_includes_vision_columns(tmp_path):
    """Clean install via _create_file_summaries_table must include them.

    The ALTER path should only be necessary for upgraded DBs.
    """
    from app.database import _create_file_summaries_table

    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    with engine.begin() as conn:
        _create_file_summaries_table(conn)

    cols = _columns(engine)
    for name in _VISION_COLUMNS:
        assert name in cols


def test_vision_status_column_accepts_expected_values(legacy_engine):
    """Spec statuses: NULL / 'pending' / 'success' / 'failed' / 'unsupported'.

    The migration uses TEXT (no CHECK constraint) so all spec values
    must insert without error. The worker/router layer owns value
    validation, but schema-level rejection would be a regression.
    """
    from app.database import _migrate_file_summaries_if_needed

    with legacy_engine.begin() as conn:
        _migrate_file_summaries_if_needed(conn)

    now = datetime.now(UTC).isoformat()
    with legacy_engine.begin() as conn:
        for i, status in enumerate(
            ("pending", "success", "failed", "unsupported")
        ):
            conn.execute(
                text(
                    "INSERT INTO file_summaries "
                    "(file_id, short_summary, long_summary, model, "
                    "context_type, context_chars, was_truncated, status, "
                    "created_at, visual_description_status) "
                    "VALUES (:fid, '', '', '', 'image', 0, 0, 'hidden', "
                    ":now, :vs)"
                ),
                {"fid": f"vf-{i}", "now": now, "vs": status},
            )


class TestEveryVisionColumnIsClearedTogether:
    """Detector for a column added to the writer and forgotten elsewhere.

    Three paths clear vision data — DELETE, the policy-off purge, and
    the worker's status writes. The reason column was added to the
    writer and missed by the purge, which left a reason to be read
    against an emptied row. Deriving the SQL from one list is the fix;
    this checks the list itself still matches the table.
    """

    def test_the_clear_covers_every_vision_column_in_the_table(self, tmp_path):
        from sqlalchemy import create_engine

        from app.database import (
            VISION_DESCRIBE_COLUMNS,
            _create_file_summaries_table,
        )

        engine = create_engine(f"sqlite:///{tmp_path}/fresh.db")
        with engine.begin() as conn:
            _create_file_summaries_table(conn)

        in_table = {
            name
            for name in _columns(engine)
            if name.startswith("visual_description")
        }
        assert in_table == set(VISION_DESCRIBE_COLUMNS), (
            "a vision column exists that the clear paths do not name"
        )

    def test_the_derived_sql_names_each_column_once(self):
        from app.database import (
            VISION_DESCRIBE_CLEAR_SQL,
            VISION_DESCRIBE_COLUMNS,
            VISION_DESCRIBE_PRESENT_SQL,
        )

        for column in VISION_DESCRIBE_COLUMNS:
            assert f"{column} = NULL" in VISION_DESCRIBE_CLEAR_SQL
            assert f"{column} IS NOT NULL" in VISION_DESCRIBE_PRESENT_SQL
        assert VISION_DESCRIBE_CLEAR_SQL.count("= NULL") == len(
            VISION_DESCRIBE_COLUMNS
        )
