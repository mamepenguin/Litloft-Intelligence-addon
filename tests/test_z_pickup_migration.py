"""Retiring ``pickup_cache``.

This migration runs inside ``init_search_db``, which the lifespan calls
with no guard, so anything it raises takes the addon down on boot — and
it only runs at all on databases that predate the change, which is every
existing installation and none of the fresh ones a test suite usually
builds. It shipped once with a NameError for exactly that reason.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine, text  # noqa: E402

from app.database import _migrate_pickup_to_rows  # noqa: E402


def _engine(tmp_path):
    return create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )


def _tables(conn):
    return {
        row[0] for row in
        conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
    }


def test_a_pre_upgrade_database_loses_the_old_table(tmp_path):
    engine = _engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE pickup_cache (drive_id TEXT, viewer_id TEXT, "
            "file_ids TEXT, computed_at TEXT, watch_history_checkpoint TEXT)"
        ))
        conn.execute(text(
            "INSERT INTO pickup_cache VALUES ('a','v1','[\"f1\"]','now','x')"
        ))

    with engine.begin() as conn:
        _migrate_pickup_to_rows(conn)
        assert "pickup_cache" not in _tables(conn)


def test_running_it_twice_is_harmless(tmp_path):
    """The sweep it belongs to runs on every boot."""
    engine = _engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE pickup_cache (drive_id TEXT)"))

    with engine.begin() as conn:
        _migrate_pickup_to_rows(conn)
        _migrate_pickup_to_rows(conn)
        assert "pickup_cache" not in _tables(conn)


def test_a_fresh_database_is_untouched(tmp_path):
    engine = _engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE something_else (x TEXT)"))

    with engine.begin() as conn:
        _migrate_pickup_to_rows(conn)
        assert _tables(conn) == {"something_else"}


def test_it_does_not_disturb_the_replacement_tables(tmp_path):
    engine = _engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE pickup_cache (drive_id TEXT)"))
        conn.execute(text(
            "CREATE TABLE pickup_item (drive_id TEXT, viewer_id TEXT, "
            "rank INTEGER, file_id TEXT, cluster_id TEXT, channel TEXT, "
            "score REAL)"
        ))
        conn.execute(text(
            "CREATE TABLE pickup_profile (drive_id TEXT, viewer_id TEXT, "
            "total INTEGER, computed_at TEXT, watch_history_checkpoint TEXT)"
        ))
        conn.execute(text(
            "INSERT INTO pickup_item VALUES ('a','v1',1,'f1','c0','clip',0.9)"
        ))

    with engine.begin() as conn:
        _migrate_pickup_to_rows(conn)
        assert {"pickup_item", "pickup_profile"} <= _tables(conn)
        kept = conn.execute(text("SELECT file_id FROM pickup_item")).fetchall()
        assert [r[0] for r in kept] == ["f1"]


def test_it_runs_under_the_real_init_path(tmp_path, monkeypatch):
    """A NameError here is only reachable on a pre-upgrade database.

    Every fixture that builds a schema from scratch skips the branch
    entirely, which is why this shipped broken: the migration is dead
    code on precisely the databases a test suite tends to create.
    """
    import app.database as database

    engine = _engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE pickup_cache (drive_id TEXT)"))

    with engine.begin() as conn:
        _migrate_pickup_to_rows(conn)

    assert not hasattr(database, "logger"), (
        "app.database has no module-level logger; every migration must "
        "bind its own or it raises NameError on the databases that need it"
    )
