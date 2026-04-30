"""Integration tests for ``IndexManager.handle_files_moved``.

Covers the happy path (drive / file_path / filename / title sync + FTS5
re-upsert) and edge cases (missing IndexedFile, unresolvable path,
policy off, no-op for empty input). Index completion flags must be
preserved so file content is not re-indexed unnecessarily.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub heavy ML/image deps before importing app modules.
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
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import Base  # noqa: E402
from app.models import IndexedFile  # noqa: E402,F401


@pytest.fixture()
def search_engine(tmp_path):
    """Build a search DB with indexed_files + minimal FTS mirrors."""
    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_files "
            "USING fts5(file_id UNINDEXED, filename, title, description, tags_text)"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_files_word "
            "USING fts5(file_id UNINDEXED, filename, title, description, tags_text)"
        ))
    return engine


@pytest.fixture()
def litloft_engine(tmp_path):
    """Build a stand-in Litloft DB with the columns ``handle_files_moved`` reads."""
    db_path = tmp_path / "litloft.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE files ("
            "  id TEXT PRIMARY KEY,"
            "  drive TEXT NOT NULL,"
            "  file_path TEXT NOT NULL,"
            "  filename TEXT NOT NULL,"
            "  title TEXT,"
            "  folder_path TEXT,"
            "  description TEXT,"
            "  file_size INTEGER,"
            "  file_type TEXT,"
            "  mime_type TEXT,"
            "  duration REAL,"
            "  deleted_at TEXT,"
            "  missing_since TEXT"
            ")"
        ))
    return engine


@pytest.fixture()
def patched_dbs(monkeypatch, search_engine, litloft_engine):
    """Wire get_search_db / get_litloft_db to in-memory engines."""
    SearchSession = sessionmaker(bind=search_engine, expire_on_commit=False)
    LitloftSession = sessionmaker(bind=litloft_engine, expire_on_commit=False)

    @contextmanager
    def _get_search_db():
        s = SearchSession()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    @contextmanager
    def _get_litloft_db():
        s = LitloftSession()
        try:
            yield s
        finally:
            s.close()

    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    monkeypatch.setattr("app.indexer.get_search_db", _get_search_db)
    monkeypatch.setattr("app.database.get_litloft_db", _get_litloft_db)
    monkeypatch.setattr("app.indexer.get_litloft_db", _get_litloft_db)
    return SearchSession, LitloftSession


@pytest.fixture()
def patch_resolve_file_path(monkeypatch):
    """Make resolve_file_path deterministic: drive + relative → /drives/<drive>/<rel>."""
    def fake_resolve(drive: str, relative: str) -> str | None:
        if drive == "_NO_MOUNT_":
            return None
        return f"/drives/{drive}/{relative}"

    monkeypatch.setattr("app.indexer.resolve_file_path", fake_resolve)
    return fake_resolve


@pytest.fixture()
def patch_policy_open(monkeypatch):
    """Default: every drive enabled. Override per-test if needed."""
    async def always_true(drive, feature):
        return True

    monkeypatch.setattr("app.policy_client.is_feature_enabled", always_true)
    return always_true


@pytest.fixture()
def make_manager():
    """Create a bare IndexManager whose handle_files_moved we exercise."""
    from app.indexer import IndexManager

    def _create():
        # Bypass __init__ side-effects (queue setup, model loading).
        return IndexManager.__new__(IndexManager)

    return _create


def _seed_indexed(SearchSession, **kwargs):
    defaults = dict(
        file_id="f1",
        drive="drive1",
        filename="old.mp4",
        file_path="/drives/drive1/旅行/old.mp4",
        file_type="video",
        mime_type="video/mp4",
        file_size=1000,
        active=True,
        metadata_indexed=True,
        clip_indexed=True,
        whisper_indexed=True,
        text_indexed=True,
        title="Old",
        description="d",
        tags_text="t",
    )
    defaults.update(kwargs)
    s = SearchSession()
    s.add(IndexedFile(**defaults))
    s.commit()
    s.close()


def _seed_litloft(LitloftSession, **kwargs):
    defaults = dict(
        id="f1",
        drive="drive1",
        file_path="アーカイブ/旅行/old.mp4",
        filename="old.mp4",
        title="Old",
        folder_path="アーカイブ/旅行",
        description="",
        file_size=1000,
        file_type="video",
        mime_type="video/mp4",
        duration=None,
    )
    defaults.update(kwargs)
    s = LitloftSession()
    s.execute(text(
        "INSERT INTO files (id, drive, file_path, filename, title, folder_path, "
        "description, file_size, file_type, mime_type, duration) "
        "VALUES (:id, :drive, :file_path, :filename, :title, :folder_path, "
        ":description, :file_size, :file_type, :mime_type, :duration)"
    ), defaults)
    s.commit()
    s.close()


class TestHandleFilesMovedHappyPath:
    @pytest.mark.asyncio
    async def test_syncs_drive_path_filename_title(
        self, patched_dbs, patch_resolve_file_path, patch_policy_open, make_manager,
    ):
        SearchSession, LitloftSession = patched_dbs
        _seed_indexed(SearchSession)  # old: drive1, filename old.mp4
        _seed_litloft(
            LitloftSession,
            drive="drive2",
            file_path="archive/2024/new.mp4",
            filename="new.mp4",
            title="New title",
        )

        manager = make_manager()
        await manager.handle_files_moved(["f1"])

        # Verify IndexedFile updated
        s = SearchSession()
        row = s.query(IndexedFile).filter_by(file_id="f1").first()
        assert row.drive == "drive2"
        assert row.file_path == "/drives/drive2/archive/2024/new.mp4"
        assert row.filename == "new.mp4"
        assert row.title == "New title"
        # Index completion flags untouched (file content unchanged)
        assert row.metadata_indexed is True
        assert row.clip_indexed is True
        assert row.whisper_indexed is True
        assert row.text_indexed is True
        s.close()

    @pytest.mark.asyncio
    async def test_fts5_row_replaced(
        self, patched_dbs, patch_resolve_file_path, patch_policy_open, make_manager,
    ):
        SearchSession, LitloftSession = patched_dbs
        _seed_indexed(SearchSession, filename="old.mp4", title="Old")
        # Pre-populate FTS5 with the old filename so we can verify replacement.
        s = SearchSession()
        s.execute(text(
            "INSERT INTO fts_files(file_id, filename, title, description, tags_text) "
            "VALUES('f1', 'old.mp4', 'Old', 'd', 't')"
        ))
        s.execute(text(
            "INSERT INTO fts_files_word(file_id, filename, title, description, tags_text) "
            "VALUES('f1', 'old.mp4', 'Old', 'd', 't')"
        ))
        s.commit()
        s.close()
        _seed_litloft(LitloftSession, filename="new.mp4", title="New")

        manager = make_manager()
        await manager.handle_files_moved(["f1"])

        s = SearchSession()
        rows = s.execute(text(
            "SELECT filename, title FROM fts_files WHERE file_id='f1'"
        )).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "new.mp4"
        assert rows[0][1] == "New"
        s.close()


class TestHandleFilesMovedEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_input_is_noop(
        self, patched_dbs, patch_resolve_file_path, patch_policy_open, make_manager,
    ):
        manager = make_manager()
        await manager.handle_files_moved([])
        # No raise, no DB access required.

    @pytest.mark.asyncio
    async def test_unindexed_id_is_skipped(
        self, patched_dbs, patch_resolve_file_path, patch_policy_open, make_manager,
    ):
        # Litloft has the file, but Intelligence hasn't indexed it yet.
        # handle_files_moved should silently drop the id (reconcile() will
        # add it on the next pass when policy permits).
        _, LitloftSession = patched_dbs
        _seed_litloft(LitloftSession, id="ghost")

        manager = make_manager()
        await manager.handle_files_moved(["ghost"])
        # No raise, no IndexedFile created.

    @pytest.mark.asyncio
    async def test_unresolvable_path_skipped_with_log(
        self, patched_dbs, patch_resolve_file_path, patch_policy_open, make_manager,
    ):
        SearchSession, LitloftSession = patched_dbs
        _seed_indexed(SearchSession, drive="drive1", file_path="/drives/drive1/old.mp4")
        # Mock returns None for unmounted drive.
        _seed_litloft(LitloftSession, drive="_NO_MOUNT_", file_path="x.mp4")

        manager = make_manager()
        await manager.handle_files_moved(["f1"])

        # IndexedFile must NOT be partially updated when path is unresolvable.
        s = SearchSession()
        row = s.query(IndexedFile).filter_by(file_id="f1").first()
        assert row.drive == "drive1"  # unchanged
        assert row.file_path == "/drives/drive1/old.mp4"  # unchanged
        s.close()

    @pytest.mark.asyncio
    async def test_policy_off_drive_is_skipped(
        self, patched_dbs, patch_resolve_file_path, monkeypatch, make_manager,
    ):
        SearchSession, LitloftSession = patched_dbs
        _seed_indexed(SearchSession, drive="drive1", filename="old.mp4")
        _seed_litloft(LitloftSession, drive="drive2", filename="new.mp4")

        async def always_false(drive, feature):
            return False

        monkeypatch.setattr("app.policy_client.is_feature_enabled", always_false)

        manager = make_manager()
        await manager.handle_files_moved(["f1"])

        s = SearchSession()
        row = s.query(IndexedFile).filter_by(file_id="f1").first()
        assert row.filename == "old.mp4"  # unchanged
        s.close()

    @pytest.mark.asyncio
    async def test_policy_lookup_failure_falls_open(
        self, patched_dbs, patch_resolve_file_path, monkeypatch, make_manager,
    ):
        """Policy client raising should not block the sync (fail open)."""
        SearchSession, LitloftSession = patched_dbs
        _seed_indexed(SearchSession, drive="drive1", filename="old.mp4")
        _seed_litloft(LitloftSession, drive="drive1", filename="new.mp4")

        async def boom(drive, feature):
            raise RuntimeError("policy service unreachable")

        monkeypatch.setattr("app.policy_client.is_feature_enabled", boom)

        manager = make_manager()
        await manager.handle_files_moved(["f1"])

        s = SearchSession()
        row = s.query(IndexedFile).filter_by(file_id="f1").first()
        # Sync still applied — fail-open semantic
        assert row.filename == "new.mp4"
        s.close()

    @pytest.mark.asyncio
    async def test_partial_litloft_match(
        self, patched_dbs, patch_resolve_file_path, patch_policy_open, make_manager,
    ):
        """Some ids exist in core, others don't — only existing ones are synced."""
        SearchSession, LitloftSession = patched_dbs
        _seed_indexed(SearchSession, file_id="f1", filename="old.mp4")
        _seed_litloft(LitloftSession, id="f1", filename="new.mp4")
        # f2 is not in core (e.g. purged just before webhook arrived).

        manager = make_manager()
        await manager.handle_files_moved(["f1", "f2"])

        s = SearchSession()
        row = s.query(IndexedFile).filter_by(file_id="f1").first()
        assert row.filename == "new.mp4"
        s.close()
