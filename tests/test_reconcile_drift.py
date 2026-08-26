"""Drift detection in reconcile() — webhook fallback path.

Phase 3 of the files.moved spec: reconcile() detects when an indexed
file's (drive, file_path, filename) snapshot disagrees with the core
DB and self-heals by reusing handle_files_moved. This is the safety
net for missed webhooks.
"""

from __future__ import annotations

import asyncio
import sys
import time
from contextlib import contextmanager
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
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import Base  # noqa: E402
from app.models import IndexedFile, TranscriptChunk  # noqa: E402,F401


@pytest.fixture()
def engines(tmp_path):
    """Build a search DB + minimal litloft DB; return (search_engine, litloft_engine)."""
    search_db = tmp_path / "search.db"
    search_engine = create_engine(
        f"sqlite:///{search_db}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(search_engine)
    with search_engine.begin() as conn:
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_files "
            "USING fts5(file_id UNINDEXED, filename, title, description, tags_text)"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_files_word "
            "USING fts5(file_id UNINDEXED, filename, title, description, tags_text)"
        ))

    litloft_db = tmp_path / "litloft.db"
    litloft_engine = create_engine(
        f"sqlite:///{litloft_db}",
        connect_args={"check_same_thread": False},
    )
    with litloft_engine.begin() as conn:
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

    return search_engine, litloft_engine


@pytest.fixture()
def patched(monkeypatch, engines):
    search_engine, litloft_engine = engines
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
    def _get_search_db_read():
        s = SearchSession()
        try:
            yield s
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
    monkeypatch.setattr("app.database.get_search_db_read", _get_search_db_read)
    monkeypatch.setattr("app.indexer.get_search_db_read", _get_search_db_read)
    monkeypatch.setattr("app.database.get_litloft_db", _get_litloft_db)
    monkeypatch.setattr("app.indexer.get_litloft_db", _get_litloft_db)

    def fake_resolve(drive, relative):
        return f"/drives/{drive}/{relative}"

    monkeypatch.setattr("app.indexer.resolve_file_path", fake_resolve)

    async def always_true(drive, feature):
        return True

    monkeypatch.setattr("app.policy_client.is_feature_enabled", always_true)
    return SearchSession, LitloftSession


@pytest.fixture()
def make_manager(monkeypatch):
    """Build an IndexManager with reconcile()'s side-helpers stubbed.

    _reset_loft_refs_with_new_vtt and _resume_incomplete are unrelated
    to drift detection but run as part of reconcile(); make them no-ops
    so the test stays focused.
    """
    from app.indexer import IndexManager

    def _create():
        m = IndexManager.__new__(IndexManager)
        m._reset_loft_refs_with_new_vtt = lambda: None

        async def _noop_resume():
            return 0

        m._resume_incomplete = _noop_resume
        return m

    return _create


def _seed_litloft_active(LitloftSession, **kwargs):
    defaults = dict(
        id="f1",
        drive="drive1",
        file_path="新/移動先/x.mp4",
        filename="x.mp4",
        title="X",
        folder_path="新/移動先",
        description="",
        file_size=1000,
        file_type="video",
        mime_type="video/mp4",
        duration=None,
        deleted_at=None,
        missing_since=None,
    )
    defaults.update(kwargs)
    s = LitloftSession()
    s.execute(text(
        "INSERT INTO files (id, drive, file_path, filename, title, folder_path, "
        "description, file_size, file_type, mime_type, duration, deleted_at, "
        "missing_since) "
        "VALUES (:id, :drive, :file_path, :filename, :title, :folder_path, "
        ":description, :file_size, :file_type, :mime_type, :duration, :deleted_at, "
        ":missing_since)"
    ), defaults)
    s.commit()
    s.close()


def _seed_indexed(SearchSession, **kwargs):
    defaults = dict(
        file_id="f1",
        drive="drive1",
        filename="x.mp4",
        file_path="/drives/drive1/旧/x.mp4",  # stale path (drift)
        file_type="video",
        mime_type="video/mp4",
        file_size=1000,
        active=True,
        metadata_indexed=True,
        clip_indexed=True,
        whisper_indexed=True,
        text_indexed=True,
        title="X",
        description="",
        tags_text="",
    )
    defaults.update(kwargs)
    s = SearchSession()
    s.add(IndexedFile(**defaults))
    s.commit()
    s.close()


class TestReconcileDriftDetection:
    @pytest.mark.asyncio
    async def test_drift_in_file_path_triggers_repair(self, patched, make_manager, caplog):
        SearchSession, LitloftSession = patched
        _seed_litloft_active(LitloftSession)
        _seed_indexed(
            SearchSession,
            file_path="/drives/drive1/旧/x.mp4",  # diverges from litloft
        )

        with caplog.at_level("WARNING"):
            manager = make_manager()
            result = await manager.reconcile()

        assert result.get("drift_repaired") == 1
        s = SearchSession()
        row = s.query(IndexedFile).filter_by(file_id="f1").first()
        assert row.file_path == "/drives/drive1/新/移動先/x.mp4"
        s.close()
        # Log out the count for ops visibility.
        assert any("drift_repaired" in r.message or "drift" in r.message.lower()
                   for r in caplog.records)

    @pytest.mark.asyncio
    async def test_drift_in_drive_triggers_repair(self, patched, make_manager):
        SearchSession, LitloftSession = patched
        _seed_litloft_active(LitloftSession, drive="drive2", file_path="x.mp4")
        _seed_indexed(SearchSession, drive="drive1", file_path="/drives/drive1/x.mp4")

        manager = make_manager()
        result = await manager.reconcile()

        assert result.get("drift_repaired") == 1
        s = SearchSession()
        row = s.query(IndexedFile).filter_by(file_id="f1").first()
        assert row.drive == "drive2"
        assert row.file_path == "/drives/drive2/x.mp4"
        s.close()

    @pytest.mark.asyncio
    async def test_drift_in_filename_triggers_repair(self, patched, make_manager):
        SearchSession, LitloftSession = patched
        _seed_litloft_active(LitloftSession, filename="new.mp4")
        _seed_indexed(SearchSession, filename="old.mp4")

        manager = make_manager()
        result = await manager.reconcile()

        assert result.get("drift_repaired") == 1
        s = SearchSession()
        row = s.query(IndexedFile).filter_by(file_id="f1").first()
        assert row.filename == "new.mp4"
        s.close()

    @pytest.mark.asyncio
    async def test_no_drift_when_in_sync(self, patched, make_manager):
        SearchSession, LitloftSession = patched
        _seed_litloft_active(LitloftSession, file_path="folder/x.mp4")
        _seed_indexed(SearchSession, file_path="/drives/drive1/folder/x.mp4")

        manager = make_manager()
        result = await manager.reconcile()

        assert result.get("drift_repaired") == 0

    @pytest.mark.asyncio
    async def test_inactive_files_skip_drift_check(self, patched, make_manager):
        """Trashed / missing files are deactivated by reconcile but not drift-checked.

        Drift in deleted_at-set rows is irrelevant: nothing serves them.
        """
        SearchSession, LitloftSession = patched
        _seed_litloft_active(
            LitloftSession,
            deleted_at="2026-01-01T00:00:00",
            file_path="新/x.mp4",
        )
        _seed_indexed(SearchSession, file_path="/drives/drive1/旧/x.mp4")

        manager = make_manager()
        result = await manager.reconcile()

        # File is deactivated, drift is not repaired (path stays stale)
        assert result["deactivated"] == 1
        assert result.get("drift_repaired", 0) == 0
        s = SearchSession()
        row = s.query(IndexedFile).filter_by(file_id="f1").first()
        assert row.active is False
        assert row.file_path == "/drives/drive1/旧/x.mp4"
        s.close()


class TestLoftTempAudioReset:
    def test_temp_audio_resets_even_when_loft_already_has_chunks(
        self, patched, tmp_path
    ):
        SearchSession, _ = patched
        loft = tmp_path / "movie.loft"
        temp_audio = tmp_path / "movie.stt_temp.m4a"
        loft.write_text("{}", encoding="utf-8")
        temp_audio.write_bytes(b"audio")

        _seed_indexed(
            SearchSession,
            file_id="loft1",
            mime_type="application/vnd.litloft.loft+json",
            file_path=str(loft),
            whisper_indexed=True,
        )
        s = SearchSession()
        try:
            s.add(
                TranscriptChunk(
                    file_id="loft1",
                    chunk_index=0,
                    text="existing vtt chunk",
                    language="ja",
                    timestamp_start=0.0,
                    timestamp_end=1.0,
                )
            )
            s.commit()
        finally:
            s.close()

        from app.indexer import IndexManager

        manager = IndexManager.__new__(IndexManager)
        manager._reset_loft_refs_with_new_vtt()

        s = SearchSession()
        try:
            row = s.query(IndexedFile).filter_by(file_id="loft1").one()
            assert row.whisper_indexed is False
        finally:
            s.close()


class TestReconcileKeepsEventLoopResponsive:
    """reconcile()'s synchronous DB work must not run on the event loop.

    A scan-complete webhook on a library of ~10k files used to hold the
    loop long enough for the watchdog to fire at 121s, making every
    endpoint unreachable and the container unhealthy.
    """

    @pytest.mark.asyncio
    async def test_slow_db_work_does_not_starve_the_loop(
        self, patched, make_manager, monkeypatch
    ):
        import app.indexer as indexer

        SearchSession, LitloftSession = patched
        _seed_litloft_active(LitloftSession)
        _seed_indexed(SearchSession, file_path="/drives/drive1/新/移動先/x.mp4")

        real = indexer._get_litloft_files

        def slow_read():
            time.sleep(0.3)
            return real()

        monkeypatch.setattr("app.indexer._get_litloft_files", slow_read)

        ticks = 0
        running = True

        async def ticker():
            nonlocal ticks
            while running:
                ticks += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(ticker())
        await asyncio.sleep(0)  # let the ticker reach its first await
        await make_manager().reconcile()
        running = False
        await task

        # Blocking inline would let through at most a tick or two.
        assert ticks > 5, (
            f"event loop advanced only {ticks} ticks during reconcile() — "
            "synchronous DB work is running on the loop"
        )
