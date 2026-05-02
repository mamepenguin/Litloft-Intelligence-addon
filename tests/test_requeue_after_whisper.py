"""Re-enqueue summaries / auto_tags after WHISPER completion.

Closes the race where METADATA-driven enqueue runs the LLM workers
before TranscriptChunk rows exist — the workers ``insufficient_content``
silent-return without writing a ``file_summaries`` / ``suggested_tags``
row, leaving the file stuck until the next intelligence restart sweep.

The hook lives on ``IndexManager.requeue_after_whisper`` and is invoked
by ``_whisper_worker`` after ``index_whisper`` succeeds. Conditions
match the existing ``enqueue_unprocessed`` sweep so user-deleted
summaries / tags are not regenerated.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

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
from app.models import IndexedFile  # noqa: E402,F401


@pytest.fixture()
def engine(tmp_path):
    """Search DB with indexed_files + file_summaries + suggested_tags."""
    db_path = tmp_path / "search.db"
    eng = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(eng)
    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS suggested_tags ("
            "  file_id TEXT PRIMARY KEY,"
            "  tags TEXT NOT NULL,"
            "  model TEXT NOT NULL,"
            "  context_type TEXT NOT NULL,"
            "  created_at TEXT NOT NULL,"
            "  status TEXT NOT NULL DEFAULT 'pending'"
            ")"
        ))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS file_summaries ("
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
            "  detailed_status TEXT,"
            "  detailed_error TEXT,"
            "  visual_description TEXT,"
            "  visual_description_generated_at TEXT,"
            "  visual_description_model TEXT,"
            "  visual_description_status TEXT"
            ")"
        ))
    return eng


@pytest.fixture()
def patched(monkeypatch, engine):
    """Wire get_search_db to the test engine."""
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _get_search_db():
        s = Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    monkeypatch.setattr("app.indexer.get_search_db", _get_search_db)
    return Session


@pytest.fixture()
def features(monkeypatch):
    """Replace ``app.indexer.settings`` with a mutable proxy.

    ``FeaturesConfig`` and ``Settings`` are frozen dataclasses, so we
    swap the whole module-level ``settings`` reference for a
    ``SimpleNamespace`` whose ``features`` attribute is mutable.
    Tests then mutate ``features.summaries`` etc. directly to exercise
    different config combinations without touching the real Settings.
    """
    from types import SimpleNamespace

    from app.config import settings as real_settings

    proxy_features = SimpleNamespace(
        indexing=real_settings.features.indexing,
        search=real_settings.features.search,
        summaries="on_index",
        detailed_summaries="false",
        auto_tags="on_index",
        rag=real_settings.features.rag,
    )
    proxy_settings = SimpleNamespace(
        features=proxy_features,
        workers=real_settings.workers,
        indexing=real_settings.indexing,
    )
    monkeypatch.setattr("app.indexer.settings", proxy_settings)
    return proxy_features


@pytest.fixture()
def workers():
    """Mock summaries_worker / auto_tags_worker with async enqueue."""
    summaries_worker = MagicMock()
    summaries_worker.enqueue = AsyncMock()
    auto_tags_worker = MagicMock()
    auto_tags_worker.enqueue = AsyncMock()
    return summaries_worker, auto_tags_worker


@pytest.fixture()
def manager(features, workers):
    """An IndexManager wired with mock workers (settings already proxied)."""
    from app.indexer import IndexManager

    summaries_worker, auto_tags_worker = workers
    return IndexManager(
        auto_tags_worker=auto_tags_worker,
        summaries_worker=summaries_worker,
    )


def _seed_indexed(
    Session,
    file_id="f1",
    *,
    metadata_indexed=True,
    whisper_indexed=True,
    active=True,
    mime_type="video/mp4",
):
    s = Session()
    s.add(IndexedFile(
        file_id=file_id,
        drive="d1",
        filename="x.mp4",
        file_path="/drives/d1/x.mp4",
        file_type="video",
        mime_type=mime_type,
        file_size=1000,
        active=active,
        metadata_indexed=metadata_indexed,
        clip_indexed=False,
        whisper_indexed=whisper_indexed,
        text_indexed=False,
        title="x",
        description="",
        tags_text="",
    ))
    s.commit()
    s.close()


def _seed_summary(Session, file_id="f1"):
    s = Session()
    s.execute(text(
        "INSERT INTO file_summaries "
        "(file_id, short_summary, long_summary, model, context_type, "
        " context_chars, created_at) "
        "VALUES (:fid, 's', 'l', 'm', 'video', 100, '2026-05-03T00:00:00')"
    ), {"fid": file_id})
    s.commit()
    s.close()


def _seed_tags(Session, file_id="f1"):
    s = Session()
    s.execute(text(
        "INSERT INTO suggested_tags "
        "(file_id, tags, model, context_type, created_at) "
        "VALUES (:fid, '[]', 'm', 'video', '2026-05-03T00:00:00')"
    ), {"fid": file_id})
    s.commit()
    s.close()


class TestRequeueAfterWhisper:
    @pytest.mark.asyncio
    async def test_basic_enqueues_both_workers(
        self, patched, manager, workers
    ):
        """metadata=1, whisper=1, no summary/tags rows → both workers called."""
        _seed_indexed(patched)
        summaries_worker, auto_tags_worker = workers

        await manager.requeue_after_whisper("f1")

        summaries_worker.enqueue.assert_awaited_once_with("f1")
        auto_tags_worker.enqueue.assert_awaited_once_with("f1")

    @pytest.mark.asyncio
    async def test_skips_summaries_when_row_exists(
        self, patched, manager, workers
    ):
        """User-deleted/regenerated semantics: existing file_summaries → skip."""
        _seed_indexed(patched)
        _seed_summary(patched)
        summaries_worker, auto_tags_worker = workers

        await manager.requeue_after_whisper("f1")

        summaries_worker.enqueue.assert_not_awaited()
        # auto_tags still enqueued because suggested_tags is empty
        auto_tags_worker.enqueue.assert_awaited_once_with("f1")

    @pytest.mark.asyncio
    async def test_skips_auto_tags_when_row_exists(
        self, patched, manager, workers
    ):
        """Existing suggested_tags row → auto_tags skipped, summaries still enqueued."""
        _seed_indexed(patched)
        _seed_tags(patched)
        summaries_worker, auto_tags_worker = workers

        await manager.requeue_after_whisper("f1")

        summaries_worker.enqueue.assert_awaited_once_with("f1")
        auto_tags_worker.enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_metadata_not_indexed(
        self, patched, manager, workers
    ):
        """Defensive: metadata_indexed=False → no enqueue (race protection)."""
        _seed_indexed(patched, metadata_indexed=False)
        summaries_worker, auto_tags_worker = workers

        await manager.requeue_after_whisper("f1")

        summaries_worker.enqueue.assert_not_awaited()
        auto_tags_worker.enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_file_inactive(
        self, patched, manager, workers
    ):
        """Trashed/missing file (active=False) → no enqueue."""
        _seed_indexed(patched, active=False)
        summaries_worker, auto_tags_worker = workers

        await manager.requeue_after_whisper("f1")

        summaries_worker.enqueue.assert_not_awaited()
        auto_tags_worker.enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_features_disabled(
        self, patched, features, manager, workers
    ):
        """All features=false → no enqueue (manual / on_index off)."""
        features.summaries = "false"
        features.detailed_summaries = "false"
        features.auto_tags = "false"
        _seed_indexed(patched)
        summaries_worker, auto_tags_worker = workers

        await manager.requeue_after_whisper("f1")

        summaries_worker.enqueue.assert_not_awaited()
        auto_tags_worker.enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_summaries_enqueued_when_only_detailed_on_index(
        self, patched, features, manager, workers
    ):
        """detailed_summaries=on_index alone routes through summaries_worker.

        Per the per-layer worker design (hako 5a9HNw29fsRuL_rKuKwcW),
        SummariesWorker.enqueue accepts files when either short/long or
        detailed is enabled. The hook must wake it up in that case too.
        """
        features.summaries = "false"
        features.detailed_summaries = "on_index"
        features.auto_tags = "false"
        _seed_indexed(patched)
        summaries_worker, auto_tags_worker = workers

        await manager.requeue_after_whisper("f1")

        summaries_worker.enqueue.assert_awaited_once_with("f1")
        auto_tags_worker.enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_workers_not_wired(
        self, patched, features
    ):
        """No workers passed (e.g. early startup) → no AttributeError."""
        from app.indexer import IndexManager

        m = IndexManager(auto_tags_worker=None, summaries_worker=None)
        _seed_indexed(patched)

        # Should not raise.
        await m.requeue_after_whisper("f1")

    @pytest.mark.asyncio
    async def test_unknown_file_id_is_noop(
        self, patched, manager, workers
    ):
        """Unknown file_id → silent skip, no enqueue, no exception."""
        # No seed.
        summaries_worker, auto_tags_worker = workers

        await manager.requeue_after_whisper("ghost")

        summaries_worker.enqueue.assert_not_awaited()
        auto_tags_worker.enqueue.assert_not_awaited()
