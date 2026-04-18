"""Drive-level purge: bulk removal triggered by per-drive policy off."""

import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
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

from app import policy_client, purge


@pytest.fixture(autouse=True)
def _reset_policy_cache():
    policy_client.reset_cache()
    yield
    policy_client.reset_cache()


def _stub_search_db(file_ids_for_drive: dict[str, list[str]], monkeypatch):
    """Patch app.purge.get_search_db with an in-memory dict-backed stub.

    Captures which file_ids the SUT discovered per drive and which it
    asked _purge_file to delete, without standing up a real SQLite DB.
    """
    asked_drive: dict[str, str] = {}

    class _Query:
        def __init__(self_inner, payload):
            self_inner._payload = payload
            self_inner._drive_filter: str | None = None
            self_inner._distinct = False

        def filter(self_inner, *args, **kwargs):
            # SQLAlchemy filter API: the test sees something like
            # IndexedFile.drive == "work". The stub just records the
            # rhs string by introspecting the BinaryExpression.
            for arg in args:
                rhs = getattr(arg, "right", None)
                value = getattr(rhs, "value", None) if rhs is not None else None
                if isinstance(value, str):
                    self_inner._drive_filter = value
            return self_inner

        def distinct(self_inner):
            self_inner._distinct = True
            return self_inner

        def all(self_inner):
            if self_inner._distinct:
                return [SimpleNamespace(drive=d) for d in self_inner._payload]
            ids = self_inner._payload.get(self_inner._drive_filter, [])
            return [SimpleNamespace(file_id=fid) for fid in ids]

    class _Session:
        def query(self_inner, _col):
            return _Query(file_ids_for_drive)

    @contextmanager
    def _get_search_db():
        yield _Session()

    monkeypatch.setattr(purge, "get_search_db", _get_search_db, raising=False)
    # Also patch the late-imported version inside purge_drive / purge_disabled_drives
    import app.database as database_module
    monkeypatch.setattr(database_module, "get_search_db", _get_search_db)
    return asked_drive


def test_purge_drive_no_files_returns_zero(monkeypatch):
    _stub_search_db({"work": []}, monkeypatch)
    deleted: list[str] = []
    monkeypatch.setattr("app.indexer._purge_file", lambda fid: deleted.append(fid))
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)

    assert purge.purge_drive("work") == 0
    assert deleted == []


def test_purge_drive_calls_per_file(monkeypatch):
    _stub_search_db({"work": ["a", "b", "c"]}, monkeypatch)
    deleted: list[str] = []
    monkeypatch.setattr("app.indexer._purge_file", lambda fid: deleted.append(fid))
    invalidated = MagicMock()
    monkeypatch.setattr("app.search.invalidate_similar_cache", invalidated)

    assert purge.purge_drive("work") == 3
    assert deleted == ["a", "b", "c"]
    invalidated.assert_called_once()


def test_purge_drive_continues_on_per_file_failure(monkeypatch):
    _stub_search_db({"work": ["a", "b", "c"]}, monkeypatch)
    deleted: list[str] = []

    def boom_on_b(fid):
        if fid == "b":
            raise RuntimeError("boom")
        deleted.append(fid)

    monkeypatch.setattr("app.indexer._purge_file", boom_on_b)
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)

    # 2 succeeded, b failed → total purged = 2
    assert purge.purge_drive("work") == 2
    assert deleted == ["a", "c"]


@pytest.mark.asyncio
async def test_purge_disabled_drives_only_purges_off(monkeypatch):
    _stub_search_db({"work": ["w1"], "private": ["p1"]}, monkeypatch)

    async def fake_policy(drive, feature):
        # work is off, private stays on
        return drive != "work"

    monkeypatch.setattr(purge, "is_feature_enabled", fake_policy, raising=False)
    monkeypatch.setattr(policy_client, "is_feature_enabled", fake_policy)

    purged: list[str] = []
    monkeypatch.setattr("app.indexer._purge_file", lambda fid: purged.append(fid))
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)

    result = await purge.purge_disabled_drives()
    assert result == {"work": 1}
    assert purged == ["w1"]


@pytest.mark.asyncio
async def test_purge_disabled_drives_skips_on_policy_lookup_failure(monkeypatch):
    """A policy lookup failure must not silently destroy data."""
    _stub_search_db({"work": ["w1"]}, monkeypatch)

    async def boom(drive, feature):
        raise RuntimeError("network down")

    monkeypatch.setattr(policy_client, "is_feature_enabled", boom)

    purged: list[str] = []
    monkeypatch.setattr("app.indexer._purge_file", lambda fid: purged.append(fid))
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)

    result = await purge.purge_disabled_drives()
    assert result == {}
    assert purged == []


# ---------------------------------------------------------------------------
# Detailed-summary cleanup on drive purge (regression coverage)
# ---------------------------------------------------------------------------


def test_purge_file_also_wipes_detailed_summary_columns(tmp_path, monkeypatch):
    """``_purge_file`` deletes the whole ``file_summaries`` row.

    Since detailed-summary state lives as columns on the same row
    (``detailed_summary``, ``detailed_status``, etc.), per-file purge
    already removes them — no additional cleanup is needed when the
    host flips ``features.index`` off for a drive. This test guards
    against a future refactor splitting detailed into its own table
    without updating the purge path.
    """
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    from app.database import (
        Base,
        _create_detailed_summary_citations_table,
        _create_file_summaries_table,
    )
    from app.models import IndexedFile

    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _create_file_summaries_table(conn)
        # Phase 1: _purge_file now also wipes citations, so the table
        # must exist even for tests that never insert citations.
        _create_detailed_summary_citations_table(conn)
        # suggested_tags is a raw-SQL table like file_summaries.
        from app.database import _create_suggested_tags_table
        _create_suggested_tags_table(conn)
        # FTS mirrors touched by _purge_file. sqlite-vec is unavailable
        # in the vanilla test image, so create FTS5 tables directly
        # without going through _create_vec_tables (which also tries
        # the vec0 extension).
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_files "
            "USING fts5(file_id, filename, title, description, "
            "tags_text, tokenize='trigram')"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_transcripts "
            "USING fts5(file_id, chunk_index, text, "
            "tokenize='trigram')"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content "
            "USING fts5(file_id, chunk_index, page, text, "
            "tokenize='trigram')"
        ))

    Session = sessionmaker(bind=engine, expire_on_commit=False)

    seed = Session()
    try:
        seed.add(
            IndexedFile(
                file_id="abc123",
                drive="work",
                filename="video.mp4",
                file_path="/drives/work/video.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=100,
                active=True,
            )
        )
        seed.commit()
    finally:
        seed.close()

    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO file_summaries "
                "(file_id, short_summary, long_summary, model, "
                "context_type, context_chars, was_truncated, status, "
                "created_at, detailed_summary, detailed_status, "
                "detailed_model, detailed_generated_at, "
                "detailed_context_chars, detailed_was_truncated) "
                "VALUES (:fid, 's', 'l', 'm', 'video', 100, 0, "
                "'generated', :now, 'detailed body', 'generated', "
                "'m', :now, 100, 0)"
            ),
            {"fid": "abc123", "now": now},
        )

    @contextmanager
    def _get_search_db():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    monkeypatch.setattr("app.indexer.get_search_db", _get_search_db)
    # invalidate_similar_cache is called from purge_drive, not _purge_file,
    # but stub defensively so this test remains robust to refactors.
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)

    from app.indexer import _purge_file

    _purge_file("abc123")

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT 1 FROM file_summaries WHERE file_id = :fid"
            ),
            {"fid": "abc123"},
        ).fetchone()

    assert row is None, (
        "file_summaries row must be deleted; detailed_* columns go with it"
    )
