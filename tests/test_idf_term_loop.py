"""SPEC-ADDON-021: the backfill loop — when passes run, and how it stops."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import time
from pathlib import Path

import pytest

from app import dependencies, tfidf
from app.models import TranscriptChunk
from tests import idf_backfill_api as api
from tests.idf_doubles import MarkerTokenizer
from tests.idf_term_db import counts, idx, no_children  # noqa: F401

WORDS = "共通語 " * 10
RARE = "珍語 " * 10


async def _until(predicate, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


@contextlib.asynccontextmanager
async def _loop():
    task = asyncio.create_task(api.run_loop())
    try:
        yield task
    finally:
        task.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(task, timeout=10)


@pytest.fixture()
def fast(monkeypatch):
    api.set_timings(monkeypatch, coalesce=0.05, retry=0.2)


@pytest.mark.asyncio
async def test_spec_addon_021_the_loop_runs_a_pass_at_startup(idx, fast):
    idx.add_file("k1", transcript=WORDS)

    async with _loop():
        assert await _until(lambda: idx.state("k1")[0])

    assert idx.terms("k1") == idx.term_set("k1")


@pytest.mark.asyncio
async def test_spec_addon_021_a_committed_change_is_written_without_another_trigger(idx, fast):
    from app.workers.whisper import _remove_whisper_data

    idx.add_file("k1", transcript=WORDS)

    async with _loop():
        assert await _until(lambda: idx.state("k1")[0])
        with idx.Session() as s:
            _remove_whisper_data(s, "k1")
            s.add(TranscriptChunk(
                file_id="k1", chunk_index=0, text=RARE,
                timestamp_start=0.0, timestamp_end=1.0,
            ))
            s.commit()
        assert await _until(lambda: idx.state("k1") == (True, 1))

    assert idx.terms("k1") == idx.term_set("k1")
    assert "共通語" not in idx.terms("k1")


@pytest.mark.asyncio
async def test_spec_addon_021_a_file_reactivated_by_reconcile_is_written_and_reconcile_does_not_wait(
    idx, monkeypatch,
):
    from tests.test_idf_term_marks import _manager

    api.set_timings(monkeypatch, coalesce=1.0, retry=0.2)
    idx.add_file("k1", transcript=WORDS, active=False)
    put = _litloft(monkeypatch, idx)
    put("k1")

    async with _loop():
        await asyncio.sleep(0.5)
        assert idx.state("k1") == (False, 0)
        await _manager().reconcile()
        assert idx.terms("k1") == set()
        assert await _until(lambda: idx.state("k1")[0])

    assert idx.terms("k1") == idx.term_set("k1")


def _litloft(monkeypatch, idx):
    from contextlib import contextmanager

    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(
        f"sqlite:///{Path(str(idx.engine.url.database)).parent / 'litloft.db'}",
        connect_args={"check_same_thread": False},
    )
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE files (id TEXT PRIMARY KEY, drive TEXT NOT NULL, "
            "file_path TEXT NOT NULL, filename TEXT NOT NULL, title TEXT, "
            "folder_path TEXT, description TEXT, file_size INTEGER, file_type TEXT, "
            "mime_type TEXT, duration REAL, deleted_at TEXT, missing_since TEXT)"
        ))
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _get_litloft_db():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    monkeypatch.setattr("app.database.get_litloft_db", _get_litloft_db)
    monkeypatch.setattr("app.indexer.get_litloft_db", _get_litloft_db)
    monkeypatch.setattr(
        "app.indexer.resolve_file_path", lambda drive, rel: f"/drives/{drive}/{rel}"
    )

    async def _open(drive, feature):
        return True

    monkeypatch.setattr("app.policy_client.is_feature_enabled", _open)

    def put(file_id):
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO files VALUES (:i, 'a', :n, :n, '', '', '', 1, 'video', "
                "'video/mp4', NULL, NULL, NULL)"
            ), {"i": file_id, "n": f"{file_id}.mp4"})

    return put


@pytest.mark.asyncio
async def test_spec_addon_021_requests_in_a_burst_become_one_pass(
    idx, monkeypatch, caplog,
):
    api.set_timings(monkeypatch, coalesce=1.0, retry=0.2)

    with caplog.at_level(logging.INFO):
        async with _loop():
            await asyncio.sleep(0.5)
            for fid in ("k1", "k2", "k3"):
                idx.add_file(fid, transcript=WORDS)
                api.request()
                await asyncio.sleep(0.1)
            assert await _until(lambda: all(idx.state(f)[0] for f in ("k1", "k2", "k3")))
            await asyncio.sleep(0.5)

    assert [c["written"] for c in counts(caplog)] == [3]


@pytest.mark.asyncio
async def test_spec_addon_021_a_request_during_a_pass_causes_one_more_pass(
    idx, fast, monkeypatch, caplog, tmp_path,
):
    monkeypatch.setattr(tfidf, "_TOKENIZER_FACTORY", MarkerTokenizer)
    gate = Path(tmp_path)
    idx.add_file("k1", transcript="GATEWORD " + WORDS)

    with caplog.at_level(logging.INFO):
        async with _loop():
            assert await _until(lambda: (gate / "entered").exists())
            assert len(multiprocessing.active_children()) <= 1
            idx.add_file("k2", transcript=RARE)
            api.request()
            idx.add_file("k3", transcript=RARE)
            api.request()
            (gate / "release").touch()
            assert await _until(lambda: all(idx.state(f)[0] for f in ("k1", "k2", "k3")))
            await asyncio.sleep(0.5)

    assert [c["written"] for c in counts(caplog)] == [1, 2]


@pytest.mark.asyncio
async def test_spec_addon_021_a_file_skipped_because_it_changed_is_written_by_a_later_pass(
    idx, fast, monkeypatch, tmp_path,
):
    monkeypatch.setattr(tfidf, "_TOKENIZER_FACTORY", MarkerTokenizer)
    gate = Path(tmp_path)
    idx.add_file("k1", transcript="GATEWORD " + WORDS)

    async with _loop():
        assert await _until(lambda: (gate / "entered").exists())
        idx.sql("UPDATE transcript_chunks SET text = :t WHERE file_id = 'k1'", t=RARE)
        idx.set_state("k1", indexed=False, gen=1)
        (gate / "release").touch()
        assert await _until(lambda: idx.state("k1") == (True, 1))

    assert idx.terms("k1") == idx.term_set("k1")


@pytest.mark.asyncio
async def test_spec_addon_021_a_write_that_raised_is_retried_by_a_later_pass(idx, fast):
    idx.add_file("k1", transcript=WORDS)
    idx.sql("CREATE TABLE fail_writes (on_ INTEGER)")
    idx.sql("INSERT INTO fail_writes VALUES (1)")
    idx.sql(
        "CREATE TRIGGER fail_k1 BEFORE INSERT ON idf_doc_terms "
        "WHEN (SELECT on_ FROM fail_writes) = 1 "
        "BEGIN SELECT RAISE(ABORT, 'write failed'); END"
    )

    async with _loop():
        await asyncio.sleep(1.0)
        assert idx.state("k1") == (False, 0)
        idx.sql("UPDATE fail_writes SET on_ = 0")
        assert await _until(lambda: idx.state("k1")[0])

    assert idx.terms("k1") == idx.term_set("k1")


@pytest.mark.asyncio
async def test_spec_addon_021_a_pass_that_raises_is_logged_and_the_loop_continues(
    idx, fast, caplog,
):
    idx.add_file("k1", transcript=WORDS)
    idx.sql("ALTER TABLE indexed_files RENAME TO indexed_files_away")

    with caplog.at_level(logging.INFO):
        async with _loop() as task:
            assert await _until(
                lambda: any(r.levelno == logging.ERROR for r in caplog.records)
            )
            assert not task.done()
            idx.sql("ALTER TABLE indexed_files_away RENAME TO indexed_files")
            assert await _until(lambda: idx.state("k1")[0])


@pytest.mark.asyncio
async def test_spec_addon_021_a_paused_queue_holds_the_backfill(idx, fast):
    manager = dependencies.get_index_manager()
    manager.pause()
    idx.add_file("k1", transcript=WORDS)

    async with _loop():
        await asyncio.sleep(1.5)
        assert idx.state("k1") == (False, 0)
        manager.resume()
        assert await _until(lambda: idx.state("k1")[0])


@pytest.mark.asyncio
async def test_spec_addon_021_cancelling_the_loop_terminates_its_child(
    idx, fast, monkeypatch, tmp_path,
):
    monkeypatch.setattr(tfidf, "_TOKENIZER_FACTORY", MarkerTokenizer)
    gate = Path(tmp_path)
    idx.add_file("k1", transcript="GATEWORD " + WORDS)

    task = asyncio.create_task(api.run_loop())
    assert await _until(lambda: (gate / "entered").exists())
    assert multiprocessing.active_children()

    task.cancel()
    with contextlib.suppress(BaseException):
        await asyncio.wait_for(task, timeout=5)

    assert no_children(timeout=5)
    assert idx.state("k1") == (False, 0)
    assert idx.terms("k1") == set()
