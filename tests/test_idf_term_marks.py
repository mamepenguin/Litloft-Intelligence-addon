"""SPEC-ADDON-021: which writes mark a file's term set stale, the backfill request
they send after commit, purge, and the startup rule-version check."""

from __future__ import annotations

import contextlib
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.models import JobRecord, TranscriptChunk
from app.workers import refine
from tests import idf_backfill_api as api
from tests.idf_term_db import idx  # noqa: F401
from tests.test_refine_fts_reindex import (
    _FAILING_TRANSCRIPT_DDL,
    FILE,
    _llm,
    _make_db,
    _patch_job,
    _seed_file,
)

WORDS = "共通語 " * 10


# ---------------------------------------------------------------------------
# Whisper: _remove_whisper_data, shared by transcription, the VTT path and
# the startup orphan cleanup
# ---------------------------------------------------------------------------


def test_spec_addon_021_removing_whisper_data_marks_the_file_and_requests_after_commit(idx):
    from app.workers.whisper import _remove_whisper_data

    idx.add_file("k1", transcript=WORDS, indexed=True, gen=4)
    api.reset_state()

    with idx.Session() as s:
        _remove_whisper_data(s, "k1")
        s.add(TranscriptChunk(
            file_id="k1", chunk_index=0, text="新しい話 " * 10,
            timestamp_start=0.0, timestamp_end=1.0,
        ))
        assert not api.request_pending()
        s.commit()

    assert idx.state("k1") == (False, 5)
    assert api.request_pending()


def test_spec_addon_021_a_rolled_back_whisper_removal_changes_nothing_and_requests_nothing(idx):
    from app.workers.whisper import _remove_whisper_data

    idx.add_file("k1", transcript=WORDS, indexed=True, gen=4)
    idx.put_terms("k1", {"共通語"})
    api.reset_state()

    with idx.Session() as s:
        _remove_whisper_data(s, "k1")
        s.rollback()

    assert idx.state("k1") == (True, 4)
    assert idx.terms("k1") == {"共通語"}
    assert not api.request_pending()


def test_spec_addon_021_the_orphan_cleanup_marks_the_file_even_when_the_removal_raises(idx):
    from app.workers.whisper import fail_orphaned_running_jobs

    idx.add_file("k1", transcript=WORDS, indexed=True, gen=2)
    with idx.Session() as s:
        s.add(JobRecord(
            file_id="k1", job_kind="transcription", provider="whisper_local",
            status="running",
        ))
        s.commit()
    idx.sql(
        "CREATE TRIGGER keep_chunks BEFORE DELETE ON transcript_chunks "
        "BEGIN SELECT RAISE(ABORT, 'delete failed'); END"
    )
    api.reset_state()

    with contextlib.suppress(Exception):
        fail_orphaned_running_jobs()

    assert idx.rows("SELECT count(*) FROM transcript_chunks WHERE file_id = 'k1'") == [(1,)]
    assert idx.state("k1") == (False, 3)
    assert api.request_pending()


# ---------------------------------------------------------------------------
# Refine: the per-window write and the re-chunk
# ---------------------------------------------------------------------------


def _set_flag(Session, file_id, indexed, gen):
    with Session() as s:
        s.execute(text(
            "UPDATE indexed_files SET idf_terms_indexed = :i, idf_terms_gen = :g "
            "WHERE file_id = :f"
        ), {"i": indexed, "g": gen, "f": file_id})
        s.commit()


def _flag(Session, file_id):
    with Session() as s:
        row = s.execute(text(
            "SELECT idf_terms_indexed, idf_terms_gen FROM indexed_files WHERE file_id = :f"
        ), {"f": file_id}).one()
    return bool(row[0]), int(row[1])


@pytest.mark.asyncio
async def test_spec_addon_021_a_refine_window_write_marks_the_file(tmp_path, monkeypatch):
    _engine, Session = _make_db(tmp_path)
    with Session() as s:
        _seed_file(s, FILE, [(1, 0, "we deployed it on cooper netties", 0.0, 4.0)], words=False)
        s.commit()
    _set_flag(Session, FILE, True, 7)
    api.reset_state()
    _patch_job(monkeypatch, Session, realign=lambda *a, **k: 0, rechunk=lambda *a, **k: [])

    await refine._run_refine_job(
        FILE, "job1", [1], _llm([(1, "We deployed it on Kubernetes.")])
    )

    indexed, gen = _flag(Session, FILE)
    assert indexed is False
    assert gen > 7
    assert api.request_pending()
    api.reset_state()


@pytest.mark.asyncio
async def test_spec_addon_021_a_rolled_back_refine_window_leaves_the_flag(tmp_path, monkeypatch):
    _engine, Session = _make_db(tmp_path, transcript_ddl=_FAILING_TRANSCRIPT_DDL)
    with Session() as s:
        _seed_file(s, FILE, [(1, 0, "segment one said cooper netties", 0.0, 4.0)])
        s.commit()
    _set_flag(Session, FILE, True, 7)
    api.reset_state()
    _patch_job(monkeypatch, Session, rechunk=lambda *a, **k: [])

    await refine._run_refine_job(FILE, "job1", [1], _llm([(1, "BOOM segment one.")]))

    assert _flag(Session, FILE) == (True, 7)
    assert not api.request_pending()


def _words_db(tmp_path):
    _engine, Session = _make_db(tmp_path)
    with Session() as s:
        _seed_file(s, FILE, [(1, 0, "alpha beta gamma delta", 0.0, 4.0)])
        s.commit()
    _set_flag(Session, FILE, True, 2)
    api.reset_state()
    return Session


def test_spec_addon_021_a_committed_rechunk_marks_the_file(tmp_path):
    Session = _words_db(tmp_path)

    with Session() as s:
        refine.rechunk_from_words(s, FILE, refined_model="m")
        assert not api.request_pending()
        s.commit()

    assert _flag(Session, FILE) == (False, 3)
    assert api.request_pending()
    api.reset_state()


def test_spec_addon_021_a_rolled_back_rechunk_leaves_the_flag(tmp_path):
    Session = _words_db(tmp_path)

    with Session() as s:
        refine.rechunk_from_words(s, FILE, refined_model="m")
        s.rollback()

    assert _flag(Session, FILE) == (True, 2)
    assert not api.request_pending()


# ---------------------------------------------------------------------------
# Moves and reconcile
# ---------------------------------------------------------------------------


@pytest.fixture()
def litloft(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'litloft.db'}",
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

    def put(file_id, *, drive="a", file_path="k1.mp4", filename="k1.mp4", deleted_at=None):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM files WHERE id = :i"), {"i": file_id})
            conn.execute(text(
                "INSERT INTO files (id, drive, file_path, filename, title, folder_path, "
                "description, file_size, file_type, mime_type, duration, deleted_at, "
                "missing_since) VALUES (:i, :d, :p, :n, '', '', '', 1, 'video', "
                "'video/mp4', NULL, :del, NULL)"
            ), {"i": file_id, "d": drive, "p": file_path, "n": filename, "del": deleted_at})

    return put


def _manager():
    from app.indexer import IndexManager

    m = IndexManager()
    m._reset_loft_refs_with_new_vtt = lambda: None

    async def _no_resume():
        return 0

    m._resume_incomplete = _no_resume
    return m


@pytest.mark.asyncio
async def test_spec_addon_021_a_move_that_renames_the_file_marks_it(idx, litloft):
    idx.add_file("k1", filename="k1.mp4", transcript=WORDS, indexed=True, gen=1)
    idx.put_terms("k1", {"共通語"})
    litloft("k1", file_path="sub/新しい名前.mp4", filename="新しい名前.mp4")
    api.reset_state()

    await _manager().handle_files_moved(["k1"])

    assert idx.state("k1") == (False, 2)
    assert idx.terms("k1") == {"共通語"}
    assert api.request_pending()


@pytest.mark.asyncio
@pytest.mark.parametrize("drive", ["a", "b"])
async def test_spec_addon_021_a_move_that_keeps_the_name_changes_nothing(idx, litloft, drive):
    idx.add_file("k1", filename="k1.mp4", transcript=WORDS, indexed=True, gen=1)
    idx.put_terms("k1", {"共通語"})
    litloft("k1", drive=drive, file_path="other/folder/k1.mp4", filename="k1.mp4")
    api.reset_state()

    await _manager().handle_files_moved(["k1"])

    assert idx.rows("SELECT drive FROM indexed_files WHERE file_id = 'k1'") == [(drive,)]
    assert idx.state("k1") == (True, 1)
    assert idx.terms("k1") == {"共通語"}
    assert not api.request_pending()


@pytest.mark.asyncio
async def test_spec_addon_021_reconcile_deactivation_and_reactivation_keep_flag_gen_and_rows(
    idx, litloft,
):
    idx.add_file("k1", filename="k1.mp4", transcript=WORDS, indexed=True, gen=6)
    idx.put_terms("k1", {"共通語"})
    litloft("k1", deleted_at="2026-10-01T00:00:00")

    await _manager().reconcile()
    assert idx.rows("SELECT active FROM indexed_files WHERE file_id = 'k1'") == [(0,)]
    assert idx.state("k1") == (True, 6)
    assert idx.terms("k1") == {"共通語"}

    litloft("k1")
    await _manager().reconcile()
    assert idx.rows("SELECT active FROM indexed_files WHERE file_id = 'k1'") == [(1,)]
    assert idx.state("k1") == (True, 6)
    assert idx.terms("k1") == {"共通語"}


@pytest.mark.asyncio
async def test_spec_addon_021_reconcile_requests_a_pass_when_it_finishes(idx, litloft):
    idx.add_file("k1", filename="k1.mp4", indexed=True, gen=0)
    litloft("k1")
    api.reset_state()

    await _manager().reconcile()

    assert api.request_pending()


@pytest.mark.asyncio
async def test_spec_addon_021_reconcile_requests_a_pass_even_when_a_step_raises(
    idx, litloft, monkeypatch,
):
    def _boom():
        raise RuntimeError("core DB unreadable")

    monkeypatch.setattr("app.indexer._get_litloft_files", _boom)
    api.reset_state()

    with contextlib.suppress(Exception):
        await _manager().reconcile()

    assert api.request_pending()


# ---------------------------------------------------------------------------
# Purge
# ---------------------------------------------------------------------------


def _purge_ready(idx, monkeypatch):
    from app.database import (
        _create_detailed_summary_citations_table,
        _create_file_summaries_table,
        _create_suggested_chapters_table,
        _create_suggested_tags_table,
    )

    with idx.engine.begin() as conn:
        _create_file_summaries_table(conn)
        _create_detailed_summary_citations_table(conn)
        _create_suggested_tags_table(conn)
        _create_suggested_chapters_table(conn)
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)
    idx.add_file("a1", transcript=WORDS)
    idx.add_file("a2", transcript=WORDS)
    idx.add_file("b1", drive="b", transcript=WORDS)
    for fid in ("a1", "a2", "b1"):
        idx.put_terms(fid, {"共通語", "珍語"})


def test_spec_addon_021_purging_a_file_removes_its_term_rows(idx, monkeypatch):
    from app.indexer import _purge_file

    _purge_ready(idx, monkeypatch)

    _purge_file("a1")

    assert idx.terms("a1") == set()
    assert idx.terms("a2") == {"共通語", "珍語"}


def test_spec_addon_021_purging_a_drive_removes_the_term_rows_of_its_files(idx, monkeypatch):
    from app.purge import purge_drive

    _purge_ready(idx, monkeypatch)

    purge_drive("a")

    assert idx.terms("a1") == set()
    assert idx.terms("a2") == set()
    assert idx.terms("b1") == {"共通語", "珍語"}


# ---------------------------------------------------------------------------
# The startup rule-version check
# ---------------------------------------------------------------------------


def _meta(idx):
    return idx.rows("SELECT value FROM index_meta WHERE key = 'idf_term_rules'")


def _seed_indexed(idx):
    idx.add_file("k1", transcript=WORDS, indexed=True, gen=3)
    idx.add_file("k2", transcript=WORDS, indexed=True, gen=1, active=False)
    idx.put_terms("k1", {"共通語"})
    idx.sql("UPDATE indexed_files SET tfidf_keywords_indexed = 1")


@pytest.mark.parametrize("recorded", [None, "an-older-version"])
def test_spec_addon_021_a_new_rule_version_flags_every_file_and_keeps_rows(idx, recorded):
    _seed_indexed(idx)
    if recorded is not None:
        idx.sql("CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT)")
        idx.sql("INSERT INTO index_meta VALUES ('idf_term_rules', :v)", v=recorded)

    api.check_rules()

    assert idx.state("k1")[0] is False
    assert idx.state("k2")[0] is False
    assert idx.terms("k1") == {"共通語"}
    assert _meta(idx) == [(api.rules_version(),)]
    assert idx.rows("SELECT DISTINCT tfidf_keywords_indexed FROM indexed_files") == [(1,)]


def test_spec_addon_021_a_matching_rule_version_changes_nothing(idx):
    _seed_indexed(idx)
    idx.sql("CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT)")
    idx.sql("INSERT INTO index_meta VALUES ('idf_term_rules', :v)", v=api.rules_version())

    api.check_rules()

    assert idx.state("k1") == (True, 3)
    assert idx.state("k2") == (True, 1)
    assert _meta(idx) == [(api.rules_version(),)]


def test_spec_addon_021_a_failing_rule_version_check_changes_neither_flags_nor_version(idx):
    assert api.rules_version() != "an-older-version"
    _seed_indexed(idx)
    idx.sql(
        "CREATE TABLE index_meta (key TEXT PRIMARY KEY, "
        "value TEXT CHECK (value = 'an-older-version'))"
    )
    idx.sql("INSERT INTO index_meta VALUES ('idf_term_rules', 'an-older-version')")

    with contextlib.suppress(Exception):
        api.check_rules()

    assert idx.state("k1") == (True, 3)
    assert idx.state("k2") == (True, 1)
    assert _meta(idx) == [("an-older-version",)]
