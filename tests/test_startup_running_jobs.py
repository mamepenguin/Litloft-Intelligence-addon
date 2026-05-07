"""Phase 1C tests: startup hook fails orphaned ``running`` JobRecords.

Spec ``2026-05-07-cloud-transcription-providers.md`` §"Hot-swap /
半端ジョブ → startup hook 実装". When the intelligence container
restarts mid-job, any ``status='running'`` rows are inherently
orphaned — no worker is going to finish them. The startup hook flips
them to ``status='failed'`` with ``error_class='ContainerRestart'``
and cleans the partial chunk writes via ``_remove_whisper_data``.

Phase 1 explicitly assumes a single intelligence container; the
multi-container variant (worker_id column + per-worker hook) is
deferred to Phase 2 alongside VibeVoice's independent service.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
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
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import Base  # noqa: E402
from app.models import IndexedFile, JobRecord  # noqa: E402


@pytest.fixture()
def engine_with_jobs(tmp_path, monkeypatch):
    db_path = tmp_path / "search.db"
    eng = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(eng)
    # FTS5 mirrors stubbed for _remove_whisper_data side-effects.
    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS fts_transcripts ("
            "  file_id TEXT, chunk_index INTEGER, text TEXT)"
        ))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS fts_transcripts_word ("
            "  file_id TEXT, chunk_index INTEGER, text TEXT)"
        ))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS vec_text ("
            "  embedding_id TEXT PRIMARY KEY, vector BLOB)"
        ))
    Session = sessionmaker(bind=eng, expire_on_commit=False)

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
    monkeypatch.setattr("app.workers.whisper.get_search_db", _get_search_db)
    return Session


def _seed(Session, *, file_id="f00000000001"):
    s = Session()
    try:
        s.add(IndexedFile(
            file_id=file_id,
            drive="d1",
            filename="x.mp4",
            file_path="/drives/d1/x.mp4",
            file_type="video",
            mime_type="video/mp4",
            file_size=1,
            active=True,
        ))
        s.commit()
    finally:
        s.close()


def test_running_records_flipped_to_failed(engine_with_jobs) -> None:
    """A ``status='running'`` row at startup must be flipped to failed."""
    from app.workers.whisper import fail_orphaned_running_jobs

    _seed(engine_with_jobs)
    s = engine_with_jobs()
    try:
        s.add(JobRecord(
            file_id="f00000000001",
            job_kind="transcription",
            provider="deepgram",
            status="running",
        ))
        s.commit()
    finally:
        s.close()

    fail_orphaned_running_jobs()

    s = engine_with_jobs()
    try:
        record = s.query(JobRecord).filter_by(file_id="f00000000001").one()
        assert record.status == "failed"
        assert record.error_class == "ContainerRestart"
        assert record.completed_at is not None
        assert "restarted" in (record.error_message or "").lower()
    finally:
        s.close()


def test_succeeded_records_left_alone(engine_with_jobs) -> None:
    """The hook only touches ``running`` rows — succeeded must not be flipped."""
    from app.workers.whisper import fail_orphaned_running_jobs

    _seed(engine_with_jobs)
    s = engine_with_jobs()
    try:
        s.add(JobRecord(
            file_id="f00000000001",
            job_kind="transcription",
            provider="whisper_local",
            status="succeeded",
            completed_at=datetime.now(UTC),
        ))
        s.commit()
    finally:
        s.close()

    fail_orphaned_running_jobs()

    s = engine_with_jobs()
    try:
        record = s.query(JobRecord).filter_by(file_id="f00000000001").one()
        assert record.status == "succeeded"
    finally:
        s.close()


def test_failed_records_left_alone(engine_with_jobs) -> None:
    """A previously-failed record must keep its original error_class."""
    from app.workers.whisper import fail_orphaned_running_jobs

    _seed(engine_with_jobs)
    s = engine_with_jobs()
    try:
        s.add(JobRecord(
            file_id="f00000000001",
            job_kind="transcription",
            provider="deepgram",
            status="failed",
            error_class="RateLimitError",
            completed_at=datetime.now(UTC),
        ))
        s.commit()
    finally:
        s.close()

    fail_orphaned_running_jobs()

    s = engine_with_jobs()
    try:
        record = s.query(JobRecord).filter_by(file_id="f00000000001").one()
        assert record.status == "failed"
        assert record.error_class == "RateLimitError"  # not "ContainerRestart"
    finally:
        s.close()


def test_running_record_purges_partial_whisper_data(engine_with_jobs) -> None:
    """Orphaned partial chunks / words must be cleaned via _remove_whisper_data."""
    from app.models import TranscriptChunk, TranscriptWord
    from app.workers.whisper import fail_orphaned_running_jobs

    _seed(engine_with_jobs)
    s = engine_with_jobs()
    try:
        s.add_all([
            JobRecord(
                file_id="f00000000001",
                job_kind="transcription",
                provider="whisper_local",
                status="running",
            ),
            TranscriptChunk(
                file_id="f00000000001",
                chunk_index=0,
                text="partial",
                language="en",
                timestamp_start=0.0,
                timestamp_end=1.0,
            ),
            TranscriptWord(
                file_id="f00000000001",
                text="partial",
                language="en",
                timestamp_start=0.0,
                timestamp_end=1.0,
            ),
        ])
        s.commit()
    finally:
        s.close()

    fail_orphaned_running_jobs()

    s = engine_with_jobs()
    try:
        chunks = s.query(TranscriptChunk).filter_by(
            file_id="f00000000001"
        ).all()
        words = s.query(TranscriptWord).filter_by(
            file_id="f00000000001"
        ).all()
    finally:
        s.close()
    assert chunks == []
    assert words == []


def test_no_running_records_is_noop(engine_with_jobs) -> None:
    """No-op exit when nothing was orphaned."""
    from app.workers.whisper import fail_orphaned_running_jobs

    _seed(engine_with_jobs)
    fail_orphaned_running_jobs()  # must not raise

    s = engine_with_jobs()
    try:
        assert s.query(JobRecord).count() == 0
    finally:
        s.close()
