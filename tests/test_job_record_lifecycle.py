"""Phase 1A foundation tests for JobRecord + speaker_id migrations.

Schema-only coverage at this layer:

* ``job_records`` is created with the right columns and accepts the
  full lifecycle (running → succeeded / failed)
* ``transcript_words`` and ``transcript_chunks`` gain ``speaker_id``
  via the new migration, with nullable semantics
* All migrations are idempotent

Worker-side lifecycle (status transitions written by index_whisper,
container-restart fixup) lands in Phase 1C; this file pins only the
schema contract that those transitions will rely on.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

# Stub heavy ML deps before importing app.database (matches conftest).
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


@pytest.fixture()
def schema_engine(tmp_path):
    """Build an empty DB and run the relevant Phase 1A migrations.

    The migrations under test are idempotent CREATE / ADD COLUMN
    helpers, so a freshly-created engine that runs them once is
    enough to exercise the schema; we don't need the full
    ``init_search_db`` orchestration here.
    """
    db_path = tmp_path / "search.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # Pre-create the legacy transcript_* tables so the column-add
    # migration has something to ALTER. The CREATE shape mirrors the
    # pre-Phase-1A version (no speaker_id) so the test exercises the
    # upgrade path rather than the fresh-install path.
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE transcript_words ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  file_id TEXT NOT NULL,"
            "  text TEXT NOT NULL,"
            "  language TEXT NOT NULL DEFAULT '',"
            "  timestamp_start REAL NOT NULL,"
            "  timestamp_end REAL NOT NULL,"
            "  created_at TIMESTAMP"
            ")"
        ))
        conn.execute(text(
            "CREATE TABLE transcript_chunks ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  file_id TEXT NOT NULL,"
            "  chunk_index INTEGER NOT NULL,"
            "  text TEXT NOT NULL,"
            "  language TEXT NOT NULL DEFAULT '',"
            "  timestamp_start REAL NOT NULL,"
            "  timestamp_end REAL NOT NULL,"
            "  text_refined_at TIMESTAMP,"
            "  created_at TIMESTAMP"
            ")"
        ))
        # indexed_files referenced by job_records FK.
        conn.execute(text(
            "CREATE TABLE indexed_files ("
            "  file_id TEXT PRIMARY KEY,"
            "  drive TEXT NOT NULL,"
            "  filename TEXT NOT NULL,"
            "  file_path TEXT NOT NULL,"
            "  file_type TEXT NOT NULL,"
            "  mime_type TEXT NOT NULL,"
            "  file_size INTEGER NOT NULL"
            ")"
        ))
        conn.execute(text(
            "INSERT INTO indexed_files "
            "(file_id, drive, filename, file_path, file_type, mime_type, file_size) "
            "VALUES ('f00000000001', 'd', 'a.wav', '/d/a.wav', 'audio', "
            "'audio/wav', 1)"
        ))
    return engine


def _columns(engine, table: str) -> set[str]:
    with engine.connect() as conn:
        return {
            row[1]
            for row in conn.execute(
                text(f"PRAGMA table_info({table})")
            ).fetchall()
        }


def test_speaker_id_added_to_transcript_words(schema_engine) -> None:
    from app.database import _migrate_transcript_speaker_id_columns

    with schema_engine.begin() as conn:
        _migrate_transcript_speaker_id_columns(conn)

    assert "speaker_id" in _columns(schema_engine, "transcript_words")


def test_speaker_id_added_to_transcript_chunks(schema_engine) -> None:
    from app.database import _migrate_transcript_speaker_id_columns

    with schema_engine.begin() as conn:
        _migrate_transcript_speaker_id_columns(conn)

    assert "speaker_id" in _columns(schema_engine, "transcript_chunks")


def test_speaker_id_accepts_null(schema_engine) -> None:
    """Existing rows must survive the migration with speaker_id = NULL."""
    from app.database import _migrate_transcript_speaker_id_columns

    with schema_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO transcript_words "
            "(file_id, text, timestamp_start, timestamp_end) "
            "VALUES ('f00000000001', 'hello', 0.0, 0.5)"
        ))

    with schema_engine.begin() as conn:
        _migrate_transcript_speaker_id_columns(conn)

    with schema_engine.connect() as conn:
        row = conn.execute(text(
            "SELECT text, speaker_id FROM transcript_words "
            "WHERE file_id = 'f00000000001'"
        )).fetchone()
    assert row is not None
    assert row[0] == "hello"
    assert row[1] is None


def test_speaker_id_accepts_string_value(schema_engine) -> None:
    from app.database import _migrate_transcript_speaker_id_columns

    with schema_engine.begin() as conn:
        _migrate_transcript_speaker_id_columns(conn)
        conn.execute(text(
            "INSERT INTO transcript_words "
            "(file_id, text, timestamp_start, timestamp_end, speaker_id) "
            "VALUES ('f00000000001', 'hi', 0.0, 0.5, 'spk_0')"
        ))

    with schema_engine.connect() as conn:
        row = conn.execute(text(
            "SELECT speaker_id FROM transcript_words WHERE text = 'hi'"
        )).fetchone()
    assert row is not None
    assert row[0] == "spk_0"


def test_speaker_id_migration_is_idempotent(schema_engine) -> None:
    from app.database import _migrate_transcript_speaker_id_columns

    with schema_engine.begin() as conn:
        _migrate_transcript_speaker_id_columns(conn)
    # Running twice must not raise (would error if ALTER ran a second
    # time without the PRAGMA-based guard).
    with schema_engine.begin() as conn:
        _migrate_transcript_speaker_id_columns(conn)

    assert "speaker_id" in _columns(schema_engine, "transcript_words")
    assert "speaker_id" in _columns(schema_engine, "transcript_chunks")


def test_job_records_table_created(schema_engine) -> None:
    from app.database import _create_job_records_table

    with schema_engine.begin() as conn:
        _create_job_records_table(conn)

    cols = _columns(schema_engine, "job_records")
    expected = {
        "id", "file_id", "job_kind", "provider", "status",
        "error_class", "error_message", "attempted_at", "completed_at",
    }
    assert expected <= cols


def test_job_records_accepts_running_status(schema_engine) -> None:
    from app.database import _create_job_records_table

    with schema_engine.begin() as conn:
        _create_job_records_table(conn)
        conn.execute(text(
            "INSERT INTO job_records "
            "(file_id, job_kind, provider, status) "
            "VALUES ('f00000000001', 'transcription', 'deepgram', 'running')"
        ))

    with schema_engine.connect() as conn:
        row = conn.execute(text(
            "SELECT job_kind, provider, status, error_class, completed_at "
            "FROM job_records WHERE file_id = 'f00000000001'"
        )).fetchone()
    assert row == ("transcription", "deepgram", "running", None, None)


def test_job_records_accepts_succeeded_lifecycle(schema_engine) -> None:
    """A succeeded row carries provider + completed_at and no error."""
    from app.database import _create_job_records_table

    with schema_engine.begin() as conn:
        _create_job_records_table(conn)
        completed = datetime.now(UTC).isoformat()
        conn.execute(
            text(
                "INSERT INTO job_records "
                "(file_id, job_kind, provider, status, completed_at) "
                "VALUES (:fid, 'transcription', 'whisper_local', "
                "'succeeded', :ca)"
            ),
            {"fid": "f00000000001", "ca": completed},
        )

    with schema_engine.connect() as conn:
        row = conn.execute(text(
            "SELECT status, error_class, completed_at FROM job_records "
            "WHERE file_id = 'f00000000001'"
        )).fetchone()
    assert row[0] == "succeeded"
    assert row[1] is None
    assert row[2] is not None


def test_job_records_accepts_failed_with_error_class(schema_engine) -> None:
    from app.database import _create_job_records_table

    with schema_engine.begin() as conn:
        _create_job_records_table(conn)
        conn.execute(
            text(
                "INSERT INTO job_records "
                "(file_id, job_kind, provider, status, error_class, "
                "error_message) "
                "VALUES (:fid, 'transcription', 'deepgram', 'failed', "
                "'RateLimitError', 'HTTP 429')"
            ),
            {"fid": "f00000000001"},
        )

    with schema_engine.connect() as conn:
        row = conn.execute(text(
            "SELECT status, error_class, error_message FROM job_records "
            "WHERE file_id = 'f00000000001'"
        )).fetchone()
    assert row == ("failed", "RateLimitError", "HTTP 429")


def test_job_records_provider_nullable(schema_engine) -> None:
    """Future job kinds may not have a provider — column must be nullable."""
    from app.database import _create_job_records_table

    with schema_engine.begin() as conn:
        _create_job_records_table(conn)
        conn.execute(text(
            "INSERT INTO job_records (file_id, job_kind, status) "
            "VALUES ('f00000000001', 'embedding', 'running')"
        ))

    with schema_engine.connect() as conn:
        row = conn.execute(text(
            "SELECT provider FROM job_records WHERE job_kind = 'embedding'"
        )).fetchone()
    assert row is not None
    assert row[0] is None


def test_job_records_table_creation_is_idempotent(schema_engine) -> None:
    from app.database import _create_job_records_table

    with schema_engine.begin() as conn:
        _create_job_records_table(conn)
    with schema_engine.begin() as conn:
        _create_job_records_table(conn)
    with schema_engine.begin() as conn:
        _create_job_records_table(conn)

    assert "job_records" in {
        r[0] for r in schema_engine.connect().execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
    }


def test_job_record_orm_model_matches_table(schema_engine) -> None:
    """The SQLAlchemy ``JobRecord`` model must match the migration shape.

    Importing the model and inserting via ORM is the most direct
    parity check between the migration runner and the ORM definition.
    """
    from app.database import _create_job_records_table
    from sqlalchemy.orm import sessionmaker

    with schema_engine.begin() as conn:
        _create_job_records_table(conn)

    Session = sessionmaker(bind=schema_engine, expire_on_commit=False)
    from app.models import JobRecord

    with Session() as session:
        rec = JobRecord(
            file_id="f00000000001",
            job_kind="transcription",
            provider="deepgram",
            status="running",
        )
        session.add(rec)
        session.commit()
        rec.status = "failed"
        rec.error_class = "FatalError"
        rec.error_message = "401 unauthorized"
        rec.completed_at = datetime.now(UTC)
        session.commit()

        fetched = session.query(JobRecord).filter_by(
            file_id="f00000000001"
        ).one()
        assert fetched.status == "failed"
        assert fetched.error_class == "FatalError"
        assert fetched.completed_at is not None
