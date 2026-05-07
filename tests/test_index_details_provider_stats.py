"""Phase 1C tests: ``provider_stats`` aggregate on /index-details.

Spec ``2026-05-07-cloud-transcription-providers.md`` §"観測性 /
index-details API 拡張". The endpoint aggregates ``JobRecord`` rows
from the last 7 days into a per-provider breakdown:

* ``calls`` — total transcription jobs for the provider
* ``failures`` — subset where ``status='failed'``
* ``last_error`` — ``error_class`` of the most-recent failed row
  (``None`` when no failure on record)

Existing fields (``status.whisper`` and the ``embeddings`` block)
must remain unchanged so the legacy frontend keeps working.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
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

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import Base  # noqa: E402
from app.models import IndexedFile, JobRecord  # noqa: E402


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    seed = Session()
    try:
        seed.add(IndexedFile(
            file_id="f00000000001",
            drive="drive1",
            filename="x.mp4",
            file_path="/drives/drive1/x.mp4",
            file_type="video",
            mime_type="video/mp4",
            file_size=100,
            active=True,
            metadata_indexed=True,
            whisper_indexed=True,
        ))
        seed.commit()
    finally:
        seed.close()

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
    monkeypatch.setattr("app.routers.files.get_search_db", _get_search_db, raising=False)
    return Session


def _add_job(
    Session,
    *,
    file_id="f00000000001",
    provider="whisper_local",
    status="succeeded",
    error_class=None,
    days_ago=0,
) -> None:
    s = Session()
    try:
        s.add(JobRecord(
            file_id=file_id,
            job_kind="transcription",
            provider=provider,
            status=status,
            error_class=error_class,
            attempted_at=datetime.now(UTC) - timedelta(days=days_ago),
            completed_at=datetime.now(UTC) - timedelta(days=days_ago),
        ))
        s.commit()
    finally:
        s.close()


@pytest.mark.asyncio
async def test_provider_stats_zero_when_no_records(search_db) -> None:
    from app.routers.files import get_index_details

    response = await get_index_details(file_id="f00000000001", drive="drive1")
    payload = response.model_dump()

    assert "provider_stats" in payload
    # Empty dict (no JobRecord rows) — frontend treats absence as zero.
    assert payload["provider_stats"] == {}


@pytest.mark.asyncio
async def test_provider_stats_aggregates_calls_per_provider(search_db) -> None:
    _add_job(search_db, provider="whisper_local", status="succeeded")
    _add_job(search_db, provider="whisper_local", status="succeeded")
    _add_job(search_db, provider="deepgram", status="succeeded")

    from app.routers.files import get_index_details

    response = await get_index_details(file_id="f00000000001", drive="drive1")
    payload = response.model_dump()

    stats = payload["provider_stats"]
    assert stats["whisper_local"]["calls"] == 2
    assert stats["whisper_local"]["failures"] == 0
    assert stats["whisper_local"]["last_error"] is None
    assert stats["deepgram"]["calls"] == 1


@pytest.mark.asyncio
async def test_provider_stats_records_last_error_class(search_db) -> None:
    _add_job(
        search_db,
        provider="deepgram",
        status="failed",
        error_class="RateLimitError",
    )

    from app.routers.files import get_index_details

    response = await get_index_details(file_id="f00000000001", drive="drive1")
    payload = response.model_dump()
    stats = payload["provider_stats"]
    assert stats["deepgram"]["calls"] == 1
    assert stats["deepgram"]["failures"] == 1
    assert stats["deepgram"]["last_error"] == "RateLimitError"


@pytest.mark.asyncio
async def test_provider_stats_excludes_records_older_than_7_days(search_db) -> None:
    _add_job(search_db, provider="deepgram", status="succeeded", days_ago=10)
    _add_job(search_db, provider="deepgram", status="succeeded", days_ago=2)

    from app.routers.files import get_index_details

    response = await get_index_details(file_id="f00000000001", drive="drive1")
    payload = response.model_dump()
    # Only the 2-days-ago row counts.
    assert payload["provider_stats"]["deepgram"]["calls"] == 1


@pytest.mark.asyncio
async def test_provider_stats_excludes_non_transcription_kinds(search_db) -> None:
    """``provider_stats`` is transcription-only — embedding / summary
    rows must not pollute the count."""
    s = search_db()
    try:
        s.add(JobRecord(
            file_id="f00000000001",
            job_kind="embedding",
            provider="whisper_local",
            status="succeeded",
        ))
        s.commit()
    finally:
        s.close()

    from app.routers.files import get_index_details

    response = await get_index_details(file_id="f00000000001", drive="drive1")
    payload = response.model_dump()
    assert payload["provider_stats"] == {}


@pytest.mark.asyncio
async def test_legacy_status_block_still_present(search_db) -> None:
    """The pre-1C ``status`` and ``embeddings`` blocks must keep working."""
    from app.routers.files import get_index_details

    response = await get_index_details(file_id="f00000000001", drive="drive1")
    payload = response.model_dump()
    assert "status" in payload
    assert payload["status"]["whisper"] is True
    assert "embeddings" in payload
