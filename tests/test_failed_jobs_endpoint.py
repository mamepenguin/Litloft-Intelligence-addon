"""Tests for the admin global failed-jobs summary endpoint.

Spec: ``docs/superpowers/specs/2026-05-24-intelligence-reindex-controls.md``
§2.2.

``GET /api/addons/intelligence/admin/failed-jobs`` aggregates
``JobRecord`` rows with ``status='failed'`` (or ``'failed'``-like
classes) keyed by ``(file_id, job_kind, provider)`` and returns the
most-recent row per group. Skipped rows (``UnsupportedMimeType``) are
explicitly excluded — they would just be re-skipped on retry.

Coverage:

* Happy path: a single failed row surfaces with ``filename`` + ``drive``
  JOIN'd from ``IndexedFile``.
* ``status='skipped'`` rows are excluded (retry-eligible filter).
* Failed rows for purged files (no ``IndexedFile`` row) are excluded.
* Rows older than 7 days are still shown until the latest terminal state
  for that group is no longer ``failed``.
* ``attempts`` counts consecutive failures since the last ``succeeded``
  or manual ``skipped`` close (history is not a lifetime total — UI
  relevance only).
* ``error_message_excerpt`` is truncated to 256 chars.
* Aggregation: multiple failed rows for the same
  ``(file_id, job_kind, provider)`` collapse into one item carrying the
  latest ``attempted_at`` / ``error_class`` / ``error_message``.

RED-phase: the handler does not exist yet. The import at the top of
each test triggers ImportError which pytest reports as RED.
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
        seed.add_all([
            IndexedFile(
                file_id="f00000000001",
                drive="drive1",
                filename="x.mp4",
                file_path="/drives/drive1/x.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=100,
                active=True,
            ),
            IndexedFile(
                file_id="f00000000002",
                drive="drive2",
                filename="y.mp4",
                file_path="/drives/drive2/y.mp4",
                file_type="video",
                mime_type="video/mp4",
                file_size=200,
                active=True,
            ),
        ])
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
    monkeypatch.setattr(
        "app.routers.admin.get_search_db", _get_search_db, raising=False,
    )
    return Session


def _add_job(
    Session,
    *,
    file_id: str = "f00000000001",
    job_kind: str = "transcription",
    provider: str | None = "whisper_local",
    status: str = "failed",
    error_class: str | None = "FatalError",
    error_message: str | None = "ffmpeg returned 1: bad input",
    days_ago: float = 0,
) -> None:
    s = Session()
    try:
        s.add(JobRecord(
            file_id=file_id,
            job_kind=job_kind,
            provider=provider,
            status=status,
            error_class=error_class,
            error_message=error_message,
            attempted_at=datetime.now(UTC) - timedelta(days=days_ago),
            completed_at=datetime.now(UTC) - timedelta(days=days_ago),
        ))
        s.commit()
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_jobs_returns_single_row_payload(search_db) -> None:
    from app.routers.admin import get_failed_jobs

    _add_job(
        search_db,
        provider="whisper_local",
        status="failed",
        error_class="FatalError",
        error_message="ffmpeg returned 1: bad input",
    )

    response = await get_failed_jobs()
    payload = response if isinstance(response, dict) else response.model_dump()

    assert payload["total"] == 1
    assert len(payload["items"]) == 1

    item = payload["items"][0]
    assert item["file_id"] == "f00000000001"
    assert item["filename"] == "x.mp4"
    assert item["drive"] == "drive1"
    assert item["job_kind"] == "transcription"
    assert item["provider"] == "whisper_local"
    assert item["error_class"] == "FatalError"
    assert "ffmpeg returned 1" in item["error_message_excerpt"]
    assert "attempted_at" in item
    assert "attempts" in item


@pytest.mark.asyncio
async def test_failed_jobs_orders_by_attempted_at_descending(
    search_db,
) -> None:
    """Most-recent failures surface first so the operator sees the
    fresh signal at the top of the modal."""
    from app.routers.admin import get_failed_jobs

    _add_job(
        search_db,
        file_id="f00000000001",
        provider="deepgram",
        days_ago=3,
        error_class="RateLimitError",
    )
    _add_job(
        search_db,
        file_id="f00000000002",
        provider="whisper_local",
        days_ago=0.5,
        error_class="FatalError",
    )

    response = await get_failed_jobs()
    payload = response if isinstance(response, dict) else response.model_dump()
    items = payload["items"]
    assert len(items) == 2
    # Newest first
    assert items[0]["file_id"] == "f00000000002"
    assert items[1]["file_id"] == "f00000000001"


# ---------------------------------------------------------------------------
# Filters (skipped / purged / windowed / non-failed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skipped_status_is_excluded(search_db) -> None:
    """``status='skipped'`` rows (UnsupportedMimeType, models.py:385)
    must not appear — retry would just be skipped again."""
    from app.routers.admin import get_failed_jobs

    _add_job(
        search_db,
        status="skipped",
        provider=None,
        error_class="UnsupportedMimeType",
        error_message="mime=application/zip",
    )

    response = await get_failed_jobs()
    payload = response if isinstance(response, dict) else response.model_dump()
    assert payload["total"] == 0
    assert payload["items"] == []


@pytest.mark.asyncio
async def test_succeeded_status_is_excluded(search_db) -> None:
    from app.routers.admin import get_failed_jobs

    _add_job(search_db, status="succeeded", error_class=None, error_message=None)
    _add_job(search_db, status="running", error_class=None, error_message=None)

    response = await get_failed_jobs()
    payload = response if isinstance(response, dict) else response.model_dump()
    assert payload["total"] == 0


@pytest.mark.asyncio
async def test_records_older_than_7_days_are_included(search_db) -> None:
    """Failures stay visible indefinitely so they cannot silently keep
    index progress below 100% after the dashboard window expires."""
    from app.routers.admin import get_failed_jobs

    _add_job(search_db, days_ago=10, error_class="FatalError")
    _add_job(search_db, days_ago=2, error_class="RateLimitError")

    response = await get_failed_jobs()
    payload = response if isinstance(response, dict) else response.model_dump()
    assert payload["total"] == 1
    assert payload["items"][0]["error_class"] == "RateLimitError"


@pytest.mark.asyncio
async def test_failed_jobs_excludes_groups_newer_success_resolved(
    search_db,
) -> None:
    """A historical failure must not remain visible after a later
    succeeded/skipped row closes the same (file, kind, provider) group."""
    from app.routers.admin import get_failed_jobs

    _add_job(search_db, days_ago=20, error_class="FatalError")
    _add_job(
        search_db,
        status="succeeded",
        error_class=None,
        error_message=None,
        days_ago=1,
    )

    response = await get_failed_jobs()
    payload = response if isinstance(response, dict) else response.model_dump()
    assert payload["total"] == 0
    assert payload["items"] == []


@pytest.mark.asyncio
async def test_purged_files_are_excluded(search_db) -> None:
    """A failed JobRecord whose IndexedFile has been hard-deleted has
    no filename / drive to JOIN. The endpoint must silently drop it
    rather than render a row with NULL fields the UI can't link to."""
    from app.routers.admin import get_failed_jobs

    # JobRecord for a file_id that does not exist in indexed_files.
    # SQLite without explicit FK enforcement (the default in
    # in-memory test engines) accepts the orphan row.
    _add_job(search_db, file_id="purgedXX001", error_class="FatalError")

    response = await get_failed_jobs()
    payload = response if isinstance(response, dict) else response.model_dump()
    assert payload["total"] == 0


# ---------------------------------------------------------------------------
# Aggregation, attempts counter, excerpt truncation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_failures_aggregate_per_provider(search_db) -> None:
    """Spec §2.2: rows are grouped by (file_id, job_kind, provider)
    and only the latest row is surfaced."""
    from app.routers.admin import get_failed_jobs

    _add_job(
        search_db, provider="deepgram", days_ago=2,
        error_class="RateLimitError", error_message="HTTP 429 old",
    )
    _add_job(
        search_db, provider="deepgram", days_ago=0.5,
        error_class="FatalError", error_message="HTTP 500 new",
    )

    response = await get_failed_jobs()
    payload = response if isinstance(response, dict) else response.model_dump()
    assert payload["total"] == 1
    item = payload["items"][0]
    assert item["provider"] == "deepgram"
    # Latest wins
    assert item["error_class"] == "FatalError"
    assert "HTTP 500 new" in item["error_message_excerpt"]


@pytest.mark.asyncio
async def test_attempts_counts_consecutive_failures_since_last_success(
    search_db,
) -> None:
    """Spec §2.2 attempts: "最後の succeeded 以降の連続失敗回数"."""
    from app.routers.admin import get_failed_jobs

    # Two failures BEFORE a success (must not be counted)
    _add_job(search_db, provider="deepgram", days_ago=5, error_class="FatalError")
    _add_job(search_db, provider="deepgram", days_ago=4, error_class="FatalError")
    # Success resets the counter
    _add_job(
        search_db, provider="deepgram", status="succeeded", days_ago=3,
        error_class=None, error_message=None,
    )
    # Two failures AFTER the success → attempts should equal 2
    _add_job(search_db, provider="deepgram", days_ago=2, error_class="FatalError")
    _add_job(search_db, provider="deepgram", days_ago=1, error_class="FatalError")

    response = await get_failed_jobs()
    payload = response if isinstance(response, dict) else response.model_dump()
    item = payload["items"][0]
    assert item["attempts"] == 2


@pytest.mark.asyncio
async def test_error_message_excerpt_truncates_to_256_chars(search_db) -> None:
    from app.routers.admin import get_failed_jobs

    long_msg = "x" * 1000
    _add_job(
        search_db, provider="whisper_local", error_message=long_msg,
        error_class="FatalError",
    )

    response = await get_failed_jobs()
    payload = response if isinstance(response, dict) else response.model_dump()
    item = payload["items"][0]
    assert len(item["error_message_excerpt"]) <= 256


# ---------------------------------------------------------------------------
# Cross-file aggregation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_different_files_each_become_separate_items(search_db) -> None:
    from app.routers.admin import get_failed_jobs

    _add_job(search_db, file_id="f00000000001", provider="whisper_local")
    _add_job(search_db, file_id="f00000000002", provider="whisper_local")

    response = await get_failed_jobs()
    payload = response if isinstance(response, dict) else response.model_dump()
    assert payload["total"] == 2
    file_ids = {item["file_id"] for item in payload["items"]}
    assert file_ids == {"f00000000001", "f00000000002"}


@pytest.mark.asyncio
async def test_different_providers_for_same_file_are_separate_items(
    search_db,
) -> None:
    """A file failing with whisper_local AND deepgram surfaces as two
    rows so the operator can retry per-provider if needed."""
    from app.routers.admin import get_failed_jobs

    _add_job(
        search_db, file_id="f00000000001", provider="whisper_local",
        error_class="FatalError",
    )
    _add_job(
        search_db, file_id="f00000000001", provider="deepgram",
        error_class="RateLimitError",
    )

    response = await get_failed_jobs()
    payload = response if isinstance(response, dict) else response.model_dump()
    assert payload["total"] == 2
    providers = {item["provider"] for item in payload["items"]}
    assert providers == {"whisper_local", "deepgram"}


# ---------------------------------------------------------------------------
# Pagination shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_shape_includes_limit_offset(search_db) -> None:
    """Spec §2.2 fixes the response shape:
    ``{items, total, limit, offset}``. Even with no failures the keys
    must exist so the frontend can render an empty state without
    optional-chaining everywhere."""
    from app.routers.admin import get_failed_jobs

    response = await get_failed_jobs()
    payload = response if isinstance(response, dict) else response.model_dump()
    assert "items" in payload
    assert "total" in payload
    assert "limit" in payload
    assert "offset" in payload
    assert payload["items"] == []
    assert payload["total"] == 0


# ---------------------------------------------------------------------------
# Manual resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_failed_job_marks_task_done_and_hides_group(
    search_db,
) -> None:
    """Admins can close a permanently failing task so progress reaches
    100% without deleting the file or rewriting failure history."""
    from app.routers.admin import get_failed_jobs, resolve_failed_job

    _add_job(search_db, days_ago=10, error_class="FatalError")

    response = await resolve_failed_job({
        "file_id": "f00000000001",
        "job_kind": "transcription",
        "provider": "whisper_local",
    })
    payload = response if isinstance(response, dict) else response.model_dump()
    assert payload["status"] == "resolved"
    assert payload["file_id"] == "f00000000001"
    assert payload["task"] == "whisper"

    s = search_db()
    try:
        indexed = s.query(IndexedFile).filter_by(file_id="f00000000001").one()
        assert indexed.whisper_indexed is True
        latest = (
            s.query(JobRecord)
            .filter_by(
                file_id="f00000000001",
                job_kind="transcription",
                provider="whisper_local",
            )
            .order_by(JobRecord.attempted_at.desc())
            .first()
        )
        assert latest is not None
        assert latest.status == "skipped"
        assert latest.error_class == "ManuallyResolved"
    finally:
        s.close()

    after = await get_failed_jobs()
    after_payload = after if isinstance(after, dict) else after.model_dump()
    assert after_payload["total"] == 0


@pytest.mark.asyncio
async def test_resolve_failed_job_rejects_unknown_job_kind(search_db) -> None:
    from app.routers.admin import resolve_failed_job

    with pytest.raises(Exception) as exc:
        await resolve_failed_job({
            "file_id": "f00000000001",
            "job_kind": "unknown",
            "provider": "whisper_local",
        })

    assert getattr(exc.value, "status_code", None) == 422
