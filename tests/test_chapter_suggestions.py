from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.database import Base, _create_suggested_chapters_table
from app.llm import (
    FAILURE_MALFORMED,
    FAILURE_TOKEN_BUDGET,
    JsonGeneration,
)
from app.models import IndexedFile, TranscriptChunk
from app.prompt_loader import render
from app.workers.chapter_suggestions import (
    ChapterSuggestionsWorker,
    _build_system_prompt,
    _build_windows,
    is_chapter_suggestions_enabled,
    normalise_chapter_candidates,
    promote_chapters_to_core,
)
from app.workers.transcription.errors import TransientError


@pytest.fixture()
def chapter_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'chapters.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _create_suggested_chapters_table(conn)
        conn.execute(text("CREATE TABLE suggested_tags(file_id TEXT PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE file_summaries(file_id TEXT PRIMARY KEY)"))
        conn.execute(text("CREATE VIRTUAL TABLE fts_files USING fts5(file_id)"))
        conn.execute(text("CREATE VIRTUAL TABLE fts_transcripts USING fts5(file_id)"))
        conn.execute(text("CREATE VIRTUAL TABLE fts_text_content USING fts5(file_id)"))
        conn.execute(text("CREATE VIRTUAL TABLE fts_files_word USING fts5(file_id)"))
        conn.execute(text("CREATE VIRTUAL TABLE fts_transcripts_word USING fts5(file_id)"))
        conn.execute(text("CREATE VIRTUAL TABLE fts_text_content_word USING fts5(file_id)"))
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def db():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("app.database.get_search_db", db)
    monkeypatch.setattr("app.database.get_search_db_read", db)
    # indexer imports the helpers by value; in a full-suite run it may have
    # been imported before this fixture, so patch that reference as well.
    monkeypatch.setattr("app.indexer.get_search_db", db)
    monkeypatch.setattr("app.routers.chapter_suggestions.get_search_db", db)
    return db


def _seed_transcript(db, *, chunks: int = 3) -> None:
    with db() as session:
        session.add(IndexedFile(
            file_id="file00000001", drive="Media", filename="talk.mp4",
            file_path="talk.mp4", file_type="video", mime_type="video/mp4",
            file_size=1,
            active=True, metadata_indexed=True, whisper_indexed=True,
        ))
        for index in range(chunks):
            session.add(TranscriptChunk(
                file_id="file00000001", chunk_index=index,
                text=f"segment-{index}-" + ("x" * 70), language="en",
                timestamp_start=float(index * 10), timestamp_end=float(index * 10 + 10),
            ))


def test_oversized_transcript_chunk_is_fully_covered_by_bounded_windows(monkeypatch):
    monkeypatch.setattr("app.workers.chapter_suggestions._WINDOW_CHAR_BUDGET", 80)
    source = "abcdefghijklmnop" * 40
    chunk = SimpleNamespace(timestamp_start=1, timestamp_end=2, text=source)
    windows = _build_windows([chunk])
    assert len(windows) > 1
    assert all(len(window) <= 80 for window in windows)
    recovered = "".join(window.split("] ", 1)[1] for window in windows)
    assert recovered == source


@pytest.mark.asyncio
async def test_chapter_policy_lookup_fails_closed(monkeypatch):
    calls = []

    async def unavailable(drive, feature, *, default_on_failure):
        calls.append((drive, feature, default_on_failure))
        raise TransientError("backend warming")

    monkeypatch.setattr("app.policy_client.is_feature_enabled", unavailable)
    assert await is_chapter_suggestions_enabled("Private") is False
    assert calls == [("Private", "chapter_suggestions", False)]


class FakeLLM:
    enabled = True

    def __init__(self):
        self.calls: list[str] = []

    async def generate_json_result(self, _system: str, user: str, **_kwargs):
        self.calls.append(user)
        if "CANDIDATES" in user:
            return JsonGeneration({"chapters": [
                {"start_time": 0, "end_time": 10, "title": "Opening"},
                {"start_time": 10, "end_time": None, "title": "Discussion"},
            ]}, None)
        first = float(user.split("[")[1].split("-")[0])
        return JsonGeneration({"chapters": [
            {"start_time": first, "end_time": first + 10, "title": f"At {first:g}"}
        ]}, None)


def test_chapter_prompts_define_semantic_navigation_granularity():
    system = _build_system_prompt("en")
    editor = render(
        "chapter_suggestions/consolidate_user.jinja2",
        candidates_json="[]",
    )

    assert "central subject, purpose, or phase" in system
    assert "speaker changes" in system
    assert "brief digressions" in system
    assert "intentionally seek" in system
    assert "transcript segment or timestamp boundary" in system
    assert "example, clarification, continuation" in editor
    assert "distinct destination" in editor
    assert "complete chronological coverage" in editor


class GranularSingleWindowLLM:
    enabled = True

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def generate_json_result(self, _system: str, user: str, **kwargs):
        self.calls.append((user, kwargs))
        if "PROVISIONAL CANDIDATES" in user:
            assert "Late ending" in user
            return JsonGeneration({"chapters": [
                {"start_time": 0, "end_time": 120, "title": "Main subject"},
                {"start_time": 120, "end_time": None, "title": "Late ending"},
            ]}, None)
        return JsonGeneration({"chapters": [
            {
                "start_time": float(index * 10),
                "end_time": float(index * 10 + 10),
                "title": "Late ending" if index == 13 else f"Micro topic {index}",
            }
            for index in range(14)
        ]}, None)


@pytest.mark.asyncio
async def test_single_window_editor_receives_all_candidates_without_head_truncation(
    chapter_db, monkeypatch
):
    _seed_transcript(chapter_db)
    monkeypatch.setattr(
        "app.workers.chapter_suggestions.settings",
        SimpleNamespace(
            features=SimpleNamespace(chapter_suggestions="manual"),
            llm=SimpleNamespace(output_language="en", model="test-model"),
        ),
    )
    monkeypatch.setattr(
        "app.workers.chapter_suggestions.is_chapter_suggestions_enabled",
        lambda _drive: _async_true(),
    )
    monkeypatch.setattr(
        "app.workers.chapter_suggestions.emit_chapter_suggestions_event",
        lambda *_args: _async_none(),
    )
    llm = GranularSingleWindowLLM()

    await ChapterSuggestionsWorker(llm)._process_file(
        "file00000001", force=True
    )

    assert len(llm.calls) == 2
    assert all("max_tokens_override" not in kwargs for _, kwargs in llm.calls)
    with chapter_db() as session:
        saved = json.loads(session.execute(text(
            "SELECT chapters_json FROM suggested_chapters WHERE file_id=:fid"
        ), {"fid": "file00000001"}).scalar_one())
    assert [chapter["title"] for chapter in saved] == [
        "Main subject",
        "Late ending",
    ]


class TokenBudgetLLM:
    """Every attempt loses its budget to the provider's own thinking."""

    enabled = True

    def __init__(self):
        self.calls: list[str] = []

    async def generate_json_result(self, _system: str, user: str, **_kwargs):
        self.calls.append(user)
        return JsonGeneration(None, FAILURE_TOKEN_BUDGET)


class InvalidJsonLLM:
    enabled = True

    def __init__(self):
        self.calls: list[str] = []

    async def generate_json_result(self, _system: str, user: str, **_kwargs):
        self.calls.append(user)
        return JsonGeneration(None, FAILURE_MALFORMED)


@pytest.mark.asyncio
async def test_invalid_json_retries_then_emits_failed_event(
    chapter_db, monkeypatch
):
    _seed_transcript(chapter_db)
    monkeypatch.setattr(
        "app.workers.chapter_suggestions.settings",
        SimpleNamespace(
            features=SimpleNamespace(chapter_suggestions="manual"),
            llm=SimpleNamespace(output_language="en", model="test-model"),
        ),
    )
    monkeypatch.setattr(
        "app.workers.chapter_suggestions.is_chapter_suggestions_enabled",
        lambda _drive: _async_true(),
    )
    emitted = []

    async def record_event(event, data):
        emitted.append((event, data))

    monkeypatch.setattr(
        "app.workers.chapter_suggestions.emit_chapter_suggestions_event",
        record_event,
    )
    llm = InvalidJsonLLM()

    await ChapterSuggestionsWorker(llm)._process_file(
        "file00000001", force=True
    )

    assert len(llm.calls) == 2
    assert "previous attempt" in llm.calls[1].lower()
    assert emitted == [(
        "intelligence.chapter_suggestions.failed",
        {
            "file_id": "file00000001",
            "drive": "Media",
            "reason": "invalid_model_output",
        },
    )]
    with chapter_db() as session:
        assert session.execute(
            text("SELECT 1 FROM suggested_chapters")
        ).first() is None


@pytest.mark.asyncio
async def test_long_transcript_covers_every_chunk_and_consolidates_candidates(chapter_db, monkeypatch):
    _seed_transcript(chapter_db, chunks=8)
    monkeypatch.setattr("app.workers.chapter_suggestions._WINDOW_CHAR_BUDGET", 180)
    monkeypatch.setattr(
        "app.workers.chapter_suggestions.settings",
        SimpleNamespace(
            features=SimpleNamespace(chapter_suggestions="manual"),
            llm=SimpleNamespace(output_language="en", model="test-model"),
        ),
    )
    monkeypatch.setattr(
        "app.workers.chapter_suggestions.is_chapter_suggestions_enabled",
        lambda _drive: _async_true(),
    )
    llm = FakeLLM()
    worker = ChapterSuggestionsWorker(llm)
    emitted = []

    async def record_event(event, data):
        emitted.append((event, data))

    monkeypatch.setattr(
        "app.workers.chapter_suggestions.emit_chapter_suggestions_event",
        record_event,
    )

    await worker._process_file("file00000001", force=True)

    window_calls = [call for call in llm.calls if "CANDIDATES" not in call]
    assert len(window_calls) > 1
    combined = "\n".join(window_calls)
    for index in range(8):
        assert f"segment-{index}-" in combined
    assert "CANDIDATES" in llm.calls[-1]
    with chapter_db() as session:
        row = session.execute(text(
            "SELECT chapters_json, status FROM suggested_chapters WHERE file_id=:fid"
        ), {"fid": "file00000001"}).one()
    assert json.loads(row[0])[0]["title"] == "Opening"
    assert row[1] == "pending"
    assert len(emitted) == 1
    assert emitted[0][0] == "intelligence.chapter_suggestions.ready"
    assert emitted[0][1]["file_id"] == "file00000001"
    assert emitted[0][1]["drive"] == "Media"
    assert emitted[0][1]["created_at"]


@pytest.mark.asyncio
async def test_worker_rechecks_policy_before_sending_transcript(
    chapter_db, monkeypatch
):
    _seed_transcript(chapter_db)
    monkeypatch.setattr(
        "app.workers.chapter_suggestions.settings",
        SimpleNamespace(
            features=SimpleNamespace(chapter_suggestions="manual"),
            llm=SimpleNamespace(output_language="en", model="test-model"),
        ),
    )

    async def denied(_drive):
        return False

    monkeypatch.setattr(
        "app.workers.chapter_suggestions.is_chapter_suggestions_enabled",
        denied,
    )
    llm = FakeLLM()
    await ChapterSuggestionsWorker(llm)._process_file(
        "file00000001", force=True
    )

    assert llm.calls == []
    with chapter_db() as session:
        assert session.execute(
            text("SELECT 1 FROM suggested_chapters")
        ).first() is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"chapters": [{"start_time": 0, "end_time": 1, "title": " A "}]}, [{"start_time": 0.0, "end_time": 1.0, "title": "A"}]),
        ({"chapters": [{"start_time": float("nan"), "title": "bad"}]}, []),
        ({"chapters": [{"start_time": 1, "end_time": float("inf"), "title": "ok"}]}, [{"start_time": 1.0, "end_time": None, "title": "ok"}]),
        ({"chapters": [{"start_time": 1, "title": "   "}]}, []),
        ({"chapters": [{"start_time": 2, "title": "B"}, {"start_time": 1, "title": "A"}]}, [{"start_time": 2.0, "end_time": None, "title": "B"}, {"start_time": 1.0, "end_time": None, "title": "A"}]),
    ],
)
def test_core_promotion_validator_parity(raw, expected):
    assert normalise_chapter_candidates(raw) == expected


@pytest.mark.asyncio
async def test_promotion_wire_shape_and_secret(monkeypatch):
    received = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received.update(
            method=request.method,
            url=str(request.url),
            secret=request.headers.get("X-Internal-Secret"),
            content_type=request.headers.get("Content-Type"),
            body=json.loads(request.content),
        )
        return httpx.Response(204)

    monkeypatch.setenv("HOMEVAULT_INTERNAL_URL", "http://core:9000")
    monkeypatch.setenv("CORE_INTERNAL_SECRET", "shared-secret")
    transport = httpx.MockTransport(handler)
    await promote_chapters_to_core(
        "file00000001",
        [{"start_time": 0.0, "end_time": None, "title": "Opening"}],
        transport=transport,
    )

    assert received == {
        "method": "PUT",
        "url": "http://core:9000/api/internal/files/file00000001/chapters",
        "secret": "shared-secret",
        "content_type": "application/json",
        "body": {"chapters": [{"start_time": 0.0, "end_time": None, "title": "Opening"}]},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [None, "   "])
async def test_promotion_requires_local_secret(monkeypatch, configured):
    if configured is None:
        monkeypatch.delenv("CORE_INTERNAL_SECRET", raising=False)
    else:
        monkeypatch.setenv("CORE_INTERNAL_SECRET", configured)
    with pytest.raises(RuntimeError, match="CORE_INTERNAL_SECRET"):
        await promote_chapters_to_core("file00000001", [{"start_time": 0, "end_time": None, "title": "A"}])


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 422, 503])
async def test_promotion_propagates_core_rejection(monkeypatch, status):
    monkeypatch.setenv("CORE_INTERNAL_SECRET", "secret")
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(status, json={"detail": "rejected"})
    )
    with pytest.raises(httpx.HTTPStatusError) as exc:
        await promote_chapters_to_core("file00000001", [{"start_time": 0, "end_time": None, "title": "A"}], transport=transport)
    assert exc.value.response.status_code == status


def test_suggested_chapters_table_schema_and_purge(chapter_db):
    _seed_transcript(chapter_db)
    with chapter_db() as session:
        session.execute(text(
            "INSERT INTO suggested_chapters(file_id, chapters_json, model, created_at, status) "
            "VALUES ('file00000001', '[]', 'm', 'now', 'dismissed')"
        ))
        cols = {r[1] for r in session.execute(text("PRAGMA table_info(suggested_chapters)"))}
    assert {"file_id", "chapters_json", "model", "created_at", "status"} <= cols

    from app.indexer import _purge_file
    _purge_file("file00000001")
    with chapter_db() as session:
        assert session.execute(text("SELECT 1 FROM suggested_chapters")).first() is None


@pytest.mark.asyncio
async def test_approve_marks_accepted_only_after_core_success(chapter_db, monkeypatch):
    _seed_transcript(chapter_db)
    with chapter_db() as session:
        session.execute(text(
            "INSERT INTO suggested_chapters(file_id, chapters_json, model, created_at, status) "
            "VALUES (:fid, :chapters, 'm', 'now', 'pending')"
        ), {
            "fid": "file00000001",
            "chapters": json.dumps([{"start_time": 0, "end_time": None, "title": "A"}]),
        })
    settings = SimpleNamespace(features=SimpleNamespace(chapter_suggestions="manual"))
    monkeypatch.setattr("app.routers.chapter_suggestions.settings", settings)
    monkeypatch.setattr(
        "app.policy_client.is_feature_enabled",
        lambda *_args, **_kwargs: _async_true(),
    )

    async def failed(*_args, **_kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr("app.routers.chapter_suggestions.promote_chapters_to_core", failed)
    from app.routers.chapter_suggestions import approve_chapter_suggestions
    with pytest.raises(Exception) as exc:
        await approve_chapter_suggestions("file00000001", "Media")
    assert getattr(exc.value, "status_code", None) == 502
    with chapter_db() as session:
        status = session.execute(text(
            "SELECT status FROM suggested_chapters WHERE file_id=:fid"
        ), {"fid": "file00000001"}).scalar_one()
    assert status == "pending"


async def _async_true():
    return True


async def _async_none():
    return None


@pytest.mark.asyncio
async def test_generate_rejects_missing_transcript(chapter_db, monkeypatch):
    with chapter_db() as session:
        session.add(IndexedFile(
            file_id="file00000001", drive="Media", filename="silent.mp4",
            file_path="silent.mp4", file_type="video", mime_type="video/mp4",
            file_size=1, active=True, metadata_indexed=True,
        ))
    monkeypatch.setattr(
        "app.routers.chapter_suggestions.settings",
        SimpleNamespace(features=SimpleNamespace(chapter_suggestions="manual")),
    )
    monkeypatch.setattr(
        "app.policy_client.is_feature_enabled",
        lambda *_args, **_kwargs: _async_true(),
    )
    monkeypatch.setattr(
        "app.routers.chapter_suggestions.get_llm_client",
        lambda: SimpleNamespace(enabled=True),
    )
    from app.routers.chapter_suggestions import generate_chapter_suggestions
    with pytest.raises(Exception) as exc:
        await generate_chapter_suggestions("file00000001", "Media")
    assert getattr(exc.value, "status_code", None) == 409


@pytest.mark.asyncio
async def test_approve_success_and_empty_rejection(chapter_db, monkeypatch):
    _seed_transcript(chapter_db)
    monkeypatch.setattr(
        "app.routers.chapter_suggestions.settings",
        SimpleNamespace(features=SimpleNamespace(chapter_suggestions="manual")),
    )
    monkeypatch.setattr(
        "app.policy_client.is_feature_enabled",
        lambda *_args, **_kwargs: _async_true(),
    )
    promoted = []

    async def success(file_id, chapters):
        promoted.append((file_id, chapters))

    monkeypatch.setattr(
        "app.routers.chapter_suggestions.promote_chapters_to_core", success
    )
    from app.routers.chapter_suggestions import approve_chapter_suggestions

    with chapter_db() as session:
        session.execute(text(
            "INSERT INTO suggested_chapters(file_id, chapters_json, model, created_at, status) "
            "VALUES (:fid, :chapters, 'm', 'now', 'pending')"
        ), {"fid": "file00000001", "chapters": json.dumps([
            {"start_time": 0, "end_time": None, "title": "A"}
        ])})
    result = await approve_chapter_suggestions("file00000001", "Media")
    assert result.status == "ok"
    assert promoted[0][0] == "file00000001"
    with chapter_db() as session:
        assert session.execute(text(
            "SELECT status FROM suggested_chapters WHERE file_id=:fid"
        ), {"fid": "file00000001"}).scalar_one() == "accepted"
        session.execute(text(
            "UPDATE suggested_chapters SET chapters_json='[]', status='pending' "
            "WHERE file_id=:fid"
        ), {"fid": "file00000001"})
    with pytest.raises(Exception) as exc:
        await approve_chapter_suggestions("file00000001", "Media")
    assert getattr(exc.value, "status_code", None) == 422


@pytest.mark.asyncio
async def test_approve_requires_pending_candidate(chapter_db, monkeypatch):
    _seed_transcript(chapter_db)
    with chapter_db() as session:
        session.execute(text(
            "INSERT INTO suggested_chapters "
            "(file_id, chapters_json, model, created_at, status) "
            "VALUES (:fid, :chapters, 'm', 'rev-a', 'dismissed')"
        ), {
            "fid": "file00000001",
            "chapters": json.dumps([
                {"start_time": 0, "end_time": None, "title": "A"}
            ]),
        })
    monkeypatch.setattr(
        "app.routers.chapter_suggestions.settings",
        SimpleNamespace(features=SimpleNamespace(chapter_suggestions="manual")),
    )
    monkeypatch.setattr(
        "app.routers.chapter_suggestions.is_chapter_suggestions_enabled",
        lambda _drive: _async_true(),
    )
    promoted = []

    async def record(*args):
        promoted.append(args)

    monkeypatch.setattr(
        "app.routers.chapter_suggestions.promote_chapters_to_core", record
    )
    from app.routers.chapter_suggestions import approve_chapter_suggestions

    with pytest.raises(Exception) as exc:
        await approve_chapter_suggestions("file00000001", "Media")

    assert getattr(exc.value, "status_code", None) == 409
    assert promoted == []


@pytest.mark.asyncio
async def test_approve_does_not_mark_a_newer_candidate_accepted(
    chapter_db, monkeypatch
):
    _seed_transcript(chapter_db)
    with chapter_db() as session:
        session.execute(text(
            "INSERT INTO suggested_chapters "
            "(file_id, chapters_json, model, created_at, status) "
            "VALUES (:fid, :chapters, 'm', 'rev-a', 'pending')"
        ), {
            "fid": "file00000001",
            "chapters": json.dumps([
                {"start_time": 0, "end_time": None, "title": "A"}
            ]),
        })
    monkeypatch.setattr(
        "app.routers.chapter_suggestions.settings",
        SimpleNamespace(features=SimpleNamespace(chapter_suggestions="manual")),
    )
    monkeypatch.setattr(
        "app.routers.chapter_suggestions.is_chapter_suggestions_enabled",
        lambda _drive: _async_true(),
    )

    async def regenerate_during_promotion(_file_id, _chapters):
        with chapter_db() as session:
            session.execute(text(
                "UPDATE suggested_chapters SET chapters_json=:chapters, "
                "created_at='rev-b', status='pending' WHERE file_id=:fid"
            ), {
                "fid": "file00000001",
                "chapters": json.dumps([
                    {"start_time": 5, "end_time": None, "title": "B"}
                ]),
            })

    monkeypatch.setattr(
        "app.routers.chapter_suggestions.promote_chapters_to_core",
        regenerate_during_promotion,
    )
    from app.routers.chapter_suggestions import approve_chapter_suggestions

    with pytest.raises(Exception) as exc:
        await approve_chapter_suggestions("file00000001", "Media")

    assert getattr(exc.value, "status_code", None) == 409
    with chapter_db() as session:
        row = session.execute(text(
            "SELECT chapters_json, created_at, status FROM suggested_chapters "
            "WHERE file_id=:fid"
        ), {"fid": "file00000001"}).one()
    assert json.loads(row[0])[0]["title"] == "B"
    assert (row[1], row[2]) == ("rev-b", "pending")


@pytest.mark.asyncio
async def test_dismiss_and_access_gates(chapter_db, monkeypatch):
    _seed_transcript(chapter_db)
    with chapter_db() as session:
        session.execute(text(
            "INSERT INTO suggested_chapters(file_id, chapters_json, model, created_at, status) "
            "VALUES (:fid, '[]', 'm', 'now', 'pending')"
        ), {"fid": "file00000001"})
    enabled = SimpleNamespace(
        features=SimpleNamespace(chapter_suggestions="manual")
    )
    monkeypatch.setattr("app.routers.chapter_suggestions.settings", enabled)
    monkeypatch.setattr(
        "app.policy_client.is_feature_enabled",
        lambda *_args, **_kwargs: _async_true(),
    )
    from app.routers.chapter_suggestions import (
        approve_chapter_suggestions,
        dismiss_chapter_suggestions,
        get_chapter_suggestions,
    )
    result = await dismiss_chapter_suggestions("file00000001", "Media")
    assert result.status == "ok"
    with chapter_db() as session:
        assert session.execute(text(
            "SELECT status FROM suggested_chapters WHERE file_id=:fid"
        ), {"fid": "file00000001"}).scalar_one() == "dismissed"

    with pytest.raises(Exception) as cross_drive:
        await dismiss_chapter_suggestions("file00000001", "Other")
    assert getattr(cross_drive.value, "status_code", None) == 404

    disabled = SimpleNamespace(
        features=SimpleNamespace(chapter_suggestions="false")
    )
    monkeypatch.setattr("app.routers.chapter_suggestions.settings", disabled)
    response = await get_chapter_suggestions("file00000001", "Media")
    assert response.enabled is False
    with pytest.raises(Exception) as feature_off:
        await approve_chapter_suggestions("file00000001", "Media")
    assert getattr(feature_off.value, "status_code", None) == 404


@pytest.mark.asyncio
async def test_per_drive_policy_off_is_hidden(chapter_db, monkeypatch):
    _seed_transcript(chapter_db)
    monkeypatch.setattr(
        "app.routers.chapter_suggestions.settings",
        SimpleNamespace(features=SimpleNamespace(chapter_suggestions="manual")),
    )

    async def denied(*_args, **_kwargs):
        return False

    monkeypatch.setattr("app.policy_client.is_feature_enabled", denied)
    from app.routers.chapter_suggestions import get_chapter_suggestions
    with pytest.raises(Exception) as exc:
        await get_chapter_suggestions("file00000001", "Media")
    assert getattr(exc.value, "status_code", None) == 404


@pytest.mark.asyncio
async def test_token_budget_failure_names_its_own_reason(
    chapter_db, monkeypatch
):
    """A budget spent on reasoning has a different remedy than bad JSON."""
    _seed_transcript(chapter_db)
    emitted: list[tuple[str, dict]] = []

    async def record_event(event: str, data: dict):
        emitted.append((event, data))

    monkeypatch.setattr(
        "app.workers.chapter_suggestions.settings",
        SimpleNamespace(
            features=SimpleNamespace(chapter_suggestions="manual"),
            llm=SimpleNamespace(output_language="en", model="test-model"),
        ),
    )
    monkeypatch.setattr(
        "app.workers.chapter_suggestions.is_chapter_suggestions_enabled",
        lambda _drive: _async_true(),
    )
    monkeypatch.setattr(
        "app.workers.chapter_suggestions.emit_chapter_suggestions_event",
        record_event,
    )
    llm = TokenBudgetLLM()

    await ChapterSuggestionsWorker(llm)._process_file(
        "file00000001", force=True
    )

    assert len(llm.calls) == 2
    assert emitted == [(
        "intelligence.chapter_suggestions.failed",
        {
            "file_id": "file00000001",
            "drive": "Media",
            "reason": "model_token_budget",
        },
    )]
