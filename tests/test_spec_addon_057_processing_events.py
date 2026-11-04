"""SPEC-ADDON-057 I6: ``processing.updated`` on the intelligence writers.

The spec is the untracked ``.flash/spec.md`` working copy. Every write that
moves ``metadata_indexed``, ``text_indexed`` or ``whisper_indexed`` from false
to true, or ends a transcription job ``failed``, must POST
``processing.updated`` through ``/api/internal/addon-events`` with the drive
set and at most 100 file ids per event. A write that leaves a flag unchanged
or clears it posts nothing.

The HTTP call is intercepted at ``httpx``'s transport so the assertion is on
the wire body, not on one internal helper name.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
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
from app.extractors.base import ExtractionResult, TextChunk  # noqa: E402
from app.models import IndexedFile, JobRecord  # noqa: E402
from app.workers.transcription.base import (  # noqa: E402
    ProviderCapabilities,
    TranscriptionSegment,
    WordToken,
)
from app.workers.transcription.errors import (  # noqa: E402
    FatalError,
    RateLimitError,
    TransientError,
)
from app.workers.transcription.retry import CircuitBreakerOpen  # noqa: E402


DRIVE = "d1"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    """A real search DB with the auxiliary tables the writers touch."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS file_summaries ("
            "  file_id TEXT, status TEXT, long_summary TEXT,"
            "  visual_description_status TEXT, visual_description TEXT)"
        ))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS vec_text ("
            "  embedding_id TEXT PRIMARY KEY, embedding_type TEXT, vector BLOB)"
        ))
        for table in (
            "fts_transcripts",
            "fts_transcripts_word",
            "fts_text_content",
            "fts_text_content_word",
        ):
            conn.execute(text(
                f"CREATE TABLE IF NOT EXISTS {table} "
                "(file_id TEXT, chunk_index TEXT, page TEXT, text TEXT)"
            ))
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _write():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def _read():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    for target in (
        "app.database",
        "app.workers.metadata",
        "app.workers.whisper",
        "app.indexer",
    ):
        monkeypatch.setattr(f"{target}.get_search_db", _write, raising=False)
        monkeypatch.setattr(f"{target}.get_search_db_read", _read, raising=False)
    return Session


@pytest.fixture()
def events(monkeypatch):
    """Capture every POST body the writers send to addon-events."""
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            captured.append(json.loads(request.content))
        except json.JSONDecodeError:
            captured.append({"_raw": request.content.decode("utf-8", "replace")})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    real_client = httpx.Client

    def async_client(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_async_client(transport=transport, **kwargs)

    def sync_client(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", async_client)
    monkeypatch.setattr(httpx, "Client", sync_client)
    return captured


def _processing(captured: list[dict]) -> list[dict]:
    return [
        body for body in captured
        if body.get("event") == "processing.updated"
    ]


def _ids(body: dict) -> list[str]:
    data = body.get("data")
    if isinstance(data, dict) and "file_ids" in data:
        return data["file_ids"]
    return body.get("file_ids")


def _assert_event(body: dict, expected_ids, *, drive: str = DRIVE) -> None:
    assert body.get("drive") == drive, body
    ids = _ids(body)
    assert isinstance(ids, list), body
    assert len(ids) <= 100, body
    assert set(ids) == set(expected_ids), body


def _seed_file(
    Session,
    file_id: str,
    *,
    drive: str = DRIVE,
    active: bool = True,
    mime_type: str = "video/mp4",
    file_path: str | None = None,
    metadata_indexed: bool = False,
    text_indexed: bool = False,
    whisper_indexed: bool = False,
    clip_indexed: bool = True,
    clip_thumbnail_indexed: bool = True,
    tfidf_keywords_indexed: bool = True,
) -> None:
    file_type = mime_type.split("/")[0]
    if file_type not in ("video", "image", "audio", "text"):
        file_type = "document"
    session = Session()
    try:
        session.add(IndexedFile(
            file_id=file_id,
            drive=drive,
            filename=Path(file_path).name if file_path else f"{file_id}.bin",
            file_path=file_path or f"/drives/{drive}/{file_id}.bin",
            file_type=file_type,
            mime_type=mime_type,
            file_size=1,
            active=active,
            metadata_indexed=metadata_indexed,
            text_indexed=text_indexed,
            whisper_indexed=whisper_indexed,
            clip_indexed=clip_indexed,
            clip_thumbnail_indexed=clip_thumbnail_indexed,
            tfidf_keywords_indexed=tfidf_keywords_indexed,
        ))
        session.commit()
    finally:
        session.close()


def _seed_job(
    Session,
    file_id: str,
    *,
    status: str = "running",
    error_class: str | None = None,
    job_kind: str = "transcription",
) -> None:
    session = Session()
    try:
        session.add(JobRecord(
            file_id=file_id,
            job_kind=job_kind,
            provider="whisper_local",
            status=status,
            error_class=error_class,
        ))
        session.commit()
    finally:
        session.close()


def _stub_embeddings(monkeypatch, module) -> None:
    def _embed(texts):
        return [np.ones(4, dtype=np.float32) for _ in texts]

    monkeypatch.setattr(module, "embed_passages", _embed)


def _assert_flag(Session, file_id: str, flag: str, expected: bool) -> None:
    session = Session()
    try:
        row = session.query(IndexedFile).filter_by(file_id=file_id).one()
        assert getattr(row, flag) is expected
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Metadata batch
# ---------------------------------------------------------------------------


def test_spec_addon_057_i6_metadata_batch_posts_once_for_flags_turned_true(
    search_db, events, monkeypatch
) -> None:
    """SPEC-ADDON-057 I6: one metadata write posts one event for the ids whose flag became true."""
    from app.workers import metadata

    _seed_file(search_db, "m1", metadata_indexed=False)
    _seed_file(search_db, "m2", metadata_indexed=False)
    _stub_embeddings(monkeypatch, metadata)

    metadata.index_metadata_batch(["m1", "m2"])

    posted = _processing(events)
    assert len(posted) == 1, posted
    _assert_event(posted[0], {"m1", "m2"})
    _assert_flag(search_db, "m1", "metadata_indexed", True)
    _assert_flag(search_db, "m2", "metadata_indexed", True)


def test_spec_addon_057_i6_metadata_batch_posts_nothing_when_flag_unchanged(
    search_db, events, monkeypatch
) -> None:
    """SPEC-ADDON-057 I6: re-writing an already true metadata flag posts nothing."""
    from app.workers import metadata

    _seed_file(search_db, "m1", metadata_indexed=True)
    _seed_file(search_db, "m2", metadata_indexed=True)
    _stub_embeddings(monkeypatch, metadata)

    metadata.index_metadata_batch(["m1", "m2"])

    assert _processing(events) == []


def test_spec_addon_057_i6_metadata_batch_posts_per_drive(
    search_db, events, monkeypatch
) -> None:
    """SPEC-ADDON-057 I6: one event per write and drive, with that drive's ids only."""
    from app.workers import metadata

    _seed_file(search_db, "d1a", drive=DRIVE, metadata_indexed=False)
    _seed_file(search_db, "d1b", drive=DRIVE, metadata_indexed=False)
    _seed_file(search_db, "d2a", drive="d2", metadata_indexed=False)
    _stub_embeddings(monkeypatch, metadata)

    metadata.index_metadata_batch(["d1a", "d1b", "d2a"])

    posted = _processing(events)
    assert len(posted) == 2, posted
    by_drive = {body["drive"]: set(_ids(body)) for body in posted}
    assert by_drive == {
        DRIVE: {"d1a", "d1b"},
        "d2": {"d2a"},
    }
    for body in posted:
        assert len(_ids(body)) <= 100


def test_spec_addon_057_i6_metadata_batch_caps_each_event_at_100_ids(
    search_db, events, monkeypatch
) -> None:
    """SPEC-ADDON-057 I6: a write of more than 100 changed files is split, never over 100 per event."""
    from app.workers import metadata

    ids = [f"cap{i:03d}" for i in range(105)]
    for file_id in ids:
        _seed_file(search_db, file_id, metadata_indexed=False)
    _stub_embeddings(monkeypatch, metadata)

    metadata.index_metadata_batch(ids)

    posted = _processing(events)
    assert posted, "a write that changed 105 flags posted no event"
    covered: set[str] = set()
    for body in posted:
        assert body.get("drive") == DRIVE
        event_ids = _ids(body)
        assert len(event_ids) <= 100, body
        covered.update(event_ids)
    assert covered == set(ids)


# ---------------------------------------------------------------------------
# Text indexing
# ---------------------------------------------------------------------------


def _stub_text_extraction(monkeypatch) -> None:
    from app.workers import metadata

    monkeypatch.setattr(
        metadata,
        "_extract_file_content",
        lambda file: ExtractionResult(
            chunks=[TextChunk("text content to index")],
            extractor="text",
        ),
    )
    _stub_embeddings(monkeypatch, metadata)


def test_spec_addon_057_i6_text_indexing_posts_when_flag_turns_true(
    search_db, events, monkeypatch
) -> None:
    """SPEC-ADDON-057 I6: text_indexed false to true posts one event with the row's drive."""
    from app.workers import metadata

    _seed_file(
        search_db, "doc1", mime_type="text/plain", text_indexed=False
    )
    _stub_text_extraction(monkeypatch)

    assert metadata.index_text_content("doc1") is True

    posted = _processing(events)
    assert len(posted) == 1, posted
    _assert_event(posted[0], {"doc1"})
    _assert_flag(search_db, "doc1", "text_indexed", True)


def test_spec_addon_057_i6_text_indexing_posts_nothing_when_flag_unchanged(
    search_db, events, monkeypatch
) -> None:
    """SPEC-ADDON-057 I6: re-indexing an already text_indexed row posts nothing."""
    from app.workers import metadata

    _seed_file(
        search_db, "doc1", mime_type="text/plain", text_indexed=True
    )
    _stub_text_extraction(monkeypatch)

    assert metadata.index_text_content("doc1") is True

    assert _processing(events) == []


# ---------------------------------------------------------------------------
# Whisper: successful/silent transcription
# ---------------------------------------------------------------------------


def _provider(segments=None, error: Exception | None = None):
    provider = MagicMock()
    provider.name = "fake"
    provider.capabilities = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
        handles_own_retry=True,
    )
    if error is not None:
        provider.transcribe = AsyncMock(side_effect=error)
    else:
        provider.transcribe = AsyncMock(return_value=segments or [])
    return provider


def _run_whisper(monkeypatch, file_id: str, provider) -> bool:
    from app.workers import whisper

    _stub_embeddings(monkeypatch, whisper)
    with (
        patch.object(whisper, "get_provider", return_value=provider),
        patch.object(whisper, "validate_file_path", return_value=True),
        patch.object(whisper, "_index_tfidf_keywords", lambda *a, **k: None),
        patch(
            "app.policy_client.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
    ):
        import asyncio

        return asyncio.run(whisper.index_whisper(file_id))


def _speech_provider():
    words = "the quick brown fox jumps over the lazy dog again".split()
    return _provider(segments=[TranscriptionSegment(
        text=" ".join(words),
        start=0.0,
        end=float(len(words)),
        language="en",
        words=[
            WordToken(
                text=word,
                start=float(index),
                end=float(index) + 0.9,
            )
            for index, word in enumerate(words)
        ],
    )])


def test_spec_addon_057_i6_whisper_success_posts_when_flag_turns_true(
    search_db, events, monkeypatch
) -> None:
    """SPEC-ADDON-057 I6: a completed transcription that sets whisper_indexed posts one event."""
    _seed_file(search_db, "vid1", mime_type="video/mp4", whisper_indexed=False)

    assert _run_whisper(monkeypatch, "vid1", _speech_provider()) is True

    posted = _processing(events)
    assert len(posted) == 1, posted
    _assert_event(posted[0], {"vid1"})
    _assert_flag(search_db, "vid1", "whisper_indexed", True)


def test_spec_addon_057_i6_whisper_silent_file_posts_when_flag_turns_true(
    search_db, events, monkeypatch
) -> None:
    """SPEC-ADDON-057 I6: the zero-segment silent path still moves whisper_indexed false to true."""
    _seed_file(search_db, "silent1", mime_type="audio/wav", whisper_indexed=False)

    assert _run_whisper(monkeypatch, "silent1", _provider(segments=[])) is True

    posted = _processing(events)
    assert len(posted) == 1, posted
    _assert_event(posted[0], {"silent1"})
    _assert_flag(search_db, "silent1", "whisper_indexed", True)


def test_spec_addon_057_i6_whisper_success_posts_nothing_when_flag_unchanged(
    search_db, events, monkeypatch
) -> None:
    """SPEC-ADDON-057 I6: a successful re-run of an already whisper_indexed file posts nothing."""
    _seed_file(search_db, "vid1", mime_type="video/mp4", whisper_indexed=True)

    assert _run_whisper(monkeypatch, "vid1", _speech_provider()) is True

    assert _processing(events) == []


# ---------------------------------------------------------------------------
# Whisper: .loft and flag-only paths
# ---------------------------------------------------------------------------


def test_spec_addon_057_i6_loft_vtt_posts_when_flag_turns_true(
    search_db, events, monkeypatch, tmp_path
) -> None:
    """SPEC-ADDON-057 I6: the .loft adjacent-VTT path sets whisper_indexed and posts one event."""
    from app.workers.whisper import LOFT_MIME

    folder = tmp_path / "loft"
    folder.mkdir()
    loft = folder / "Talk.loft"
    loft.write_text("{}", encoding="utf-8")
    (folder / "Talk.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nthe transcript of the talk\n",
        encoding="utf-8",
    )
    _seed_file(
        search_db,
        "loft1",
        mime_type=LOFT_MIME,
        file_path=str(loft),
        whisper_indexed=False,
    )

    assert _run_whisper(monkeypatch, "loft1", _provider()) is True

    posted = _processing(events)
    assert len(posted) == 1, posted
    _assert_event(posted[0], {"loft1"})
    _assert_flag(search_db, "loft1", "whisper_indexed", True)


def test_spec_addon_057_i6_unsupported_mime_flag_only_posts_when_flag_turns_true(
    search_db, events, monkeypatch
) -> None:
    """SPEC-ADDON-057 I6: the unsupported-MIME flag-only path posts for whisper_indexed false to true."""
    _seed_file(search_db, "img1", mime_type="image/jpeg", whisper_indexed=False)

    assert _run_whisper(monkeypatch, "img1", _provider()) is True

    posted = _processing(events)
    assert len(posted) == 1, posted
    _assert_event(posted[0], {"img1"})
    _assert_flag(search_db, "img1", "whisper_indexed", True)


def test_spec_addon_057_i6_loft_consumed_temp_audio_posts_when_flag_turns_true(
    search_db, events
) -> None:
    """SPEC-ADDON-057 I6: the .loft consumed-temp-audio flag-only write posts one event."""
    from app.workers import whisper

    _seed_file(
        search_db,
        "loft2",
        mime_type=whisper.LOFT_MIME,
        whisper_indexed=False,
    )

    whisper._mark_loft_stt_temp_consumed("loft2")

    posted = _processing(events)
    assert len(posted) == 1, posted
    _assert_event(posted[0], {"loft2"})
    _assert_flag(search_db, "loft2", "whisper_indexed", True)


# ---------------------------------------------------------------------------
# Whisper: a transcription job ends failed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(FatalError("fatal"), id="FatalError"),
        pytest.param(TransientError("transient"), id="TransientError"),
        pytest.param(RateLimitError("rate"), id="RateLimitError"),
        pytest.param(CircuitBreakerOpen("open"), id="CircuitBreakerOpen"),
    ],
)
def test_spec_addon_057_i6_failed_transcription_job_posts(
    search_db, events, monkeypatch, error
) -> None:
    """SPEC-ADDON-057 I6: a transcription job that ends failed posts even when the whisper flag did not move."""
    _seed_file(search_db, "vid1", mime_type="video/mp4", whisper_indexed=False)

    assert _run_whisper(monkeypatch, "vid1", _provider(error=error)) is False

    posted = _processing(events)
    assert len(posted) == 1, posted
    _assert_event(posted[0], {"vid1"})
    _assert_flag(search_db, "vid1", "whisper_indexed", False)


def test_spec_addon_057_i6_container_restart_posts_when_running_job_ends_failed(
    search_db, events
) -> None:
    """SPEC-ADDON-057 I6: the startup hook that ends an orphaned running job failed posts."""
    from app.workers.whisper import fail_orphaned_running_jobs

    _seed_file(search_db, "vid1", mime_type="video/mp4", whisper_indexed=False)
    _seed_job(search_db, "vid1", status="running")

    fail_orphaned_running_jobs()

    posted = _processing(events)
    assert len(posted) == 1, posted
    _assert_event(posted[0], {"vid1"})


# ---------------------------------------------------------------------------
# Indexer resume
# ---------------------------------------------------------------------------


def _resume_manager(monkeypatch):
    from app.indexer import IndexManager

    manager = IndexManager()

    async def _noop_enqueue(task, *, force: bool = False) -> None:
        return None

    monkeypatch.setattr(manager, "_enqueue", _noop_enqueue)
    return manager


@pytest.mark.asyncio
async def test_spec_addon_057_i6_resume_posts_when_it_flips_a_flag_true(
    search_db, events, monkeypatch
) -> None:
    """SPEC-ADDON-057 I6: a resume write that sets text_indexed true posts one event."""
    _seed_file(
        search_db,
        "resume1",
        mime_type="audio/mpeg",
        metadata_indexed=True,
        clip_indexed=True,
        clip_thumbnail_indexed=True,
        whisper_indexed=False,
        text_indexed=False,
        tfidf_keywords_indexed=True,
    )
    manager = _resume_manager(monkeypatch)

    await manager._resume_incomplete()

    posted = _processing(events)
    assert len(posted) == 1, posted
    _assert_event(posted[0], {"resume1"})
    _assert_flag(search_db, "resume1", "text_indexed", True)


@pytest.mark.asyncio
async def test_spec_addon_057_i6_resume_posts_nothing_when_it_only_enqueues(
    search_db, events, monkeypatch
) -> None:
    """SPEC-ADDON-057 I6: a resume pass that changes no flag posts nothing."""
    _seed_file(
        search_db,
        "resume2",
        mime_type="video/mp4",
        metadata_indexed=False,
        clip_indexed=True,
        clip_thumbnail_indexed=True,
        whisper_indexed=False,
        text_indexed=True,
        tfidf_keywords_indexed=True,
    )
    manager = _resume_manager(monkeypatch)

    await manager._resume_incomplete()

    assert _processing(events) == []


# ---------------------------------------------------------------------------
# A cleared flag posts nothing
# ---------------------------------------------------------------------------


def test_spec_addon_057_i6_reset_loft_ref_posts_nothing_when_flag_cleared(
    search_db, events, tmp_path
) -> None:
    """SPEC-ADDON-057 I6: a reset that clears whisper_indexed posts nothing."""
    from app.indexer import IndexManager
    from app.workers.whisper import LOFT_MIME

    folder = tmp_path / "reset"
    folder.mkdir()
    loft = folder / "Talk.loft"
    loft.write_text("{}", encoding="utf-8")
    (folder / "Talk.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nnew captions\n",
        encoding="utf-8",
    )
    _seed_file(
        search_db,
        "loft3",
        mime_type=LOFT_MIME,
        file_path=str(loft),
        whisper_indexed=True,
    )

    IndexManager.__new__(IndexManager)._reset_loft_refs_with_new_vtt()

    _assert_flag(search_db, "loft3", "whisper_indexed", False)
    assert _processing(events) == []
