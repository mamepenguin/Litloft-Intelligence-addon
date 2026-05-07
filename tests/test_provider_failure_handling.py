"""Phase 1C tests: ``transcribe_with_retry`` + JobRecord lifecycle.

Spec ``2026-05-07-cloud-transcription-providers.md`` §"Retry policy"
§"429 連続時の circuit breaker" §"_index_whisper_sync の戻り値ロジック修正".

This file pins the retry budget, fatal-vs-transient classification,
the circuit-breaker hand-off, AND the integration with
``index_whisper`` (provider error → JobRecord.failed →
whisper_indexed stays False → no requeue_after_whisper).
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub heavy ML deps before app imports.
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
from app.models import IndexedFile, JobRecord  # noqa: E402,F401

from app.workers.transcription.base import (  # noqa: E402
    ProviderCapabilities,
    TranscriptionSegment,
    WordToken,
)
from app.workers.transcription.circuit_breaker import (  # noqa: E402
    ProviderCircuitBreaker,
)
from app.workers.transcription.errors import (  # noqa: E402
    FatalError,
    RateLimitError,
    TransientError,
)
from app.workers.transcription.retry import (  # noqa: E402
    CircuitBreakerOpen,
    transcribe_with_retry,
)


# --------------------------------------------------------------------------
# transcribe_with_retry behaviour
# --------------------------------------------------------------------------


class _ProgrammableProvider:
    """Provider whose ``transcribe`` returns / raises a fixed sequence."""

    name = "fake"
    capabilities = ProviderCapabilities(
        sends_audio_offhost=True,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )

    def __init__(self, sequence: list) -> None:
        self._sequence = list(sequence)
        self.calls = 0

    async def transcribe(self, file_path: str, **kwargs):
        self.calls += 1
        item = self._sequence.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture()
def fake_sleep(monkeypatch):
    """Replace asyncio.sleep with a no-op coroutine."""

    async def _noop(*args, **kwargs):
        return None

    return _noop


@pytest.mark.asyncio
async def test_succeeds_first_attempt(fake_sleep) -> None:
    breaker = ProviderCircuitBreaker(threshold=10, window_s=60, pause_s=60)
    seg = TranscriptionSegment(
        text="hi", start=0.0, end=1.0, language="en", words=[]
    )
    provider = _ProgrammableProvider([[seg]])
    result = await transcribe_with_retry(
        provider,
        "/tmp/a.wav",
        circuit_breaker=breaker,
        sleep=fake_sleep,
    )
    assert result == [seg]
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_transient_then_success(fake_sleep) -> None:
    breaker = ProviderCircuitBreaker(threshold=10, window_s=60, pause_s=60)
    seg = TranscriptionSegment(
        text="hi", start=0.0, end=1.0, language="en", words=[]
    )
    provider = _ProgrammableProvider([
        TransientError("blip"),
        TransientError("blip"),
        [seg],
    ])
    result = await transcribe_with_retry(
        provider,
        "/tmp/a.wav",
        circuit_breaker=breaker,
        sleep=fake_sleep,
    )
    assert result == [seg]
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_transient_exhausts_budget_and_raises(fake_sleep) -> None:
    breaker = ProviderCircuitBreaker(threshold=10, window_s=60, pause_s=60)
    provider = _ProgrammableProvider([
        TransientError("blip"),
        TransientError("blip"),
        TransientError("blip"),
    ])
    with pytest.raises(TransientError):
        await transcribe_with_retry(
            provider,
            "/tmp/a.wav",
            circuit_breaker=breaker,
            sleep=fake_sleep,
        )
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_fatal_short_circuits_no_retry(fake_sleep) -> None:
    breaker = ProviderCircuitBreaker(threshold=10, window_s=60, pause_s=60)
    provider = _ProgrammableProvider([
        FatalError("401"),
    ])
    with pytest.raises(FatalError):
        await transcribe_with_retry(
            provider,
            "/tmp/a.wav",
            circuit_breaker=breaker,
            sleep=fake_sleep,
        )
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_rate_limit_records_failure_and_retries(fake_sleep) -> None:
    breaker = ProviderCircuitBreaker(threshold=10, window_s=60, pause_s=60)
    seg = TranscriptionSegment(
        text="hi", start=0.0, end=1.0, language="en", words=[]
    )
    provider = _ProgrammableProvider([
        RateLimitError("429"),
        [seg],
    ])
    result = await transcribe_with_retry(
        provider,
        "/tmp/a.wav",
        circuit_breaker=breaker,
        sleep=fake_sleep,
    )
    assert result == [seg]
    # Breaker recorded one 429.
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_breaker_open_short_circuits(fake_sleep) -> None:
    """An already-open breaker must skip provider.transcribe entirely."""
    breaker = ProviderCircuitBreaker(threshold=1, window_s=60, pause_s=60)
    # Prime the breaker.
    breaker.record_failure("fake")
    breaker.record_failure("fake")
    assert breaker.is_open("fake") is True

    provider = _ProgrammableProvider([])
    with pytest.raises(CircuitBreakerOpen):
        await transcribe_with_retry(
            provider,
            "/tmp/a.wav",
            circuit_breaker=breaker,
            sleep=fake_sleep,
        )
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_rate_limit_trips_breaker_during_retry(fake_sleep) -> None:
    """Repeated 429s should trip the breaker mid-retry and abort."""
    breaker = ProviderCircuitBreaker(threshold=1, window_s=60, pause_s=60)
    provider = _ProgrammableProvider([
        RateLimitError("429"),
        RateLimitError("429"),
        RateLimitError("429"),
    ])
    with pytest.raises(CircuitBreakerOpen):
        await transcribe_with_retry(
            provider,
            "/tmp/a.wav",
            circuit_breaker=breaker,
            sleep=fake_sleep,
        )


# --------------------------------------------------------------------------
# index_whisper integration: failure → JobRecord.failed, whisper_indexed=False
# --------------------------------------------------------------------------


@pytest.fixture()
def search_engine(tmp_path):
    """Search DB with the schema needed by _index_whisper integration."""
    db_path = tmp_path / "search.db"
    eng = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(eng)
    # FTS5 mirrors used by upsert_fts_transcripts / delete_fts_transcripts;
    # missing in test schema by default. Create as regular tables since
    # we don't exercise FTS5 features here.
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
    return eng


@pytest.fixture()
def patched_db(monkeypatch, search_engine):
    Session = sessionmaker(bind=search_engine, expire_on_commit=False)

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


def _seed_file(Session, *, file_id="f00000000001", drive="d1") -> None:
    s = Session()
    s.add(IndexedFile(
        file_id=file_id,
        drive=drive,
        filename="x.mp4",
        file_path="/drives/d1/x.mp4",
        file_type="video",
        mime_type="video/mp4",
        file_size=1000,
        active=True,
        metadata_indexed=True,
        clip_indexed=False,
        whisper_indexed=False,
        text_indexed=False,
        title="x",
        description="",
        tags_text="",
    ))
    s.commit()
    s.close()


@pytest.mark.asyncio
async def test_provider_failure_writes_failed_jobrecord(patched_db) -> None:
    """A FatalError from the provider must:
    * write JobRecord.status='failed' with error_class
    * leave whisper_indexed=False (so the file can be re-indexed later)
    * return False so ``requeue_after_whisper`` is NOT called
    """
    from app.models import IndexedFile, JobRecord
    from app.workers import whisper as whisper_module

    _seed_file(patched_db)

    failing_provider = MagicMock()
    failing_provider.name = "fake"
    failing_provider.capabilities = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )
    failing_provider.transcribe = AsyncMock(side_effect=FatalError("401 unauthorized"))

    with (
        patch.object(whisper_module, "get_provider", return_value=failing_provider),
        patch.object(whisper_module, "validate_file_path", return_value=True),
        patch(
            "app.policy_client.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
    ):
        result = await whisper_module.index_whisper("f00000000001")

    assert result is False

    s = patched_db()
    try:
        file = s.query(IndexedFile).filter_by(file_id="f00000000001").one()
        assert file.whisper_indexed is False
        records = s.query(JobRecord).filter_by(file_id="f00000000001").all()
    finally:
        s.close()
    assert len(records) == 1
    assert records[0].status == "failed"
    assert records[0].error_class == "FatalError"
    assert records[0].provider == "fake"
    assert records[0].completed_at is not None


@pytest.mark.asyncio
async def test_provider_zero_segments_writes_succeeded_jobrecord(patched_db) -> None:
    """Empty / silent file is a *succeeded* run with 0 segments.

    The legacy path also flipped whisper_indexed=True for empty
    results — that behaviour is preserved (silent video shouldn't
    keep getting re-attempted).
    """
    from app.models import IndexedFile, JobRecord
    from app.workers import whisper as whisper_module

    _seed_file(patched_db)

    silent_provider = MagicMock()
    silent_provider.name = "whisper_local"
    silent_provider.capabilities = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )
    silent_provider.transcribe = AsyncMock(return_value=[])

    with (
        patch.object(whisper_module, "get_provider", return_value=silent_provider),
        patch.object(whisper_module, "validate_file_path", return_value=True),
        patch(
            "app.policy_client.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
    ):
        result = await whisper_module.index_whisper("f00000000001")

    assert result is True
    s = patched_db()
    try:
        file = s.query(IndexedFile).filter_by(file_id="f00000000001").one()
        assert file.whisper_indexed is True
        records = s.query(JobRecord).filter_by(file_id="f00000000001").all()
    finally:
        s.close()
    assert len(records) == 1
    assert records[0].status == "succeeded"
    assert records[0].error_class is None


@pytest.fixture()
def settings_with_provider(monkeypatch):
    """Replace ``app.workers.whisper.settings`` with a mutable proxy.

    ``Settings`` and ``TranscriptionConfig`` are frozen dataclasses, so
    we swap the module-level reference for a ``SimpleNamespace`` whose
    ``transcription`` attribute is mutable. Tests then mutate
    ``transcription.provider`` etc. directly.
    """
    from types import SimpleNamespace

    from app.config import settings as real_settings

    proxy_transcription = SimpleNamespace(
        provider=real_settings.transcription.provider,
        language_hint=real_settings.transcription.language_hint,
        hotwords=real_settings.transcription.hotwords,
        whisper_local=real_settings.transcription.whisper_local,
        openai_compatible=real_settings.transcription.openai_compatible,
        deepgram=real_settings.transcription.deepgram,
        elevenlabs_scribe=real_settings.transcription.elevenlabs_scribe,
    )
    proxy_settings = SimpleNamespace(
        transcription=proxy_transcription,
        indexing=real_settings.indexing,
    )
    monkeypatch.setattr("app.workers.whisper.settings", proxy_settings)
    return proxy_transcription


@pytest.mark.asyncio
async def test_cloud_provider_falls_back_when_policy_off(
    patched_db, settings_with_provider
) -> None:
    """``transcription_cloud=false`` must force whisper_local fallback.

    The cloud provider's transcribe must NEVER be called when policy
    says no.
    """
    from app.workers import whisper as whisper_module

    _seed_file(patched_db)
    settings_with_provider.provider = "deepgram"

    cloud_provider = MagicMock()
    cloud_provider.name = "deepgram"
    cloud_provider.capabilities = ProviderCapabilities(
        sends_audio_offhost=True,
        supports_diarization=True,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )
    cloud_provider.transcribe = AsyncMock(return_value=[])

    local_provider = MagicMock()
    local_provider.name = "whisper_local"
    local_provider.capabilities = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )
    local_provider.transcribe = AsyncMock(return_value=[])

    def _get_provider(name: str):
        if name == "whisper_local":
            return local_provider
        return cloud_provider

    async def _policy(drive, feature, *, default_on_failure=True):
        if feature == "transcription_cloud":
            return False
        if feature == "index":
            return True
        return True

    with (
        patch.object(whisper_module, "get_provider", side_effect=_get_provider),
        patch.object(whisper_module, "validate_file_path", return_value=True),
        patch("app.policy_client.is_feature_enabled", new=AsyncMock(side_effect=_policy)),
    ):
        await whisper_module.index_whisper("f00000000001")

    cloud_provider.transcribe.assert_not_awaited()
    local_provider.transcribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_index_off_skips_transcription_entirely(patched_db) -> None:
    """Layer 2 gate: ``intelligence.index=false`` returns False without
    calling the provider at all (re-evaluation at dequeue)."""
    from app.workers import whisper as whisper_module

    _seed_file(patched_db)

    provider = MagicMock()
    provider.name = "whisper_local"
    provider.capabilities = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )
    provider.transcribe = AsyncMock(return_value=[])

    async def _policy(drive, feature, *, default_on_failure=True):
        if feature == "index":
            return False
        return True

    with (
        patch.object(whisper_module, "get_provider", return_value=provider),
        patch.object(whisper_module, "validate_file_path", return_value=True),
        patch("app.policy_client.is_feature_enabled", new=AsyncMock(side_effect=_policy)),
    ):
        result = await whisper_module.index_whisper("f00000000001")

    assert result is False
    provider.transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_emits_transcription_completed_on_success(patched_db) -> None:
    """Successful transcription emits intelligence.transcription.completed."""
    from app.workers import whisper as whisper_module

    _seed_file(patched_db)

    seg = TranscriptionSegment(
        text="hello", start=0.0, end=1.0, language="en",
        words=[WordToken(text="hello", start=0.0, end=1.0)],
    )

    provider = MagicMock()
    provider.name = "whisper_local"
    provider.capabilities = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )
    provider.transcribe = AsyncMock(return_value=[seg])

    emitted: list[tuple[str, dict]] = []

    async def _capture(event, data):
        emitted.append((event, data))

    with (
        patch.object(whisper_module, "get_provider", return_value=provider),
        patch.object(whisper_module, "validate_file_path", return_value=True),
        patch.object(whisper_module, "_emit_ws_event", new=_capture),
        patch.object(whisper_module, "embed_passages", return_value=None),
        patch(
            "app.policy_client.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
    ):
        await whisper_module.index_whisper("f00000000001")

    completed = [e for e in emitted if e[0] == "intelligence.transcription.completed"]
    assert completed, f"expected completed event in {emitted!r}"
    payload = completed[0][1]
    assert payload["file_id"] == "f00000000001"
    assert payload["provider"] == "whisper_local"
    assert payload["has_diarization"] is False


@pytest.mark.asyncio
async def test_emits_transcription_failed_on_provider_error(
    patched_db, settings_with_provider
) -> None:
    """Provider error emits intelligence.transcription.failed."""
    from app.workers import whisper as whisper_module

    _seed_file(patched_db)
    settings_with_provider.provider = "deepgram"

    provider = MagicMock()
    provider.name = "deepgram"
    provider.capabilities = ProviderCapabilities(
        sends_audio_offhost=False,  # disable cloud-policy gate for this test
        supports_diarization=True,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )
    provider.transcribe = AsyncMock(side_effect=FatalError("401"))

    emitted: list[tuple[str, dict]] = []

    async def _capture(event, data):
        emitted.append((event, data))

    with (
        patch.object(whisper_module, "get_provider", return_value=provider),
        patch.object(whisper_module, "validate_file_path", return_value=True),
        patch.object(whisper_module, "_emit_ws_event", new=_capture),
        patch(
            "app.policy_client.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
    ):
        await whisper_module.index_whisper("f00000000001")

    failed = [e for e in emitted if e[0] == "intelligence.transcription.failed"]
    assert failed, f"expected failed event in {emitted!r}"
    payload = failed[0][1]
    assert payload["file_id"] == "f00000000001"
    assert payload["provider"] == "deepgram"
    assert payload["error_class"] == "FatalError"
