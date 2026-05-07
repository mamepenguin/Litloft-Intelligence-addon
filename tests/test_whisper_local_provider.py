"""Tests for :class:`WhisperLocalProvider`.

The provider is a thin async wrapper over the existing synchronous
``_transcribe_file`` from ``app.workers.whisper``. We patch that
helper to keep the test surface tiny — the actual faster-whisper
behaviour is covered by the legacy whisper tests; here we only verify:

* Protocol conformance (name / capabilities / async transcribe shape)
* Conversion from the legacy ``list[dict]`` shape to
  ``list[TranscriptionSegment]`` with ``WordToken(speaker_id=None)``
* The wrapper genuinely runs the sync work off the event-loop thread
  (``asyncio.to_thread``) so heavy CPU work cannot starve the loop
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from app.workers.transcription import (
    ProviderCapabilities,
    TranscriptionProvider,
    TranscriptionSegment,
    WordToken,
)
from app.workers.transcription.whisper_local import WhisperLocalProvider


def test_provider_declared_name() -> None:
    provider = WhisperLocalProvider()
    assert provider.name == "whisper_local"


def test_provider_capabilities_match_spec() -> None:
    """``whisper_local`` runs entirely on-host; no diarization / hotwords."""
    provider = WhisperLocalProvider()
    assert provider.capabilities == ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )


def test_provider_satisfies_protocol() -> None:
    """``TranscriptionProvider`` is a Protocol — assignment is the check."""
    provider: TranscriptionProvider = WhisperLocalProvider()
    assert provider.name == "whisper_local"


@pytest.mark.asyncio
async def test_transcribe_converts_legacy_dicts_to_segments() -> None:
    """Legacy ``_transcribe_file`` returns dicts; provider converts them.

    The dict contract is the one used inside ``whisper.py``:
    ``{text, start, end, language, words: [{word, start, end}, ...]}``.
    Each dict becomes a ``TranscriptionSegment``; each word dict becomes
    a ``WordToken`` with ``speaker_id=None`` (this provider does not
    diarize).
    """
    canned = [
        {
            "text": "hello world",
            "start": 0.0,
            "end": 1.5,
            "language": "en",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.5},
            ],
        },
        {
            "text": "second segment",
            "start": 1.5,
            "end": 3.0,
            "language": "en",
            "words": [
                {"word": "second", "start": 1.5, "end": 2.2},
                {"word": "segment", "start": 2.2, "end": 3.0},
            ],
        },
    ]

    with patch(
        "app.workers.transcription.whisper_local._transcribe_file",
        return_value=canned,
    ):
        provider = WhisperLocalProvider()
        result = await provider.transcribe("/tmp/fake.wav")

    assert len(result) == 2
    assert all(isinstance(seg, TranscriptionSegment) for seg in result)
    assert result[0].text == "hello world"
    assert result[0].start == 0.0
    assert result[0].end == 1.5
    assert result[0].language == "en"
    assert result[0].words == [
        WordToken(text="hello", start=0.0, end=0.5, speaker_id=None),
        WordToken(text="world", start=0.5, end=1.5, speaker_id=None),
    ]
    assert all(w.speaker_id is None for seg in result for w in seg.words)


@pytest.mark.asyncio
async def test_transcribe_handles_segments_without_words() -> None:
    """Batched mode occasionally omits ``words`` on tiny segments.

    The provider must tolerate a missing ``words`` key (and also an
    empty list) by emitting a segment with ``words=[]``.
    """
    canned = [
        {"text": "uh", "start": 0.0, "end": 0.1, "language": "en"},
        {
            "text": "hi",
            "start": 0.1,
            "end": 0.3,
            "language": "en",
            "words": [],
        },
    ]
    with patch(
        "app.workers.transcription.whisper_local._transcribe_file",
        return_value=canned,
    ):
        provider = WhisperLocalProvider()
        result = await provider.transcribe("/tmp/fake.wav")

    assert len(result) == 2
    assert result[0].words == []
    assert result[1].words == []


@pytest.mark.asyncio
async def test_transcribe_returns_empty_list_when_underlying_returns_empty() -> None:
    """Silence / VAD-rejected audio yields ``[]`` — the provider must propagate."""
    with patch(
        "app.workers.transcription.whisper_local._transcribe_file",
        return_value=[],
    ):
        provider = WhisperLocalProvider()
        assert await provider.transcribe("/tmp/silence.wav") == []


@pytest.mark.asyncio
async def test_transcribe_runs_sync_helper_in_thread() -> None:
    """``_transcribe_file`` is CPU-bound — must NOT run in the loop thread.

    We capture the thread that runs the patched helper and assert it is
    not the main thread. This is what protects the indexer event loop
    from blocking under heavy CPU.
    """
    main_thread = threading.get_ident()
    captured: dict[str, int] = {}

    def fake(file_path: str) -> list[dict]:
        captured["thread"] = threading.get_ident()
        return []

    with patch(
        "app.workers.transcription.whisper_local._transcribe_file",
        side_effect=fake,
    ):
        provider = WhisperLocalProvider()
        await provider.transcribe("/tmp/x.wav")

    assert captured["thread"] != main_thread


@pytest.mark.asyncio
async def test_transcribe_ignores_progress_callback() -> None:
    """The Protocol allows ``progress`` for streaming providers; whisper_local
    runs to completion in a single shot, so the callback is unused but must
    not break anything when supplied."""
    calls: list[float] = []

    with patch(
        "app.workers.transcription.whisper_local._transcribe_file",
        return_value=[],
    ):
        provider = WhisperLocalProvider()
        await provider.transcribe(
            "/tmp/x.wav",
            progress=lambda f: calls.append(f),
        )

    # whisper_local has no streaming progress; the callback is decorative
    # in Phase 1 (per spec). It must remain unused, not crash.
    assert calls == []


@pytest.mark.asyncio
async def test_transcribe_passes_file_path_through() -> None:
    """The wrapper must forward ``file_path`` verbatim to the legacy helper."""
    seen: dict[str, str] = {}

    def fake(file_path: str) -> list[dict]:
        seen["path"] = file_path
        return []

    with patch(
        "app.workers.transcription.whisper_local._transcribe_file",
        side_effect=fake,
    ):
        provider = WhisperLocalProvider()
        await provider.transcribe("/drives/foo/bar.mp4")

    assert seen["path"] == "/drives/foo/bar.mp4"
