"""On-host transcription via faster-whisper (``whisper_local`` provider).

This module is a thin async-shaped wrapper around the existing
synchronous helpers in :mod:`app.workers.whisper`. The legacy code is
preserved verbatim — Phase 1B only adds the Provider surface; Phase 1C
will move ``_transcribe_file`` here and retire the legacy import path.

Design notes:

* ``_transcribe_file`` is CPU-bound (faster-whisper runs on a
  background thread internally but blocks the caller until done). We
  hand it to :func:`asyncio.to_thread` so it does not stall the
  intelligence indexer's event loop.
* Diarisation is not supported by faster-whisper, so every emitted
  ``WordToken.speaker_id`` is ``None``. The chunker treats ``None`` as
  "no speaker boundary", matching the legacy behaviour.
* The ``progress`` callback in the Protocol is intentionally ignored:
  faster-whisper has no streaming progress hook in our current
  pipeline, and wiring a fake "50% then 100%" would mislead the UI.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from app.workers.transcription.base import (
    ProviderCapabilities,
    TranscriptionSegment,
    WordToken,
)
from app.workers.whisper import _transcribe_file


def _segment_from_dict(raw: dict) -> TranscriptionSegment:
    """Convert a legacy ``_transcribe_file`` dict into a TranscriptionSegment.

    The legacy contract is documented in :func:`_transcribe_file`:
    ``{text, start, end, language, words: [{word, start, end}, ...]}``.
    Some segments (very short utterances under batched mode) come back
    without a ``words`` key; we treat that the same as an empty list.
    """
    raw_words = raw.get("words") or []
    words = [
        WordToken(
            text=str(w["word"]),
            start=float(w["start"]),
            end=float(w["end"]),
            speaker_id=None,
        )
        for w in raw_words
    ]
    return TranscriptionSegment(
        text=str(raw["text"]),
        start=float(raw["start"]),
        end=float(raw["end"]),
        language=str(raw.get("language", "")),
        words=words,
    )


class WhisperLocalProvider:
    """faster-whisper backed Provider (default, on-host)."""

    name = "whisper_local"
    capabilities = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
        # faster-whisper's decode_audio() loads the full file into a
        # single float32 numpy array before transcribing, so memory
        # scales linearly with duration. A 9 h video peaked at ~14 GB
        # and crash-looped the container. 50 MB of normalized 16 kHz
        # mono FLAC (~20 KB/s × 0.8 safety) yields ~33 min chunks,
        # capping per-chunk peak at a few hundred MB.
        max_input_bytes=50 * 1024 * 1024,
        accepts_initial_prompt=True,    # forwards to faster-whisper's prompt
        handles_own_retry=False,
    )

    async def transcribe(
        self,
        file_path: str,
        *,
        language_hint: str | None = None,
        hotwords: list[str] | None = None,
        initial_prompt: str | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> list[TranscriptionSegment]:
        """Run faster-whisper off the event-loop thread.

        ``language_hint`` and ``hotwords`` are accepted for Protocol
        conformance but ignored here — the existing
        :func:`_transcribe_file` already does its own language
        detection and reads ``settings.transcription.whisper_local`` for
        the language-default prompt. Wiring per-call hints would
        require a deeper refactor that belongs in Phase 1C / 2.

        ``initial_prompt`` is forwarded as a precedence-1 override
        (Phase 2B): when a non-empty string is supplied the caller's
        value replaces both the user-configured override and the
        per-language default; when ``None`` / empty the legacy
        resolution chain runs unchanged so non-chunked transcription
        keeps its existing behaviour.
        """
        raw = await asyncio.to_thread(
            _transcribe_file, file_path, initial_prompt
        )
        return [_segment_from_dict(seg) for seg in raw]
