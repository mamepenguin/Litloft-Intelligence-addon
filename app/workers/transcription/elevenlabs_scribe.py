"""ElevenLabs Scribe transcription provider.

Calls ElevenLabs' speech-to-text endpoint via direct httpx multipart
upload. Same shape as :class:`DeepgramProvider`: returns one
:class:`TranscriptionSegment` per call with all words; chunking and
speaker-boundary segmentation belong to Phase 1C.

Spec: ``2026-05-07-cloud-transcription-providers.md``.

Notes:

* Authentication uses the ``xi-api-key`` HTTP header — this is
  ElevenLabs' standard for every product (TTS, voice cloning, STT).
* Scribe's response interleaves entries with ``type`` of ``"word"``,
  ``"spacing"``, and ``"audio_event"`` (laughter, music, etc.). We
  forward only ``"word"`` rows: spacing tokens carry no speech
  content; audio events would pollute the transcript with bracketed
  pseudo-words downstream search has no use for. Phase 2 may surface
  audio events separately.
* Speaker IDs are already strings (``"speaker_0"``); we pass them
  through verbatim.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import httpx

import app.config as config
from app.workers.transcription.base import (
    ProviderCapabilities,
    TranscriptionSegment,
    WordToken,
)
from app.workers.transcription.errors import (
    FatalError,
    RateLimitError,
    TransientError,
)

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
_FATAL_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 409, 413, 415, 422})


class ElevenLabsScribeProvider:
    """Cloud transcription via ElevenLabs Scribe."""

    name = "elevenlabs_scribe"
    capabilities = ProviderCapabilities(
        sends_audio_offhost=True,
        supports_diarization=True,
        supports_hotwords=False,
        supports_word_timestamps=True,
        max_input_bytes=None,
        accepts_initial_prompt=False,
        handles_own_retry=False,
    )

    def __init__(self) -> None:
        api_key = os.getenv("ELEVENLABS_API_KEY", "")
        if not api_key:
            raise FatalError(
                "ELEVENLABS_API_KEY not configured. "
                "Set the env var to enable the elevenlabs_scribe "
                "transcription provider."
            )
        cfg = config.settings.transcription.elevenlabs_scribe
        self._api_key = api_key
        self._model_id = cfg.model_id
        self._diarize = cfg.diarize
        self._timeout_s = cfg.timeout_s
        # See DeepgramProvider — short-lived ``httpx.AsyncClient`` per
        # call (hako ``W0F1YQspXF-lVYgaDb6V1``). ``_transport`` is the
        # test-only injection slot.
        self._transport: httpx.BaseTransport | None = None

    async def transcribe(
        self,
        file_path: str,
        *,
        language_hint: str | None = None,
        hotwords: list[str] | None = None,
        initial_prompt: str | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> list[TranscriptionSegment]:
        del progress, hotwords, initial_prompt

        data = {
            "model_id": self._model_id,
            "diarize": "true" if self._diarize else "false",
            "timestamps_granularity": "word",
        }
        if language_hint:
            data["language_code"] = language_hint
        headers = {"xi-api-key": self._api_key}

        try:
            client_kwargs: dict = {"timeout": float(self._timeout_s)}
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
            with open(file_path, "rb") as audio:
                files = {
                    "file": (os.path.basename(file_path), audio, "application/octet-stream"),
                }
                async with httpx.AsyncClient(**client_kwargs) as client:
                    response = await client.post(
                        ELEVENLABS_STT_URL,
                        headers=headers,
                        data=data,
                        files=files,
                    )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise TransientError(
                f"ElevenLabs Scribe network error: {exc}"
            ) from exc
        except OSError as exc:
            raise FatalError(
                f"Cannot read audio file {file_path}: {exc}"
            ) from exc

        if response.status_code == 429:
            raise RateLimitError(
                f"ElevenLabs Scribe rate limit (429): {response.text[:200]}"
            )
        if response.status_code in _FATAL_STATUS_CODES:
            raise FatalError(
                f"ElevenLabs Scribe HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        if 500 <= response.status_code < 600:
            raise TransientError(
                f"ElevenLabs Scribe HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        if response.status_code != 200:
            raise FatalError(
                f"ElevenLabs Scribe unexpected HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        return _parse_response(response.json())


def _parse_response(payload: dict) -> list[TranscriptionSegment]:
    raw_words = payload.get("words") or []
    if not raw_words:
        return []

    language = payload.get("language_code") or ""

    words: list[WordToken] = []
    for w in raw_words:
        # Skip non-speech entries (spacing / audio_event). Some
        # responses omit ``type`` entirely on word entries; default to
        # ``"word"`` so we do not silently drop legitimate words from
        # older API versions.
        kind = w.get("type", "word")
        if kind != "word":
            continue
        text = (w.get("text") or "").strip()
        if not text:
            continue
        words.append(
            WordToken(
                text=text,
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                speaker_id=w.get("speaker_id"),
            )
        )
    if not words:
        return []

    transcript_text = (
        payload.get("text") or " ".join(w.text for w in words)
    )
    return [
        TranscriptionSegment(
            text=transcript_text,
            start=words[0].start,
            end=words[-1].end,
            language=language,
            words=words,
        )
    ]
