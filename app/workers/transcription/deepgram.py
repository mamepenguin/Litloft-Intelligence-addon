"""Deepgram transcription provider.

Calls Deepgram's pre-recorded ``/v1/listen`` endpoint via direct HTTP
(no SDK — :mod:`httpx` is already a transitive dependency, and the
SDK adds a non-trivial import surface for very little). Diarisation
is enabled by default; speaker labels are forwarded to the indexer
through ``WordToken.speaker_id``.

Spec: ``2026-05-07-cloud-transcription-providers.md``.

Phase 1B contract:

* The provider returns **one** :class:`TranscriptionSegment` containing
  every word from channel 0. The Phase 1C chunker is responsible for
  re-segmenting on speaker change / silence / punctuation; this keeps
  the provider boundary purely "faithfully report what the API said".
* Multichannel responses are clipped to channel 0. Litloft is mono-
  audio centric (extracted with ffmpeg before transcription); doubling
  up channel 1 would emit duplicate words.
* ``punctuated_word`` is preferred over ``word`` for the token text so
  downstream search and subtitle rendering get proper casing /
  punctuation. Fallback to ``word`` keeps things working with
  ``smart_format=false`` configurations.
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

DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"

_FATAL_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 409, 413, 415, 422})


class DeepgramProvider:
    """Cloud transcription via Deepgram /v1/listen."""

    name = "deepgram"
    capabilities = ProviderCapabilities(
        sends_audio_offhost=True,
        supports_diarization=True,
        supports_hotwords=False,
        supports_word_timestamps=True,
        max_input_bytes=None,           # Deepgram has no practical cap
        accepts_initial_prompt=False,
        handles_own_retry=False,
    )

    def __init__(self) -> None:
        api_key = os.getenv("DEEPGRAM_API_KEY", "")
        if not api_key:
            raise FatalError(
                "DEEPGRAM_API_KEY not configured. "
                "Set the env var to enable the deepgram transcription "
                "provider."
            )
        cfg = config.settings.transcription.deepgram
        self._api_key = api_key
        self._model = cfg.model
        self._diarize = cfg.diarize
        self._smart_format = cfg.smart_format
        self._detect_language = cfg.detect_language
        self._timeout_s = cfg.timeout_s
        # The previous design stored an ``httpx.AsyncClient`` on the
        # instance. Combined with ``get_provider()`` returning a fresh
        # provider per request, this leaked sockets/fds under batch
        # indexing because ``aclose()`` was never called. We now build
        # a short-lived client inside ``transcribe()`` via ``async
        # with`` so the socket pool is released between jobs (hako
        # pattern ``W0F1YQspXF-lVYgaDb6V1``).
        # ``_transport`` is a test-only injection point for
        # ``httpx.MockTransport`` instances.
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
        # Deepgram has no "prior text" channel, so initial_prompt is
        # ignored. Capability matrix flags this via
        # ``accepts_initial_prompt=False``.
        del progress, hotwords, initial_prompt

        params = {
            "model": self._model,
            "diarize": "true" if self._diarize else "false",
            "smart_format": "true" if self._smart_format else "false",
            "punctuate": "true",
            "utterances": "false",
        }
        # Deepgram offers EITHER detect_language OR language=xx, not
        # both. A language_hint always wins because the caller is more
        # informed than the model.
        if language_hint:
            params["language"] = language_hint
        elif self._detect_language:
            params["detect_language"] = "true"

        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "application/octet-stream",
        }

        try:
            with open(file_path, "rb") as audio:
                body = audio.read()
            client_kwargs: dict = {"timeout": float(self._timeout_s)}
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(
                    DEEPGRAM_LISTEN_URL,
                    params=params,
                    headers=headers,
                    content=body,
                )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise TransientError(f"Deepgram network error: {exc}") from exc
        except OSError as exc:
            raise FatalError(f"Cannot read audio file {file_path}: {exc}") from exc

        if response.status_code == 429:
            raise RateLimitError(
                f"Deepgram rate limit (429): {response.text[:200]}"
            )
        if response.status_code in _FATAL_STATUS_CODES:
            raise FatalError(
                f"Deepgram HTTP {response.status_code}: {response.text[:200]}"
            )
        if 500 <= response.status_code < 600:
            raise TransientError(
                f"Deepgram HTTP {response.status_code}: {response.text[:200]}"
            )
        if response.status_code != 200:
            raise FatalError(
                f"Deepgram unexpected HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        return _parse_response(response.json())


def _parse_response(payload: dict) -> list[TranscriptionSegment]:
    """Convert a Deepgram /v1/listen JSON body into TranscriptionSegments.

    See the docstring in :class:`DeepgramProvider` for the contract:
    one segment per call, channel 0 only, ``speaker`` (int) is
    stringified into ``speaker_id``.
    """
    channels = (payload.get("results") or {}).get("channels") or []
    if not channels:
        return []
    alternatives = channels[0].get("alternatives") or []
    if not alternatives:
        return []
    alt = alternatives[0]
    raw_words = alt.get("words") or []
    if not raw_words:
        # Silence / no speech — succeeded with zero words. Same
        # treatment as faster-whisper's empty result: the indexer
        # marks the file ``whisper_indexed=True`` without writing
        # chunks.
        return []

    language = (
        channels[0].get("detected_language")
        or (payload.get("metadata") or {}).get("detected_language")
        or ""
    )

    words: list[WordToken] = []
    for w in raw_words:
        text = (w.get("punctuated_word") or w.get("word") or "").strip()
        if not text:
            continue
        speaker = w.get("speaker")
        words.append(
            WordToken(
                text=text,
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                speaker_id=str(speaker) if speaker is not None else None,
            )
        )
    if not words:
        return []

    transcript_text = (
        alt.get("transcript")
        or " ".join(w.text for w in words)
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
