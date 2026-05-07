"""OpenAI-compatible Whisper transcription provider.

Works against any backend that implements OpenAI's
``/audio/transcriptions`` API: the official OpenAI endpoint, plus
self-hosted / managed compatible backends like Groq, Fireworks, and
vLLM. Selection is via :data:`settings.transcription.openai_compatible.base_url`.

Spec: ``2026-05-07-cloud-transcription-providers.md``.

Design notes:

* The SDK is initialised lazily inside ``__init__`` so a mistyped
  ``base_url`` still surfaces at startup, not the first request.
* ``OPENAI_API_KEY`` env is required; we raise :class:`FatalError`
  rather than substituting ``"not-needed"`` because the spec demands
  fail-loud behaviour (the LLM client can use a placeholder because it
  also runs against ollama with no auth).
* The 25 MB pre-check is gated on the official OpenAI hostname only —
  Groq / Fireworks / self-hosted endpoints have no equivalent cap and
  silently blocking 26 MB clips on those backends would be wrong.
* OpenAI's ``timestamp_granularities=["word"]`` is honoured by the
  official endpoint and most well-maintained compatibles, but at least
  one Groq variant has been observed returning empty ``words`` arrays.
  We treat that as :class:`FatalError` so the indexer surfaces the
  problem in ``JobRecord`` instead of silently producing search results
  with no word-level seek data.
* HTTP failure classification mirrors the LLM client's mapping (5xx /
  network / timeout = transient, 429 = rate limit, the rest = fatal).
"""

from __future__ import annotations

import os
from collections.abc import Callable

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

# Imported lazily inside __init__ so unit tests that patch ``settings``
# / monkeypatch env do not pay an SDK import cost when only the type
# is needed.
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
)
from openai import (
    RateLimitError as _OpenAIRateLimitError,
)

OPENAI_OFFICIAL_HOST = "api.openai.com"
OPENAI_FILE_SIZE_LIMIT = 25 * 1024 * 1024  # 25 MB hard limit

# 4xx codes that classify as fatal (no point retrying).
_FATAL_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 409, 413, 415, 422})


class OpenAICompatibleProvider:
    """Transcribe via OpenAI's /audio/transcriptions API or a compatible."""

    name = "openai_compatible"
    capabilities = ProviderCapabilities(
        sends_audio_offhost=True,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
    )

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise FatalError(
                "OPENAI_API_KEY not configured. "
                "The openai_compatible transcription provider requires an "
                "API key (use a placeholder string for self-hosted backends "
                "with no auth)."
            )

        cfg = config.settings.transcription.openai_compatible
        self._base_url = cfg.base_url
        self._model = cfg.model
        self._timeout_s = cfg.timeout_s
        self._is_openai_official = OPENAI_OFFICIAL_HOST in (cfg.base_url or "")
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=cfg.base_url,
            timeout=float(cfg.timeout_s) if cfg.timeout_s else None,
        )

    async def transcribe(
        self,
        file_path: str,
        *,
        language_hint: str | None = None,
        hotwords: list[str] | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> list[TranscriptionSegment]:
        # Hotwords are ignored: OpenAI's Whisper API has no first-class
        # hotword field. Some compatibles repurpose ``prompt`` for that
        # purpose, but the behaviour is provider-specific and breaks
        # the abstraction; we surface it as ``supports_hotwords=False``.
        del progress, hotwords

        self._pre_check_size(file_path)

        try:
            with open(file_path, "rb") as audio:
                response = await self._client.audio.transcriptions.create(
                    file=audio,
                    model=self._model,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                    language=language_hint or None,
                )
        except _OpenAIRateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status is None:
                # Some SDK versions stash it on response only.
                status = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(status, int) and status in _FATAL_STATUS_CODES:
                raise FatalError(f"HTTP {status}: {exc}") from exc
            if isinstance(status, int) and 500 <= status < 600:
                raise TransientError(f"HTTP {status}: {exc}") from exc
            raise FatalError(str(exc)) from exc
        except (APITimeoutError, APIConnectionError, InternalServerError) as exc:
            raise TransientError(str(exc)) from exc

        return _parse_response(response)

    def _pre_check_size(self, file_path: str) -> None:
        """Reject >25 MB files when the target is api.openai.com.

        The official OpenAI endpoint hard-fails with HTTP 413 on
        anything over 25 MB; the failure is permanent and there is no
        Range upload, so retrying compounds JobRecord noise. Catching
        it here means the user gets an actionable error message
        suggesting Deepgram / ElevenLabs / Groq alternatives instead of
        a cryptic 413 from the SDK.
        """
        if not self._is_openai_official:
            return
        try:
            size = os.path.getsize(file_path)
        except OSError as exc:
            raise FatalError(f"Cannot stat audio file {file_path}: {exc}") from exc
        if size > OPENAI_FILE_SIZE_LIMIT:
            raise FatalError(
                f"OpenAI Whisper API has a 25MB file size limit "
                f"(got {size} bytes). Use Deepgram or ElevenLabs Scribe "
                "for long-form audio, or self-host whisper via Groq/"
                "Fireworks (no 25MB cap)."
            )


def _parse_response(response: object) -> list[TranscriptionSegment]:
    """Convert an OpenAI verbose_json response into TranscriptionSegments.

    The SDK returns a Pydantic model with ``segments`` (each with
    ``words``) and a top-level ``words`` list. We prefer per-segment
    words to keep the segment / word grouping faithful — when a
    backend returns segments with empty ``words`` everywhere we fail
    loud rather than silently emit chunks with no seek data.
    """
    segments_raw = getattr(response, "segments", None) or []
    language = getattr(response, "language", "") or ""

    # Empty audio / silence: legitimate "succeeded with zero segments"
    # case. The indexer marks the file ``whisper_indexed=True`` without
    # writing any chunks.
    if not segments_raw:
        return []

    if not any(getattr(seg, "words", None) for seg in segments_raw):
        raise FatalError(
            "Provider returned no word timestamps. "
            "Verify timestamp_granularities=['word'] is supported by "
            "the configured base_url."
        )

    segments: list[TranscriptionSegment] = []
    for seg in segments_raw:
        raw_words = getattr(seg, "words", None) or []
        words = [
            WordToken(
                text=str(getattr(w, "word", "")),
                start=float(getattr(w, "start", 0.0)),
                end=float(getattr(w, "end", 0.0)),
                speaker_id=None,
            )
            for w in raw_words
        ]
        segments.append(
            TranscriptionSegment(
                text=str(getattr(seg, "text", "")),
                start=float(getattr(seg, "start", 0.0)),
                end=float(getattr(seg, "end", 0.0)),
                language=str(getattr(seg, "language", language) or language),
                words=words,
            )
        )
    return segments
