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
* OpenAI's ``timestamp_granularities=["word", "segment"]`` is honoured by the
  official endpoint and most well-maintained compatibles, but at least
  one Groq variant has been observed returning empty ``words`` arrays.
  We treat that as :class:`FatalError` so the indexer surfaces the
  problem in ``JobRecord`` instead of silently producing search results
  with no word-level seek data.
* HTTP failure classification mirrors the LLM client's mapping (5xx /
  network / timeout = transient, 429 = rate limit, the rest = fatal).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import app.config as config

logger = logging.getLogger(__name__)
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
        self._api_key = api_key
        self._base_url = cfg.base_url
        self._model = cfg.model
        self._timeout_s = cfg.timeout_s
        # Strict host check via urlparse (was substring match — see hako
        # ``tJV51mfYZWLqMBIHm9Qvi``). ``api.openai.com.attacker.com``
        # used to slip past as "official" and trigger the 25 MB
        # pre-check on a non-OpenAI endpoint.
        from urllib.parse import urlparse

        parsed = urlparse(cfg.base_url or "")
        host = (parsed.hostname or "").lower()
        self._is_openai_official = host == OPENAI_OFFICIAL_HOST
        # We previously stored an ``AsyncOpenAI`` here, but
        # ``get_provider()`` returns a fresh provider per request and
        # the SDK never got ``aclose()``-ed. Build a short-lived client
        # inside ``transcribe()`` via ``async with`` so sockets are
        # released between jobs (hako ``W0F1YQspXF-lVYgaDb6V1``).
        # ``_client`` remains a test injection point: when set, the
        # tests pre-build a mock and the production lifecycle is
        # bypassed.
        self._client: AsyncOpenAI | None = None

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

        try:
            with open(file_path, "rb") as audio:
                # TOCTOU-safe size check: stat the open fd, not the
                # path, so a file replaced/grown between stat and open
                # cannot bypass the 25 MB OpenAI cap. Phase 1 impact is
                # low (API would 413), but the spec contract for
                # actionable error message would otherwise break in a
                # race.
                if self._is_openai_official:
                    self._pre_check_size_fd(audio.fileno(), file_path)
                # Send as (synthetic_filename, fileobj). We use a
                # generated ASCII filename of the form ``audio.<ext>``
                # rather than the user's original basename for two
                # reasons:
                #   1. Without an explicit tuple the SDK derives the
                #      filename from fileobj.name = the full absolute
                #      path, which OpenAI fails to parse and rejects
                #      as "Invalid file format".
                #   2. Original basenames may contain non-ASCII
                #      characters (Japanese, hash, spaces, etc.) that
                #      httpx multipart encoding mangles or that OpenAI
                #      silently misroutes (returning an empty
                #      transcription with 0 segments instead of an
                #      error).
                # The extension is what OpenAI uses for format
                # detection, so we preserve it. NOTE: extension and
                # actual container must still match (e.g., M4A audio
                # in a .mp4 wrapper is rejected — see docs/PROVIDERS.md).
                ext = os.path.splitext(file_path)[1].lower() or ".bin"
                synthetic_name = f"audio{ext}"
                file_arg = (synthetic_name, audio)
                if self._client is not None:
                    # Test path: a pre-built mock client is installed.
                    response = await self._client.audio.transcriptions.create(
                        file=file_arg,
                        model=self._model,
                        response_format="verbose_json",
                        timestamp_granularities=["word", "segment"],
                        language=language_hint or None,
                    )
                else:
                    async with AsyncOpenAI(
                        api_key=self._api_key,
                        base_url=self._base_url,
                        timeout=float(self._timeout_s) if self._timeout_s else None,
                    ) as client:
                        response = await client.audio.transcriptions.create(
                            file=file_arg,
                            model=self._model,
                            response_format="verbose_json",
                            timestamp_granularities=["word", "segment"],
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

    def _pre_check_size_fd(self, fd: int, file_path: str) -> None:
        """Reject >25 MB files when the target is api.openai.com.

        The official OpenAI endpoint hard-fails with HTTP 413 on
        anything over 25 MB; the failure is permanent and there is no
        Range upload, so retrying compounds JobRecord noise. Catching
        it here means the user gets an actionable error message
        suggesting Deepgram / ElevenLabs / Groq alternatives instead of
        a cryptic 413 from the SDK.

        Stats the open file descriptor (``os.fstat``) rather than the
        path so a TOCTOU race (file replaced/grown between stat and
        SDK read) cannot bypass the cap. Same fd ⇒ same content.
        """
        try:
            size = os.fstat(fd).st_size
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

    With ``timestamp_granularities=["word", "segment"]`` the SDK
    returns segments (text + start/end) at one level and words at
    another. Real-world response shapes vary by backend:

    * **OpenAI official**: ``response.segments[*].text|start|end``
      with empty per-segment ``words``, and a top-level ``response.words``
      array of ``{word, start, end}``.
    * **Some self-hosted backends** (Groq / Fireworks / vLLM whisper):
      put words inside each ``segment.words`` instead.

    We accept both shapes: prefer per-segment words when present,
    otherwise distribute the top-level word list to segments by
    timestamp window.
    """
    segments_raw = getattr(response, "segments", None) or []
    top_words_raw = getattr(response, "words", None) or []
    language = getattr(response, "language", "") or ""

    # Empty audio / silence: legitimate "succeeded with zero segments"
    # case. The indexer marks the file ``whisper_indexed=True`` without
    # writing any chunks.
    if not segments_raw:
        text = getattr(response, "text", "") or ""
        duration = getattr(response, "duration", 0.0) or 0.0
        logger.info(
            "OpenAI returned 0 segments. language=%r duration=%s text_len=%d "
            "text_preview=%r",
            language, duration, len(text), text[:200],
        )
        return []

    has_per_segment_words = any(
        getattr(seg, "words", None) for seg in segments_raw
    )
    if not has_per_segment_words and not top_words_raw:
        raise FatalError(
            "Provider returned no word timestamps. "
            "Verify timestamp_granularities=['word', 'segment'] is supported by "
            "the configured base_url."
        )

    # Pre-build top-level WordToken list (used when segments lack words)
    top_words: list[WordToken] = [
        WordToken(
            text=str(getattr(w, "word", "")),
            start=float(getattr(w, "start", 0.0)),
            end=float(getattr(w, "end", 0.0)),
            speaker_id=None,
        )
        for w in top_words_raw
    ]

    segments: list[TranscriptionSegment] = []
    for seg in segments_raw:
        seg_start = float(getattr(seg, "start", 0.0))
        seg_end = float(getattr(seg, "end", 0.0))
        raw_words = getattr(seg, "words", None) or []
        if raw_words:
            words = [
                WordToken(
                    text=str(getattr(w, "word", "")),
                    start=float(getattr(w, "start", 0.0)),
                    end=float(getattr(w, "end", 0.0)),
                    speaker_id=None,
                )
                for w in raw_words
            ]
        else:
            # Distribute top-level words by timestamp window. A word is
            # included in this segment if its start falls within the
            # segment's [start, end) window (inclusive at both ends for
            # the final segment to avoid losing trailing words).
            words = [
                w for w in top_words
                if seg_start <= w.start <= seg_end
            ]
        segments.append(
            TranscriptionSegment(
                text=str(getattr(seg, "text", "")),
                start=seg_start,
                end=seg_end,
                language=str(getattr(seg, "language", language) or language),
                words=words,
            )
        )
    return segments
