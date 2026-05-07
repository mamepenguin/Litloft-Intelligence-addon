"""Gemini transcription provider.

Uses Google's ``google-genai`` SDK. Audio is uploaded to Gemini's File
API, then ``generate_content`` is called with the file handle plus a
structured-output prompt asking for SRT-like segments. The response is
parsed into segments + synthetic word timestamps (Gemini does not
return native word-level timing).

Phase 2A contract evolution: this is the first provider with
``supports_word_timestamps=False``. The chunker is unchanged — we
synthesise non-empty ``words`` lists per segment so downstream code
keeps working; the dispatch layer logs a one-shot WARN to surface the
precision drop.

Lifecycle: ``__init__`` only validates env + lazy-imports the SDK so
a missing dependency surfaces as ``FatalError`` (not bare
``ImportError``). The ``genai.Client`` is built per-call so HTTP
sockets and File API handles do not leak between jobs (hako
``W0F1YQspXF-lVYgaDb6V1``).

Spec: ``2026-05-08-transcription-providers-phase-2a.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json as _json
import logging
import os
from collections.abc import Callable

import regex as re_grapheme  # NOTE: third-party regex package, not stdlib re

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

logger = logging.getLogger(__name__)

GEMINI_FILE_SIZE_LIMIT = 2 * 1024 * 1024 * 1024  # 2 GB Gemini File API cap
GEMINI_FILE_ACTIVE_POLL_INTERVAL_S = 2.0

# Languages without whitespace word boundaries. Detection is language-
# driven (not script-driven) because Gemini's per-segment ``language``
# is more reliable than a code-point heuristic over short text.
_NO_WHITESPACE_LANGS = ("ja", "zh", "yue", "ko", "th", "lo", "km", "my")

_GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start": {"type": "NUMBER"},
                    "end": {"type": "NUMBER"},
                    "text": {"type": "STRING"},
                    "language": {"type": "STRING"},
                },
                "required": ["start", "end", "text"],
            },
        }
    },
    "required": ["segments"],
}


class GeminiProvider:
    """Cloud transcription via Google Gemini File API + generate_content."""

    name = "gemini"
    capabilities = ProviderCapabilities(
        sends_audio_offhost=True,
        supports_diarization=False,  # Gemini has no native diarization API.
        supports_hotwords=True,  # Hotwords forwarded via system prompt.
        supports_word_timestamps=False,  # Synthetic — see _synthetic_words.
    )

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise FatalError(
                "GEMINI_API_KEY not configured. "
                "Set the env var to enable the gemini transcription provider."
            )
        # Lazy SDK import: a missing dependency must surface as
        # FatalError so the dispatch layer's ``except (ValueError,
        # FatalError)`` catches it and JobRecord captures the right
        # error class. Bare ImportError would escape unwrapped.
        try:
            import google.genai  # noqa: F401
        except ImportError as exc:
            raise FatalError(
                "google-genai SDK not installed. "
                "Add 'google-genai>=1.10' to "
                "addons/intelligence/requirements.txt."
            ) from exc
        cfg = config.settings.transcription.gemini
        self._api_key = api_key
        self._model = cfg.model
        self._output_language = cfg.output_language
        self._upload_wait_sec = cfg.upload_wait_sec
        self._timeout_s = cfg.timeout_s
        # Test injection slot — production builds the client per call
        # to honour the lifecycle rule.
        self._client = None  # type: ignore[assignment]

    async def transcribe(
        self,
        file_path: str,
        *,
        language_hint: str | None = None,
        hotwords: list[str] | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> list[TranscriptionSegment]:
        del progress  # See ProviderCapabilities

        # TOCTOU-safe size check: stat the open fd, not the path
        # (mirrors openai_compatible / assemblyai).
        try:
            with open(file_path, "rb") as audio:
                self._pre_check_size_fd(audio.fileno(), file_path)
        except OSError as exc:
            raise FatalError(
                f"Cannot read audio file {file_path}: {exc}"
            ) from exc

        from google import genai
        from google.genai import errors as genai_errors
        from google.genai import types as genai_types

        client = self._client or genai.Client(api_key=self._api_key)
        loop = asyncio.get_running_loop()

        try:
            # Wrap upload with asyncio.wait_for so a stuck POST does
            # not block the worker indefinitely. _wait_for_active has
            # its own deadline; this guard only covers the upload phase.
            uploaded = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: client.files.upload(file=file_path)
                ),
                timeout=float(self._timeout_s),
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise TransientError(
                f"Gemini upload timeout after {self._timeout_s}s"
            ) from exc
        except genai_errors.ClientError as exc:
            status = _status_code(exc)
            if status == 429:
                raise RateLimitError(str(exc)) from exc
            raise FatalError(f"Gemini upload client error: {exc}") from exc
        except genai_errors.ServerError as exc:
            raise TransientError(f"Gemini upload server error: {exc}") from exc

        try:
            uploaded = await self._wait_for_active(client, uploaded.name, loop)
            prompt = self._build_prompt(language_hint, hotwords)
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=self._model,
                        contents=[uploaded, prompt],
                        config=genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=_GEMINI_SCHEMA,
                            temperature=0.0,
                        ),
                    ),
                )
            except genai_errors.ClientError as exc:
                status = _status_code(exc)
                if status == 429:
                    raise RateLimitError(str(exc)) from exc
                raise FatalError(
                    f"Gemini generate_content client error: {exc}"
                ) from exc
            except genai_errors.ServerError as exc:
                raise TransientError(
                    f"Gemini generate_content server error: {exc}"
                ) from exc
        finally:
            # Cleanup is best-effort — failure here is logged but not
            # surfaced because the user already paid for the inference.
            with contextlib.suppress(Exception):
                await loop.run_in_executor(
                    None, lambda: client.files.delete(name=uploaded.name)
                )

        return _convert_gemini_response(
            response, language_hint or self._output_language
        )

    async def _wait_for_active(self, client, file_name: str, loop):
        """Poll ``client.files.get`` until ``state.name == "ACTIVE"``.

        The Gemini File API processes uploads asynchronously. Until the
        state becomes ``ACTIVE`` the file cannot be referenced in
        ``generate_content``. ``FAILED`` is unrecoverable. Polling
        runs in the executor because the SDK is synchronous.
        """
        from google.genai import errors as genai_errors

        deadline = loop.time() + float(self._upload_wait_sec)
        while True:
            try:
                f = await loop.run_in_executor(
                    None, lambda: client.files.get(name=file_name)
                )
            except genai_errors.ServerError as exc:
                if loop.time() > deadline:
                    raise TransientError(
                        f"Gemini file polling failed: {exc}"
                    ) from exc
                await asyncio.sleep(GEMINI_FILE_ACTIVE_POLL_INTERVAL_S)
                continue
            state_obj = getattr(f, "state", None)
            state = (
                getattr(state_obj, "name", None)
                or str(state_obj or "")
            ).upper()
            if state == "ACTIVE":
                return f
            if state == "FAILED":
                raise FatalError(
                    f"Gemini File API processing failed for {file_name}"
                )
            if loop.time() > deadline:
                raise TransientError(
                    f"Gemini file did not become ACTIVE within "
                    f"{self._upload_wait_sec}s (state={state})"
                )
            await asyncio.sleep(GEMINI_FILE_ACTIVE_POLL_INTERVAL_S)

    def _build_prompt(
        self,
        language_hint: str | None,
        hotwords: list[str] | None,
    ) -> str:
        target_language = language_hint or self._output_language
        hotword_line = (
            f"固有名詞のヒント: {', '.join(hotwords)}\n" if hotwords else ""
        )
        return (
            "あなたは音声書き起こしの構造化出力ツールです。"
            "提供された音声ファイル全体を、自然な句読点を入れて"
            "書き起こしてください。出力は JSON で、各 segment ごとに "
            "start (秒、小数1位)、end (秒、小数1位)、text (書き起こし文字列)、"
            "language (BCP-47 コード) を返してください。"
            "1 segment は概ね 1 文または 1 発話単位にしてください。\n"
            f"出力言語: {target_language}\n"
            f"{hotword_line}"
        )

    def _pre_check_size_fd(self, fd: int, file_path: str) -> None:
        """Reject >2 GB files — Gemini File API hard cap."""
        try:
            size = os.fstat(fd).st_size
        except OSError as exc:
            raise FatalError(
                f"Cannot stat audio file {file_path}: {exc}"
            ) from exc
        if size > GEMINI_FILE_SIZE_LIMIT:
            raise FatalError(
                f"Gemini File API has a 2GB file size limit "
                f"(got {size} bytes). Phase 2B will add ffmpeg-based "
                "splitting for long-form audio."
            )


def _status_code(exc: Exception) -> int | None:
    """Best-effort extraction of the HTTP status from a SDK exception.

    google-genai exposes the status as either ``code`` (newer
    releases) or ``status_code`` (older). Both are tried; the first
    integer wins.
    """
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _is_no_whitespace_language(language: str | None) -> bool:
    if not language:
        return False
    code = language.lower().split("-")[0]
    return code in _NO_WHITESPACE_LANGS


def _split_to_tokens(text: str, language: str | None) -> list[str]:
    """Split segment text into pseudo-words for synthetic timing.

    For CJK + Thai/Lao/Khmer/Burmese we use grapheme clusters via the
    third-party ``regex`` package (``\\X``), which handles combining
    marks and surrogate pairs correctly. For everything else we
    whitespace-split after stripping. Empty text returns an empty list.
    """
    text = (text or "").strip()
    if not text:
        return []
    if _is_no_whitespace_language(language):
        return [g for g in re_grapheme.findall(r"\X", text) if g.strip()]
    return text.split()


def _synthetic_words(
    segment_text: str,
    start: float,
    end: float,
    language: str | None,
) -> list[WordToken]:
    """Synthesise word timestamps by splitting segment duration evenly.

    Used when the provider only returns segment-level timing
    (Gemini). The result is intentionally coarse — chunker / search
    work, but per-word seek alignment is approximate. WhisperX-style
    forced realignment would restore native precision; that lives in
    the existing ``transcript_refine`` pass.
    """
    tokens = _split_to_tokens(segment_text, language)
    if not tokens:
        return []
    duration = max(0.0, end - start)
    per_token = duration / len(tokens) if tokens else 0.0
    return [
        WordToken(
            text=tok,
            start=start + i * per_token,
            end=start + (i + 1) * per_token,
            speaker_id=None,  # No diarization.
        )
        for i, tok in enumerate(tokens)
    ]


def _convert_gemini_response(
    response: object,
    fallback_language: str,
) -> list[TranscriptionSegment]:
    """Parse Gemini's structured-output JSON into TranscriptionSegments.

    The SDK's ``response.text`` contains the JSON string; we parse,
    validate the structure, and synthesise word timestamps per
    segment. Parsing failures are FatalError — empty / malformed
    output should fail loud rather than silently produce zero results.
    """
    raw_text = getattr(response, "text", None)
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise FatalError("Gemini response has no JSON text")
    try:
        payload = _json.loads(raw_text)
    except (ValueError, _json.JSONDecodeError) as exc:
        raise FatalError(
            f"Gemini response is not valid JSON: {raw_text[:200]}"
        ) from exc
    if not isinstance(payload, dict):
        raise FatalError(
            f"Gemini response is not a JSON object: {type(payload).__name__}"
        )
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise FatalError(
            "Gemini response missing 'segments' array"
        )
    if not raw_segments:
        return []

    segments: list[TranscriptionSegment] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(raw.get("start", 0.0))
            end = float(raw.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        language = str(raw.get("language") or fallback_language)
        words = _synthetic_words(text, start, end, language)
        if not words:
            continue
        segments.append(
            TranscriptionSegment(
                text=text,
                start=start,
                end=end,
                language=language,
                words=words,
            )
        )
    return segments
