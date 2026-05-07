"""AssemblyAI transcription provider.

Calls AssemblyAI's v2 REST API directly via httpx (no SDK — same
reasoning as deepgram.py: httpx is already a transitive dependency
and the SDK adds a non-trivial import surface for very little).

Wire flow per AssemblyAI v2 docs:
1. ``POST /v2/upload`` with ``content-type: application/octet-stream``
   uploads the raw audio bytes; response carries an ``upload_url``.
2. ``POST /v2/transcript`` with ``content-type: application/json`` and
   the ``upload_url`` queues the transcription; response carries
   ``id``.
3. ``GET /v2/transcript/{id}`` is polled until ``status`` is
   ``completed`` or ``error``.
4. The ``words`` list returned alongside the transcript is converted
   to :class:`WordToken` plus a single :class:`TranscriptionSegment`,
   matching the Phase 1 deepgram / elevenlabs contract (the chunker is
   responsible for re-segmenting on speaker change / silence).

Spec: ``2026-05-08-transcription-providers-phase-2a.md``.
"""

from __future__ import annotations

import asyncio
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

ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2"
ASSEMBLYAI_FILE_SIZE_LIMIT = 5 * 1024 * 1024 * 1024  # 5 GB hard cap

_FATAL_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 409, 413, 415, 422})


class AssemblyAIProvider:
    """Cloud transcription via AssemblyAI v2 REST API."""

    name = "assemblyai"
    capabilities = ProviderCapabilities(
        sends_audio_offhost=True,
        supports_diarization=True,
        supports_hotwords=True,
        supports_word_timestamps=True,
        max_input_bytes=ASSEMBLYAI_FILE_SIZE_LIMIT,
        accepts_initial_prompt=False,   # AssemblyAI has no prior-text channel
        handles_own_retry=False,
    )

    def __init__(self) -> None:
        api_key = os.getenv("ASSEMBLYAI_API_KEY", "")
        if not api_key:
            raise FatalError(
                "ASSEMBLYAI_API_KEY not configured. "
                "Set the env var to enable the assemblyai transcription provider."
            )
        cfg = config.settings.transcription.assemblyai
        self._api_key = api_key
        self._model = cfg.model  # "best" or "nano"
        self._language_detection = cfg.language_detection
        self._speaker_labels = cfg.speaker_labels
        self._timeout_s = cfg.timeout_s
        self._poll_interval_s = cfg.poll_interval_s
        # Test injection point (mirrors deepgram.py / elevenlabs_scribe.py).
        # Production lifecycle builds short-lived clients per request so
        # AssemblyAI sockets / fds are released between jobs (hako
        # ``W0F1YQspXF-lVYgaDb6V1``).
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
        del progress, initial_prompt  # See ProviderCapabilities

        try:
            with open(file_path, "rb") as audio:
                # TOCTOU-safe size check: stat the open fd, not the
                # path, so a file replaced/grown between stat and read
                # cannot bypass the 5GB cap (mirrors openai_compatible).
                self._pre_check_size_fd(audio.fileno(), file_path)
                body = audio.read()
        except OSError as exc:
            raise FatalError(
                f"Cannot read audio file {file_path}: {exc}"
            ) from exc

        client_kwargs: dict = {"timeout": float(self._timeout_s)}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        # AssemblyAI's auth header is plain ``authorization: <key>``
        # without the ``Bearer`` prefix.
        common_headers = {"authorization": self._api_key}

        async with httpx.AsyncClient(**client_kwargs) as client:
            upload_url = await self._upload(client, common_headers, body)
            transcript_id = await self._submit(
                client, common_headers, upload_url, language_hint, hotwords
            )
            transcript = await self._poll(
                client, common_headers, transcript_id
            )

        return _parse_assemblyai_response(transcript)

    def _pre_check_size_fd(self, fd: int, file_path: str) -> None:
        """Reject >5 GB files — AssemblyAI's hard upload cap.

        Same TOCTOU rationale as openai_compatible's 25 MB pre-check:
        stat the open fd so a swap between stat and read cannot
        smuggle a larger body past the gate. Phase 2B will add an
        ffmpeg-based splitter so long-form audio survives without
        per-provider caps; until then we fail loud with an actionable
        message.
        """
        try:
            size = os.fstat(fd).st_size
        except OSError as exc:
            raise FatalError(
                f"Cannot stat audio file {file_path}: {exc}"
            ) from exc
        if size > ASSEMBLYAI_FILE_SIZE_LIMIT:
            raise FatalError(
                f"AssemblyAI has a 5GB file size limit "
                f"(got {size} bytes). Phase 2B will add ffmpeg-based "
                "splitting for long-form audio."
            )

    async def _upload(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        body: bytes,
    ) -> str:
        upload_headers = {
            **headers,
            "content-type": "application/octet-stream",
        }
        try:
            response = await client.post(
                f"{ASSEMBLYAI_BASE_URL}/upload",
                headers=upload_headers,
                content=body,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise TransientError(
                f"AssemblyAI upload network error: {exc}"
            ) from exc
        _classify_status(response, "upload")
        try:
            payload = response.json()
        except ValueError as exc:
            raise FatalError(
                f"AssemblyAI upload returned non-JSON: {response.text[:200]}"
            ) from exc
        upload_url = payload.get("upload_url")
        if not isinstance(upload_url, str) or not upload_url:
            raise FatalError(
                f"AssemblyAI upload response missing upload_url: "
                f"{response.text[:200]}"
            )
        return upload_url

    async def _submit(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        upload_url: str,
        language_hint: str | None,
        hotwords: list[str] | None,
    ) -> str:
        payload: dict = {
            "audio_url": upload_url,
            "speech_model": self._model,
            "speaker_labels": self._speaker_labels,
            "punctuate": True,
            "format_text": True,
        }
        # Mirror Deepgram's resolution rule (deepgram.py:104-110): an
        # explicit ``language_hint`` always wins over auto-detection
        # because the caller is more informed than the model.
        if language_hint:
            payload["language_code"] = language_hint
        elif self._language_detection:
            payload["language_detection"] = True
        if hotwords:
            payload["word_boost"] = list(hotwords)

        submit_headers = {**headers, "content-type": "application/json"}
        try:
            response = await client.post(
                f"{ASSEMBLYAI_BASE_URL}/transcript",
                headers=submit_headers,
                json=payload,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise TransientError(
                f"AssemblyAI submit network error: {exc}"
            ) from exc
        _classify_status(response, "submit")
        try:
            data = response.json()
        except ValueError as exc:
            raise FatalError(
                f"AssemblyAI submit returned non-JSON: "
                f"{response.text[:200]}"
            ) from exc
        transcript_id = data.get("id")
        if not isinstance(transcript_id, str) or not transcript_id:
            raise FatalError(
                f"AssemblyAI submit response missing id: "
                f"{response.text[:200]}"
            )
        return transcript_id

    async def _poll(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        transcript_id: str,
    ) -> dict:
        deadline = asyncio.get_event_loop().time() + float(self._timeout_s)
        url = f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}"
        while True:
            try:
                response = await client.get(url, headers=headers)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                raise TransientError(
                    f"AssemblyAI poll network error: {exc}"
                ) from exc
            _classify_status(response, "poll")
            try:
                data = response.json()
            except ValueError as exc:
                raise FatalError(
                    f"AssemblyAI poll returned non-JSON: "
                    f"{response.text[:200]}"
                ) from exc
            status = data.get("status")
            if status == "completed":
                return data
            if status == "error":
                raise FatalError(
                    f"AssemblyAI transcription error: {data.get('error')}"
                )
            if status not in ("queued", "processing"):
                raise FatalError(
                    f"AssemblyAI unexpected status: {status!r}"
                )
            if asyncio.get_event_loop().time() > deadline:
                raise TransientError(
                    f"AssemblyAI polling timeout after {self._timeout_s}s "
                    f"(transcript {transcript_id} still {status})"
                )
            await asyncio.sleep(float(self._poll_interval_s))


def _classify_status(response: httpx.Response, phase: str) -> None:
    """Map an HTTP status to TranscriptionError subclasses.

    ``phase`` is the wire stage label ("upload" / "submit" / "poll"),
    surfaced in the error message so JobRecord rows pinpoint where a
    cloud round-trip failed without needing to scrape logs.
    """
    if response.status_code == 200:
        return
    if response.status_code == 429:
        raise RateLimitError(
            f"AssemblyAI {phase} rate limit (429): {response.text[:200]}"
        )
    if response.status_code in _FATAL_STATUS_CODES:
        raise FatalError(
            f"AssemblyAI {phase} HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )
    if 500 <= response.status_code < 600:
        raise TransientError(
            f"AssemblyAI {phase} HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )
    raise FatalError(
        f"AssemblyAI {phase} unexpected HTTP {response.status_code}: "
        f"{response.text[:200]}"
    )


def _parse_assemblyai_response(payload: dict) -> list[TranscriptionSegment]:
    """Convert an AssemblyAI completed transcript into TranscriptionSegments.

    Same one-segment-per-call contract as deepgram (the chunker handles
    re-segmentation). ``words`` are mandatory; an empty list is treated
    as silence and the indexer marks the file ``whisper_indexed=True``
    without writing chunks (parity with deepgram.py).
    """
    raw_words = payload.get("words") or []
    if not raw_words:
        return []

    language = (
        payload.get("language_code")
        or payload.get("language_detected")
        or ""
    )

    words: list[WordToken] = []
    for w in raw_words:
        text = str(w.get("text") or "").strip()
        if not text:
            continue
        speaker = w.get("speaker")
        # AssemblyAI returns timestamps in milliseconds; convert to seconds
        # to match the WordToken contract (mirrors elevenlabs_scribe).
        start_ms = float(w.get("start", 0.0))
        end_ms = float(w.get("end", 0.0))
        words.append(
            WordToken(
                text=text,
                start=start_ms / 1000.0,
                end=end_ms / 1000.0,
                speaker_id=str(speaker) if speaker is not None else None,
            )
        )
    if not words:
        return []

    transcript_text = (
        payload.get("text")
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
