"""On-host Japanese transcription via sherpa-onnx (``haas_local`` provider).

Runs the quantized ONNX conversion of Haas-ja-v1.0 on the CPU. The model
is Japanese only: other languages produce meaningless text without an
error.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import re
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

import app.config as config
from app.workers.transcription.base import (
    ProviderCapabilities,
    TranscriptionSegment,
    WordToken,
)
from app.workers.transcription.errors import FatalError, TransientError

logger = logging.getLogger(__name__)

MODEL_REPO = "oboroge0/Haas-ja-v1.0-onnx-quantized"
# Pinned: the conversion reaches sherpa-onnx's Parakeet TDT path through
# its metadata, so a moving ref could change what loads.
MODEL_COMMIT = "7f5ab0a4b1bf39db78afd6691eeecbc4f9d42e39"
MODEL_FILES = (
    "encoder.quantized.onnx",
    "encoder.data",
    "decoder.onnx",
    "joiner.onnx",
    "tokens.txt",
    "LICENSE",
    "NOTICE",
)

_SAMPLE_RATE = 16000
_LANGUAGE = "ja"
_BYTE_TOKEN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")

# The recognizer is process-wide and not thread-safe. ``_inflight`` is only
# touched on the event loop; the lock guards fetch, load, release and decode.
_lock = threading.Lock()
_recognizer: object | None = None
_last_used: float = 0.0
_inflight = 0


def _fetch_model_dir() -> Path:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:
        raise FatalError(f"huggingface_hub is not available: {exc}") from exc

    cache_dir = str(config.settings.model_cache_dir)

    def complete(path: Path) -> bool:
        return all((path / name).exists() for name in MODEL_FILES)

    def download(*, local_files_only: bool) -> Path:
        return Path(
            snapshot_download(
                MODEL_REPO,
                revision=MODEL_COMMIT,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=list(MODEL_FILES),
            )
        )

    try:
        cached = download(local_files_only=True)
        if complete(cached):
            return cached
    except Exception:
        pass

    logger.info("Fetching %s@%s", MODEL_REPO, MODEL_COMMIT)
    try:
        fetched = download(local_files_only=False)
    except LocalEntryNotFoundError as exc:
        raise TransientError(
            f"Could not fetch {MODEL_REPO}: {exc}"
        ) from exc
    except (
        RepositoryNotFoundError,
        RevisionNotFoundError,
        EntryNotFoundError,
    ) as exc:
        raise FatalError(
            f"{MODEL_REPO}@{MODEL_COMMIT} is not available on Hugging Face: "
            f"{exc}"
        ) from exc
    except Exception as exc:
        raise TransientError(f"Could not fetch {MODEL_REPO}: {exc}") from exc
    if not complete(fetched):
        raise TransientError(
            f"Fetched snapshot of {MODEL_REPO} is incomplete in {fetched}"
        )
    return fetched


def _load_recognizer() -> object:
    """Build the recognizer. The caller holds ``_lock``."""
    global _recognizer, _last_used

    try:
        import sherpa_onnx
    except ImportError as exc:
        raise FatalError(f"sherpa_onnx is not available: {exc}") from exc

    model_dir = _fetch_model_dir()
    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(model_dir / "encoder.quantized.onnx"),
            decoder=str(model_dir / "decoder.onnx"),
            joiner=str(model_dir / "joiner.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=config.settings.transcription.haas_local.num_threads,
            model_type="nemo_transducer",
            feature_dim=128,
        )
    except Exception as exc:
        raise FatalError(
            f"Could not load {MODEL_REPO} from {model_dir}; delete that "
            f"directory to fetch it again: {exc}"
        ) from exc

    _recognizer = recognizer
    _last_used = time.monotonic()
    logger.info("Haas recognizer loaded from %s", model_dir)
    return recognizer


def _release_recognizer() -> None:
    """Drop the recognizer. The caller holds ``_lock``."""
    global _recognizer

    if _recognizer is None:
        return
    _recognizer = None
    logger.info("Haas recognizer unloaded")

    from app.utils.memory import malloc_trim

    gc.collect()
    malloc_trim()


def check_idle_unload() -> None:
    """Release the recognizer once it has been idle past the configured time.

    Runs on the event loop, so it never waits: a held lock or a call in
    progress means nothing is released this round.
    """
    timeout = config.settings.memory.whisper_idle_unload
    if timeout <= 0 or _recognizer is None or _inflight:
        return
    if time.monotonic() - _last_used <= timeout:
        return
    if not _lock.acquire(blocking=False):
        return
    try:
        if (
            _recognizer is not None
            and not _inflight
            and time.monotonic() - _last_used > timeout
        ):
            _release_recognizer()
    finally:
        _lock.release()


def _decode_audio(file_path: str) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", file_path,
        "-vn", "-ac", "1", "-ar", str(_SAMPLE_RATE), "-f", "f32le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise FatalError("ffmpeg is not installed") from exc
    if proc.returncode < 0:
        raise TransientError(
            f"ffmpeg was killed by signal {-proc.returncode} on {file_path}"
        )
    if proc.returncode > 0:
        logger.warning(
            "ffmpeg exited with status %d on %s: %s",
            proc.returncode,
            file_path,
            proc.stderr.decode("utf-8", "replace").strip()[-300:],
        )
        return np.zeros(0, dtype=np.float32)
    usable = len(proc.stdout) - len(proc.stdout) % 4
    return np.frombuffer(proc.stdout[:usable], dtype=np.float32)


def _speech_spans(samples: np.ndarray) -> list[tuple[int, int]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        max_speech_duration_s=(
            config.settings.transcription.haas_local.max_segment_seconds
        ),
        min_silence_duration_ms=300,
        speech_pad_ms=200,
    )
    try:
        raw = get_speech_timestamps(samples, options)
    except IndexError:
        return []
    except Exception as exc:
        raise FatalError(f"Voice activity detection failed: {exc}") from exc
    spans = []
    for item in raw:
        start = max(0, int(item["start"]))
        end = min(len(samples), int(item["end"]))
        if end > start:
            spans.append((start, end))
    return spans


def _words(
    tokens: list[str],
    timestamps: list[float],
    durations: list[float],
    span_start: float,
    span_end: float,
) -> list[WordToken]:
    """Turn one recognizer result into words with file-offset times."""
    has_durations = len(durations) == len(tokens)

    # (text, start offset into the span, duration or None)
    items: list[tuple[str, float, float | None]] = []
    i = 0
    while i < len(tokens):
        if _BYTE_TOKEN.match(tokens[i]):
            j = i
            while j < len(tokens) and _BYTE_TOKEN.match(tokens[j]):
                j += 1
            raw = bytes(
                int(_BYTE_TOKEN.match(t).group(1), 16) for t in tokens[i:j]
            )
            text = (
                raw.decode("utf-8", errors="ignore")
                .replace("�", "")
                .strip()
            )
            duration = (
                timestamps[j - 1] + durations[j - 1] - timestamps[i]
                if has_durations
                else None
            )
            start = timestamps[i]
            i = j
        else:
            text = tokens[i].strip()
            duration = durations[i] if has_durations else None
            start = timestamps[i]
            i += 1
        if text:
            items.append((text, start, duration))

    words: list[WordToken] = []
    previous_end = span_start
    for n, (text, offset, duration) in enumerate(items):
        raw_start = span_start + offset
        following = (
            span_start + items[n + 1][1] if n + 1 < len(items) else None
        )
        start = min(max(raw_start, previous_end, span_start), span_end)
        end = (
            start + duration
            if duration is not None
            else (following if following is not None else span_end)
        )
        end = min(end, span_end)
        if following is not None:
            end = min(end, following)
        end = max(end, start)
        words.append(WordToken(text=text, start=start, end=end, speaker_id=None))
        previous_end = end
    return words


def _transcribe_blocking(file_path: str) -> list[TranscriptionSegment]:
    global _last_used

    samples = _decode_audio(file_path)
    if len(samples) == 0:
        return []
    spans = _speech_spans(samples)
    if not spans:
        return []

    segments: list[TranscriptionSegment] = []
    for start_i, end_i in spans:
        with _lock:
            recognizer = _recognizer or _load_recognizer()
            try:
                stream = recognizer.create_stream()
                stream.accept_waveform(_SAMPLE_RATE, samples[start_i:end_i])
                recognizer.decode_stream(stream)
                result = stream.result
                tokens = list(result.tokens)
                timestamps = list(result.timestamps)
                durations = list(getattr(result, "durations", None) or [])
            except Exception as exc:
                _release_recognizer()
                raise FatalError(
                    f"Haas recognizer failed on {file_path}: {exc}"
                ) from exc
            _last_used = time.monotonic()

        words = _words(
            tokens,
            timestamps,
            durations,
            start_i / _SAMPLE_RATE,
            end_i / _SAMPLE_RATE,
        )
        if words:
            segments.append(
                TranscriptionSegment(
                    text="".join(w.text for w in words),
                    start=words[0].start,
                    end=words[-1].end,
                    language=_LANGUAGE,
                    words=words,
                )
            )
    return segments


class HaasLocalProvider:
    """sherpa-onnx backed Provider for Japanese (on-host)."""

    name = "haas_local"
    capabilities = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
        # Same cap as whisper_local: the decoded samples of a chunk this
        # size bound the per-call memory.
        max_input_bytes=50 * 1024 * 1024,
        accepts_initial_prompt=False,
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
        """Transcribe on a worker thread; the hints and prompt are ignored."""
        global _inflight

        _inflight += 1
        try:
            return await asyncio.to_thread(_transcribe_blocking, file_path)
        finally:
            _inflight -= 1
