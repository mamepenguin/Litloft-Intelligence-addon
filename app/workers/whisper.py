"""Whisper transcription worker.

Transcribes audio from video/audio files using a configurable
:class:`~app.workers.transcription.base.TranscriptionProvider`. The
local faster-whisper backend stays the default; cloud providers
(Deepgram, ElevenLabs Scribe, OpenAI-compatible) are wired through
:func:`~app.workers.transcription.get_provider` and gated by
per-drive cloud policy at both enqueue and dequeue time.

Only one Whisper task runs at a time (controlled by the indexer's
semaphore).

Spec: ``2026-05-07-cloud-transcription-providers.md``.
"""

import asyncio
import logging
import os
import re
import threading
import time
import uuid
from datetime import UTC, datetime

from app.config import settings, validate_file_path
from app.database import delete_fts_transcripts, get_search_db, get_search_db_read, upsert_fts_transcripts
from app.models import (
    Embedding,
    IndexedFile,
    JobRecord,
    TranscriptChunk,
    TranscriptWord,
)
from app.workers.embedder import embed_passages
from app.workers.transcription import get_provider
from app.workers.transcription.errors import (
    FatalError,
    RateLimitError,
    TranscriptionError,
    TransientError,
)
from app.workers.transcription.retry import (
    CircuitBreakerOpen,
    transcribe_with_retry,
)
from app.workers.whisper_prompts import resolve_initial_prompt
from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)


async def _emit_ws_event(event: str, data: dict) -> None:
    """Best-effort WebSocket event emission via the host's internal API.

    Mirrors :func:`app.workers.refine._emit_ws_event` /
    :func:`app.workers.summaries._emit_ws_event`. The host forwards
    the posted JSON to its WebSocket broadcaster; delivery failures
    are swallowed so a flaky core never fails a transcription job.
    Tests monkeypatch this function.

    Drive scoping: when ``data`` carries a ``drive`` key we lift it to
    the top-level ``AddonEventRequest.drive`` so the host's
    ``ConnectionManager.broadcast`` filter can suppress delivery to
    viewers without access to the protected drive. Without this, the
    event would broadcast to every connected viewer (hako pattern
    ``HpeftQ_io8n7sJ5xxlasC``).
    """
    logger.info("transcription-event %s %s", event, data)

    base = os.environ.get(
        "HOMEVAULT_INTERNAL_API_URL", "http://backend:8000/api/internal"
    )
    url = f"{base}/addon-events"
    payload: dict = {"event": event, "data": data}
    drive = data.get("drive") if isinstance(data, dict) else None
    if drive:
        payload["drive"] = drive
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(url, json=payload)
    except Exception:
        return

# Model state (lazy-loaded, with idle unload support)
_lock = threading.Lock()
_model: object | None = None
_batched_pipeline: object | None = None
_loaded = False
_last_used: float = 0.0

# Audio/video types that can be transcribed.
#
# Phase 2F additions: ``audio/mp4`` is the IANA-registered MIME for
# AAC-in-MP4 audio (most ``.m4a`` files); ``audio/opus`` is the
# IANA-registered MIME for Opus audio. Both are required because
# Linux Docker's ``mimetypes`` DB lacks the ``.m4a`` / ``.opus``
# entries so backend classify() falls back to extension lookup and
# emits the IANA names (hako A-gF1mK3kDjRjS_dfuq1B). The de-facto
# Apple variants (``audio/m4a``, ``audio/x-m4a``) are kept for
# files registered through other paths.
TRANSCRIBABLE_TYPES = {
    "video/mp4", "video/webm", "video/quicktime", "video/x-matroska",
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg", "audio/flac",
    "audio/aac", "audio/m4a", "audio/x-m4a",
    "audio/mp4", "audio/opus",
}

# External source files (.loft) that use adjacent .vtt instead of Whisper
LOFT_MIME = "application/vnd.litloft.loft+json"
LOFT_STT_TEMP_SUFFIX = ".stt_temp.m4a"


def _loft_stt_temp_path(file_path: str) -> str | None:
    from pathlib import Path

    loft_path = Path(file_path)
    candidate = loft_path.parent / f"{loft_path.stem}{LOFT_STT_TEMP_SUFFIX}"
    if candidate.is_file():
        return str(candidate)
    return None


def _cleanup_loft_stt_temp(file_path: str | None) -> None:
    if not file_path:
        return
    from pathlib import Path

    try:
        path = Path(file_path)
        if path.is_file():
            path.unlink()
    except FileNotFoundError:
        return
    except Exception:
        logger.warning("Failed to remove loft STT temp file: %s", file_path)


def _mark_loft_stt_temp_consumed(file_id: str) -> None:
    """Close a one-shot temp-audio attempt after provider failure.

    Normal media files keep ``whisper_indexed=False`` after provider
    failures because the source file remains available for retry. A
    Media Import temp audio is deleted after the attempt by design; if we
    left the flag false, the next resume pass would enqueue the .loft
    again and the no-temp/no-VTT path would mark it indexed without the
    failed JobRecord being the terminal state.
    """
    try:
        with get_search_db() as session:
            file = session.query(IndexedFile).filter_by(file_id=file_id).first()
            if file is not None:
                file.whisper_indexed = True
    except Exception:
        logger.exception("Failed to mark loft STT temp consumed for %s", file_id)


def _ensure_loaded() -> tuple[object, object | None]:
    """Lazy-load the Whisper model on first use.

    Returns:
        Tuple of (WhisperModel, BatchedInferencePipeline or None).
    """
    global _model, _batched_pipeline, _loaded, _last_used

    if _loaded and _model is not None:
        _last_used = time.monotonic()
        return _model, _batched_pipeline

    with _lock:
        if _loaded and _model is not None:
            _last_used = time.monotonic()
            return _model, _batched_pipeline

        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel

            model_size = _resolve_model_size(settings.models.whisper)
            cache_dir = str(settings.model_cache_dir)

            logger.info("Loading Whisper model: %s (size: %s)", settings.models.whisper, model_size)

            _model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                download_root=cache_dir,
            )

            batch_size = settings.indexing.whisper.batch_size
            if batch_size > 0:
                _batched_pipeline = BatchedInferencePipeline(model=_model)
                logger.info("Batched inference enabled (batch_size=%d)", batch_size)
            else:
                _batched_pipeline = None

            _loaded = True
            _last_used = time.monotonic()
            logger.info("Whisper model loaded successfully")
            return _model, _batched_pipeline

        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)
            raise RuntimeError(f"Whisper model load failed: {e}") from e


def _resolve_model_size(config_name: str) -> str:
    """Resolve config model name to a faster-whisper model identifier.

    Known ``openai/whisper-*`` aliases are mapped to faster-whisper size
    shortcuts. Any other value is returned verbatim so users can point at
    faster-whisper size strings (e.g. ``large-v3-turbo``) or
    CT2-compatible HuggingFace repo IDs (e.g.
    ``deepdml/faster-whisper-large-v3-turbo-ct2``) without a schema
    change. Invalid names surface as a clear faster-whisper load error
    rather than silently falling back to a different model.
    """
    size_map = {
        "openai/whisper-tiny": "tiny",
        "openai/whisper-base": "base",
        "openai/whisper-small": "small",
        "openai/whisper-medium": "medium",
        "openai/whisper-large": "large-v3",
        "openai/whisper-large-v3": "large-v3",
        "openai/whisper-large-v3-turbo": "large-v3-turbo",
    }
    return size_map.get(config_name, config_name)


def unload_model() -> None:
    """Unload the Whisper model to free RAM.

    Safe to call even if the model is not loaded.
    """
    global _model, _batched_pipeline, _loaded

    with _lock:
        if _model is not None:
            logger.info("Unloading Whisper model to free RAM")
            _model = None
            _batched_pipeline = None
            _loaded = False

    import gc

    from app.utils.memory import malloc_trim

    gc.collect()
    malloc_trim()


def check_idle_unload() -> None:
    """Check if the model should be unloaded due to idle timeout.

    Called periodically by the indexer's background task.
    """
    idle_timeout = settings.memory.whisper_idle_unload
    if idle_timeout <= 0:
        return

    if not _loaded or _model is None:
        return

    elapsed = time.monotonic() - _last_used
    if elapsed > idle_timeout:
        unload_model()


# Below this language-detection probability we treat the result as
# unreliable and skip the default prompt rather than feeding Whisper a
# mismatched language hint.
_LANG_DETECT_MIN_PROB = 0.5


def _detect_language(model: object, file_path: str) -> str | None:
    """Run Whisper's lightweight language detector on a media file.

    Uses faster-whisper's ``detect_language`` which only consumes a
    short audio sample (~30 s), so the cost relative to a full
    transcription is negligible. Returns ``None`` on low confidence
    or any failure — callers must tolerate an absent language.
    """
    try:
        language, probability, _ = model.detect_language(file_path)
    except Exception as e:
        logger.warning(
            "Language detection failed for %s: %s", file_path, e
        )
        return None

    if probability < _LANG_DETECT_MIN_PROB:
        logger.info(
            "Language detection low confidence for %s: %s (%.2f) — "
            "skipping default initial_prompt",
            file_path, language, probability,
        )
        return None
    return language


def _transcribe_file(
    file_path: str,
    initial_prompt_override: str | None = None,
) -> list[dict]:
    """Transcribe a media file using faster-whisper.

    Uses BatchedInferencePipeline when batch_size > 0 for faster throughput.
    Falls back to sequential transcription otherwise.

    Args:
        file_path: Path to the audio/video file.
        initial_prompt_override: When non-empty, replaces both the
            user-configured override and the language-default chain
            entirely. Used by ``WhisperLocalProvider`` (Phase 2B) so
            ``SplittingTranscriber`` can seed chunk N with chunk N-1's
            tail without going through the language-default fallback.

    Returns:
        List of segment dicts with keys: text, start, end, language.
    """
    model, batched = _ensure_loaded()
    whisper_config = settings.indexing.whisper

    if initial_prompt_override and initial_prompt_override.strip():
        # Phase 2B precedence (1): caller-supplied prior text wins
        # over both the search-config override and the per-language
        # default. This only fires for chunked Whisper calls (chunk
        # N>0); chunk 0 / un-chunked passes ``None`` and falls into
        # the legacy resolution below (R1 spec M-R1-2).
        initial_prompt: str | None = initial_prompt_override
    else:
        override = whisper_config.initial_prompt or ""
        if override.strip():
            initial_prompt = override
        else:
            detected = _detect_language(model, file_path)
            initial_prompt = resolve_initial_prompt(detected, "")

    if batched is not None:
        return _transcribe_batched(
            batched, file_path, whisper_config, initial_prompt
        )
    return _transcribe_sequential(
        model, file_path, whisper_config, initial_prompt
    )


def _transcribe_batched(
    pipeline: object,
    file_path: str,
    whisper_config: object,
    initial_prompt: str | None,
) -> list[dict]:
    """Transcribe using BatchedInferencePipeline for faster throughput."""
    try:
        transcribe_kwargs: dict = {
            "batch_size": whisper_config.batch_size,
            "beam_size": whisper_config.beam_size,
            "language": None,
            "word_timestamps": True,
            "initial_prompt": initial_prompt,
        }
        if whisper_config.compression_ratio_threshold > 0:
            transcribe_kwargs["compression_ratio_threshold"] = (
                whisper_config.compression_ratio_threshold
            )
        if whisper_config.no_speech_threshold > 0:
            transcribe_kwargs["no_speech_threshold"] = (
                whisper_config.no_speech_threshold
            )
        if whisper_config.log_prob_threshold != 0:
            transcribe_kwargs["log_prob_threshold"] = (
                whisper_config.log_prob_threshold
            )
        segments_iter, info = pipeline.transcribe(
            file_path, **transcribe_kwargs
        )

        detected_language = info.language
        logger.info(
            "Transcribing %s (detected language: %s, duration: %.1fs, batched=%d)",
            file_path, detected_language, info.duration, whisper_config.batch_size,
        )

        segments: list[dict] = []
        for segment in segments_iter:
            text = segment.text.strip()
            if text:
                words = [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in (segment.words or [])
                ]
                segments = [
                    *segments,
                    {
                        "text": text,
                        "start": segment.start,
                        "end": segment.end,
                        "language": detected_language,
                        "words": words,
                    },
                ]

        return segments

    except Exception as e:
        logger.warning(
            "Batched transcription failed for %s: %s, falling back to sequential",
            file_path, e,
        )
        model, _ = _ensure_loaded()
        return _transcribe_sequential(
            model, file_path, whisper_config, initial_prompt
        )


def _transcribe_sequential(
    model: object,
    file_path: str,
    whisper_config: object,
    initial_prompt: str | None,
) -> list[dict]:
    """Transcribe sequentially with VAD fallback."""
    # Try with VAD first, fall back without VAD if it fails.
    # faster-whisper's VAD can raise IndexError ("tuple index out of range")
    # on certain audio streams (very short, silence-only, or corrupted).
    for attempt, use_vad in enumerate([True, False]):
        try:
            transcribe_kwargs: dict = {
                "beam_size": whisper_config.beam_size,
                "language": None,
                "vad_filter": use_vad,
                "condition_on_previous_text": whisper_config.condition_on_previous_text,
                "word_timestamps": True,
                "initial_prompt": initial_prompt,
            }
            if whisper_config.compression_ratio_threshold > 0:
                transcribe_kwargs["compression_ratio_threshold"] = (
                    whisper_config.compression_ratio_threshold
                )
            if whisper_config.no_speech_threshold > 0:
                transcribe_kwargs["no_speech_threshold"] = (
                    whisper_config.no_speech_threshold
                )
            if whisper_config.log_prob_threshold != 0:
                transcribe_kwargs["log_prob_threshold"] = (
                    whisper_config.log_prob_threshold
                )
            if use_vad:
                transcribe_kwargs["vad_parameters"] = {
                    "min_silence_duration_ms": 500
                }

            segments_iter, info = model.transcribe(
                file_path, **transcribe_kwargs
            )

            detected_language = info.language
            logger.info(
                "Transcribing %s (detected language: %s, duration: %.1fs, vad=%s)",
                file_path, detected_language, info.duration, use_vad,
            )

            segments: list[dict] = []
            for segment in segments_iter:
                text = segment.text.strip()
                if text:
                    words = [
                        {"word": w.word, "start": w.start, "end": w.end}
                        for w in (segment.words or [])
                    ]
                    segments = [
                        *segments,
                        {
                            "text": text,
                            "start": segment.start,
                            "end": segment.end,
                            "language": detected_language,
                            "words": words,
                        },
                    ]

            return segments

        except (IndexError, RuntimeError) as e:
            if attempt == 0:
                logger.warning(
                    "Transcription with VAD failed for %s: %s, retrying without VAD",
                    file_path, e,
                )
                continue
            logger.error("Transcription failed for %s: %s", file_path, e)
            return []
        except Exception as e:
            logger.error("Transcription failed for %s: %s", file_path, e)
            return []

    return []


_PUNCT_BREAK = frozenset(".。!?！？…")
_PUNCT_SOFT = frozenset(",、;:：；")
_SILENCE_GAP = 0.4  # seconds; larger gaps are treated as sentence boundaries


def _flatten_words(segments: list[dict]) -> list[dict]:
    """Flatten per-segment ``words`` lists into a single ordered list.

    Segments without word timestamps are skipped (batched mode occasionally
    omits words for very short utterances). The resulting list is ordered
    by the underlying segment iteration so timestamps stay monotonic.

    Carries ``speaker_id`` through when present (Phase 1C: diarized
    cloud providers stash speaker labels per word). Legacy callers
    that omit the field continue to work — ``get`` returns None.
    """
    flat: list[dict] = []
    for seg in segments:
        language = seg.get("language", "")
        for w in seg.get("words") or []:
            text = (w.get("word") or "").strip()
            if not text:
                continue
            flat.append({
                "text": text,
                "start": float(w["start"]),
                "end": float(w["end"]),
                "language": language,
                "speaker_id": w.get("speaker_id"),
            })
    return flat


def _majority_speaker(words: list[dict]) -> str | None:
    """Return the most common ``speaker_id`` among ``words`` (None tie-broken).

    Used when emitting a chunk: if every word inside agrees on a
    speaker, that label is preserved; if the chunk happened to
    straddle speakers (e.g. an under-min flush that ignored R4),
    majority wins. ``None`` entries are not counted — a chunk with
    half NULL / half "spk_0" reports ``"spk_0"``.
    """
    counts: dict[str, int] = {}
    for w in words:
        sid = w.get("speaker_id")
        if sid is None:
            continue
        counts[sid] = counts.get(sid, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _build_chunks_from_words(
    words: list[dict],
    min_duration: float,
    max_duration: float,
) -> list[dict]:
    """Build search-oriented transcript chunks from word-level timestamps.

    Chunks are cut preferentially at sentence boundaries (terminal
    punctuation, then soft punctuation, then silence gaps). The min/max
    duration bounds are hard constraints: chunks shorter than min are
    extended past a break candidate; chunks at or above max are flushed
    unconditionally to avoid the unbounded-chunk regression that the old
    ``_merge_segments`` allowed when a single Whisper segment exceeded
    max_duration (ref hako qx19g-IBnLc7_C-WBo-rf).
    """
    if not words:
        return []

    def _is_break(word_text: str, gap_to_next: float) -> int:
        """Return 2 for hard break, 1 for soft break, 0 otherwise."""
        if word_text and word_text[-1] in _PUNCT_BREAK:
            return 2
        if gap_to_next >= _SILENCE_GAP:
            return 2
        if word_text and word_text[-1] in _PUNCT_SOFT:
            return 1
        return 0

    chunks: list[dict] = []
    current: list[dict] = []
    chunk_start = words[0]["start"]
    language = words[0].get("language", "")

    for i, word in enumerate(words):
        current.append(word)
        chunk_end = word["end"]
        duration = chunk_end - chunk_start
        next_start = words[i + 1]["start"] if i + 1 < len(words) else chunk_end
        gap = max(0.0, next_start - chunk_end)
        break_strength = _is_break(word["text"], gap)

        # R4 (Phase 1C): speaker change between this word and the next
        # is treated as a hard boundary, but only past min_duration so
        # rapid Q/A exchanges don't shred chunks under the searchable
        # floor. The check requires both speaker IDs non-NULL — mixed
        # streams (cloud-diarized followed by re-indexed local-whisper
        # rows) fall back to the legacy R1-R3 conditions.
        speaker_change = False
        if i + 1 < len(words):
            this_speaker = word.get("speaker_id")
            next_speaker = words[i + 1].get("speaker_id")
            if (
                this_speaker is not None
                and next_speaker is not None
                and this_speaker != next_speaker
            ):
                speaker_change = True

        should_flush = False
        if duration >= max_duration:
            should_flush = True
        elif duration >= min_duration and break_strength == 2:
            should_flush = True
        elif duration >= min_duration and speaker_change:
            should_flush = True
        elif duration >= min_duration * 1.5 and break_strength == 1:
            should_flush = True

        if should_flush:
            chunks.append({
                "text": _join_words([w["text"] for w in current], language),
                "start": chunk_start,
                "end": chunk_end,
                "language": language,
                "speaker_id": _majority_speaker(current),
            })
            current = []
            if i + 1 < len(words):
                chunk_start = words[i + 1]["start"]

    if current:
        chunks.append({
            "text": _join_words([w["text"] for w in current], language),
            "start": chunk_start,
            "end": current[-1]["end"],
            "language": language,
            "speaker_id": _majority_speaker(current),
        })

    return chunks


_RE_CJK_SPACE = re.compile(
    r"(?<=[ぁ-んァ-ヶー一-龯々]) (?=[ぁ-んァ-ヶー一-龯々])"
)


def _strip_cjk_spaces(text: str) -> str:
    """Remove half-width spaces between CJK characters.

    Whisper sometimes inserts spurious spaces inside Japanese tokens
    (e.g. "お 香" → "お香"). Spaces between CJK and Latin are kept.
    """
    return _RE_CJK_SPACE.sub("", text)


def _join_words(tokens: list[str], language: str) -> str:
    """Join tokens with a space when the language expects inter-word spaces.

    CJK languages (ja/zh/ko/th) are joined without a separator. Other
    languages get a single-space join. The caller passes a stripped
    token per word, so we never produce leading/trailing whitespace.
    """
    if not tokens:
        return ""
    lang = (language or "").lower()
    if lang.startswith(("ja", "jp", "zh", "ko", "th")):
        return _strip_cjk_spaces("".join(tokens))
    return " ".join(tokens)


def _merge_segments(
    segments: list[dict],
    min_duration: int,
    max_duration: int,
) -> list[dict]:
    """Merge cue-shaped segments into larger chunks.

    Used for the LoftRef path where raw inputs are WebVTT cues (no word
    timestamps). For the main Whisper path use ``_build_chunks_from_words``
    instead — it respects sentence boundaries and enforces max_duration
    strictly.
    """
    if not segments:
        return []

    chunks: list[dict] = []
    current_texts: list[str] = []
    current_start: float = segments[0]["start"]
    current_end: float = segments[0]["end"]
    current_language: str = segments[0].get("language", "")

    for segment in segments:
        segment_duration = segment["end"] - current_start

        if segment_duration > max_duration and current_texts:
            # Flush current chunk
            chunks = [
                *chunks,
                {
                    "text": _join_words(current_texts, current_language),
                    "start": current_start,
                    "end": current_end,
                    "language": current_language,
                },
            ]
            current_texts = [segment["text"]]
            current_start = segment["start"]
            current_end = segment["end"]
        else:
            current_texts = [*current_texts, segment["text"]]
            current_end = segment["end"]

        # Flush if we've exceeded min_duration
        if current_end - current_start >= min_duration:
            chunks = [
                *chunks,
                {
                    "text": _join_words(current_texts, current_language),
                    "start": current_start,
                    "end": current_end,
                    "language": current_language,
                },
            ]
            current_texts = []
            current_start = current_end

    # Flush remaining
    if current_texts:
        chunks = [
            *chunks,
            {
                "text": _join_words(current_texts, current_language),
                "start": current_start,
                "end": current_end,
                "language": current_language,
            },
        ]

    return chunks


async def index_whisper(file_id: str) -> bool:
    """Transcribe a media file and index the transcript.

    Phase 1C orchestration:

    * Resolves the file's drive + path from the search DB
    * Routes ``.loft`` files to the legacy adjacent-VTT path
      (untouched — those don't go through a TranscriptionProvider)
    * Selects the configured provider via :func:`get_provider`
    * Layer 1 policy: cloud providers (`sends_audio_offhost=True`)
      fall back to ``whisper_local`` when ``transcription_cloud=false``
      for the drive (fail-closed via ``default_on_failure=False``)
    * Layer 2 policy: re-evaluates ``intelligence.index`` at dequeue
      so a recent flip is honoured even if the file was enqueued
      before the change
    * Hands off to :func:`_do_transcribe_and_index`, which owns the
      provider call, retry / circuit breaker, JobRecord lifecycle,
      and DB write phase

    Returns:
        True iff the file was indexed (transcript written or
        legitimate zero-segment success). False on provider error or
        skipped-by-policy — both leave ``whisper_indexed=False`` so a
        future re-index can re-attempt.
    """
    # --- Resolve file_id → (path, drive, mime) ---
    def _resolve() -> tuple[str, str, str, bool] | None:
        """None means "no such active file"; else (mime, path, drive, skip)."""
        unsupported = False
        with get_search_db() as session:
            file = session.query(IndexedFile).filter_by(
                file_id=file_id, active=True
            ).first()
            if file is None:
                return None
            mime_type = file.mime_type
            file_path = file.file_path
            drive = file.drive

            # Unsupported MIME: mark indexed so reconcile won't re-enqueue
            # the file, but record the skip in JobRecord + INFO log so
            # operators can answer "why isn't this file transcribed?"
            # without scraping logs (Phase 2F, hako A-gF1mK3kDjRjS_dfuq1B).
            if mime_type not in TRANSCRIBABLE_TYPES and mime_type != LOFT_MIME:
                file.whisper_indexed = True
                session.commit()
                logger.info(
                    "File %s skipped for transcription: mime=%s not in "
                    "TRANSCRIBABLE_TYPES",
                    file_id, mime_type,
                )
                unsupported = True
        return mime_type, file_path, drive, unsupported

    # This coroutine is awaited straight from ``_whisper_worker``, so the
    # lookup runs on the indexer's event loop unless it hops off. The
    # query is one indexed row; waiting on the write lock is the part
    # that stalls every endpoint.
    resolved = await asyncio.to_thread(_resolve)
    if resolved is None:
        return False
    mime_type, file_path, drive, unsupported_mime = resolved

    # Recorded outside the block above: _record_skipped_job() opens its own
    # write session, so running it on a worker thread while this coroutine
    # still held the (non-reentrant) write lock deadlocked both.
    if unsupported_mime:
        await asyncio.to_thread(
            _record_skipped_job,
            file_id,
            f"mime={mime_type}",
        )
        return True

    loft_temp_path: str | None = None
    if mime_type == LOFT_MIME:
        loft_temp_path = _loft_stt_temp_path(file_path)
        if loft_temp_path is None:
            # External source carve-out (adjacent .vtt → not a provider
            # call). Called outside the DB context to avoid self-deadlock
            # on the internal write lock.
            return await asyncio.to_thread(_index_loft_vtt, file_id, file_path)
        file_path = loft_temp_path

    if not validate_file_path(file_path):
        logger.error("File path validation failed for %s: %s", file_id, file_path)
        return False

    configured = settings.transcription.provider
    try:
        provider = get_provider(configured)
    except (ValueError, FatalError) as exc:
        # Misconfigured provider name OR missing API key. Record the
        # failure so operators see it, then bail.
        logger.error(
            "Provider %r init failed for %s: %s", configured, file_id, exc
        )
        await _record_failed_job(
            file_id=file_id,
            provider=configured,
            error=exc,
        )
        await _emit_ws_event(
            "intelligence.transcription.failed",
            {
                "file_id": file_id,
                "drive": drive,
                "provider": configured,
                "error_class": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        return False

    # Layer 1 — per-drive cloud policy (fail closed).
    if provider.capabilities.sends_audio_offhost:
        from app.policy_client import is_feature_enabled

        try:
            allowed = await is_feature_enabled(
                drive,
                "transcription_cloud",
                default_on_failure=False,
            )
        except TransientError:
            # Cold-start grace path — leave the job for a future retry.
            logger.info(
                "Drive %s: cloud policy lookup transient error, deferring %s",
                drive, file_id,
            )
            return False
        if not allowed:
            logger.info(
                "Drive %s: transcription_cloud=false, falling back to "
                "whisper_local for %s",
                drive, file_id,
            )
            try:
                provider = get_provider("whisper_local")
            except (ValueError, FatalError) as exc:
                logger.error(
                    "whisper_local fallback init failed for %s: %s",
                    file_id, exc,
                )
                await _record_failed_job(
                    file_id=file_id, provider="whisper_local", error=exc,
                )
                return False

    # Layer 2 — re-evaluate ``intelligence.index`` at dequeue.
    from app.policy_client import is_feature_enabled

    if not await is_feature_enabled(drive, "index"):
        logger.info(
            "Drive %s: intelligence.index=false at dequeue, skipping %s",
            drive, file_id,
        )
        return False

    # Phase 2A: providers without native word timestamps synthesise
    # them by uniformly splitting segment text. Surface a one-shot WARN
    # per dispatch so operators know alignment precision is reduced;
    # the chunker itself does not branch on this flag.
    if not provider.capabilities.supports_word_timestamps:
        logger.warning(
            "Provider %s does not return native word timestamps; "
            "synthetic alignment will be used (lower precision) for %s",
            provider.name, file_id,
        )

    try:
        success = await _do_transcribe_and_index(file_id, file_path, drive, provider)
        if not success and loft_temp_path is not None:
            await asyncio.to_thread(_mark_loft_stt_temp_consumed, file_id)
        return success
    finally:
        _cleanup_loft_stt_temp(loft_temp_path)


async def _record_failed_job(
    *,
    file_id: str,
    provider: str | None,
    error: Exception,
) -> None:
    """Insert a single ``JobRecord`` row marking a transcription failure.

    Called from ``index_whisper`` for early-exit failures (provider
    factory error, missing API key, fallback init failure) where no
    "running" row exists yet.
    """
    def _write() -> None:
        with get_search_db() as session:
            session.add(JobRecord(
                file_id=file_id,
                job_kind="transcription",
                provider=provider,
                status="failed",
                error_class=type(error).__name__,
                error_message=str(error)[:1000],
                completed_at=datetime.now(UTC),
            ))

    try:
        await asyncio.to_thread(_write)
    except Exception:
        # JobRecord write is best-effort observability; never fail the
        # caller because of an audit-log row.
        logger.exception("Failed to write JobRecord for %s", file_id)


def _record_skipped_job(file_id: str, reason: str) -> None:
    """Insert a status='skipped' JobRecord row.

    Phase 2F: replaces the legacy silent-skip path so admins can
    SELECT ``status='skipped'`` and see exactly why a file's
    transcript is missing (hako A-gF1mK3kDjRjS_dfuq1B). The row has
    ``provider=None`` because no provider was contacted, and a
    non-retryable ``error_class`` so this is not lumped together
    with transient cloud failures. ``whisper_indexed=True`` is set
    by the caller so reconcile does not re-enqueue the file.
    """
    try:
        with get_search_db() as session:
            session.add(JobRecord(
                file_id=file_id,
                job_kind="transcription",
                provider=None,
                status="skipped",
                error_class="UnsupportedMimeType",
                error_message=reason[:1000],
                completed_at=datetime.now(UTC),
            ))
    except Exception:
        logger.exception("Failed to write skipped JobRecord for %s", file_id)


async def _do_transcribe_and_index(
    file_id: str,
    file_path: str,
    drive: str,
    provider,
) -> bool:
    """Run a provider transcription and persist the result.

    Owns the JobRecord lifecycle:

    * insert ``status='running'`` row before invoking the provider
    * on success / 0-segment-silent: update to ``status='succeeded'``
      and flip ``whisper_indexed=True``
    * on provider error: update to ``status='failed'`` with the
      classified ``error_class``, leave ``whisper_indexed=False`` so
      a future re-index can re-attempt, and return False so
      :meth:`IndexManager.requeue_after_whisper` is NOT called (would
      otherwise enqueue summaries / auto_tags with no transcript).

    Emits ``intelligence.transcription.completed`` /
    ``intelligence.transcription.failed`` WS events on the way out.
    """
    job_id = await asyncio.to_thread(_insert_running_job, file_id, provider.name)

    try:
        if provider.capabilities.handles_own_retry:
            # Phase 2B: ``SplittingTranscriber`` runs per-chunk
            # ``transcribe_with_retry`` itself. Wrapping again here
            # would double-count failures on the inner provider's
            # circuit breaker and discard already-transcribed chunks
            # when the outer retry kicks in.
            segments = await provider.transcribe(
                file_path,
                language_hint=settings.transcription.language_hint or None,
                hotwords=list(settings.transcription.hotwords) or None,
            )
        else:
            segments = await transcribe_with_retry(
                provider,
                file_path,
                language_hint=settings.transcription.language_hint or None,
                hotwords=list(settings.transcription.hotwords) or None,
            )
    except TranscriptionError as exc:
        # All classified provider errors land here (TransientError /
        # RateLimitError exhausted, FatalError, CircuitBreakerOpen).
        await asyncio.to_thread(
            _finish_job_failed, job_id, exc,
        )
        await _emit_ws_event(
            "intelligence.transcription.failed",
            {
                "file_id": file_id,
                "drive": drive,
                "provider": provider.name,
                "error_class": type(exc).__name__,
                "error_message": str(exc)[:500],
            },
        )
        return False
    except Exception as exc:
        # Unclassified bug in the provider — record as TransientError
        # so the operator sees something but the job can be retried.
        logger.exception("Unclassified provider error for %s", file_id)
        await asyncio.to_thread(_finish_job_failed, job_id, exc)
        await _emit_ws_event(
            "intelligence.transcription.failed",
            {
                "file_id": file_id,
                "drive": drive,
                "provider": provider.name,
                "error_class": type(exc).__name__,
                "error_message": str(exc)[:500],
            },
        )
        return False

    # Convert TranscriptionSegment → legacy dict shape that the chunk
    # builder + DB writer already understand.
    raw_segments = _segments_to_dicts(segments)
    has_diarization = any(
        w.get("speaker_id") is not None
        for seg in raw_segments
        for w in seg.get("words") or []
    )

    await asyncio.to_thread(
        _persist_transcript,
        file_id,
        provider.name,
        raw_segments,
        job_id,
    )

    # Snapshot counts for the completion event after persist.
    words_count = sum(len(seg.get("words") or []) for seg in raw_segments)
    await _emit_ws_event(
        "intelligence.transcription.completed",
        {
            "file_id": file_id,
            "drive": drive,
            "provider": provider.name,
            "segments_count": len(raw_segments),
            "words_count": words_count,
            "has_diarization": has_diarization,
        },
    )
    return True


def _segments_to_dicts(segments) -> list[dict]:
    """Convert a list of :class:`TranscriptionSegment` to legacy dicts.

    Carries ``speaker_id`` through from :class:`WordToken`. The
    legacy dict shape is what ``_flatten_words`` /
    ``_build_chunks_from_words`` consume.
    """
    out: list[dict] = []
    for seg in segments:
        out.append({
            "text": seg.text,
            "start": seg.start,
            "end": seg.end,
            "language": seg.language,
            "words": [
                {
                    "word": w.text,
                    "start": w.start,
                    "end": w.end,
                    "speaker_id": w.speaker_id,
                }
                for w in seg.words
            ],
        })
    return out


def _insert_running_job(file_id: str, provider_name: str) -> int:
    """Insert a fresh JobRecord with status='running' and return its id."""
    with get_search_db() as session:
        record = JobRecord(
            file_id=file_id,
            job_kind="transcription",
            provider=provider_name,
            status="running",
        )
        session.add(record)
        session.flush()
        return int(record.id)


def _finish_job_failed(job_id: int, exc: Exception) -> None:
    """Mark a previously-inserted JobRecord as failed."""
    with get_search_db() as session:
        record = session.query(JobRecord).filter_by(id=job_id).first()
        if record is None:
            # Should never happen — we wrote the row a moment ago.
            return
        record.status = "failed"
        record.error_class = type(exc).__name__
        record.error_message = str(exc)[:1000]
        record.completed_at = datetime.now(UTC)


def _finish_job_succeeded(job_id: int) -> None:
    """Mark a previously-inserted JobRecord as succeeded."""
    with get_search_db() as session:
        record = session.query(JobRecord).filter_by(id=job_id).first()
        if record is None:
            return
        record.status = "succeeded"
        record.completed_at = datetime.now(UTC)


def _index_tfidf_keywords(file_id: str, chunk_texts: list[str]) -> None:
    """Build and store a TF-IDF keyword embedding for a file.

    Extracts top keywords from the supplied transcript chunks, embeds
    them as a single dense vector via ``embed_passages``, and saves the
    result to ``vec_text`` with ``embedding_type="tfidf_keywords"``.
    Sets ``tfidf_keywords_indexed=True`` on success (or when the
    transcript is too short to be worth indexing).

    Leaves the flag ``False`` on embedding failures so the backfill
    worker can retry on the next startup sweep.
    """
    text = " ".join(chunk_texts).strip()

    def _mark_done() -> None:
        with get_search_db() as session:
            session.execute(
                sql_text(
                    "UPDATE indexed_files SET tfidf_keywords_indexed=1 "
                    "WHERE file_id=:fid"
                ),
                {"fid": file_id},
            )

    if len(text) < 20:
        _mark_done()
        return

    with get_search_db_read() as session:
        row = (
            session.query(IndexedFile.filename)
            .filter_by(file_id=file_id)
            .first()
        )
    filename = row.filename if row else ""

    from app.tfidf import extract_top_keywords
    keywords = extract_top_keywords(text, filename)
    if not keywords:
        _mark_done()
        return

    keyword_string = " ".join(keywords)
    try:
        vectors = embed_passages([keyword_string])
    except Exception as e:
        logger.warning("TF-IDF keyword embedding failed for %s: %s", file_id, e)
        return  # leave tfidf_keywords_indexed=False for backfill retry

    vector = vectors[0]
    embedding_id = f"tk_{file_id}_{uuid.uuid4().hex[:8]}"

    with get_search_db() as session:
        old_embs = (
            session.query(Embedding)
            .filter_by(file_id=file_id, embedding_type="tfidf_keywords")
            .all()
        )
        for old in old_embs:
            session.execute(
                sql_text("DELETE FROM vec_text WHERE embedding_id = :id"),
                {"id": old.id},
            )
            session.delete(old)

        embedding_record = Embedding(
            id=embedding_id,
            file_id=file_id,
            embedding_type="tfidf_keywords",
            vector_table="vec_text",
            content_preview=keyword_string[:200],
        )
        session.add(embedding_record)
        session.flush()

        session.execute(
            sql_text(
                "INSERT INTO vec_text(embedding_id, vector) VALUES(:id, :vec)"
            ),
            {"id": embedding_id, "vec": vector.tobytes()},
        )

        session.execute(
            sql_text(
                "UPDATE indexed_files SET tfidf_keywords_indexed=1 "
                "WHERE file_id=:fid"
            ),
            {"fid": file_id},
        )

    # Same reasoning as the thumbnail route: this is the secondary signal
    # behind every video and audio similar-files answer, the cache has no
    # TTL, and a backfill write reaches it outside any webhook.
    from app.search import invalidate_similar_cache
    invalidate_similar_cache()

    logger.debug(
        "TF-IDF keyword embedding indexed for %s (%d keywords)",
        file_id, len(keywords),
    )


def index_tfidf_keywords_backfill(file_id: str) -> bool:
    """Backfill TF-IDF keyword embedding for a file that already has a transcript.

    Used by the TFIDF_KEYWORDS task worker to process files indexed
    before this feature was introduced.

    Returns True on success (including "too short to index"),
    False when the embedding step fails.
    """
    with get_search_db_read() as session:
        chunks = (
            session.query(TranscriptChunk.text)
            .filter_by(file_id=file_id)
            .order_by(TranscriptChunk.chunk_index)
            .all()
        )
    chunk_texts = [c.text for c in chunks if c.text.strip()]

    text = " ".join(chunk_texts).strip()
    if len(text) < 20:
        with get_search_db() as session:
            f = session.query(IndexedFile).filter_by(file_id=file_id).first()
            if f is not None:
                f.tfidf_keywords_indexed = True
        return True

    # Check if embedding already exists (race guard)
    with get_search_db_read() as session:
        existing = (
            session.query(Embedding)
            .filter_by(file_id=file_id, embedding_type="tfidf_keywords")
            .first()
        )
    if existing is not None:
        with get_search_db() as session:
            f = session.query(IndexedFile).filter_by(file_id=file_id).first()
            if f is not None:
                f.tfidf_keywords_indexed = True
        return True

    _index_tfidf_keywords(file_id, chunk_texts)

    # Verify the flag was set
    with get_search_db_read() as session:
        f = session.query(IndexedFile.tfidf_keywords_indexed).filter_by(file_id=file_id).first()
    return bool(f and f.tfidf_keywords_indexed)


def _persist_transcript(
    file_id: str,
    provider_name: str,
    raw_segments: list[dict],
    job_id: int,
) -> None:
    """DB write phase: chunks + words + embeddings + JobRecord update.

    Also handles the "0 segments / 0 chunks" silent-audio case by
    flipping ``whisper_indexed=True`` and marking the JobRecord as
    succeeded — preserving legacy behaviour so silent files don't get
    re-attempted forever.
    """
    if not raw_segments:
        with get_search_db() as session:
            file = session.query(IndexedFile).filter_by(file_id=file_id).first()
            if file is not None:
                file.whisper_indexed = True
        _finish_job_succeeded(job_id)
        return

    whisper_config = settings.transcription.whisper_local
    words = _flatten_words(raw_segments)
    chunks = _build_chunks_from_words(
        words,
        whisper_config.min_segment_duration,
        whisper_config.max_segment_duration,
    )

    if not chunks:
        with get_search_db() as session:
            file = session.query(IndexedFile).filter_by(file_id=file_id).first()
            if file is not None:
                file.whisper_indexed = True
        _finish_job_succeeded(job_id)
        return

    chunk_texts = [c["text"] for c in chunks if c["text"].strip()]
    vectors = None
    if chunk_texts:
        try:
            vectors = embed_passages(chunk_texts)
        except Exception as e:
            logger.error("Whisper embedding failed for %s: %s", file_id, e)

    with get_search_db() as session:
        _remove_whisper_data(session, file_id)

        for idx, chunk in enumerate(chunks):
            transcript = TranscriptChunk(
                file_id=file_id,
                chunk_index=idx,
                text=chunk["text"],
                language=chunk["language"],
                timestamp_start=chunk["start"],
                timestamp_end=chunk["end"],
                speaker_id=chunk.get("speaker_id"),
            )
            session.add(transcript)

        if words:
            session.bulk_insert_mappings(
                TranscriptWord,
                [
                    {
                        "file_id": file_id,
                        "text": w["text"],
                        "language": w.get("language", ""),
                        "timestamp_start": w["start"],
                        "timestamp_end": w["end"],
                        "speaker_id": w.get("speaker_id"),
                    }
                    for w in words
                ],
            )

        if vectors is not None:
            text_idx = 0
            for idx, chunk in enumerate(chunks):
                if not chunk["text"].strip():
                    continue

                try:
                    embedding_id = f"wh_{file_id}_{idx}_{uuid.uuid4().hex[:8]}"

                    embedding_record = Embedding(
                        id=embedding_id,
                        file_id=file_id,
                        embedding_type="whisper",
                        vector_table="vec_text",
                        content_preview=chunk["text"][:200],
                        timestamp_start=chunk["start"],
                        timestamp_end=chunk["end"],
                    )
                    session.add(embedding_record)
                    session.flush()

                    vec_bytes = vectors[text_idx].tobytes()
                    session.execute(
                        sql_text(
                            "INSERT INTO vec_text(embedding_id, vector) VALUES(:id, :vec)"
                        ),
                        {"id": embedding_id, "vec": vec_bytes},
                    )

                    text_idx += 1

                except Exception as e:
                    logger.error(
                        "Failed to store whisper embedding %d for %s: %s",
                        idx, file_id, e,
                    )

        fts_chunks = [
            {"chunk_index": idx, "text": chunk["text"]}
            for idx, chunk in enumerate(chunks)
            if chunk["text"].strip()
        ]
        if fts_chunks:
            upsert_fts_transcripts(session, file_id, fts_chunks)

        file = session.query(IndexedFile).filter_by(file_id=file_id).first()
        if file is not None:
            file.whisper_indexed = True

        # Mark the JobRecord succeeded inside the same session so the
        # state flip and the transcript write commit atomically.
        record = session.query(JobRecord).filter_by(id=job_id).first()
        if record is not None:
            record.status = "succeeded"
            record.completed_at = datetime.now(UTC)

    _index_tfidf_keywords(file_id, chunk_texts)


def fail_orphaned_running_jobs() -> None:
    """Flip ``status='running'`` JobRecords to ``failed`` at startup.

    Spec ``2026-05-07-cloud-transcription-providers.md`` §"Hot-swap /
    半端ジョブ". When the intelligence container restarts mid-job,
    any "running" rows are inherently orphaned — the in-memory worker
    that owned them is gone. We mark them failed with
    ``error_class='ContainerRestart'`` so operators can distinguish
    a crash from a real provider error, and call
    :func:`_remove_whisper_data` to drop any partial chunks / words /
    embeddings the dead worker left behind.

    Safe to call repeatedly; a no-op when there are no running rows.

    Phase 1 assumes a single intelligence container — the multi-worker
    variant (worker_id column + per-worker scoping) is Phase 2.
    """
    with get_search_db() as session:
        running = session.query(JobRecord).filter_by(status="running").all()
        if not running:
            return
        now = datetime.now(UTC)
        for record in running:
            record.status = "failed"
            record.error_class = "ContainerRestart"
            record.error_message = (
                "Container restarted while job was in progress"
            )
            record.completed_at = now
            try:
                _remove_whisper_data(session, record.file_id)
            except Exception:
                # Cleanup is best-effort: we don't want a sqlite-vec
                # absence (test envs without the loadable extension)
                # to block startup. The "running" → "failed" status
                # flip is the critical part.
                logger.exception(
                    "Failed to purge partial whisper data for %s",
                    record.file_id,
                )
        logger.info(
            "Marked %d orphaned 'running' transcription job(s) as failed",
            len(running),
        )


def _remove_whisper_data(session: object, file_id: str) -> None:
    """Remove existing Whisper data (transcripts + embeddings) for a file.

    Args:
        session: Database session.
        file_id: The file ID.
    """
    # Remove transcript chunks, word-level rows, and FTS5 mirror
    session.query(TranscriptChunk).filter_by(file_id=file_id).delete()
    session.query(TranscriptWord).filter_by(file_id=file_id).delete()
    delete_fts_transcripts(session, file_id)

    # Remove whisper embeddings and their vectors
    existing = (
        session.query(Embedding)
        .filter_by(file_id=file_id, embedding_type="whisper")
        .all()
    )

    if existing:
        for emb in existing:
            session.execute(
                sql_text("DELETE FROM vec_text WHERE embedding_id = :id"),
                {"id": emb.id},
            )

        for emb in existing:
            session.delete(emb)

    session.flush()


def _parse_vtt_cues(vtt_path: str) -> list[dict]:
    """Parse a WebVTT file into timestamped segments.

    Returns a list of dicts with keys: text, start, end, language.
    """
    import re
    from pathlib import Path

    path = Path(vtt_path)
    if not path.exists():
        return []

    # Detect language from filename: stem.lang.vtt
    lang = ""
    parts = path.stem.rsplit(".", 1)
    if len(parts) == 2 and re.match(r"^[a-zA-Z]{2}(?:-[a-zA-Z]{2,4})?$", parts[1]):
        lang = parts[1]

    content = path.read_text(encoding="utf-8", errors="replace")

    # Detect language from VTT header (e.g. "Language: ja")
    if not lang:
        lang_match = re.search(r"^Language:\s*(\S+)", content, re.MULTILINE)
        if lang_match:
            lang = lang_match.group(1)
    timestamp_re = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
    )
    tag_re = re.compile(r"<[^>]+>")

    cues: list[dict] = []
    current_start = 0.0
    current_end = 0.0
    current_text_lines: list[str] = []
    in_cue = False

    for line in content.splitlines():
        line = line.strip()

        m = timestamp_re.match(line)
        if m:
            if in_cue and current_text_lines:
                text = " ".join(current_text_lines).strip()
                if text:
                    cues.append({
                        "text": text,
                        "start": current_start,
                        "end": current_end,
                        "language": lang,
                    })

            h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(x) for x in m.groups())
            current_start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
            current_end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
            current_text_lines = []
            in_cue = True
            continue

        if not line:
            if in_cue and current_text_lines:
                text = " ".join(current_text_lines).strip()
                if text:
                    cues.append({
                        "text": text,
                        "start": current_start,
                        "end": current_end,
                        "language": lang,
                    })
                current_text_lines = []
                in_cue = False
            continue

        if line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue

        if in_cue:
            cleaned = tag_re.sub("", line)
            if cleaned.strip():
                current_text_lines.append(cleaned.strip())

    if in_cue and current_text_lines:
        text = " ".join(current_text_lines).strip()
        if text:
            cues.append({
                "text": text,
                "start": current_start,
                "end": current_end,
                "language": lang,
            })

    return cues


def _dedup_rolling_cues(cues: list[dict]) -> list[dict]:
    """Remove duplicates caused by YouTube's rolling-subtitle format.

    YouTube auto-captions use a "scroll" style where each cue carries
    the tail of the previous cue as its first line(s), plus new text on
    the last line.  Snapshot cues (≤0.01 s) are pure duplicates of the
    preceding text and are dropped entirely.  For remaining cues, any
    leading lines that appeared in the previous cue's text are stripped
    so only genuinely new content survives.
    """
    if not cues:
        return []

    # Phase 1: drop snapshot cues (duration ≤ 0.01s)
    filtered = [c for c in cues if c["end"] - c["start"] > 0.015]
    if not filtered:
        return []

    # Phase 2: strip overlapping leading lines between consecutive cues
    deduped: list[dict] = []
    prev_text = ""
    for cue in filtered:
        text = cue["text"]
        # If this cue's text starts with the previous cue's text, keep
        # only the new suffix.
        if prev_text and text.startswith(prev_text):
            new_text = text[len(prev_text):].strip()
        else:
            # Fallback: try line-level dedup.  Split both into lines and
            # drop leading lines that match the tail of the previous cue.
            prev_lines = prev_text.split(" ") if prev_text else []
            cur_lines = text.split(" ")
            # Find how many leading lines of cur match trailing lines of prev
            overlap = 0
            for k in range(1, min(len(prev_lines), len(cur_lines)) + 1):
                if prev_lines[-k:] == cur_lines[:k]:
                    overlap = k
            new_text = " ".join(cur_lines[overlap:]).strip() if overlap else text
        if new_text:
            deduped.append({
                **cue,
                "text": new_text,
            })
        prev_text = text

    return deduped


def _index_loft_vtt(file_id: str, file_path: str) -> bool:
    """Index a .loft file using adjacent .vtt subtitles instead of Whisper.

    Reads .vtt file(s) next to the .loft, parses cues into segments,
    merges them, creates TranscriptChunks and embeddings — same output
    as the Whisper path.
    """
    from pathlib import Path

    loft_path = Path(file_path)
    stem = loft_path.stem
    parent = loft_path.parent

    vtt_candidates = sorted(parent.glob(f"{stem}*.vtt"))
    if not vtt_candidates:
        logger.info("No adjacent VTT for loft ref %s, marking as indexed", file_id)
        with get_search_db() as session:
            file = session.query(IndexedFile).filter_by(file_id=file_id).first()
            if file is not None:
                file.whisper_indexed = True
        return True

    best_vtt = vtt_candidates[0]
    for c in vtt_candidates:
        if c.name == f"{stem}.vtt":
            best_vtt = c
            break

    raw_segments = _dedup_rolling_cues(_parse_vtt_cues(str(best_vtt)))
    if not raw_segments:
        logger.info("VTT empty for loft ref %s, marking as indexed", file_id)
        with get_search_db() as session:
            file = session.query(IndexedFile).filter_by(file_id=file_id).first()
            if file is not None:
                file.whisper_indexed = True
        return True

    whisper_config = settings.indexing.whisper
    chunks = _merge_segments(
        raw_segments,
        whisper_config.min_segment_duration,
        whisper_config.max_segment_duration,
    )

    if not chunks:
        with get_search_db() as session:
            file = session.query(IndexedFile).filter_by(file_id=file_id).first()
            if file is not None:
                file.whisper_indexed = True
        return True

    chunk_texts = [c["text"] for c in chunks if c["text"].strip()]
    vectors = None
    if chunk_texts:
        try:
            vectors = embed_passages(chunk_texts)
        except Exception as e:
            logger.error("VTT embedding failed for %s: %s", file_id, e)

    with get_search_db() as session:
        _remove_whisper_data(session, file_id)

        for idx, chunk in enumerate(chunks):
            transcript = TranscriptChunk(
                file_id=file_id,
                chunk_index=idx,
                text=chunk["text"],
                language=chunk["language"],
                timestamp_start=chunk["start"],
                timestamp_end=chunk["end"],
            )
            session.add(transcript)

        if vectors is not None:
            text_idx = 0
            for idx, chunk in enumerate(chunks):
                if not chunk["text"].strip():
                    continue
                try:
                    embedding_id = f"wh_{file_id}_{idx}_{uuid.uuid4().hex[:8]}"
                    embedding_record = Embedding(
                        id=embedding_id,
                        file_id=file_id,
                        embedding_type="whisper",
                        vector_table="vec_text",
                        content_preview=chunk["text"][:200],
                        timestamp_start=chunk["start"],
                        timestamp_end=chunk["end"],
                    )
                    session.add(embedding_record)
                    session.flush()

                    vec_bytes = vectors[text_idx].tobytes()
                    session.execute(
                        sql_text(
                            "INSERT INTO vec_text(embedding_id, vector) VALUES(:id, :vec)"
                        ),
                        {"id": embedding_id, "vec": vec_bytes},
                    )
                    text_idx += 1
                except Exception as e:
                    logger.error(
                        "Failed to store VTT embedding %d for %s: %s",
                        idx, file_id, e,
                    )

        fts_chunks = [
            {"chunk_index": idx, "text": chunk["text"]}
            for idx, chunk in enumerate(chunks)
            if chunk["text"].strip()
        ]
        if fts_chunks:
            upsert_fts_transcripts(session, file_id, fts_chunks)

        file = session.query(IndexedFile).filter_by(file_id=file_id).first()
        if file is not None:
            file.whisper_indexed = True

    _index_tfidf_keywords(file_id, chunk_texts)
    logger.info("Indexed loft ref VTT transcript for %s (%d chunks)", file_id, len(chunks))
    return True
