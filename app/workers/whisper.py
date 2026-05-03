"""Whisper transcription worker.

Transcribes audio from video/audio files using faster-whisper (CTranslate2).
The model is lazy-loaded and can be unloaded after idle to save RAM.

Only one Whisper task runs at a time (controlled by the indexer's semaphore).
"""

import asyncio
import logging
import re
import threading
import time
import uuid

from app.config import settings, validate_file_path
from app.database import delete_fts_transcripts, get_search_db, upsert_fts_transcripts
from app.models import Embedding, IndexedFile, TranscriptChunk, TranscriptWord
from app.workers.embedder import embed_passages
from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)

# Model state (lazy-loaded, with idle unload support)
_lock = threading.Lock()
_model: object | None = None
_batched_pipeline: object | None = None
_loaded = False
_last_used: float = 0.0

# Audio/video types that can be transcribed
TRANSCRIBABLE_TYPES = {
    "video/mp4", "video/webm", "video/quicktime", "video/x-matroska",
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg", "audio/flac",
    "audio/aac", "audio/m4a", "audio/x-m4a",
}

# External source files (.loft) that use adjacent .vtt instead of Whisper
LOFT_MIME = "application/vnd.litloft.loft+json"


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


def _transcribe_file(file_path: str) -> list[dict]:
    """Transcribe a media file using faster-whisper.

    Uses BatchedInferencePipeline when batch_size > 0 for faster throughput.
    Falls back to sequential transcription otherwise.

    Args:
        file_path: Path to the audio/video file.

    Returns:
        List of segment dicts with keys: text, start, end, language.
    """
    model, batched = _ensure_loaded()
    whisper_config = settings.indexing.whisper

    if batched is not None:
        return _transcribe_batched(batched, file_path, whisper_config)
    return _transcribe_sequential(model, file_path, whisper_config)


def _transcribe_batched(
    pipeline: object,
    file_path: str,
    whisper_config: object,
) -> list[dict]:
    """Transcribe using BatchedInferencePipeline for faster throughput."""
    try:
        transcribe_kwargs: dict = {
            "batch_size": whisper_config.batch_size,
            "beam_size": whisper_config.beam_size,
            "language": None,
            "word_timestamps": True,
            "initial_prompt": whisper_config.initial_prompt or None,
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
        return _transcribe_sequential(model, file_path, whisper_config)


def _transcribe_sequential(
    model: object,
    file_path: str,
    whisper_config: object,
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
                "initial_prompt": whisper_config.initial_prompt or None,
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
            })
    return flat


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

        should_flush = False
        if duration >= max_duration:
            should_flush = True
        elif duration >= min_duration and break_strength == 2:
            should_flush = True
        elif duration >= min_duration * 1.5 and break_strength == 1:
            should_flush = True

        if should_flush:
            chunks.append({
                "text": _join_words([w["text"] for w in current], language),
                "start": chunk_start,
                "end": chunk_end,
                "language": language,
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
    if lang.startswith(("ja", "zh", "ko", "th")):
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

    Runs transcription in a thread to avoid blocking the event loop.

    Args:
        file_id: The file ID to transcribe and index.

    Returns:
        True if indexing succeeded.
    """
    return await asyncio.to_thread(_index_whisper_sync, file_id)


def _index_whisper_sync(file_id: str) -> bool:
    """Synchronous Whisper indexing implementation.

    Splits into read → compute → write phases to minimize DB lock duration.

    Args:
        file_id: The file ID to transcribe and index.

    Returns:
        True if indexing succeeded.
    """
    # --- Phase 1: Read file info (short DB access) ---
    with get_search_db() as session:
        file = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()

        if file is None:
            return False

        mime_type = file.mime_type
        file_path = file.file_path

        if mime_type not in TRANSCRIBABLE_TYPES and mime_type != LOFT_MIME:
            file.whisper_indexed = True
            return True

    # External source: use adjacent .vtt instead of Whisper.
    # Called outside get_search_db() to avoid self-deadlock on _write_lock
    # (_index_loft_vtt internally acquires get_search_db()).
    if mime_type == LOFT_MIME:
        return _index_loft_vtt(file_id, file_path)

    if not validate_file_path(file_path):
        logger.error("File path validation failed for %s: %s", file_id, file_path)
        return False

    # --- Phase 2: Transcribe + embed (no DB access, may be very slow) ---
    raw_segments = _transcribe_file(file_path)
    if not raw_segments:
        with get_search_db() as session:
            file = session.query(IndexedFile).filter_by(file_id=file_id).first()
            if file is not None:
                file.whisper_indexed = True
        return True

    whisper_config = settings.indexing.whisper
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
        return True

    chunk_texts = [c["text"] for c in chunks if c["text"].strip()]
    vectors = None
    if chunk_texts:
        try:
            vectors = embed_passages(chunk_texts)
        except Exception as e:
            logger.error("Whisper embedding failed for %s: %s", file_id, e)

    # --- Phase 3: Write all results to DB (short transaction) ---
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

        # Write transcript chunks to FTS5 for keyword search
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

    return True


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

    logger.info("Indexed loft ref VTT transcript for %s (%d chunks)", file_id, len(chunks))
    return True
