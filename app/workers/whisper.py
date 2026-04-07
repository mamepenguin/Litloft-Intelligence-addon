"""Whisper transcription worker.

Transcribes audio from video/audio files using faster-whisper (CTranslate2).
The model is lazy-loaded and can be unloaded after idle to save RAM.

Only one Whisper task runs at a time (controlled by the indexer's semaphore).
"""

import asyncio
import logging
import threading
import time
import uuid

from app.config import settings, validate_file_path
from app.database import delete_fts_transcripts, get_search_db, upsert_fts_transcripts
from app.models import Embedding, IndexedFile, TranscriptChunk
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
    """Resolve config model name to faster-whisper model size.

    Args:
        config_name: Model name from config (e.g., "openai/whisper-small").

    Returns:
        Model size string for faster-whisper.
    """
    size_map = {
        "openai/whisper-tiny": "tiny",
        "openai/whisper-base": "base",
        "openai/whisper-small": "small",
        "openai/whisper-medium": "medium",
        "openai/whisper-large": "large-v3",
    }
    return size_map.get(config_name, "small")


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
    gc.collect()


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
        segments_iter, info = pipeline.transcribe(
            file_path,
            batch_size=whisper_config.batch_size,
            beam_size=whisper_config.beam_size,
            language=None,
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
                segments = [
                    *segments,
                    {
                        "text": text,
                        "start": segment.start,
                        "end": segment.end,
                        "language": detected_language,
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
            }
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
                    segments = [
                        *segments,
                        {
                            "text": text,
                            "start": segment.start,
                            "end": segment.end,
                            "language": detected_language,
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


def _merge_segments(
    segments: list[dict],
    min_duration: int,
    max_duration: int,
) -> list[dict]:
    """Merge small Whisper segments into larger chunks.

    Groups segments to target the min_duration while not exceeding
    max_duration. Preserves start/end timestamps.

    Args:
        segments: Raw Whisper segments.
        min_duration: Minimum target chunk duration in seconds.
        max_duration: Maximum chunk duration in seconds.

    Returns:
        List of merged chunk dicts.
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
                    "text": " ".join(current_texts),
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
                    "text": " ".join(current_texts),
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
                "text": " ".join(current_texts),
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

        if file.mime_type not in TRANSCRIBABLE_TYPES:
            file.whisper_indexed = True
            return True

        file_path = file.file_path

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
    # Remove transcript chunks (ORM + FTS5)
    session.query(TranscriptChunk).filter_by(file_id=file_id).delete()
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
