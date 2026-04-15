"""Forced alignment wrapper for the refine worker.

Wraps WhisperX's wav2vec2-based ``align()`` so that ``refine`` can
replace time-proportional word timestamp allocation with acoustic
alignment. Called from ``refine.realign_words_for_chunk`` after the
LLM has rewritten a chunk's text.

Model state is module-level, keyed by language code, and explicitly
released via :func:`release_all` at refine-job completion — per-language
wav2vec2 models are ~1 GB and we don't want them resident between jobs.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterator

import app.config as config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# LRU-capped so pathological language diversity can't pin unbounded
# memory (each wav2vec2 model is ~1 GB). OrderedDict preserves insertion
# order; move_to_end marks recent use; popitem(last=False) evicts LRU.
_MODEL_CACHE_MAX = 2
_align_models: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
_waveform_cache: dict[str, Any] = {}
# Reference counter: folder-batch refine can fan out multiple parallel
# _run_refine_job tasks; we want to release the wav2vec2 model only
# after the **last** one finishes, not whenever any one ends (that
# would thrash the ~1 GB model reload across concurrent jobs).
_active_jobs: int = 0

# Scripts without whitespace word boundaries — align on characters so
# the resulting TranscriptWord rows match the existing CJK-char grid
# that subtitle_builder.py already expects.
_CJK_LANGS = frozenset({"ja", "zh", "ko", "th"})


def _normalise_lang(language: str) -> str:
    return (language or "").lower().strip().split("-", 1)[0]


def _load_align_model(language: str) -> tuple[Any, Any] | None:
    """Lazy-load the wav2vec2 alignment model for ``language``.

    Returns ``None`` when the language is unsupported by WhisperX or
    the load fails for any other reason (network, OOM, missing model).
    """
    lang = _normalise_lang(language)
    if not lang:
        return None

    with _lock:
        cached = _align_models.get(lang)
        if cached is not None:
            _align_models.move_to_end(lang)
            return cached
        try:
            import whisperx

            model_dir = str(config.settings.model_cache_dir)
            logger.info("aligner: loading wav2vec2 model lang=%s", lang)
            model, metadata = whisperx.load_align_model(
                language_code=lang,
                device="cpu",
                model_dir=model_dir,
            )
            while len(_align_models) >= _MODEL_CACHE_MAX:
                evicted_lang, _ = _align_models.popitem(last=False)
                logger.info(
                    "aligner: evicting LRU model lang=%s (cap=%d)",
                    evicted_lang, _MODEL_CACHE_MAX,
                )
            _align_models[lang] = (model, metadata)
            return (model, metadata)
        except Exception as e:
            logger.warning(
                "aligner: load_align_model failed lang=%s (%s)", lang, e
            )
            return None


def load_waveform(file_path: str) -> Any | None:
    """Load a 16 kHz mono waveform once per refine job.

    Cached on ``file_path`` so a single audio/video file is decoded at
    most once regardless of chunk count. Returns ``None`` when the
    file is missing or ffmpeg decoding fails — callers must treat
    that as the signal to skip alignment for every chunk of the file.
    """
    cached = _waveform_cache.get(file_path)
    if cached is not None:
        return cached
    try:
        import whisperx

        waveform = whisperx.load_audio(file_path)
    except Exception as e:
        logger.warning("aligner: load_audio failed for %s (%s)", file_path, e)
        return None
    _waveform_cache[file_path] = waveform
    return waveform


def align_segment(
    waveform: Any,
    chunk_start: float,
    chunk_end: float,
    text: str,
    language: str,
) -> list[dict] | None:
    """Forced-align ``text`` against the audio inside ``[chunk_start, chunk_end]``.

    Returns a list of ``{"text", "timestamp_start", "timestamp_end"}``
    dicts in the same absolute-seconds timeline the caller passed in,
    or ``None`` on any failure. CJK languages receive character-level
    rows (matching the existing char grid in ``transcript_words``);
    everything else receives word-level rows.
    """
    if waveform is None or not text or not text.strip():
        return None
    if chunk_end <= chunk_start:
        return None

    loaded = _load_align_model(language)
    if loaded is None:
        return None
    model, metadata = loaded
    lang = _normalise_lang(language)
    want_chars = lang in _CJK_LANGS

    try:
        import whisperx

        transcript = [{
            "start": float(chunk_start),
            "end": float(chunk_end),
            "text": text,
        }]
        result = whisperx.align(
            transcript=transcript,
            model=model,
            align_model_metadata=metadata,
            audio=waveform,
            device="cpu",
            return_char_alignments=want_chars,
        )
    except Exception as e:
        logger.warning(
            "aligner: align failed lang=%s (%s)", lang, type(e).__name__
        )
        return None

    segments = result.get("segments") if isinstance(result, dict) else None
    if not segments:
        return None
    seg = segments[0] if isinstance(segments[0], dict) else None
    if seg is None:
        return None

    key = "chars" if want_chars else "words"
    items = seg.get(key) or []
    units: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        token = item.get("char") if want_chars else item.get("word")
        if token is None:
            continue
        token_str = str(token)
        if not token_str.strip():
            continue
        ts_raw = item.get("start")
        te_raw = item.get("end")
        if ts_raw is None or te_raw is None:
            continue
        try:
            ts = float(ts_raw)
            te = float(te_raw)
        except (TypeError, ValueError):
            continue
        if te < ts:
            continue
        # Clamp to the caller's window: WhisperX wav2vec2 CTC can emit
        # edge tokens with timestamps that slightly overflow the
        # [chunk_start, chunk_end] segment (rounding, padding frames).
        # Dropping those here is the aligner's responsibility, not the
        # caller's — every downstream consumer gets a window-bounded
        # result by construction.
        if ts < chunk_start or te > chunk_end:
            continue
        units.append({
            "text": token_str,
            "timestamp_start": ts,
            "timestamp_end": te,
        })

    return units or None


@contextmanager
def job_scope() -> Iterator[None]:
    """Ref-counted scope for a refine job.

    Pair entry/exit so wav2vec2 models stay resident across parallel
    folder-batch jobs and release once all exit (normal, exception, or
    asyncio cancellation — all of which drive ``__exit__``). Prefer
    this over raw :func:`acquire_job` / :func:`release_job` so the
    release cannot be skipped by an early return / unanticipated
    exception path.
    """
    acquire_job()
    try:
        yield
    finally:
        try:
            release_job()
        except Exception:
            logger.warning("aligner.release_job failed", exc_info=True)


def acquire_job() -> None:
    """Mark the start of a refine job using the aligner.

    Pair with :func:`release_job` in a ``try/finally`` block. Multiple
    concurrent jobs share one loaded wav2vec2 model; the model is only
    freed after the last outstanding job calls :func:`release_job`.
    """
    global _active_jobs
    with _lock:
        _active_jobs += 1


def release_job() -> None:
    """Mark the end of a refine job and, if last, free loaded models.

    ~1 GB per language model, so we don't keep them warm between
    batches — refine is manual / on-index-tail and rare enough that
    the reload cost is acceptable. Safe to call unpaired (bottoms out
    at 0).
    """
    global _active_jobs, _align_models, _waveform_cache
    do_release = False
    with _lock:
        _active_jobs -= 1
        if _active_jobs <= 0:
            _active_jobs = 0
            if _align_models or _waveform_cache:
                logger.info(
                    "aligner: releasing %d model(s)", len(_align_models)
                )
                _align_models = OrderedDict()
                _waveform_cache = {}
                do_release = True
    if do_release:
        import gc

        gc.collect()


def release_all() -> None:
    """Force-release every loaded align model and waveform (test hook).

    Production code should use :func:`release_job` instead so the
    reference counter stays consistent. This exists mainly so tests
    can reset module state between runs.
    """
    global _align_models, _waveform_cache, _active_jobs
    with _lock:
        if _align_models:
            logger.info("aligner: force-releasing %d model(s)", len(_align_models))
        _align_models = OrderedDict()
        _waveform_cache = {}
        _active_jobs = 0
    import gc

    gc.collect()
