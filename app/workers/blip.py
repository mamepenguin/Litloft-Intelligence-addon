"""BLIP image captioning module.

Generates English text descriptions of images using BLIP
(Bootstrapping Language-Image Pre-training). Captions are stored
in the embeddings table and consumed by the auto_tags worker
to produce meaningful tags.

The model is lazy-loaded on first use and unloaded after an idle
period (controlled by settings.memory.blip_idle_unload) to free
the ~1GB of RAM the model occupies. Only active when models.blip
is set in search-config.yml.
"""

import logging
import threading
import time

from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

# Model state (lazy-loaded, with idle unload support)
_lock = threading.Lock()
_model: object | None = None
_processor: object | None = None
_loaded = False
_last_used: float = 0.0


def is_enabled() -> bool:
    """Check if BLIP model is configured.

    Returns:
        True if models.blip is set to a non-empty string.
    """
    return bool(settings.models.blip)


def _ensure_loaded() -> tuple[object, object]:
    """Lazy-load the BLIP model on first use.

    Uses double-checked locking for thread safety.
    Loads BlipForConditionalGeneration and BlipProcessor
    from the HuggingFace model specified in settings.

    Returns:
        Tuple of (model, processor).

    Raises:
        RuntimeError: If BLIP is disabled or model loading fails.
    """
    global _model, _processor, _loaded, _last_used

    if _loaded and _model is not None:
        _last_used = time.monotonic()
        return _model, _processor

    with _lock:
        if _loaded and _model is not None:
            _last_used = time.monotonic()
            return _model, _processor

        if not is_enabled():
            raise RuntimeError("BLIP is not configured (models.blip is empty)")

        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            model_name = settings.models.blip
            cache_dir = str(settings.model_cache_dir)

            logger.info("Loading BLIP model: %s", model_name)

            _processor = BlipProcessor.from_pretrained(
                model_name, cache_dir=cache_dir
            )
            _model = BlipForConditionalGeneration.from_pretrained(
                model_name, cache_dir=cache_dir
            )
            _model.eval()

            _loaded = True
            _last_used = time.monotonic()
            logger.info("BLIP model loaded successfully")
            return _model, _processor

        except Exception as e:
            logger.error("Failed to load BLIP model: %s", e)
            raise RuntimeError(f"BLIP model load failed: {e}") from e


def unload_model() -> None:
    """Unload the BLIP model to free RAM (~1GB).

    Safe to call even if the model is not loaded.
    """
    global _model, _processor, _loaded

    with _lock:
        if _model is not None:
            logger.info("Unloading BLIP model to free RAM")
            _model = None
            _processor = None
            _loaded = False

    import gc
    gc.collect()


def check_idle_unload() -> None:
    """Check if the model should be unloaded due to idle timeout.

    Called periodically by the indexer's background task.
    Respects settings.memory.blip_idle_unload (0 = never unload).
    """
    idle_timeout = settings.memory.blip_idle_unload
    if idle_timeout <= 0:
        return

    if not _loaded or _model is None:
        return

    elapsed = time.monotonic() - _last_used
    if elapsed > idle_timeout:
        unload_model()


def generate_caption(image: Image.Image) -> str | None:
    """Generate a caption for a single image.

    Runs BLIP conditional image captioning to produce an English
    text description. This is a synchronous, CPU-bound operation
    and should be called from a thread (e.g. via run_in_executor).

    Args:
        image: PIL Image in RGB mode.

    Returns:
        Caption string, or None if BLIP is disabled or captioning fails.
    """
    if not is_enabled():
        return None

    try:
        import torch

        model, processor = _ensure_loaded()

        inputs = processor(image, return_tensors="pt")

        generate_kwargs: dict = {"max_new_tokens": settings.models.blip_max_tokens}
        if settings.models.blip_num_beams > 1:
            generate_kwargs["num_beams"] = settings.models.blip_num_beams

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generate_kwargs)

        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip() if caption else None

    except Exception as e:
        logger.error("BLIP captioning failed: %s", e)
        return None
