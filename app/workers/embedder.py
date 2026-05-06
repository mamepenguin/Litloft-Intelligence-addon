"""Text embedding worker using sentence-transformers.

Provides shared text-to-vector embedding functionality used by
both metadata indexing and whisper transcript indexing.

The model is lazy-loaded on first use and kept in memory.
Supports multilingual-e5 and Ruri model families with automatic
prefix detection.
"""

import logging
import threading
from functools import lru_cache

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Model state (protected by lock for thread safety)
_lock = threading.Lock()
_model: object | None = None
_loaded = False

# Default dimension (overwritten after model loads with actual value)
EMBEDDING_DIM = 384

# Model name → embedding dimension mapping
_MODEL_DIMS: dict[str, int] = {
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    "cl-nagoya/ruri-v3-30m": 256,
    "cl-nagoya/ruri-v3-130m": 768,
    "cl-nagoya/ruri-v3-310m": 1024,
}

# Model family → (query_prefix, passage_prefix)
_MODEL_PREFIXES: dict[str, tuple[str, str]] = {
    "e5": ("query: ", "passage: "),
    "ruri": ("検索クエリ: ", "検索文書: "),
}


def _detect_prefix_family(model_name: str) -> str:
    """Detect the prefix family from the model name.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Prefix family key ("e5", "ruri", etc.).
    """
    lower = model_name.lower()
    if "ruri" in lower:
        return "ruri"
    return "e5"


def _get_prefixes() -> tuple[str, str]:
    """Get the (query_prefix, passage_prefix) for the configured model.

    Priority: explicit config values > auto-detect from model name.
    If both text_query_prefix and text_passage_prefix are set in config,
    those are used directly. Otherwise, falls back to model name detection.

    Returns:
        Tuple of (query_prefix, passage_prefix) strings.
    """
    models_config = settings.models
    if models_config.text_query_prefix or models_config.text_passage_prefix:
        return (models_config.text_query_prefix, models_config.text_passage_prefix)
    family = _detect_prefix_family(models_config.text_embedding)
    return _MODEL_PREFIXES.get(family, _MODEL_PREFIXES["e5"])


def _ensure_loaded() -> object:
    """Lazy-load the text embedding model on first use.

    Returns:
        SentenceTransformer model instance.

    Raises:
        RuntimeError: If model loading fails.
    """
    global _model, _loaded, EMBEDDING_DIM

    if _loaded and _model is not None:
        return _model

    with _lock:
        if _loaded and _model is not None:
            return _model

        try:
            from sentence_transformers import SentenceTransformer

            model_name = settings.models.text_embedding
            cache_dir = str(settings.model_cache_dir)

            logger.info("Loading text embedding model: %s", model_name)

            _model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                device="cpu",
            )

            # Update dimension from actual model
            actual_dim = _model.get_sentence_embedding_dimension()
            if actual_dim is not None:
                EMBEDDING_DIM = actual_dim
            elif model_name in _MODEL_DIMS:
                EMBEDDING_DIM = _MODEL_DIMS[model_name]

            _loaded = True
            logger.info(
                "Text embedding model loaded: %s (dim=%d)", model_name, EMBEDDING_DIM
            )
            return _model

        except Exception as e:
            logger.error("Failed to load text embedding model: %s", e)
            raise RuntimeError(f"Text embedding model load failed: {e}") from e


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a batch of texts.

    Args:
        texts: List of text strings to embed.
            Should already include "query: " or "passage: " prefix.

    Returns:
        numpy array of shape (len(texts), EMBEDDING_DIM).
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = _ensure_loaded()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


@lru_cache(maxsize=512)
def embed_query(query: str) -> np.ndarray:
    """Embed a search query text.

    Adds the appropriate query prefix for the configured model.
    Results are cached (LRU, 512 entries) — the returned array is
    read-only to prevent accidental cache corruption.

    Args:
        query: The search query string.

    Returns:
        numpy array of shape (EMBEDDING_DIM,).
    """
    query_prefix, _ = _get_prefixes()
    prefixed = f"{query_prefix}{query}"
    arr = embed_texts([prefixed])[0]
    arr.flags.writeable = False
    return arr


def embed_passages(passages: list[str]) -> np.ndarray:
    """Embed document passages for indexing.

    Adds the appropriate passage prefix for the configured model.

    Args:
        passages: List of document text passages.

    Returns:
        numpy array of shape (len(passages), EMBEDDING_DIM).
    """
    _, passage_prefix = _get_prefixes()
    prefixed = [f"{passage_prefix}{p}" for p in passages]
    return embed_texts(prefixed)
