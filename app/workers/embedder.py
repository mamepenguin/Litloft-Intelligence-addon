"""Text embedding worker using multilingual-e5-small via sentence-transformers.

Provides shared text-to-vector embedding functionality used by
both metadata indexing and whisper transcript indexing.

The model is lazy-loaded on first use and kept in memory.
Uses sentence-transformers which handles ONNX export and inference internally.
"""

import logging
import threading

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Model state (protected by lock for thread safety)
_lock = threading.Lock()
_model: object | None = None
_loaded = False

# multilingual-e5-small produces 384-dimensional vectors
EMBEDDING_DIM = 384


def _ensure_loaded() -> object:
    """Lazy-load the text embedding model on first use.

    Returns:
        SentenceTransformer model instance.

    Raises:
        RuntimeError: If model loading fails.
    """
    global _model, _loaded

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

            _loaded = True
            logger.info("Text embedding model loaded successfully")
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


def embed_query(query: str) -> np.ndarray:
    """Embed a search query text.

    Adds the "query: " prefix as per multilingual-e5 convention.

    Args:
        query: The search query string.

    Returns:
        numpy array of shape (EMBEDDING_DIM,).
    """
    prefixed = f"query: {query}"
    result = embed_texts([prefixed])
    return result[0]


def embed_passages(passages: list[str]) -> np.ndarray:
    """Embed document passages for indexing.

    Adds the "passage: " prefix as per multilingual-e5 convention.

    Args:
        passages: List of document text passages.

    Returns:
        numpy array of shape (len(passages), EMBEDDING_DIM).
    """
    prefixed = [f"passage: {p}" for p in passages]
    return embed_texts(prefixed)
