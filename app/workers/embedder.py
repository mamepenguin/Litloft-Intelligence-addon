"""Text embedding worker using multilingual-e5-small (ONNX).

Provides shared text-to-vector embedding functionality used by
both metadata indexing and whisper transcript indexing.

The model is lazy-loaded on first use and kept in memory.
"""

import logging
import threading
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Model state (protected by lock for thread safety)
_lock = threading.Lock()
_tokenizer: object | None = None
_session: object | None = None
_loaded = False

# multilingual-e5-small produces 384-dimensional vectors
EMBEDDING_DIM = 384


def _ensure_loaded() -> tuple[object, object]:
    """Lazy-load the text embedding model on first use.

    Returns:
        Tuple of (tokenizer, onnx_session).

    Raises:
        RuntimeError: If model loading fails.
    """
    global _tokenizer, _session, _loaded

    if _loaded and _tokenizer is not None and _session is not None:
        return _tokenizer, _session

    with _lock:
        # Double-check after acquiring lock
        if _loaded and _tokenizer is not None and _session is not None:
            return _tokenizer, _session

        try:
            from transformers import AutoTokenizer
            import onnxruntime as ort

            model_name = settings.models.text_embedding
            cache_dir = str(settings.model_cache_dir)

            logger.info("Loading text embedding model: %s", model_name)

            _tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir
            )

            model_path = _find_or_export_onnx(model_name, cache_dir)

            _session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )

            _loaded = True
            logger.info("Text embedding model loaded successfully")
            return _tokenizer, _session

        except Exception as e:
            logger.error("Failed to load text embedding model: %s", e)
            raise RuntimeError(f"Text embedding model load failed: {e}") from e


def _find_or_export_onnx(model_name: str, cache_dir: str) -> Path:
    """Find existing ONNX model or export from HuggingFace.

    Args:
        model_name: HuggingFace model identifier.
        cache_dir: Directory for caching models.

    Returns:
        Path to the ONNX model file.
    """
    onnx_dir = Path(cache_dir) / "text_embedding_onnx"
    onnx_path = onnx_dir / "model.onnx"

    if onnx_path.exists():
        return onnx_path

    logger.info("Exporting text embedding model to ONNX: %s", model_name)

    from optimum.onnxruntime import ORTModelForFeatureExtraction

    model = ORTModelForFeatureExtraction.from_pretrained(
        model_name,
        export=True,
        cache_dir=cache_dir,
    )
    model.save_pretrained(str(onnx_dir))

    return onnx_path


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a batch of texts.

    Uses the multilingual-e5-small convention of prefixing
    with "query: " or "passage: " for better retrieval.

    Args:
        texts: List of text strings to embed.
            Should already include "query: " or "passage: " prefix.

    Returns:
        numpy array of shape (len(texts), EMBEDDING_DIM).
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    tokenizer, session = _ensure_loaded()

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )

    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)

    feeds = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    # Add token_type_ids if the model expects it
    input_names = {inp.name for inp in session.get_inputs()}
    if "token_type_ids" in input_names:
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
        feeds["token_type_ids"] = token_type_ids

    outputs = session.run(None, feeds)

    # Mean pooling over token embeddings, masked by attention
    token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)
    mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    pooled = summed / counts

    # L2 normalize
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    normalized = pooled / norms

    return normalized.astype(np.float32)


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
