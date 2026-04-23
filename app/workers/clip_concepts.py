"""CLIP zero-shot concept scoring for auto-tag candidate generation.

Loads a curated concept vocabulary (clip_concepts.json) plus any
user-defined tags fetched from the Litloft DB, encodes each concept
into a CLIP text embedding once (cached at module level), and scores
file CLIP embeddings against the concept set via cosine similarity.

For videos with multiple frame embeddings, aggregates by counting the
frames in which each concept scores above the threshold, then ranking
concepts by (frequency * max_score). This correctly handles videos
that cover several distinct concepts in different scenes.

The concept vocabulary is built lazily on first call and invalidated
automatically if the CLIP model changes.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np
from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_litloft_db, get_search_db
from app.models import Embedding
from app.workers.clip import embed_text_clip

logger = logging.getLogger(__name__)

_CONCEPTS_JSON = Path(__file__).parent.parent / "data" / "clip_concepts.json"

# Module-level caches, guarded by _lock.
_lock = threading.Lock()
_cached_concepts: dict[str, np.ndarray] | None = None
_cached_model_key: str | None = None


def load_preset_concepts(path: Path | None = None) -> list[str]:
    """Load the curated concept vocabulary from clip_concepts.json.

    Keys starting with underscore (e.g. ``_comment``) are ignored so
    the JSON can carry metadata without polluting the vocabulary.

    Args:
        path: Optional override for the JSON file path.

    Returns:
        Deduplicated list of concept strings.
    """
    target = path or _CONCEPTS_JSON
    if not target.exists():
        logger.warning("Concept JSON not found at %s", target)
        return []

    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to load concept JSON: %s", e)
        return []

    concepts: list[str] = []
    seen: set[str] = set()
    for key, values in data.items():
        if key.startswith("_"):
            continue
        if not isinstance(values, list):
            continue
        for v in values:
            if not isinstance(v, str):
                continue
            v = v.strip()
            if not v or v in seen:
                continue
            seen.add(v)
            concepts.append(v)
    return concepts


def load_user_tags() -> list[str]:
    """Fetch distinct tag names from the Litloft DB.

    Best-effort: returns an empty list if the DB is unavailable (e.g.
    under tests) rather than propagating the error. User tags enrich
    the concept pool so the model can suggest vocabulary the user
    already uses.

    Returns:
        List of tag name strings, deduplicated by the DB query.
    """
    try:
        with get_litloft_db() as session:
            rows = session.execute(
                sql_text("SELECT DISTINCT name FROM tags ORDER BY name")
            ).fetchall()
            return [row[0] for row in rows if row[0]]
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not load user tags for concepts: %s", e)
        return []


def build_concept_pool(
    *,
    preset_path: Path | None = None,
    include_user_tags: bool = True,
) -> list[str]:
    """Merge preset concepts with user tags into the final vocabulary.

    Args:
        preset_path: Optional override for the preset JSON path.
        include_user_tags: When True, fetches tags from Litloft DB.

    Returns:
        Merged, deduplicated list of concept strings.
    """
    preset = load_preset_concepts(preset_path)
    if not include_user_tags:
        return preset

    user_tags = load_user_tags()
    seen = {c for c in preset}
    merged = list(preset)
    for tag in user_tags:
        if tag not in seen:
            seen.add(tag)
            merged.append(tag)
    return merged


def _encode_concepts(concepts: list[str]) -> dict[str, np.ndarray]:
    """Encode each concept string via CLIP text encoder.

    Failures on individual concepts are logged and skipped rather than
    aborting the whole batch — a tokenizer issue with one exotic term
    shouldn't wipe out the vocabulary for an entire session.
    """
    out: dict[str, np.ndarray] = {}
    for concept in concepts:
        try:
            vec = embed_text_clip(concept)
            out[concept] = vec
        except Exception as e:
            logger.warning("Failed to encode concept %r: %s", concept, e)
    return out


def get_concept_embeddings(
    *,
    preset_path: Path | None = None,
    include_user_tags: bool = True,
    force_reload: bool = False,
) -> dict[str, np.ndarray]:
    """Return the concept → embedding map, computing on first call.

    The cache is keyed by CLIP model name so switching models
    automatically triggers recomputation. Thread-safe via the module
    lock; concurrent first-calls will cooperate rather than double-
    compute.

    Args:
        preset_path: Optional override (mostly for tests).
        include_user_tags: Include Litloft user tags in the pool.
        force_reload: Discard cache and rebuild from scratch.

    Returns:
        Map of concept string → normalized CLIP text embedding.
    """
    global _cached_concepts, _cached_model_key

    model_key = settings.models.clip
    with _lock:
        cache_valid = (
            not force_reload
            and _cached_concepts is not None
            and _cached_model_key == model_key
        )
        if cache_valid:
            return _cached_concepts  # type: ignore[return-value]

        concepts = build_concept_pool(
            preset_path=preset_path,
            include_user_tags=include_user_tags,
        )
        if not concepts:
            _cached_concepts = {}
            _cached_model_key = model_key
            return {}

        logger.info(
            "Encoding %d concepts with CLIP model %s", len(concepts), model_key
        )
        _cached_concepts = _encode_concepts(concepts)
        _cached_model_key = model_key
        return _cached_concepts


def _load_file_clip_vectors(file_id: str) -> list[np.ndarray]:
    """Load all CLIP embeddings stored for a file from vec_clip."""
    with get_search_db() as session:
        embeddings = (
            session.query(Embedding.id)
            .filter(
                Embedding.file_id == file_id,
                Embedding.embedding_type == "clip",
            )
            .all()
        )
        embedding_ids = [e.id for e in embeddings]

    if not embedding_ids:
        return []

    from app.database import get_search_engine

    vectors: list[np.ndarray] = []
    with get_search_engine().connect() as conn:
        for eid in embedding_ids:
            row = conn.execute(
                sql_text("SELECT vector FROM vec_clip WHERE embedding_id = :eid"),
                {"eid": eid},
            ).fetchone()
            if row and row[0]:
                vec = np.frombuffer(row[0], dtype=np.float32)
                vectors.append(vec)
    return vectors


def score_vectors_against_concepts(
    vectors: list[np.ndarray],
    concept_embeddings: dict[str, np.ndarray],
    *,
    threshold: float = 0.25,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Score one or more image/frame vectors against the concept set.

    Vectors are assumed to already be L2-normalized (CLIP encoder
    output is), so the dot product equals cosine similarity.

    Aggregation rule (case A from the spec): for each frame, collect
    concepts scoring >= threshold; then rank concepts by
    (frequency * max_score) across frames. Single-image files degrade
    naturally — one frame, frequency=1, score = max_score.

    Args:
        vectors: List of per-frame (or per-image) normalized vectors.
        concept_embeddings: Map from concept string to its embedding.
        threshold: Minimum cosine similarity to consider a match.
        top_k: Max number of concepts to return.

    Returns:
        Sorted list of (concept, aggregated_score) tuples, highest
        first. ``aggregated_score`` is the max cosine similarity the
        concept achieved across frames (useful for downstream filters),
        not the frequency product.
    """
    if not vectors or not concept_embeddings:
        return []

    concept_names = list(concept_embeddings.keys())
    concept_matrix = np.stack([concept_embeddings[c] for c in concept_names])

    # Per-concept aggregates
    frame_count: dict[str, int] = {}
    max_score: dict[str, float] = {}

    for vec in vectors:
        # Cosine similarity = dot product for normalized vectors
        sims = concept_matrix @ vec
        for idx, score in enumerate(sims):
            s = float(score)
            if s < threshold:
                continue
            name = concept_names[idx]
            frame_count[name] = frame_count.get(name, 0) + 1
            if s > max_score.get(name, -1.0):
                max_score[name] = s

    if not frame_count:
        return []

    # Rank by frequency * max_score (favors concepts that recur AND match well).
    ranked = sorted(
        frame_count.keys(),
        key=lambda c: frame_count[c] * max_score[c],
        reverse=True,
    )[:top_k]
    return [(c, max_score[c]) for c in ranked]


def score_file_concepts(
    file_id: str,
    concept_embeddings: dict[str, np.ndarray] | None = None,
    *,
    threshold: float = 0.25,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Score a file's CLIP embeddings against the concept vocabulary.

    Convenience wrapper that loads the file's vectors from vec_clip
    and uses the cached concept embeddings. Returns empty list if the
    file has no CLIP embeddings (e.g. it's a document) or no concepts
    pass the threshold.

    Args:
        file_id: The file ID to score.
        concept_embeddings: Optional injection for tests; when None,
            uses the module-level cache.
        threshold: Minimum cosine similarity to consider a match.
        top_k: Max number of concepts to return.

    Returns:
        Sorted list of (concept, max_score) tuples.
    """
    vectors = _load_file_clip_vectors(file_id)
    if not vectors:
        return []

    if concept_embeddings is None:
        concept_embeddings = get_concept_embeddings()
    if not concept_embeddings:
        return []

    return score_vectors_against_concepts(
        vectors,
        concept_embeddings,
        threshold=threshold,
        top_k=top_k,
    )


def reset_cache() -> None:
    """Clear the module-level concept cache (primarily for tests)."""
    global _cached_concepts, _cached_model_key
    with _lock:
        _cached_concepts = None
        _cached_model_key = None
