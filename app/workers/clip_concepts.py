"""CLIP zero-shot concept scoring for auto-tag candidate generation.

Loads a curated concept vocabulary (clip_concepts.json) plus the tags
the user defined *in the drive being tagged*, encodes each concept
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
import time
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
#
# The preset vocabulary carries no user data, so it is encoded once and
# shared. Tag vocabulary is per drive — a drive is a security boundary,
# so another drive's tag names must never enter this drive's candidate
# vocabulary — and is therefore cached under the drive it came from.
_lock = threading.Lock()
_cached_preset: dict[str, np.ndarray] | None = None
_cached_drive_tags: dict[str, dict[str, np.ndarray]] = {}
_cached_model_key: str | None = None
_last_used: float = 0.0


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


def load_user_tags(drive: str) -> list[str]:
    """Fetch one drive's distinct tag names from the Litloft DB.

    Best-effort: returns an empty list if the DB is unavailable (e.g.
    under tests) rather than propagating the error. User tags enrich
    the concept pool so the model can suggest vocabulary the user
    already uses — but only vocabulary from the drive being tagged.

    Args:
        drive: The drive whose tags may enter the vocabulary.

    Returns:
        List of tag name strings, deduplicated by the DB query.
    """
    try:
        with get_litloft_db() as session:
            rows = session.execute(
                sql_text(
                    "SELECT DISTINCT name FROM tags "
                    "WHERE drive = :drive ORDER BY name"
                ),
                {"drive": drive},
            ).fetchall()
            return [row[0] for row in rows if row[0]]
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not load user tags for concepts: %s", e)
        return []


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
    drive: str | None,
    preset_path: Path | None = None,
    include_user_tags: bool = True,
    force_reload: bool = False,
) -> dict[str, np.ndarray]:
    """Return the concept → embedding map for one drive, computing on first call.

    The preset half of the vocabulary is encoded once and reused; the
    tag half is encoded per drive and never crosses between drives.
    Both caches are keyed by CLIP model name so switching models
    automatically triggers recomputation. Thread-safe via the module
    lock; concurrent first-calls will cooperate rather than double-
    compute.

    Args:
        drive: The drive being tagged. ``None`` yields the preset
            vocabulary alone — no drive, no tag vocabulary.
        preset_path: Optional override (mostly for tests).
        include_user_tags: Include the drive's Litloft tags in the pool.
        force_reload: Discard cache and rebuild from scratch.

    Returns:
        Map of concept string → normalized CLIP text embedding.
    """
    global _cached_preset, _cached_drive_tags, _cached_model_key, _last_used

    model_key = settings.models.clip
    with _lock:
        if force_reload or _cached_model_key != model_key:
            _cached_preset = None
            _cached_drive_tags = {}
            _cached_model_key = model_key

        if _cached_preset is None:
            preset_names = load_preset_concepts(preset_path)
            logger.info(
                "Encoding %d preset concepts with CLIP model %s",
                len(preset_names), model_key,
            )
            _cached_preset = _encode_concepts(preset_names)
        preset = _cached_preset

        tags: dict[str, np.ndarray] = {}
        if include_user_tags and drive:
            cached_tags = _cached_drive_tags.get(drive)
            if cached_tags is None:
                # Tags already covered by the preset need no second
                # encoding — the embedding would be identical.
                names = [t for t in load_user_tags(drive) if t not in preset]
                if names:
                    logger.info(
                        "Encoding %d tag concepts for drive %s", len(names), drive
                    )
                cached_tags = _encode_concepts(names) if names else {}
                _cached_drive_tags[drive] = cached_tags
            tags = cached_tags

        _last_used = time.monotonic()
        if not tags:
            return preset
        return {**preset, **tags}


def check_idle_unload() -> None:
    """Discard concept embeddings after an idle period to free RAM.

    Respects settings.memory.clip_concepts_idle_unload (0 = never unload).
    The cache rebuilds automatically on the next auto-tag request.
    """
    global _cached_preset, _cached_drive_tags, _cached_model_key, _last_used

    idle_timeout = settings.memory.clip_concepts_idle_unload
    if idle_timeout <= 0:
        return

    with _lock:
        if _cached_preset is None and not _cached_drive_tags:
            return
        if time.monotonic() - _last_used > idle_timeout:
            logger.info("Releasing CLIP concept embeddings (idle timeout)")
            _cached_preset = None
            _cached_drive_tags = {}
            _cached_model_key = None


def load_file_clip_vectors(file_id: str) -> list[np.ndarray]:
    """Load all CLIP embeddings stored for a file from vec_clip.

    Shared by both concept scoring (this module) and k-NN tag
    recommendation (``app.workers.tag_knn``) so a file's vectors are
    fetched from the DB once per candidate-generation pass rather than
    once per pipeline.
    """
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

    placeholders = ",".join(f":id{i}" for i in range(len(embedding_ids)))
    params = {f"id{i}": eid for i, eid in enumerate(embedding_ids)}
    vectors: list[np.ndarray] = []
    with get_search_engine().connect() as conn:
        rows = conn.execute(
            sql_text(
                f"SELECT vector FROM vec_clip WHERE embedding_id IN ({placeholders})"
            ),
            params,
        ).fetchall()
    for row in rows:
        if row[0]:
            vectors.append(np.frombuffer(row[0], dtype=np.float32))
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
    drive: str | None,
    threshold: float = 0.25,
    top_k: int = 10,
    vectors: list[np.ndarray] | None = None,
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
        drive: The file's drive, deciding whose tag vocabulary joins
            the preset concepts. Ignored when ``concept_embeddings``
            is injected.
        threshold: Minimum cosine similarity to consider a match.
        top_k: Max number of concepts to return.
        vectors: Optional pre-loaded CLIP vectors for this file (see
            ``load_file_clip_vectors``). Pass this when the caller
            already fetched the vectors for another pipeline (e.g.
            k-NN) to avoid a redundant DB round trip; when None, this
            function loads them itself.

    Returns:
        Sorted list of (concept, max_score) tuples.
    """
    if vectors is None:
        vectors = load_file_clip_vectors(file_id)
    if not vectors:
        return []

    if concept_embeddings is None:
        concept_embeddings = get_concept_embeddings(drive=drive)
    if not concept_embeddings:
        return []

    return score_vectors_against_concepts(
        vectors,
        concept_embeddings,
        threshold=threshold,
        top_k=top_k,
    )


def reset_cache() -> None:
    """Clear the module-level concept caches (primarily for tests)."""
    global _cached_preset, _cached_drive_tags, _cached_model_key
    with _lock:
        _cached_preset = None
        _cached_drive_tags = {}
        _cached_model_key = None
