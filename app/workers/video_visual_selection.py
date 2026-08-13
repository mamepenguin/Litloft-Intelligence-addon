"""Deterministic candidate-scene selection for the video visual index.

Implements the "Video Visual Index" design doc §6: a duration-aware,
bounded scene budget, then a deterministic selection over the existing
scene-CLIP candidate pool via bucketed temporal coverage, near-duplicate
removal, and greedy farthest-point fill. Pure functions over plain
:class:`Candidate` values so the algorithm is unit-testable without a
database or a loaded CLIP model.

The candidate pool itself (``Embedding`` rows with
``embedding_type="clip"``, vectors from ``vec_clip``) is read by the
caller (the video-visual worker); this module never touches the DB.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

# Pilot defaults (design doc §6.2 / §16) — not exposed as operator
# settings. Change only after the pilot review shows a systematic
# coverage or cost problem.
MIN_TARGET = 6
MAX_TARGET = 24
DURATION_SECONDS_PER_SCENE = 300  # ceil(duration / this) drives the target
UNKNOWN_DURATION_TARGET = 12
NEAR_DUPLICATE_COSINE_THRESHOLD = 0.97


@dataclass(frozen=True)
class Candidate:
    """One scene-CLIP embedding row eligible for visual-index selection."""

    embedding_id: str
    timestamp: float
    vector: np.ndarray


def compute_target_count(duration_seconds: float | None, candidate_count: int) -> int:
    """Duration-aware, bounded scene budget (design doc §6.2).

    ``target = clamp(MIN_TARGET, ceil(duration_seconds / 300), MAX_TARGET)``.
    Unknown, non-finite, or non-positive duration uses
    ``UNKNOWN_DURATION_TARGET``. Never exceeds ``candidate_count``.
    """
    if candidate_count <= 0:
        return 0
    if (
        duration_seconds is None
        or not math.isfinite(duration_seconds)
        or duration_seconds <= 0
    ):
        base = UNKNOWN_DURATION_TARGET
    else:
        base = math.ceil(duration_seconds / DURATION_SECONDS_PER_SCENE)
    target = max(MIN_TARGET, min(MAX_TARGET, base))
    return min(target, candidate_count)


def _filter_valid(candidates: list[Candidate]) -> list[Candidate]:
    """Drop non-finite/negative timestamps and missing vectors (§6.3-1)."""
    out: list[Candidate] = []
    for c in candidates:
        if c.vector is None:
            continue
        if not math.isfinite(c.timestamp) or c.timestamp < 0:
            continue
        out.append(c)
    return out


def compute_candidate_fingerprint(candidates: list[Candidate]) -> str:
    """Hash the ordered candidate embedding IDs + timestamps (design doc §5.1).

    Order-sensitive by design: rebuilding scene CLIP changes candidate
    content/order even when the id set happens to coincide, and the
    fingerprint must catch that. Callers pass candidates in a stable
    order (the CLIP worker's storage order, i.e. ``timestamp_start``).
    """
    h = hashlib.sha256()
    for c in candidates:
        h.update(c.embedding_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(f"{c.timestamp:.6f}".encode("ascii"))
        h.update(b"\x01")
    return h.hexdigest()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _bucket_coverage(
    candidates: list[Candidate], bucket_count: int
) -> list[Candidate]:
    """Pick one candidate nearest each non-empty time-bucket center (§6.3-2/3).

    Partitions ``[min(timestamp), max(timestamp)]`` into ``bucket_count``
    equal-width half-open buckets (the final bucket is closed on the
    right so the latest timestamp always lands somewhere), then keeps
    the candidate nearest each non-empty bucket's center.
    """
    if bucket_count <= 0 or not candidates:
        return []
    timestamps = [c.timestamp for c in candidates]
    lo, hi = min(timestamps), max(timestamps)
    span = hi - lo
    if span <= 0:
        # All candidates share one timestamp (or a single candidate) —
        # one bucket, one representative.
        return [min(candidates, key=lambda c: (c.timestamp, c.embedding_id))]

    width = span / bucket_count
    selected: list[Candidate] = []
    for i in range(bucket_count):
        b_lo = lo + i * width
        b_hi = hi if i == bucket_count - 1 else lo + (i + 1) * width
        center = (b_lo + b_hi) / 2
        in_bucket = [
            c for c in candidates
            if (b_lo <= c.timestamp < b_hi)
            or (i == bucket_count - 1 and c.timestamp == b_hi)
        ]
        if not in_bucket:
            continue
        best = min(
            in_bucket,
            key=lambda c: (abs(c.timestamp - center), c.timestamp, c.embedding_id),
        )
        selected.append(best)
    return selected


def _dedupe_near_duplicates(selected: list[Candidate]) -> list[Candidate]:
    """Drop near-duplicate scenes by cosine similarity (§6.3-4).

    Deterministic single pass in timestamp order: a candidate is
    dropped when it is a near-duplicate of any candidate already kept.
    """
    ordered = sorted(selected, key=lambda c: (c.timestamp, c.embedding_id))
    kept: list[Candidate] = []
    for c in ordered:
        if any(
            _cosine_similarity(c.vector, k.vector) >= NEAR_DUPLICATE_COSINE_THRESHOLD
            for k in kept
        ):
            continue
        kept.append(c)
    return kept


def _farthest_point_fill(
    pool: list[Candidate],
    selected: list[Candidate],
    budget: int,
) -> list[Candidate]:
    """Greedy farthest-point fill over CLIP distance (§6.3-5).

    Repeatedly adds the remaining candidate whose minimum cosine
    distance to the current selected set is largest. Ties are broken
    by earliest timestamp, then by ``embedding_id`` for full
    determinism given the same candidate set.
    """
    selected_ids = {c.embedding_id for c in selected}
    remaining = [c for c in pool if c.embedding_id not in selected_ids]
    result = list(selected)

    if not result and remaining and budget > 0:
        seed = min(remaining, key=lambda c: (c.timestamp, c.embedding_id))
        result.append(seed)
        remaining = [c for c in remaining if c.embedding_id != seed.embedding_id]

    while remaining and len(result) < budget:
        distances = {
            c.embedding_id: min(
                1.0 - _cosine_similarity(c.vector, s.vector) for s in result
            )
            for c in remaining
        }
        max_dist = max(distances.values())
        tied = [c for c in remaining if distances[c.embedding_id] == max_dist]
        best = min(tied, key=lambda c: (c.timestamp, c.embedding_id))
        result.append(best)
        remaining = [c for c in remaining if c.embedding_id != best.embedding_id]

    return result


def select_candidates(
    candidates: list[Candidate],
    *,
    duration_seconds: float | None,
) -> list[Candidate]:
    """Deterministically select representative scenes (design doc §6.3).

    Returns candidates sorted by timestamp (callers assign dense
    ``ordering`` from the returned list's index). Empty input, or input
    that is entirely filtered out as invalid, returns an empty list.
    """
    valid = _filter_valid(candidates)
    if not valid:
        return []

    target = compute_target_count(duration_seconds, len(valid))
    if target <= 0:
        return []

    bucket_count = min(target, len(valid))
    covered = _bucket_coverage(valid, bucket_count)
    deduped = _dedupe_near_duplicates(covered)
    filled = _farthest_point_fill(valid, deduped, target)

    return sorted(filled, key=lambda c: (c.timestamp, c.embedding_id))


__all__ = [
    "Candidate",
    "MAX_TARGET",
    "MIN_TARGET",
    "NEAR_DUPLICATE_COSINE_THRESHOLD",
    "UNKNOWN_DURATION_TARGET",
    "compute_candidate_fingerprint",
    "compute_target_count",
    "select_candidates",
]
