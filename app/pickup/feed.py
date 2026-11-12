"""Assembling the Pickup feed from scored lanes.

Stride scheduling. Each lane emits its candidates at positions spaced by
the reciprocal of its weight, so a lane's share of the output converges
on its share of the total weight, and stays there however deep the feed
is cut. That is what gives the weight floor its meaning: an interest the
viewer has not touched in months holds a predictable fraction rather
than one that thins out with depth.

The convergence is not immediate, and the difference matters. A lane of
weight ``w`` places its first item at key ``1/w``, so every lane has
appeared only once the feed is about ``sum(weights) / min(weight)``
items long. Measured with 24 lanes — 3 at 1.0 and 21 at the 0.25 floor,
the realistic worst case for three channels at ``_K_MAX``:

    depth   lanes seen   floor-lane share   (0.636 expected)
       12        6 / 24              0.250
       24       18 / 24              0.625
       40       24 / 24              0.525
      100       24 / 24              0.630

Below roughly thirty items the feed is granular and favours the heaviest
lanes. Anything showing a short slice of it must therefore sample from a
window past that point rather than take the head — which is what the
endpoint's daily window is sized for, and why that sizing is
load-bearing rather than cosmetic.

Deduplication runs before ranking, and the order matters. A file
reachable from several lanes is credited to the lane where its
similarity is highest and removed from the others; only then is each
lane's surviving list numbered. Numbering first would leave a lane that
lost its third candidate emitting positions 1, 2, 4, 5 — quietly
forfeiting a turn it was owed, and breaking the proportionality this
module exists to keep.

Credit follows similarity, not the ranking score. Freshness belongs to
the candidate, not to the lane's interest in it: a file kept by two lanes
is the same file, and the lane that resembles it more is the one whose
subject it serves, even when a decay has pushed both scores to zero.

Within a lane, survivors are ordered by ranking score (similarity times
freshness), then similarity, then file id. That is the same order
``score_lanes`` uses for its cut, so the feed and the lane agree on what
"best" means.

What this does not do is contain a binge. Watching forty episodes of one
series produces several lanes rather than one, because k-means splits a
dense blob and no clustering parameter recovers the subject boundary —
see the note in ``profile``. The feed reports what the history looks
like; narrowing a subject is left to an explicit control, where the
person supplies the meaning the vectors do not carry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.pickup.profile import Lane


@dataclass(frozen=True)
class FeedItem:
    """One row of the feed, and enough of its provenance to explain it."""

    file_id: str
    rank: int
    cluster_id: str
    channel: str
    similarity: float
    score: float


def interleave(
    lanes: Sequence[Lane],
    scored: Mapping[str, Sequence[tuple[str, float, float]]],
    *,
    depth: int,
) -> list[FeedItem]:
    """Weave each lane's candidates into one ordered feed.

    Args:
        lanes: The viewer's interests, with normalized weights.
        scored: Per ``cluster_id``, its candidates as
            ``(file_id, similarity, ranking_score)``. Order is not
            relied upon; this ranks them itself.
        depth: Maximum items to return.

    Returns:
        Feed items in order, ``rank`` counting from 1 within each lane.
        ``score`` is the ranking score and ``similarity`` the lane
        similarity behind it.
    """
    if depth <= 0 or not lanes:
        return []

    by_id = {lane.cluster_id: lane for lane in lanes}

    # Credit each file to the lane where its similarity is highest. Ties
    # fall to the lane listed first, which ``build_lanes`` orders by
    # weight, so the decision is stable across runs.
    owner: dict[str, tuple[str, float, float]] = {}
    for lane in lanes:
        for file_id, similarity, ranking in scored.get(lane.cluster_id, ()):
            held = owner.get(file_id)
            if held is None or similarity > held[1]:
                owner[file_id] = (lane.cluster_id, similarity, ranking)

    survivors: dict[str, list[tuple[str, float, float]]] = {
        lane.cluster_id: [] for lane in lanes
    }
    for file_id, (cluster_id, similarity, ranking) in owner.items():
        survivors[cluster_id].append((file_id, similarity, ranking))

    keyed: list[tuple[float, str, int, str, float, float]] = []
    for cluster_id, candidates in survivors.items():
        weight = by_id[cluster_id].weight
        if weight <= 0.0:
            # A lane with no weight has no turn to take. Reaching this
            # would mean the profile emitted a lane it had already
            # decided was worthless, but dividing by it would be worse.
            continue
        # Ranked after deduplication, so ``rank`` counts only what will
        # actually be emitted.
        candidates.sort(key=lambda row: (-row[2], -row[1], row[0]))
        for rank, (file_id, similarity, score) in enumerate(candidates, start=1):
            keyed.append(
                (rank / weight, cluster_id, rank, file_id, similarity, score)
            )

    # ``cluster_id`` and ``rank`` break ties deterministically; two lanes
    # of equal weight would otherwise alternate in dictionary order.
    keyed.sort(key=lambda row: (row[0], row[1], row[2]))

    return [
        FeedItem(
            file_id=file_id,
            rank=rank,
            cluster_id=cluster_id,
            channel=by_id[cluster_id].channel,
            similarity=similarity,
            score=score,
        )
        for _, cluster_id, rank, file_id, similarity, score in keyed[:depth]
    ]
