"""Assembling the Pickup feed from scored lanes.

Stride scheduling. Each lane emits its candidates at positions spaced by
the reciprocal of its weight, so a lane's share of the output is
proportional to its weight *at every prefix length* — cut the feed at
ten items or at three hundred and the proportions match. That is what
gives the weight floor its meaning: an interest the viewer has not
touched in months holds a predictable fraction of the feed rather than
one that thins out with depth.

Deduplication runs before ranking, and the order matters. A file
reachable from several lanes is credited to the lane that scores it
highest and removed from the others; only then is each lane's surviving
list numbered. Numbering first would leave a lane that lost its third
candidate emitting positions 1, 2, 4, 5 — quietly forfeiting a turn it
was owed, and breaking the proportionality this module exists to keep.

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
    score: float


def interleave(
    lanes: Sequence[Lane],
    scored: Mapping[str, Sequence[tuple[str, float]]],
    *,
    depth: int,
) -> list[FeedItem]:
    """Weave each lane's candidates into one ordered feed.

    Args:
        lanes: The viewer's interests, with normalized weights.
        scored: Per ``cluster_id``, its candidates as (file_id, score).
            Order is not relied upon; this ranks them itself.
        depth: Maximum items to return.

    Returns:
        Feed items in order, ``rank`` counting from 1 within each lane.
    """
    if depth <= 0 or not lanes:
        return []

    by_id = {lane.cluster_id: lane for lane in lanes}

    # Credit each file to the lane that scores it highest. Ties fall to
    # the lane listed first, which ``build_lanes`` orders by weight, so
    # the decision is stable across runs.
    owner: dict[str, tuple[str, float]] = {}
    for lane in lanes:
        for file_id, score in scored.get(lane.cluster_id, ()):
            held = owner.get(file_id)
            if held is None or score > held[1]:
                owner[file_id] = (lane.cluster_id, score)

    survivors: dict[str, list[tuple[str, float]]] = {
        lane.cluster_id: [] for lane in lanes
    }
    for file_id, (cluster_id, score) in owner.items():
        survivors[cluster_id].append((file_id, score))

    keyed: list[tuple[float, str, int, str, float]] = []
    for cluster_id, candidates in survivors.items():
        weight = by_id[cluster_id].weight
        if weight <= 0.0:
            # A lane with no weight has no turn to take. Reaching this
            # would mean the profile emitted a lane it had already
            # decided was worthless, but dividing by it would be worse.
            continue
        # Ranked after deduplication, so ``rank`` counts only what will
        # actually be emitted.
        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        for rank, (file_id, score) in enumerate(candidates, start=1):
            keyed.append((rank / weight, cluster_id, rank, file_id, score))

    # ``cluster_id`` and ``rank`` break ties deterministically; two lanes
    # of equal weight would otherwise alternate in dictionary order.
    keyed.sort(key=lambda row: (row[0], row[1], row[2]))

    return [
        FeedItem(
            file_id=file_id,
            rank=rank,
            cluster_id=cluster_id,
            channel=by_id[cluster_id].channel,
            score=score,
        )
        for _, cluster_id, rank, file_id, score in keyed[:depth]
    ]
