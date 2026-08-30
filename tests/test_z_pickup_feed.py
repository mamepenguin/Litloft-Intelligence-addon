"""Assembling the feed: dedupe, rank, interleave.

The guarantee this file exists to hold is that a lane's share of the
feed is proportional to its weight *at every prefix length*. Cut the
feed at ten items or at three hundred and the proportions are the same.
That is what makes the weight floor mean something: an interest the
viewer has not touched in months keeps a predictable fraction of the
output rather than a fraction that decays with depth.

What this file does NOT claim is that a binge occupies one lane. It
does not — k-means splits a dense blob, and neither a similarity
threshold nor a silhouette criterion recovers the subject boundary,
because "sub-structure of one interest" and "two interests" differ
semantically and not geometrically. The feed reports what the history
looks like.
"""

from __future__ import annotations

import sys
from collections import Counter
from unittest.mock import MagicMock

import numpy as np
import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.pickup import feed  # noqa: E402
from app.pickup.profile import Lane  # noqa: E402


def _lane(cluster_id: str, weight: float, channel: str = "clip_thumbnail") -> Lane:
    return Lane(
        channel=channel,
        cluster_id=cluster_id,
        centroid=np.zeros(4, dtype=np.float32),
        member_count=1,
        raw_weight=weight,
        weight=weight,
    )


def _share(items, cluster_id: str) -> float:
    if not items:
        return 0.0
    return sum(1 for i in items if i.cluster_id == cluster_id) / len(items)


# ---------------------------------------------------------------------------
# The proportionality guarantee
# ---------------------------------------------------------------------------


def test_a_lanes_share_holds_at_every_depth():
    lanes = [_lane("loud", 1.0), _lane("quiet", 0.25)]
    scored = {
        "loud": [(f"L{i}", 0.9 - i * 0.001) for i in range(400)],
        "quiet": [(f"Q{i}", 0.9 - i * 0.001) for i in range(400)],
    }

    items = feed.interleave(lanes, scored, depth=300)

    expected = 1.0 / 1.25
    for depth in (10, 50, 150, 300):
        got = _share(items[:depth], "loud")
        assert got == pytest.approx(expected, abs=0.06), f"at depth {depth}"


def test_the_quietest_lane_is_never_starved():
    lanes = [_lane(f"loud{i}", 1.0) for i in range(5)] + [_lane("quiet", 0.25)]
    scored = {
        lane.cluster_id: [(f"{lane.cluster_id}-{i}", 0.9) for i in range(200)]
        for lane in lanes
    }

    items = feed.interleave(lanes, scored, depth=300)

    assert _share(items, "quiet") > 0.03
    # 0.25 / (5 + 0.25)
    assert _share(items, "quiet") == pytest.approx(0.25 / 5.25, abs=0.03)


def test_a_heavier_lane_leads():
    lanes = [_lane("light", 0.25), _lane("heavy", 1.0)]
    scored = {
        "light": [("l0", 0.99)],
        "heavy": [("h0", 0.50)],
    }

    items = feed.interleave(lanes, scored, depth=10)

    assert items[0].file_id == "h0"


# ---------------------------------------------------------------------------
# Dedupe happens before ranking
# ---------------------------------------------------------------------------


def test_a_file_reachable_from_two_lanes_appears_once():
    lanes = [_lane("a", 1.0), _lane("b", 1.0)]
    scored = {
        "a": [("shared", 0.9), ("a1", 0.8)],
        "b": [("shared", 0.7), ("b1", 0.6)],
    }

    items = feed.interleave(lanes, scored, depth=10)

    assert [i.file_id for i in items].count("shared") == 1
    assert next(i for i in items if i.file_id == "shared").cluster_id == "a"


def test_the_losing_lane_forfeits_no_turn():
    """Ranking must follow deduplication, not precede it.

    If ``j`` were fixed before the shared file was removed, lane b would
    emit keys 1/w, 3/w, 4/w — silently skipping a turn it was owed and
    breaking the proportionality above.
    """
    lanes = [_lane("a", 1.0), _lane("b", 1.0)]
    scored = {
        "a": [("shared", 0.99), ("a1", 0.9), ("a2", 0.8), ("a3", 0.7)],
        "b": [("b1", 0.95), ("shared", 0.5), ("b2", 0.85), ("b3", 0.75)],
    }

    items = feed.interleave(lanes, scored, depth=8)
    b_ranks = [i.rank for i in items if i.cluster_id == "b"]

    assert [i.file_id for i in items if i.cluster_id == "b"] == ["b1", "b2", "b3"]
    assert b_ranks == sorted(b_ranks)
    assert len(set(b_ranks)) == len(b_ranks)


def test_a_file_is_credited_to_the_lane_that_scores_it_highest():
    lanes = [_lane("weak", 1.0), _lane("strong", 1.0)]
    scored = {"weak": [("f", 0.5)], "strong": [("f", 0.95)]}

    items = feed.interleave(lanes, scored, depth=5)

    assert [(i.file_id, i.cluster_id) for i in items] == [("f", "strong")]


def test_channels_share_one_sequence():
    lanes = [
        _lane("visual", 1.0, channel="clip_thumbnail"),
        _lane("textual", 1.0, channel="text_content"),
    ]
    scored = {
        "visual": [(f"v{i}", 0.9) for i in range(50)],
        "textual": [(f"t{i}", 0.9) for i in range(50)],
    }

    items = feed.interleave(lanes, scored, depth=40)
    kinds = Counter(i.channel for i in items)

    assert kinds["clip_thumbnail"] == pytest.approx(20, abs=3)
    assert kinds["text_content"] == pytest.approx(20, abs=3)


# ---------------------------------------------------------------------------
# Ordering, ranks and depth
# ---------------------------------------------------------------------------


def test_ranks_are_dense_and_start_at_one():
    lanes = [_lane("a", 1.0)]
    scored = {"a": [(f"a{i}", 0.9 - i * 0.01) for i in range(5)]}

    items = feed.interleave(lanes, scored, depth=10)

    assert [i.rank for i in items] == [1, 2, 3, 4, 5]


def test_within_a_lane_the_better_score_comes_first():
    lanes = [_lane("a", 1.0)]
    scored = {"a": [("mid", 0.7), ("best", 0.95), ("worst", 0.5)]}

    items = feed.interleave(lanes, scored, depth=10)

    assert [i.file_id for i in items] == ["best", "mid", "worst"]


def test_depth_truncates():
    lanes = [_lane("a", 1.0)]
    scored = {"a": [(f"a{i}", 0.9) for i in range(500)]}

    assert len(feed.interleave(lanes, scored, depth=300)) == 300


def test_the_result_is_deterministic():
    lanes = [_lane("a", 1.0), _lane("b", 1.0), _lane("c", 0.5)]
    scored = {
        "a": [(f"a{i}", 0.9) for i in range(20)],
        "b": [(f"b{i}", 0.9) for i in range(20)],
        "c": [(f"c{i}", 0.9) for i in range(20)],
    }

    first = feed.interleave(lanes, scored, depth=40)
    second = feed.interleave(lanes, scored, depth=40)

    assert [(i.file_id, i.rank) for i in first] == \
           [(i.file_id, i.rank) for i in second]


# ---------------------------------------------------------------------------
# Empty and degenerate inputs
# ---------------------------------------------------------------------------


def test_no_lanes_yields_no_feed():
    assert feed.interleave([], {}, depth=10) == []


def test_a_lane_with_no_candidates_is_skipped():
    lanes = [_lane("empty", 1.0), _lane("full", 0.5)]
    scored = {"empty": [], "full": [("f0", 0.9)]}

    items = feed.interleave(lanes, scored, depth=10)

    assert [i.file_id for i in items] == ["f0"]


def test_a_zero_weight_lane_does_not_divide_by_zero():
    lanes = [_lane("dead", 0.0), _lane("live", 1.0)]
    scored = {"dead": [("d0", 0.9)], "live": [("l0", 0.9)]}

    items = feed.interleave(lanes, scored, depth=10)

    assert "l0" in [i.file_id for i in items]


def test_depth_of_zero_yields_nothing():
    lanes = [_lane("a", 1.0)]
    assert feed.interleave(lanes, {"a": [("x", 0.9)]}, depth=0) == []
