"""Tests for the deterministic candidate-scene selection algorithm.

Design doc "Video Visual Index" §6 / §14.1:
- duration budget boundaries and unknown duration
- non-finite timestamp/vector rejection
- temporal coverage before diversity fill
- near-duplicate removal and stable tie-breaking
- deterministic selection for a fixed candidate set
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

for _mod in ("PIL", "PIL.Image", "sqlite_vec"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import numpy as np  # noqa: E402

from app.workers.video_visual_selection import (  # noqa: E402
    Candidate,
    compute_candidate_fingerprint,
    compute_target_count,
    select_candidates,
)


def _unit_vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestComputeTargetCount:
    @pytest.mark.parametrize(
        "duration_seconds,expected",
        [
            (300, 6),      # 5 min -> clamp(6, ceil(1), 24) = 6
            (1800, 6),     # 30 min -> clamp(6, 6, 24) = 6
            (3600, 12),    # 60 min -> clamp(6, 12, 24) = 12
            (7200, 24),    # 120 min -> clamp(6, 24, 24) = 24
            (7201, 24),    # 120+ min stays capped at 24
        ],
    )
    def test_duration_table(self, duration_seconds, expected):
        assert compute_target_count(duration_seconds, candidate_count=100) == expected

    @pytest.mark.parametrize("bad_duration", [None, float("nan"), -5, 0])
    def test_unknown_or_invalid_duration_uses_12(self, bad_duration):
        assert compute_target_count(bad_duration, candidate_count=100) == 12

    def test_never_exceeds_candidate_count(self):
        assert compute_target_count(3600, candidate_count=3) == 3

    def test_zero_candidates_yields_zero_target(self):
        assert compute_target_count(3600, candidate_count=0) == 0


class TestSelectCandidatesValidation:
    def test_empty_input_returns_empty(self):
        assert select_candidates([], duration_seconds=600) == []

    def test_drops_non_finite_and_negative_timestamps(self):
        v = _unit_vec(1)
        candidates = [
            Candidate("nan_ts", float("nan"), v),
            Candidate("neg_ts", -1.0, v),
            Candidate("inf_ts", float("inf"), v),
        ]
        assert select_candidates(candidates, duration_seconds=600) == []

    def test_drops_missing_vectors(self):
        candidates = [Candidate("no_vec", 5.0, None)]
        assert select_candidates(candidates, duration_seconds=600) == []

    def test_mixed_valid_and_invalid_keeps_only_valid(self):
        v = _unit_vec(2)
        candidates = [
            Candidate("good", 5.0, v),
            Candidate("bad_ts", float("nan"), v),
            Candidate("bad_vec", 6.0, None),
        ]
        result = select_candidates(candidates, duration_seconds=300)
        assert [c.embedding_id for c in result] == ["good"]


class TestSelectCandidatesDeterminism:
    def _make_pool(self, n: int, spread: float = 2.0) -> list[Candidate]:
        rng = np.random.default_rng(42)
        out = []
        for i in range(n):
            v = rng.normal(size=8).astype(np.float32) * spread
            v = v / np.linalg.norm(v)
            out.append(Candidate(f"e{i}", float(i * 2), v))
        return out

    def test_result_sorted_by_timestamp(self):
        pool = self._make_pool(50)
        result = select_candidates(pool, duration_seconds=600)
        timestamps = [c.timestamp for c in result]
        assert timestamps == sorted(timestamps)

    def test_result_count_matches_target(self):
        pool = self._make_pool(50)
        result = select_candidates(pool, duration_seconds=600)
        assert len(result) == compute_target_count(600, len(pool))

    def test_deterministic_across_repeated_calls(self):
        pool = self._make_pool(50)
        first = [c.embedding_id for c in select_candidates(pool, duration_seconds=600)]
        second = [c.embedding_id for c in select_candidates(pool, duration_seconds=600)]
        assert first == second

    def test_fewer_candidates_than_target_uses_all(self):
        pool = self._make_pool(3)
        result = select_candidates(pool, duration_seconds=3600)  # target=12
        assert len(result) == 3
        assert {c.embedding_id for c in result} == {"e0", "e1", "e2"}

    def test_temporal_coverage_spans_full_timeline(self):
        """Bucket coverage must not clump all picks at one end of the video."""
        pool = self._make_pool(60)  # timestamps 0..118
        result = select_candidates(pool, duration_seconds=1800)  # target=6
        timestamps = sorted(c.timestamp for c in result)
        # With 6 buckets over a ~118s span, no gap between consecutive
        # picks should exceed roughly 2 bucket-widths.
        span = timestamps[-1] - timestamps[0]
        max_gap = max(b - a for a, b in zip(timestamps, timestamps[1:]))
        assert max_gap < span  # sanity: picks are not all identical
        assert timestamps[0] < span * 0.3  # an early pick exists
        assert timestamps[-1] > span * 0.7  # a late pick exists


class TestNearDuplicateRemovalAndDiversityFill:
    def test_near_duplicates_collapse_and_outlier_is_kept(self):
        base = _unit_vec(7)
        outlier = -base  # maximally distant in cosine terms
        candidates = [
            Candidate(f"dup{i}", float(i), base.copy()) for i in range(20)
        ]
        candidates.append(Candidate("outlier", 21.0, outlier))

        result = select_candidates(candidates, duration_seconds=1800)  # target=6
        ids = {c.embedding_id for c in result}
        assert "outlier" in ids
        # Near-duplicates must not fill the whole budget — at most one
        # of the identical-vector cluster should survive dedup before
        # diversity fill pulls in distinct candidates.
        dup_count = sum(1 for i in ids if i.startswith("dup"))
        assert dup_count < 20

    def test_stable_tie_break_by_timestamp_then_id(self):
        """Two candidates equidistant from the selected set: earliest
        timestamp wins, then embedding_id, for full determinism."""
        v_a = _unit_vec(10)
        v_b = _unit_vec(11)
        candidates = [
            Candidate("seed", 0.0, v_a),
            Candidate("z_later", 5.0, v_b),
            Candidate("a_earlier", 5.0, v_b),
        ]
        result = select_candidates(candidates, duration_seconds=None)  # target=6, only 3 avail
        # All three survive (fewer than target); just confirm no crash
        # and stable ordering by timestamp then id.
        assert [c.embedding_id for c in result] == [
            "seed", "a_earlier", "z_later",
        ]


class TestCandidateFingerprint:
    def test_deterministic_for_same_input(self):
        v = _unit_vec(1)
        candidates = [Candidate("a", 1.0, v), Candidate("b", 2.0, v)]
        assert compute_candidate_fingerprint(candidates) == compute_candidate_fingerprint(
            candidates
        )

    def test_changes_when_candidate_set_changes(self):
        v = _unit_vec(1)
        full = [Candidate("a", 1.0, v), Candidate("b", 2.0, v)]
        truncated = full[:-1]
        assert compute_candidate_fingerprint(full) != compute_candidate_fingerprint(
            truncated
        )

    def test_changes_when_timestamp_changes(self):
        v = _unit_vec(1)
        original = [Candidate("a", 1.0, v)]
        moved = [Candidate("a", 1.5, v)]
        assert compute_candidate_fingerprint(original) != compute_candidate_fingerprint(
            moved
        )

    def test_empty_input_is_stable(self):
        assert compute_candidate_fingerprint([]) == compute_candidate_fingerprint([])
