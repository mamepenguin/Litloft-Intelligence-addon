"""Tests for app.workers.tag_knn (CLIP-similarity-based tag recommender)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.workers import tag_knn


class TestRecommendTagsBySimilarity:
    """Tests cover ranking, min_support, and cold-start behavior.

    Neighbor lookup and tag fetch are stubbed so the tests don't need
    a real vec_clip index or Litloft DB connection.
    """

    def _setup(
        self,
        monkeypatch,
        *,
        query_vec=np.array([1.0, 0.0], dtype=np.float32),
        neighbors=None,
        tags_by_file=None,
    ):
        neighbors = neighbors or []
        tags_by_file = tags_by_file or {}
        monkeypatch.setattr(
            tag_knn, "_average_clip_vector", lambda fid: query_vec
        )
        monkeypatch.setattr(
            tag_knn,
            "_query_nearest_file_ids",
            lambda q, src, k: neighbors,
        )
        monkeypatch.setattr(
            tag_knn, "_load_tags_for_files", lambda ids: tags_by_file
        )

    def test_cold_start_returns_empty(self, monkeypatch):
        self._setup(monkeypatch, query_vec=None)
        assert tag_knn.recommend_tags_by_similarity("f1") == []

    def test_no_neighbors_returns_empty(self, monkeypatch):
        self._setup(monkeypatch, neighbors=[])
        assert tag_knn.recommend_tags_by_similarity("f1") == []

    def test_no_tagged_neighbors_returns_empty(self, monkeypatch):
        self._setup(
            monkeypatch,
            neighbors=[("f2", 0.9), ("f3", 0.8)],
            tags_by_file={},
        )
        assert tag_knn.recommend_tags_by_similarity("f1") == []

    def test_basic_aggregation_weights_by_similarity(self, monkeypatch):
        self._setup(
            monkeypatch,
            neighbors=[("f2", 0.9), ("f3", 0.5)],
            tags_by_file={
                "f2": ["料理", "和食"],
                "f3": ["料理", "レシピ"],
            },
        )
        result = tag_knn.recommend_tags_by_similarity("f1", min_support=1)
        # 料理 scored from both neighbors → highest
        assert result[0][0] == "料理"
        assert result[0][1] == pytest.approx(1.4, abs=1e-5)

    def test_min_support_drops_single_occurrence_tags(self, monkeypatch):
        self._setup(
            monkeypatch,
            neighbors=[("f2", 0.9), ("f3", 0.8)],
            tags_by_file={
                "f2": ["料理", "和食"],
                "f3": ["料理", "レシピ"],  # unique: 和食 and レシピ
            },
        )
        result = tag_knn.recommend_tags_by_similarity("f1", min_support=2)
        words = [r[0] for r in result]
        # 和食 and レシピ each appear in only 1 file — dropped
        assert "料理" in words
        assert "和食" not in words
        assert "レシピ" not in words

    def test_top_tags_limits_output(self, monkeypatch):
        # Build a scenario where many tags pass min_support
        neighbors = [(f"f{i}", 0.5) for i in range(10)]
        # Every tag appears on every neighbor
        tags_by_file = {
            fid: [f"tag{j}" for j in range(20)] for fid, _ in neighbors
        }
        self._setup(monkeypatch, neighbors=neighbors, tags_by_file=tags_by_file)
        result = tag_knn.recommend_tags_by_similarity(
            "f1", top_tags=5, min_support=2
        )
        assert len(result) == 5

    def test_similarity_weight_ordering(self, monkeypatch):
        """A tag on very similar neighbors outranks a tag on
        many weakly-similar neighbors."""
        self._setup(
            monkeypatch,
            neighbors=[
                ("f2", 0.95),  # very similar
                ("f3", 0.1),   # barely similar
                ("f4", 0.1),
            ],
            tags_by_file={
                "f2": ["A"],
                "f3": ["B"],
                "f4": ["B"],
            },
        )
        # A gets 0.95; B gets 0.1 + 0.1 = 0.2 → A wins despite fewer neighbors
        result = tag_knn.recommend_tags_by_similarity(
            "f1", min_support=1
        )
        assert result[0][0] == "A"
