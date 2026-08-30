"""Tests for app.workers.clip_concepts module.

Covers preset loading, scoring aggregation, cache invalidation,
and the pure vector scoring logic without loading the real CLIP
model.
"""

import json

import numpy as np
import pytest

from app.workers import clip_concepts


# ---------------------------------------------------------------------------
# load_preset_concepts
# ---------------------------------------------------------------------------


class TestLoadPresetConcepts:
    """Tests for load_preset_concepts: parses clip_concepts.json."""

    def test_loads_and_flattens_categories(self, tmp_path):
        path = tmp_path / "concepts.json"
        path.write_text(
            json.dumps({
                "scenes": ["風景", "屋内"],
                "subjects": ["人物", "動物"],
            }),
            encoding="utf-8",
        )
        result = clip_concepts.load_preset_concepts(path)
        assert set(result) == {"風景", "屋内", "人物", "動物"}

    def test_ignores_underscore_keys(self, tmp_path):
        path = tmp_path / "concepts.json"
        path.write_text(
            json.dumps({
                "_comment": "metadata should be ignored",
                "scenes": ["風景"],
            }),
            encoding="utf-8",
        )
        result = clip_concepts.load_preset_concepts(path)
        assert result == ["風景"]

    def test_deduplicates_across_categories(self, tmp_path):
        path = tmp_path / "concepts.json"
        path.write_text(
            json.dumps({
                "cat_a": ["料理", "食べ物"],
                "cat_b": ["料理", "飲み物"],
            }),
            encoding="utf-8",
        )
        result = clip_concepts.load_preset_concepts(path)
        # Order preserved, duplicates removed
        assert result == ["料理", "食べ物", "飲み物"]

    def test_missing_file_returns_empty(self, tmp_path):
        result = clip_concepts.load_preset_concepts(tmp_path / "nope.json")
        assert result == []

    def test_invalid_json_returns_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json", encoding="utf-8")
        result = clip_concepts.load_preset_concepts(path)
        assert result == []

    def test_default_vocabulary_file_loads(self):
        """The shipped clip_concepts.json must parse without error."""
        result = clip_concepts.load_preset_concepts()
        # Ship with at least 100 concepts to provide meaningful coverage
        assert len(result) >= 100


# ---------------------------------------------------------------------------
# score_vectors_against_concepts
# ---------------------------------------------------------------------------


def _normalized(v: list[float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


class TestScoreVectorsAgainstConcepts:
    """Tests for score_vectors_against_concepts: vector math is the core."""

    def test_returns_empty_on_no_vectors(self):
        result = clip_concepts.score_vectors_against_concepts(
            [], {"料理": _normalized([1.0, 0.0])}
        )
        assert result == []

    def test_returns_empty_on_no_concepts(self):
        result = clip_concepts.score_vectors_against_concepts(
            [_normalized([1.0, 0.0])], {}
        )
        assert result == []

    def test_single_vector_matches_aligned_concept(self):
        # Vector and concept point in the same direction → cos sim = 1.0
        vectors = [_normalized([1.0, 0.0, 0.0])]
        concepts = {
            "料理": _normalized([1.0, 0.0, 0.0]),
            "風景": _normalized([0.0, 1.0, 0.0]),
        }
        result = clip_concepts.score_vectors_against_concepts(
            vectors, concepts, threshold=0.5
        )
        assert len(result) == 1
        assert result[0][0] == "料理"
        assert result[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_threshold_filters_low_scores(self):
        vectors = [_normalized([1.0, 0.0, 0.0])]
        concepts = {
            "near": _normalized([0.9, 0.1, 0.0]),
            "far": _normalized([0.1, 0.9, 0.0]),
        }
        # "far" should score below 0.5
        result = clip_concepts.score_vectors_against_concepts(
            vectors, concepts, threshold=0.5
        )
        names = [r[0] for r in result]
        assert "near" in names
        assert "far" not in names

    def test_video_frequency_aggregation(self):
        """A concept appearing in multiple frames ranks higher than a
        concept appearing in one frame, given equal peak scores."""
        # 3 frames all pointing at "料理", 1 frame at "風景"
        cooking = _normalized([1.0, 0.0, 0.0])
        landscape = _normalized([0.0, 1.0, 0.0])
        vectors = [cooking, cooking, cooking, landscape]
        concepts = {
            "料理": cooking,
            "風景": landscape,
        }
        result = clip_concepts.score_vectors_against_concepts(
            vectors, concepts, threshold=0.5
        )
        # Both pass threshold, but 料理 appears more often → ranked first
        assert result[0][0] == "料理"
        assert result[1][0] == "風景"

    def test_top_k_limits_output(self):
        v = _normalized([1.0, 0.0])
        concepts = {
            f"c{i}": v for i in range(10)
        }
        result = clip_concepts.score_vectors_against_concepts(
            [v], concepts, threshold=0.0, top_k=3
        )
        assert len(result) == 3

    def test_max_score_reported_not_frequency(self):
        """The score returned per concept is its peak cosine similarity,
        not the frequency — downstream filters want the raw signal."""
        high = _normalized([1.0, 0.0])
        low = _normalized([0.7, 0.7])  # cos = 0.7 with high

        # "料理" has one high-score frame and one low-score frame
        vectors = [high, low]
        concepts = {"料理": high}
        result = clip_concepts.score_vectors_against_concepts(
            vectors, concepts, threshold=0.5
        )
        assert result[0][0] == "料理"
        # Should report the max (from the first frame, score ≈ 1.0)
        assert result[0][1] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Vocabulary assembly
# ---------------------------------------------------------------------------


class TestVocabularyAssembly:
    """The pool handed to CLIP is preset concepts + the drive's tags."""

    def _fake_encode(self, calls):
        def _encode(concepts):
            calls.append(list(concepts))
            return {c: np.array([1.0, 0.0], dtype=np.float32) for c in concepts}

        return _encode

    def test_excludes_user_tags_when_flag_off(
        self, tmp_path, monkeypatch, make_settings
    ):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"x": ["a", "b"]}), encoding="utf-8")
        clip_concepts.reset_cache()
        monkeypatch.setattr(clip_concepts, "settings", make_settings())
        monkeypatch.setattr(
            clip_concepts, "_encode_concepts", self._fake_encode([])
        )
        monkeypatch.setattr(
            clip_concepts, "load_user_tags", lambda drive: ["user_only"]
        )

        result = clip_concepts.get_concept_embeddings(
            drive="main", preset_path=path, include_user_tags=False
        )

        assert sorted(result) == ["a", "b"]
        clip_concepts.reset_cache()

    def test_merges_preset_and_user_tags(
        self, tmp_path, monkeypatch, make_settings
    ):
        path = tmp_path / "c.json"
        path.write_text(
            json.dumps({"x": ["preset1", "preset2"]}), encoding="utf-8"
        )
        clip_concepts.reset_cache()
        monkeypatch.setattr(clip_concepts, "settings", make_settings())
        monkeypatch.setattr(
            clip_concepts, "_encode_concepts", self._fake_encode([])
        )
        monkeypatch.setattr(
            clip_concepts, "load_user_tags", lambda drive: ["user1", "user2"]
        )

        result = clip_concepts.get_concept_embeddings(
            drive="main", preset_path=path
        )

        assert sorted(result) == ["preset1", "preset2", "user1", "user2"]
        clip_concepts.reset_cache()

    def test_user_tag_duplicating_preset_is_encoded_once(
        self, tmp_path, monkeypatch, make_settings
    ):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"x": ["料理"]}), encoding="utf-8")
        clip_concepts.reset_cache()
        encode_calls: list[list[str]] = []
        monkeypatch.setattr(clip_concepts, "settings", make_settings())
        monkeypatch.setattr(
            clip_concepts, "_encode_concepts", self._fake_encode(encode_calls)
        )
        monkeypatch.setattr(
            clip_concepts, "load_user_tags", lambda drive: ["料理", "独自タグ"]
        )

        result = clip_concepts.get_concept_embeddings(
            drive="main", preset_path=path
        )

        assert sorted(result) == sorted(["料理", "独自タグ"])
        assert encode_calls == [["料理"], ["独自タグ"]]
        clip_concepts.reset_cache()

    def test_no_drive_means_no_tag_vocabulary(
        self, tmp_path, monkeypatch, make_settings
    ):
        """A caller without a drive gets presets only, never every drive's tags."""
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"x": ["preset1"]}), encoding="utf-8")
        clip_concepts.reset_cache()
        monkeypatch.setattr(clip_concepts, "settings", make_settings())
        monkeypatch.setattr(
            clip_concepts, "_encode_concepts", self._fake_encode([])
        )

        def _explode(drive):
            raise AssertionError("tags must not be loaded without a drive")

        monkeypatch.setattr(clip_concepts, "load_user_tags", _explode)

        result = clip_concepts.get_concept_embeddings(
            drive=None, preset_path=path
        )

        assert sorted(result) == ["preset1"]
        clip_concepts.reset_cache()


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


class TestConceptEmbeddingCache:
    """Tests for get_concept_embeddings: caching and invalidation."""

    def test_cache_invalidates_on_model_change(
        self, tmp_path, monkeypatch, make_settings
    ):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"x": ["tag1"]}), encoding="utf-8")

        clip_concepts.reset_cache()

        encode_calls: list[list[str]] = []

        def fake_encode(concepts):
            encode_calls.append(list(concepts))
            return {c: np.array([1.0, 0.0], dtype=np.float32) for c in concepts}

        monkeypatch.setattr(clip_concepts, "_encode_concepts", fake_encode)
        monkeypatch.setattr(clip_concepts, "load_user_tags", lambda drive: [])

        # First model
        from app.config import ModelConfig
        settings_a = make_settings(models=ModelConfig(clip="model-a"))
        monkeypatch.setattr(clip_concepts, "settings", settings_a)
        clip_concepts.get_concept_embeddings(drive="main", preset_path=path)
        assert len(encode_calls) == 1

        # Same model, second call → cached
        clip_concepts.get_concept_embeddings(drive="main", preset_path=path)
        assert len(encode_calls) == 1

        # Different model → re-encodes
        settings_b = make_settings(models=ModelConfig(clip="model-b"))
        monkeypatch.setattr(clip_concepts, "settings", settings_b)
        clip_concepts.get_concept_embeddings(drive="main", preset_path=path)
        assert len(encode_calls) == 2

        clip_concepts.reset_cache()

    def test_force_reload_bypasses_cache(
        self, tmp_path, monkeypatch, make_settings
    ):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"x": ["tag1"]}), encoding="utf-8")

        clip_concepts.reset_cache()
        encode_calls: list[list[str]] = []

        def fake_encode(concepts):
            encode_calls.append(list(concepts))
            return {c: np.array([1.0, 0.0], dtype=np.float32) for c in concepts}

        monkeypatch.setattr(clip_concepts, "_encode_concepts", fake_encode)
        monkeypatch.setattr(clip_concepts, "load_user_tags", lambda drive: [])
        monkeypatch.setattr(clip_concepts, "settings", make_settings())

        clip_concepts.get_concept_embeddings(drive="main", preset_path=path)
        clip_concepts.get_concept_embeddings(
            drive="main", preset_path=path, force_reload=True
        )
        assert len(encode_calls) == 2

        clip_concepts.reset_cache()
