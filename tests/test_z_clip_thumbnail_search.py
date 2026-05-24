"""Tests for the search-layer plumbing of ``embedding_type="clip_thumbnail"``.

Spec ``2026-05-02-thumbnail-clip-default-shallow-search.md``:

- ``SearchConfig`` exposes ``min_score_clip_thumbnail`` /
  ``rrf_weight_clip_thumbnail`` / ``type_weight_clip_thumbnail``
  knobs separate from scene CLIP.
- ``_vector_search_clip`` filters to ``clip_thumbnail`` only by
  default; ``include_scene_clip=True`` unions in scene CLIP.
- ``_combine_scores_cosine`` / ``_combine_scores_rrf`` label hits
  with the actual ``embedding_type`` so the UI can distinguish.

This file is named ``test_z_*`` for the same reason as
``test_z_clip_thumbnail_dispatch.py``: pytest-asyncio AUTO mode +
sibling tests using the deprecated ``asyncio.get_event_loop()``
pattern make ordering fragile. Sorting after them avoids pollution.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Stub heavy ML deps before importing app.search (transitive open_clip,
# torch, etc.). conftest already covers most but ``test_z_*`` runs late
# enough that some have already been replaced by real imports.
for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.config import SearchConfig  # noqa: E402
from app.search import _TYPE_WEIGHTS  # noqa: E402


# ---------------------------------------------------------------------------
# SearchConfig defaults
# ---------------------------------------------------------------------------


def test_search_config_exposes_clip_thumbnail_threshold():
    cfg = SearchConfig()
    assert hasattr(cfg, "min_score_clip_thumbnail")
    assert isinstance(cfg.min_score_clip_thumbnail, float)


def test_search_config_clip_thumbnail_threshold_default():
    """Default follows the tuned shallow-search threshold."""
    cfg = SearchConfig()
    assert cfg.min_score_clip_thumbnail == 0.05
    # And it is strictly looser than (or equal to) scene CLIP.
    assert cfg.min_score_clip_thumbnail <= cfg.min_score_clip


def test_search_config_clip_thumbnail_rrf_weight_default():
    """clip_thumbnail ranks at parity, scene CLIP at half."""
    cfg = SearchConfig()
    assert cfg.rrf_weight_clip_thumbnail == 1.0
    assert cfg.rrf_weight_clip == 0.5


def test_search_config_clip_thumbnail_type_weight_default():
    cfg = SearchConfig()
    assert cfg.type_weight_clip_thumbnail == 1.0


# ---------------------------------------------------------------------------
# _TYPE_WEIGHTS map
# ---------------------------------------------------------------------------


def test_type_weights_map_includes_clip_thumbnail():
    """The map drives both _combine_scores_cosine match labels and weights."""
    assert "clip_thumbnail" in _TYPE_WEIGHTS
    assert _TYPE_WEIGHTS["clip_thumbnail"] == "type_weight_clip_thumbnail"


def test_type_weights_map_keeps_scene_clip_separate():
    """Scene CLIP and thumbnail CLIP must not share a weight knob."""
    assert _TYPE_WEIGHTS["clip"] == "type_weight_clip"
    assert _TYPE_WEIGHTS["clip"] != _TYPE_WEIGHTS["clip_thumbnail"]
