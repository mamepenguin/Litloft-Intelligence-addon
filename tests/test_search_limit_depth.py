"""SPEC-ADDON-046: the first list reaches as many of the drive's nearest
representative frames as the effective ``limit`` asks for, and ``limit`` may
reach 1000.

Distances are L2 between unit vectors, so a row at distance ``d`` scores
``1 - d^2 / 2``.
"""

from __future__ import annotations

import pytest

from app import search as search_mod
from app.config import SearchConfig
from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    api,
    patch_everywhere,
    search_index,
)
from tests.test_search_knn_partition import (  # noqa: F401
    HOME,
    OTHER,
    RRF_CANDIDATES,
    _distance,
    _rows,
    _spread,
    _use_search_config,
    embedded_search,
    image_depth,
)
from tests.vec_partition_db import unit

#: The floor's default is 0.05; a 0.01 band just above it passes the floor and
#: the margin together, so only the depth decides how many rows come back.
IN_BAND = (0.051, 0.061)


def _ids(body) -> list[str]:
    return [r["file_id"] for r in body["results"]]


@pytest.fixture()
def search_api(api, search_index, monkeypatch):  # noqa: F811
    """``/search`` over HTTP with the image and text models stubbed."""
    dim = search_index.dim("vec_clip")
    text_dim = search_index.dim("vec_text")
    patch_everywhere(monkeypatch, "embed_text_clip", lambda *_a, **_k: unit(dim, 0))
    patch_everywhere(monkeypatch, "embed_query", lambda *_a, **_k: unit(text_dim, 0))

    async def _hydrate(_ids):
        return {}

    patch_everywhere(monkeypatch, "hydrate_files", _hydrate)

    def _get(params):
        return api("/search", params)

    return _get


# ---------------------------------------------------------------------------
# SPEC-ADDON-046 I1: limit up to 1000, and the same first list for limit <= 50
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "limit, status",
    [
        pytest.param(1000, 200, id="1000-accepted"),
        pytest.param(1001, 422, id="1001-422"),
    ],
)
def test_spec_addon_046_i1_search_limit_bound(search_api, limit, status):
    response = search_api({"q": "zzzz", "limit": limit})

    assert response.status_code == status, response.text


def test_spec_addon_046_max_limit_defaults_to_1000():
    assert SearchConfig().max_limit == 1000


def test_spec_addon_046_i1_limit_at_or_below_50_keeps_the_50_frame_depth(
    search_index, monkeypatch, embedded_search, image_depth,
):
    """SPEC-ADDON-046 I1: with ``limit`` <= 50 the first list is the one the
    change leaves untouched."""
    _use_search_config(monkeypatch, score_gap_threshold=0.0,
                       min_score_clip_thumbnail=-1.0, score_cutoff_margin=10.0)
    _rows(search_index, "vec_clip", "clip_thumbnail",
          [_distance(s) for s in _spread(60, *IN_BAND)], "pic")

    response = search_mod.search("zzzz", limit=50, drive=HOME)

    assert image_depth == [RRF_CANDIDATES]
    # Scores rise with the index, so the 50 nearest are pic59 down to pic10.
    assert [r.file_id for r in response.results] == [
        f"pic{i}" for i in range(59, 9, -1)
    ]


# ---------------------------------------------------------------------------
# SPEC-ADDON-046 I2: recall mode and Scene search keep the image depth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode, scene",
    [
        pytest.param("recall", False, id="recall"),
        pytest.param("recall", True, id="recall-scene-on"),
        pytest.param("precision", True, id="precision-scene-on"),
    ],
)
def test_spec_addon_046_i2_the_image_channel_takes_at_most_rrf_candidates_frames(
    search_index, monkeypatch, embedded_search, image_depth, mode, scene,
):
    _use_search_config(monkeypatch, score_gap_threshold=0.0,
                       min_score_clip_thumbnail=-1.0, score_cutoff_margin=10.0)
    _rows(search_index, "vec_clip", "clip_thumbnail",
          [_distance(s) for s in _spread(60, *IN_BAND)], "pic")
    _rows(search_index, "vec_clip", "clip",
          [_distance(s) for s in _spread(60, *IN_BAND)], "scene", axis0=3000)

    response = search_mod.search("zzzz", limit=1000, drive=HOME, mode=mode,
                                 include_scene_clip=scene)

    assert image_depth == [RRF_CANDIDATES]
    image_hits = [
        r for r in response.results
        if {"clip", "clip_thumbnail"} & set(r.match_types)
    ]
    assert len(image_hits) <= RRF_CANDIDATES


# ---------------------------------------------------------------------------
# SPEC-ADDON-046 I3: every in-band frame of the drive is in the response while
# it holds fewer than effective_limit results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "limit, n_thumbnails, crowd, expected",
    [
        pytest.param(1000, 60, 4000, 60, id="past-the-50-cap-inside-the-4096-window"),
        pytest.param(50, 60, 0, 50, id="limit-50-unchanged"),
        pytest.param(1000, 1005, 0, 1000, id="stops-at-effective_limit"),
    ],
)
def test_spec_addon_046_i3_every_in_band_frame_is_in_the_response(
    search_index, search_api, limit, n_thumbnails, crowd, expected,
):
    if crowd:
        _rows(search_index, "vec_clip", "clip_thumbnail",
              _spread(crowd, 0.10, 0.11), "crowd", drive=OTHER,
              rows_per_file=50, axis0=10000)
    scores = _spread(n_thumbnails, *IN_BAND)
    _rows(search_index, "vec_clip", "clip_thumbnail",
          [_distance(s) for s in scores], "pic", drive=DRIVE, axis0=20000)

    response = search_api({"q": "zzzz", "limit": limit})

    assert response.status_code == 200, response.text
    ids = _ids(response.json())
    assert len(ids) == expected
    # Scores rise with the index, so the deepest response is the last ids first.
    assert ids == [
        f"pic{i}"
        for i in range(n_thumbnails - 1, n_thumbnails - 1 - expected, -1)
    ]


def test_spec_addon_046_i3_a_requested_limit_above_the_default_max_limit_stops_at_1000(
    search_index, monkeypatch, embedded_search,
):
    _use_search_config(monkeypatch, score_gap_threshold=0.0,
                       min_score_clip_thumbnail=-1.0, score_cutoff_margin=10.0)
    _rows(search_index, "vec_clip", "clip_thumbnail",
          [_distance(s) for s in _spread(1005, *IN_BAND)], "pic")

    response = search_mod.search("zzzz", limit=2000, drive=HOME)

    assert len(response.results) == 1000
