"""SPEC-ADDON-043 / SPEC-ADDON-044: image matches survive the score gates of
precision search.

Distances are L2 between unit vectors, so a row at distance ``d`` scores
``1 - d^2 / 2``.
"""

from __future__ import annotations

import numpy as np
import pytest

from app import search as search_mod
from tests.test_search_knn_partition import (  # noqa: F401
    HOME,
    OTHER,
    RRF_CANDIDATES,
    _by_id,
    _clip,
    _distance,
    _far_scene_rows,
    _far_thumbnails_of_other_files,
    _row,
    _rows,
    _scores,
    _spread,
    _use_search_config,
    clip_index,
)
from tests.vec_partition_db import search_index, unit  # noqa: F401

FLAT = [0.800 + 0.002 * i for i in range(10)]
LOW_CV = [0.83] + [0.7995 + 0.00002 * i for i in range(39)]


def _clip_scene_on(index, mode="precision"):
    return search_mod._vector_search_clip(
        unit(index.dim("vec_clip"), 0), RRF_CANDIDATES, mode=mode,
        include_scene_clip=True,
    )


# ---------------------------------------------------------------------------
# SPEC-ADDON-043 I1: no standout needed, Scene search off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scores",
    [
        pytest.param(FLAT, id="gap-below-threshold"),
        pytest.param(LOW_CV, id="gap-above-threshold-cv-below-001"),
    ],
)
@pytest.mark.parametrize("far", ["none", "scene", "other-drive", "inactive"])
def test_spec_addon_043_thumbnails_without_a_standout_are_returned_scene_off(
    clip_index, monkeypatch, scores, far,
):
    _use_search_config(monkeypatch)
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in scores], "t")
    if far == "scene":
        _far_scene_rows(clip_index, n=10)
    elif far != "none":
        _far_thumbnails_of_other_files(clip_index, far)

    got = _by_id(_clip(clip_index))

    assert set(got) == {f"e_t{i}" for i in range(len(scores))}


def test_spec_addon_043_a_lone_thumbnail_is_returned(clip_index, monkeypatch):
    _use_search_config(monkeypatch)
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(0.8)], "t")

    assert set(_by_id(_clip(clip_index))) == {"e_t0"}


# ---------------------------------------------------------------------------
# SPEC-ADDON-043 I2: Scene search on, thumbnails and scene frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scores", [FLAT, LOW_CV], ids=["flat", "low-cv"])
def test_spec_addon_043_thumbnails_and_scene_frames_without_a_standout_are_returned_scene_on(
    clip_index, monkeypatch, scores,
):
    _use_search_config(monkeypatch)
    half = len(scores) // 2
    _rows(clip_index, "vec_clip", "clip_thumbnail",
          [_distance(s) for s in scores[:half]], "t")
    _rows(clip_index, "vec_clip", "clip", [_distance(s) for s in scores[half:]], "s",
          axis0=3000)

    got = _by_id(_clip_scene_on(clip_index))

    assert set(got) == (
        {f"e_t{i}" for i in range(half)}
        | {f"e_s{i}" for i in range(len(scores) - half)}
    )


# ---------------------------------------------------------------------------
# SPEC-ADDON-043 I3: floors and margin still decide
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["precision", "recall"])
def test_spec_addon_043_scene_on_drops_rows_below_each_types_floor_and_beyond_the_margin(
    clip_index, monkeypatch, mode,
):
    _use_search_config(monkeypatch, min_score_clip_thumbnail=0.50, min_score_clip=0.70,
                       score_cutoff_margin=10.0)
    rows = [
        ("t_keep", "clip_thumbnail", 0.80),
        ("t_floor", "clip_thumbnail", 0.45),
        ("s_keep", "clip", 0.75),
        ("s_floor", "clip", 0.65),
    ]
    for i, (emb_id, etype, score) in enumerate(rows):
        _row(clip_index, "vec_clip", emb_id, etype, _distance(score), i)

    got = _scores(_clip_scene_on(clip_index, mode=mode))

    assert set(got) == {"t_keep", "s_keep"}


def test_spec_addon_043_margin_cuts_rows_far_below_the_best_when_scores_are_flat(
    clip_index, monkeypatch,
):
    _use_search_config(monkeypatch, score_cutoff_margin=0.05)
    scores = [0.80, 0.79, 0.78, 0.70, 0.60]
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in scores], "t")

    assert set(_by_id(_clip(clip_index))) == {"e_t0", "e_t1", "e_t2"}


def test_spec_addon_043_recall_mode_returns_a_flat_set(clip_index, monkeypatch):
    _use_search_config(monkeypatch)
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in FLAT], "t")

    got = _by_id(_clip(clip_index, mode="recall"))

    assert set(got) == {f"e_t{i}" for i in range(len(FLAT))}


# ---------------------------------------------------------------------------
# SPEC-ADDON-043 I4: the text channel keeps its gap and CV checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scores",
    [
        pytest.param([0.93 + 0.002 * i for i in range(10)], id="gap-below-threshold"),
        pytest.param([0.97] + [0.9395 + 0.00002 * i for i in range(39)], id="cv-below-001"),
    ],
)
def test_spec_addon_043_text_channel_still_discards_a_flat_set(
    search_index, monkeypatch, scores,  # noqa: F811
):
    _use_search_config(monkeypatch, score_cutoff_margin=10.0)
    gap = max(scores) - float(np.mean(scores))
    cv = float(np.std(scores)) / float(np.mean(scores))
    assert gap < 0.02 or cv < 0.01
    for i, s in enumerate(scores):
        _row(search_index, "vec_text", f"m{i}", "metadata", _distance(s), i)

    got = search_mod._vector_search_text(
        unit(search_index.dim("vec_text"), 0), 50, mode="precision",
    )

    assert got == []


def test_spec_addon_043_text_channel_still_returns_a_standout(search_index, monkeypatch):  # noqa: F811
    _use_search_config(monkeypatch, score_cutoff_margin=0.10)
    scores = [0.98] + [0.90 + 0.001 * i for i in range(9)]
    for i, s in enumerate(scores):
        _row(search_index, "vec_text", f"m{i}", "metadata", _distance(s), i)

    got = _scores(search_mod._vector_search_text(
        unit(search_index.dim("vec_text"), 0), 50, mode="precision",
    ))

    assert "m0" in got


# ---------------------------------------------------------------------------
# SPEC-ADDON-044 I5 / I6 / I7 / I8: the combined cutoff in _build_results
# ---------------------------------------------------------------------------

STRONG = "metadata"


def _fs(file_id, score, types):
    return search_mod._FileScore(
        file_id=file_id,
        combined_score=score,
        matches=[search_mod.MatchInfo(match_type=t, text="x", score=score) for t in types],
        match_types=set(types),
    )


@pytest.fixture()
def files(search_index):  # noqa: F811
    for fid in ("top", "mid", "kw", "weak", "img", "thumb", "both", "scene",
                "other", "doc", "gone", "outside"):
        mime = "text/plain" if fid == "doc" else "image/png"
        search_index.add_file(
            fid, drive=OTHER if fid == "other" else HOME, active=fid != "gone",
            mime_type=mime,
        )
    return search_index


def _build(scores, *, mode="precision", limit=50, **kw):
    return search_mod._build_results(
        {s.file_id: s for s in scores}, kw.pop("file_type", None),
        kw.pop("drive", HOME), limit, mode=mode, **kw,
    )


def test_spec_addon_044_image_matches_below_the_ratio_survive_and_others_are_dropped(
    files, monkeypatch,
):
    _use_search_config(monkeypatch)
    results = _build([
        _fs("top", 1.0, [STRONG]),
        _fs("weak", 0.1, [STRONG]),
        _fs("thumb", 0.1, ["clip_thumbnail"]),
        _fs("scene", 0.1, ["clip"]),
        _fs("both", 0.1, [STRONG, "clip_thumbnail"]),
    ])

    assert [r.file_id for r in results][0] == "top"
    assert {r.file_id for r in results} == {"top", "thumb", "scene", "both"}


def test_spec_addon_044_a_result_exactly_at_the_ratio_is_kept(files, monkeypatch):
    _use_search_config(monkeypatch)
    results = _build([_fs("top", 1.0, [STRONG]), _fs("mid", 0.5, [STRONG]),
                      _fs("weak", 0.49, [STRONG])])

    assert {r.file_id for r in results} == {"top", "mid"}


def test_spec_addon_044_an_image_match_raises_top_score_for_the_cutoff_of_others(
    files, monkeypatch,
):
    _use_search_config(monkeypatch)
    without = _build([_fs("top", 0.6, [STRONG]), _fs("mid", 0.4, [STRONG])])
    with_image = _build([_fs("top", 0.6, [STRONG]), _fs("mid", 0.4, [STRONG]),
                         _fs("both", 1.0, [STRONG, "clip_thumbnail"])])

    assert {r.file_id for r in without} == {"top", "mid"}
    assert {r.file_id for r in with_image} == {"top", "both"}


def test_spec_addon_044_skip_cutoff_keeps_every_result(files, monkeypatch):
    _use_search_config(monkeypatch)
    results = _build([_fs("top", 1.0, [STRONG]), _fs("weak", 0.01, [STRONG])],
                     skip_cutoff=True)

    assert {r.file_id for r in results} == {"top", "weak"}


def test_spec_addon_044_recall_mode_drops_image_matches_below_the_recall_ratio(
    files, monkeypatch,
):
    _use_search_config(monkeypatch)
    ratio = search_mod._RECALL_PARAMS.score_cutoff_ratio
    results = _build([
        _fs("top", 1.0, [STRONG]),
        _fs("thumb", ratio * 0.5, ["clip_thumbnail"]),
        _fs("scene", ratio * 0.5, ["clip"]),
        _fs("mid", ratio * 1.5, ["clip_thumbnail"]),
        _fs("weak", ratio * 0.5, [STRONG]),
    ], mode="recall")

    assert {r.file_id for r in results} == {"top", "mid"}


def test_spec_addon_044_results_are_sorted_by_score_with_exempt_ones_in_place(
    files, monkeypatch,
):
    _use_search_config(monkeypatch)
    results = _build([
        _fs("thumb", 0.1, ["clip_thumbnail"]),
        _fs("mid", 0.7, [STRONG]),
        _fs("top", 1.0, [STRONG]),
        _fs("img", 0.3, ["clip"]),
    ])

    assert [r.file_id for r in results] == ["top", "mid", "img", "thumb"]


def test_spec_addon_044_limit_truncates_after_the_cutoff_and_never_prefers_an_exempt_result(
    files, monkeypatch,
):
    _use_search_config(monkeypatch)
    scores = [_fs("top", 1.0, [STRONG]), _fs("mid", 0.7, [STRONG]),
              _fs("thumb", 0.1, ["clip_thumbnail"]), _fs("weak", 0.1, [STRONG])]

    assert [r.file_id for r in _build(scores, limit=2)] == ["top", "mid"]
    assert [r.file_id for r in _build(scores, limit=3)] == ["top", "mid", "thumb"]


@pytest.mark.parametrize(
    "kwargs, excluded",
    [
        pytest.param({}, {"other", "gone"}, id="drive-and-inactive"),
        pytest.param({"file_type": "image"}, {"other", "gone", "doc"}, id="file-type"),
        pytest.param({"file_id_scope": ["top", "thumb", "scene"]},
                     {"other", "gone", "doc", "both", "img"}, id="scope"),
    ],
)
def test_spec_addon_044_exempt_results_are_still_filtered(files, monkeypatch, kwargs, excluded):
    _use_search_config(monkeypatch)
    ids = ["thumb", "scene", "both", "img", "other", "gone", "doc"]
    scores = [_fs("top", 1.0, [STRONG])] + [_fs(i, 0.1, ["clip_thumbnail"]) for i in ids]

    got = {r.file_id for r in _build(scores, **kwargs)}

    assert got & excluded == set()
    assert {"thumb", "scene"} <= got


# ---------------------------------------------------------------------------
# SPEC-ADDON-043 I9: search() end to end
# ---------------------------------------------------------------------------


@pytest.fixture()
def embedded_query(clip_index, monkeypatch):
    dim = clip_index.dim("vec_clip")
    text_dim = clip_index.dim("vec_text")
    monkeypatch.setattr(search_mod, "embed_text_clip", lambda *_a, **_k: unit(dim, 0))
    monkeypatch.setattr(search_mod, "embed_query", lambda *_a, **_k: unit(text_dim, 0))
    _use_search_config(monkeypatch)
    return clip_index


def test_spec_addon_044_i9_search_returns_all_60_in_band_thumbnails_at_limit_100_and_the_50_nearest_at_limit_50(
    embedded_query,
):
    """SPEC-ADDON-044 I9 (revised): 60 thumbnails inside a 0.01 band above the
    floor, and no other match."""
    _rows(embedded_query, "vec_clip", "clip_thumbnail",
          [_distance(s) for s in _spread(60, 0.051, 0.061)], "pic")

    at_100 = [r.file_id for r in search_mod.search("zzzz", limit=100, drive=HOME).results]
    at_50 = [r.file_id for r in search_mod.search("zzzz", limit=50, drive=HOME).results]

    assert len(at_100) == 60
    assert set(at_100) == {f"pic{i}" for i in range(60)}
    # The band rises with the index, so the 50 nearest are pic10..pic59.
    assert len(at_50) == 50
    assert set(at_50) == {f"pic{i}" for i in range(10, 60)}


def test_spec_addon_044_i9_search_lists_a_keyword_match_first_and_all_60_image_matches_after_it(
    embedded_query,
):
    _rows(embedded_query, "vec_clip", "clip_thumbnail",
          [_distance(s) for s in _spread(60, 0.051, 0.061)], "pic")
    embedded_query.add_file("doc", drive=HOME)
    embedded_query.sql(
        "INSERT INTO fts_files(file_id, filename, title, description, tags_text) "
        "VALUES('doc', 'zzzz.txt', 'zzzz', '', '')"
    )

    ids = [r.file_id for r in search_mod.search("zzzz", limit=100, drive=HOME).results]

    assert len(ids) == 61
    assert ids[0] == "doc"
    assert set(ids[1:]) == {f"pic{i}" for i in range(60)}
