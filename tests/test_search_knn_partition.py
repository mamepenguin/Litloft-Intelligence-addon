"""SPEC-ADDON-025 / SPEC-ADDON-026: the search CLIP channel with Scene search off
scans only the ``clip_thumbnail`` partition, and Ask's coarse retrieval scans only
the ``metadata`` partition.

Distances are L2 between unit vectors, so a row at distance ``d`` from the query
scores ``1 - d^2 / 2``. Most tests put more rows of another type nearer to the
query than the KNN fetches, so only a scan confined to the partition reaches the
rows the test expects.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from app import search as search_mod
from app.config import SearchConfig
from app.rag import coarse_retriever as cr_mod
from tests.vec_partition_db import (  # noqa: F401
    at_distance,
    search_index,
    unit,
)

HOME = "home"
OTHER = "elsewhere"
RRF_CANDIDATES = 50

PERMISSIVE = dict(
    score_gap_threshold=0.0,
    min_score_clip_thumbnail=-1.0,
    min_score_clip=-1.0,
    score_cutoff_margin=10.0,
)


def _score(d: float) -> float:
    return 1.0 - d * d / 2.0


def _distance(score: float) -> float:
    return float(np.sqrt(2.0 * (1.0 - score)))


def _use_search_config(monkeypatch, **kwargs) -> None:
    monkeypatch.setattr(
        search_mod, "settings",
        dataclasses.replace(search_mod.settings, search=SearchConfig(**kwargs)),
    )


def _rows(index, table, etype, distances, prefix, *, drive=HOME, active=True,
          rows_per_file=1, axis0=0):
    dim = index.dim(table)
    index.add_many([
        (
            f"{prefix}{i // rows_per_file}", drive, active, f"e_{prefix}{i}", etype, table,
            at_distance(dim, d, axis0 + i),
        )
        for i, d in enumerate(distances)
    ])


def _spread(n, near, far):
    return [near + (far - near) * i / max(n - 1, 1) for i in range(n)]


def _scene_rows_nearer_than_everything(index, n=RRF_CANDIDATES + 10):
    _rows(index, "vec_clip", "clip", _spread(n, 0.05, 0.25), "scene",
          rows_per_file=10, axis0=5000)


def _clip(index, limit=RRF_CANDIDATES, mode="precision", drive=HOME):
    query = unit(index.dim("vec_clip"), 0)
    return search_mod._vector_search_clip(
        query, limit, mode=mode, include_scene_clip=False, drive=drive,
    )


def _other_drive_and_inactive_thumbnails_nearer_than_everything(
    index, n=RRF_CANDIDATES + 10,
):
    _rows(index, "vec_clip", "clip_thumbnail", _spread(n, 0.05, 0.25), "otherdrive",
          drive=OTHER, rows_per_file=10, axis0=6000)
    _rows(index, "vec_clip", "clip_thumbnail", _spread(n, 0.06, 0.26), "inactive",
          active=False, rows_per_file=10, axis0=6500)


@pytest.fixture()
def clip_index(search_index):  # noqa: F811
    return search_index


def _by_id(matches):
    return {m.embedding_id: m for m in matches}


# ---------------------------------------------------------------------------
# SPEC-ADDON-025 I1 (r5): Scene search off returns the nearest clip_thumbnail rows
# of the requested drive's active files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_thumbnails, limit",
    [
        pytest.param(2, RRF_CANDIDATES, id="two-thumbnails"),
        pytest.param(10, RRF_CANDIDATES, id="fewer-than-rrf_candidates"),
        pytest.param(RRF_CANDIDATES, RRF_CANDIDATES, id="exactly-rrf_candidates"),
        pytest.param(RRF_CANDIDATES + 10, RRF_CANDIDATES, id="more-than-rrf_candidates"),
        pytest.param(30, 20, id="smaller-limit"),
    ],
)
def test_spec_addon_025_scene_off_returns_the_drives_nearest_active_thumbnails_however_many_other_rows_are_nearer(
    clip_index, monkeypatch, n_thumbnails, limit,
):
    _use_search_config(monkeypatch, **PERMISSIVE)
    distances = _spread(n_thumbnails, 0.30, 1.20)
    _rows(clip_index, "vec_clip", "clip_thumbnail", distances, "t")
    _scene_rows_nearer_than_everything(clip_index)
    _other_drive_and_inactive_thumbnails_nearer_than_everything(clip_index)

    got = _by_id(_clip(clip_index, limit=limit))

    expected = {f"e_t{i}": distances[i] for i in range(min(limit, n_thumbnails))}
    assert set(got) == set(expected)
    assert all(m.embedding_type == "clip_thumbnail" for m in got.values())
    for emb_id, d in expected.items():
        assert got[emb_id].score == pytest.approx(_score(d), abs=1e-4)


def test_spec_addon_025_scene_off_never_returns_a_thumbnail_row_without_a_partition_value(
    clip_index, monkeypatch,
):
    _use_search_config(monkeypatch, **PERMISSIVE)
    _rows(clip_index, "vec_clip", "clip_thumbnail", _spread(5, 0.30, 0.90), "t")
    clip_index.add_file("unkeyed")
    clip_index.add_embedding("e_unkeyed", "unkeyed", "clip_thumbnail", "vec_clip")
    clip_index.put_vector(
        "vec_clip", "e_unkeyed", at_distance(clip_index.dim("vec_clip"), 0.10, 900),
    )

    got = _by_id(_clip(clip_index))

    assert set(got) == {f"e_t{i}" for i in range(5)}


def test_spec_addon_025_scene_off_reaches_the_drives_thumbnails_only_within_the_4096_nearest(
    clip_index, monkeypatch,
):
    """4,090 other-drive thumbnails are nearer, so of the drive's ten thumbnails the
    first six rank inside the 4,096 nearest ``clip_thumbnail`` rows and the rest do
    not."""
    _use_search_config(monkeypatch, **PERMISSIVE)
    _rows(clip_index, "vec_clip", "clip_thumbnail", _spread(4090, 0.05, 0.25), "crowd",
          drive=OTHER, rows_per_file=50, axis0=10000)
    _rows(clip_index, "vec_clip", "clip_thumbnail", _spread(10, 0.30, 0.39), "t")

    got = _by_id(_clip(clip_index))

    assert set(got) == {f"e_t{i}" for i in range(6)}


# ---------------------------------------------------------------------------
# SPEC-ADDON-025 I3 (r5): the discard stages use the configured values. With Scene
# search off and a requested drive, the gap and CV are computed over the rows kept
# after the drive and activity filter
# ---------------------------------------------------------------------------


def _far_scene_rows(index, n=5):
    """Scene rows far from the query: inside the KNN today, so they widen the score
    distribution the gap and CV checks see, but below every thumbnail."""
    _rows(index, "vec_clip", "clip", _spread(n, 1.15, 1.20), "farscene", axis0=7000)


@pytest.mark.parametrize(
    "gap_threshold, thumbnail_scores",
    [
        pytest.param(0.02, [0.800 + 0.002 * i for i in range(10)], id="default-gap-flat-set"),
        pytest.param(0.10, [0.90] + [0.80 + 0.007 * i for i in range(9)], id="configured-gap"),
    ],
)
def test_spec_addon_025_a_thumbnail_set_without_a_standout_is_discarded_by_the_gap_check(
    clip_index, monkeypatch, gap_threshold, thumbnail_scores,
):
    _use_search_config(monkeypatch, score_gap_threshold=gap_threshold)
    _rows(clip_index, "vec_clip", "clip_thumbnail",
          [_distance(s) for s in thumbnail_scores], "t")
    _far_scene_rows(clip_index)

    assert _clip(clip_index) == []


def test_spec_addon_025_a_thumbnail_set_whose_spread_is_below_the_cv_cutoff_is_discarded(
    clip_index, monkeypatch,
):
    """One standout clears the default gap threshold, but the scores vary by less
    than 1% of their mean."""
    _use_search_config(monkeypatch)
    scores = [0.83] + [0.7995 + 0.00002 * i for i in range(39)]
    mean = float(np.mean(scores))
    assert scores[0] - mean > SearchConfig().score_gap_threshold
    assert float(np.std(scores)) / mean < 0.01
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in scores], "t")
    _far_scene_rows(clip_index, n=10)

    assert _clip(clip_index) == []


@pytest.mark.parametrize(
    "config, scores, kept",
    [
        pytest.param(
            dict(score_gap_threshold=0.0, min_score_clip_thumbnail=0.5,
                 score_cutoff_margin=10.0),
            [0.95, 0.80, 0.60, 0.45, 0.30], 3, id="min_score_clip_thumbnail",
        ),
        pytest.param(
            dict(score_gap_threshold=0.0, score_cutoff_margin=0.10),
            [0.95, 0.90, 0.86, 0.80, 0.60, 0.50], 3, id="score_cutoff_margin",
        ),
        pytest.param(
            dict(),
            [0.95, 0.80, 0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73, 0.72], 1, id="defaults",
        ),
    ],
)
def test_spec_addon_025_precision_mode_keeps_the_thumbnails_the_configured_stages_pass(
    clip_index, monkeypatch, config, scores, kept,
):
    _use_search_config(monkeypatch, **config)
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in scores], "t")
    _scene_rows_nearer_than_everything(clip_index)

    got = _by_id(_clip(clip_index))

    assert set(got) == {f"e_t{i}" for i in range(kept)}


def test_spec_addon_025_recall_mode_uses_the_recall_margin(clip_index, monkeypatch):
    _use_search_config(monkeypatch, score_gap_threshold=0.0, score_cutoff_margin=10.0)
    margin = search_mod._RECALL_PARAMS.score_cutoff_margin
    top = 0.95
    scores = [top, top - margin / 3, top - 2 * margin / 3,
              top - margin - 0.03, top - margin - 0.10, top - margin - 0.20]
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in scores], "t")
    _scene_rows_nearer_than_everything(clip_index)

    got = _by_id(_clip(clip_index, mode="recall"))

    assert set(got) == {"e_t0", "e_t1", "e_t2"}


def _far_thumbnails_of_other_files(index, which, n=10):
    """Thumbnails far below the drive's own, which widen the score distribution if
    the gap and CV checks count them."""
    if which == "other-drive":
        _rows(index, "vec_clip", "clip_thumbnail", _spread(n, 1.15, 1.20), "farother",
              drive=OTHER, axis0=8000)
    else:
        _rows(index, "vec_clip", "clip_thumbnail", _spread(n, 1.15, 1.20), "farinactive",
              active=False, axis0=8500)


GATES_ONLY = dict(score_gap_threshold=0.02, min_score_clip_thumbnail=-1.0,
                  score_cutoff_margin=10.0)


@pytest.mark.parametrize("which", ["other-drive", "inactive"])
def test_spec_addon_025_gap_check_judges_the_drives_active_thumbnails_not_far_rows_of_other_files(
    clip_index, monkeypatch, which,
):
    _use_search_config(monkeypatch, **GATES_ONLY)
    scores = [0.800 + 0.002 * i for i in range(10)]
    assert max(scores) - float(np.mean(scores)) < 0.02
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in scores], "t")
    _far_thumbnails_of_other_files(clip_index, which)

    assert _clip(clip_index) == []


@pytest.mark.parametrize("which", ["other-drive", "inactive"])
def test_spec_addon_025_cv_check_judges_the_drives_active_thumbnails_not_far_rows_of_other_files(
    clip_index, monkeypatch, which,
):
    _use_search_config(monkeypatch, **GATES_ONLY)
    scores = [0.83] + [0.7995 + 0.00002 * i for i in range(39)]
    mean = float(np.mean(scores))
    assert scores[0] - mean > 0.02
    assert float(np.std(scores)) / mean < 0.01
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in scores], "t")
    _far_thumbnails_of_other_files(clip_index, which)

    assert _clip(clip_index) == []


@pytest.mark.parametrize("drive, active", [(OTHER, True), (HOME, False)],
                         ids=["other-drive", "inactive"])
def test_spec_addon_025_a_flat_crowd_of_other_files_nearer_does_not_discard_the_drives_standout(
    clip_index, monkeypatch, drive, active,
):
    _use_search_config(monkeypatch, score_gap_threshold=0.02,
                       min_score_clip_thumbnail=-1.0, score_cutoff_margin=0.10)
    scores = [0.95, 0.80, 0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73, 0.72]
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in scores], "t")
    crowd = [0.970 - 0.00003 * i for i in range(RRF_CANDIDATES + 10)]
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in crowd], "crowd",
          drive=drive, active=active, rows_per_file=10, axis0=9000)

    got = _by_id(_clip(clip_index))

    assert set(got) == {"e_t0"}
    assert got["e_t0"].score == pytest.approx(0.95, abs=1e-4)


def test_spec_addon_025_scene_on_gap_check_counts_every_row_the_knn_returned(
    clip_index, monkeypatch,
):
    """Otherwise-branch of I3: with Scene search on, far scene rows inside the KNN
    widen the distribution, so a flat thumbnail set passes the gap check."""
    _use_search_config(monkeypatch, score_gap_threshold=0.02,
                       min_score_clip_thumbnail=-1.0, min_score_clip=-1.0,
                       score_cutoff_margin=0.10)
    scores = [0.800 + 0.002 * i for i in range(10)]
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in scores], "t")
    _far_scene_rows(clip_index, n=10)

    got = _by_id(search_mod._vector_search_clip(
        unit(clip_index.dim("vec_clip"), 0), RRF_CANDIDATES, mode="precision",
        include_scene_clip=True,
    ))

    assert set(got) == {f"e_t{i}" for i in range(10)}


# ---------------------------------------------------------------------------
# SPEC-ADDON-025 I9 (r5): with Scene search off, the CLIP channel's matches for a
# drive do not depend on other drives' thumbnails
# ---------------------------------------------------------------------------


def _home_layout(mode):
    if mode == "precision":
        return [0.95, 0.93, 0.91, 0.80, 0.75, 0.70, 0.65, 0.60]
    margin = search_mod._RECALL_PARAMS.score_cutoff_margin
    top = 0.95
    return [top, top - margin / 3, top - 2 * margin / 3,
            top - margin - 0.03, top - margin - 0.10, top - margin - 0.20]


OTHER_DRIVE_LAYOUTS = {
    "none": [],
    "one-far-above": [0.99],
    "flat-crowd-nearer": [0.970 - 0.00003 * i for i in range(RRF_CANDIDATES + 10)],
    "interleaved": [0.96, 0.94, 0.92, 0.90, 0.85, 0.78, 0.72, 0.66, 0.62],
    "wide-and-far": _spread(30, 0.40, 0.20),
}


@pytest.mark.parametrize("mode", ["precision", "recall"])
@pytest.mark.parametrize("layout", list(OTHER_DRIVE_LAYOUTS))
def test_spec_addon_025_scene_off_matches_for_a_drive_do_not_depend_on_other_drives_thumbnails(
    clip_index, monkeypatch, layout, mode,
):
    _use_search_config(monkeypatch, score_gap_threshold=0.02,
                       min_score_clip_thumbnail=0.5, score_cutoff_margin=0.05)
    scores = _home_layout(mode)
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in scores], "t")
    others = OTHER_DRIVE_LAYOUTS[layout]
    if others:
        _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in others],
              "o", drive=OTHER, rows_per_file=10, axis0=9500)

    got = _scores(_clip(clip_index, mode=mode))

    assert set(got) == {"e_t0", "e_t1", "e_t2"}
    for i in range(3):
        assert got[f"e_t{i}"] == pytest.approx(scores[i], abs=1e-4)


def test_spec_addon_025_search_with_scene_off_finds_the_drives_thumbnails_behind_a_crowd_of_another_drive(
    clip_index, monkeypatch,
):
    dim = clip_index.dim("vec_clip")
    text_dim = clip_index.dim("vec_text")
    monkeypatch.setattr(search_mod, "embed_text_clip", lambda *_a, **_k: unit(dim, 0))
    monkeypatch.setattr(
        search_mod, "embed_query", lambda *_a, **_k: unit(text_dim, 0),
    )
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in (0.95, 0.94, 0.93)],
          "home")
    _rows(clip_index, "vec_clip", "clip_thumbnail", _spread(20, 0.90, 1.00), "low",
          axis0=300)
    crowd = [0.970 - 0.00003 * i for i in range(RRF_CANDIDATES + 10)]
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in crowd], "crowd",
          drive=OTHER, rows_per_file=10, axis0=9000)

    response = search_mod.search("住宅", drive=HOME)

    ids = {r.file_id if not isinstance(r, dict) else r["file_id"] for r in response.results}
    assert ids == {"home0", "home1", "home2"}


# ---------------------------------------------------------------------------
# SPEC-ADDON-025 I2 / I6: Scene search on and the text channel take their
# candidates from the nearest rows of the whole table, whatever their type
# ---------------------------------------------------------------------------


def _row(index, table, emb_id, etype, d, axis, *, keyed=True):
    file_id = f"f_{emb_id}"
    index.add_file(file_id)
    index.add_embedding(emb_id, file_id, etype, table)
    vector = at_distance(index.dim(table), d, axis)
    if keyed:
        index.put_vector(table, emb_id, vector, partition=etype)
    else:
        index.put_vector(table, emb_id, vector)


def _scores(matches):
    return {m.embedding_id: m.score for m in matches}


def test_spec_addon_025_scene_on_takes_the_nearest_rows_of_every_type_and_returns_thumbnails_and_scenes(
    clip_index, monkeypatch,
):
    _use_search_config(monkeypatch, **PERMISSIVE)
    rows = [
        ("scene_a", "clip", 0.20, True),
        ("unkeyed_thumb", "clip_thumbnail", 0.25, False),
        ("thumb_a", "clip_thumbnail", 0.30, True),
        ("foreign", "unlisted_type", 0.35, True),
        ("scene_b", "clip", 0.40, True),
        ("unkeyed_scene", "clip", 0.45, False),
        ("thumb_b", "clip_thumbnail", 0.50, True),
        ("scene_c", "clip", 0.60, True),
        ("thumb_c", "clip_thumbnail", 0.70, True),
        ("scene_d", "clip", 0.80, True),
        ("thumb_far", "clip_thumbnail", 1.00, True),
        ("scene_far", "clip", 1.10, True),
    ]
    for i, (emb_id, etype, d, keyed) in enumerate(rows):
        _row(clip_index, "vec_clip", emb_id, etype, d, i, keyed=keyed)

    got = _scores(search_mod._vector_search_clip(
        unit(clip_index.dim("vec_clip"), 0), 10, mode="precision", include_scene_clip=True,
    ))

    expected = {e: _score(d) for e, t, d, _ in rows[:10] if t != "unlisted_type"}
    assert set(got) == set(expected)
    for emb_id, score in expected.items():
        assert got[emb_id] == pytest.approx(score, abs=1e-4)


def test_spec_addon_025_text_channel_takes_the_nearest_rows_of_every_type(
    search_index, monkeypatch,  # noqa: F811
):
    _use_search_config(
        monkeypatch, score_gap_threshold=0.0, min_score_text=-1.0, score_cutoff_margin=10.0,
    )
    rows = [
        ("speech_a", "whisper", 0.20, True),
        ("meta_a", "metadata", 0.25, True),
        ("unkeyed_meta", "metadata", 0.30, False),
        ("visual", "video_visual_scene", 0.35, True),
        ("doc_a", "text_content", 0.40, True),
        ("unkeyed_speech", "whisper", 0.45, False),
        ("speech_b", "whisper", 0.50, True),
        ("meta_b", "metadata", 0.60, True),
        ("doc_b", "text_content", 0.70, True),
        ("speech_c", "whisper", 0.80, True),
        ("meta_far", "metadata", 1.00, True),
        ("doc_far", "text_content", 1.10, True),
    ]
    for i, (emb_id, etype, d, keyed) in enumerate(rows):
        _row(search_index, "vec_text", emb_id, etype, d, i, keyed=keyed)

    got = _scores(search_mod._vector_search_text(
        unit(search_index.dim("vec_text"), 0), 10, mode="precision",
    ))

    expected = {e: _score(d) for e, t, d, _ in rows[:10] if t != "video_visual_scene"}
    assert set(got) == set(expected)
    for emb_id, score in expected.items():
        assert got[emb_id] == pytest.approx(score, abs=1e-4)


# ---------------------------------------------------------------------------
# SPEC-ADDON-025 I8: search() keeps the drive and lifecycle filters although the
# CLIP channel's matches span every drive
# ---------------------------------------------------------------------------


def test_spec_addon_025_search_with_scene_off_returns_only_the_drives_active_files(
    clip_index, monkeypatch,
):
    dim = clip_index.dim("vec_clip")
    text_dim = clip_index.dim("vec_text")
    monkeypatch.setattr(search_mod, "embed_text_clip", lambda *_a, **_k: unit(dim, 0))
    monkeypatch.setattr(
        search_mod, "embed_query", lambda *_a, **_k: unit(text_dim, 0),
    )
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in (0.95, 0.94, 0.93)],
          "home")
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(s) for s in (0.955, 0.945)],
          "other", drive=OTHER, axis0=100)
    _rows(clip_index, "vec_clip", "clip_thumbnail", [_distance(0.935)], "inactive",
          active=False, axis0=200)
    _rows(clip_index, "vec_clip", "clip_thumbnail", _spread(20, 0.90, 1.00), "low",
          axis0=300)
    _scene_rows_nearer_than_everything(clip_index)

    response = search_mod.search("住宅", drive=HOME)

    ids = {r.file_id if not isinstance(r, dict) else r["file_id"] for r in response.results}
    assert ids == {"home0", "home1", "home2"}


# ---------------------------------------------------------------------------
# SPEC-ADDON-026 I4 / I5: coarse retrieval shortlists the drive's nearest
# metadata files, however many rows of other types are nearer
# ---------------------------------------------------------------------------


@pytest.fixture()
def coarse_query(search_index, monkeypatch):  # noqa: F811
    dim = search_index.dim("vec_text")
    monkeypatch.setattr(cr_mod, "embed_query", lambda *_a, **_k: unit(dim, 0))
    return search_index


def _whisper_rows_nearer_than_everything(index, n):
    _rows(index, "vec_text", "whisper", _spread(n, 0.01, 0.09), "speech",
          rows_per_file=50, axis0=5000)


def _over_fetch(top_k):
    return max(top_k * 50, 500)


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [1, 3, 10], ids=["top_k-1", "top_k-trims", "top_k-above-files"])
async def test_spec_addon_026_coarse_shortlist_is_the_drives_active_metadata_files_nearest_first(
    coarse_query, top_k,
):
    index = coarse_query
    dim = index.dim("vec_text")
    index.add_many([
        ("a0", HOME, True, "e_a_far", "metadata", "vec_text", at_distance(dim, 0.40, 1)),
        ("a0", HOME, True, "e_a_near", "metadata", "vec_text", at_distance(dim, 0.30, 2)),
    ])
    _rows(index, "vec_text", "metadata", [0.35], "b", axis0=10)
    _rows(index, "vec_text", "metadata", [0.45], "c", axis0=20)
    _rows(index, "vec_text", "metadata", [0.50], "d", axis0=30)
    _rows(index, "vec_text", "metadata", [0.10], "x", drive=OTHER, axis0=40)
    _rows(index, "vec_text", "metadata", [0.12], "y", active=False, axis0=50)
    _whisper_rows_nearer_than_everything(index, _over_fetch(top_k) + 20)

    result = await cr_mod.coarse_retrieve("q", drive=HOME, top_k=top_k)

    files = [("a0", 0.30), ("b0", 0.35), ("c0", 0.45), ("d0", 0.50)]
    expected = files[:top_k]
    assert result.file_ids == tuple(f for f, _ in expected)
    assert list(result.scores) == pytest.approx([_score(d) for _, d in expected], abs=1e-4)
    assert result.top_score == pytest.approx(_score(0.30), abs=1e-4)


@pytest.mark.asyncio
async def test_spec_addon_026_coarse_shortlist_reaches_exactly_the_over_fetch_nearest_metadata_rows(
    coarse_query,
):
    index = coarse_query
    top_k = 5
    k = _over_fetch(top_k)
    _rows(index, "vec_text", "metadata", _spread(k - 1, 0.10, 0.25), "x", drive=OTHER,
          axis0=100)
    _rows(index, "vec_text", "metadata", [0.30], "h", axis0=10)
    _rows(index, "vec_text", "metadata", [0.32], "j", axis0=20)
    _whisper_rows_nearer_than_everything(index, k + 100)

    result = await cr_mod.coarse_retrieve("q", drive=HOME, top_k=top_k)

    assert result.file_ids == ("h0",)


@pytest.mark.asyncio
async def test_spec_addon_026_coarse_shortlist_never_holds_a_metadata_row_without_a_partition_value(
    coarse_query,
):
    index = coarse_query
    _rows(index, "vec_text", "metadata", [0.30, 0.40], "m")
    index.add_file("unkeyed")
    index.add_embedding("e_unkeyed", "unkeyed", "metadata", "vec_text")
    index.put_vector("vec_text", "e_unkeyed", at_distance(index.dim("vec_text"), 0.10, 900))

    result = await cr_mod.coarse_retrieve("q", drive=HOME, top_k=10)

    assert result.file_ids == ("m0", "m1")
