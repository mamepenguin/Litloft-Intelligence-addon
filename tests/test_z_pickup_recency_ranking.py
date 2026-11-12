"""Freshness, and the lane ranking it feeds: similarity x freshness.

SPEC-ID: SPEC-ADDON-102
"""

from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.pickup import feed  # noqa: E402
from app.pickup import retrieval  # noqa: E402
from app.pickup.profile import Lane  # noqa: E402
from tests.pickup_accuracy_rig import CoreDB, SearchDB, at_angle, ts, wire  # noqa: E402

NOW = datetime(2026, 11, 1, 12, 0, 0)


@pytest.fixture()
def fresh():
    return importlib.import_module("app.pickup.freshness")


def _days(n: float) -> timedelta:
    return timedelta(days=n)


# ---------------------------------------------------------------------------
# freshness()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("half_life", "floor", "age_days", "expected"),
    [
        (30.0, 0.5, 0.0, 1.0),
        (30.0, 0.5, 30.0, 0.75),
        (30.0, 0.5, 60.0, 0.625),
        (7.0, 0.0, 7.0, 0.5),
        (7.0, 0.0, 14.0, 0.25),
        (90.0, 0.0, 45.0, 0.5 ** 0.5),
    ],
    ids=["age 0", "one half-life", "two half-lives", "floor 0, one half-life",
         "floor 0, two half-lives", "half a half-life"],
)
def test_spec_addon_102_freshness_follows_the_formula(fresh, half_life, floor, age_days, expected):
    """SPEC-ADDON-102: FLOOR + (1 - FLOOR) * 0.5 ** (age / HALF_LIFE_DAYS)."""
    got = fresh.freshness(NOW - _days(age_days), NOW, half_life_days=half_life, floor=floor)

    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "ahead",
    [timedelta(seconds=1), timedelta(hours=12), timedelta(days=1)],
    ids=["one second ahead", "half a day ahead", "exactly one day ahead"],
)
def test_spec_addon_102_i3_up_to_a_day_in_the_future_is_age_zero(fresh, ahead):
    """SPEC-ADDON-102 I3: a created_at up to one day after now has freshness 1."""
    assert fresh.freshness(NOW + ahead, NOW, half_life_days=7.0, floor=0.5) == 1.0


@pytest.mark.parametrize(
    "created",
    [None, NOW + timedelta(days=1, seconds=1), NOW + _days(400)],
    ids=["unparseable", "just over a day ahead", "far in the future"],
)
@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_spec_addon_102_i3_unknown_or_future_dates_get_the_floor(fresh, created, floor):
    """SPEC-ADDON-102 I3: freshness exactly FLOOR, the value of an infinitely old file."""
    assert fresh.freshness(created, NOW, half_life_days=30.0, floor=floor) == floor


@pytest.mark.parametrize(
    "created",
    [None, NOW, NOW - _days(10_000), NOW + _days(5), NOW + timedelta(hours=3)],
    ids=["unparseable", "now", "ancient", "future", "skew"],
)
def test_spec_addon_102_floor_one_is_freshness_one_everywhere(fresh, created):
    """SPEC-ADDON-102: with floor = 1 it returns 1.0 for every input."""
    assert fresh.freshness(created, NOW, half_life_days=7.0, floor=1.0) == 1.0


def test_spec_addon_102_i2_freshness_falls_strictly_with_age(fresh):
    """SPEC-ADDON-102 I2: a later created_at is never less fresh, and is fresher while above FLOOR."""
    ages = [0.0, 0.5, 1.0, 10.0, 30.0, 100.0, 365.0]
    values = [
        fresh.freshness(NOW - _days(a), NOW, half_life_days=30.0, floor=0.5) for a in ages
    ]

    assert all(a > b for a, b in zip(values, values[1:])), values


def test_spec_addon_102_very_old_files_tie_on_the_floor(fresh):
    """SPEC-ADDON-102: about 53 half-lives on, freshness equals FLOOR exactly."""
    sixty = fresh.freshness(NOW - _days(60 * 7), NOW, half_life_days=7.0, floor=0.5)
    seventy = fresh.freshness(NOW - _days(70 * 7), NOW, half_life_days=7.0, floor=0.5)

    assert sixty == seventy == 0.5


def test_spec_addon_102_the_defaults_are_the_module_constants(fresh):
    """SPEC-ADDON-102: HALF_LIFE_DAYS and FLOOR are what an unparameterised call uses."""
    created = NOW - _days(45)

    assert fresh.freshness(created, NOW) == fresh.freshness(
        created, NOW, half_life_days=fresh.HALF_LIFE_DAYS, floor=fresh.FLOOR,
    )


def test_spec_addon_102_reference_now_is_naive_utc(fresh):
    """SPEC-ADDON-102: reference_now() is naive UTC now."""
    got = fresh.reference_now()
    utc = datetime.now(UTC).replace(tzinfo=None)

    assert got.tzinfo is None
    assert abs((got - utc).total_seconds()) < 5


# ---------------------------------------------------------------------------
# score_lanes()
# ---------------------------------------------------------------------------

DIM = 8


def _centroid() -> np.ndarray:
    return at_angle(0.0)


def _cos_vector(cos: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = cos
    v[1] = float(np.sqrt(max(0.0, 1.0 - cos * cos)))
    return v


def _candidates(rows: list[tuple[str, float]]) -> retrieval.CandidateSet:
    """Candidates in the given matrix order, each at the given cosine to the centroid."""
    return retrieval.CandidateSet(
        channel="clip_thumbnail", drive="a",
        file_ids=tuple(f for f, _ in rows),
        matrix=np.stack([_cos_vector(c) for _, c in rows]).astype(np.float32),
    )


def _score(rows, freshness=..., *, exclude=(), limit=120, centroids=None):
    kwargs = {} if freshness is ... else {"freshness": freshness}
    return retrieval.score_lanes(
        _candidates(rows),
        centroids if centroids is not None else [_centroid()],
        channel="clip_thumbnail",
        exclude_file_ids=set(exclude),
        limit=limit,
        **kwargs,
    )


def test_spec_addon_102_lanes_are_ranked_by_similarity_times_freshness():
    """SPEC-ADDON-102: a closer but staler candidate yields to a fresher one."""
    lane = _score(
        [("old", 0.95), ("new", 0.70), ("mid", 0.80)],
        freshness={"old": 0.2, "new": 1.0, "mid": 0.5},
    )[0]

    assert [f for f, _, _ in lane] == ["new", "mid", "old"]
    assert [r for _, _, r in lane] == pytest.approx([0.70, 0.40, 0.19], rel=1e-6)


def test_spec_addon_102_a_lane_entry_is_file_similarity_and_ranking_score():
    """SPEC-ADDON-102: similarity is float(float32), ranking_score is their float64 product."""
    lane = _score([("a", 0.8), ("b", 0.6)], freshness={"a": 0.3, "b": 0.9})[0]

    for file_id, similarity, ranking in lane:
        assert type(similarity) is float
        assert similarity == float(np.float32(similarity))
        expected_fresh = {"a": 0.3, "b": 0.9}[file_id]
        assert ranking == similarity * expected_fresh
    assert {f: s for f, s, _ in lane} == pytest.approx({"a": 0.8, "b": 0.6}, abs=1e-6)


@pytest.mark.parametrize(
    "freshness",
    [..., None],
    ids=["omitted", "None"],
)
def test_spec_addon_102_i4_no_freshness_ranks_by_similarity_alone(freshness):
    """SPEC-ADDON-102 I4: freshness 1 everywhere, so the ranking score is the similarity exactly."""
    lane = _score([("mid", 0.7), ("best", 0.9), ("worst", 0.5)], freshness)[0]

    assert [f for f, _, _ in lane] == ["best", "mid", "worst"]
    assert all(r == s for _, s, r in lane)


def test_spec_addon_102_i4_freshness_one_everywhere_equals_no_freshness():
    """SPEC-ADDON-102 I4: an all-1.0 mapping gives the same triples as None."""
    rows = [("c", 0.7), ("a", 0.9), ("d", 0.5), ("b", 0.8)]

    with_ones = _score(rows, {f: 1.0 for f, _ in rows})
    without = _score(rows, None)

    assert with_ones == without


def test_spec_addon_102_a_file_absent_from_the_mapping_gets_the_floor(fresh):
    """SPEC-ADDON-102: a file missing from a given mapping is scored at FLOOR."""
    lane = _score([("known", 0.6), ("unknown", 0.6)], {"known": 1.0})[0]
    by_id = {f: (s, r) for f, s, r in lane}

    assert by_id["unknown"][1] == pytest.approx(by_id["unknown"][0] * fresh.FLOOR, rel=1e-12)
    assert by_id["known"][1] == by_id["known"][0]


def test_spec_addon_102_an_empty_mapping_is_not_no_mapping(fresh):
    """SPEC-ADDON-102: {} puts every file at FLOOR, where None puts every file at 1.0."""
    lane = _score([("a", 0.8), ("b", 0.6)], {})[0]

    for _, similarity, ranking in lane:
        assert ranking == pytest.approx(similarity * fresh.FLOOR, rel=1e-12)


def test_spec_addon_102_i1_qualification_is_on_similarity_alone():
    """SPEC-ADDON-102 I1: [0.45, 0.999) on similarity, whatever the freshness; opened files out."""
    rows = [
        ("weak_but_new", 0.44),
        ("floor_and_ancient", 0.46),
        ("good_and_dead", 0.80),
        ("duplicate", 0.9995),
        ("opened", 0.85),
        ("plain", 0.60),
    ]
    freshness = {
        "weak_but_new": 1.0, "floor_and_ancient": 0.0, "good_and_dead": 0.0,
        "duplicate": 1.0, "opened": 1.0, "plain": 0.5,
    }

    lane = _score(rows, freshness, exclude={"opened"})[0]

    assert {f for f, _, _ in lane} == {"floor_and_ancient", "good_and_dead", "plain"}


def test_spec_addon_102_the_per_lane_cut_keeps_the_best_by_ranking_score():
    """SPEC-ADDON-102: the lane keeps its best `limit` by ranking score, not by similarity."""
    rows = [(f"old{i}", 0.95 - i * 0.001) for i in range(5)] + [("new", 0.50)]
    freshness = {f: 0.1 for f, _ in rows} | {"new": 1.0}

    lane = _score(rows, freshness, limit=1)[0]

    assert [f for f, _, _ in lane] == ["new"]


def test_spec_addon_102_full_ties_fall_to_file_id_including_at_the_cut():
    """SPEC-ADDON-102: equal similarity and freshness are ordered by file id, not matrix row."""
    rows = [("c", 0.7), ("b", 0.7), ("a", 0.7), ("d", 0.7)]
    freshness = {f: 0.5 for f, _ in rows}

    assert [f for f, _, _ in _score(rows, freshness)[0]] == ["a", "b", "c", "d"]
    assert [f for f, _, _ in _score(rows, freshness, limit=2)[0]] == ["a", "b"]


def test_spec_addon_102_i2_of_equal_similarity_the_fresher_ranks_higher():
    """SPEC-ADDON-102 I2: identical vectors, the later created_at first, whatever the file ids."""
    created = {"a_older": NOW - _days(40), "z_newer": NOW - _days(2)}
    fresh = importlib.import_module("app.pickup.freshness")
    freshness = {
        f: fresh.freshness(c, NOW, half_life_days=30.0, floor=0.5) for f, c in created.items()
    }

    lane = _score([("a_older", 0.7), ("z_newer", 0.7)], freshness)[0]

    assert [f for f, _, _ in lane] == ["z_newer", "a_older"]


def test_spec_addon_102_i3_an_unknown_date_never_outranks_a_fresher_equal(fresh):
    """SPEC-ADDON-102 I3: unknown created_at is at FLOOR, below any equal-similarity file above it."""
    freshness = {
        "a_unknown": fresh.freshness(None, NOW, half_life_days=30.0, floor=0.5),
        "z_old": fresh.freshness(NOW - _days(300), NOW, half_life_days=30.0, floor=0.5),
    }

    lane = _score([("a_unknown", 0.7), ("z_old", 0.7)], freshness)[0]

    assert [f for f, _, _ in lane] == ["z_old", "a_unknown"]


def test_spec_addon_102_zero_ranking_scores_still_qualify_and_order_by_similarity():
    """SPEC-ADDON-102: at FLOOR = 0 a decayed candidate stays, ordered by similarity then id."""
    rows = [("b_tie", 0.6), ("low", 0.5), ("a_tie", 0.6), ("high", 0.9), ("live", 0.46)]
    freshness = {f: 0.0 for f, _ in rows} | {"live": 1.0}

    lane = _score(rows, freshness)[0]

    assert [f for f, _, _ in lane] == ["live", "high", "a_tie", "b_tie", "low"]
    assert [r for _, _, r in lane][1:] == [0.0, 0.0, 0.0, 0.0]


def test_spec_addon_102_every_lane_uses_the_same_freshness():
    """SPEC-ADDON-102: freshness is per file, not per lane."""
    rows = [("x", 0.8), ("y", 0.7)]

    lanes = _score(rows, {"x": 0.1, "y": 1.0}, centroids=[_centroid(), _centroid()])

    assert [f for f, _, _ in lanes[0]] == ["y", "x"]
    assert lanes[0] == lanes[1]


# ---------------------------------------------------------------------------
# interleave()
# ---------------------------------------------------------------------------


def _lane(cluster_id: str, weight: float = 1.0) -> Lane:
    return Lane(
        channel="clip_thumbnail", cluster_id=cluster_id,
        centroid=np.zeros(4, dtype=np.float32), member_count=1,
        raw_weight=weight, weight=weight,
    )


def test_spec_addon_102_i5_a_shared_file_goes_to_its_most_similar_lane_even_at_zero():
    """SPEC-ADDON-102 I5: credited by similarity, even when FLOOR = 0 decayed every score to 0."""
    lanes = [_lane("first"), _lane("second")]
    scored = {
        "first": [("f", 0.50, 0.0)],
        "second": [("f", 0.90, 0.0)],
    }

    items = feed.interleave(lanes, scored, depth=10)

    assert [(i.file_id, i.cluster_id) for i in items] == [("f", "second")]


def test_spec_addon_102_i5_a_similarity_tie_goes_to_the_lane_listed_first():
    """SPEC-ADDON-102 I5: equal similarity, the first lane wins."""
    lanes = [_lane("first"), _lane("second")]
    scored = {
        "first": [("f", 0.70, 0.35)],
        "second": [("f", 0.70, 0.35)],
    }

    items = feed.interleave(lanes, scored, depth=10)

    assert [(i.file_id, i.cluster_id) for i in items] == [("f", "first")]


def test_spec_addon_102_survivors_are_sorted_in_ranking_order():
    """SPEC-ADDON-102: ranking score, then similarity, then file id, whatever the input order."""
    lanes = [_lane("a")]
    scored = {"a": [
        ("stale", 0.95, 0.10),
        ("q_lowsim", 0.60, 0.40),
        ("z_tie", 0.50, 0.50),
        ("p_highsim", 0.80, 0.40),
        ("y_tie", 0.50, 0.50),
    ]}

    items = feed.interleave(lanes, scored, depth=10)

    assert [i.file_id for i in items] == ["y_tie", "z_tie", "p_highsim", "q_lowsim", "stale"]
    assert [i.rank for i in items] == [1, 2, 3, 4, 5]


def test_spec_addon_102_a_feed_item_carries_ranking_score_and_similarity():
    """SPEC-ADDON-102: FeedItem.score is the ranking score; FeedItem.similarity is the similarity."""
    items = feed.interleave([_lane("a")], {"a": [("f", 0.8, 0.4)]}, depth=5)

    assert items[0].score == 0.4
    assert items[0].similarity == 0.8


# ---------------------------------------------------------------------------
# live_videos()
# ---------------------------------------------------------------------------


def test_spec_addon_102_live_videos_maps_each_live_video_to_its_parsed_created_at(monkeypatch, tmp_path):
    """SPEC-ADDON-102: naive UTC, aware values converted, unparseable values None; live videos only."""
    from app.pickup import profile

    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    wire(monkeypatch, core, search)
    core.add_file("naive", created_at=ts(datetime(2026, 5, 1, 8, 30)))
    core.add_file("aware", created_at="2026-05-01T09:00:00+09:00")
    core.add_file("bad", created_at="not-a-date")
    core.add_file("doc", kind="document", created_at=ts(datetime(2026, 5, 1)))
    core.add_file("trashed", deleted=True, created_at=ts(datetime(2026, 5, 1)))
    core.add_file("gone", missing=True, created_at=ts(datetime(2026, 5, 1)))
    core.add_file("elsewhere", drive="b", created_at=ts(datetime(2026, 5, 1)))

    got = profile.live_videos("a")

    assert got == {
        "naive": datetime(2026, 5, 1, 8, 30),
        "aware": datetime(2026, 5, 1, 0, 0),
        "bad": None,
    }
    assert all(v is None or v.tzinfo is None for v in got.values())
