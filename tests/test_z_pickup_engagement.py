"""How much each watched video shapes the Pickup profile.

SPEC-ID: SPEC-ADDON-095
SPEC-ID: SPEC-ADDON-096
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy import text

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.pickup import profile  # noqa: E402
from tests.pickup_accuracy_rig import (  # noqa: E402
    LOFT_MIME,
    CoreDB,
    SearchDB,
    at_angle,
    ts,
    wire,
)

NOW = datetime(2026, 6, 15, 12, 0, 0)
PLAYED = NOW - timedelta(days=1)


@pytest.fixture()
def core(monkeypatch, tmp_path):
    db = CoreDB(tmp_path / "litloft.db")
    wire(monkeypatch, db, SearchDB(tmp_path / "search.db"))
    return db


def _engagements(**kwargs) -> dict[str, float]:
    return {
        w.file_id: w.engagement
        for w in profile.profile_history("a", "v1", now=NOW, **kwargs)
    }


# ---------------------------------------------------------------------------
# Engagement of one history row
# ---------------------------------------------------------------------------

#: (case, file kwargs, watch kwargs, this viewer's segments, other viewer's
#: segments, expected engagement; None means the row does not shape the profile)
_ENGAGEMENT_ROWS = [
    ("under a tenth", {}, {"position": 9.99}, [], [], None),
    ("exactly a tenth", {}, {"position": 10.0}, [], [], 0.2),
    ("a quarter", {}, {"position": 25.0}, [], [], 0.5),
    ("half counts fully", {}, {"position": 50.0}, [], [], 1.0),
    ("to the end", {}, {"position": 100.0}, [], [], 1.0),
    ("position past the duration", {}, {"position": 150.0}, [], [], 1.0),
    ("played seconds summed", {}, {"position": 0.0}, [10.0, 20.0], [], 0.6),
    ("played seconds capped at the duration", {}, {"position": 0.0}, [80.0, 80.0], [], 1.0),
    ("position above played seconds", {}, {"position": 40.0}, [20.0], [], 0.8),
    ("played seconds above position", {}, {"position": 5.0}, [15.0, 15.0], [], 0.6),
    ("another viewer's seconds do not count", {}, {"position": 5.0}, [], [50.0], None),
    ("liked, barely opened", {"liked": True}, {"position": 1.0}, [], [], 1.0),
    ("favourite, barely opened", {"favourite": True}, {"position": 1.0}, [], [], 1.0),
    ("liked, never played", {"liked": True}, {"position": 0.0, "duration": 0.0}, [], [], 1.0),
    (
        "liked beats dismissed", {"liked": True},
        {"position": 1.0, "dismissed_at": PLAYED + timedelta(hours=1)}, [], [], 1.0,
    ),
    (
        "favourite beats dismissed", {"favourite": True},
        {"position": 1.0, "dismissed_at": PLAYED + timedelta(hours=1)}, [], [], 1.0,
    ),
    ("dismissed after the play", {}, {"dismissed_at": PLAYED + timedelta(hours=1)}, [], [], None),
    ("dismissed at the same instant", {}, {"dismissed_at": PLAYED}, [], [], None),
    (
        "a play after the dismissal undoes it", {},
        {"dismissed_at": PLAYED - timedelta(hours=1)}, [], [], 1.0,
    ),
    (
        "dismissal an hour later, written with an offset", {},
        {
            "played_at": PLAYED.strftime("%Y-%m-%d %H:%M:%S"),
            "dismissed_at": PLAYED.strftime("%Y-%m-%dT%H:%M:%S") + "-01:00",
        },
        [], [], None,
    ),
    (
        "dismissal an hour earlier in a T-form offset", {},
        {
            "played_at": PLAYED.strftime("%Y-%m-%d %H:%M:%S"),
            "dismissed_at": PLAYED.strftime("%Y-%m-%dT%H:%M:%S") + "+01:00",
        },
        [], [], 1.0,
    ),
    (
        "a T-form play dismissed later in space form", {},
        {
            "played_at": PLAYED.strftime("%Y-%m-%dT%H:%M:%S") + "+00:00",
            "dismissed_at": (PLAYED + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
        },
        [], [], None,
    ),
    ("an unparseable dismissal is not one", {}, {"dismissed_at": "not-a-date"}, [], [], 1.0),
    ("embed that never reported", {"mime": LOFT_MIME}, {"position": 0.0, "duration": 0.0}, [], [], 0.5),
    ("embed with an infinite duration", {"mime": LOFT_MIME}, {"position": 3.0, "duration": math.inf}, [], [], 0.5),
    ("embed that reported a brief open", {"mime": LOFT_MIME}, {"position": 5.0}, [], [], None),
    ("embed that reported half", {"mime": LOFT_MIME}, {"position": 50.0}, [], [], 1.0),
    ("native video never reported", {}, {"position": 0.0, "duration": 0.0}, [], [], None),
    ("native video with a negative duration", {}, {"position": 3.0, "duration": -5.0}, [], [], None),
    ("native video with an infinite duration", {}, {"position": 3.0, "duration": math.inf}, [], [], None),
]


@pytest.mark.parametrize(
    ("file_kw", "watch_kw", "mine", "theirs", "expected"),
    [row[1:] for row in _ENGAGEMENT_ROWS],
    ids=[row[0] for row in _ENGAGEMENT_ROWS],
)
def test_spec_addon_095_i1_engagement_of_a_video(core, file_kw, watch_kw, mine, theirs, expected):
    """SPEC-ADDON-095 I1: e from likes, favourites, dismissal and the watched fraction."""
    core.add_file("v", **file_kw)
    watch = {"played_at": PLAYED, "position": 100.0, "duration": 100.0, **watch_kw}
    core.watch("v", **watch)
    if mine:
        core.segments("v", mine)
    if theirs:
        core.watch("v", viewer="v2", played_at=PLAYED)
        core.segments("v", theirs, viewer="v2")

    got = _engagements()

    if expected is None:
        assert "v" not in got
        assert profile.watched_file_ids("a", "v1") == {"v"}
    else:
        assert got["v"] == pytest.approx(expected)


def test_spec_addon_095_unweighted_history_gives_every_row_one(core):
    """SPEC-ADDON-095: weighted=False keeps dismissed and brief rows, each at 1."""
    core.add_file("brief")
    core.watch("brief", played_at=PLAYED, position=1.0)
    core.add_file("dismissed")
    core.watch("dismissed", played_at=PLAYED, dismissed_at=PLAYED + timedelta(hours=1))
    core.add_file("half")
    core.watch("half", played_at=PLAYED, position=25.0)

    assert _engagements(weighted=False) == {"brief": 1.0, "dismissed": 1.0, "half": 1.0}


# ---------------------------------------------------------------------------
# Which kinds of file the history holds
# ---------------------------------------------------------------------------


def test_spec_addon_096_the_profile_history_holds_live_videos_only(core):
    """SPEC-ADDON-096: documents, images, audio and dead videos are not profiled."""
    for file_id, kw in (
        ("video", {}),
        ("embed", {"mime": LOFT_MIME}),
        ("doc", {"kind": "document"}),
        ("image", {"kind": "image"}),
        ("audio", {"kind": "audio"}),
        ("trashed", {"deleted": True}),
        ("missing", {"missing": True}),
    ):
        core.add_file(file_id, **kw)
        core.watch(file_id, played_at=PLAYED)

    assert set(_engagements()) == {"video", "embed"}
    assert profile.watched_file_ids("a", "v1") == {
        "video", "embed", "doc", "image", "audio", "trashed", "missing",
    }


def test_spec_addon_096_every_kind_when_asked_with_other_kinds_at_one(core):
    """SPEC-ADDON-096: videos_only=False keeps live files of every kind; non-videos get e=1."""
    core.add_file("doc", kind="document")
    core.watch("doc", played_at=PLAYED, position=0.0, duration=0.0)
    core.add_file("brief")
    core.watch("brief", played_at=PLAYED, position=1.0)
    core.add_file("half")
    core.watch("half", played_at=PLAYED, position=25.0)
    core.add_file("trashed_doc", kind="document", deleted=True)
    core.watch("trashed_doc", played_at=PLAYED)

    assert _engagements(videos_only=False) == {"doc": 1.0, "half": 0.5}


def test_spec_addon_096_the_drive_boundary_holds_for_videos(core):
    """SPEC-ADDON-096: another drive's videos are not read."""
    core.add_file("mine")
    core.watch("mine", played_at=PLAYED)
    core.add_file("theirs", drive="b")
    core.watch("theirs", played_at=PLAYED)

    assert set(_engagements()) == {"mine"}


# ---------------------------------------------------------------------------
# Filters run before the cap, at a given reference time
# ---------------------------------------------------------------------------


def test_spec_addon_095_i2_dropped_rows_do_not_use_up_the_cap(core):
    """SPEC-ADDON-095 I2: newer documents, brief opens and bad rows never push a kept row out."""
    for i in range(4):
        core.add_file(f"keep{i}")
        core.watch(f"keep{i}", played_at=NOW - timedelta(days=5 + i))
    for i in range(3):
        core.add_file(f"doc{i}", kind="document")
        core.watch(f"doc{i}", played_at=NOW - timedelta(days=1 + i * 0.1))
    core.add_file("brief")
    core.watch("brief", played_at=NOW - timedelta(hours=2), position=1.0)
    core.add_file("dismissed")
    core.watch("dismissed", played_at=NOW - timedelta(hours=3),
               dismissed_at=NOW - timedelta(hours=1))
    core.add_file("bad")
    core.watch("bad", played_at="not-a-date")
    core.add_file("future")
    core.watch("future", played_at=NOW + timedelta(days=5))

    got = profile.profile_history("a", "v1", cap=3, now=NOW)

    assert [w.file_id for w in got] == ["keep0", "keep1", "keep2"]


def test_spec_addon_095_i2_more_than_the_default_cap_of_kept_rows(core):
    """SPEC-ADDON-095 I2: with over 2000 kept rows, newer dropped rows do not change the history."""
    rows = []
    for i in range(profile.PROFILE_VECTOR_CAP + 5):
        rows.append({"id": f"k{i:05d}", "t": ts(NOW - timedelta(days=1, minutes=i))})
    with core.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO files (id, drive, file_path, file_type, mime_type,"
                " created_at, updated_at) VALUES (:id, 'a', :id, 'video',"
                " 'video/mp4', '2025-01-01 00:00:00', '2025-01-01 00:00:00')"
            ),
            rows,
        )
        conn.execute(
            text(
                "INSERT INTO watch_history (viewer_id, file_id, playback_position,"
                " duration, last_played_at) VALUES ('v1', :id, 100, 100, :t)"
            ),
            rows,
        )
    before = [(w.file_id, w.engagement) for w in profile.profile_history("a", "v1", now=NOW)]

    for i in range(20):
        core.add_file(f"doc{i}", kind="document")
        core.watch(f"doc{i}", played_at=NOW - timedelta(hours=1, minutes=i))
        core.add_file(f"brief{i}")
        core.watch(f"brief{i}", played_at=NOW - timedelta(hours=2, minutes=i), position=1.0)
        core.add_file(f"old{i}", kind="document")
        core.watch(f"old{i}", played_at=NOW - timedelta(days=400 + i))
    after = [(w.file_id, w.engagement) for w in profile.profile_history("a", "v1", now=NOW)]

    assert len(before) == profile.PROFILE_VECTOR_CAP
    assert before[0][0] == "k00000"
    assert after == before


def test_spec_addon_095_the_window_is_measured_from_the_reference_time(core):
    """SPEC-ADDON-095: 365 days back, 1 day ahead, and strictly before until."""
    ref = NOW - timedelta(days=500)
    for file_id, played in (
        ("in_window", ref - timedelta(days=364)),
        ("past_window", ref - timedelta(days=366)),
        ("slight_skew", ref + timedelta(hours=12)),
        ("far_future", ref + timedelta(days=2)),
    ):
        core.add_file(file_id)
        core.watch(file_id, played_at=played)

    got = profile.profile_history("a", "v1", now=ref)

    assert [w.file_id for w in got] == ["slight_skew", "in_window"]


def test_spec_addon_095_until_is_an_exclusive_upper_bound(core):
    """SPEC-ADDON-095: rows at or after until are not read."""
    cut = NOW - timedelta(days=10)
    for file_id, played in (
        ("before", cut - timedelta(seconds=1)),
        ("at", cut),
        ("after", cut + timedelta(hours=1)),
    ):
        core.add_file(file_id)
        core.watch(file_id, played_at=played)

    got = profile.profile_history("a", "v1", now=cut, until=cut)

    assert [w.file_id for w in got] == ["before"]


# ---------------------------------------------------------------------------
# Ages and weights
# ---------------------------------------------------------------------------


def test_spec_addon_095_age_is_measured_from_the_reference_time():
    """SPEC-ADDON-095: age_days(..., now=) uses the given instant."""
    played = datetime(2026, 1, 1, 0, 0, 0)

    assert profile.age_days(played, now=played + timedelta(days=2.5)) == pytest.approx(2.5)


def test_spec_addon_095_raw_weight_multiplies_decay_by_engagement():
    """SPEC-ADDON-095 I1: raw_weight = log1p(sum(w * decay(a)))."""
    hl = profile.HALF_LIFE_DAYS

    assert profile.raw_weight([0.0, hl], [0.5, 1.0]) == pytest.approx(math.log1p(1.0))
    assert profile.raw_weight([0.0, 0.0], [0.25, 0.25]) == pytest.approx(math.log1p(0.5))
    assert profile.raw_weight([0.0, 0.0]) == pytest.approx(math.log1p(2.0))


# ---------------------------------------------------------------------------
# build_lanes weights every mean by engagement
# ---------------------------------------------------------------------------


def _source(vectors: dict[str, np.ndarray]):
    def lookup(file_ids, channel):
        if channel != "clip_thumbnail":
            return {}
        return {f: vectors[f] for f in file_ids if f in vectors}

    return lookup


def test_spec_addon_095_i1_the_lane_centroid_is_the_engagement_weighted_mean():
    """SPEC-ADDON-095 I1: centroid = normalize(sum(e * v)), raw weight = log1p(sum(e * decay))."""
    u, w = at_angle(0), at_angle(90)
    history = [
        profile.WatchedFile("a", NOW, 1.0),
        profile.WatchedFile("b", NOW, 0.2),
        profile.WatchedFile("c", NOW, 0.2),
    ]

    lanes = profile.build_lanes(
        history, key="a:v1", now=NOW, vectors=_source({"a": u, "b": w, "c": w}),
    )

    expected = u + 0.4 * w
    expected = expected / np.linalg.norm(expected)
    assert len(lanes) == 1
    assert np.allclose(lanes[0].centroid, expected, atol=1e-5)
    assert lanes[0].member_count == 3
    assert lanes[0].raw_weight == pytest.approx(math.log1p(1.4))


def test_spec_addon_095_i1_ages_in_the_raw_weight_come_from_the_reference_time():
    """SPEC-ADDON-095 I1: decay is measured from now, scaled by e."""
    u = at_angle(0)
    hl = profile.HALF_LIFE_DAYS
    history = [
        profile.WatchedFile("a", NOW - timedelta(days=hl), 1.0),
        profile.WatchedFile("b", NOW, 0.5),
        profile.WatchedFile("c", NOW, 0.5),
    ]

    lanes = profile.build_lanes(
        history, key="a:v1", now=NOW, vectors=_source({"a": u, "b": u, "c": u}),
    )

    assert len(lanes) == 1
    assert lanes[0].raw_weight == pytest.approx(math.log1p(1.5))


def test_spec_addon_095_i1_the_directionless_check_uses_the_weighted_mean():
    """SPEC-ADDON-095 I1: opposite members cancel only when their engagements do."""
    u = at_angle(0)
    history = [
        profile.WatchedFile("a1", NOW, 1.0),
        profile.WatchedFile("a2", NOW, 1.0),
        profile.WatchedFile("b1", NOW, 0.2),
        profile.WatchedFile("b2", NOW, 0.2),
    ]

    lanes = profile.build_lanes(
        history, key="a:v1", now=NOW,
        vectors=_source({"a1": u, "a2": u, "b1": -u, "b2": -u}),
    )

    assert len(lanes) == 1
    assert np.allclose(lanes[0].centroid, u, atol=1e-5)


def test_spec_addon_095_i2_a_document_or_unengaged_row_leaves_the_lanes_alone(core, monkeypatch):
    """SPEC-ADDON-095 I2: such a row changes only the exclusion set."""
    vectors = {}
    for i in range(5):
        core.add_file(f"keep{i}")
        core.watch(f"keep{i}", played_at=NOW - timedelta(days=2 + i))
        vectors[f"keep{i}"] = at_angle(i * 3.0)

    def lanes():
        return [
            (l.channel, l.cluster_id, l.member_count, round(l.raw_weight, 9),
             round(l.weight, 9), tuple(np.round(l.centroid, 6)))
            for l in profile.build_lanes(
                profile.profile_history("a", "v1", now=NOW),
                key="a:v1", now=NOW, vectors=_source(vectors),
            )
        ]

    base = lanes()
    core.add_file("doc", kind="document")
    core.watch("doc", played_at=NOW - timedelta(hours=1))
    vectors["doc"] = at_angle(90)
    core.add_file("brief")
    core.watch("brief", played_at=NOW - timedelta(hours=2), position=2.0)
    vectors["brief"] = at_angle(80)
    core.add_file("dismissed")
    core.watch("dismissed", played_at=NOW - timedelta(hours=3),
               dismissed_at=NOW - timedelta(hours=2))
    vectors["dismissed"] = at_angle(70)

    assert base
    assert lanes() == base
    assert {"doc", "brief", "dismissed"} <= profile.watched_file_ids("a", "v1")
