"""The Pickup worker: what it reads, how often, and per whom.

Two things here are regressions waiting to happen, and both were real
defects in the shape this replaced.

The candidate matrices do not depend on the viewer, so they are built
once per drive and scored against by everyone. Moving that inside the
viewer loop would multiply the index reads by the number of viewers
without changing a single result.

The checkpoint is computed over one viewer's own watched set. The
previous version hashed the first twenty ids of a drive-wide list
ordered by recency, so a second viewer's checkpoint barely moved when
their own history did.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.models import Base, PickupItem, PickupProfile  # noqa: E402
from app.pickup.profile import Lane  # noqa: E402
from app.pickup.retrieval import CandidateSet  # noqa: E402
from app.workers import pickup as worker_mod  # noqa: E402


@pytest.fixture()
def rig(monkeypatch, tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_cm():
        s = maker()
        try:
            yield s
        finally:
            s.close()

    monkeypatch.setattr(worker_mod, "get_search_db", session_cm)
    monkeypatch.setattr(worker_mod, "get_search_db_read", session_cm)

    state = {
        "viewers": ["v1"],
        "watched": {"v1": {"seen1"}},
        "history": {"v1": ["seen1"]},
        "lanes": [Lane(
            channel="clip_thumbnail", cluster_id="clip_thumbnail:0",
            centroid=np.zeros(4, dtype=np.float32), member_count=3,
            raw_weight=1.0, weight=1.0,
        )],
        "loads": [],
        "scores": [],
        "played": {},
    }

    monkeypatch.setattr(worker_mod.profile_mod, "viewer_ids",
                        lambda drive: list(state["viewers"]))
    monkeypatch.setattr(worker_mod.profile_mod, "watched_file_ids",
                        lambda drive, v: set(state["watched"].get(v, ())))
    monkeypatch.setattr(
        worker_mod.profile_mod, "watch_signature",
        lambda drive, v: [
            (f, state["played"].get((v, f), "2026-08-01 00:00:00"))
            for f in sorted(state["watched"].get(v, ()))
        ],
    )
    monkeypatch.setattr(worker_mod.profile_mod, "profile_history",
                        lambda drive, v: list(state["history"].get(v, ())))
    monkeypatch.setattr(worker_mod.profile_mod, "build_lanes",
                        lambda history, key: list(state["lanes"]))
    monkeypatch.setattr(worker_mod.profile_mod, "CHANNELS",
                        ("clip_thumbnail", "tfidf_keywords"))

    def fake_load(*, drive, channel):
        state["loads"].append((drive, channel))
        if channel in state.get("failing_channels", ()):
            raise RuntimeError("index unavailable")
        if channel != "clip_thumbnail":
            return CandidateSet(channel=channel, drive=drive, file_ids=(),
                                matrix=np.zeros((0, 0), dtype=np.float32))
        return CandidateSet(
            channel=channel, drive=drive,
            file_ids=("c1", "c2", "c3"),
            matrix=np.eye(3, 4, dtype=np.float32),
        )

    def fake_score(candidates, centroids, *, channel, exclude_file_ids, limit):
        assert channel == candidates.channel, (
            f"{channel} lanes scored against {candidates.channel} rows"
        )
        state["scores"].append(set(exclude_file_ids))
        hits = [(f, 0.9 - i * 0.01)
                for i, f in enumerate(candidates.file_ids)
                if f not in exclude_file_ids]
        return [list(hits[:limit]) for _ in centroids]

    monkeypatch.setattr(worker_mod, "load_candidates", fake_load)
    monkeypatch.setattr(worker_mod, "score_lanes", fake_score)
    state["orig_score"] = fake_score

    def rows(drive="a", viewer="v1"):
        s = maker()
        try:
            return (
                s.query(PickupItem)
                .filter_by(drive_id=drive, viewer_id=viewer)
                .order_by(PickupItem.rank).all()
            )
        finally:
            s.close()

    def header(drive="a", viewer="v1"):
        s = maker()
        try:
            return s.query(PickupProfile).filter_by(
                drive_id=drive, viewer_id=viewer).first()
        finally:
            s.close()

    state["rows"] = rows
    state["header"] = header
    return state


async def test_a_feed_is_built_and_stored(rig):
    await worker_mod.PickupWorker()._compute_for_drive("a")

    got = rig["rows"]()
    assert [r.file_id for r in got] == ["c1", "c2", "c3"]
    assert [r.rank for r in got] == [1, 2, 3]
    assert rig["header"]().total == 3
    assert got[0].cluster_id == "clip_thumbnail:0"


async def test_watched_files_are_passed_as_the_exclusion(rig):
    rig["watched"]["v1"] = {"c1", "seen1"}

    await worker_mod.PickupWorker()._compute_for_drive("a")

    assert rig["scores"][0] == {"c1", "seen1"}
    assert [r.file_id for r in rig["rows"]()] == ["c2", "c3"]


async def test_candidates_are_loaded_once_per_drive_not_per_viewer(rig):
    rig["viewers"] = ["v1", "v2", "v3"]
    rig["watched"] = {v: {f"seen-{v}"} for v in rig["viewers"]}
    rig["history"] = {v: [f"seen-{v}"] for v in rig["viewers"]}

    await worker_mod.PickupWorker()._compute_for_drive("a")

    assert rig["loads"] == [("a", "clip_thumbnail"), ("a", "tfidf_keywords")]
    assert len(rig["scores"]) == 3


async def test_an_unchanged_history_skips_recomputation(rig):
    w = worker_mod.PickupWorker()
    await w._compute_for_drive("a")
    before = len(rig["scores"])

    await w._compute_for_drive("a")

    assert len(rig["scores"]) == before


async def test_a_changed_history_recomputes(rig):
    w = worker_mod.PickupWorker()
    await w._compute_for_drive("a")
    before = len(rig["scores"])
    rig["watched"]["v1"] = {"seen1", "seen2"}

    await w._compute_for_drive("a")

    assert len(rig["scores"]) == before + 1


async def test_one_viewers_activity_does_not_move_anothers_checkpoint(rig):
    """The regression test for hashing a drive-wide recency slice."""
    rig["viewers"] = ["quiet", "loud"]
    rig["watched"] = {"quiet": {"q1"}, "loud": {"l1"}}
    rig["history"] = {"quiet": ["q1"], "loud": ["l1"]}
    w = worker_mod.PickupWorker()
    await w._compute_for_drive("a")
    quiet_before = rig["header"](viewer="quiet").watch_history_checkpoint

    rig["watched"]["loud"] = {f"l{i}" for i in range(200)}
    await w._compute_for_drive("a")

    assert rig["header"](viewer="quiet").watch_history_checkpoint == quiet_before


async def test_a_viewer_with_no_history_gets_no_feed(rig):
    rig["watched"]["v1"] = set()

    await worker_mod.PickupWorker()._compute_for_drive("a")

    assert rig["rows"]() == []
    assert rig["header"]() is None


async def test_a_rebuild_replaces_rows_rather_than_layering_them(rig):
    """Ranks are positional; two rankings interleaved would be nonsense."""
    w = worker_mod.PickupWorker()
    await w._compute_for_drive("a")
    rig["watched"]["v1"] = {"seen1", "c1"}

    await w._compute_for_drive("a")

    got = rig["rows"]()
    assert [r.file_id for r in got] == ["c2", "c3"]
    assert [r.rank for r in got] == [1, 2]
    assert rig["header"]().total == 2


async def test_an_empty_channel_is_skipped_without_failing(rig):
    """``tfidf_keywords`` has no rows in this rig and must not be scored."""
    await worker_mod.PickupWorker()._compute_for_drive("a")

    assert len(rig["scores"]) == 1


async def test_a_drive_with_no_viewers_does_nothing(rig):
    rig["viewers"] = []

    await worker_mod.PickupWorker()._compute_for_drive("a")

    assert rig["loads"] == []


# ---------------------------------------------------------------------------
# A transient failure must not settle into a permanent empty feed
# ---------------------------------------------------------------------------


async def test_an_index_error_leaves_the_previous_feed_alone(rig):
    """The checkpoint tracks history, so a bad sweep would outlive itself.

    Storing an empty feed against a fresh checkpoint means the next
    sweep sees the viewer as up to date and skips them. One transient
    error would then persist until they happened to watch something new.
    """
    w = worker_mod.PickupWorker()
    await w._compute_for_drive("a")
    assert len(rig["rows"]()) == 3
    good = rig["header"]().watch_history_checkpoint

    rig["watched"]["v1"] = {"seen1", "seen2"}
    rig["failing_channels"] = {"clip_thumbnail"}
    await w._compute_for_drive("a")

    assert [r.file_id for r in rig["rows"]()] == ["c1", "c2", "c3"]
    assert rig["header"]().watch_history_checkpoint == good


async def test_the_feed_is_rebuilt_once_the_index_recovers(rig):
    w = worker_mod.PickupWorker()
    rig["failing_channels"] = {"clip_thumbnail"}
    await w._compute_for_drive("a")
    assert rig["header"]() is None

    rig["failing_channels"] = set()
    await w._compute_for_drive("a")

    assert [r.file_id for r in rig["rows"]()] == ["c1", "c2", "c3"]


async def test_an_empty_feed_is_not_checkpointed(rig):
    """An empty feed is never settled.

    A viewer whose watched files are not embedded yet produces nothing
    now and something later, while their history — the only thing the
    checkpoint watches — does not move at all.
    """
    rig["watched"]["v1"] = {"c1", "c2", "c3", "seen1"}

    await worker_mod.PickupWorker()._compute_for_drive("a")

    assert rig["rows"]() == []
    assert rig["header"]().total == 0
    assert rig["header"]().watch_history_checkpoint is None


async def test_an_empty_feed_is_retried_on_the_next_sweep(rig):
    rig["watched"]["v1"] = {"c1", "c2", "c3", "seen1"}
    w = worker_mod.PickupWorker()
    await w._compute_for_drive("a")
    before = len(rig["scores"])

    await w._compute_for_drive("a")

    assert len(rig["scores"]) == before + 1


async def test_a_lane_less_profile_is_not_checkpointed(rig):
    rig["lanes"] = []

    await worker_mod.PickupWorker()._compute_for_drive("a")

    assert rig["header"]().watch_history_checkpoint is None


# ---------------------------------------------------------------------------
# Doing no work when there is none
# ---------------------------------------------------------------------------


async def test_an_idle_drive_does_not_touch_the_index(rig):
    """Loading the candidate matrices reads every vector in the drive.

    On a quiet hour nothing would be done with them, and the read is
    held under the drive lock the whole time.
    """
    w = worker_mod.PickupWorker()
    await w._compute_for_drive("a")
    rig["loads"].clear()

    await w._compute_for_drive("a")

    assert rig["loads"] == []


async def test_one_stale_viewer_is_enough_to_load(rig):
    rig["viewers"] = ["v1", "v2"]
    rig["watched"] = {"v1": {"a1"}, "v2": {"b1"}}
    rig["history"] = {"v1": ["a1"], "v2": ["b1"]}
    w = worker_mod.PickupWorker()
    await w._compute_for_drive("a")
    rig["loads"].clear()
    rig["scores"].clear()
    rig["watched"]["v2"] = {"b1", "b2"}

    await w._compute_for_drive("a")

    assert rig["loads"] != []
    assert len(rig["scores"]) == 1


# ---------------------------------------------------------------------------
# Storage is scoped to one viewer
# ---------------------------------------------------------------------------


async def test_rebuilding_one_viewer_leaves_the_other_alone(rig):
    """The delete is positional and must not reach past its owner.

    Unscoped it would strip the other viewer's rows while leaving their
    header, so their endpoint would answer total=N with an empty page.
    """
    rig["viewers"] = ["v1", "v2"]
    rig["watched"] = {"v1": {"a1"}, "v2": {"b1"}}
    rig["history"] = {"v1": ["a1"], "v2": ["b1"]}
    w = worker_mod.PickupWorker()
    await w._compute_for_drive("a")
    assert len(rig["rows"](viewer="v2")) == 3

    rig["watched"]["v1"] = {"a1", "a2"}
    await w._compute_for_drive("a")

    assert len(rig["rows"](viewer="v2")) == 3
    assert rig["header"](viewer="v2").total == 3


# ---------------------------------------------------------------------------
# The checkpoint covers the whole set, not a slice of it
# ---------------------------------------------------------------------------


async def test_a_change_beyond_the_first_twenty_ids_is_noticed(rig):
    """The shape this replaced hashed a 20-item slice of a recency list.

    Anything the viewer watched after that prefix left the hash
    unchanged, so their feed silently stopped being rebuilt.
    """
    rig["watched"]["v1"] = {f"f{i:03d}" for i in range(60)}
    rig["history"]["v1"] = sorted(rig["watched"]["v1"])
    w = worker_mod.PickupWorker()
    await w._compute_for_drive("a")
    before = len(rig["scores"])

    rig["watched"]["v1"].add("f999")
    await w._compute_for_drive("a")

    assert len(rig["scores"]) == before + 1


# ---------------------------------------------------------------------------
# A lane is only ever scored against its own channel
# ---------------------------------------------------------------------------


async def test_a_lane_is_scored_against_its_own_channels_candidates(rig):
    """Channels hold vectors of different widths in production.

    Crossing them raises here, but the guard is the last line of
    defence: this asserts the pairing itself, so the mistake is caught
    even where the widths happen to agree.
    """
    seen = []

    def recording_score(candidates, centroids, *, channel, exclude_file_ids, limit):
        seen.append((channel, candidates.channel, len(centroids)))
        return [[] for _ in centroids]

    rig["lanes"] = [
        Lane(channel="clip_thumbnail", cluster_id="clip_thumbnail:0",
             centroid=np.zeros(4, dtype=np.float32), member_count=2,
             raw_weight=1.0, weight=1.0),
        Lane(channel="tfidf_keywords", cluster_id="tfidf_keywords:0",
             centroid=np.zeros(4, dtype=np.float32), member_count=2,
             raw_weight=1.0, weight=1.0),
    ]
    import app.workers.pickup as mod
    mod.score_lanes = recording_score

    try:
        await worker_mod.PickupWorker()._compute_for_drive("a")
    finally:
        mod.score_lanes = rig["orig_score"]

    # tfidf_keywords has no rows in this rig, so only the visual lane is
    # scored — and it is scored against the visual candidate set.
    assert seen == [("clip_thumbnail", "clip_thumbnail", 1)]


# ---------------------------------------------------------------------------
# A trigger arriving mid-sweep
# ---------------------------------------------------------------------------


async def test_a_trigger_during_a_sweep_is_not_dropped(rig):
    """A scan finishing mid-run is when the feed is most out of date."""
    w = worker_mod.PickupWorker()
    lock = w._lock_for("a")
    await lock.acquire()
    try:
        await w._guarded_compute("a")
    finally:
        lock.release()

    assert "a" in w._missed_triggers


async def test_the_missed_trigger_is_served_before_the_lock_is_released(rig):
    w = worker_mod.PickupWorker()
    calls = []
    original = w._compute_for_drive

    async def counting(drive):
        calls.append(drive)
        if len(calls) == 1:
            w._missed_triggers.add(drive)
        await original(drive)

    w._compute_for_drive = counting
    await w._guarded_compute("a")

    assert calls == ["a", "a"]


# ---------------------------------------------------------------------------
# The checkpoint has to move with recency, not only with membership
# ---------------------------------------------------------------------------


async def test_reopening_a_watched_file_rebuilds_the_feed(rig):
    """Re-watching changes no ids, and changes the profile completely.

    Lane weights are made of recency. A viewer returning to something
    they had drifted away from is reviving that interest, and a feed
    keyed on the set alone would go on describing the old one.
    """
    w = worker_mod.PickupWorker()
    await w._compute_for_drive("a")
    before = len(rig["scores"])

    rig["played"][("v1", "seen1")] = "2026-08-30 12:00:00"
    await w._compute_for_drive("a")

    assert len(rig["scores"]) == before + 1


def test_the_checkpoint_moves_with_the_day():
    """The window and the half-life drift while the viewer does nothing.

    Without this a viewer who stops watching is frozen at whatever their
    profile said the day they stopped.
    """
    from datetime import UTC, datetime
    from unittest.mock import patch

    sig = [("f1", "2026-01-01 00:00:00")]

    with patch.object(worker_mod, "datetime") as clock:
        clock.now.return_value = datetime(2026, 3, 2, tzinfo=UTC)
        monday = worker_mod._checkpoint(sig)
        clock.now.return_value = datetime(2026, 3, 3, tzinfo=UTC)
        tuesday = worker_mod._checkpoint(sig)
        clock.now.return_value = datetime(2026, 3, 3, 23, tzinfo=UTC)
        same_day = worker_mod._checkpoint(sig)

    assert monday != tuesday
    assert tuesday == same_day


def test_the_checkpoint_still_moves_with_membership():
    a = worker_mod._checkpoint([("f1", "t"), ("f2", "t")])
    b = worker_mod._checkpoint([("f1", "t")])
    assert a != b


def test_the_checkpoint_is_order_independent_over_pairs():
    a = worker_mod._checkpoint([("f1", "t1"), ("f2", "t2")])
    b = worker_mod._checkpoint([("f2", "t2"), ("f1", "t1")])
    assert a == b
