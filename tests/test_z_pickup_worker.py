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
    }

    monkeypatch.setattr(worker_mod.profile_mod, "viewer_ids",
                        lambda drive: list(state["viewers"]))
    monkeypatch.setattr(worker_mod.profile_mod, "watched_file_ids",
                        lambda drive, v: set(state["watched"].get(v, ())))
    monkeypatch.setattr(worker_mod.profile_mod, "profile_history",
                        lambda drive, v: list(state["history"].get(v, ())))
    monkeypatch.setattr(worker_mod.profile_mod, "build_lanes",
                        lambda history, key: list(state["lanes"]))
    monkeypatch.setattr(worker_mod.profile_mod, "CHANNELS",
                        ("clip_thumbnail", "tfidf_keywords"))

    def fake_load(*, drive, channel):
        state["loads"].append((drive, channel))
        if channel != "clip_thumbnail":
            return CandidateSet(channel=channel, drive=drive, file_ids=(),
                                matrix=np.zeros((0, 0), dtype=np.float32))
        return CandidateSet(
            channel=channel, drive=drive,
            file_ids=("c1", "c2", "c3"),
            matrix=np.eye(3, 4, dtype=np.float32),
        )

    def fake_score(candidates, centroids, *, exclude_file_ids, limit):
        state["scores"].append(set(exclude_file_ids))
        hits = [(f, 0.9 - i * 0.01)
                for i, f in enumerate(candidates.file_ids)
                if f not in exclude_file_ids]
        return [list(hits[:limit]) for _ in centroids]

    monkeypatch.setattr(worker_mod, "load_candidates", fake_load)
    monkeypatch.setattr(worker_mod, "score_lanes", fake_score)

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
