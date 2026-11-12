"""The Pickup worker with freshness, and the rebuilds new arrivals cause.

SPEC-ID: SPEC-ADDON-102
SPEC-ID: SPEC-ADDON-103
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.models import PickupItem, PickupProfile  # noqa: E402
from app.pickup import profile as profile_mod  # noqa: E402
from app.pickup import retrieval  # noqa: E402
from app.workers import pickup as worker_mod  # noqa: E402
from tests.pickup_accuracy_rig import (  # noqa: E402
    CoreDB,
    SearchDB,
    at_angle,
    ts,
    utcnow,
    wire,
)


class _Rig:
    def __init__(self, core: CoreDB, search: SearchDB):
        self.core = core
        self.search = search
        self.worker = worker_mod.PickupWorker()

    def watched(self, file_id, *, angle=None, kind="video", age_days=1.0, viewer="v1",
                drive="a"):
        self.core.add_file(file_id, kind=kind, drive=drive)
        self.core.watch(file_id, viewer=viewer, played_at=utcnow() - timedelta(days=age_days))
        if angle is not None:
            self.search.index(file_id, at_angle(angle), kind=kind, drive=drive)

    def candidate(self, file_id, angle, *, kind="video", channel="clip_thumbnail",
                  drive="a", **file_kw):
        self.core.add_file(file_id, kind=kind, drive=drive, **file_kw)
        self.search.index(file_id, at_angle(angle), channel=channel, kind=kind, drive=drive)

    def kept(self, prefix="h", n=5, angle=0.0, viewer="v1"):
        for i in range(n):
            self.watched(f"{prefix}{i}", angle=angle, age_days=1 + i, viewer=viewer)

    async def sweep(self, drive="a"):
        time.sleep(0.01)
        await self.worker._compute_for_drive(drive)

    def items(self, viewer="v1", drive="a"):
        s = self.search.sessions()
        try:
            return (
                s.query(PickupItem)
                .filter_by(drive_id=drive, viewer_id=viewer)
                .order_by(PickupItem.rank).all()
            )
        finally:
            s.close()

    def feed(self, viewer="v1", drive="a"):
        return [r.file_id for r in self.items(viewer, drive)]

    def header(self, viewer="v1", drive="a"):
        s = self.search.sessions()
        try:
            return s.query(PickupProfile).filter_by(drive_id=drive, viewer_id=viewer).first()
        finally:
            s.close()


@pytest.fixture()
def rig(monkeypatch, tmp_path):
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    wire(monkeypatch, core, search)
    return _Rig(core, search)


def _patch_everywhere(monkeypatch, name, value, home, *, raising=True):
    """Replace ``name`` in its home module and wherever it was imported by name."""
    monkeypatch.setattr(home, name, value, raising=raising)
    for mod in (worker_mod, retrieval, profile_mod):
        if mod is not home and hasattr(mod, name):
            monkeypatch.setattr(mod, name, value)


def _use_freshness(monkeypatch, *, now, half_life, floor):
    """Fix the reference time and the worker's constants for one test."""
    fresh = importlib.import_module("app.pickup.freshness")
    monkeypatch.setattr(fresh, "HALF_LIFE_DAYS", half_life)
    monkeypatch.setattr(fresh, "FLOOR", floor)
    # Defaults are bound when the function is defined, so the module constants
    # alone would not reach a call that relies on them.
    defaults = dict(fresh.freshness.__kwdefaults__ or {})
    defaults.update(half_life_days=half_life, floor=floor)
    monkeypatch.setattr(fresh.freshness, "__kwdefaults__", defaults)
    _patch_everywhere(monkeypatch, "reference_now", lambda: now, fresh)
    return fresh


# ---------------------------------------------------------------------------
# SPEC-ADDON-102: the stored feed
# ---------------------------------------------------------------------------


async def test_spec_addon_102_a_newer_match_outranks_a_closer_old_one(rig, monkeypatch):
    """SPEC-ADDON-102: the stored feed follows similarity x freshness at the sweep's reference time."""
    now = utcnow()
    _use_freshness(monkeypatch, now=now, half_life=7.0, floor=0.5)
    rig.kept()
    rig.candidate("old_close", 3.0, created_at=now - timedelta(days=400))
    rig.candidate("new_far", 20.0, created_at=now - timedelta(hours=1))

    await rig.sweep()

    assert rig.feed() == ["new_far", "old_close"]


async def test_spec_addon_102_the_stored_score_is_the_ranking_score(rig, monkeypatch):
    """SPEC-ADDON-102: pickup_item.score holds similarity x freshness, computed at reference_now()."""
    now = utcnow().replace(microsecond=0)
    fresh = _use_freshness(monkeypatch, now=now, half_life=7.0, floor=0.5)
    rig.kept()
    created = {
        "c_old": now - timedelta(days=30),
        "c_mid": now - timedelta(days=7),
        "c_new": now - timedelta(days=1),
    }
    angles = {"c_old": 5.0, "c_mid": 10.0, "c_new": 15.0}
    for file_id, at in created.items():
        rig.candidate(file_id, angles[file_id], created_at=at)

    await rig.sweep()

    import math

    got = {r.file_id: r.score for r in rig.items()}
    expected = {
        f: math.cos(math.radians(angles[f]))
        * fresh.freshness(created[f], now, half_life_days=7.0, floor=0.5)
        for f in created
    }
    assert got == pytest.approx(expected, rel=1e-4)
    assert rig.feed() == sorted(expected, key=expected.get, reverse=True)


async def test_spec_addon_102_i1_a_decayed_candidate_still_qualifies(rig, monkeypatch):
    """SPEC-ADDON-102 I1: at FLOOR = 0 an ancient close match stays in the feed, last."""
    now = utcnow()
    _use_freshness(monkeypatch, now=now, half_life=7.0, floor=0.0)
    rig.kept()
    rig.candidate("ancient", 3.0, created_at=now - timedelta(days=11_000))
    rig.candidate("new", 20.0, created_at=now - timedelta(hours=2))
    rig.candidate("noise", 70.0, created_at=now - timedelta(hours=1))

    await rig.sweep()

    assert rig.feed() == ["new", "ancient"]


async def test_spec_addon_102_i3_unknown_and_future_dates_rank_at_the_floor(rig, monkeypatch):
    """SPEC-ADDON-102 I3: unparseable or over a day ahead is FLOOR; up to a day ahead is age 0."""
    now = utcnow()
    _use_freshness(monkeypatch, now=now, half_life=7.0, floor=0.5)
    rig.kept()
    rig.candidate("bad", 3.0, created_at="not-a-date")
    rig.candidate("future", 4.0, created_at=now + timedelta(days=3))
    rig.candidate("skew", 15.0, created_at=now + timedelta(hours=12))
    rig.candidate("new", 10.0, created_at=now - timedelta(hours=1))

    await rig.sweep()

    assert rig.feed() == ["new", "skew", "bad", "future"]


async def test_spec_addon_102_profile_version_is_4_in_the_feed_log(rig, monkeypatch):
    """SPEC-ADDON-102: PROFILE_VERSION 4, written on new feed log rows."""
    rig.search.create_feed_log()
    rig.kept()
    rig.candidate("c0", 5.0)

    await rig.sweep()

    assert profile_mod.PROFILE_VERSION == 4
    rows = rig.search.query("SELECT profile_version, file_ids FROM pickup_feed_log")
    assert [(v, json.loads(f)) for v, f in rows] == [(4, ["c0"])]


# ---------------------------------------------------------------------------
# SPEC-ADDON-103: arrivals move the checkpoint
# ---------------------------------------------------------------------------


async def _settled(rig):
    """Two sweeps, the second a no-op: the precondition every arrival test starts from."""
    await rig.sweep()
    first = rig.header()
    assert first is not None and first.watch_history_checkpoint is not None
    await rig.sweep()
    held = rig.header()
    assert held.computed_at == first.computed_at
    assert held.watch_history_checkpoint == first.watch_history_checkpoint
    return held


def _base(rig):
    rig.kept()
    rig.candidate("c0", 5.0)
    rig.candidate("c1", 10.0)
    rig.candidate("was_trashed", 12.0, deleted=True)
    rig.core.add_file("late")


def _new_video(rig):
    rig.candidate("arrived", 8.0)


def _new_video_row_only(rig):
    rig.core.add_file("arrived_row")


def _embedding(channel):
    def change(rig):
        rig.search.index("late", at_angle(6.0), channel=channel)
    change.__name__ = f"_embedding_{channel}"
    return change


def _trash(rig):
    rig.core.execute("UPDATE files SET deleted_at = :t WHERE id = 'c1'", {"t": ts(utcnow())})


def _restore(rig):
    rig.core.execute("UPDATE files SET deleted_at = NULL WHERE id = 'was_trashed'")


def _goes_missing(rig):
    rig.core.execute("UPDATE files SET missing_since = :t WHERE id = 'c1'", {"t": ts(utcnow())})


_ARRIVALS = [
    _new_video,
    _new_video_row_only,
    _embedding("clip_thumbnail"),
    _embedding("tfidf_keywords"),
    _embedding("text_content"),
    _trash,
    _restore,
    _goes_missing,
]
_ARRIVAL_IDS = [
    "new video with its embedding",
    "new live video row",
    "clip_thumbnail embedding for a live video",
    "tfidf_keywords embedding for a live video",
    "text_content embedding for a live video",
    "trash",
    "restore",
    "goes missing",
]


@pytest.mark.parametrize("change", _ARRIVALS, ids=_ARRIVAL_IDS)
async def test_spec_addon_103_i6_an_arrival_rebuilds_every_viewer_once(rig, change):
    """SPEC-ADDON-103 I6: every viewer's checkpoint moves, then holds on the next computation."""
    _base(rig)
    rig.kept("g", viewer="v2")
    await _settled(rig)
    before = {v: rig.header(v) for v in ("v1", "v2")}

    change(rig)
    await rig.sweep()
    moved = {v: rig.header(v) for v in ("v1", "v2")}

    for v in ("v1", "v2"):
        assert moved[v].watch_history_checkpoint != before[v].watch_history_checkpoint, v
        assert moved[v].computed_at != before[v].computed_at, v

    await rig.sweep()
    for v in ("v1", "v2"):
        held = rig.header(v)
        assert held.watch_history_checkpoint == moved[v].watch_history_checkpoint, v
        assert held.computed_at == moved[v].computed_at, v


async def test_spec_addon_103_a_new_match_reaches_the_stored_feed(rig):
    """SPEC-ADDON-103: a new matching video is in the feed after the next computation."""
    _base(rig)
    await _settled(rig)

    rig.candidate("arrived", 8.0)
    await rig.sweep()

    assert "arrived" in rig.feed()


def _image_embedding(rig):
    rig.search.index("img", at_angle(4.0), kind="image")


def _document_embedding(rig):
    rig.search.index("doc", at_angle(4.0), channel="text_content", kind="document")


def _non_pickup_embedding(rig):
    rig.search.index("late", at_angle(4.0), channel="whisper")


def _another_drives_video(rig):
    rig.candidate("b_new", 4.0, drive="b")


def _another_drives_trash(rig):
    rig.core.execute("UPDATE files SET deleted_at = :t WHERE id = 'b_old'", {"t": ts(utcnow())})


@pytest.mark.parametrize(
    "change",
    [_image_embedding, _document_embedding, _non_pickup_embedding,
     _another_drives_video, _another_drives_trash],
    ids=["image embedding", "document embedding", "non-Pickup channel embedding",
         "another drive's new video", "another drive's trash"],
)
async def test_spec_addon_103_i6_i8_what_does_not_move_the_stamp(rig, change):
    """SPEC-ADDON-103 I6, I8: these leave the checkpoint alone; a video embedding then moves it."""
    _base(rig)
    rig.core.add_file("img", kind="image")
    rig.core.add_file("doc", kind="document")
    rig.candidate("b_old", 4.0, drive="b")
    settled = await _settled(rig)

    change(rig)
    await rig.sweep()
    held = rig.header()

    assert held.watch_history_checkpoint == settled.watch_history_checkpoint
    assert held.computed_at == settled.computed_at

    rig.search.index("late", at_angle(6.0))
    await rig.sweep()

    assert rig.header().watch_history_checkpoint != settled.watch_history_checkpoint


async def test_spec_addon_103_a_settled_empty_feed_is_rebuilt_on_arrival(rig):
    """SPEC-ADDON-095 I6 as revised: no kept row and an unchanged history, but the stamp moved."""
    for i in range(5):
        rig.watched(f"d{i}", angle=0.0, kind="document", age_days=1 + i)
    rig.candidate("c0", 5.0)
    await rig.sweep()
    first = rig.header()
    assert first.total == 0 and first.watch_history_checkpoint is not None
    await rig.sweep()
    assert rig.header().computed_at == first.computed_at

    rig.candidate("arrived", 8.0)
    await rig.sweep()

    assert rig.header().computed_at != first.computed_at


async def test_spec_addon_103_i10_an_embedding_written_mid_computation_rebuilds_next_time(rig, monkeypatch):
    """SPEC-ADDON-103 I10: the stamp is read before the candidate load, so a late write is caught."""
    _base(rig)
    await _settled(rig)

    original = worker_mod.load_candidates
    written = []

    def load_then_race(*, drive, channel):
        if not written:
            rig.search.index("late", at_angle(6.0))
            written.append(channel)
        return original(drive=drive, channel=channel)

    monkeypatch.setattr(worker_mod, "load_candidates", load_then_race)
    rig.core.watch("c0", played_at=utcnow() - timedelta(minutes=5))
    await rig.sweep()
    raced = rig.header()
    assert written

    monkeypatch.setattr(worker_mod, "load_candidates", original)
    await rig.sweep()
    after = rig.header()

    assert after.computed_at != raced.computed_at
    assert after.watch_history_checkpoint != raced.watch_history_checkpoint

    await rig.sweep()
    assert rig.header().computed_at == after.computed_at


async def test_spec_addon_103_the_stamp_is_read_before_the_live_video_read(rig, monkeypatch):
    """SPEC-ADDON-103: a video added after the live-video read is caught by the next computation."""
    _base(rig)
    await _settled(rig)
    added = []

    def racing(original):
        def read(drive):
            got = original(drive)
            if not added:
                rig.core.add_file("arrived_mid")
                added.append(drive)
            return got
        return read

    for name in ("live_videos", "live_video_ids"):
        if hasattr(profile_mod, name):
            monkeypatch.setattr(profile_mod, name, racing(getattr(profile_mod, name)))
    rig.core.watch("c0", played_at=utcnow() - timedelta(minutes=5))
    await rig.sweep()
    raced = rig.header()
    assert added

    await rig.sweep()
    after = rig.header()

    assert after.computed_at != raced.computed_at
    assert after.watch_history_checkpoint != raced.watch_history_checkpoint

    await rig.sweep()
    assert rig.header().computed_at == after.computed_at


# ---------------------------------------------------------------------------
# SPEC-ADDON-103: the stamp read itself
# ---------------------------------------------------------------------------


async def test_spec_addon_103_i7_a_failed_stamp_read_abandons_the_sweep(rig, monkeypatch, caplog):
    """SPEC-ADDON-103 I7: no pickup row of the drive changes, and logger.exception reports it."""
    _base(rig)
    await _settled(rig)
    before_feed = rig.feed()
    before = rig.header()

    def refuse(drive):
        raise RuntimeError("stamp unavailable")

    _patch_everywhere(monkeypatch, "arrival_stamp", refuse, retrieval, raising=False)
    rig.core.watch("c0", played_at=utcnow() - timedelta(minutes=5))
    rig.candidate("arrived", 8.0)
    caplog.set_level(logging.ERROR)

    await rig.sweep()

    after = rig.header()
    assert rig.feed() == before_feed
    assert after.computed_at == before.computed_at
    assert after.watch_history_checkpoint == before.watch_history_checkpoint
    assert any(
        r.levelno == logging.ERROR and r.exc_info is not None
        and r.getMessage().startswith("Pickup: could not")
        and "skipping this sweep" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


async def test_spec_addon_103_the_stamp_is_read_once_per_computation_with_viewers(rig, monkeypatch):
    """SPEC-ADDON-103: one stamp read per drive computation that found viewers, none without."""
    calls = []
    real = retrieval.arrival_stamp

    def counting(drive):
        calls.append(drive)
        return real(drive)

    _patch_everywhere(monkeypatch, "arrival_stamp", counting, retrieval)
    rig.candidate("c0", 5.0)

    await rig.sweep()
    assert calls == []

    rig.kept()
    rig.kept("g", viewer="v2")
    await rig.sweep()
    assert calls == ["a"]


def test_spec_addon_103_arrival_stamp_is_a_deterministic_string(rig):
    """SPEC-ADDON-103: the same data gives the same stamp; a new video embedding changes it."""
    stamp = retrieval.arrival_stamp
    rig.candidate("c0", 5.0)
    rig.core.add_file("late")

    first = stamp("a")
    again = stamp("a")
    rig.search.index("late", at_angle(6.0))
    moved = stamp("a")

    assert isinstance(first, str)
    assert first == again
    assert moved != first


@pytest.mark.parametrize("database", ["search", "core"])
def test_spec_addon_103_arrival_stamp_raises_on_a_read_failure(rig, database):
    """SPEC-ADDON-103: a failure of either read raises rather than returning a partial stamp."""
    stamp = retrieval.arrival_stamp
    rig.candidate("c0", 5.0)
    if database == "search":
        rig.search.execute("DROP TABLE embeddings")
    else:
        rig.core.execute("DROP TABLE files")

    with pytest.raises(Exception):
        stamp("a")


def test_spec_addon_103_arrival_stamp_ignores_other_drives(rig):
    """SPEC-ADDON-103 I8: another drive's videos, rows and embeddings leave the stamp alone."""
    stamp = retrieval.arrival_stamp
    rig.candidate("c0", 5.0)
    first = stamp("a")

    rig.candidate("b0", 5.0, drive="b", created_at=datetime(2030, 1, 1))
    rig.core.add_file("b1", drive="b")

    assert stamp("a") == first
