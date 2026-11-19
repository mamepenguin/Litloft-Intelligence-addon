"""The Pickup worker under coverage: glanced videos are candidates, watched ones are not.

SPEC-ID: SPEC-ADDON-149
SPEC-ID: SPEC-ADDON-150
"""

from __future__ import annotations

import sys
import time
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.models import PickupItem, PickupProfile  # noqa: E402
from app.pickup import profile  # noqa: E402
from app.routers import pickup as endpoint  # noqa: E402
from app.workers import pickup as worker_mod  # noqa: E402
from tests.pickup_accuracy_rig import (  # noqa: E402
    LOFT_MIME,
    CoreDB,
    SearchDB,
    at_angle,
    utcnow,
    wire,
)

LEGACY = "2026-10-20 12:00:00"


class _Rig:
    def __init__(self, core: CoreDB, search: SearchDB):
        self.core = core
        self.search = search
        self.worker = worker_mod.PickupWorker()

    def segments(self, file_id, indexes, *, seconds=5.0, viewer="v1"):
        for n in indexes:
            self.core.execute(
                "INSERT INTO played_segments VALUES (:v, :f, :n, :s)",
                {"v": viewer, "f": file_id, "n": n, "s": seconds},
            )

    def video(self, file_id, angle=None, **file_kw):
        self.core.add_file(file_id, **file_kw)
        if angle is not None:
            self.search.index(file_id, at_angle(angle), kind=file_kw.get("kind", "video"))

    def watched(self, file_id, angle=None, *, hours=24.0, **file_kw):
        self.video(file_id, angle, **file_kw)
        self.core.watch(file_id, played_at=utcnow() - timedelta(hours=hours), position=0.0)
        self.segments(file_id, range(10))

    def glanced(self, file_id, angle=None, *, hours=24.0, **file_kw):
        self.video(file_id, angle, **file_kw)
        self.core.watch(file_id, played_at=utcnow() - timedelta(hours=hours), position=100.0)

    def kept(self, n=5, angle=0.0):
        for i in range(n):
            self.watched(f"h{i}", angle, hours=24.0 * (1 + i))

    async def sweep(self):
        time.sleep(0.01)
        await self.worker._compute_for_drive("a")

    def feed(self, viewer="v1"):
        s = self.search.sessions()
        try:
            return [
                r.file_id for r in s.query(PickupItem)
                .filter_by(drive_id="a", viewer_id=viewer)
                .order_by(PickupItem.rank).all()
            ]
        finally:
            s.close()

    def header(self, viewer="v1"):
        s = self.search.sessions()
        try:
            return s.query(PickupProfile).filter_by(drive_id="a", viewer_id=viewer).first()
        finally:
            s.close()


@pytest.fixture()
def rig(monkeypatch, tmp_path):
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    wire(monkeypatch, core, search)
    return _Rig(core, search)


# ---------------------------------------------------------------------------
# Rule 1: what the stored feed may hold, and what seeds it
# ---------------------------------------------------------------------------


async def test_spec_addon_149_150_a_glance_is_a_candidate_and_no_seed(rig):
    """SPEC-ADDON-149 rule 1 / SPEC-ADDON-150: glanced videos come back; watched, legacy and unparseable rows do not."""
    rig.kept()
    rig.glanced("glanced", 5.0)
    rig.watched("watched", 6.0)
    rig.video("legacy", 7.0)
    rig.core.watch("legacy", played_at=LEGACY, position=1.0)
    rig.video("unparsed", 8.0)
    rig.core.watch("unparsed", played_at="not-a-date", position=0.0, duration=0.0)
    rig.video("never", 9.0)
    for i in range(3):
        rig.glanced(f"far{i}", 90.0, hours=2.0)
    rig.video("near_far", 86.0)

    await rig.sweep()

    assert rig.feed() == ["glanced", "never"]


async def test_spec_addon_149_i5_the_sweep_the_exclusion_read_and_the_feed_read_agree(rig):
    """SPEC-ADDON-149 I5: one dataset, one exclusion set, three readers."""
    rig.kept()
    rig.watched("w", 0.0)
    rig.glanced("g", 4.0)
    rig.glanced("liked", 0.0, liked=True)
    rig.glanced("fav", 0.0, favourite=True)
    rig.video("legacy", 7.0)
    rig.core.watch("legacy", played_at=LEGACY, position=1.0)
    rig.video("short", 0.0)
    rig.core.watch("short", played_at=utcnow() - timedelta(hours=3), position=6.0, duration=10.0)
    rig.video("short_glance", 9.0)
    rig.core.watch("short_glance", played_at=utcnow() - timedelta(hours=3), position=4.0, duration=10.0)
    rig.video("embed", 10.0, mime=LOFT_MIME)
    rig.core.watch("embed", played_at=utcnow() - timedelta(hours=3), position=0.0, duration=0.0)
    rig.video("never", 11.0)
    rig.glanced("doc", 12.0, kind="document")

    excluded = profile.watched_file_ids("a", "v1")
    await rig.sweep()
    stored = rig.feed()
    served = await endpoint.pickup_endpoint(
        drive="a", viewer_id="v1", limit=60, offset=0, window=None, date=None,
    )

    assert excluded == {
        "h0", "h1", "h2", "h3", "h4", "w", "liked", "fav", "legacy", "short", "doc",
    }
    assert stored == ["g", "short_glance", "embed", "never"]
    assert served == {"file_ids": stored, "total": 4}


async def test_spec_addon_095_i2_revised_a_glance_changes_neither_the_lanes_nor_the_exclusion(rig):
    """SPEC-ADDON-095 I2 as revised: a glanced row on a candidate gives exactly the feed without it."""
    rig.kept()
    for file_id, angle in (("c0", 4.0), ("c1", 8.0), ("c2", 12.0), ("c3", 16.0)):
        rig.video(file_id, angle)
    await rig.sweep()
    base = rig.feed()
    assert base == ["c0", "c1", "c2", "c3"]

    rig.core.watch("c1", played_at=utcnow() - timedelta(hours=1), position=100.0)
    rig.segments("c1", range(9))
    await rig.sweep()

    assert rig.feed() == base


# ---------------------------------------------------------------------------
# Rule 2: the checkpoint follows watched state and engagement
# ---------------------------------------------------------------------------


def _coverage_only(rig):
    rig.core.execute(
        "UPDATE played_segments SET segment_index = 30 WHERE file_id = 'h0' AND segment_index = 9"
    )


def _more_coverage(rig):
    rig.segments("h0", [12])


def _longer(rig):
    rig.core.execute("UPDATE watch_history SET duration = 110 WHERE file_id = 'h0'")


def _shorter(rig):
    rig.core.execute("UPDATE watch_history SET duration = 90 WHERE file_id = 'h0'")


def _position(rig):
    rig.core.execute("UPDATE watch_history SET playback_position = 30 WHERE file_id = 'h0'")


def _played_seconds_only(rig):
    rig.core.execute(
        "UPDATE played_segments SET played_seconds = 3.0 WHERE file_id = 'h0' AND segment_index = 0"
    )


def _mime(rig):
    rig.core.execute("UPDATE files SET mime_type = :m WHERE id = 'h0'", {"m": LOFT_MIME})


def _file_type(rig):
    rig.core.execute("UPDATE files SET file_type = 'document' WHERE id = 'h0'")


def _glance_covered(rig):
    rig.segments("g", range(10))


def _glance_duration(rig):
    rig.core.execute("UPDATE watch_history SET duration = 200 WHERE file_id = 'g'")


@pytest.mark.parametrize(
    "change",
    [_coverage_only, _more_coverage, _longer, _shorter, _position, _played_seconds_only,
     _mime, _file_type, _glance_covered, _glance_duration],
    ids=["coverage only", "more coverage", "duration longer", "duration shorter", "position",
         "played seconds only", "mime type", "file type", "a glance reaches half",
         "a glance's duration"],
)
async def test_spec_addon_149_rule_2_a_change_without_a_new_play_rebuilds_once(rig, change):
    """SPEC-ADDON-149 rule 2 / SPEC-ADDON-095 I5 as revised: the checkpoint moves, then holds."""
    rig.kept()
    rig.glanced("g", 4.0)
    for file_id, angle in (("c0", 5.0), ("c1", 10.0)):
        rig.video(file_id, angle)
    await rig.sweep()
    assert rig.feed() == ["g", "c0", "c1"]
    before = rig.header().watch_history_checkpoint
    assert before is not None

    change(rig)
    await rig.sweep()
    moved = rig.header()

    assert moved.watch_history_checkpoint != before

    await rig.sweep()
    held = rig.header()

    assert held.watch_history_checkpoint == moved.watch_history_checkpoint
    assert held.computed_at == moved.computed_at


async def test_spec_addon_149_a_glance_watched_since_the_sweep_leaves_the_next_feed(rig):
    """SPEC-ADDON-149 rules 1 and 2: once covered, the rebuilt feed drops it."""
    rig.kept()
    rig.glanced("g", 4.0)
    rig.video("c0", 5.0)
    await rig.sweep()
    assert rig.feed() == ["g", "c0"]

    rig.segments("g", range(10))
    await rig.sweep()

    assert rig.feed() == ["c0"]


# ---------------------------------------------------------------------------
# Failure: a sweep that cannot read history or segments writes nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["played_segments", "watch_history"])
async def test_spec_addon_149_an_unreadable_history_leaves_the_previous_feed(rig, table):
    """SPEC-ADDON-149 failure cases: no feed or profile row is written; the next sweep retries."""
    rig.kept()
    rig.glanced("g", 4.0)
    rig.video("c0", 5.0)
    await rig.sweep()
    before = rig.header()
    assert rig.feed() == ["g", "c0"]

    rig.segments("g", range(10))
    rig.core.execute(f"ALTER TABLE {table} RENAME TO {table}_away")
    try:
        await rig.sweep()
    except Exception:
        pass

    after = rig.header()
    assert rig.feed() == ["g", "c0"]
    assert after.watch_history_checkpoint == before.watch_history_checkpoint
    assert after.computed_at == before.computed_at

    rig.core.execute(f"ALTER TABLE {table}_away RENAME TO {table}")
    await rig.sweep()

    assert rig.feed() == ["c0"]


def test_spec_addon_149_profile_version_is_5():
    """SPEC-ADDON-149 rule 5 / I5: every feed is rebuilt after deployment."""
    assert profile.PROFILE_VERSION == 5
