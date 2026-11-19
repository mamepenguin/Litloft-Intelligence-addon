"""The Pickup feed read removes what the viewer has watched since the last sweep.

SPEC-ID: SPEC-ADDON-149
"""

from __future__ import annotations

import sys
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.models import PickupItem, PickupProfile  # noqa: E402
from app.routers import pickup as endpoint  # noqa: E402
from tests.pickup_accuracy_rig import CoreDB, SearchDB, utcnow, wire  # noqa: E402

PLAYED = utcnow() - timedelta(days=1)


class _Store:
    def __init__(self, core: CoreDB, search: SearchDB):
        self.core = core
        self.search = search

    def feed(self, file_ids, *, drive="a", viewer="v1"):
        s = self.search.sessions()
        try:
            s.query(PickupItem).filter_by(drive_id=drive, viewer_id=viewer).delete()
            s.query(PickupProfile).filter_by(drive_id=drive, viewer_id=viewer).delete()
            s.add(PickupProfile(
                drive_id=drive, viewer_id=viewer,
                total=len(file_ids), watch_history_checkpoint="x",
            ))
            for rank, file_id in enumerate(file_ids, start=1):
                s.add(PickupItem(
                    drive_id=drive, viewer_id=viewer, rank=rank, file_id=file_id,
                    cluster_id="c0", channel="clip_thumbnail", score=0.9,
                ))
            s.commit()
        finally:
            s.close()

    def files(self, file_ids, *, drive="a"):
        for file_id in file_ids:
            self.core.add_file(file_id, drive=drive)

    def watched(self, file_id, *, viewer="v1"):
        self.core.watch(file_id, viewer=viewer, played_at=PLAYED, position=0.0)
        self.segments(file_id, range(20), viewer=viewer)

    def glanced(self, file_id, *, viewer="v1"):
        self.core.watch(file_id, viewer=viewer, played_at=PLAYED, position=100.0)

    def segments(self, file_id, indexes, *, viewer="v1"):
        for n in indexes:
            self.core.execute(
                "INSERT INTO played_segments VALUES (:v, :f, :n, 5.0)",
                {"v": viewer, "f": file_id, "n": n},
            )


@pytest.fixture()
def store(monkeypatch, tmp_path):
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    wire(monkeypatch, core, search)
    return _Store(core, search)


async def _call(**kw):
    kw.setdefault("drive", "a")
    kw.setdefault("viewer_id", "v1")
    kw.setdefault("limit", 12)
    kw.setdefault("offset", 0)
    kw.setdefault("window", None)
    kw.setdefault("date", None)
    return await endpoint.pickup_endpoint(**kw)


FEED = [f"f{i:02d}" for i in range(20)]


def _mixed(store: _Store) -> list[str]:
    """A stored feed of 20 ids, five of them in the viewer's exclusion set."""
    store.files(FEED)
    store.watched("f02")
    store.core.execute("UPDATE files SET file_type = 'document' WHERE id = 'f05'")
    store.glanced("f05")
    store.glanced("f07")
    store.core.watch("f09", played_at="2026-10-01 12:00:00", position=1.0)
    store.core.execute("UPDATE files SET liked_at = '2026-11-10 00:00:00' WHERE id = 'f11'")
    store.glanced("f11")
    store.watched("f13", viewer="v2")
    store.core.watch("f15", played_at="not-a-date")
    store.feed(FEED)
    return [f for f in FEED if f not in {"f02", "f05", "f09", "f11", "f15"}]


# ---------------------------------------------------------------------------
# Rule 4: the stored feed minus the current exclusion set
# ---------------------------------------------------------------------------


async def test_spec_addon_149_i4_the_feed_read_removes_the_exclusion_set_in_rank_order(store):
    """SPEC-ADDON-149 I4: watched, non-video, legacy and liked rows are removed; total is the filtered length."""
    expected = _mixed(store)

    got = await _call(limit=40)

    assert got == {"file_ids": expected, "total": 15}


@pytest.mark.parametrize(
    ("offset", "limit", "window"),
    [(0, 5, slice(0, 5)), (5, 5, slice(5, 10)), (10, 12, slice(10, 15)), (14, 1, slice(14, 15)),
     (15, 5, slice(15, 15)), (40, 5, slice(40, 40))],
    ids=["first page", "second page", "last partial page", "last item", "at the end", "past the end"],
)
async def test_spec_addon_149_i4_offset_and_limit_slice_the_filtered_list(store, offset, limit, window):
    """SPEC-ADDON-149 I4: pages stay full and total is what a reader reaches."""
    expected = _mixed(store)

    got = await _call(offset=offset, limit=limit)

    assert got == {"file_ids": expected[window], "total": 15}


async def test_spec_addon_149_i4_every_item_excluded_leaves_an_empty_feed(store):
    """SPEC-ADDON-149 I4: zero left, total 0, for paging and for the daily window."""
    store.files(["a0", "a1"])
    store.watched("a0")
    store.watched("a1")
    store.feed(["a0", "a1"])

    assert await _call() == {"file_ids": [], "total": 0}
    assert (await _call(window="daily", date="2026-11-19"))["file_ids"] == []


async def test_spec_addon_149_i4_a_video_watched_since_the_sweep_leaves_at_once(store):
    """SPEC-ADDON-149 rule 4: a glanced video stays until its coverage reaches half."""
    store.files(["x0", "x1", "x2"])
    store.glanced("x1")
    store.feed(["x0", "x1", "x2"])
    assert (await _call())["file_ids"] == ["x0", "x1", "x2"]

    store.segments("x1", range(10))

    assert await _call() == {"file_ids": ["x0", "x2"], "total": 2}


async def test_spec_addon_149_i4_another_viewers_or_drives_history_filters_nothing(store):
    """SPEC-ADDON-149 I4: the exclusion set is this viewer's, in this drive."""
    store.files(["y0", "y1", "y2"])
    store.files(["z0"], drive="b")
    store.watched("y0", viewer="v2")
    store.watched("y1")
    store.watched("z0")
    store.feed(["y0", "y1", "y2"])
    store.feed(["y0", "y1", "y2"], viewer="v2")

    assert (await _call(viewer_id="v1"))["file_ids"] == ["y0", "y2"]
    assert (await _call(viewer_id="v2"))["file_ids"] == ["y1", "y2"]


# ---------------------------------------------------------------------------
# The daily window is drawn from the filtered list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stored", [60, 51, 20], ids=["over 40 left", "exactly 40 left", "under 40 left"])
@pytest.mark.parametrize("date", ["2026-11-19", "2026-12-01"])
async def test_spec_addon_149_i4_the_daily_window_is_the_window_of_the_filtered_list(store, stored, date):
    """SPEC-ADDON-149 I4: the same window as a stored feed that never held the excluded ids."""
    ids = [f"w{i:02d}" for i in range(stored)]
    removed = {f"w{i:02d}" for i in range(0, min(stored, 10))} | {ids[-1]}
    kept = [f for f in ids if f not in removed]
    store.files(ids)

    store.feed(kept)
    oracle = await _call(window="daily", date=date)

    for file_id in sorted(removed):
        store.watched(file_id)
    store.feed(ids)
    got = await _call(window="daily", date=date)

    assert got == oracle
    assert set(got["file_ids"]).isdisjoint(removed)
    assert got["total"] == len(kept)


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["watch_history", "played_segments", "files"])
async def test_spec_addon_149_an_unreadable_exclusion_set_fails_the_request(store, table):
    """SPEC-ADDON-149 failure cases: the read fails as a failed feed read does, never unfiltered."""
    store.files(["u0", "u1"])
    store.watched("u0")
    store.feed(["u0", "u1"])
    store.core.execute(f"DROP TABLE {table}")

    with pytest.raises(Exception):
        await _call()
