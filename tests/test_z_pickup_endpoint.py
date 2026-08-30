"""The Pickup endpoint: paging, and the carousel's daily window.

The feed itself must stay in stable rank order. Reshuffling a paged
list produces duplicates on one page and gaps on the next — the same
trap as a randomised sort with an offset. The carousel does not page,
so it can rotate; the feed cannot, so it does not.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.models import Base, PickupItem, PickupProfile  # noqa: E402
from app.routers import pickup as endpoint  # noqa: E402


@pytest.fixture()
def store(monkeypatch, tmp_path):
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

    monkeypatch.setattr(endpoint, "get_search_db", session_cm)

    def seed(count, drive="a", viewer="v1"):
        s = maker()
        try:
            s.add(PickupProfile(
                drive_id=drive, viewer_id=viewer,
                total=count, watch_history_checkpoint="x",
            ))
            for i in range(count):
                s.add(PickupItem(
                    drive_id=drive, viewer_id=viewer, rank=i + 1,
                    file_id=f"f{i:03d}", cluster_id="c0",
                    channel="clip_thumbnail", score=0.9,
                ))
            s.commit()
        finally:
            s.close()
    return seed


async def _call(**kw):
    kw.setdefault("drive", "a")
    kw.setdefault("viewer_id", "v1")
    kw.setdefault("limit", 12)
    kw.setdefault("offset", 0)
    kw.setdefault("window", None)
    kw.setdefault("date", None)
    return await endpoint.pickup_endpoint(**kw)


# ---------------------------------------------------------------------------
# Paging
# ---------------------------------------------------------------------------


async def test_pages_do_not_overlap_or_skip(store):
    store(100)

    first = await _call(limit=40, offset=0)
    second = await _call(limit=40, offset=40)

    assert len(first["file_ids"]) == 40
    assert len(second["file_ids"]) == 40
    assert set(first["file_ids"]).isdisjoint(second["file_ids"])
    assert first["file_ids"] + second["file_ids"] == [f"f{i:03d}" for i in range(80)]


async def test_paging_ignores_the_date(store):
    store(100)

    a = await _call(limit=40, offset=40, date="2026-01-01")
    b = await _call(limit=40, offset=40, date="2026-06-15")

    assert a["file_ids"] == b["file_ids"]


async def test_total_is_rows_held_not_the_page(store):
    store(100)

    got = await _call(limit=10)

    assert got["total"] == 100
    assert len(got["file_ids"]) == 10


async def test_an_offset_past_the_end_is_empty(store):
    store(10)

    got = await _call(limit=10, offset=50)

    assert got["file_ids"] == []
    assert got["total"] == 10


# ---------------------------------------------------------------------------
# The daily window
# ---------------------------------------------------------------------------


async def test_the_same_day_gives_the_same_window(store):
    store(100)

    a = await _call(window="daily", date="2026-03-02")
    b = await _call(window="daily", date="2026-03-02")

    assert a["file_ids"] == b["file_ids"]


async def test_adjacent_days_differ(store):
    store(100)

    a = await _call(window="daily", date="2026-03-02")
    b = await _call(window="daily", date="2026-03-03")

    assert a["file_ids"] != b["file_ids"]


async def test_the_window_is_drawn_from_the_top_of_the_feed(store):
    store(100)

    got = await _call(window="daily", date="2026-03-02")

    assert len(got["file_ids"]) == 12
    assert len(set(got["file_ids"])) == 12
    top = {f"f{i:03d}" for i in range(endpoint._WINDOW_POOL)}
    assert set(got["file_ids"]) <= top


async def test_a_short_feed_still_fills_the_window_without_repeats(store):
    store(15)

    got = await _call(window="daily", date="2026-03-02", limit=12)

    assert len(got["file_ids"]) == 12
    assert len(set(got["file_ids"])) == 12


async def test_a_feed_shorter_than_the_limit_returns_what_there_is(store):
    store(5)

    got = await _call(window="daily", date="2026-03-02", limit=12)

    assert len(got["file_ids"]) == 5


async def test_two_viewers_get_different_windows(store):
    store(100, viewer="v1")
    store(100, viewer="v2")

    a = await _call(window="daily", date="2026-03-02", viewer_id="v1")
    b = await _call(window="daily", date="2026-03-02", viewer_id="v2")

    assert a["file_ids"] != b["file_ids"]


async def test_a_malformed_date_does_not_error(store):
    store(100)

    got = await _call(window="daily", date="not-a-date")

    assert len(got["file_ids"]) == 12


# ---------------------------------------------------------------------------
# Cold and empty paths
# ---------------------------------------------------------------------------


async def test_no_viewer_id_returns_nothing(store):
    store(100)

    got = await _call(viewer_id=None)

    assert got == {"file_ids": [], "total": 0}


async def test_a_viewer_with_no_feed_returns_nothing(store):
    store(100, viewer="v1")

    got = await _call(viewer_id="v2")

    assert got == {"file_ids": [], "total": 0}


async def test_another_drive_is_not_served(store):
    store(100, drive="a")

    got = await _call(drive="b")

    assert got == {"file_ids": [], "total": 0}
