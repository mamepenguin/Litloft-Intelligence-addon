"""Pickup's not-interested marks: the endpoints, the feed, the sweep, purge and evaluate.

SPEC-ID: SPEC-ADDON-154
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402

from app import policy_client, purge  # noqa: E402
from app.models import IndexedFile, PickupItem, PickupProfile  # noqa: E402
from app.pickup import profile  # noqa: E402
from app.routers import pickup as endpoint  # noqa: E402
from app.workers import pickup as worker_mod  # noqa: E402
from tests.pickup_accuracy_rig import (  # noqa: E402
    CoreDB,
    SearchDB,
    at_angle,
    ts,
    utcnow,
    wire,
)
from tests.test_z_pickup_evaluate import _fingerprint, _run, _sections, _table, _today  # noqa: E402

TABLE = "pickup_not_interested"
OLD_MARK = "2026-01-02 03:04:05.000000"


def _ids(n: int, prefix: str = "f") -> list[str]:
    return [f"{prefix}{i:011d}" for i in range(n)]


class _Rig:
    def __init__(self, core: CoreDB, search: SearchDB):
        self.core = core
        self.search = search
        self.worker = worker_mod.PickupWorker()
        app = FastAPI()
        app.include_router(endpoint.router)
        self.client = TestClient(app)

    # --- the two endpoints and the feed read, as the proxy calls them ---

    def put(self, file_id, *, drive="a", viewer="v1"):
        return self.client.put(f"/pickup/not-interested/{file_id}", headers=self._headers(drive, viewer))

    def delete(self, file_id, *, drive="a", viewer="v1"):
        return self.client.delete(f"/pickup/not-interested/{file_id}", headers=self._headers(drive, viewer))

    def read(self, *, drive="a", viewer="v1", **params):
        params.setdefault("limit", 40)
        r = self.client.get("/pickup", params=params, headers=self._headers(drive, viewer))
        assert r.status_code == 200, r.text
        return r.json()

    @staticmethod
    def _headers(drive, viewer):
        h = {"X-Lit-Drive": drive}
        if viewer is not None:
            h["X-Lit-Viewer-Id"] = viewer
        return h

    # --- marks, seeded and read as rows ---

    def mark(self, file_id, *, drive="a", viewer="v1", marked_at=OLD_MARK):
        self.search.execute(
            f"INSERT INTO {TABLE} (drive_id, viewer_id, file_id, marked_at) VALUES (:d, :v, :f, :m)",
            {"d": drive, "v": viewer, "f": file_id, "m": marked_at},
        )

    def unmark(self, file_id, *, drive="a", viewer="v1"):
        self.search.execute(
            f"DELETE FROM {TABLE} WHERE drive_id = :d AND viewer_id = :v AND file_id = :f",
            {"d": drive, "v": viewer, "f": file_id},
        )

    def marks(self):
        return sorted(self.search.query(
            f"SELECT drive_id, viewer_id, file_id, marked_at FROM {TABLE}"
        ))

    def unreadable_marks(self):
        self.search.execute(f"ALTER TABLE {TABLE} RENAME TO {TABLE}_away")
        self.search.execute(f"CREATE TABLE {TABLE} (unrelated INTEGER)")

    def readable_marks(self):
        self.search.execute(f"DROP TABLE {TABLE}")
        self.search.execute(f"ALTER TABLE {TABLE}_away RENAME TO {TABLE}")

    # --- a stored feed, as the sweep leaves it ---

    def feed(self, file_ids, *, drive="a", viewer="v1"):
        s = self.search.sessions()
        try:
            s.query(PickupItem).filter_by(drive_id=drive, viewer_id=viewer).delete()
            s.query(PickupProfile).filter_by(drive_id=drive, viewer_id=viewer).delete()
            s.add(PickupProfile(
                drive_id=drive, viewer_id=viewer, total=len(file_ids), watch_history_checkpoint="x",
            ))
            for rank, file_id in enumerate(file_ids, start=1):
                s.add(PickupItem(
                    drive_id=drive, viewer_id=viewer, rank=rank, file_id=file_id,
                    cluster_id="c0", channel="clip_thumbnail", score=0.9,
                ))
            s.commit()
        finally:
            s.close()

    def stored(self, *, drive="a", viewer="v1"):
        s = self.search.sessions()
        try:
            return [
                r.file_id for r in s.query(PickupItem)
                .filter_by(drive_id=drive, viewer_id=viewer)
                .order_by(PickupItem.rank).all()
            ]
        finally:
            s.close()

    def header(self, *, drive="a", viewer="v1"):
        s = self.search.sessions()
        try:
            return s.query(PickupProfile).filter_by(drive_id=drive, viewer_id=viewer).first()
        finally:
            s.close()

    # --- history and candidates for a sweep ---

    def files(self, file_ids, *, drive="a"):
        for file_id in file_ids:
            self.core.add_file(file_id, drive=drive)

    def video(self, file_id, angle, *, drive="a"):
        self.core.add_file(file_id, drive=drive)
        self.search.index(file_id, at_angle(angle), drive=drive)

    def covered(self, file_id, *, viewer="v1", hours=24.0):
        self.core.watch(file_id, viewer=viewer, played_at=utcnow() - timedelta(hours=hours), position=0.0)
        self.core.segments(file_id, [5.0] * 10, viewer=viewer)

    def history(self, *, viewers=("v1",), drive="a"):
        for i in range(5):
            file_id = f"h{i}{drive}"
            self.video(file_id, 0.0, drive=drive)
            for viewer in viewers:
                self.covered(file_id, viewer=viewer, hours=24.0 * (1 + i))

    async def sweep(self, drive="a"):
        time.sleep(0.01)
        await self.worker._compute_for_drive(drive)


@pytest.fixture()
def rig(monkeypatch, tmp_path):
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    wire(monkeypatch, core, search)
    return _Rig(core, search)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Rule 1: the table
# ---------------------------------------------------------------------------


def test_spec_addon_154_rule_1_one_mark_per_drive_viewer_and_file(rig):
    """SPEC-ADDON-154 rule 1: the primary key is (drive_id, viewer_id, file_id)."""
    rig.mark("abcdefghijkl")
    rig.mark("abcdefghijkl", viewer="v2")
    rig.mark("abcdefghijkl", drive="b")
    rig.mark("abcdefghijk0")

    with pytest.raises(IntegrityError):
        rig.mark("abcdefghijkl")

    assert len(rig.marks()) == 4


# ---------------------------------------------------------------------------
# Rule 2: the proxy route
# ---------------------------------------------------------------------------


def test_spec_addon_154_rule_2_the_route_is_gated_like_the_feed():
    """SPEC-ADDON-154 rule 2 / I2: PUT and DELETE, behind the same `index` gate as /pickup."""
    manifest = json.loads((Path(__file__).resolve().parents[1] / "manifest.json").read_text("utf-8"))
    routes = manifest["proxy"]["routes"]
    feed = next(r for r in routes if r["path"] == "/pickup")
    marks = [r for r in routes if r["path"] == "/pickup/not-interested/{file_id}"]

    assert sorted(m for r in marks for m in r["methods"]) == ["DELETE", "PUT"]
    for route in marks:
        assert {k: v for k, v in route.items() if k not in ("path", "methods")} == {
            k: v for k, v in feed.items() if k not in ("path", "methods")
        }
        assert route["pre_check"] == {"type": "addon_feature", "feature": "index"}


# ---------------------------------------------------------------------------
# Rule 3 / I2: the endpoints
# ---------------------------------------------------------------------------


def test_spec_addon_154_rule_3_put_stores_a_mark_for_the_viewer_and_drive(rig):
    """SPEC-ADDON-154 rule 3: 204, one row keyed by the request's drive and viewer."""
    r = rig.put("abcdefghijkl", drive="a", viewer="v1")

    assert r.status_code == 204, r.text
    rows = rig.marks()
    assert [row[:3] for row in rows] == [("a", "v1", "abcdefghijkl")]
    assert rows[0][3]


def test_spec_addon_154_i2_a_repeated_put_keeps_the_first_marked_at(rig):
    """SPEC-ADDON-154 I2: a repeat answers 204 and leaves the row as it was."""
    rig.mark("abcdefghijkl", marked_at=OLD_MARK)

    first = rig.put("abcdefghijkl")
    second = rig.put("abcdefghijkl")

    assert (first.status_code, second.status_code) == (204, 204)
    assert rig.marks() == [("a", "v1", "abcdefghijkl", OLD_MARK)]


def test_spec_addon_154_i2_delete_removes_only_that_viewers_mark_in_that_drive(rig):
    """SPEC-ADDON-154 rule 3 / I1: DELETE touches one (drive, viewer, file)."""
    for drive, viewer, file_id in (
        ("a", "v1", "abcdefghijkl"), ("a", "v2", "abcdefghijkl"),
        ("b", "v1", "abcdefghijkl"), ("a", "v1", "abcdefghijk0"),
    ):
        rig.mark(file_id, drive=drive, viewer=viewer)

    r = rig.delete("abcdefghijkl", drive="a", viewer="v1")

    assert r.status_code == 204, r.text
    assert [row[:3] for row in rig.marks()] == [
        ("a", "v1", "abcdefghijk0"), ("a", "v2", "abcdefghijkl"), ("b", "v1", "abcdefghijkl"),
    ]


def test_spec_addon_154_i2_delete_without_a_mark_answers_204(rig):
    """SPEC-ADDON-154 I2: DELETE with no mark is 204 and writes nothing."""
    rig.mark("abcdefghijk0")

    r = rig.delete("abcdefghijkl")

    assert r.status_code == 204, r.text
    assert [row[:3] for row in rig.marks()] == [("a", "v1", "abcdefghijk0")]


@pytest.mark.parametrize("method", ["put", "delete"])
def test_spec_addon_154_i2_without_a_viewer_answers_400_and_writes_nothing(rig, method):
    """SPEC-ADDON-154 I2: 400 viewer_required; the table is unchanged."""
    rig.mark("abcdefghijkl", viewer="v1")
    before = rig.marks()

    r = getattr(rig, method)("abcdefghijkl", viewer=None)

    assert r.status_code == 400, r.text
    assert "viewer_required" in r.text
    assert rig.marks() == before


@pytest.mark.parametrize("method", ["put", "delete"])
@pytest.mark.parametrize(
    "file_id",
    ["abcdefghijk", "abcdefghijklm", "abcdefghij.k", "abcdefghij k", "abcdefghijk~", "あいうえおかきくけこさし"],
    ids=["11 chars", "13 chars", "a dot", "a space", "a tilde", "outside ASCII"],
)
def test_spec_addon_154_i2_a_file_id_outside_the_core_alphabet_is_422(rig, method, file_id):
    """SPEC-ADDON-154 I2: ^[A-Za-z0-9_-]{12}$ or 422, writing nothing."""
    rig.mark("abcdefghijkl")
    before = rig.marks()

    r = getattr(rig, method)(file_id)

    assert r.status_code == 422, r.text
    assert rig.marks() == before


@pytest.mark.parametrize(
    "file_id", ["ABCDEFGHIJKL", "012345678901", "a_b-c_d-e_f-", "____________", "------------"],
)
def test_spec_addon_154_i2_every_id_of_the_core_alphabet_is_accepted(rig, file_id):
    """SPEC-ADDON-154 I2: the whole alphabet, including `_` and `-`, round-trips."""
    assert rig.put(file_id).status_code == 204
    assert [row[2] for row in rig.marks()] == [file_id]
    assert rig.delete(file_id).status_code == 204
    assert rig.marks() == []


def test_spec_addon_154_i1_a_mark_and_its_undo_write_no_core_table(rig, tmp_path):
    """SPEC-ADDON-154 rule 5 / I1: no watched state, engagement or seed moves; the core DB is untouched."""
    rig.history()
    rig.video("cand00000000", 5.0)
    rig.core.engine.dispose()
    core_before = _sha(rig.core.path)
    watched_before = profile.watched_file_ids("a", "v1")

    assert rig.put("cand00000000").status_code == 204
    assert profile.watched_file_ids("a", "v1") == watched_before
    assert rig.delete("cand00000000").status_code == 204

    rig.core.engine.dispose()
    assert _sha(rig.core.path) == core_before
    assert "cand00000000" not in watched_before


# ---------------------------------------------------------------------------
# Rule 4 / SPEC-ADDON-149 I4 as revised: the feed read
# ---------------------------------------------------------------------------

FEED = _ids(20)


def _mixed(rig: _Rig) -> list[str]:
    """A feed of 20: one watched, three marked by v1 in drive a, two marked elsewhere."""
    rig.files(FEED)
    rig.covered(FEED[2])
    for i in (4, 9, 19):
        rig.mark(FEED[i])
    rig.mark(FEED[5], viewer="v2")
    rig.mark(FEED[6], drive="b")
    rig.feed(FEED)
    return [f for i, f in enumerate(FEED) if i not in (2, 4, 9, 19)]


def test_spec_addon_149_i4_revised_the_feed_read_removes_marks_and_watched_in_rank_order(rig):
    """SPEC-ADDON-149 I4 as revised: stored feed minus exclusion set minus marks; total is the filtered length."""
    expected = _mixed(rig)

    assert rig.read(limit=40, offset=0) == {"file_ids": expected, "total": 16}


@pytest.mark.parametrize(
    ("offset", "limit", "window"),
    [(0, 5, slice(0, 5)), (5, 5, slice(5, 10)), (10, 12, slice(10, 16)), (15, 1, slice(15, 16)),
     (16, 5, slice(16, 16))],
    ids=["first page", "second page", "last partial page", "last item", "at the end"],
)
def test_spec_addon_154_rule_4_pages_of_the_filtered_list_stay_full(rig, offset, limit, window):
    """SPEC-ADDON-154 rule 4: marks are removed before paging."""
    expected = _mixed(rig)

    assert rig.read(offset=offset, limit=limit) == {"file_ids": expected[window], "total": 16}


def test_spec_addon_154_i1_another_viewers_or_drives_marks_filter_nothing(rig):
    """SPEC-ADDON-154 I1: a mark is this viewer's, in this drive."""
    ids = _ids(3, "y")
    idsb = _ids(3, "z")
    rig.files(ids)
    rig.files(idsb, drive="b")
    rig.mark(ids[0], viewer="v2")
    rig.mark(ids[1], viewer="v1")
    rig.mark(ids[2], viewer="v1", drive="b")
    rig.mark(idsb[2], viewer="v1", drive="b")
    for viewer in ("v1", "v2"):
        rig.feed(ids, drive="a", viewer=viewer)
        rig.feed(idsb, drive="b", viewer=viewer)

    assert rig.read(viewer="v1")["file_ids"] == [ids[0], ids[2]]
    assert rig.read(viewer="v2")["file_ids"] == [ids[1], ids[2]]
    assert rig.read(viewer="v1", drive="b")["file_ids"] == [idsb[0], idsb[1]]
    assert rig.read(viewer="v2", drive="b")["file_ids"] == idsb


def test_spec_addon_154_rule_4_a_mark_leaves_the_feed_at_once_and_its_undo_restores_the_rank(rig):
    """SPEC-ADDON-154 rule 4: PUT then the next read drops it; DELETE puts it back where it ranked."""
    ids = _ids(5, "x")
    rig.files(ids)
    rig.feed(ids)

    assert rig.put(ids[2]).status_code == 204
    assert rig.read() == {"file_ids": [ids[0], ids[1], ids[3], ids[4]], "total": 4}

    assert rig.delete(ids[2]).status_code == 204
    assert rig.read() == {"file_ids": ids, "total": 5}


def test_spec_addon_154_rule_4_every_item_marked_leaves_an_empty_feed(rig):
    """SPEC-ADDON-154 rule 4: zero left, total 0, for paging and for the daily window."""
    ids = _ids(2, "e")
    rig.files(ids)
    for file_id in ids:
        rig.mark(file_id)
    rig.feed(ids)

    assert rig.read(limit=12) == {"file_ids": [], "total": 0}
    assert rig.read(limit=12, window="daily", date="2026-11-19")["file_ids"] == []


@pytest.mark.parametrize("stored", [60, 51, 20], ids=["over 40 left", "exactly 40 left", "under 40 left"])
def test_spec_addon_149_i4_revised_the_daily_window_is_drawn_from_the_list_without_marks(rig, stored):
    """SPEC-ADDON-149 I4 as revised: the window equals that of a feed that never held the marked ids."""
    ids = _ids(stored, "w")
    marked = set(ids[:10]) | {ids[-1]}
    kept = [f for f in ids if f not in marked]
    rig.files(ids)

    rig.feed(kept)
    oracle = rig.read(limit=12, window="daily", date="2026-11-19")

    for file_id in sorted(marked):
        rig.mark(file_id)
    rig.feed(ids)
    got = rig.read(limit=12, window="daily", date="2026-11-19")

    assert got == oracle
    assert set(got["file_ids"]).isdisjoint(marked)
    assert got["total"] == len(kept)


async def test_spec_addon_154_an_unreadable_mark_table_fails_the_feed_read(rig):
    """SPEC-ADDON-154 failure cases / SPEC-ADDON-149: never served unfiltered."""
    ids = _ids(2, "u")
    rig.files(ids)
    rig.mark(ids[0])
    rig.feed(ids)
    rig.unreadable_marks()

    with pytest.raises(Exception):
        await endpoint.pickup_endpoint(
            drive="a", viewer_id="v1", limit=12, offset=0, window=None, date=None,
        )


# ---------------------------------------------------------------------------
# Rule 4 / SPEC-ADDON-149 I5, SPEC-ADDON-102 I1, SPEC-ADDON-095 I5 as revised: the sweep
# ---------------------------------------------------------------------------


async def test_spec_addon_149_i5_revised_the_sweep_and_the_feed_read_drop_marks_and_watched_ids_does_not(rig):
    """SPEC-ADDON-149 I5 / SPEC-ADDON-102 I1 as revised: one set for both readers; watched_file_ids holds no marks."""
    rig.history(viewers=("v1", "v2"))
    for file_id, angle in (("c0", 4.0), ("c1", 8.0), ("c2", 12.0), ("c3", 16.0)):
        rig.video(file_id, angle)
    rig.mark("c1", viewer="v1")
    rig.mark("c2", viewer="v1", drive="b")

    await rig.sweep()

    assert rig.stored(viewer="v1") == ["c0", "c2", "c3"]
    assert rig.stored(viewer="v2") == ["c0", "c1", "c2", "c3"]
    assert rig.read(viewer="v1") == {"file_ids": ["c0", "c2", "c3"], "total": 3}
    assert "c1" not in profile.watched_file_ids("a", "v1")


async def test_spec_addon_095_i5_revised_adding_and_removing_a_mark_rebuilds_once_each(rig):
    """SPEC-ADDON-095 I5 / SPEC-ADDON-154 rule 4: the checkpoint moves with the viewer's marks, then holds."""
    rig.history()
    for file_id, angle in (("c0", 4.0), ("c1", 8.0), ("c2", 12.0)):
        rig.video(file_id, angle)
    await rig.sweep()
    assert rig.stored() == ["c0", "c1", "c2"]
    base = rig.header().watch_history_checkpoint

    rig.mark("c1")
    await rig.sweep()
    marked = rig.header()
    assert rig.stored() == ["c0", "c2"]
    assert marked.watch_history_checkpoint != base

    await rig.sweep()
    held = rig.header()
    assert held.watch_history_checkpoint == marked.watch_history_checkpoint
    assert held.computed_at == marked.computed_at

    rig.unmark("c1")
    await rig.sweep()
    assert rig.stored() == ["c0", "c1", "c2"]
    assert rig.header().watch_history_checkpoint == base


async def test_spec_addon_154_i1_another_viewers_or_drives_mark_moves_no_checkpoint(rig):
    """SPEC-ADDON-154 I1: v2's mark, or v1's mark in drive b, leaves v1's drive-a feed as it was."""
    rig.history()
    for file_id, angle in (("c0", 4.0), ("c1", 8.0), ("c2", 12.0)):
        rig.video(file_id, angle)
    rig.mark("c2")
    await rig.sweep()
    assert rig.stored() == ["c0", "c1"]
    before = rig.header()

    rig.mark("c1", viewer="v2")
    rig.mark("c1", drive="b")
    await rig.sweep()
    after = rig.header()

    assert rig.stored() == ["c0", "c1"]
    assert after.watch_history_checkpoint == before.watch_history_checkpoint
    assert after.computed_at == before.computed_at


async def test_spec_addon_154_i3_a_sweep_that_cannot_read_marks_writes_nothing_and_retries(rig):
    """SPEC-ADDON-154 rule 6 / I3: a rebuild is due, yet no feed and no checkpoint are written; the next sweep rebuilds."""
    rig.history()
    rig.video("c0", 4.0)
    rig.video("c1", 8.0)
    await rig.sweep()
    before = rig.header()
    assert rig.stored() == ["c0", "c1"]

    rig.mark("c1")
    rig.video("c2", 12.0)
    rig.unreadable_marks()
    try:
        await rig.sweep()
    except Exception:
        pass

    after = rig.header()
    assert rig.stored() == ["c0", "c1"]
    assert after.watch_history_checkpoint == before.watch_history_checkpoint
    assert after.computed_at == before.computed_at

    rig.readable_marks()
    await rig.sweep()

    assert rig.stored() == ["c0", "c2"]


# ---------------------------------------------------------------------------
# Rule 7 / I3: purge
# ---------------------------------------------------------------------------


@pytest.fixture()
def purging(rig, monkeypatch):
    policy_client.reset_cache()

    async def policy(drive, feature):
        return drive != "off"

    monkeypatch.setattr(policy_client, "is_feature_enabled", policy)
    monkeypatch.setattr(purge, "is_feature_enabled", policy, raising=False)
    purged: list[str] = []
    monkeypatch.setattr("app.indexer._purge_file", lambda fid: purged.append(fid))
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)
    yield purged
    policy_client.reset_cache()


def _indexed(rig: _Rig, file_id: str, drive: str) -> None:
    s = rig.search.sessions()
    try:
        s.add(IndexedFile(
            file_id=file_id, drive=drive, filename=f"{file_id}.mp4",
            file_path=f"/drives/{drive}/{file_id}.mp4", file_type="video",
            mime_type="video/mp4", file_size=1, active=True,
        ))
        s.commit()
    finally:
        s.close()


def test_spec_addon_154_rule_7_purge_drive_deletes_a_drives_marks_without_indexed_files(rig, purging):
    """SPEC-ADDON-154 rule 7 / I3: every viewer's marks of that drive, and no other drive's."""
    rig.mark("abcdefghijkl", drive="off", viewer="v1")
    rig.mark("abcdefghijk0", drive="off", viewer="v2")
    rig.mark("abcdefghijkl", drive="on", viewer="v1")

    purge.purge_drive("off")

    assert [row[:3] for row in rig.marks()] == [("on", "v1", "abcdefghijkl")]


async def test_spec_addon_154_i3_the_startup_purge_visits_a_drive_that_only_has_marks(rig, purging):
    """SPEC-ADDON-154 rule 7 / I3: a policy-off drive with marks and no indexed files is purged."""
    _indexed(rig, "onfile000000", "on")
    rig.mark("abcdefghijkl", drive="off")
    rig.mark("abcdefghijkl", drive="on")
    rig.mark("abcdefghijkl", drive="elsewhere")

    await purge.purge_disabled_drives()

    assert [row[:3] for row in rig.marks()] == [
        ("elsewhere", "v1", "abcdefghijkl"), ("on", "v1", "abcdefghijkl"),
    ]
    assert purging == []


async def test_spec_addon_154_rule_7_a_failed_mark_delete_is_retried_on_the_next_startup(rig, purging):
    """SPEC-ADDON-154 rule 7: the purge goes on past the failure; the drive is visited again."""
    _indexed(rig, "offfile00000", "off")
    _indexed(rig, "offfile00001", "off")
    rig.mark("abcdefghijkl", drive="off")
    rig.mark("abcdefghijkl", drive="on")
    rig.search.execute(
        f"CREATE TRIGGER refuse BEFORE DELETE ON {TABLE} BEGIN SELECT RAISE(ABORT, 'refused'); END"
    )

    await purge.purge_disabled_drives()

    assert sorted(purging) == ["offfile00000", "offfile00001"]
    assert [row[:3] for row in rig.marks()] == [
        ("off", "v1", "abcdefghijkl"), ("on", "v1", "abcdefghijkl"),
    ]

    rig.search.execute("DROP TRIGGER refuse")
    rig.search.execute("DELETE FROM indexed_files WHERE drive = 'off'")
    await purge.purge_disabled_drives()

    assert [row[:3] for row in rig.marks()] == [("on", "v1", "abcdefghijkl")]


# ---------------------------------------------------------------------------
# Rule 8 / SPEC-ADDON-098 I9 as revised: evaluate
# ---------------------------------------------------------------------------

_UNCHANGED = ("before", "depth", "video", "random", "newest")


def _replay_dataset(tmp_path):
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    t0 = _today() - timedelta(days=1)
    old = _today() - timedelta(days=30)
    for i in range(5):
        core.add_file(f"h{i}", created_at=old)
        core.watch(f"h{i}", played_at=t0 - timedelta(days=5, hours=i), position=0.0)
        core.segments(f"h{i}", [5.0] * 10)
        search.index(f"h{i}", at_angle(0.0))
    for file_id, angle, hour in (("X", 18.0, 1), ("Y", 19.0, 2)):
        core.add_file(file_id, created_at=_today() - timedelta(days=25))
        core.watch(file_id, played_at=t0 + timedelta(hours=hour), position=0.0)
        core.segments(file_id, [5.0] * 10)
        search.index(file_id, at_angle(angle))
    return core, search, t0


def _mark_raw(search: SearchDB, file_id: str, marked_at, *, viewer="v1", drive="a"):
    search.execute(
        f"INSERT INTO {TABLE} (drive_id, viewer_id, file_id, marked_at) VALUES (:d, :v, :f, :m)",
        {"d": drive, "v": viewer, "f": file_id, "m": ts(marked_at)},
    )


def test_spec_addon_098_i9_revised_replay_excludes_marks_made_before_the_cut_point(tmp_path):
    """SPEC-ADDON-154 rule 8 / SPEC-ADDON-098 I9 as revised: after, grid and fresh rows only; marks before t only."""
    core, search, t0 = _replay_dataset(tmp_path)
    _mark_raw(search, "X", t0 - timedelta(hours=1))
    _mark_raw(search, "Y", t0 + timedelta(minutes=30))
    _mark_raw(search, "Y", t0 - timedelta(hours=1), viewer="v2")
    _mark_raw(search, "Y", t0 - timedelta(hours=1), drive="b")
    core.engine.dispose()
    search.engine.dispose()

    result = _run(core.path, search.path, "replay", "--drive", "a", "--days", "1", "--horizon", "1")

    assert result.returncode == 0, result.stderr
    v1 = _sections(result.stdout)["v1"]
    assert "candidate videos: 2 .. 2" in [line.strip() for line in v1]
    _, rows = _table(v1)
    rows = {name.rstrip("*"): cells for name, cells in rows.items()}
    for config in _UNCHANGED:
        assert rows[config] == ["1", "0", "2", "1.000", "1.000", "1.000"], config
    excluding = [name for name in rows if name not in _UNCHANGED]
    assert "after" in excluding
    for config in excluding:
        assert rows[config] == ["1", "0", "2", "0.500", "0.500", "1.000"], config


def test_spec_addon_154_rule_8_evaluate_fails_and_writes_nothing_when_marks_cannot_be_read(tmp_path):
    """SPEC-ADDON-154 rule 8: an error exit, both databases unchanged."""
    core, search, t0 = _replay_dataset(tmp_path)
    search.execute(f"DROP TABLE IF EXISTS {TABLE}")
    search.execute(f"CREATE TABLE {TABLE} (unrelated INTEGER)")
    core.engine.dispose()
    search.engine.dispose()
    before = _fingerprint(core.path, search.path)

    result = _run(core.path, search.path, "replay", "--drive", "a", "--days", "1", "--horizon", "1")

    assert result.returncode != 0, result.stdout
    assert _fingerprint(core.path, search.path) == before
