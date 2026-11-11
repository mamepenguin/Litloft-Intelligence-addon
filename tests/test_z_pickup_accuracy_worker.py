"""The Pickup worker end to end: videos only, engagement, settling, and the feed log.

SPEC-ID: SPEC-ADDON-095
SPEC-ID: SPEC-ADDON-096
SPEC-ID: SPEC-ADDON-097
"""

from __future__ import annotations

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
from app.pickup import profile  # noqa: E402
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

    def watched(self, file_id, *, angle=None, kind="video", channel="clip_thumbnail",
                age_days=1.0, viewer="v1", file_kw=None, **watch_kw):
        self.core.add_file(file_id, kind=kind, **(file_kw or {}))
        self.core.watch(file_id, viewer=viewer,
                        played_at=utcnow() - timedelta(days=age_days), **watch_kw)
        if angle is not None:
            self.search.index(file_id, at_angle(angle), channel=channel, kind=kind)

    def candidate(self, file_id, angle, *, kind="video", channel="clip_thumbnail", **file_kw):
        self.core.add_file(file_id, kind=kind, **file_kw)
        self.search.index(file_id, at_angle(angle), channel=channel, kind=kind)

    def kept(self, prefix="h", n=5, angle=0.0, channel="clip_thumbnail"):
        for i in range(n):
            self.watched(f"{prefix}{i}", angle=angle, channel=channel, age_days=1 + i)

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

    def logs(self, drive="a", viewer="v1"):
        return self.search.query(
            "SELECT computed_at, profile_version, file_ids FROM pickup_feed_log"
            " WHERE drive_id = :d AND viewer_id = :v ORDER BY computed_at",
            {"d": drive, "v": viewer},
        )


@pytest.fixture()
def rig(monkeypatch, tmp_path):
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    wire(monkeypatch, core, search)
    return _Rig(core, search)


def _as_naive(value) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return datetime.fromisoformat(str(value)).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Videos only
# ---------------------------------------------------------------------------


async def test_spec_addon_096_i3_a_stored_feed_holds_live_videos_only(rig):
    """SPEC-ADDON-096 I3: documents, images, trashed and missing videos never appear."""
    rig.kept()
    for file_id, angle in (("c0", 4.0), ("c1", 8.0), ("c2", 12.0)):
        rig.candidate(file_id, angle)
    rig.candidate("doc", 4.5, kind="document")
    rig.candidate("img", 5.5, kind="image")
    rig.candidate("trashed", 3.0, deleted=True)
    rig.candidate("missing", 3.5, missing=True)

    await rig.sweep()

    assert rig.feed() == ["c0", "c1", "c2"]
    assert rig.header().total == 3


async def test_spec_addon_096_document_history_does_not_shape_the_feed(rig):
    """SPEC-ADDON-096: documents watched near another subject pull no videos from it."""
    rig.kept(angle=0.0)
    for i in range(5):
        rig.watched(f"doc{i}", angle=90.0, kind="document", age_days=0.5)
    rig.candidate("near_videos", 5.0)
    rig.candidate("near_docs", 88.0)

    await rig.sweep()

    assert rig.feed() == ["near_videos"]


async def test_spec_addon_095_brief_and_dismissed_videos_do_not_shape_the_feed(rig):
    """SPEC-ADDON-095: a video opened briefly or dismissed pulls nothing similar in."""
    rig.kept(angle=0.0)
    for i in range(3):
        rig.watched(f"brief{i}", angle=90.0, age_days=0.5, position=1.0)
    for i in range(2):
        rig.watched(f"gone{i}", angle=90.0, age_days=0.5,
                    dismissed_at=ts(utcnow()))
    rig.candidate("near_kept", 5.0)
    rig.candidate("near_brief", 85.0)

    await rig.sweep()

    assert rig.feed() == ["near_kept"]


async def test_spec_addon_095_i2_an_unengaged_or_document_row_only_excludes(rig):
    """SPEC-ADDON-095 I2: the feed is the old one with that file taken out."""
    rig.kept(angle=0.0)
    for file_id, angle in (("p3", 3.0), ("m5", -5.0), ("p9", 9.0), ("m12", -12.0),
                           ("p16", 16.0), ("p22", 22.0), ("brief", 25.0)):
        rig.candidate(file_id, angle)
    await rig.sweep()
    base = rig.feed()
    assert "brief" in base

    rig.core.watch("brief", played_at=utcnow() - timedelta(hours=1), position=1.0)
    rig.watched("doc", angle=90.0, kind="document", age_days=0.1)
    await rig.sweep()

    assert rig.feed() == [f for f in base if f != "brief"]


# ---------------------------------------------------------------------------
# Settled and retrying empty feeds
# ---------------------------------------------------------------------------


def _documents_only(rig):
    for i in range(5):
        rig.watched(f"d{i}", angle=0.0, kind="document", age_days=1 + i)


def _brief_only(rig):
    for i in range(5):
        rig.watched(f"b{i}", angle=0.0, age_days=1 + i, position=1.0)


def _dismissed_only(rig):
    for i in range(5):
        rig.watched(f"x{i}", angle=0.0, age_days=1 + i, dismissed_at=ts(utcnow()))


@pytest.mark.parametrize(
    "history", [_documents_only, _brief_only, _dismissed_only],
    ids=["documents only", "brief opens only", "dismissed only"],
)
async def test_spec_addon_095_i6_no_kept_row_settles_an_empty_feed(rig, history):
    """SPEC-ADDON-095 I6: empty, checkpoint stored, not rebuilt again that day."""
    history(rig)
    for file_id, angle in (("c0", 5.0), ("c1", 10.0), ("c2", 15.0)):
        rig.candidate(file_id, angle)

    await rig.sweep()
    first = rig.header()

    assert rig.feed() == []
    assert first.total == 0
    assert first.watch_history_checkpoint is not None

    await rig.sweep()
    second = rig.header()

    assert second.computed_at == first.computed_at
    assert second.watch_history_checkpoint == first.watch_history_checkpoint


def _only_watched_videos_are_candidates(rig):
    rig.kept(angle=0.0)
    for i, angle in enumerate((3.0, 6.0, 9.0)):
        rig.candidate(f"doc{i}", angle, kind="document")


def _kept_rows_without_vectors(rig):
    for i in range(5):
        rig.watched(f"novec{i}", angle=None, age_days=1 + i)
    for i in range(5):
        rig.watched(f"doc{i}", angle=0.0, kind="document", age_days=0.5)
    for i, angle in enumerate((3.0, 6.0, 9.0)):
        rig.candidate(f"c{i}", angle)


def _nothing_survives(rig):
    rig.kept(angle=0.0)
    rig.candidate("far", 90.0)
    for i, angle in enumerate((3.0, 6.0)):
        rig.candidate(f"doc{i}", angle, kind="document")


@pytest.mark.parametrize(
    "setup", [_only_watched_videos_are_candidates, _kept_rows_without_vectors, _nothing_survives],
    ids=["only watched videos are candidates", "kept rows but no vector", "nothing passes the floor"],
)
async def test_spec_addon_095_i6_an_empty_feed_with_kept_rows_retries(rig, setup):
    """SPEC-ADDON-095 I6: empty with a null checkpoint, rebuilt on every sweep."""
    setup(rig)

    await rig.sweep()
    first = rig.header()

    assert rig.feed() == []
    assert first.total == 0
    assert first.watch_history_checkpoint is None

    await rig.sweep()

    assert rig.header().computed_at != first.computed_at


async def test_spec_addon_096_an_unreadable_video_set_leaves_the_feed(rig):
    """SPEC-ADDON-096: when the live-video read fails, the stored feed and checkpoint stay."""
    rig.kept()
    rig.candidate("c0", 5.0)
    await rig.sweep()
    before = rig.header()
    assert before.total == 1

    rig.core.execute(
        "CREATE TABLE files_without_kind AS SELECT id, filename, title, drive,"
        " folder_path, file_path, file_size, mime_type, duration, liked_at,"
        " is_favorite, created_at, updated_at, deleted_at, missing_since FROM files"
    )
    rig.core.execute("DROP TABLE files")
    rig.core.execute("ALTER TABLE files_without_kind RENAME TO files")
    rig.core.watch("c0", played_at=utcnow() - timedelta(hours=1))
    try:
        await rig.sweep()
    except Exception:
        pass

    after = rig.header()
    assert rig.feed() == ["c0"]
    assert after.watch_history_checkpoint == before.watch_history_checkpoint
    assert after.computed_at == before.computed_at


# ---------------------------------------------------------------------------
# The rebuild trigger
# ---------------------------------------------------------------------------


def _like(rig):
    rig.core.execute("UPDATE files SET liked_at = :t WHERE id = 'h0'", {"t": ts(utcnow())})


def _unlike(rig):
    rig.core.execute("UPDATE files SET liked_at = NULL WHERE id = 'h0'")


def _favourite(rig):
    rig.core.execute("UPDATE files SET is_favorite = 1 WHERE id = 'h0'")


def _unfavourite(rig):
    rig.core.execute("UPDATE files SET is_favorite = 0 WHERE id = 'h0'")


def _played_ranges(rig):
    rig.core.segments("h0", [30.0])


def _dismiss(rig):
    rig.core.execute(
        "UPDATE watch_history SET dismissed_at = :t WHERE file_id = 'h0'",
        {"t": ts(utcnow())},
    )


@pytest.mark.parametrize(
    ("initial", "change"),
    [
        ({}, _like),
        ({"liked": True}, _unlike),
        ({}, _favourite),
        ({"favourite": True}, _unfavourite),
        ({}, _played_ranges),
        ({}, _dismiss),
    ],
    ids=["like", "unlike", "favourite", "unfavourite", "played ranges", "dismiss"],
)
async def test_spec_addon_095_i5_a_change_without_a_new_play_rebuilds_once(rig, initial, change):
    """SPEC-ADDON-095 I5: the checkpoint moves, then holds on an unchanged history."""
    rig.watched("h0", angle=0.0, age_days=1, file_kw=initial)
    for i in range(1, 5):
        rig.watched(f"h{i}", angle=0.0, age_days=1 + i)
    for file_id, angle in (("c0", 5.0), ("c1", 10.0)):
        rig.candidate(file_id, angle)
    await rig.sweep()
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


async def test_spec_addon_095_i5_the_checkpoint_ignores_row_order_and_float_noise(monkeypatch, tmp_path):
    """SPEC-ADDON-095 I5: the same history hashes the same, however it was written."""
    checkpoints = []
    played_at = utcnow() - timedelta(days=3)
    for name, order, seconds in (
        ("one", range(5), [0.1, 0.2, 0.3]),
        ("two", reversed(range(5)), [0.3, 0.2, 0.1]),
    ):
        (tmp_path / name).mkdir()
        rig = _Rig(CoreDB(tmp_path / name / "litloft.db"), SearchDB(tmp_path / name / "search.db"))
        wire(monkeypatch, rig.core, rig.search)
        for i in order:
            rig.core.add_file(f"h{i}")
            rig.core.watch(f"h{i}", played_at=played_at)
            rig.search.index(f"h{i}", at_angle(0.0))
        rig.candidate("c0", 5.0)
        await rig.sweep()
        unchanged = rig.header().watch_history_checkpoint

        rig.core.segments("h0", seconds)
        await rig.sweep()
        checkpoints.append((unchanged, rig.header().watch_history_checkpoint))

    (first_before, first_after), (second_before, second_after) = checkpoints
    assert first_after != first_before
    assert first_before == second_before
    assert first_after == second_after


# ---------------------------------------------------------------------------
# The feed log
# ---------------------------------------------------------------------------


async def test_spec_addon_097_i7_a_long_feed_logs_its_first_40_in_stored_order(rig):
    """SPEC-ADDON-097 I7: one row, the first 40 ids by stored rank, the profile's computed_at."""
    rig.search.create_feed_log()
    rig.kept("hc", angle=0.0, channel="clip_thumbnail")
    rig.kept("ht", angle=0.0, channel="tfidf_keywords")
    for i in range(30):
        rig.candidate(f"cc{i:02d}", 10.0 + i, channel="clip_thumbnail")
        rig.candidate(f"ct{i:02d}", 10.5 + i, channel="tfidf_keywords")

    await rig.sweep()

    stored = rig.feed()
    header = rig.header()
    logs = rig.logs()
    assert header.total > 40
    assert len(logs) == 1
    computed_at, version, file_ids = logs[0]
    assert json.loads(file_ids) == stored[:40]
    assert version == profile.PROFILE_VERSION == 3
    assert _as_naive(computed_at) == _as_naive(header.computed_at)


async def test_spec_addon_097_i7_a_short_feed_logs_all_of_it_once(rig):
    """SPEC-ADDON-097 I7: fewer than 40 ids, and no second row without a rebuild."""
    rig.search.create_feed_log()
    rig.kept()
    for file_id, angle in (("c0", 5.0), ("c1", 10.0), ("c2", 15.0)):
        rig.candidate(file_id, angle)

    await rig.sweep()
    await rig.sweep()

    logs = rig.logs()
    assert len(logs) == 1
    assert json.loads(logs[0][2]) == ["c0", "c1", "c2"]


async def test_spec_addon_097_i7_an_empty_feed_writes_no_log_row(rig):
    """SPEC-ADDON-097 I7: settled or retrying, an empty feed is not logged."""
    rig.search.create_feed_log()
    _documents_only(rig)
    rig.watched("v_only", angle=None, viewer="v2")
    rig.candidate("c0", 5.0)

    await rig.sweep()

    assert rig.header("v1").total == 0
    assert rig.header("v2").total == 0
    assert rig.logs(viewer="v1") == []
    assert rig.logs(viewer="v2") == []
    assert rig.search.query("SELECT COUNT(*) FROM pickup_feed_log") == [(0,)]


async def test_spec_addon_097_i7_rows_older_than_180_days_are_pruned(rig):
    """SPEC-ADDON-097 I7: a log write removes old rows of any drive and viewer."""
    rig.search.create_feed_log()
    now = utcnow()
    rig.search.add_feed_log(drive="z", viewer="q", computed_at=now - timedelta(days=181), file_ids=["old"])
    rig.search.add_feed_log(drive="z", viewer="q", computed_at=now - timedelta(days=179), file_ids=["recent"])
    rig.kept()
    rig.candidate("c0", 5.0)

    await rig.sweep()

    assert [json.loads(r[2]) for r in rig.logs(drive="z", viewer="q")] == [["recent"]]
    assert len(rig.logs()) == 1


async def test_spec_addon_097_i7_a_failed_log_write_keeps_the_feed(rig, caplog):
    """SPEC-ADDON-097 I7: the feed stays committed and a warning is logged."""
    rig.search.create_feed_log()
    rig.search.execute(
        "CREATE TRIGGER refuse_log BEFORE INSERT ON pickup_feed_log"
        " BEGIN SELECT RAISE(ABORT, 'log refused'); END"
    )
    rig.kept()
    for file_id, angle in (("c0", 5.0), ("c1", 10.0)):
        rig.candidate(file_id, angle)
    caplog.set_level(logging.WARNING)

    await rig.sweep()

    assert rig.feed() == ["c0", "c1"]
    assert rig.header().total == 2
    assert rig.header().watch_history_checkpoint is not None
    assert rig.logs() == []
    assert any(
        "could not write feed log drive=a viewer=v1" in r.getMessage()
        and r.levelno == logging.WARNING
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Restricting candidates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("keep", "expected"),
    [({"b", "d", "absent"}, {"b", "d"}), (set(), set()), ({"a", "b", "c", "d"}, {"a", "b", "c", "d"})],
    ids=["some", "none", "all"],
)
def test_spec_addon_096_restricting_candidates_keeps_ids_and_rows_together(keep, expected):
    """SPEC-ADDON-096: the subset's ids and matrix rows stay paired."""
    import numpy as np

    from app.pickup import retrieval

    ids = ("a", "b", "c", "d")
    matrix = np.stack([at_angle(10.0 * i) for i in range(4)]).astype(np.float32)
    full = retrieval.CandidateSet(channel="clip_thumbnail", drive="a", file_ids=ids, matrix=matrix)

    got = retrieval.restrict_candidates(full, keep)

    assert set(got.file_ids) == expected
    assert len(got.file_ids) == len(expected)
    assert got.matrix.shape[0] == len(expected)
    for row, file_id in enumerate(got.file_ids):
        assert np.array_equal(got.matrix[row], matrix[ids.index(file_id)])
    assert got.channel == "clip_thumbnail"
    assert got.drive == "a"
