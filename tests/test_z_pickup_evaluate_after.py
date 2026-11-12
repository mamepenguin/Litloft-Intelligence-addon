"""`after` is the worker's feed with freshness off, and the marked row is the shipped one.

SPEC-ID: SPEC-ADDON-098
"""

from __future__ import annotations

import asyncio
import importlib
import time
from datetime import timedelta

from app.models import PickupItem
from app.pickup import evaluate
from app.pickup import profile as profile_mod
from app.pickup.retrieval import load_candidates
from app.workers import pickup as worker_mod
from tests.pickup_accuracy_rig import CoreDB, SearchDB, at_angle, utcnow, wire


def _dataset(directory):
    directory.mkdir(parents=True, exist_ok=True)
    core = CoreDB(directory / "litloft.db")
    search = SearchDB(directory / "search.db")
    now = utcnow()
    old = now - timedelta(days=30)

    for i in range(5):
        core.add_file(f"h{i}", created_at=old)
        core.watch(f"h{i}", played_at=now - timedelta(days=10 + i))
        search.index(f"h{i}", at_angle(0.0))
    # Dismissed videos: kept by the unweighted configurations, dropped by
    # `after`, so a wrong weighting shows in the ranking.
    for i in range(3):
        core.add_file(f"d{i}", created_at=old)
        core.watch(
            f"d{i}", played_at=now - timedelta(days=10),
            dismissed_at=now - timedelta(days=9),
        )
        search.index(f"d{i}", at_angle(90.0))
    # Ages that run against similarity, so freshness 1 (the `after` row)
    # and the shipped constants order these differently.
    ages = [400, 3, 200, 1, 90, 15, 600, 7, 45, 0.5, 120, 30]
    for i, age in enumerate(ages):
        core.add_file(f"c{i:02d}", created_at=now - timedelta(days=age))
        search.index(f"c{i:02d}", at_angle(4.0 + i))
    for i in range(4):
        core.add_file(f"p{i:02d}", kind="document", created_at=old)
        search.index(f"p{i:02d}", at_angle(6.0 + i), kind="document")
    core.engine.dispose()
    search.engine.dispose()
    return core, search


def _stored_feed(search: SearchDB, *, viewer: str = "v1") -> list[str]:
    session = search.sessions()
    try:
        return [
            row.file_id
            for row in session.query(PickupItem)
            .filter_by(drive_id="a", viewer_id=viewer)
            .order_by(PickupItem.rank)
            .all()
        ]
    finally:
        session.close()


def _set_constants(monkeypatch, *, half_life, floor):
    fresh = importlib.import_module("app.pickup.freshness")
    monkeypatch.setattr(fresh, "HALF_LIFE_DAYS", half_life)
    monkeypatch.setattr(fresh, "FLOOR", floor)
    # Defaults are bound when the function is defined, so the module
    # constants alone would not reach a call that relies on them.
    defaults = dict(fresh.freshness.__kwdefaults__ or {})
    defaults.update(half_life_days=half_life, floor=floor)
    monkeypatch.setattr(fresh.freshness, "__kwdefaults__", defaults)
    return fresh


def _bind_at_t(monkeypatch, t):
    original_history = profile_mod.profile_history
    original_lanes = profile_mod.build_lanes

    def history_at_t(drive, viewer_id, *args, **kwargs):
        kwargs.setdefault("now", t)
        kwargs.setdefault("until", t)
        return original_history(drive, viewer_id, *args, **kwargs)

    def lanes_at_t(history, *, key, now=None,
                   vectors=profile_mod.representative_vectors):
        return original_lanes(
            history, key=key, now=t if now is None else now, vectors=vectors,
        )

    # Bind the worker's defaults to the replay's reference time, so both
    # build the feed from the same history at the same instant.
    monkeypatch.setattr(profile_mod, "profile_history", history_at_t)
    monkeypatch.setattr(profile_mod, "build_lanes", lanes_at_t)
    fresh = importlib.import_module("app.pickup.freshness")
    monkeypatch.setattr(fresh, "reference_now", lambda: t)


def _worker_feed(monkeypatch, directory, *, t, half_life, floor):
    core, search = _dataset(directory)
    wire(monkeypatch, core, search)
    _bind_at_t(monkeypatch, t)
    _set_constants(monkeypatch, half_life=half_life, floor=floor)
    time.sleep(0.01)
    asyncio.run(worker_mod.PickupWorker()._compute_for_drive("a"))
    return _stored_feed(search)


def _replay_rankings(t):
    files = evaluate._read_files("a")
    rows = evaluate._read_history("a", "v1")
    candidates = {
        channel: load_candidates(drive="a", channel=channel)
        for channel in profile_mod.CHANNELS
    }
    rankings, _ = evaluate._rank_configs(
        "a", "v1", rows, files, candidates,
        evaluate._vector_lookup(evaluate._history_vectors(rows)),
        t=t,
    )
    return rankings


def _unmarked(rankings):
    return {name.rstrip("*"): ranking for name, ranking in rankings.items()}


def test_spec_addon_098_i9_revised_after_is_the_worker_with_floor_one(monkeypatch, tmp_path):
    """SPEC-ADDON-098 I9 as revised: `after` is the worker's feed with FLOOR = 1."""
    core, search = _dataset(tmp_path / "replay")
    wire(monkeypatch, core, search)
    t = utcnow()
    _bind_at_t(monkeypatch, t)

    rankings = _unmarked(_replay_rankings(t))
    stored = _worker_feed(
        monkeypatch, tmp_path / "floor1", t=t, half_life=30.0, floor=1.0,
    )

    assert stored
    assert rankings["after"] == stored
    assert rankings["fresh_h7_f0"] != stored


def test_spec_addon_098_i9_revised_the_marked_row_is_the_shipped_worker(
    monkeypatch, tmp_path,
):
    """SPEC-ADDON-098 I9 as revised: the `*` row ranks as the worker with its constants."""
    fresh = importlib.import_module("app.pickup.freshness")
    half_life, floor = fresh.HALF_LIFE_DAYS, fresh.FLOOR

    core, search = _dataset(tmp_path / "replay")
    wire(monkeypatch, core, search)
    t = utcnow()
    _bind_at_t(monkeypatch, t)

    rankings = _replay_rankings(t)
    stored = _worker_feed(
        monkeypatch, tmp_path / "shipped", t=t,
        half_life=half_life, floor=floor,
    )

    mark = (
        "after" if floor == 1
        else f"fresh_h{half_life:g}_f{floor:g}"
    )
    assert stored
    assert rankings[f"{mark}*"] == stored
    if mark != "after":
        assert rankings["after"] != stored
