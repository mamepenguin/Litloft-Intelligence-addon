"""`replay` with freshness: the grid rows, the marked row, and parity with the worker.

SPEC-ID: SPEC-ADDON-102
"""

from __future__ import annotations

import asyncio
import importlib
import time
from datetime import timedelta

import pytest

from app.models import PickupItem
from app.pickup import evaluate
from app.pickup import profile as profile_mod
from app.pickup import retrieval
from app.pickup.retrieval import load_candidates
from app.workers import pickup as worker_mod
from tests.pickup_accuracy_rig import CoreDB, SearchDB, at_angle, utcnow, wire
from tests.test_z_pickup_evaluate import _run, _sections, _table, _today

GRID = [(h, f) for h in (7, 30, 90, 365) for f in (0, 0.5)]
GRID_NAMES = [f"fresh_h{h:g}_f{f:g}" for h, f in GRID]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _new_target_dataset(tmp_path):
    """One evaluated cut point at T0, where the target is new and less similar than 15 old videos."""
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    t0 = _today() - timedelta(days=1)
    old = _today() - timedelta(days=30)
    for i in range(5):
        core.add_file(f"h{i}", created_at=old)
        core.watch(f"h{i}", played_at=t0 - timedelta(days=9, hours=i))
        search.index(f"h{i}", at_angle(0.0))
    for i in range(15):
        core.add_file(f"k{i:02d}", created_at=old + timedelta(hours=i))
        search.index(f"k{i:02d}", at_angle(2.0 + i))
    core.add_file("X", created_at=t0 - timedelta(hours=12))
    core.watch("X", played_at=t0 + timedelta(hours=1))
    search.index("X", at_angle(30.0))
    core.engine.dispose()
    search.engine.dispose()
    return core, search


def _marked(fresh) -> str:
    if fresh.FLOOR == 1:
        return "after*"
    return f"fresh_h{fresh.HALF_LIFE_DAYS:g}_f{fresh.FLOOR:g}*"


def test_spec_addon_102_replay_reports_the_grid_rows_in_order(tmp_path):
    """SPEC-ADDON-102: the six rows stay; eight fresh rows, H outer and F inner; one row marked."""
    core, search = _new_target_dataset(tmp_path)

    result = _run(core.path, search.path, "replay", "--drive", "a", "--days", "3", "--horizon", "1")

    assert result.returncode == 0, result.stderr
    _, rows = _table(_sections(result.stdout)["v1"])
    names = list(rows)
    base = [n.rstrip("*") for n in names if not n.startswith("fresh_")]
    assert base == ["before", "depth", "video", "after", "random", "newest"]
    assert [n.rstrip("*") for n in names if n.startswith("fresh_")] == GRID_NAMES
    fresh = importlib.import_module("app.pickup.freshness")
    assert [n for n in names if n.endswith("*")] == [_marked(fresh)]


def test_spec_addon_102_a_fresh_row_finds_the_new_target_after_misses(tmp_path):
    """SPEC-ADDON-102: same targets as `after`; the steepest decay lifts the new target into the top 12."""
    core, search = _new_target_dataset(tmp_path)

    result = _run(core.path, search.path, "replay", "--drive", "a", "--days", "3", "--horizon", "1")

    assert result.returncode == 0, result.stderr
    _, rows = _table(_sections(result.stdout)["v1"])
    rows = {n.rstrip("*"): v for n, v in rows.items()}
    assert rows["after"] == ["1", "2", "1", "0.000", "1.000", "1.000"]
    assert rows["fresh_h7_f0"] == ["1", "2", "1", "1.000", "1.000", "1.000"]
    for name in GRID_NAMES:
        assert rows[name][:3] == rows["after"][:3], name


# ---------------------------------------------------------------------------
# Parity with the worker (SPEC-ADDON-098 I9 as revised)
# ---------------------------------------------------------------------------


def _parity_dataset(tmp_path):
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    now = utcnow()
    old = now - timedelta(days=30)
    for i in range(5):
        core.add_file(f"h{i}", created_at=old)
        core.watch(f"h{i}", played_at=now - timedelta(days=10 + i))
        search.index(f"h{i}", at_angle(0.0))
    for i in range(3):
        core.add_file(f"d{i}", created_at=old)
        core.watch(f"d{i}", played_at=now - timedelta(days=10),
                   dismissed_at=now - timedelta(days=9))
        search.index(f"d{i}", at_angle(90.0))
    # Ages that run against similarity, so every grid setting orders them differently.
    ages = [400, 3, 200, 1, 90, 15, 600, 7, 45, 0.5, 120, 30]
    for i, age in enumerate(ages):
        core.add_file(f"c{i:02d}", created_at=now - timedelta(days=age))
        search.index(f"c{i:02d}", at_angle(4.0 + 2 * i))
    for i in range(4):
        core.add_file(f"p{i:02d}", kind="document", created_at=now - timedelta(days=1))
        search.index(f"p{i:02d}", at_angle(6.0 + i), kind="document")
    core.engine.dispose()
    search.engine.dispose()
    return core, search


def _stored_feed(search: SearchDB) -> list[str]:
    session = search.sessions()
    try:
        return [
            row.file_id
            for row in session.query(PickupItem)
            .filter_by(drive_id="a", viewer_id="v1")
            .order_by(PickupItem.rank)
            .all()
        ]
    finally:
        session.close()


def _bind_worker_to(monkeypatch, t, *, half_life, floor):
    fresh = importlib.import_module("app.pickup.freshness")
    original_history = profile_mod.profile_history
    original_lanes = profile_mod.build_lanes

    def history_at_t(drive, viewer_id, *args, **kwargs):
        kwargs.setdefault("now", t)
        kwargs.setdefault("until", t)
        return original_history(drive, viewer_id, *args, **kwargs)

    def lanes_at_t(history, *, key, now=None, vectors=profile_mod.representative_vectors):
        return original_lanes(history, key=key, now=t if now is None else now, vectors=vectors)

    monkeypatch.setattr(profile_mod, "profile_history", history_at_t)
    monkeypatch.setattr(profile_mod, "build_lanes", lanes_at_t)

    monkeypatch.setattr(fresh, "HALF_LIFE_DAYS", half_life)
    monkeypatch.setattr(fresh, "FLOOR", floor)
    # Defaults are bound when the function is defined, so the module constants
    # alone would not reach a call that relies on them.
    defaults = dict(fresh.freshness.__kwdefaults__ or {})
    defaults.update(half_life_days=half_life, floor=floor)
    monkeypatch.setattr(fresh.freshness, "__kwdefaults__", defaults)
    for mod in (fresh, worker_mod, retrieval, profile_mod):
        if mod is fresh or hasattr(mod, "reference_now"):
            monkeypatch.setattr(mod, "reference_now", lambda: t)


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
    return {name.rstrip("*"): ranking for name, ranking in rankings.items()}


def _worker_feed(monkeypatch, tmp_path, *, half_life, floor):
    core, search = _parity_dataset(tmp_path)
    wire(monkeypatch, core, search)
    t = utcnow()
    _bind_worker_to(monkeypatch, t, half_life=half_life, floor=floor)
    time.sleep(0.01)
    asyncio.run(worker_mod.PickupWorker()._compute_for_drive("a"))
    return t, _stored_feed(search)


@pytest.mark.parametrize(("half_life", "floor"), GRID, ids=GRID_NAMES)
def test_spec_addon_102_i9_a_fresh_row_ranks_as_the_worker(monkeypatch, tmp_path, half_life, floor):
    """SPEC-ADDON-102 I9: fresh_h<H>_f<F> at t is the worker's feed with those constants at t."""
    t, stored = _worker_feed(monkeypatch, tmp_path, half_life=float(half_life), floor=float(floor))

    rankings = _replay_rankings(t)

    assert stored
    assert rankings[f"fresh_h{half_life:g}_f{floor:g}"] == stored
    assert rankings["after"] != stored


def test_spec_addon_098_i9_revised_after_is_the_worker_with_floor_one(monkeypatch, tmp_path):
    """SPEC-ADDON-098 I9 as revised: `after` is the worker's feed with FLOOR = 1."""
    t, stored = _worker_feed(monkeypatch, tmp_path, half_life=30.0, floor=1.0)

    rankings = _replay_rankings(t)

    assert stored
    assert rankings["after"] == stored
    assert rankings["fresh_h7_f0"] != stored


def test_spec_addon_098_i9_revised_the_marked_row_is_the_shipped_worker(monkeypatch, tmp_path):
    """SPEC-ADDON-098 I9 as revised: the `*` row ranks as the worker with its own constants."""
    fresh = importlib.import_module("app.pickup.freshness")
    t, stored = _worker_feed(
        monkeypatch, tmp_path, half_life=fresh.HALF_LIFE_DAYS, floor=fresh.FLOOR,
    )

    rankings = _replay_rankings(t)

    assert rankings[_marked(fresh).rstrip("*")] == stored
