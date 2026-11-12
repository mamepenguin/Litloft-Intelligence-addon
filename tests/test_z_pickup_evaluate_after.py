"""`after` is the feed the worker would store at the same reference time.

SPEC-ID: SPEC-ADDON-098
"""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta

from app.models import PickupItem
from app.pickup import evaluate
from app.pickup import profile as profile_mod
from app.pickup.retrieval import load_candidates
from app.workers import pickup as worker_mod
from tests.pickup_accuracy_rig import CoreDB, SearchDB, at_angle, utcnow, wire


def _dataset(tmp_path):
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
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
    for i in range(12):
        core.add_file(f"c{i:02d}", created_at=old)
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


def test_spec_addon_098_i9_after_matches_the_worker_feed(monkeypatch, tmp_path):
    """SPEC-ADDON-098 I9: `after` is the worker's path at the cut point."""
    core, search = _dataset(tmp_path)
    wire(monkeypatch, core, search)

    t = utcnow()
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

    time.sleep(0.01)
    asyncio.run(worker_mod.PickupWorker()._compute_for_drive("a"))

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

    stored = _stored_feed(search)
    assert stored
    assert rankings["after"] == stored
