"""The evaluation commands under coverage: `after` follows it, version 4's rows do not.

SPEC-ID: SPEC-ADDON-150
SPEC-ID: SPEC-ADDON-149
"""

from __future__ import annotations

from datetime import timedelta

from tests.pickup_accuracy_rig import CoreDB, SearchDB, at_angle, utcnow
from tests.test_z_pickup_evaluate import _run, _sections, _table, _today


def _covered(core: CoreDB, file_id: str, *, seconds: float = 5.0) -> None:
    for n in range(10):
        core.execute(
            "INSERT INTO played_segments VALUES ('v1', :f, :n, :s)",
            {"f": file_id, "n": n, "s": seconds},
        )


def _history(core: CoreDB, search: SearchDB, t0) -> None:
    old = _today() - timedelta(days=30)
    for i in range(5):
        core.add_file(f"h{i}", created_at=old)
        core.watch(f"h{i}", played_at=t0 - timedelta(days=5, hours=i), position=0.0)
        _covered(core, f"h{i}")
        search.index(f"h{i}", at_angle(0.0))


def test_spec_addon_150_replay_after_ranks_glanced_videos_and_version_4_rows_do_not(tmp_path):
    """SPEC-ADDON-150 evaluation / SPEC-ADDON-098 I9 as revised: the exclusion of each row."""
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    t0 = _today() - timedelta(days=1)
    old = _today() - timedelta(days=30)
    _history(core, search, t0)
    for i in range(11):
        core.add_file(f"g{i:02d}", created_at=old)
        core.watch(f"g{i:02d}", played_at=t0 - timedelta(days=2), position=100.0)
        search.index(f"g{i:02d}", at_angle(5.0 + 0.2 * i))
    core.add_file("legacy", created_at=old)
    core.watch("legacy", played_at="2026-10-20 12:00:00", position=1.0)
    search.index("legacy", at_angle(4.0))
    core.add_file("unparsed", created_at=old)
    core.watch("unparsed", played_at="not-a-date", position=0.0, duration=0.0)
    search.index("unparsed", at_angle(4.5))
    for file_id, angle, hour in (("X", 18.0, 1), ("Y", 19.0, 2)):
        core.add_file(file_id, created_at=_today() - timedelta(days=25))
        core.watch(file_id, played_at=t0 + timedelta(hours=hour), position=0.0)
        _covered(core, file_id)
        search.index(file_id, at_angle(angle))
    core.engine.dispose()
    search.engine.dispose()

    result = _run(core.path, search.path, "replay", "--drive", "a", "--days", "1", "--horizon", "1")

    assert result.returncode == 0, result.stderr
    v1 = _sections(result.stdout)["v1"]
    assert "candidate videos: 2 .. 2" in [line.strip() for line in v1]
    _, rows = _table(v1)
    rows = {name.rstrip("*"): cells for name, cells in rows.items()}
    for config in ("before", "depth", "video", "random", "newest"):
        assert rows[config] == ["1", "0", "2", "1.000", "1.000", "1.000"], config
    assert rows["after"] == ["1", "0", "2", "0.500", "1.000", "1.000"]


def test_spec_addon_150_replay_targets_are_rows_with_engagement_under_coverage(tmp_path):
    """SPEC-ADDON-150 evaluation: a played position without coverage is no target; coverage without position is."""
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    t0 = _today() - timedelta(days=1)
    _history(core, search, t0)
    created = _today() - timedelta(days=25)
    core.add_file("X", created_at=created)
    core.watch("X", played_at=t0 + timedelta(hours=1), position=0.0)
    _covered(core, "X")
    search.index("X", at_angle(10.0))
    core.add_file("Y", created_at=created)
    core.watch("Y", played_at=t0 + timedelta(hours=2), position=100.0)
    search.index("Y", at_angle(12.0))
    for file_id, angle, hour in (("Z1", 14.0, 3), ("Z2", 16.0, 4)):
        core.add_file(file_id, created_at=created)
        core.watch(file_id, played_at=t0 + timedelta(hours=hour), position=1.0)
        _covered(core, file_id, seconds=0.5)
        search.index(file_id, at_angle(angle))
    core.engine.dispose()
    search.engine.dispose()

    result = _run(core.path, search.path, "replay", "--drive", "a", "--days", "1", "--horizon", "1")

    assert result.returncode == 0, result.stderr
    _, rows = _table(_sections(result.stdout)["v1"])
    rows = {name.rstrip("*"): cells for name, cells in rows.items()}
    for config in ("before", "depth", "video", "after", "random", "newest"):
        assert rows[config][:3] == ["1", "0", "3"], config


def test_spec_addon_150_online_targets_are_rows_with_engagement_under_coverage(tmp_path):
    """SPEC-ADDON-150 evaluation: `online` counts the same targets as `replay`."""
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    search.create_feed_log()
    played = utcnow() - timedelta(days=3)
    created = played - timedelta(days=100)
    core.add_file("X", created_at=created)
    core.watch("X", played_at=played, position=0.0)
    _covered(core, "X")
    core.add_file("Y", created_at=created)
    core.watch("Y", played_at=played, position=100.0)
    for file_id in ("Z1", "Z2"):
        core.add_file(file_id, created_at=created)
        core.watch(file_id, played_at=played, position=1.0)
        _covered(core, file_id, seconds=0.5)
    search.add_feed_log(
        computed_at=played - timedelta(hours=10), version=5,
        file_ids=["X", "Y", "Z1", "Z2"],
    )
    core.engine.dispose()
    search.engine.dispose()

    result = _run(core.path, search.path, "online", "--drive", "a", "--days", "30")

    assert result.returncode == 0, result.stderr
    _, rows = _table(_sections(result.stdout)["v1"])
    assert rows["total"][:4] == ["3", "0", "1.000", "1.000"]
