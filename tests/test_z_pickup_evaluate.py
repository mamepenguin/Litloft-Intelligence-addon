"""The Pickup evaluation commands, run as the operator runs them.

SPEC-ID: SPEC-ADDON-098
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.pickup_accuracy_rig import CoreDB, SearchDB, at_angle, utcnow

_ADDON_ROOT = Path(__file__).resolve().parents[1]

# The test image has no ML packages.
_BOOT = (
    "import sys, runpy\n"
    "from unittest.mock import MagicMock\n"
    "for m in ('PIL', 'PIL.Image', 'open_clip', 'torch', 'sentence_transformers',\n"
    "          'faster_whisper', 'onnxruntime', 'transformers', 'janome',\n"
    "          'janome.tokenizer', 'sqlite_vec'):\n"
    "    sys.modules.setdefault(m, MagicMock())\n"
    "sys.argv = ['app.pickup.evaluate', *sys.argv[1:]]\n"
    "runpy.run_module('app.pickup.evaluate', run_name='__main__', alter_sys=True)\n"
)


def _run(core_path: Path, search_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "HOMEVAULT_DB_PATH": str(core_path),
        "INTELLIGENCE_SEARCH_DB_PATH": str(search_path),
    }
    return subprocess.run(
        [sys.executable, "-c", _BOOT, *args],
        cwd=_ADDON_ROOT, env=env, capture_output=True, text=True, timeout=600,
    )


def _sections(stdout: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    current = None
    for line in stdout.splitlines():
        if line.startswith("viewer "):
            current = line[len("viewer "):].strip()
            out[current] = []
        elif current is not None:
            out[current].append(line)
    return out


def _viewer_order(stdout: str) -> list[str]:
    return [l[len("viewer "):].strip() for l in stdout.splitlines() if l.startswith("viewer ")]


def _table(lines: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    rows = [
        [c.strip() for c in l.strip().strip("|").split("|")]
        for l in lines if l.strip().startswith("|")
    ]
    header, body = rows[0], rows[1:]
    body = [r for r in body if not all(set(c) <= set("-: ") for c in r)]
    return header, {r[0]: r[1:] for r in body}


def _table_rows(lines: list[str]) -> list[list[str]]:
    rows = [
        [c.strip() for c in l.strip().strip("|").split("|")]
        for l in lines if l.strip().startswith("|")
    ]
    return [r for r in rows[1:] if not all(set(c) <= set("-: ") for c in r)]


def _fingerprint(*paths: Path) -> dict:
    state = {}
    for p in paths:
        state[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        for suffix in ("-wal", "-shm", "-journal"):
            state[p.name + suffix] = Path(str(p) + suffix).exists()
    return state


def _tables(db: SearchDB | CoreDB) -> list[str]:
    from sqlalchemy import text

    with db.engine.connect() as conn:
        return sorted(r[0] for r in conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'index', 'trigger')"
        )))


def _today() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def _replay_documents_dataset(tmp_path):
    """One evaluated cut point at T0 - 1d, where documents outscore the target."""
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    t0 = _today() - timedelta(days=1)
    old = _today() - timedelta(days=30)

    for i in range(5):
        core.add_file(f"h{i}", created_at=old)
        core.watch(f"h{i}", played_at=t0 - timedelta(days=9, hours=i))
        search.index(f"h{i}", at_angle(0.0))
    for i in range(15):
        core.add_file(f"dc{i:02d}", kind="document", created_at=old)
        search.index(f"dc{i:02d}", at_angle(6.0 + i * 0.25), kind="document")
    core.add_file("X", created_at=_today() - timedelta(days=25))
    core.watch("X", played_at=t0 + timedelta(hours=1))
    search.index("X", at_angle(18.0))
    core.add_file("Q", created_at=_today() - timedelta(days=25))
    core.watch("Q", played_at=t0 + timedelta(hours=2))
    for i in range(13):
        core.add_file(f"o{i:02d}", created_at=_today() - timedelta(days=20) + timedelta(hours=i))
        search.index(f"o{i:02d}", at_angle(30.0 + 2 * i))
    core.watch("o12", played_at=t0 + timedelta(hours=3), position=1.0)
    core.add_file("Z", created_at=t0 + timedelta(minutes=30))
    core.watch("Z", played_at=t0 + timedelta(hours=4))
    search.index("Z", at_angle(18.5))
    core.add_file("Tr", deleted=True, created_at=old)
    core.watch("Tr", played_at=t0 + timedelta(hours=5))
    search.index("Tr", at_angle(19.0))
    core.add_file("Dt", kind="document", created_at=old)
    core.watch("Dt", played_at=t0 + timedelta(hours=6))

    core.add_file("Bad", created_at=old)
    core.watch("Bad", played_at="not-a-date")
    search.index("Bad", at_angle(17.0))
    core.add_file("Bc", created_at="not-a-date")
    core.watch("Bc", played_at=t0 + timedelta(minutes=90))
    search.index("Bc", at_angle(17.5))

    core.add_file("v2doc", kind="document", created_at=old)
    core.watch("v2doc", viewer="v2", played_at=t0 - timedelta(days=9))
    core.engine.dispose()
    search.engine.dispose()
    return core, search


def test_spec_addon_098_i9_replay_scores_each_configuration(tmp_path):
    """SPEC-ADDON-098 I9: targets, universe and recall@k per configuration, per viewer."""
    core, search = _replay_documents_dataset(tmp_path)

    result = _run(core.path, search.path, "replay", "--drive", "a", "--days", "3", "--horizon", "1")

    assert result.returncode == 0, result.stderr
    assert _viewer_order(result.stdout) == ["v1", "v2"]
    v1 = _sections(result.stdout)["v1"]
    assert "candidate videos: 14 .. 14" in [l.strip() for l in v1]
    header, rows = _table(v1)
    assert header == ["config", "cuts", "skipped", "targets", "recall@12", "recall@40", "hit@40"]
    expected = ["before", "depth", "video", "after"]
    expected += [
        f"fresh_h{half_life:g}_f{floor:g}"
        for half_life in (7, 30, 90, 365)
        for floor in (0, 0.5)
    ]
    expected += ["random", "newest"]
    assert [name.rstrip("*") for name in rows] == expected
    assert rows["before"] == ["1", "2", "2", "0.000", "0.500", "1.000"]
    assert rows["depth"] == ["1", "2", "2", "0.000", "0.500", "1.000"]
    assert rows["video"] == ["1", "2", "2", "0.500", "0.500", "1.000"]
    assert rows["after"] == ["1", "2", "2", "0.500", "0.500", "1.000"]
    assert rows["random"][:3] == ["1", "2", "2"]
    assert rows["random"][4:] == ["0.500", "1.000"]
    assert rows["newest"] == ["1", "2", "2", "0.000", "0.500", "1.000"]


def test_spec_addon_098_a_viewer_with_no_evaluated_cut_point(tmp_path):
    """SPEC-ADDON-098: zero counts, '-' means and '-' candidate counts."""
    core, search = _replay_documents_dataset(tmp_path)

    result = _run(core.path, search.path, "replay", "--drive", "a", "--days", "3", "--horizon", "1")

    assert result.returncode == 0, result.stderr
    v2 = _sections(result.stdout)["v2"]
    assert "candidate videos: - .. -" in [l.strip() for l in v2]
    _, rows = _table(v2)
    for config in ("before", "depth", "video", "after", "random", "newest"):
        assert rows[config] == ["0", "3", "0", "-", "-", "-"]


def test_spec_addon_098_replay_for_one_viewer(tmp_path):
    """SPEC-ADDON-098: --viewer limits the report to that viewer."""
    core, search = _replay_documents_dataset(tmp_path)

    result = _run(core.path, search.path, "replay", "--drive", "a", "--days", "3",
                  "--horizon", "1", "--viewer", "v1")

    assert result.returncode == 0, result.stderr
    assert _viewer_order(result.stdout) == ["v1"]


def test_spec_addon_098_i9_replay_weights_by_engagement_at_the_cut(tmp_path):
    """SPEC-ADDON-098 I9: dismissed history pulls 'before' and 'video' off the target, not 'after'."""
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    t0 = _today() - timedelta(days=1)
    old = _today() - timedelta(days=30)
    for i in range(4):
        core.add_file(f"h{i}", created_at=old)
        core.watch(f"h{i}", played_at=t0 - timedelta(days=9, hours=i))
        search.index(f"h{i}", at_angle(0.0))
    for i in range(4):
        core.add_file(f"g{i}", created_at=old)
        core.watch(f"g{i}", played_at=t0 - timedelta(days=9, hours=i),
                   dismissed_at=t0 - timedelta(days=8))
        search.index(f"g{i}", at_angle(90.0))
    for i in range(15):
        core.add_file(f"k{i:02d}", created_at=_today() - timedelta(days=20) + timedelta(hours=i))
        search.index(f"k{i:02d}", at_angle(26.0 + i))
    core.add_file("X", created_at=_today() - timedelta(days=25))
    core.watch("X", played_at=t0 + timedelta(hours=1))
    search.index("X", at_angle(18.0))
    core.engine.dispose()
    search.engine.dispose()

    result = _run(core.path, search.path, "replay", "--drive", "a", "--days", "3", "--horizon", "1")

    assert result.returncode == 0, result.stderr
    v1 = _sections(result.stdout)["v1"]
    assert "candidate videos: 16 .. 16" in [l.strip() for l in v1]
    _, rows = _table(v1)
    assert rows["before"] == ["1", "2", "1", "0.000", "1.000", "1.000"]
    assert rows["depth"] == ["1", "2", "1", "1.000", "1.000", "1.000"]
    assert rows["video"] == ["1", "2", "1", "0.000", "1.000", "1.000"]
    assert rows["after"] == ["1", "2", "1", "1.000", "1.000", "1.000"]
    assert rows["random"][4:] == ["1.000", "1.000"]
    assert rows["newest"] == ["1", "2", "1", "0.000", "1.000", "1.000"]


def test_spec_addon_098_i9_the_target_window_includes_t_and_excludes_t_plus_horizon(tmp_path):
    """SPEC-ADDON-098 I9: a play exactly at t is a target, one exactly at t + horizon is not."""
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    t0 = _today() - timedelta(days=1)
    old = _today() - timedelta(days=30)
    for i in range(5):
        core.add_file(f"h{i}", created_at=old)
        core.watch(f"h{i}", played_at=t0 - timedelta(days=9, hours=i))
        search.index(f"h{i}", at_angle(0.0))
    core.add_file("at_t", created_at=old)
    core.watch("at_t", played_at=t0)
    search.index("at_t", at_angle(10.0))
    core.add_file("at_end", created_at=old)
    core.watch("at_end", played_at=_today())
    search.index("at_end", at_angle(12.0))
    core.engine.dispose()
    search.engine.dispose()

    result = _run(core.path, search.path, "replay", "--drive", "a", "--days", "1", "--horizon", "1")

    assert result.returncode == 0, result.stderr
    v1 = _sections(result.stdout)["v1"]
    assert "candidate videos: 2 .. 2" in [l.strip() for l in v1]
    _, rows = _table(v1)
    assert rows["after"] == ["1", "0", "1", "1.000", "1.000", "1.000"]


# ---------------------------------------------------------------------------
# online
# ---------------------------------------------------------------------------


def _online_dataset(tmp_path):
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    search.create_feed_log()
    now = utcnow()
    old = now - timedelta(days=400)
    fill = [f"f{i:02d}" for i in range(40)]

    played = {
        "X1": now - timedelta(days=2),
        "X2": now - timedelta(days=6),
        "X3": now - timedelta(days=10),
        "X4": now - timedelta(days=14),
    }
    for file_id, at in played.items():
        core.add_file(file_id, created_at=old)
        core.watch(file_id, played_at=at)
    for file_id in ("W", "O", "Tv", "D"):
        core.add_file(file_id, created_at=old,
                      kind="document" if file_id == "D" else "video",
                      deleted=file_id == "Tv")
    core.watch("W", played_at=now - timedelta(days=2), position=1.0)
    core.watch("O", played_at=now - timedelta(days=40))
    core.watch("Tv", played_at=now - timedelta(days=2))
    core.watch("D", played_at=now - timedelta(days=2))
    core.watch("X4", viewer="v2", played_at=now - timedelta(days=60))
    core.add_file("Y", created_at=old)
    core.watch("Y", viewer="v3", played_at=now - timedelta(days=90))

    search.add_feed_log(computed_at=played["X1"] - timedelta(hours=10), version=2,
                        file_ids=["W", "D", "Tv", "f00", "X1"] + fill[1:30])
    search.add_feed_log(computed_at=played["X2"] - timedelta(hours=20), version=3,
                        file_ids=fill[:19] + ["X2"] + fill[19:39])
    search.add_feed_log(computed_at=played["X3"] - timedelta(hours=50), version=9,
                        file_ids=["X3"] + fill[:10])
    search.add_feed_log(computed_at=played["X3"] + timedelta(hours=1), version=9,
                        file_ids=["X3"] + fill[:10])
    search.add_feed_log(computed_at=played["X4"] - timedelta(hours=1), version=1,
                        file_ids=fill[:40])
    search.add_feed_log(viewer="v2", computed_at=played["X4"] - timedelta(hours=1),
                        version=7, file_ids=["X4"])
    search.add_feed_log(drive="b", computed_at=played["X4"] - timedelta(hours=1),
                        version=8, file_ids=["X4"])
    search.add_feed_log(computed_at=now - timedelta(days=40, hours=1), version=6,
                        file_ids=["O"])
    core.engine.dispose()
    search.engine.dispose()
    return core, search, played


def _week(at: datetime) -> str:
    year, week, _ = at.isocalendar()
    return f"{year}-W{week:02d}"


def _versions(cell: str) -> list[str]:
    return [v.strip() for v in cell.split(",")]


def _ratio(hits: int, n: int) -> str:
    return "-" if n == 0 else f"{hits / n:.3f}"


def test_spec_addon_098_i10_online_scores_targets_against_feeds_shown_before(tmp_path):
    """SPEC-ADDON-098 I10: hits within 48 h before the play, unscored when no feed."""
    core, search, played = _online_dataset(tmp_path)

    result = _run(core.path, search.path, "online", "--drive", "a", "--days", "30")

    assert result.returncode == 0, result.stderr
    assert _viewer_order(result.stdout) == ["v1", "v2", "v3"]
    v1 = _sections(result.stdout)["v1"]
    header, rows = _table(v1)
    assert header == ["week", "targets", "unscored", "recall@12", "recall@40", "versions"]
    assert rows["total"][:4] == ["3", "1", "0.333", "0.667"]
    assert _versions(rows["total"][4]) == ["1", "2", "3"]

    outcome = {"X1": (True, True, 2), "X2": (False, True, 3), "X3": None, "X4": (False, False, 1)}
    weeks: dict[str, list] = {}
    for file_id, at in sorted(played.items(), key=lambda kv: kv[1]):
        weeks.setdefault(_week(at), []).append(outcome[file_id])
    expected = []
    for week, results in weeks.items():
        scored = [r for r in results if r is not None]
        expected.append([
            week, str(len(scored)), str(len(results) - len(scored)),
            _ratio(sum(r[0] for r in scored), len(scored)),
            _ratio(sum(r[1] for r in scored), len(scored)),
            [str(v) for v in sorted({r[2] for r in scored})],
        ])
    got = _table_rows(v1)
    assert [r[0] for r in got] == [e[0] for e in expected] + ["total"]
    for g, e in zip(got, expected):
        assert g[:5] == e[:5]
        if e[1] != "0":
            assert _versions(g[5]) == e[5]


def test_spec_addon_098_i10_a_feed_exactly_48_hours_before_counts(tmp_path):
    """SPEC-ADDON-098 I10: the window [last_played_at - 48 h, last_played_at) includes its start."""
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    search.create_feed_log()
    played = utcnow() - timedelta(days=3)
    core.add_file("X", created_at=played - timedelta(days=100))
    core.watch("X", played_at=played)
    search.add_feed_log(computed_at=played - timedelta(hours=48), version=3, file_ids=["X"])
    core.engine.dispose()
    search.engine.dispose()

    result = _run(core.path, search.path, "online", "--drive", "a", "--days", "30")

    assert result.returncode == 0, result.stderr
    _, rows = _table(_sections(result.stdout)["v1"])
    assert rows["total"][:4] == ["1", "0", "1.000", "1.000"]


def test_spec_addon_098_online_viewers_without_targets_get_zero_counts(tmp_path):
    """SPEC-ADDON-098: a viewer with no log row, or no target, prints zeros and '-'."""
    core, search, _ = _online_dataset(tmp_path)

    result = _run(core.path, search.path, "online", "--drive", "a", "--days", "30")

    assert result.returncode == 0, result.stderr
    for viewer in ("v2", "v3"):
        rows = _table_rows(_sections(result.stdout)[viewer])
        assert [r[0] for r in rows] == ["total"]
        assert rows[0][1:5] == ["0", "0", "-", "-"]


def test_spec_addon_098_online_without_a_feed_log(tmp_path):
    """SPEC-ADDON-098 I8: 'no feed log' on stderr, exit 0, and no table created."""
    core, search, _ = _online_dataset(tmp_path)
    search.drop_feed_log()
    search.engine.dispose()
    tables = _tables(search)
    search.engine.dispose()
    before = _fingerprint(core.path, search.path)

    result = _run(core.path, search.path, "online", "--drive", "a")

    assert result.returncode == 0, result.stderr
    assert "no feed log" in result.stderr
    assert _fingerprint(core.path, search.path) == before
    assert _tables(search) == tables


# ---------------------------------------------------------------------------
# Read-only, and the edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["replay", "online"])
def test_spec_addon_098_i8_the_commands_change_neither_database(tmp_path, command):
    """SPEC-ADDON-098 I8: both files byte-identical, no journal left, no table added."""
    core, search, _ = _online_dataset(tmp_path)
    search.engine.dispose()
    core_tables, search_tables = _tables(core), _tables(search)
    core.engine.dispose()
    search.engine.dispose()
    before = _fingerprint(core.path, search.path)

    result = _run(core.path, search.path, command, "--drive", "a", "--days", "30")

    assert result.returncode == 0, result.stderr
    assert _fingerprint(core.path, search.path) == before
    assert _tables(core) == core_tables
    assert _tables(search) == search_tables


@pytest.mark.parametrize(
    "args",
    [
        ["replay", "--drive", "nowhere"],
        ["online", "--drive", "nowhere"],
        ["replay", "--drive", "a", "--viewer", "ghost"],
        ["online", "--drive", "a", "--viewer", "ghost"],
    ],
    ids=["replay, unknown drive", "online, unknown drive",
         "replay, unknown viewer", "online, unknown viewer"],
)
def test_spec_addon_098_no_viewers(tmp_path, args):
    """SPEC-ADDON-098: the single line 'no viewers', exit 0."""
    core, search, _ = _online_dataset(tmp_path)

    result = _run(core.path, search.path, *args)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "no viewers"


@pytest.mark.parametrize("command", ["replay", "online"])
@pytest.mark.parametrize("missing", ["core", "search"])
def test_spec_addon_098_a_missing_database_fails(tmp_path, command, missing):
    """SPEC-ADDON-098: non-zero exit with the error on stderr, and nothing created."""
    core, search, _ = _online_dataset(tmp_path)
    ok = _run(core.path, search.path, command, "--drive", "a")
    assert ok.returncode == 0, ok.stderr

    absent = tmp_path / "absent.db"
    paths = (absent, search.path) if missing == "core" else (core.path, absent)
    result = _run(*paths, command, "--drive", "a")

    assert result.returncode != 0
    assert result.stderr.strip()
    assert not absent.exists()
