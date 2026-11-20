"""The Pickup interest report, run as the operator runs it.

SPEC-ID: SPEC-ADDON-156
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy import create_engine, text

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.pickup import profile  # noqa: E402
from tests.pickup_accuracy_rig import (  # noqa: E402
    DIM,
    LOFT_MIME,
    CoreDB,
    SearchDB,
    at_angle,
    ts,
    wire,
)
from tests.test_z_pickup_evaluate import _fingerprint  # noqa: E402

_ADDON_ROOT = Path(__file__).resolve().parents[1]

R = profile.RECORDING_START
NOW = R + timedelta(days=10)
CREATED = NOW - timedelta(days=100)
KEY = "a\x1fv1"

# The test image has no ML packages. The clock is pinned through
# `reference_now`, the one instant the command is required to take.
_BOOT = (
    "import os, sys, runpy\n"
    "from datetime import datetime\n"
    "from unittest.mock import MagicMock\n"
    "for m in ('PIL', 'PIL.Image', 'open_clip', 'torch', 'sentence_transformers',\n"
    "          'faster_whisper', 'onnxruntime', 'transformers', 'janome',\n"
    "          'janome.tokenizer', 'sqlite_vec'):\n"
    "    sys.modules.setdefault(m, MagicMock())\n"
    "_now = datetime.fromisoformat(os.environ['PICKUP_TEST_NOW'])\n"
    "import app.pickup.freshness\n"
    "for _m in list(sys.modules.values()):\n"
    "    if getattr(_m, '__name__', '').startswith('app') and hasattr(_m, 'reference_now'):\n"
    "        _m.reference_now = lambda: _now\n"
    "sys.argv = ['app.pickup.evaluate', *sys.argv[1:]]\n"
    "runpy.run_module('app.pickup.evaluate', run_name='__main__', alter_sys=True)\n"
)

_ERROR_LINE = re.compile(r"^[A-Za-z_][\w.]*: .+")

HEADER = (
    "| # | type | weight | members | 14d | 90d | median age | recorded | glanced"
    " | glance rate | set aside | marked |"
)
SEPARATOR = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def _interests(core_path: Path, search_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "HOMEVAULT_DB_PATH": str(core_path),
        "INTELLIGENCE_SEARCH_DB_PATH": str(search_path),
        "PICKUP_TEST_NOW": NOW.isoformat(),
    }
    return subprocess.run(
        [sys.executable, "-c", _BOOT, "interests", *args],
        cwd=_ADDON_ROOT, env=env, capture_output=True, text=True, timeout=600,
    )


def _blocks(stdout: str) -> list[str]:
    return stdout.rstrip("\n").split("\n\n")


def _viewers(stdout: str) -> dict[str, list[str]]:
    """Each viewer's blocks after its `viewer <id>` block."""
    out: dict[str, list[str]] = {}
    current = None
    for block in _blocks(stdout):
        if block.startswith("viewer "):
            current = block[len("viewer "):]
            out[current] = []
        else:
            assert current is not None, block
            out[current].append(block)
    return out


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _counts(watched=0, glanced=0, set_aside=0, no_length=0, legacy_zero=0, marked=0) -> str:
    return (
        f"watched {watched}, glanced {glanced}, set aside {set_aside},"
        f" no length {no_length}, legacy zero {legacy_zero}, marked {marked}"
    )


def _ago(days: float):
    return NOW - timedelta(days=days)


def _cats(degrees: float) -> np.ndarray:
    return at_angle(degrees)


def _trains(degrees: float) -> np.ndarray:
    return at_angle(degrees, 2, 3)


def _covered(core: CoreDB, file_id: str, *, viewer: str = "v1") -> None:
    """Over half of the default 100-second duration, in 5-second segments."""
    core.segments(file_id, [5.0] * 10, viewer=viewer)


def _file(core: CoreDB, file_id: str, *, title: str | None = None,
          filename: str | None = None, **kw) -> None:
    core.add_file(file_id, created_at=CREATED, **kw)
    core.execute(
        "UPDATE files SET title = :t, filename = :n WHERE id = :i",
        {"t": f"T-{file_id}" if title is None else title,
         "n": f"{file_id}.mp4" if filename is None else filename, "i": file_id},
    )


def _mark(search: SearchDB, file_id: str, *, drive: str = "a", viewer: str = "v1") -> None:
    search.execute(
        "INSERT INTO pickup_not_interested (drive_id, viewer_id, file_id, marked_at)"
        " VALUES (:d, :v, :f, :m)",
        {"d": drive, "v": viewer, "f": file_id, "m": ts(_ago(1))},
    )


def _oracle_lanes(core: CoreDB, search: SearchDB, viewer: str = "v1"):
    with pytest.MonkeyPatch.context() as mp:
        wire(mp, core, search)
        history = profile.profile_history("a", viewer, now=NOW)
        return profile.build_lanes(history, key=f"a\x1f{viewer}", now=NOW)


def _dispose(*dbs) -> None:
    for db in dbs:
        db.engine.dispose()


# ---------------------------------------------------------------------------
# The library: two interests, every row class, marks, and rows that are not
# classified at all
# ---------------------------------------------------------------------------


def _build_library(directory: Path) -> tuple[CoreDB, SearchDB]:
    directory.mkdir(parents=True, exist_ok=True)
    core = CoreDB(directory / "litloft.db")
    search = SearchDB(directory / "search.db")

    tilted = np.zeros(DIM, dtype=np.float32)
    tilted[0], tilted[4] = math.cos(math.radians(10)), math.sin(math.radians(10))

    files = {
        # cats: lane members
        "c0": _cats(0), "c1": _cats(0), "c2": _cats(0), "c3": _cats(0),
        "c4": _cats(0), "c5": _cats(0), "k1": _cats(0), "c6": tilted,
        # trains: lane members
        "fv1": _trains(0), **{f"t{i}": _trains(0) for i in range(7)},
        # glanced, set aside, no length, legacy zero
        "g1": _cats(8), "g2": _cats(9), "g3": _trains(8), "fut": _cats(7),
        "s1": _trains(9), "s2": _cats(40), "n2": _trains(10),
        "n1": _cats(10), "L0": _cats(30), "L1": _cats(50), "L2": None,
        # rows the classification does not keep
        "d1": _cats(5.4), "tv1": _cats(5.0), "mv1": _cats(5.2),
        "bad1": _cats(20), "far": _cats(21), "old1": _cats(22),
        # never played by v1
        "p1": _cats(6), "p2": _cats(15), "q1": _trains(6), "m1": _trains(7),
    }
    special = {
        "c1": {"title": "", "filename": "c1-file.mp4"},
        "c2": {"title": "", "filename": ""},
        "k1": {"liked": True},
        "fv1": {"favourite": True},
        "L2": {"liked": True},
        "d1": {"kind": "document"},
        "tv1": {"deleted": True},
        "mv1": {"missing": True},
    }
    for file_id, vec in files.items():
        _file(core, file_id, **special.get(file_id, {}))
        if vec is not None:
            kind = "document" if file_id == "d1" else "video"
            search.index(file_id, vec, kind=kind)
    _file(core, "b1", drive="b")
    search.index("b1", _cats(5.6), drive="b")

    # v2 and v3 are written first: the order of the report is not insertion order.
    core.watch("p2", viewer="v2", played_at=_ago(1), position=0.0)
    core.watch("b1", viewer="v3", played_at=_ago(1), position=0.0)

    def watch(file_id, days, **kw):
        played = _ago(days)
        if kw.pop("dismissed", False):
            kw["dismissed_at"] = played + timedelta(hours=1)
        core.watch(file_id, played_at=played, **kw)

    for file_id, days in (("c0", 0.5), ("c1", 3), ("c2", 7), ("c5", 10)):
        watch(file_id, days, position=0.0)
        _covered(core, file_id)
    watch("k1", 2, position=0.0, dismissed=True)
    for file_id, days in (("c3", 14), ("c4", 15), ("c6", 29)):
        watch(file_id, days, position=100.0)
    watch("fv1", 1, position=0.0)
    watch("t0", 31, position=25.0)
    for file_id, days in (("t1", 40), ("t2", 89), ("t3", 90), ("t4", 91), ("t5", 100), ("t6", 120)):
        watch(file_id, days, position=100.0)

    watch("g1", 2, position=0.0)
    watch("g2", 4, position=100.0)
    watch("g3", 6, position=0.0)
    core.watch("fut", played_at=NOW + timedelta(days=0.5), position=0.0)
    watch("s1", 5, position=0.0, dismissed=True)
    watch("s2", 3, position=0.0, dismissed=True)
    _covered(core, "s2")
    watch("n2", 6, position=0.0, duration=0.0, dismissed=True)
    watch("n1", 2, position=0.0, duration=0.0)
    watch("L0", 35, position=0.0)
    watch("L1", 40, position=100.0, dismissed=True)
    watch("L2", 45, position=0.0)

    watch("d1", 2, position=0.0)
    watch("tv1", 2, position=0.0)
    watch("mv1", 2, position=0.0)
    core.watch("bad1", played_at="not-a-date", position=0.0)
    core.watch("far", played_at=NOW + timedelta(days=2), position=0.0)
    watch("old1", profile.PROFILE_WINDOW_DAYS + 1, position=0.0)
    watch("b1", 2, position=0.0)

    for file_id in ("g1", "m1", "d1", "tv1"):
        _mark(search, file_id)
    _mark(search, "b1", drive="b")
    _mark(search, "p1", viewer="v2")
    return core, search


class _Library:
    def __init__(self, directory: Path):
        core, search = _build_library(directory)
        self.lanes = _oracle_lanes(core, search)
        _dispose(core, search)
        self.core_path, self.search_path = core.path, search.path
        self._runs: dict[tuple, subprocess.CompletedProcess] = {}

    def run(self, *args: str) -> subprocess.CompletedProcess:
        if args not in self._runs:
            self._runs[args] = _interests(self.core_path, self.search_path, *args)
        return self._runs[args]

    def ok(self, *args: str) -> subprocess.CompletedProcess:
        result = self.run("--drive", "a", *args)
        assert result.returncode == 0, result.stderr
        return result

    def copy(self, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        core = directory / self.core_path.name
        search = directory / self.search_path.name
        shutil.copy(self.core_path, core)
        shutil.copy(self.search_path, search)
        return core, search

    def is_cats(self, lane) -> bool:
        return abs(float(lane.centroid[0])) > 0.5


@pytest.fixture(scope="module")
def library(tmp_path_factory) -> _Library:
    return _Library(tmp_path_factory.mktemp("interests"))


_CATS_ROW = ["8", "6", "8", "8.5", "5", "3", "0.375", "1", "1"]
_TRAINS_ROW = ["8", "1", "5", "89.5", "1", "1", "0.500", "2", "1"]
_CATS_MEMBERS = [
    "- T-c0 (1.00)", "- c1-file.mp4 (1.00)", "- c2 (1.00)", "- T-c3 (1.00)",
    "- T-c4 (1.00)", "- T-c5 (1.00)", "- T-k1 (1.00)", "- T-c6 (1.00)",
]
_TRAINS_MEMBERS = ["- T-fv1 (1.00)", "- T-t0 (0.50)"] + [f"- T-t{i} (1.00)" for i in range(1, 7)]
_CATS_PICKS = ["- T-p1", "- T-fut", "- T-g2"]
_TRAINS_PICKS = ["- T-q1", "- T-g3", "- T-s1"]


def _lane_block(n: int, members: list[str] | None, picks: list[str] | None) -> str:
    lines = [f"### {n} clip_thumbnail"]
    if members is not None:
        lines += ["members", *members]
    if picks is not None:
        lines += ["picks", *picks]
    return "\n".join(lines)


def _expected_lane_blocks(library, *, titles: int, picks: int) -> list[str]:
    out = []
    for n, lane in enumerate(library.lanes, start=1):
        cats = library.is_cats(lane)
        members = (_CATS_MEMBERS if cats else _TRAINS_MEMBERS)[:titles] if titles else None
        chosen = (_CATS_PICKS if cats else _TRAINS_PICKS)[:picks] if picks else None
        out.append(_lane_block(n, members, chosen))
    return out


# ---------------------------------------------------------------------------
# Item 1.3, 9: viewers and the layout of the report
# ---------------------------------------------------------------------------


def test_spec_addon_156_viewers_in_the_drive_sorted_by_id(library):
    """SPEC-ADDON-156 1.3: viewers with history in the drive, ascending; another drive's viewer is not one."""
    result = library.ok()

    assert list(_viewers(result.stdout)) == ["v1", "v2"]


def test_spec_addon_156_the_report_layout(library):
    """SPEC-ADDON-156 9: blocks joined by one blank line, lines by one newline, in the stated order."""
    result = library.ok()

    assert "\n\n\n" not in result.stdout
    blocks = _blocks(result.stdout)
    assert len(blocks) == 8
    assert blocks[0] == "viewer v1"
    assert blocks[1] == _counts(watched=17, glanced=4, set_aside=3, no_length=1,
                                legacy_zero=2, marked=2)
    table = blocks[2].split("\n")
    assert table[:2] == [HEADER, SEPARATOR]
    assert len(table) == 4
    assert blocks[3:5] == _expected_lane_blocks(library, titles=5, picks=3)
    assert blocks[5:] == ["viewer v2", _counts(glanced=1, marked=1), "no lanes"]


def test_spec_addon_156_i2_the_lanes_are_build_lanes_at_the_same_instant(library):
    """SPEC-ADDON-156 I2: count, order, type, member count and weight equal build_lanes'."""
    result = library.ok()

    assert len(library.lanes) == 2
    assert sorted(library.is_cats(lane) for lane in library.lanes) == [False, True]
    rows = [_cells(line) for line in _blocks(result.stdout)[2].split("\n")[2:]]
    assert [row[:4] for row in rows] == [
        [str(n), lane.channel, f"{lane.weight:.3f}", str(lane.member_count)]
        for n, lane in enumerate(library.lanes, start=1)
    ]


def test_spec_addon_156_i4_i7_the_lane_table(library):
    """SPEC-ADDON-156 I4 I7: ages, recorded, attributed glanced, set-aside and marked, glance rate."""
    result = library.ok()

    rows = [_cells(line) for line in _blocks(result.stdout)[2].split("\n")[2:]]
    for row, lane in zip(rows, library.lanes):
        assert row[3:] == (_CATS_ROW if library.is_cats(lane) else _TRAINS_ROW), row


def test_spec_addon_156_i5_i6_row_classes_and_marked(library):
    """SPEC-ADDON-156 I5 I6: each kept row in one class; marks on live videos of this viewer and drive."""
    result = library.ok()

    assert _viewers(result.stdout)["v1"][0] == _counts(
        watched=17, glanced=4, set_aside=3, no_length=1, legacy_zero=2, marked=2,
    )


def test_spec_addon_156_i8_members_and_picks(library):
    """SPEC-ADDON-156 I8: members by cosine with the centroid, then id; picks are score_lanes' first entries."""
    result = library.ok()

    assert _blocks(result.stdout)[3:5] == _expected_lane_blocks(library, titles=5, picks=3)


@pytest.mark.parametrize(
    ("titles", "picks"),
    [(20, 0), (0, 1), (0, 0), (1, 20)],
    ids=["all members, no picks", "no members, one pick", "heading only", "one member, all picks"],
)
def test_spec_addon_156_titles_and_picks_limits(library, titles, picks):
    """SPEC-ADDON-156 9: a list's label is left out at 0; the list holds at most n items."""
    result = library.ok("--titles", str(titles), "--picks", str(picks))

    blocks = _blocks(result.stdout)
    if picks == 20:
        for n, lane in enumerate(library.lanes, start=1):
            lines = blocks[2 + n].split("\n")
            expected = _CATS_MEMBERS if library.is_cats(lane) else _TRAINS_MEMBERS
            assert lines[:4] == [f"### {n} clip_thumbnail", "members", expected[0], "picks"]
            chosen = lines[4:]
            assert len(chosen) <= 20
            leading = _CATS_PICKS if library.is_cats(lane) else _TRAINS_PICKS
            assert chosen[:3] == leading
            assert not {"- T-tv1", "- T-mv1", "- T-d1", "- T-b1", "- T-g1", "- T-m1"} & set(chosen)
    else:
        assert blocks[3:5] == _expected_lane_blocks(library, titles=titles, picks=picks)


def test_spec_addon_156_i9_one_viewer(library):
    """SPEC-ADDON-156 1.3 I9: --viewer reports that viewer only, from its own history and marks."""
    result = library.ok("--viewer", "v2")

    assert _blocks(result.stdout) == ["viewer v2", _counts(glanced=1, marked=1), "no lanes"]


def test_spec_addon_156_i1_a_run_changes_neither_database(library, tmp_path):
    """SPEC-ADDON-156 I1: both files byte-identical after a successful run."""
    core, search = library.copy(tmp_path)
    before = _fingerprint(core, search)

    result = _interests(core, search, "--drive", "a")

    assert result.returncode == 0, result.stderr
    assert "viewer v1" in result.stdout
    assert _fingerprint(core, search) == before


# ---------------------------------------------------------------------------
# Item 2: failure cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [["--drive", "nowhere"], ["--drive", "a", "--viewer", "ghost"], ["--drive", "b", "--viewer", "v2"]],
    ids=["unknown drive", "unknown viewer", "viewer with no history in that drive"],
)
def test_spec_addon_156_no_viewers(library, args):
    """SPEC-ADDON-156 2: `no viewers`, exit 0."""
    assert library.ok().stdout.startswith("viewer v1")

    result = library.run(*args)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "no viewers"


@pytest.mark.parametrize("missing", ["core", "search"])
def test_spec_addon_156_a_database_that_cannot_be_opened(library, tmp_path, missing):
    """SPEC-ADDON-156 2: exit 1, `<ErrorType>: <message>` on stderr, stdout empty, nothing created."""
    core, search = library.copy(tmp_path)
    assert _interests(core, search, "--drive", "a").returncode == 0

    absent = tmp_path / "absent.db"
    paths = (absent, search) if missing == "core" else (core, absent)
    result = _interests(*paths, "--drive", "a")

    assert result.returncode == 1
    assert result.stdout == ""
    assert _ERROR_LINE.match(result.stderr), result.stderr
    assert not absent.exists()


def test_spec_addon_156_i1_i10_unreadable_marks_fail_the_run(library, tmp_path):
    """SPEC-ADDON-156 I1 I10: a failed read exits 1, prints nothing to stdout and writes nothing."""
    core, search_path = library.copy(tmp_path)
    assert _interests(core, search_path, "--drive", "a").returncode == 0
    search = SearchDB(search_path, create_models=False)
    search.execute("ALTER TABLE pickup_not_interested RENAME TO pickup_not_interested_away")
    search.execute("CREATE TABLE pickup_not_interested (unrelated INTEGER)")
    search.engine.dispose()
    before = _fingerprint(core, search_path)

    result = _interests(core, search_path, "--drive", "a")

    assert result.returncode == 1
    assert result.stdout == ""
    assert _ERROR_LINE.match(result.stderr), result.stderr
    assert _fingerprint(core, search_path) == before


def test_spec_addon_156_i10_a_width_mismatch_fails_the_whole_run(library, tmp_path):
    """SPEC-ADDON-156 I10: an embedding of another width exits 1 with nothing on stdout."""
    core_path, search_path = library.copy(tmp_path)
    assert _interests(core_path, search_path, "--drive", "a").returncode == 0
    core = CoreDB.__new__(CoreDB)
    core.path, core.engine = core_path, create_engine(f"sqlite:///{core_path}")
    _file(core, "w")
    core.watch("w", viewer="v2", played_at=_ago(1), position=0.0)
    core.engine.dispose()
    search = SearchDB(search_path, create_models=False)
    search.index("w", np.ones(DIM // 2, dtype=np.float32) / 2)
    search.engine.dispose()

    result = _interests(core_path, search_path, "--drive", "a")

    assert result.returncode == 1
    assert result.stdout == ""
    assert _ERROR_LINE.match(result.stderr), result.stderr


@pytest.mark.parametrize(
    ("option", "value"),
    [("--titles", "21"), ("--titles", "-1"), ("--titles", "x"),
     ("--picks", "21"), ("--picks", "-1"), ("--picks", "1.5")],
)
def test_spec_addon_156_titles_and_picks_out_of_range(library, option, value):
    """SPEC-ADDON-156 2: argparse's usage error, exit 2."""
    assert library.ok(option, "20").returncode == 0
    assert library.ok(option, "0").returncode == 0

    result = library.run("--drive", "a", option, value)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "usage:" in result.stderr
    assert option in result.stderr


# ---------------------------------------------------------------------------
# Item 1.6: one row per viewer, so each counts line holds a single class
# ---------------------------------------------------------------------------

_DISMISSED_LATER = object()
_DISMISSED_SAME = object()
_DISMISSED_EARLIER = object()

#: (viewer, file kwargs, last_played_at, watch kwargs, covered, expected class)
_CLASS_ROWS = [
    ("liked_dism", {"liked": True}, _ago(1), {"position": 0.0, "dismissed_at": _DISMISSED_LATER}, False, "watched"),
    ("fav_nolen", {"favourite": True}, _ago(1), {"position": 0.0, "duration": 0.0}, False, "watched"),
    ("liked_legacy", {"liked": True}, _ago(20), {"position": 0.0}, False, "watched"),
    ("legacy_dism", {}, _ago(20), {"position": 100.0, "dismissed_at": _DISMISSED_LATER}, False, "legacy zero"),
    ("legacy_quarter", {}, _ago(20), {"position": 25.0}, False, "watched"),
    ("legacy_brief", {}, _ago(20), {"position": 5.0}, False, "legacy zero"),
    ("legacy_nolen", {}, _ago(20), {"position": 0.0, "duration": 0.0}, False, "legacy zero"),
    ("legacy_embed", {"mime": LOFT_MIME}, _ago(20), {"position": 0.0, "duration": 0.0}, False, "watched"),
    ("before_r", {}, R - timedelta(seconds=1), {"position": 0.0}, False, "legacy zero"),
    ("at_r_covered", {}, R, {"position": 0.0}, True, "watched"),
    ("at_r_position", {}, R, {"position": 100.0}, False, "glanced"),
    ("dism_covered", {}, _ago(1), {"position": 0.0, "dismissed_at": _DISMISSED_LATER}, True, "set aside"),
    ("dism_nolen", {}, _ago(1), {"position": 0.0, "duration": 0.0, "dismissed_at": _DISMISSED_LATER},
     False, "set aside"),
    ("dism_same", {}, _ago(1), {"position": 0.0, "dismissed_at": _DISMISSED_SAME}, False, "set aside"),
    ("dism_earlier", {}, _ago(1), {"position": 0.0, "dismissed_at": _DISMISSED_EARLIER}, False, "glanced"),
    ("nolen_zero", {}, _ago(1), {"position": 0.0, "duration": 0.0}, False, "no length"),
    ("nolen_negative", {}, _ago(1), {"position": 3.0, "duration": -5.0}, False, "no length"),
    ("embed_nolen", {"mime": LOFT_MIME}, _ago(1), {"position": 0.0, "duration": 0.0}, False, "no length"),
    ("covered", {}, _ago(1), {"position": 0.0}, True, "watched"),
    ("position_only", {}, _ago(1), {"position": 100.0}, False, "glanced"),
    ("future_inside", {}, NOW + timedelta(days=0.5), {"position": 0.0}, False, "glanced"),
    ("window_inside", {}, _ago(profile.PROFILE_WINDOW_DAYS - 1), {"position": 25.0}, False, "watched"),
]

#: Rows the classification does not keep, all for viewer `filtered`, beside one glanced row.
_FILTERED_ROWS = [
    ("document", {"kind": "document"}, _ago(1), {"position": 0.0, "duration": 0.0}),
    ("trashed", {"deleted": True}, _ago(1), {"position": 0.0}),
    ("missing", {"missing": True}, _ago(1), {"position": 0.0}),
    ("unparseable", {}, "not-a-date", {"position": 0.0}),
    ("future_beyond", {}, NOW + timedelta(days=2), {"position": 0.0}),
    ("window_beyond", {}, _ago(profile.PROFILE_WINDOW_DAYS + 1), {"position": 25.0}),
    ("kept", {}, _ago(1), {"position": 0.0}),
]

_CLASS_FIELD = {
    "watched": "watched", "glanced": "glanced", "set aside": "set_aside",
    "no length": "no_length", "legacy zero": "legacy_zero",
}


def _dismissal(marker, played):
    if marker is _DISMISSED_LATER:
        return played + timedelta(hours=1)
    if marker is _DISMISSED_SAME:
        return played
    if marker is _DISMISSED_EARLIER:
        return played - timedelta(hours=1)
    return marker


@pytest.fixture(scope="module")
def classes_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("classes")
    core = CoreDB(directory / "litloft.db")
    search = SearchDB(directory / "search.db")
    for viewer, file_kw, played, watch_kw, covered, _ in _CLASS_ROWS:
        file_id = f"f_{viewer}"[:12]
        _file(core, file_id, **file_kw)
        kw = dict(watch_kw)
        if "dismissed_at" in kw:
            kw["dismissed_at"] = _dismissal(kw["dismissed_at"], played)
        core.watch(file_id, viewer=viewer, played_at=played, **kw)
        if covered:
            _covered(core, file_id, viewer=viewer)
    for name, file_kw, played, watch_kw in _FILTERED_ROWS:
        file_id = f"x_{name}"[:12]
        _file(core, file_id, **file_kw)
        core.watch(file_id, viewer="filtered", played_at=played, **watch_kw)
    _dispose(core, search)
    return _interests(core.path, search.path, "--drive", "a")


@pytest.mark.parametrize(
    ("viewer", "expected"),
    [(row[0], row[5]) for row in _CLASS_ROWS],
    ids=[row[0] for row in _CLASS_ROWS],
)
def test_spec_addon_156_i5_i6_the_first_class_that_matches(classes_run, viewer, expected):
    """SPEC-ADDON-156 1.6 I5 I6: liked/favourite, then legacy, dismissed, no length, watched, glanced."""
    assert classes_run.returncode == 0, classes_run.stderr

    section = _viewers(classes_run.stdout)[viewer]

    assert section == [_counts(**{_CLASS_FIELD[expected]: 1}), "no lanes"]


def test_spec_addon_156_i5_rows_outside_the_profile_filters_are_not_classified(classes_run):
    """SPEC-ADDON-156 I5: not live, not a video, unparseable, too far ahead, outside the window."""
    assert classes_run.returncode == 0, classes_run.stderr

    assert _viewers(classes_run.stdout)["filtered"] == [_counts(glanced=1), "no lanes"]


# ---------------------------------------------------------------------------
# Item 7 and the edges of the table and the lists
# ---------------------------------------------------------------------------


def test_spec_addon_156_i4_i7_no_vector_no_lane_and_an_empty_pick_list(tmp_path):
    """SPEC-ADDON-156 I4 I7 9: unattributed without a vector of the lane's type, `-` rate, `- none`."""
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    for i in range(11):
        _file(core, f"o{i:02d}")
        core.watch(f"o{i:02d}", played_at=_ago(31 + i), position=100.0)
        search.index(f"o{i:02d}", _cats(0))
    _file(core, "gx")
    core.watch("gx", played_at=_ago(1), position=0.0)
    search.index("gx", _cats(0), channel="tfidf_keywords")
    _file(core, "gy")
    core.watch("gy", played_at=_ago(1), position=0.0)
    _file(core, "sx")
    core.watch("sx", played_at=_ago(1), position=0.0, dismissed_at=_ago(1) + timedelta(hours=1))
    search.index("sx", _cats(0), channel="tfidf_keywords")
    _file(core, "mz")
    _mark(search, "mz")
    lanes = _oracle_lanes(core, search)
    _dispose(core, search)

    result = _interests(core.path, search.path, "--drive", "a", "--titles", "0")

    assert result.returncode == 0, result.stderr
    assert len(lanes) == 1
    blocks = _blocks(result.stdout)
    assert blocks[1] == _counts(watched=11, glanced=2, set_aside=1, marked=1)
    table = blocks[2].split("\n")
    assert table[:2] == [HEADER, SEPARATOR]
    assert _cells(table[2]) == [
        "1", "clip_thumbnail", f"{lanes[0].weight:.3f}", "11",
        "0", "11", "36.0", "0", "0", "-", "0", "0",
    ]
    assert blocks[3] == "### 1 clip_thumbnail\npicks\n- none"


def test_spec_addon_156_i4_a_tie_goes_to_the_lane_listed_first(tmp_path):
    """SPEC-ADDON-156 1.7: equal cosine with two centroids attributes to the first lane only."""
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    for i in range(6):
        for prefix, vec in (("a", _cats(0)), ("b", _trains(0))):
            file_id = f"{prefix}{i}"
            _file(core, file_id)
            core.watch(file_id, played_at=_ago(1 + i), position=0.0)
            _covered(core, file_id)
            search.index(file_id, vec)
    between = np.zeros(DIM, dtype=np.float32)
    between[0] = between[2] = np.float32(1 / math.sqrt(2))
    _file(core, "tie")
    core.watch("tie", played_at=_ago(1), position=0.0)
    search.index("tie", between)
    lanes = _oracle_lanes(core, search)
    _dispose(core, search)

    result = _interests(core.path, search.path, "--drive", "a")

    assert result.returncode == 0, result.stderr
    assert len(lanes) == 2
    rows = [_cells(line) for line in _blocks(result.stdout)[2].split("\n")[2:]]
    assert [row[8] for row in rows] == ["1", "0"]


def test_spec_addon_156_i4_attribution_is_per_embedding_type(tmp_path):
    """SPEC-ADDON-156 1.7 I4: a file is attributed once in each type it has a vector of."""
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    for i in range(6):
        file_id = f"m{i}"
        _file(core, file_id)
        core.watch(file_id, played_at=_ago(1 + i), position=0.0)
        _covered(core, file_id)
        search.index(file_id, _cats(0))
        search.index(file_id, _trains(0), channel="tfidf_keywords")
    _file(core, "both")
    core.watch("both", played_at=_ago(1), position=0.0)
    search.index("both", _cats(10))
    search.index("both", _trains(10), channel="tfidf_keywords")
    _file(core, "clip_only")
    core.watch("clip_only", played_at=_ago(1), position=0.0)
    search.index("clip_only", _cats(10))
    lanes = _oracle_lanes(core, search)
    _dispose(core, search)

    result = _interests(core.path, search.path, "--drive", "a")

    assert result.returncode == 0, result.stderr
    assert sorted(lane.channel for lane in lanes) == ["clip_thumbnail", "tfidf_keywords"]
    rows = [_cells(line) for line in _blocks(result.stdout)[2].split("\n")[2:]]
    assert {row[1]: row[8] for row in rows} == {"clip_thumbnail": "2", "tfidf_keywords": "1"}


@pytest.mark.parametrize(
    ("rows", "capped"),
    [(profile.PROFILE_VECTOR_CAP - 1, False), (profile.PROFILE_VECTOR_CAP, True),
     (profile.PROFILE_VECTOR_CAP + 1, True)],
    ids=["one under the cap", "at the cap", "one over the cap"],
)
def test_spec_addon_156_the_capped_line(tmp_path, rows, capped):
    """SPEC-ADDON-156 9 I5: the line appears when profile_history returned the cap; counts are uncapped."""
    core = CoreDB(tmp_path / "litloft.db")
    search = SearchDB(tmp_path / "search.db")
    with core.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO files (id, drive, file_path, file_type, mime_type, liked_at,"
                " created_at, updated_at) VALUES (:i, 'a', :p, 'video', 'video/mp4', :l, :c, :c)"
            ),
            [{"i": f"z{i:05d}", "p": f"/z{i:05d}", "l": ts(_ago(1)), "c": ts(CREATED)}
             for i in range(rows)],
        )
        conn.execute(
            text(
                "INSERT INTO watch_history (viewer_id, file_id, playback_position, duration,"
                " last_played_at) VALUES ('v1', :i, 0, 100, :t)"
            ),
            [{"i": f"z{i:05d}", "t": ts(_ago(1 + i / 1000))} for i in range(rows)],
        )
    _dispose(core, search)

    result = _interests(core.path, search.path, "--drive", "a")

    assert result.returncode == 0, result.stderr
    counts = _counts(watched=rows)
    if capped:
        counts += f"\nmembers capped at {profile.PROFILE_VECTOR_CAP}"
    assert _blocks(result.stdout) == ["viewer v1", counts, "no lanes"]


# ---------------------------------------------------------------------------
# Item 5: cluster_lanes, and build_lanes delegating to it
# ---------------------------------------------------------------------------


def _same_lanes(got, expected) -> None:
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert (g.channel, g.cluster_id, g.member_count, g.raw_weight, g.weight) == \
               (e.channel, e.cluster_id, e.member_count, e.raw_weight, e.weight)
        assert np.array_equal(g.centroid, e.centroid)


@pytest.fixture()
def wired_library(monkeypatch, tmp_path):
    core, search = _build_library(tmp_path)
    wire(monkeypatch, core, search)
    return core, search


def test_spec_addon_156_i2_cluster_lanes_lists_build_lanes_with_their_members(wired_library):
    """SPEC-ADDON-156 1.5 I2: the same lanes in order, each with its members' unit vectors."""
    history = profile.profile_history("a", "v1", now=NOW)

    pairs = profile.cluster_lanes(history, key=KEY, now=NOW)

    _same_lanes([lane for lane, _ in pairs], profile.build_lanes(history, key=KEY, now=NOW))
    members = {}
    for lane, vectors in pairs:
        assert len(vectors) == lane.member_count
        stored = profile.representative_vectors(list(vectors), lane.channel)
        for file_id, vec in vectors.items():
            assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)
            np.testing.assert_allclose(vec, stored[file_id] / np.linalg.norm(stored[file_id]), atol=1e-6)
        members[abs(float(lane.centroid[0])) > 0.5] = set(vectors)
    assert members == {
        True: {"c0", "c1", "c2", "c3", "c4", "c5", "c6", "k1"},
        False: {"fv1", *(f"t{i}" for i in range(7))},
    }


def test_spec_addon_156_i3_build_lanes_and_cluster_lanes_use_the_vectors_passed(wired_library):
    """SPEC-ADDON-156 1.5 I3: `vectors` reaches the clustering through both functions."""
    history = profile.profile_history("a", "v1", now=NOW)
    flipped = np.zeros(DIM, dtype=np.float32)
    flipped[5] = 1.0

    def vectors(file_ids, channel):
        got = profile.representative_vectors(file_ids, channel)
        return {f: (flipped if f.startswith("t") else v) for f, v in got.items()}

    pairs = profile.cluster_lanes(history, key=KEY, now=NOW, vectors=vectors)

    _same_lanes([lane for lane, _ in pairs],
                profile.build_lanes(history, key=KEY, now=NOW, vectors=vectors))
    seen = set()
    for _, member_vectors in pairs:
        for file_id, vec in member_vectors.items():
            if file_id.startswith("t"):
                seen.add(file_id)
                assert np.array_equal(vec, flipped)
    assert seen == {f"t{i}" for i in range(7)}
