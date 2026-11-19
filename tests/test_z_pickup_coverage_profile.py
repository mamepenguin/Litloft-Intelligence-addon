"""Pickup judges a video watched by the share of it that was played.

SPEC-ID: SPEC-ADDON-149
SPEC-ID: SPEC-ADDON-150
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.pickup import profile  # noqa: E402
from tests.pickup_accuracy_rig import (  # noqa: E402
    LOFT_MIME,
    CoreDB,
    SearchDB,
    wire,
)

R = datetime(2026, 11, 5, 0, 0, 0)
NOW = datetime(2026, 11, 15, 12, 0, 0)
PLAYED = NOW - timedelta(days=1)
LEGACY = datetime(2026, 10, 20, 12, 0, 0)


@pytest.fixture()
def core(monkeypatch, tmp_path):
    db = CoreDB(tmp_path / "litloft.db")
    wire(monkeypatch, db, SearchDB(tmp_path / "search.db"))
    return db


def _engagements() -> dict[str, float]:
    return {w.file_id: w.engagement for w in profile.profile_history("a", "v1", now=NOW)}


def _segments(core: CoreDB, file_id: str, indexes, *, seconds: float = 5.0, viewer: str = "v1"):
    for n in indexes:
        core.execute(
            "INSERT INTO played_segments VALUES (:v, :f, :n, :s)",
            {"v": viewer, "f": file_id, "n": n, "s": seconds},
        )


# ---------------------------------------------------------------------------
# One video row: whether it is watched, and its engagement
# ---------------------------------------------------------------------------

#: (file id, file kwargs, watch kwargs, this viewer's segments as (indexes, seconds),
#:  other viewer's segment indexes, expected engagement or None when the row is
#:  no seed, whether the video is excluded)
_ROWS = [
    ("opened_nothing_played", {}, {"position": 0.0, "duration": 0.0}, None, None, None, False),
    ("position_at_end", {}, {"position": 100.0}, None, None, None, False),
    ("position_past_end", {}, {"position": 150.0}, None, None, None, False),
    ("nine_of_twenty", {}, {"position": 100.0}, (range(9), 5.0), None, None, False),
    ("ten_of_twenty", {}, {"position": 0.0}, (range(10), 5.0), None, 1.0, True),
    ("every_segment", {}, {"position": 3.0}, (range(20), 5.0), None, 1.0, True),
    ("whole_however_little", {}, {"position": 0.0}, (range(10), 0.1), None, 1.0, True),
    ("replayed_counts_once", {}, {"position": 0.0}, (range(9), 60.0), None, None, False),
    ("zero_seconds_do_not_count", {}, {"position": 100.0}, (range(20), 0.0), None, None, False),
    ("spread_reach_half", {}, {"position": 0.0}, (range(0, 20, 2), 5.0), None, 1.0, True),
    ("other_viewers_segments", {}, {"position": 100.0}, None, range(20), None, False),
    ("past_duration_ignored", {}, {"position": 0.0}, (list(range(9)) + [20, 21, 40], 5.0),
     None, None, False),
    ("last_segment_4s_49_of_99", {}, {"position": 0.0, "duration": 99.0},
     (list(range(9)) + [19], 5.0), None, None, False),
    ("last_segment_4s_54_of_99", {}, {"position": 0.0, "duration": 99.0},
     (list(range(10)) + [19], 5.0), None, 1.0, True),
    ("unknown_duration_segments", {}, {"position": 0.0, "duration": 0.0}, (range(20), 5.0),
     None, None, False),
    ("infinite_duration_segments", {}, {"position": 0.0, "duration": math.inf}, (range(20), 5.0),
     None, None, False),
    ("negative_duration_segments", {}, {"position": 0.0, "duration": -100.0}, (range(20), 5.0),
     None, None, False),
    ("liked_never_played", {"liked": True}, {"position": 0.0, "duration": 0.0}, None, None, 1.0, True),
    ("favourite_never_played", {"favourite": True}, {"position": 0.0, "duration": 0.0},
     None, None, 1.0, True),
    ("liked_beats_dismissed", {"liked": True},
     {"position": 0.0, "dismissed_at": PLAYED + timedelta(hours=1)}, None, None, 1.0, True),
    ("watched_dismissed", {}, {"position": 0.0, "dismissed_at": PLAYED + timedelta(hours=1)},
     (range(20), 5.0), None, None, True),
    ("glanced_dismissed", {}, {"position": 0.0, "dismissed_at": PLAYED + timedelta(hours=1)},
     None, None, None, False),
    ("play_after_dismissal", {}, {"position": 0.0, "dismissed_at": PLAYED - timedelta(hours=1)},
     (range(20), 5.0), None, 1.0, True),
    ("embed_never_reported", {"mime": LOFT_MIME}, {"position": 0.0, "duration": 0.0},
     None, None, None, False),
    ("embed_infinite_duration", {"mime": LOFT_MIME}, {"position": 3.0, "duration": math.inf},
     None, None, None, False),
    ("embed_over_half", {"mime": LOFT_MIME}, {"position": 0.0}, (range(12), 5.0), None, 1.0, True),
    ("short_half_by_position", {}, {"position": 5.0, "duration": 10.0}, None, None, 1.0, True),
    ("short_under_half", {}, {"position": 4.9, "duration": 10.0}, None, None, None, False),
    ("short_covered_positioned_early", {}, {"position": 2.0, "duration": 19.9}, (range(4), 5.0),
     None, None, False),
    ("short_19_9_half_by_position", {}, {"position": 10.0, "duration": 19.9}, None, None, 1.0, True),
    ("short_infinite_position", {}, {"position": math.inf, "duration": 10.0}, None, None, None, False),
    ("short_negative_position", {}, {"position": -15.0, "duration": 10.0}, None, None, None, False),
    ("twenty_position_alone", {}, {"position": 20.0, "duration": 20.0}, None, None, None, False),
    ("twenty_two_segments", {}, {"position": 0.0, "duration": 20.0}, (range(2), 5.0), None, 1.0, True),
]


def test_spec_addon_149_150_i1_i2_each_video_since_recording_start(core):
    """SPEC-ADDON-149 / SPEC-ADDON-150 I1 I2: watched by coverage, engagement 1 or none."""
    for file_id, file_kw, watch_kw, mine, theirs, _, _ in _ROWS:
        core.add_file(file_id, **file_kw)
        core.watch(file_id, **{"played_at": PLAYED, "position": 100.0, "duration": 100.0, **watch_kw})
        if mine is not None:
            indexes, seconds = mine
            _segments(core, file_id, indexes, seconds=seconds)
        if theirs is not None:
            core.watch(file_id, viewer="v2", played_at=PLAYED)
            _segments(core, file_id, theirs, viewer="v2")

    got = _engagements()

    assert got == pytest.approx({row[0]: row[5] for row in _ROWS if row[5] is not None})
    assert profile.watched_file_ids("a", "v1") == {row[0] for row in _ROWS if row[6]}


# ---------------------------------------------------------------------------
# Recording start
# ---------------------------------------------------------------------------

#: (file id, last_played_at, position, duration, mime, expected engagement, excluded)
_LEGACY_ROWS = [
    ("second_before_r", R - timedelta(seconds=1), 25.0, 100.0, None, 0.5, True),
    ("exactly_r", R, 25.0, 100.0, None, None, False),
    ("second_after_r", R + timedelta(seconds=1), 25.0, 100.0, None, None, False),
    ("legacy_brief_open", LEGACY, 5.0, 100.0, None, None, True),
    ("legacy_half", LEGACY, 50.0, 100.0, None, 1.0, True),
    ("legacy_embed_unreported", LEGACY, 0.0, 0.0, LOFT_MIME, 0.5, True),
    ("offset_before_r", "2026-11-05T08:00:00+09:00", 25.0, 100.0, None, 0.5, True),
    ("offset_after_r", "2026-11-04T23:30:00-01:00", 25.0, 100.0, None, None, False),
    ("unparseable", "not-a-date", 0.0, 0.0, None, None, True),
]


def test_spec_addon_149_150_i3_rows_before_recording_start_keep_version_4(core):
    """SPEC-ADDON-149 / SPEC-ADDON-150 I3: L < R or unparseable is excluded and weighted as at version 4."""
    for file_id, played_at, position, duration, mime, _, _ in _LEGACY_ROWS:
        core.add_file(file_id, mime=mime)
        core.watch(file_id, played_at=played_at, position=position, duration=duration)

    got = _engagements()

    assert got == pytest.approx({row[0]: row[5] for row in _LEGACY_ROWS if row[5] is not None})
    assert profile.watched_file_ids("a", "v1") == {row[0] for row in _LEGACY_ROWS if row[6]}


# ---------------------------------------------------------------------------
# The exclusion set
# ---------------------------------------------------------------------------


def test_spec_addon_149_rule_1_the_exclusion_set(core):
    """SPEC-ADDON-149 rule 1: every non-video row and every watched video, dead ones included."""
    for file_id, kind in (("doc", "document"), ("image", "image"), ("audio", "audio")):
        core.add_file(file_id, kind=kind)
        core.watch(file_id, played_at=PLAYED, position=0.0, duration=0.0)
    for file_id, kw in (("watched", {}), ("watched_trashed", {"deleted": True}),
                        ("watched_missing", {"missing": True})):
        core.add_file(file_id, **kw)
        core.watch(file_id, played_at=PLAYED, position=0.0)
        _segments(core, file_id, range(10))
    for file_id, kw in (("glanced", {}), ("glanced_trashed", {"deleted": True}),
                        ("glanced_missing", {"missing": True})):
        core.add_file(file_id, **kw)
        core.watch(file_id, played_at=PLAYED, position=100.0)
    core.add_file("other_drive", drive="b")
    core.watch("other_drive", played_at=PLAYED, position=0.0)
    _segments(core, "other_drive", range(20))
    core.add_file("never_opened")

    assert profile.watched_file_ids("a", "v1") == {
        "doc", "image", "audio", "watched", "watched_trashed", "watched_missing",
    }


# ---------------------------------------------------------------------------
# States: what moves a video between glanced and watched, and what does not
# ---------------------------------------------------------------------------


def _excluded() -> set[str]:
    return profile.watched_file_ids("a", "v1")


def _set(core, sql, **params):
    core.execute(sql, params)


def test_spec_addon_149_states_glanced_to_watched_and_back(core):
    """SPEC-ADDON-149 states: coverage, likes, the reported duration and removing history move a video."""
    core.add_file("v")
    core.watch("v", played_at=PLAYED, position=0.0, duration=0.0)
    assert _excluded() == set()

    _set(core, "UPDATE watch_history SET duration = 100, playback_position = 30")
    _segments(core, "v", range(6))
    assert _excluded() == set()

    _segments(core, "v", range(6, 10))
    assert _excluded() == {"v"}

    _set(core, "UPDATE watch_history SET duration = 120")
    assert _excluded() == set()

    _set(core, "UPDATE watch_history SET duration = 90")
    assert _excluded() == {"v"}

    _set(core, "UPDATE watch_history SET duration = 120")
    _set(core, "UPDATE files SET liked_at = '2026-11-14 00:00:00'")
    assert _excluded() == {"v"}
    _set(core, "UPDATE files SET liked_at = NULL")
    assert _excluded() == set()
    _set(core, "UPDATE files SET is_favorite = 1")
    assert _excluded() == {"v"}
    _set(core, "UPDATE files SET is_favorite = 0")
    assert _excluded() == set()

    _set(core, "UPDATE watch_history SET duration = 100")
    assert _excluded() == {"v"}
    _set(core, "DELETE FROM watch_history")
    _set(core, "DELETE FROM played_segments")
    assert _excluded() == set()


def test_spec_addon_149_states_a_legacy_row_played_again_is_judged_by_coverage(core):
    """SPEC-ADDON-149 states: L moved past R without enough coverage makes it glanced."""
    core.add_file("v")
    core.watch("v", played_at=LEGACY, position=100.0)
    assert _excluded() == {"v"}

    _set(core, "UPDATE watch_history SET last_played_at = :t", t=str(PLAYED))
    assert _excluded() == set()


def test_spec_addon_149_states_a_short_video_saved_below_half(core):
    """SPEC-ADDON-149 states: a short video replayed and saved below half is glanced."""
    core.add_file("v")
    core.watch("v", played_at=PLAYED, position=10.0, duration=15.0)
    assert _excluded() == {"v"}

    _set(core, "UPDATE watch_history SET playback_position = 7.0")
    assert _excluded() == set()


def _move_position(core):
    _set(core, "UPDATE watch_history SET playback_position = 100")


def _dismiss(core):
    _set(core, "UPDATE watch_history SET dismissed_at = :t", t=str(PLAYED + timedelta(hours=1)))


def _trash(core):
    _set(core, "UPDATE files SET deleted_at = '2026-11-14 00:00:00'")


def _restore(core):
    _set(core, "UPDATE files SET deleted_at = NULL")


def _missing(core):
    _set(core, "UPDATE files SET missing_since = '2026-11-14 00:00:00'")


def _other_viewer(core):
    for file_id in ("w", "g"):
        core.watch(file_id, viewer="v2", played_at=PLAYED, position=0.0)
        _segments(core, file_id, range(20), viewer="v2")


@pytest.mark.parametrize(
    "changes",
    [[_move_position], [_dismiss], [_trash], [_trash, _restore], [_missing], [_other_viewer]],
    ids=["position", "dismiss", "trash", "trash then restore", "missing", "another viewer"],
)
def test_spec_addon_149_states_what_never_changes_watched(core, changes):
    """SPEC-ADDON-149 states, not allowed: none of these changes watched for a non-short video."""
    for file_id, segments in (("w", 10), ("g", 4)):
        core.add_file(file_id)
        core.watch(file_id, played_at=PLAYED, position=1.0)
        _segments(core, file_id, range(segments))
    assert _excluded() == {"w"}

    for change in changes:
        change(core)

    assert _excluded() == {"w"}
