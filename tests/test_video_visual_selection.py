"""The pure selection of SPEC-ADDON-112.

The sections and frame times it must produce from real stored rows are pinned
through ``GET /files/{id}/scenes`` in ``test_spec_addon_112_scene_sections.py``.
These unit tests cover the contract around that: an empty pool, optional
inputs, order independence (I2), coverage (I3), the section ceiling (I4), the
Ask rule (I5) and the steady second (I6).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

for _mod in ("PIL", "PIL.Image", "sqlite_vec"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import numpy as np  # noqa: E402

from app.workers.video_visual_selection import (  # noqa: E402
    Candidate,
    Frame,
    FrameText,
    Scan,
    SelectionInputs,
    TranscriptEmbedding,
    compute_candidate_fingerprint,
    select_scenes,
)


def _basis(dim: int, axis: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def _clip(timestamp: float, axis: int, dim: int = 8) -> Frame:
    return Frame(f"clip{axis}_{timestamp}", timestamp, _basis(dim, axis))


def _change(timestamp: float, axis: int, dim: int = 8) -> Frame:
    return Frame(f"chg_{timestamp}", timestamp, _basis(dim, axis), "clip_change")


def _clips(times: list[float], dim: int = 8) -> tuple[Frame, ...]:
    return tuple(_clip(t, i % dim, dim) for i, t in enumerate(times))


class TestEmptyAndOptionalInputs:
    def test_an_empty_pool_gives_both_lists_empty(self):
        selection = select_scenes(SelectionInputs(), 0.0, 600.0)

        assert selection.navigation == ()
        assert selection.ask == ()
        assert selection.candidate_count == 0

    def test_a_lone_frame_spans_the_whole_span(self):
        selection = select_scenes(
            SelectionInputs(scene_candidates=(_clip(100.0, 0),)), 0.0, 600.0,
        )

        [item] = selection.navigation
        assert (item.section_start, item.section_end, item.frame_time) == (0.0, 600.0, 100.0)
        assert item.kind == "start"
        assert item.embedding_type == "clip"

    def test_an_unknown_end_is_one_second_past_the_last_frame(self):
        inputs = SelectionInputs(
            scene_candidates=(_clip(100.0, 0), _clip(300.0, 1), _clip(500.0, 2)),
            duration=None,
        )

        selection = select_scenes(inputs, 0.0, None)

        assert [item.section_end for item in selection.navigation] == [300.0, 501.0]

    def test_a_span_ending_before_it_starts_is_empty(self):
        inputs = SelectionInputs(scene_candidates=(_clip(100.0, 0),))

        assert select_scenes(inputs, 600.0, 600.0).navigation == ()
        assert select_scenes(inputs, 700.0, 600.0).navigation == ()

    def test_no_scan_no_transcript_and_no_ocr_are_all_optional(self):
        inputs = SelectionInputs(scene_candidates=_clips([5.0, 100.0, 300.0]))

        selection = select_scenes(inputs, 0.0, 300.0)

        assert selection.navigation


class TestPool:
    def test_a_change_frame_within_three_seconds_of_a_scene_candidate_is_left_out(self):
        inputs = SelectionInputs(
            scene_candidates=(_clip(100.0, 0), _clip(300.0, 1)),
            change_frames=(
                _change(103.0, 2),
                _change(104.0, 3),
                _change(200.0, 4),
            ),
        )

        selection = select_scenes(inputs, 0.0, 600.0)

        # 103 is exactly 3 s away, so it joins the pool; 104 is more than 3 s.
        assert selection.candidate_count == 4

    def test_change_frames_alone_make_a_pool(self):
        inputs = SelectionInputs(change_frames=(
            _change(100.0, 0), _change(300.0, 1), _change(500.0, 2),
        ))

        selection = select_scenes(inputs, 0.0, 600.0)

        assert selection.candidate_count == 3
        assert [item.embedding_type for item in selection.navigation] == [
            "clip_change", "clip_change", "clip_change",
        ]


class TestOrderIndependence:
    def test_the_same_rows_in_any_order_give_the_same_lists(self):
        clips = _clips([5.0 + 10 * k for k in range(60)])
        changes = (_change(52.0, 3), _change(320.0, 4))
        transcript = tuple(
            TranscriptEmbedding(f"wh{i}", t, _basis(8, 0 if t < 300 else 1))
            for i, t in enumerate([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
        )
        texts = (FrameText(0.0, 200.0, "a" * 250), FrameText(200.0, 600.0, "b" * 250))

        forward = select_scenes(
            SelectionInputs(clips, changes, None, transcript, texts), 0.0, 600.0,
        )
        backward = select_scenes(
            SelectionInputs(
                tuple(reversed(clips)), tuple(reversed(changes)), None,
                tuple(reversed(transcript)), tuple(reversed(texts)),
            ),
            0.0, 600.0,
        )

        assert forward == backward


class TestCoverage:
    def test_sections_cover_the_span_without_gaps_and_hold_their_frame(self):
        inputs = SelectionInputs(scene_candidates=_clips([5.0 + 10 * k for k in range(60)]))

        selection = select_scenes(inputs, 0.0, 600.0)
        items = selection.navigation

        assert items[0].section_start == 0.0
        assert items[-1].section_end == 600.0
        for current, following in zip(items, items[1:]):
            assert current.section_end == following.section_start
        for item in items:
            assert item.section_start <= item.frame_time < item.section_end

    def test_every_pool_frame_lies_in_exactly_one_section(self):
        inputs = SelectionInputs(scene_candidates=_clips([5.0 + 10 * k for k in range(60)]))
        pool = [frame.timestamp for frame in inputs.scene_candidates]

        items = select_scenes(inputs, 0.0, 600.0).navigation

        assert sum(
            1 for t in pool if sum(
                item.section_start <= t < item.section_end for item in items
            ) == 1
        ) == len(set(pool))

    def test_the_section_count_stops_at_the_ceiling(self):
        inputs = SelectionInputs(
            scene_candidates=_clips([6.0 + 12 * k for k in range(600)], dim=64),
        )

        selection = select_scenes(inputs, 0.0, 7200.0)

        assert len(selection.navigation) <= 40
        assert selection.candidate_count == 600


class TestMergingAndAsk:
    def test_identical_pictures_merge_into_one_section(self):
        inputs = SelectionInputs(scene_candidates=tuple(
            Frame(f"c{i}", 5.0 + 10 * i, _basis(4, 0)) for i in range(60)
        ))

        selection = select_scenes(inputs, 0.0, 600.0)

        assert [(item.section_start, item.section_end) for item in selection.navigation] == [
            (0.0, 600.0),
        ]
        assert len(selection.ask) == 1

    def test_the_ask_list_drops_a_shorter_section_showing_what_a_kept_one_shows(self):
        mix = np.zeros(8, dtype=np.float32)
        mix[0] = 0.96
        mix[1] = float(np.sqrt(1.0 - 0.96 * 0.96))
        close = np.zeros(8, dtype=np.float32)
        close[0] = 0.94
        close[1] = float(np.sqrt(1.0 - 0.94 * 0.94))

        dropped = select_scenes(SelectionInputs(scene_candidates=(
            _clip(10.0, 0), _clip(40.0, 1), Frame("third", 70.0, mix),
        )), 0.0, 80.0)
        kept = select_scenes(SelectionInputs(scene_candidates=(
            _clip(10.0, 0), _clip(40.0, 1), Frame("third", 70.0, close),
        )), 0.0, 80.0)

        assert [item.frame_time for item in dropped.ask] == [10.0, 40.0]
        assert [item.frame_time for item in kept.ask] == [10.0, 40.0, 70.0]


class TestTextual:
    def _chars(self, first, count):
        return "".join(chr(0x4E00 + i) for i in range(first, first + count))

    @pytest.mark.parametrize("shift, sections", [(25, 2), (10, 1)], ids=["0.82", "0.92"])
    def test_identical_pictures_are_told_apart_by_their_bigram_jaccard(
        self, shift, sections,
    ):
        frames = tuple(Frame(f"c{i}", 5.0 + 10 * i, _basis(4, 0)) for i in range(60))

        selection = select_scenes(SelectionInputs(
            scene_candidates=frames,
            frame_texts=(FrameText(0.0, 300.0, self._chars(0, 251)),
                         FrameText(300.0, 600.0, self._chars(shift, 251))),
        ), 0.0, 600.0)

        assert len(selection.navigation) == sections


class TestSteadySecond:
    def _inputs(self, scan):
        return SelectionInputs(
            scene_candidates=tuple(
                Frame(f"c{i}", 5.5 + 10 * i, _basis(8, int((5.5 + 10 * i) // 75)))
                for i in range(60)
            ),
            scan=scan,
        )

    def _scan(self, *, sizes=None):
        count = 600
        return Scan(
            seconds=count,
            hashes=np.zeros((count, 256), dtype=bool),
            sizes=np.array(
                [sizes.get(s, 1000) if sizes else 1000 for s in range(count)],
                dtype=np.int64,
            ),
        )

    def test_a_steady_second_moves_the_frame_off_its_pool_time(self):
        item = select_scenes(self._inputs(self._scan()), 0.0, 600.0).navigation[0]

        assert item.frame_time == 6.0

    def test_no_scan_keeps_the_pool_frame_time(self):
        item = select_scenes(self._inputs(None), 0.0, 600.0).navigation[0]

        assert item.frame_time == 5.5

    def test_a_second_below_the_size_floor_is_skipped(self):
        item = select_scenes(
            self._inputs(self._scan(sizes={6: 449})), 0.0, 600.0,
        ).navigation[0]

        assert item.frame_time == 7.0


class TestCandidateFingerprint:
    def test_deterministic_for_same_input(self):
        v = _basis(8, 1)
        candidates = [Candidate("a", 1.0, v), Candidate("b", 2.0, v)]
        assert compute_candidate_fingerprint(candidates) == compute_candidate_fingerprint(
            candidates
        )

    def test_changes_when_candidate_set_changes(self):
        v = _basis(8, 1)
        full = [Candidate("a", 1.0, v), Candidate("b", 2.0, v)]
        truncated = full[:-1]
        assert compute_candidate_fingerprint(full) != compute_candidate_fingerprint(
            truncated
        )

    def test_changes_when_timestamp_changes(self):
        v = _basis(8, 1)
        original = [Candidate("a", 1.0, v)]
        moved = [Candidate("a", 1.5, v)]
        assert compute_candidate_fingerprint(original) != compute_candidate_fingerprint(
            moved
        )

    def test_empty_input_is_stable(self):
        assert compute_candidate_fingerprint([]) == compute_candidate_fingerprint([])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_end_is_empty(bad):
    inputs = SelectionInputs(scene_candidates=(_clip(100.0, 0),))

    assert select_scenes(inputs, 0.0, bad).navigation == ()
