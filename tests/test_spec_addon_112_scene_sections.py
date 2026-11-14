"""SPEC-ADDON-112 (and SPEC-ADDON-113 item 4, SPEC-ADDON-031 I4/I5 as revised):
the representative-frame selection, read through ``GET /files/{id}/scenes`` over a
real search DB.

Each fixture is built so the expected sections can be worked out by hand: frames
every 10 s, basis vectors, and exact ties broken by time.
"""

from __future__ import annotations

import pytest

from tests.scene_sections_db import Video, basis, distinct_chars, mix
from tests.search_endpoint_db import DRIVE, api  # noqa: F401 - fixture
from tests.vec_partition_db import search_index  # noqa: F401 - fixture

EVERY_10S = [5.0 + 10 * k for k in range(60)]


def _scenes(api, file_id, params=None):
    response = api(f"/files/{file_id}/scenes", params or {}, drive=DRIVE)
    assert response.status_code == 200, response.text
    return response.json()


def _sections(body):
    return [(s["start"], s["end"], s.get("frame_time")) for s in body["scenes"]]


def _blocks(video, times, width):
    """Frame at ``t`` gets the basis vector of block ``floor(t / width)``."""
    for t in times:
        video.clip(t, basis(video.clip_dim, int(t // width)))
    return video


def _static(video, times=EVERY_10S):
    for t in times:
        video.clip(t, basis(video.clip_dim, 0))
    return video


def _distinct(video, times=EVERY_10S):
    for n, t in enumerate(times):
        video.clip(t, basis(video.clip_dim, n))
    return video


# Cut by time rank (every distance equal): 555, 525, ... 375, then coverage splits
# of [0, 375) at 345, 315, 285, 255, reaching min(ceil(1.5 * 8), 40) = 12 sections.
TWELVE_CUTS = [0.0, 255.0, 285.0, 315.0, 345.0, 375.0, 405.0, 435.0, 465.0, 495.0,
               525.0, 555.0, 600.0]


class TestSectionCount:
    def test_spec_addon_112_a_ten_minute_video_is_cut_into_eight_sections_at_its_changes(
        self, api, search_index,
    ):
        video = _blocks(Video(search_index, "blocks8"), EVERY_10S, 75.0)
        video.commit()

        body = _scenes(api, "blocks8")

        assert _sections(body) == [
            (0.0, 75.0, 5.0), (75.0, 155.0, 85.0), (155.0, 225.0, 165.0),
            (225.0, 305.0, 235.0), (305.0, 375.0, 315.0), (375.0, 455.0, 385.0),
            (455.0, 525.0, 465.0), (525.0, 600.0, 535.0),
        ]
        assert body["candidate_count"] == 60

    def test_spec_addon_112_coverage_splits_the_long_section_until_the_section_limit(
        self, api, search_index,
    ):
        _distinct(Video(search_index, "distinct12")).commit()

        body = _scenes(api, "distinct12")

        starts = TWELVE_CUTS[:-1]
        assert _sections(body) == [
            (start, end, 5.0 if start == 0.0 else start + 10.0)
            for start, end in zip(starts, TWELVE_CUTS[1:])
        ]

    def test_spec_addon_112_two_hours_give_forty_sections_and_never_more(
        self, api, search_index,
    ):
        video = Video(search_index, "twohours", duration=7200.0)
        _blocks(video, [6.0 + 12 * k for k in range(600)], 144.0).commit()

        sections = _sections(_scenes(api, "twohours"))

        assert len(sections) == 40
        assert sections[0] == (0.0, 1158.0, 6.0)
        for (start, end, frame), expected_start in zip(
            sections[1:-1], [1158.0 + 144 * j for j in range(38)],
        ):
            assert (start, end, frame) == (expected_start, expected_start + 144, start + 12)
        assert sections[-1] == (6630.0, 7200.0, 6774.0)

    def test_spec_addon_112_the_count_is_capped_by_the_pool_and_a_lone_frame_spans_all(
        self, api, search_index,
    ):
        video = Video(search_index, "three")
        for n, t in enumerate([100.0, 300.0, 500.0]):
            video.clip(t, basis(video.clip_dim, n))
        video.commit()
        Video(search_index, "one").clip(100.0, basis(video.clip_dim, 0)).commit()

        assert _sections(_scenes(api, "three")) == [
            (0.0, 300.0, 100.0), (300.0, 500.0, 300.0), (500.0, 600.0, 500.0),
        ]
        assert _sections(_scenes(api, "one")) == [(0.0, 600.0, 100.0)]

    def test_spec_addon_112_unknown_duration_ends_the_span_one_second_after_the_last_frame(
        self, api, search_index,
    ):
        video = Video(search_index, "nodur", duration=None)
        for n, t in enumerate([100.0, 300.0, 500.0]):
            video.clip(t, basis(video.clip_dim, n))
        video.commit()

        assert _sections(_scenes(api, "nodur")) == [(0.0, 300.0, 100.0), (300.0, 501.0, 500.0)]


class TestMerging:
    def test_spec_addon_112_a_video_that_never_changes_is_one_section(
        self, api, search_index,
    ):
        _static(Video(search_index, "static")).commit()

        assert _sections(_scenes(api, "static")) == [(0.0, 600.0, 5.0)]

    @pytest.mark.parametrize(
        "cos, expected",
        [
            pytest.param(0.95, [(0.0, 600.0, 5.0)], id="cos-0.95-merged"),
            pytest.param(0.90, [(0.0, 305.0, 5.0), (305.0, 600.0, 315.0)], id="cos-0.90-kept"),
        ],
    )
    def test_spec_addon_112_neighbours_merge_only_at_cosine_0_93(
        self, api, search_index, cos, expected,
    ):
        video = Video(search_index, "halves")
        for t in EVERY_10S:
            video.clip(t, basis(video.clip_dim, 0) if t < 300 else mix(video.clip_dim, cos))
        video.commit()

        sections = _sections(_scenes(api, "halves"))

        assert sections == expected


def _two_texts(video, first, second):
    video.ocr(0.0, 300.0, first)
    if second is not None:
        video.ocr(300.0, 600.0, second)
    return video


class TestTextual:
    @pytest.mark.parametrize(
        "first, second, expected",
        [
            pytest.param(
                distinct_chars(0, 251), distinct_chars(25, 251),
                [(0.0, 315.0, 5.0), (315.0, 600.0, 565.0)],
                id="textual-jaccard-0.82-kept-apart",
            ),
            pytest.param(
                distinct_chars(0, 251), distinct_chars(10, 251),
                [(0.0, 600.0, 5.0)],
                id="textual-jaccard-0.92-merged",
            ),
            pytest.param(
                distinct_chars(0, 200), distinct_chars(25, 200),
                [(0.0, 315.0, 5.0), (315.0, 600.0, 565.0)],
                id="length-200-is-textual",
            ),
            pytest.param(
                distinct_chars(0, 199), distinct_chars(25, 199),
                [(0.0, 600.0, 5.0)],
                id="length-199-falls-back-to-cosine",
            ),
            pytest.param(
                distinct_chars(0, 251), None,
                [(0.0, 315.0, 5.0), (315.0, 600.0, 565.0)],
                id="textual-and-not-textual-never-same",
            ),
        ],
    )
    def test_spec_addon_112_identical_pictures_are_told_apart_by_their_text(
        self, api, search_index, first, second, expected,
    ):
        _two_texts(_static(Video(search_index, "slides")), first, second).commit()

        assert _sections(_scenes(api, "slides")) == expected

    def test_spec_addon_112_different_pictures_with_the_same_text_merge(
        self, api, search_index,
    ):
        video = Video(search_index, "sametext")
        for t in EVERY_10S:
            video.clip(t, basis(video.clip_dim, 0 if t < 300 else 1))
        body = distinct_chars(0, 251)
        _two_texts(video, body, body).commit()

        assert _sections(_scenes(api, "sametext")) == [(0.0, 600.0, 5.0)]

    def test_spec_addon_112_the_longest_chunk_sets_the_length(self, api, search_index):
        video = _static(Video(search_index, "chunks"))
        video.ocr(0.0, 300.0, distinct_chars(0, 150))
        video.ocr(0.0, 300.0, distinct_chars(150, 120))
        video.ocr(300.0, 600.0, distinct_chars(500, 150))
        video.ocr(300.0, 600.0, distinct_chars(650, 120))
        video.commit()

        assert _sections(_scenes(api, "chunks")) == [(0.0, 600.0, 5.0)]


def _topic(video, times, change_at):
    for t in times:
        video.whisper(t, basis(video.text_dim, 0 if t < change_at else 1))
    return video


class TestTopic:
    def test_spec_addon_112_a_topic_change_cuts_a_section_that_is_never_merged(
        self, api, search_index,
    ):
        video = _static(Video(search_index, "topic7"))
        _topic(video, [262.0, 282.0, 302.0, 322.0, 342.0, 362.0, 382.0], 322.0).commit()

        assert _sections(_scenes(api, "topic7")) == [
            (0.0, 322.0, 5.0), (322.0, 600.0, 325.0),
        ]

    def test_spec_addon_112_six_transcript_embeddings_give_no_topic_boundary(
        self, api, search_index,
    ):
        video = _static(Video(search_index, "topic6"))
        _topic(video, [262.0, 282.0, 302.0, 322.0, 342.0, 362.0], 322.0).commit()

        assert _sections(_scenes(api, "topic6")) == [(0.0, 600.0, 5.0)]

    def test_spec_addon_112_a_topic_change_near_a_strong_visual_boundary_joins_it(
        self, api, search_index,
    ):
        video = _static(Video(search_index, "joined"))
        _topic(video, [437.0, 457.0, 477.0, 497.0, 517.0, 537.0, 557.0], 497.0).commit()

        assert _sections(_scenes(api, "joined")) == [
            (0.0, 495.0, 5.0), (495.0, 600.0, 565.0),
        ]


SHIFTED = [5.5 + 10 * k for k in range(60)]


def _steady_video(search_index, **scan):
    video = _blocks(Video(search_index, "steady"), SHIFTED, 75.0)
    if scan:
        video.scan(**scan)
    video.commit()
    return video


class TestSteadySecond:
    @pytest.mark.parametrize(
        "scan, frame_time",
        [
            pytest.param({}, 5.5, id="no-scan-own-time"),
            pytest.param({"seconds": 600}, 6.0, id="steady-at-ceil"),
            pytest.param({"seconds": 600, "hashes": {6: 23}}, 6.0, id="23-bits-steady"),
            pytest.param({"seconds": 600, "hashes": {6: 24}}, 7.0, id="24-bits-moves-on"),
            pytest.param({"seconds": 600, "sizes": {6: 451}}, 6.0, id="size-above-floor"),
            pytest.param({"seconds": 600, "sizes": {6: 449}}, 7.0, id="size-below-floor"),
            pytest.param(
                {"seconds": 600, "sizes": {6: 0, 7: 0, 8: 0}}, 16.0,
                id="best-has-none-next-best-taken",
            ),
            pytest.param(
                {"seconds": 600, "sizes": {s: 0 for s in
                                           (6, 7, 8, 16, 17, 18, 26, 27, 28, 36, 37, 38)}},
                5.5, id="four-best-have-none-own-time",
            ),
            pytest.param({"seconds": 6}, 5.5, id="second-outside-the-scan"),
            pytest.param({"seconds": 7}, 5.5, id="next-second-outside-the-scan"),
            pytest.param({"seconds": 8}, 6.0, id="second-and-next-in-the-scan"),
            pytest.param({"seconds": 600, "version": 2}, 5.5, id="other-version-is-no-scan"),
            pytest.param(
                {"seconds": 600, "hash_bytes": b"\0" * (600 * 32 - 1)}, 5.5,
                id="short-hash-blob-is-no-scan",
            ),
            pytest.param(
                {"seconds": 600, "size_bytes": b"\0" * (600 * 2 + 2)}, 5.5,
                id="long-size-blob-is-no-scan",
            ),
        ],
    )
    def test_spec_addon_112_the_frame_sits_on_a_steady_second(
        self, api, search_index, scan, frame_time,
    ):
        _steady_video(search_index, **scan)

        first = _scenes(api, "steady")["scenes"][0]

        assert (first["start"], first["end"]) == (0.0, 75.5)
        assert first.get("frame_time") == frame_time


class TestPool:
    def test_spec_addon_112_change_frames_alone_make_a_pool(self, api, search_index):
        video = Video(search_index, "changes")
        for n, t in enumerate([100.0, 300.0, 500.0]):
            video.change(t, basis(video.clip_dim, n))
        video.commit()

        body = _scenes(api, "changes")

        assert body["candidate_count"] == 3
        assert _sections(body) == [
            (0.0, 300.0, 100.0), (300.0, 500.0, 300.0), (500.0, 600.0, 500.0),
        ]

    def test_spec_addon_112_change_frames_within_3_s_of_a_scene_candidate_are_left_out(
        self, api, search_index,
    ):
        video = Video(search_index, "mixed")
        for n, t in enumerate([0.0, 100.0, 200.0, 300.0, 400.0, 500.0]):
            video.clip(t, basis(video.clip_dim, n))
        video.change(50.0, basis(video.clip_dim, 10))
        video.change(103.0, basis(video.clip_dim, 11))
        video.change(198.0, basis(video.clip_dim, 12))
        video.change(304.0, basis(video.clip_dim, 13))
        video.change(450.0, None)
        video.commit()

        assert _scenes(api, "mixed")["candidate_count"] == 8


class TestWindowAndOrder:
    def test_spec_addon_031_i5_a_window_cuts_its_own_span_from_its_own_pool(
        self, api, search_index,
    ):
        """SPEC-ADDON-031 I4/I5 as revised by SPEC-ADDON-112."""
        video = Video(search_index, "window")
        for n, t in enumerate([100.0, 300.0, 500.0]):
            video.clip(t, basis(video.clip_dim, n))
        video.commit()

        body = _scenes(api, "window", {"start": 200})

        assert body["candidate_count"] == 2
        assert _sections(body) == [(200.0, 500.0, 300.0), (500.0, 600.0, 500.0)]

    def test_spec_addon_112_i2_row_order_does_not_change_the_result(
        self, api, search_index,
    ):
        _distinct(Video(search_index, "ordered")).commit()
        _distinct(Video(search_index, "reversed")).commit(reverse=True)

        ordered = _sections(_scenes(api, "ordered"))

        assert ordered == _sections(_scenes(api, "reversed"))
        assert [start for start, _, _ in ordered] == TWELVE_CUTS[:-1]

    def test_spec_addon_113_the_excerpt_is_taken_around_the_frame_time(
        self, api, search_index,
    ):
        video = _blocks(Video(search_index, "excerpt"), EVERY_10S, 75.0)
        video.transcript_chunk(68.0, 70.0, "near-the-section-start")
        video.transcript_chunk(88.0, 90.0, "near-the-frame")
        video.commit()

        second = _scenes(api, "excerpt")["scenes"][1]

        assert (second["start"], second.get("frame_time")) == (75.0, 85.0)
        assert "near-the-frame" in (second["transcript_excerpt"] or "")
        assert "near-the-section-start" not in (second["transcript_excerpt"] or "")
