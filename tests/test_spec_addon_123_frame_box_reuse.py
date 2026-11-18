"""SPEC-ADDON-123: a text box whose pixels have not changed since its text was
recognised takes that text again instead of being recognised a second time.

Item 1 steps 3 and 4, I3, I4, I5, I9. Every state of a scene is one read frame
(see ``tests.frame_box_doubles``); ``<name>#<k>`` in a stored line says the box's
text was recognised in state ``k``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from app.models import VideoFrameText  # noqa: F401 - registers the table
from tests.frame_box_doubles import (
    BoxEngine,
    DirectEngine,
    State,
    encode_scene,
    frame_texts,
    install,
)
from tests.ocr_fixtures import (
    SearchDB,
    build_search_db,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    seed_indexed_file,
)
from tests.test_spec_addon_133_frame_ocr_reread import wire_reader

pytestmark = pytest.mark.usefixtures("pillow_real")

MKV = "video/x-matroska"


class Run:
    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.db: SearchDB = build_search_db(tmp_path, monkeypatch)
        self.module, _source = wire_reader(self.db, monkeypatch)
        self.engine_lost = False
        monkeypatch.setattr(
            self.module, "_get_engine",
            lambda: None if self.engine_lost else DirectEngine(),
        )

    def scene(self, states: list[State], **flags) -> BoxEngine:
        path = encode_scene(self.tmp_path / "scene.mkv", states)
        seed_indexed_file(self.db, "vid1", MKV, path, file_type="video", **flags)
        self.path = path
        fake = BoxEngine(states)
        install(self.monkeypatch, fake)
        return fake

    def read(self):
        return self.module.index_frame_ocr("vid1")


@pytest.fixture()
def run(tmp_path, monkeypatch) -> Run:
    return Run(tmp_path, monkeypatch)


def lines(*names: str) -> str:
    return "\n".join(names)


def around(rect, by: int = 8):
    x0, y0, x1, y1 = rect
    return (max(0, x0 - by), max(0, y0 - by), x1 + by, y1 + by)


A = (20, 60, 110, 80)


# ---------------------------------------------------------------------------
# Item 1 step 3.2-3.6, I4, I9 - reuse of an unchanged box, two calls per frame
# ---------------------------------------------------------------------------


def test_spec_addon_123_i4_an_unchanged_box_takes_its_earlier_line_and_is_not_recognised_again(run):
    fake = run.scene([State(boxes=[("A", A)]) for _ in range(3)])

    run.read()

    assert frame_texts(run.db) == [
        lines("A#0", "tick#0"), lines("A#0", "tick#1"), lines("A#0", "tick#2"),
    ]
    assert fake.recognised() == {0: [["A", "tick"]], 1: [["tick"]], 2: [["tick"]]}


def test_spec_addon_123_i9_each_read_frame_makes_one_detection_and_one_recognition_call(run):
    fake = run.scene([State(boxes=[("A", A), ("B", (20, 100, 200, 120))]) for _ in range(4)])

    run.read()

    assert fake.detects() == [0, 1, 2, 3]
    assert {k: len(calls) for k, calls in fake.recognised().items()} == {0: 1, 1: 1, 2: 1, 3: 1}


def test_spec_addon_123_item_3_6_the_frame_text_is_its_lines_in_detection_order(run):
    first, second = ("Z", (20, 20, 120, 40)), ("Y", (20, 100, 120, 120))
    fake = run.scene([
        State(boxes=[second, first]),
        State(boxes=[first, second]),
    ])

    run.read()

    assert frame_texts(run.db) == [lines("Y#0", "Z#0", "tick#0"), lines("Z#0", "Y#0", "tick#1")]
    assert fake.recognised()[1] == [["tick"]]


# ---------------------------------------------------------------------------
# Item 1 step 3.3, I3 - the window mean, its 2.0 boundary and drift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param(129, "A#0", id="mean-1.0-is-unchanged"),
        pytest.param(130, "A#1", id="mean-2.0-is-changed"),
    ],
)
def test_spec_addon_123_i3_a_window_mean_of_two_or_more_is_recognised_again(run, value, expected):
    run.scene([
        State(boxes=[("A", A)]),
        State(boxes=[("A", A)], paint=[(around(A), value)]),
    ])

    run.read()

    assert frame_texts(run.db)[1] == lines(expected, "tick#1")


def test_spec_addon_123_i3_a_reused_box_is_compared_with_the_frame_it_was_recognised_in(run):
    """Each step moves the box by 1.0; the third state is 2.0 from the recognition frame."""
    fake = run.scene([
        State(boxes=[("A", A)]),
        State(boxes=[("A", A)], paint=[(around(A), 129)]),
        State(boxes=[("A", A)], paint=[(around(A), 130)]),
    ])

    run.read()

    assert frame_texts(run.db) == [
        lines("A#0", "tick#0"), lines("A#0", "tick#1"), lines("A#2", "tick#2"),
    ]
    assert fake.recognised()[2] == [["A", "tick"]]


LONG = (10, 20, 210, 40)
TALL = (400, 40, 420, 240)


@pytest.mark.parametrize(
    "box, change, value",
    [
        pytest.param(LONG, (105, 25, 115, 35), 188, id="horizontal-middle"),
        pytest.param(TALL, (405, 135, 415, 145), 188, id="vertical-middle"),
        pytest.param(LONG, (206, 20, 210, 40), 138, id="horizontal-last-cut-window"),
    ],
)
def test_spec_addon_123_i3_a_small_change_in_a_long_box_is_recognised_again(
    run, box, change, value,
):
    """Over the whole padded box each change averages under 2.0; within one window it does not."""
    fake = run.scene([
        State(boxes=[("A", box)]),
        State(boxes=[("A", box)], paint=[(change, value)]),
    ])

    run.read()

    assert frame_texts(run.db)[1] == lines("A#1", "tick#1")
    assert fake.recognised()[1] == [["A", "tick"]]


@pytest.mark.parametrize("box", [pytest.param(LONG, id="horizontal"), pytest.param(TALL, id="vertical")])
def test_spec_addon_123_i3_a_long_box_with_no_change_is_reused(run, box):
    run.scene([State(boxes=[("A", box)]) for _ in range(2)])

    run.read()

    assert frame_texts(run.db)[1] == lines("A#0", "tick#1")


# ---------------------------------------------------------------------------
# Item 1 step 3.2, I3 - matching by intersection over union
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shift, expected",
    [
        pytest.param(10, "A#0", id="iou-0.80-matches"),
        pytest.param(11, "A#1", id="iou-0.78-does-not"),
    ],
)
def test_spec_addon_123_i3_a_box_matches_at_an_iou_of_at_least_0_8(run, shift, expected):
    x0, y0, x1, y1 = A
    run.scene([
        State(boxes=[("A", A)]),
        State(boxes=[("A", (x0 + shift, y0, x1 + shift, y1))]),
    ])

    run.read()

    assert frame_texts(run.db)[1] == lines(expected, "tick#1")


@pytest.mark.parametrize(
    "order, expected",
    [
        pytest.param(("P1", "P2"), "P1#0", id="P1-first"),
        pytest.param(("P2", "P1"), "P2#0", id="P2-first"),
    ],
)
def test_spec_addon_123_i3_a_tie_goes_to_the_earlier_box_of_the_previous_frame(run, order, expected):
    rects = {"P1": (0, 100, 100, 120), "P2": (10, 100, 110, 120)}
    run.scene([
        State(boxes=[(name, rects[name]) for name in order]),
        State(boxes=[("C", (5, 100, 105, 120))]),
    ])

    run.read()

    assert frame_texts(run.db)[1] == lines(expected, "tick#1")


def test_spec_addon_123_i3_a_picked_box_is_taken_even_when_its_region_changed(run):
    """C1 picks P and fails the region test; C2 would match P but P is taken."""
    p = (0, 20, 100, 40)
    c1 = (5, 20, 105, 40)
    fake = run.scene([
        State(boxes=[("P", p)]),
        State(boxes=[("C1", c1), ("C2", p)], paint=[((106, 20, 109, 40), 255)]),
    ])

    run.read()

    assert frame_texts(run.db)[1] == lines("C1#1", "C2#1", "tick#1")
    assert fake.recognised()[1] == [["C1", "C2", "tick"]]


def test_spec_addon_123_i3_a_box_takes_the_highest_iou_not_the_first_candidate(run):
    run.scene([
        State(boxes=[("far", (14, 60, 104, 80)), ("near", (21, 60, 111, 80))]),
        State(boxes=[("A", A)]),
    ])

    run.read()

    assert frame_texts(run.db)[1] == lines("near#0", "tick#1")


# ---------------------------------------------------------------------------
# Item 1 step 3.4, I4 - a dropped box stays dropped
# ---------------------------------------------------------------------------


def test_spec_addon_123_i4_a_reused_box_the_engine_dropped_gives_no_line(run):
    fake = run.scene([
        State(boxes=[("D", A)], texts={"D": None}),
        State(boxes=[("D", A)]),
    ])

    run.read()

    assert frame_texts(run.db) == ["tick#0", "tick#1"]
    assert fake.recognised()[1] == [["tick"]]


# ---------------------------------------------------------------------------
# Item 1 step 4, I5 - a failed frame
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("failing", ["detect", "recognise"])
def test_spec_addon_123_i5_a_failed_frame_stores_no_text_and_offers_no_boxes(run, failing):
    fake = run.scene([State(boxes=[("A", A)]) for _ in range(3)])
    if failing == "detect":
        fake.detect_raises.add(1)
    else:
        fake.recognise_raises.add(1)

    run.read()

    assert [text for text in frame_texts(run.db) if text] == [
        lines("A#0", "tick#0"), lines("A#2", "tick#2"),
    ]
    assert fake.recognised()[2] == [["A", "tick"]]


@pytest.mark.parametrize("earlier", [False, True], ids=["first-read", "re-read"])
def test_spec_addon_123_item_4_an_engine_lost_between_the_two_calls_writes_nothing(run, earlier):
    flags = dict(frame_ocr_read=True, frame_ocr_indexed=True) if earlier else {}
    fake = run.scene([State(boxes=[("A", A)]) for _ in range(3)], **flags)
    if earlier:
        run.db.execute(
            "INSERT INTO video_frame_texts(file_id, chunk_index, start_time, end_time, text) "
            "VALUES('vid1', 0, 0.0, 9.0, 'EARLIER')"
        )

    def lose(k: int) -> None:
        if k == 1:
            run.engine_lost = True

    fake.on_detect = lose

    run.read()

    assert fake.detects() == [0, 1]
    assert 1 not in fake.recognised()
    assert frame_texts(run.db) == (["EARLIER"] if earlier else [])
    assert bool(run.db.flag("vid1", "frame_ocr_read")) is earlier
    assert bool(run.db.flag("vid1", "frame_ocr_indexed")) is earlier
    assert run.db.flag("vid1", "frame_ocr_rule") is None


# ---------------------------------------------------------------------------
# I9 - the pause between the two calls of one frame
# ---------------------------------------------------------------------------


def test_spec_addon_123_i9_no_engine_call_starts_while_paused_even_between_detection_and_recognition(run):
    fake = run.scene([State(boxes=[("A", A)]) for _ in range(2)])
    state = {"paused": True}

    def pause_after_detection(k: int) -> None:
        if k == 0:
            state["paused"] = True

    fake.on_detect = pause_after_detection
    result = {}

    def work():
        result["value"] = run.module._read_video(str(run.path), paused=lambda: state["paused"])

    thread = threading.Thread(target=work)
    thread.start()
    time.sleep(0.6)
    assert fake.log == []

    state["paused"] = False
    deadline = time.monotonic() + 10
    while not fake.detects() and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.6)
    assert fake.detects() == [0]
    assert fake.recognised() == {}

    state["paused"] = False
    fake.on_detect = None
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert fake.recognised()[0] == [["A", "tick"]]
