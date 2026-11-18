"""SPEC-ADDON-134: a stored frame read keeps each segment's lines with their quads
in ``video_frame_lines``, and writes ``frame_ocr_rule`` 3.

Every state of a scene is one read frame (``tests.frame_box_doubles``). The frame
is 480x400 and the engine is faked at ``detect_boxes`` / ``recognise_boxes``.
"""

from __future__ import annotations

import pytest

from app.extractors.base import ContentExtractor
from app.models import VideoFrameText  # noqa: F401 - registers the table
from tests.frame_box_doubles import (
    HEIGHT,
    TICKER,
    WIDTH,
    BoxEngine,
    DirectEngine,
    State,
    encode_scene,
    frame_texts,
    install,
    quad,
)
from tests.frame_line_doubles import (
    ShapedEngine,
    chunk_rows,
    decoded,
    line_rows,
    norm,
    numbers_in,
    quads_of,
    seed_line_row,
    texts_of,
)
from tests.ocr_fixtures import (
    SearchDB,
    build_search_db,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    seed_indexed_file,
)
from tests.test_frame_page_ocr_readers import media  # noqa: F401 - fixture
from tests.test_spec_addon_133_frame_ocr_reread import wire_reader

pytestmark = pytest.mark.usefixtures("pillow_real")

MKV = "video/x-matroska"
A = (20, 60, 110, 80)
B = (20, 100, 200, 120)


class Run:
    def __init__(self, tmp_path, monkeypatch) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.db: SearchDB = build_search_db(tmp_path, monkeypatch)
        self.module, _source = wire_reader(self.db, monkeypatch)
        monkeypatch.setattr(self.module, "_get_engine", lambda: DirectEngine())

    def scene(self, count: int, fake, *, paint=None, **flags):
        states = [State(paint=list((paint or {}).get(k, []))) for k in range(count)]
        path = encode_scene(self.tmp_path / "scene.mkv", states)
        seed_indexed_file(self.db, "vid1", MKV, path, file_type="video", **flags)
        install(self.monkeypatch, fake)
        return fake

    def box_scene(self, states: list[State], **flags) -> BoxEngine:
        path = encode_scene(self.tmp_path / "scene.mkv", states)
        seed_indexed_file(self.db, "vid1", MKV, path, file_type="video", **flags)
        fake = BoxEngine(states)
        install(self.monkeypatch, fake)
        return fake

    def read(self):
        return self.module.index_frame_ocr("vid1")

    def rule(self):
        return self.db.flag("vid1", "frame_ocr_rule")


@pytest.fixture()
def run(tmp_path, monkeypatch) -> Run:
    return Run(tmp_path, monkeypatch)


def rect_points(rect) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in quad(rect)]


TICK_QUAD = norm(rect_points(TICKER), WIDTH, HEIGHT)


# ---------------------------------------------------------------------------
# Item 1.2, 1.4, I1, I5 - one row per segment, lines in join order
# ---------------------------------------------------------------------------


def test_spec_addon_134_i1_each_segment_gets_one_row_matching_its_chunk(run):
    run.box_scene([State(boxes=[("A", A), ("B", B)]) for _ in range(3)])

    run.read()

    chunks = chunk_rows(run.db)
    assert [c[3] for c in chunks] == [
        "A#0\nB#0\ntick#0", "A#0\nB#0\ntick#1", "A#0\nB#0\ntick#2",
    ]
    rows = line_rows(run.db)
    assert [row[0] for row in rows] == [0, 1, 2]
    assert [(row[1], row[2]) for row in rows] == [(c[1], c[2]) for c in chunks]
    assert ["\n".join(texts_of(row[3])) for row in rows] == [c[3] for c in chunks]
    assert run.rule() == 3


def test_spec_addon_134_i5_lines_json_holds_one_object_per_line_with_text_and_eight_floats(run):
    fake = ShapedEngine([[("A", rect_points(A))]], [{"A": "日本語の行"}])
    run.scene(1, fake)

    run.read()

    [row] = line_rows(run.db)
    lines = decoded(row[3])
    assert [line["text"] for line in lines] == ["日本語の行", "tick#0"]
    for line in lines:
        assert set(line) == {"text", "quad"}
        assert isinstance(line["text"], str)
        assert len(line["quad"]) == 8
        assert all(type(v) is float for v in line["quad"])


def test_spec_addon_134_item_1_2_lines_follow_the_order_they_are_joined_in(run):
    first, second = ("Z", (20, 20, 120, 40)), ("Y", (20, 100, 120, 120))
    run.box_scene([State(boxes=[second, first]), State(boxes=[first, second])])

    run.read()

    rows = line_rows(run.db)
    assert [texts_of(row[3]) for row in rows] == [
        ["Y#0", "Z#0", "tick#0"], ["Z#0", "Y#0", "tick#1"],
    ]
    assert quads_of(rows[0][3])["Y#0"] == norm(rect_points(second[1]), WIDTH, HEIGHT)
    assert quads_of(rows[1][3])["Y#0"] == norm(rect_points(second[1]), WIDTH, HEIGHT)


def test_spec_addon_134_i1_a_segment_split_into_several_chunks_still_has_one_row(run):
    long_line = " ".join(f"word{i:04d}" for i in range(260))
    fake = ShapedEngine([[("A", rect_points(A))]], [{"A": long_line}])
    run.scene(1, fake)

    run.read()

    chunks = chunk_rows(run.db)
    assert len(chunks) > 1
    assert {(c[1], c[2]) for c in chunks} == {(chunks[0][1], chunks[0][2])}
    [row] = line_rows(run.db)
    assert (row[0], row[1], row[2]) == (0, chunks[0][1], chunks[0][2])
    assert "\n".join(texts_of(row[3])) == long_line + "\ntick#0"
    assert ContentExtractor.chunk_text("\n".join(texts_of(row[3]))) == [c[3] for c in chunks]


# ---------------------------------------------------------------------------
# Item 1.2, I2 - the quad
# ---------------------------------------------------------------------------


def test_spec_addon_134_i2_a_quad_is_the_box_over_the_engine_image_rounded_to_four_places(run):
    fake = ShapedEngine([[("A", rect_points(A))]])
    run.scene(1, fake)

    run.read()

    assert fake.seen_shape[0] == (400, 480)
    [row] = line_rows(run.db)
    assert quads_of(row[3])["A#0"] == [0.0417, 0.15, 0.2292, 0.15, 0.2292, 0.2, 0.0417, 0.2]
    assert quads_of(row[3])["tick#0"] == TICK_QUAD


@pytest.mark.parametrize(
    "points, expected",
    [
        pytest.param(
            [[110, 80], [20, 80], [20, 60], [110, 60]],
            [0.2292, 0.2, 0.0417, 0.2, 0.0417, 0.15, 0.2292, 0.15],
            id="starting-bottom-right",
        ),
        pytest.param(
            [[23, 61], [113, 63], [109, 83], [19, 81]],
            [0.0479, 0.1525, 0.2354, 0.1575, 0.2271, 0.2075, 0.0396, 0.2025],
            id="skewed",
        ),
        pytest.param(
            [[-6, 390], [490, 390], [490, 405], [-6, 405]],
            [0.0, 0.975, 1.0, 0.975, 1.0, 1.0, 0.0, 1.0],
            id="past-both-edges-is-clamped",
        ),
    ],
)
def test_spec_addon_134_i2_a_quad_keeps_the_engine_corner_order_and_is_clamped(run, points, expected):
    fake = ShapedEngine([[("A", points)]])
    run.scene(1, fake)

    run.read()

    assert frame_texts(run.db) == ["A#0\ntick#0"]
    [row] = line_rows(run.db)
    assert quads_of(row[3])["A#0"] == expected


def test_spec_addon_134_i2_a_box_the_engine_dropped_adds_no_line(run):
    fake = ShapedEngine(
        [[("D", rect_points(A)), ("B", rect_points(B))]], [{"D": None}],
    )
    run.scene(1, fake)

    run.read()

    [row] = line_rows(run.db)
    assert texts_of(row[3]) == ["B#0", "tick#0"]
    assert len(decoded(row[3])) == 2


def test_spec_addon_134_i2_item_3_a_reused_box_gives_its_earlier_line_and_this_frames_quad(run):
    x0, y0, x1, y1 = A
    moved = (x0 + 10, y0, x1 + 10, y1)
    fake = run.box_scene([State(boxes=[("A", A)]), State(boxes=[("A", moved)])])

    run.read()

    assert fake.recognised()[1] == [["tick"]]
    rows = line_rows(run.db)
    assert [texts_of(row[3]) for row in rows] == [["A#0", "tick#0"], ["A#0", "tick#1"]]
    assert quads_of(rows[0][3])["A#0"] == norm(rect_points(A), WIDTH, HEIGHT)
    assert quads_of(rows[1][3])["A#0"] == norm(rect_points(moved), WIDTH, HEIGHT)


def test_spec_addon_134_item_1_2_a_frame_continuing_the_segment_leaves_its_lines(run):
    moved = [[30, 60], [120, 60], [120, 80], [30, 80]]
    fake = ShapedEngine(
        [[("A", rect_points(A))], [("A", moved)], [("A", moved)]],
        [{"A": "SAME", "tick": "T"}] * 3,
    )
    run.scene(3, fake)

    run.read()

    assert frame_texts(run.db) == ["SAME\nT"]
    [row] = line_rows(run.db)
    assert row[0] == 0
    assert quads_of(row[3])["SAME"] == norm(rect_points(A), WIDTH, HEIGHT)


# ---------------------------------------------------------------------------
# Item 1.4, §2.1 - segment numbering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("failing", ["detect", "recognise"])
def test_spec_addon_134_failure_1_a_failed_frame_opens_no_segment_and_takes_no_number(run, failing):
    fake = run.box_scene([State(boxes=[("A", A)]) for _ in range(3)])
    if failing == "detect":
        fake.detect_raises.add(1)
    else:
        fake.recognise_raises.add(1)

    run.read()

    rows = line_rows(run.db)
    assert [row[0] for row in rows] == [0, 1]
    assert [texts_of(row[3]) for row in rows] == [["A#0", "tick#0"], ["A#2", "tick#2"]]


# ---------------------------------------------------------------------------
# §2.2, I4 - unusable quads
# ---------------------------------------------------------------------------


THREE_POINTS = [[20, 60], [110, 60], [110, 80]]
FIVE_POINTS = [[20, 60], [65, 58], [110, 60], [110, 80], [20, 80]]


def _unusable_scene(bad: dict[str, list]) -> ShapedEngine:
    middle = [("A", bad.get("A", rect_points(A))), ("B", bad.get("B", rect_points(B)))]
    return ShapedEngine(
        [
            [("A", rect_points(A)), ("B", rect_points(B))],
            middle,
            [("A", rect_points(A)), ("B", rect_points(B))],
        ],
        [{}, {"A": "MID-A", "B": "MID-B"}, {}],
    )


def around(rect, by: int = 8):
    x0, y0, x1, y1 = rect
    return (x0 - by, y0 - by, x1 + by, y1 + by)


MIDDLE_PAINT = {1: [(around(A), 200), (around(B), 200)]}


def _outcome(run: Run) -> dict:
    return {
        "frames": chunk_rows(run.db),
        "fts": sorted(row[2] for row in run.db.ocr_fts("vid1")),
        "embeddings": len(run.db.ocr_embeddings("vid1")),
        "flags": (
            bool(run.db.flag("vid1", "frame_ocr_read")),
            bool(run.db.flag("vid1", "frame_ocr_indexed")),
            run.rule(),
        ),
    }


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"A": THREE_POINTS}, id="three-points"),
        pytest.param({"A": FIVE_POINTS}, id="five-points"),
        pytest.param({"A": THREE_POINTS, "B": FIVE_POINTS}, id="two-bad-lines-one-segment"),
    ],
)
def test_spec_addon_134_i4_a_segment_with_an_unusable_quad_gets_no_row_and_one_warning(
    tmp_path, monkeypatch, caplog, bad,
):
    control = Run(tmp_path / "control", monkeypatch)
    control.scene(3, _unusable_scene({}), paint=MIDDLE_PAINT)
    with caplog.at_level("WARNING"):
        control.read()
    expected = _outcome(control)
    control_warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert [row[0] for row in line_rows(control.db)] == [0, 1, 2]
    caplog.clear()

    broken = Run(tmp_path / "broken", monkeypatch)
    broken.scene(3, _unusable_scene(bad), paint=MIDDLE_PAINT)
    with caplog.at_level("WARNING"):
        broken.read()
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]

    assert _outcome(broken) == expected
    assert expected["flags"] == (True, True, 3)
    assert [row[0] for row in line_rows(broken.db)] == [0, 2]
    extra = [w for w in warnings if w not in control_warnings]
    assert len(warnings) == len(control_warnings) + 1, warnings
    [message] = extra
    middle_start = [c for c in chunk_rows(broken.db) if "MID-A" in c[3]][0][1]
    assert "vid1" in message
    assert any(abs(n - middle_start) < 0.05 for n in numbers_in(message)), (message, middle_start)


NAN_QUAD = [[20, 60], [110, float("nan")], [110, 80], [20, 80]]
INF_QUAD = [[20, 60], [float("inf"), 60], [110, 80], [20, 80]]


@pytest.mark.parametrize(
    "points",
    [pytest.param(NAN_QUAD, id="nan"), pytest.param(INF_QUAD, id="inf")],
)
def test_spec_addon_134_failure_2_a_non_finite_box_fails_its_frame_with_no_text_and_no_row(
    run, points,
):
    run.scene(3, _unusable_scene({"A": points}), paint=MIDDLE_PAINT)

    run.read()

    assert [c[3] for c in chunk_rows(run.db)] == ["A#0\nB#0\ntick#0", "A#2\nB#2\ntick#2"]
    rows = line_rows(run.db)
    assert [row[0] for row in rows] == [0, 1]
    assert [texts_of(row[3]) for row in rows] == [
        ["A#0", "B#0", "tick#0"], ["A#2", "B#2", "tick#2"],
    ]


# ---------------------------------------------------------------------------
# Item 1.4 - the store deletes the file's earlier rows
# ---------------------------------------------------------------------------


def test_spec_addon_134_i1_a_store_replaces_every_earlier_row_of_the_file(run):
    run.box_scene([State(boxes=[("A", A)]) for _ in range(2)])
    seed_line_row(run.db, segment_index=0, lines_json="STALE")
    seed_line_row(run.db, segment_index=7, lines_json="STALE")
    seed_line_row(run.db, "other", segment_index=0, lines_json="OTHER FILE")

    run.read()

    rows = line_rows(run.db)
    assert [row[0] for row in rows] == [0, 1]
    assert all(row[3] != "STALE" for row in rows)
    assert [row[3] for row in line_rows(run.db, "other")] == ["OTHER FILE"]


def test_spec_addon_134_i1_an_empty_read_leaves_no_row(run):
    fake = ShapedEngine([[], [], []], [{"tick": None}] * 3)
    run.scene(3, fake)
    seed_line_row(run.db, segment_index=0, lines_json="STALE")

    run.read()

    assert frame_texts(run.db) == []
    assert line_rows(run.db) == []
    assert bool(run.db.flag("vid1", "frame_ocr_read")) is True


def test_spec_addon_134_item_1_4_an_unreadable_first_read_deletes_the_rows_left_behind(
    tmp_path, monkeypatch, media,  # noqa: F811
):
    run = Run(tmp_path, monkeypatch)
    seed_indexed_file(run.db, "vid1", "video/mp4", media["corrupt"], file_type="video")
    install(monkeypatch, ShapedEngine([[]]))
    seed_line_row(run.db, segment_index=0, lines_json="STALE")

    run.read()

    assert line_rows(run.db) == []
    assert frame_texts(run.db) == []


# ---------------------------------------------------------------------------
# Item 1.6, §3 - a read below rule 3 is read again and stored with its lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", [None, 1, 2])
def test_spec_addon_134_item_1_6_a_read_below_rule_three_is_read_again_with_its_lines(run, rule):
    flags = dict(frame_ocr_read=True, frame_ocr_indexed=True)
    if rule is not None:
        flags["frame_ocr_rule"] = rule
    fake = run.box_scene([State(boxes=[("A", A)]) for _ in range(2)], **flags)
    run.db.execute(
        "INSERT INTO video_frame_texts(file_id, chunk_index, start_time, end_time, text) "
        "VALUES('vid1', 0, 0.0, 6.0, 'EARLIER')"
    )

    run.read()

    assert fake.detects() == [0, 1]
    assert frame_texts(run.db) == ["A#0\ntick#0", "A#0\ntick#1"]
    assert [texts_of(row[3]) for row in line_rows(run.db)] == [
        ["A#0", "tick#0"], ["A#0", "tick#1"],
    ]
    assert run.rule() == 3
