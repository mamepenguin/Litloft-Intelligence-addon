"""SPEC-ADDON-136 items 1-3, I3: ``video_frame_lines`` gains ``frame_width`` and
``frame_height``, and the frame reader's store step writes the size of the image
the engine read for each segment's opening frame.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from tests.frame_box_doubles import HEIGHT, WIDTH, SimpleBoxEngine, State, install
from tests.loft_media_world import LoftWorld, fetch
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.test_frame_page_ocr_schema import init_database  # noqa: F401 - fixture
from tests.test_spec_addon_119_loft_fetch_counting import (  # noqa: F401 - fixtures
    VIDEO_SECONDS,
    fresh_worker_state,
    gradient,
)
from tests.test_spec_addon_134_frame_lines_store import A, B, Run
from tests.vec_partition_db import search_index  # noqa: F401 - fixture
from tests.youtube_frame_doubles import needs_ffmpeg, youtube  # noqa: F401 - fixtures

pytestmark = pytest.mark.usefixtures("pillow_real")

OLD_TABLE = (
    "CREATE TABLE video_frame_lines ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "file_id TEXT NOT NULL, "
    "segment_index INTEGER NOT NULL, "
    "start_time REAL NOT NULL, "
    "end_time REAL NOT NULL, "
    "lines_json TEXT NOT NULL)"
)


def _size_columns(engine) -> dict[str, tuple[str, int]]:
    with engine.connect() as conn:
        return {
            row[1]: (str(row[2]).upper(), row[3])
            for row in conn.execute(text("PRAGMA table_info(video_frame_lines)")).fetchall()
            if row[1] in ("frame_width", "frame_height")
        }


def _sizes(engine) -> list[tuple]:
    with engine.connect() as conn:
        return [
            tuple(row) for row in conn.execute(text(
                "SELECT segment_index, frame_width, frame_height FROM video_frame_lines "
                "ORDER BY segment_index"
            )).fetchall()
        ]


# ---------------------------------------------------------------------------
# Item 1, I3 - the migration
# ---------------------------------------------------------------------------


def test_spec_addon_136_item_1_the_migration_adds_both_columns_null_and_can_run_twice(
    init_database,  # noqa: F811
):
    start, path = init_database
    engine = create_engine(f"sqlite:///{path}")
    from app.database import Base

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS video_frame_lines"))
        conn.execute(text(OLD_TABLE))
        conn.execute(text(
            "INSERT INTO video_frame_lines(file_id, segment_index, start_time, end_time, lines_json) "
            "VALUES('vid1', 0, 0.0, 3.0, '[]'), ('vid1', 1, 3.0, 6.0, '[]')"
        ))
    engine.dispose()

    start()

    reader = create_engine(f"sqlite:///{path}")
    columns = _size_columns(reader)
    assert set(columns) == {"frame_width", "frame_height"}
    for declared, notnull in columns.values():
        assert "INT" in declared
        assert notnull == 0
    assert _sizes(reader) == [(0, None, None), (1, None, None)]
    with reader.begin() as conn:
        conn.execute(text(
            "UPDATE video_frame_lines SET frame_width = 1280, frame_height = 720 "
            "WHERE segment_index = 1"
        ))
    reader.dispose()

    start()

    reader = create_engine(f"sqlite:///{path}")
    assert _sizes(reader) == [(0, None, None), (1, 1280, 720)]
    reader.dispose()


# ---------------------------------------------------------------------------
# Item 2, I3 - the store step writes the engine image's size
# ---------------------------------------------------------------------------


@pytest.fixture()
def run(tmp_path, monkeypatch) -> Run:
    return Run(tmp_path, monkeypatch)


def test_spec_addon_136_i3_each_stored_row_has_the_engine_images_width_and_height(run):
    fake = run.box_scene([
        State(boxes=[("A", A)]),
        State(boxes=[("A", A), ("B", B)]),
        State(boxes=[("B", B)]),
    ])

    run.read()

    rows = [
        tuple(row) for row in run.db.rows(
            "SELECT segment_index, frame_width, frame_height FROM video_frame_lines "
            "WHERE file_id = 'vid1' ORDER BY segment_index"
        )
    ]
    assert [row[0] for row in rows] == [0, 1, 2]
    assert [row[1:] for row in rows] == [(WIDTH, HEIGHT)] * 3
    for k in range(3):
        assert fake.seen_grey[k].shape == (HEIGHT, WIDTH)


class _ShapeRecordingEngine(SimpleBoxEngine):
    def __init__(self) -> None:
        super().__init__()
        self.shapes: set[tuple[int, int]] = set()

    def detect_boxes(self, engine, image):
        from tests.frame_box_doubles import grey_of

        height, width = grey_of(image).shape
        self.shapes.add((width, height))
        return super().detect_boxes(engine, image)


@pytest.fixture()
def loft_world(search_index, youtube, monkeypatch, tmp_path, make_settings):  # noqa: F811
    return LoftWorld(search_index, youtube, monkeypatch, tmp_path, make_settings)


@needs_ffmpeg
def test_spec_addon_136_i3_a_loft_read_stores_the_size_of_the_image_the_engine_read(
    loft_world, gradient, monkeypatch,  # noqa: F811
):
    loft_world.configure(frame_ocr="on_index")
    fake = _ShapeRecordingEngine()
    install(monkeypatch, fake)
    loft_world.serve(gradient, duration=VIDEO_SECONDS)
    loft_world.add_loft("ytloft000001", frame_scan_indexed=True,
                        frame_ocr_read=False, frame_ocr_indexed=False)

    fetch("ytloft000001")

    with loft_world.index.engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT frame_width, frame_height FROM video_frame_lines "
            "WHERE file_id = 'ytloft000001'"
        )).fetchall()
    assert rows
    assert len(fake.shapes) == 1, fake.shapes
    assert {tuple(row) for row in rows} == fake.shapes
