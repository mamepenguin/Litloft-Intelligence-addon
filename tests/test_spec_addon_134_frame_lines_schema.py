"""SPEC-ADDON-134 §4, item 1.6, item 1.7, I7: the ``video_frame_lines`` table and
its migration, its purge, rule 3 as the current rule, and a ``.loft`` read.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from tests.frame_box_doubles import SimpleBoxEngine, install
from tests.frame_line_doubles import texts_of
from tests.loft_media_world import LoftWorld, fetch
from tests.ocr_fixtures import (
    SearchDB,
    build_search_db,
    patch_features,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    seed_indexed_file,
)
from tests.test_frame_page_ocr_schema import init_database  # noqa: F401 - fixture
from tests.test_spec_addon_119_loft_fetch_counting import (  # noqa: F401 - fixtures
    VIDEO_SECONDS,
    fresh_worker_state,
    gradient,
)
from tests.vec_partition_db import search_index  # noqa: F401 - fixture
from tests.youtube_frame_doubles import needs_ffmpeg, youtube  # noqa: F401 - fixtures

pytestmark = pytest.mark.usefixtures("pillow_real")

LOFT_MIME = "application/vnd.litloft.loft+json"
ALL_DONE = dict(
    metadata_indexed=True,
    clip_indexed=True,
    clip_thumbnail_indexed=True,
    whisper_indexed=True,
    text_indexed=True,
    tfidf_keywords_indexed=True,
    frame_scan_indexed=True,
    frame_ocr_read=True,
    frame_ocr_indexed=True,
)


# ---------------------------------------------------------------------------
# §4 - the table and its migration
# ---------------------------------------------------------------------------


def _affinity(declared: str) -> str:
    declared = declared.upper()
    if "INT" in declared:
        return "INTEGER"
    if any(word in declared for word in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if any(word in declared for word in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    return declared


def _columns(engine) -> dict[str, tuple[str, int, int]]:
    with engine.connect() as conn:
        return {
            row[1]: (_affinity(str(row[2])), row[3], row[5])
            for row in conn.execute(text("PRAGMA table_info(video_frame_lines)")).fetchall()
        }


def _indexes(engine) -> list[tuple[bool, list[str]]]:
    with engine.connect() as conn:
        found = []
        for row in conn.execute(text("PRAGMA index_list(video_frame_lines)")).fetchall():
            columns = [
                info[2] for info in conn.execute(text(f"PRAGMA index_info('{row[1]}')")).fetchall()
            ]
            found.append((bool(row[2]), columns))
        return found


def _insert(conn, segment_index: int, lines_json: str = "[]") -> None:
    conn.execute(text(
        "INSERT INTO video_frame_lines(file_id, segment_index, start_time, end_time, lines_json) "
        "VALUES('vid1', :i, 0.0, 3.0, :j)"
    ), {"i": segment_index, "j": lines_json})


def test_spec_addon_134_section_4_the_migration_creates_the_table_twice_keeping_rows(
    init_database,  # noqa: F811
):
    start, path = init_database
    engine = create_engine(f"sqlite:///{path}")
    from app.database import Base

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS video_frame_lines"))
    engine.dispose()

    start()

    reader = create_engine(f"sqlite:///{path}")
    columns = _columns(reader)
    assert set(columns) == {
        "id", "file_id", "segment_index", "start_time", "end_time", "lines_json",
    }
    assert (columns["id"][0], columns["id"][2]) == ("INTEGER", 1)
    assert {name: columns[name][:2] for name in columns if name != "id"} == {
        "file_id": ("TEXT", 1),
        "segment_index": ("INTEGER", 1),
        "start_time": ("REAL", 1),
        "end_time": ("REAL", 1),
        "lines_json": ("TEXT", 1),
    }
    indexes = _indexes(reader)
    assert (True, ["file_id", "segment_index"]) in indexes
    assert (False, ["file_id"]) in indexes
    with reader.begin() as conn:
        _insert(conn, 0, '[{"text":"KEEP","quad":[0,0,1,0,1,1,0,1]}]')
    with pytest.raises(IntegrityError), reader.begin() as conn:
        _insert(conn, 0)
    reader.dispose()

    start()

    reader = create_engine(f"sqlite:///{path}")
    with reader.connect() as conn:
        assert conn.execute(text(
            "SELECT file_id, segment_index, start_time, end_time, lines_json FROM video_frame_lines"
        )).fetchall() == [("vid1", 0, 0.0, 3.0, '[{"text":"KEEP","quad":[0,0,1,0,1,1,0,1]}]')]
    reader.dispose()


# ---------------------------------------------------------------------------
# §4, I7 - the purge
# ---------------------------------------------------------------------------


def _add_file(session, file_id: str) -> None:
    from app.models import IndexedFile

    session.add(IndexedFile(
        file_id=file_id, drive="drive1", filename=f"{file_id}.mp4",
        file_path=f"/drives/drive1/{file_id}.mp4", file_type="video",
        mime_type="video/mp4", file_size=1, active=True,
    ))


def _rows(session, table: str, file_id: str) -> list:
    return session.execute(
        text(f"SELECT * FROM {table} WHERE file_id = :f"), {"f": file_id},
    ).fetchall()


def test_spec_addon_134_i7_a_purge_deletes_the_files_lines_and_completes_without_the_table(
    search_index,  # noqa: F811
):
    import app.database as database
    from app.indexer import _purge_file

    with database.get_search_db() as session:
        for file_id in ("vid1", "vid2", "vid3"):
            _add_file(session, file_id)
    with database.get_search_db() as session:
        for file_id in ("vid1", "vid2"):
            for segment in (0, 1):
                session.execute(text(
                    "INSERT INTO video_frame_lines"
                    "(file_id, segment_index, start_time, end_time, lines_json) "
                    "VALUES(:f, :i, 0.0, 3.0, '[]')"
                ), {"f": file_id, "i": segment})
        session.execute(text(
            "INSERT INTO video_frame_texts(file_id, chunk_index, start_time, end_time, text) "
            "VALUES('vid3', 0, 0.0, 3.0, 'frame text')"
        ))

    _purge_file("vid1")

    with database.get_search_db_read() as session:
        assert _rows(session, "video_frame_lines", "vid1") == []
        assert len(_rows(session, "video_frame_lines", "vid2")) == 2

    with database.get_search_db() as session:
        session.execute(text("DROP TABLE video_frame_lines"))

    _purge_file("vid3")

    with database.get_search_db_read() as session:
        assert _rows(session, "video_frame_texts", "vid3") == []
        assert _rows(session, "indexed_files", "vid3") == []


# ---------------------------------------------------------------------------
# Item 1.6, §3 - rule 3 is current; NULL, 1 and 2 are outdated under on_index
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path, monkeypatch) -> SearchDB:
    return build_search_db(tmp_path, monkeypatch)


def _video(db, tmp_path, file_id: str, **flags) -> None:
    seed_indexed_file(
        db, file_id, "video/mp4", tmp_path / f"{file_id}.mp4", file_type="video",
        **{**ALL_DONE, **flags},
    )


def _loft(db, tmp_path, file_id: str, **flags) -> None:
    path = tmp_path / f"{file_id}.loft"
    path.write_text('{"provider": "youtube", "url": "https://www.youtube.com/watch?v=x"}')
    seed_indexed_file(db, file_id, LOFT_MIME, path, file_type="video", **{**ALL_DONE, **flags})
    db.execute(
        "UPDATE indexed_files SET loft_source = 'youtube' WHERE file_id = :f", f=file_id,
    )


async def _queued(manager) -> list[tuple[str, int]]:
    from app.indexer import TaskType

    return sorted(
        (t.file_id, t.priority)
        for t in await manager._collect_batch(TaskType("frame_ocr"), 10)
    )


async def test_spec_addon_134_item_1_6_rule_two_is_outdated_and_rule_three_is_current(
    db, tmp_path, monkeypatch,
):
    from app.indexer import IndexManager

    patch_features(monkeypatch, video_frame_ocr="on_index")
    _video(db, tmp_path, "none")
    _video(db, tmp_path, "one", frame_ocr_rule=1)
    _video(db, tmp_path, "two", frame_ocr_rule=2)
    _video(db, tmp_path, "three", frame_ocr_rule=3)
    _loft(db, tmp_path, "yt2", frame_ocr_rule=2)
    _loft(db, tmp_path, "ytnone")
    manager = IndexManager()

    await manager._resume_incomplete()

    assert await _queued(manager) == [("none", -10), ("one", -10), ("two", -10)]


@pytest.mark.parametrize("mode", ["manual", "false"])
async def test_spec_addon_134_item_1_6_outside_on_index_a_rule_two_read_is_not_queued(
    db, tmp_path, monkeypatch, mode,
):
    from app.indexer import IndexManager

    _video(db, tmp_path, "two", frame_ocr_rule=2)
    patch_features(monkeypatch, video_frame_ocr=mode)
    off = IndexManager()
    await off._resume_incomplete()
    assert await _queued(off) == []

    patch_features(monkeypatch, video_frame_ocr="on_index")
    on = IndexManager()
    await on._resume_incomplete()
    assert await _queued(on) == [("two", -10)]
    assert db.flag("two", "frame_ocr_rule") == 2


async def test_spec_addon_134_item_1_6_prioritize_raises_a_rule_two_read_to_100(
    db, tmp_path, monkeypatch,
):
    from app.indexer import IndexManager

    patch_features(monkeypatch, video_frame_ocr="on_index")
    _video(db, tmp_path, "two", frame_ocr_rule=2)
    _video(db, tmp_path, "three", frame_ocr_rule=3)
    manager = IndexManager()

    await manager.prioritize("two")
    await manager.prioritize("three")

    assert await _queued(manager) == [("two", 100)]


def test_spec_addon_134_item_1_6_status_counts_rule_two_as_pending_and_rule_three_as_indexed(
    db, tmp_path, monkeypatch,
):
    from app import main

    patch_features(monkeypatch, video_frame_ocr="on_index")
    _video(db, tmp_path, "base", frame_ocr_read=False, frame_ocr_indexed=False)
    manager = main.IndexManager()
    monkeypatch.setattr(main.dependencies, "get_index_manager", lambda: manager)
    monkeypatch.setattr(main.dependencies, "_video_visual_worker", None)
    baseline = main.status_endpoint()
    _video(db, tmp_path, "two", frame_ocr_rule=2)
    _video(db, tmp_path, "three", frame_ocr_rule=3)

    status = main.status_endpoint()

    assert status.pending["frame_ocr"] - baseline.pending["frame_ocr"] == 1
    assert status.pending["total"] - baseline.pending["total"] == 1
    assert status.indexed["frame_ocr"] - baseline.indexed["frame_ocr"] == 1


async def test_spec_addon_134_item_1_6_index_details_show_rule_two_as_not_done(
    db, tmp_path, monkeypatch,
):
    from app.routers.files import get_index_details

    patch_features(monkeypatch, video_frame_ocr="on_index")
    _video(db, tmp_path, "two", frame_ocr_rule=2)
    _video(db, tmp_path, "three", frame_ocr_rule=3)

    async def done(file_id: str) -> bool:
        response = await get_index_details(file_id=file_id, drive="drive1")
        payload = response if isinstance(response, dict) else response.model_dump()
        return bool(payload["status"]["frame_ocr"])

    assert await done("two") is False
    assert await done("three") is True


# ---------------------------------------------------------------------------
# Item 1.7, §3 - a .loft read is stored with its lines under rule 3
# ---------------------------------------------------------------------------


@pytest.fixture()
def world(search_index, youtube, monkeypatch, tmp_path, make_settings):  # noqa: F811
    return LoftWorld(search_index, youtube, monkeypatch, tmp_path, make_settings)


@needs_ffmpeg
def test_spec_addon_134_item_1_7_a_loft_read_stores_its_lines_under_rule_three(
    world, gradient, monkeypatch,  # noqa: F811
):
    world.configure(frame_ocr="on_index")
    install(monkeypatch, SimpleBoxEngine())
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001", frame_scan_indexed=True,
                   frame_ocr_read=False, frame_ocr_indexed=False)

    fetch("ytloft000001")

    chunks = world.frame_texts("ytloft000001")
    assert chunks != []
    assert world.value("ytloft000001", "frame_ocr_rule") == 3
    with world.index.engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT segment_index, start_time, end_time, lines_json FROM video_frame_lines "
            "WHERE file_id = 'ytloft000001' ORDER BY segment_index"
        )).fetchall()
    assert rows
    spans = {(c[1], c[2]) for c in chunks}
    assert {(row[1], row[2]) for row in rows} == spans
    for row in rows:
        segment_chunks = [c[3] for c in chunks if (c[1], c[2]) == (row[1], row[2])]
        assert "\n".join(texts_of(row[3])) == segment_chunks[0]
