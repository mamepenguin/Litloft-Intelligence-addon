"""SPEC-ADDON-134 I6 and the revised SPEC-ADDON-133 I8: only the store step writes
``video_frame_lines``. The build, a reindex request, a failed store and a re-read
that does not complete leave the file's rows byte for byte as they were.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time

import numpy as np
import pytest
from fastapi.responses import JSONResponse

from app.models import VideoFrameText  # noqa: F401 - registers the table
from tests.frame_box_doubles import (
    BoxEngine,
    DirectEngine,
    State,
    encode_scene,
    frame_texts,
    install,
)
from tests.frame_line_doubles import all_line_rows, line_rows, seed_line_row, texts_of
from tests.ocr_fixtures import (
    build_search_db,
    patch_features,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    seed_indexed_file,
    seed_ocr_rows,
)
from tests.test_frame_page_ocr_readers import media  # noqa: F401 - fixture
from tests.test_spec_addon_133_frame_ocr_reread import wire_reader

pytestmark = pytest.mark.usefixtures("pillow_real")

MKV = "video/x-matroska"
A = (20, 60, 110, 80)
EARLIER = ("EARLIER ONE", "EARLIER TWO")
KEPT = '[{"text":"EARLIER ONE","quad":[0.1,0.1,0.9,0.1,0.9,0.2,0.1,0.2]}]'


@pytest.fixture()
def db(tmp_path, monkeypatch):
    return build_search_db(tmp_path, monkeypatch)


@pytest.fixture()
def reader(db, monkeypatch):
    module, source = wire_reader(db, monkeypatch)
    return module, source


@pytest.fixture()
def scene(tmp_path, monkeypatch):
    states = [State(boxes=[("A", A)]) for _ in range(3)]
    path = encode_scene(tmp_path / "scene.mkv", states)
    fake = BoxEngine(states)
    install(monkeypatch, fake)
    return path, fake


def _seed(db, path, *, mime: str = MKV, **flags) -> None:
    values = dict(frame_ocr_read=True, frame_ocr_indexed=True)
    values.update(flags)
    seed_indexed_file(db, "vid1", mime, path, file_type="video", **values)
    for chunk, line in enumerate(EARLIER):
        db.execute(
            "INSERT INTO video_frame_texts"
            "(file_id, chunk_index, start_time, end_time, text) "
            "VALUES('vid1', :c, :s, :e, :t)",
            c=chunk, s=chunk * 10.0, e=chunk * 10.0 + 10.0, t=line,
        )
    seed_ocr_rows(db, "vid1", EARLIER)
    seed_line_row(db, segment_index=0, start=0.0, end=10.0, lines_json=KEPT)
    seed_line_row(db, segment_index=2, start=10.0, end=20.0, lines_json=KEPT)


def _snapshot(db) -> dict:
    return {
        "frames": frame_texts(db),
        "lines": all_line_rows(db),
        "embeddings": db.ocr_embeddings("vid1"),
        "fts": db.ocr_fts("vid1"),
        "read": bool(db.flag("vid1", "frame_ocr_read")),
        "indexed": bool(db.flag("vid1", "frame_ocr_indexed")),
    }


# ---------------------------------------------------------------------------
# Item 1.5, I6 - the build step writes no row
# ---------------------------------------------------------------------------


def test_spec_addon_134_i6_the_build_after_a_store_leaves_the_rows_as_the_store_wrote_them(
    db, reader, scene, monkeypatch,
):
    module, _source = reader
    monkeypatch.setattr(module, "_get_engine", lambda: DirectEngine())
    path, fake = scene
    seed_indexed_file(db, "vid1", MKV, path, file_type="video")

    def down(_texts):
        raise RuntimeError("embedder down")

    monkeypatch.setattr(module, "embed_passages", down)
    module.index_frame_ocr("vid1")
    assert bool(db.flag("vid1", "frame_ocr_indexed")) is False
    assert [texts_of(row[3]) for row in line_rows(db)] == [
        ["A#0", "tick#0"], ["A#0", "tick#1"], ["A#0", "tick#2"],
    ]
    db.execute("DELETE FROM video_frame_lines WHERE segment_index = 1")
    db.execute("UPDATE video_frame_lines SET lines_json = 'MARK' WHERE segment_index = 2")
    stored = all_line_rows(db)

    monkeypatch.setattr(module, "embed_passages", lambda texts: [
        np.zeros(4, dtype="float32") for _ in texts
    ])
    detects = len(fake.detects())
    module.index_frame_ocr("vid1")

    assert len(fake.detects()) == detects
    assert bool(db.flag("vid1", "frame_ocr_indexed")) is True
    assert all_line_rows(db) == stored


@pytest.mark.parametrize("rule", [None, 2, 3])
def test_spec_addon_134_i6_a_rebuild_only_task_leaves_the_rows(db, reader, media, rule):  # noqa: F811
    module, source = reader
    flags = dict(frame_ocr_indexed=False)
    if rule is not None:
        flags["frame_ocr_rule"] = rule
    _seed(db, media["landscape"], mime="video/mp4", **flags)
    lines = all_line_rows(db)

    module.index_frame_ocr("vid1")

    assert source.engine.calls == 0
    assert bool(db.flag("vid1", "frame_ocr_indexed")) is True
    assert all_line_rows(db) == lines


# ---------------------------------------------------------------------------
# I6 - a reindex request
# ---------------------------------------------------------------------------


async def test_spec_addon_134_i6_a_reindex_request_leaves_the_rows(db, tmp_path, monkeypatch):
    from app.indexer import IndexManager
    from app.routers import files as files_router

    patch_features(monkeypatch, video_frame_ocr="on_index")
    _seed(db, tmp_path / "v.mp4", mime="video/mp4", frame_ocr_rule=3)
    lines = all_line_rows(db)
    manager = IndexManager()
    monkeypatch.setattr(files_router, "get_index_manager", lambda: manager)

    result = await files_router.reindex_file(
        "vid1", body={"tasks": ["frame_ocr"]}, drive="drive1",
    )
    if isinstance(result, JSONResponse):
        result = json.loads(result.body)
    elif not isinstance(result, dict):
        result = result.model_dump()

    assert result["status"] == "accepted"
    assert bool(db.flag("vid1", "frame_ocr_read")) is False
    assert db.flag("vid1", "frame_ocr_rule") == 3
    assert all_line_rows(db) == lines


# ---------------------------------------------------------------------------
# §2.3, I6 - a store transaction that fails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["video_frame_texts", "video_frame_lines"])
def test_spec_addon_134_i6_a_failed_store_writes_nothing_including_the_lines(
    db, reader, scene, monkeypatch, table,
):
    module, _source = reader
    monkeypatch.setattr(module, "_get_engine", lambda: DirectEngine())
    path, fake = scene
    _seed(db, path)
    before = _snapshot(db)
    db.execute(
        f"CREATE TRIGGER refuse_store BEFORE INSERT ON {table} "
        "BEGIN SELECT RAISE(ABORT, 'disk full'); END"
    )

    with contextlib.suppress(Exception):
        module.index_frame_ocr("vid1")

    assert fake.detects() == [0, 1, 2]
    assert _snapshot(db) == before
    assert db.flag("vid1", "frame_ocr_rule") is None


# ---------------------------------------------------------------------------
# Revised SPEC-ADDON-133 I8 - re-reads that do not complete
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["unreadable", "no-frame", "engine-raises"])
def test_spec_addon_134_revised_133_i8_an_incomplete_reread_keeps_the_lines_and_sets_rule_three(
    db, reader, media, monkeypatch, outcome,  # noqa: F811
):
    module, _source = reader
    clip = media["corrupt"] if outcome == "unreadable" else media["landscape"]
    _seed(db, clip, mime="video/mp4")
    if outcome == "no-frame":
        monkeypatch.setattr(module, "_frame_command", lambda *args: ["sh", "-c", "exit 0"])
    before = _snapshot(db)

    module.index_frame_ocr("vid1")

    assert _snapshot(db) == before
    assert db.flag("vid1", "frame_ocr_rule") == 3


@pytest.mark.parametrize("rule", [None, 2])
def test_spec_addon_134_revised_133_i8_a_reread_that_cannot_reach_the_file_changes_nothing(
    db, reader, tmp_path, rule,
):
    module, _source = reader
    flags = {} if rule is None else dict(frame_ocr_rule=rule)
    _seed(db, tmp_path / "gone.mp4", mime="video/mp4", **flags)
    before = _snapshot(db)

    module.index_frame_ocr("vid1")

    assert _snapshot(db) == before
    assert db.flag("vid1", "frame_ocr_rule") == rule


def test_spec_addon_134_revised_133_i8_a_running_reread_changes_no_row_before_its_store(
    db, reader, scene, monkeypatch,
):
    module, _source = reader
    monkeypatch.setattr(module, "_get_engine", lambda: DirectEngine())
    path, fake = scene
    _seed(db, path, frame_ocr_rule=2)
    before = _snapshot(db)
    reached, release = threading.Event(), threading.Event()

    def hold(k: int) -> None:
        if k == 2:
            reached.set()
            release.wait(timeout=20)

    fake.on_recognise = hold
    thread = threading.Thread(target=module.index_frame_ocr, args=("vid1",))
    thread.start()
    try:
        assert reached.wait(timeout=20)
        time.sleep(0.2)
        during = _snapshot(db)
        rule_during = db.flag("vid1", "frame_ocr_rule")
    finally:
        release.set()
        thread.join(timeout=20)

    assert during == before
    assert rule_during == 2
    assert [texts_of(row[3]) for row in line_rows(db)] == [
        ["A#0", "tick#0"], ["A#0", "tick#1"], ["A#0", "tick#2"],
    ]
    assert db.flag("vid1", "frame_ocr_rule") == 3
