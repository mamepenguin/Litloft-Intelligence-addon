"""SPEC-ADDON-133 with an engine that answers: a stored read writes the current rule, a
completed re-read replaces the earlier read through the store-then-build path,
and an incomplete or failed one leaves it.

Item 1 step 6, §2, §3, I6, I8. The engine is faked at ``detect_boxes`` /
``recognise_boxes`` (``tests.frame_box_doubles``).
"""

from __future__ import annotations

import contextlib
import threading
import time

import numpy as np
import pytest

from app.models import VideoFrameText  # noqa: F401 - registers the table
from tests.frame_box_doubles import (
    BoxEngine,
    DirectEngine,
    SimpleBoxEngine,
    State,
    encode_scene,
    frame_texts,
    install,
)
from tests.ocr_fixtures import (
    build_search_db,
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
NEW = ["A#0\ntick#0", "A#0\ntick#1", "A#0\ntick#2"]


@pytest.fixture()
def db(tmp_path, monkeypatch):
    return build_search_db(tmp_path, monkeypatch)


@pytest.fixture()
def reader(db, monkeypatch):
    module, _source = wire_reader(db, monkeypatch)
    monkeypatch.setattr(module, "_get_engine", lambda: DirectEngine())
    return module


@pytest.fixture()
def scene(tmp_path, monkeypatch):
    states = [State(boxes=[("A", A)]) for _ in range(3)]
    path = encode_scene(tmp_path / "scene.mkv", states)
    fake = BoxEngine(states)
    install(monkeypatch, fake)
    return path, fake


def _seed(db, path, *, earlier: bool, mime: str = MKV, **flags) -> None:
    values = dict(frame_ocr_read=earlier, frame_ocr_indexed=earlier)
    values.update(flags)
    seed_indexed_file(db, "vid1", mime, path, file_type="video", **values)
    if earlier:
        for chunk, line in enumerate(EARLIER):
            db.execute(
                "INSERT INTO video_frame_texts"
                "(file_id, chunk_index, start_time, end_time, text) "
                "VALUES('vid1', :c, :s, :e, :t)",
                c=chunk, s=chunk * 10.0, e=chunk * 10.0 + 10.0, t=line,
            )
        seed_ocr_rows(db, "vid1", EARLIER)


def _state(db) -> tuple:
    return (
        bool(db.flag("vid1", "frame_ocr_read")),
        bool(db.flag("vid1", "frame_ocr_indexed")),
        db.flag("vid1", "frame_ocr_rule"),
    )


def _snapshot(db) -> dict:
    return {
        "frames": frame_texts(db),
        "embeddings": db.ocr_embeddings("vid1"),
        "fts": db.ocr_fts("vid1"),
        "read": bool(db.flag("vid1", "frame_ocr_read")),
        "indexed": bool(db.flag("vid1", "frame_ocr_indexed")),
    }


def _searchable(db) -> list[str]:
    return sorted(row[2] for row in db.ocr_fts("vid1"))


# ---------------------------------------------------------------------------
# Item 1 step 6, §3, I6 - a stored read writes the current rule
# ---------------------------------------------------------------------------


def test_spec_addon_133_i6_a_first_read_is_stored_under_the_current_rule(db, reader, scene):
    path, _fake = scene
    _seed(db, path, earlier=False)

    assert reader.index_frame_ocr("vid1") is True

    assert frame_texts(db) == NEW
    assert _state(db) == (True, True, 3)


def test_spec_addon_133_section_3_a_completed_reread_replaces_the_earlier_read(db, reader, scene):
    path, fake = scene
    _seed(db, path, earlier=True)

    reader.index_frame_ocr("vid1")

    assert fake.detects() == [0, 1, 2]
    assert frame_texts(db) == NEW
    assert _state(db) == (True, True, 3)
    searchable = _searchable(db)
    assert searchable
    assert not any(line in text for text in searchable for line in EARLIER)
    assert any("A#0" in text for text in searchable)


def test_spec_addon_133_i6_the_store_sets_indexed_false_and_the_build_sets_it_true(
    db, reader, scene, monkeypatch,
):
    path, fake = scene
    _seed(db, path, earlier=True)

    def down(_texts):
        raise RuntimeError("embedder down")

    monkeypatch.setattr(reader, "embed_passages", down)
    reader.index_frame_ocr("vid1")

    assert frame_texts(db) == NEW
    assert _state(db) == (True, False, 3)
    assert db.ocr_embeddings("vid1") == []
    assert db.ocr_fts("vid1") == []

    detects = len(fake.detects())
    monkeypatch.setattr(reader, "embed_passages", lambda texts: [
        np.zeros(4, dtype="float32") for _ in texts
    ])
    reader.index_frame_ocr("vid1")

    assert len(fake.detects()) == detects
    assert _state(db) == (True, True, 3)
    assert db.ocr_embeddings("vid1") != []
    assert any("A#0" in text for text in _searchable(db))


def test_spec_addon_133_i6_a_reread_whose_store_fails_writes_nothing_and_is_tried_again(
    db, reader, scene,
):
    import asyncio

    from app.indexer import IndexManager, TaskType

    path, fake = scene
    _seed(db, path, earlier=True)
    db.execute(
        "CREATE TRIGGER refuse_second_chunk BEFORE INSERT ON video_frame_texts "
        "WHEN NEW.chunk_index = 1 AND NEW.text <> 'EARLIER TWO' "
        "BEGIN SELECT RAISE(ABORT, 'disk full'); END"
    )
    before = _snapshot(db)

    with contextlib.suppress(Exception):
        reader.index_frame_ocr("vid1")

    assert fake.detects() == [0, 1, 2]
    assert _snapshot(db) == before
    assert db.flag("vid1", "frame_ocr_rule") is None

    async def queued():
        manager = IndexManager()
        await manager._resume_incomplete()
        return [(t.file_id, t.priority) for t in await manager._collect_batch(TaskType.FRAME_OCR, 10)]

    assert asyncio.run(queued()) == [("vid1", -10)]


# ---------------------------------------------------------------------------
# I8 - a running re-read changes nothing before its store step
# ---------------------------------------------------------------------------


def test_spec_addon_133_i8_a_running_reread_leaves_the_earlier_read_searchable(db, reader, scene):
    path, fake = scene
    _seed(db, path, earlier=True)
    before = _snapshot(db)
    reached, release = threading.Event(), threading.Event()

    def hold(k: int) -> None:
        if k == 2:
            reached.set()
            release.wait(timeout=20)

    fake.on_recognise = hold
    thread = threading.Thread(target=reader.index_frame_ocr, args=("vid1",))
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
    assert rule_during is None
    assert frame_texts(db) == NEW
    assert _state(db) == (True, True, 3)


# ---------------------------------------------------------------------------
# §2, I8 - incomplete re-reads with an answering engine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["ended-early", "one-frame-raised"])
def test_spec_addon_133_i8_an_incomplete_reread_that_read_text_keeps_the_earlier_read(
    db, reader, media, monkeypatch, outcome,  # noqa: F811
):
    fake = SimpleBoxEngine()
    install(monkeypatch, fake)
    if outcome == "ended-early":
        # A truncated file still ends with ffmpeg's exit 0, so the decode is
        # the real one with a non-zero exit after its last frame.
        clip = media["landscape"]
        command = reader._frame_command
        monkeypatch.setattr(
            reader, "_frame_command",
            lambda *args: ["sh", "-c", '"$@"; exit 1', "sh", *command(*args)],
        )
    else:
        clip = media["landscape"]
        fake.recognise_raises_on = 2
    _seed(db, clip, earlier=True, mime="video/mp4")
    before = _snapshot(db)

    reader.index_frame_ocr("vid1")

    assert fake.recognises >= 1
    assert _snapshot(db) == before
    assert db.flag("vid1", "frame_ocr_rule") == 3


def test_spec_addon_133_failure_a_first_read_that_ended_early_keeps_its_frames_under_the_current_rule(
    db, reader, media, monkeypatch,  # noqa: F811
):
    fake = SimpleBoxEngine()
    install(monkeypatch, fake)
    _seed(db, media["large_tail"], earlier=False, mime="video/mp4")

    reader.index_frame_ocr("vid1")

    assert fake.recognises >= 1
    assert frame_texts(db)
    assert _state(db) == (True, True, 3)
