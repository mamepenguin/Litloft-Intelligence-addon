"""Acceptance tests for the picture OCR worker (SPEC-ADDON-050).

The engine is not installed, so each test replaces the single place the worker
creates it with a fake that answers the lines it is given. What is held here is
the worker's contract, not the engine's reading.
"""

from __future__ import annotations

import asyncio
import logging
import re
import socket
import time
from pathlib import Path

import pytest

from app.extractors.base import ContentExtractor
from tests.ocr_fixtures import (
    FakeEngine,
    SearchDB,
    animated_gif,
    build_search_db,
    heic_picture,
    huge_pixel_png,
    install,
    jpeg_with_exif_orientation_6,
    oversize_png,
    patch_features,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    picture,
    probe_write_outside_the_lock,
    rgba_picture,
    seed_indexed_file,
    seed_ocr_rows,
)

pytestmark = pytest.mark.usefixtures("pillow_real")

ALL_DONE = dict(
    metadata_indexed=True,
    clip_indexed=True,
    clip_thumbnail_indexed=True,
    whisper_indexed=True,
    text_indexed=True,
    tfidf_keywords_indexed=True,
)


@pytest.fixture()
def search_db(tmp_path, monkeypatch) -> SearchDB:
    return build_search_db(tmp_path, monkeypatch)


def run_worker(harness, file_id: str) -> bool:
    return harness.module.index_ocr(file_id)


async def wait_for(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not reached")


def make_manager():
    from app.indexer import IndexManager

    return IndexManager()


# ---------------------------------------------------------------------------
# I1 - the rows written
# ---------------------------------------------------------------------------


def test_stores_exactly_the_engine_lines_joined_by_newlines(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-050 I1: the kept lines, in engine order, become the text."""
    path = picture(tmp_path / "sign.png")
    seed_indexed_file(search_db, "pic1", "image/png", path, **ALL_DONE)
    lines = ["会議は10時から", "first line", "second line"]
    harness = install(monkeypatch, lines=lines)

    run_worker(harness, "pic1")

    joined = "\n".join(lines)
    assert bool(search_db.flag("pic1", "ocr_indexed")) is True
    assert search_db.ocr_embeddings("pic1") == [(0, None, joined)]
    assert search_db.ocr_fts("pic1") == [(0, "", joined)]
    assert harness.engine.seen[0].mode == "RGB"


def test_chunks_the_joined_lines_at_1000_over_200(search_db, tmp_path, monkeypatch):
    """SPEC-ADDON-050 I1: a long reading is chunked at the extractor's defaults."""
    path = picture(tmp_path / "page.png")
    seed_indexed_file(search_db, "pic1", "image/png", path, **ALL_DONE)
    lines = tuple(f"line {index}: " + "x" * 290 for index in range(6))
    joined = "\n".join(lines)
    expected = ContentExtractor.chunk_text(joined)
    assert len(expected) > 1
    harness = install(monkeypatch, lines=lines)

    run_worker(harness, "pic1")

    assert [row[2] for row in search_db.ocr_embeddings("pic1")] == [
        chunk[:200] for chunk in expected
    ]
    assert search_db.ocr_fts("pic1") == [
        (index, "", chunk) for index, chunk in enumerate(expected)
    ]
    assert [row[0] for row in search_db.ocr_embeddings("pic1")] == list(
        range(len(expected))
    )
    assert all(row[1] is None for row in search_db.ocr_embeddings("pic1"))
    assert all(len(chunk) <= 1000 for chunk in expected)


def test_replaces_earlier_rows_of_this_picture_and_no_other_files(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-050 I1/I6: earlier OCR rows go; another file's rows do not."""
    mine = picture(tmp_path / "mine.png")
    other = picture(tmp_path / "other.png")
    seed_indexed_file(search_db, "pic1", "image/png", mine, **ALL_DONE)
    seed_indexed_file(search_db, "pic2", "image/png", other, **ALL_DONE)
    seed_ocr_rows(search_db, "pic1", ("stale one", "stale two"))
    seed_ocr_rows(search_db, "pic2", ("theirs",))
    harness = install(monkeypatch, lines=["fresh"])

    run_worker(harness, "pic1")

    assert search_db.ocr_fts("pic1") == [(0, "", "fresh")]
    assert [row[2] for row in search_db.ocr_embeddings("pic1")] == ["fresh"]
    assert search_db.ocr_fts("pic2") == [(0, "3", "theirs")]
    assert [row[2] for row in search_db.ocr_embeddings("pic2")] == ["theirs"]


# ---------------------------------------------------------------------------
# I2 - reached but nothing kept
# ---------------------------------------------------------------------------


def _garbage(path: Path) -> Path:
    path.write_bytes(b"this is not any picture format" * 8)
    return path


@pytest.mark.parametrize(
    "case,builder,engine_error,engine_calls,warning",
    [
        ("no-lines", picture, None, 1, False),
        ("undecodable", _garbage, None, 0, True),
        ("over-pixel-cap", huge_pixel_png, None, 0, True),
        ("over-byte-cap", oversize_png, None, 0, True),
        ("engine-raises", picture, RuntimeError("engine blew up"), 1, True),
    ],
    ids=["no-lines", "undecodable", "over-pixel-cap", "over-byte-cap", "engine-raises"],
)
def test_a_reached_but_unreadable_picture_ends_indexed_with_no_rows(
    search_db, tmp_path, monkeypatch, caplog, case, builder, engine_error,
    engine_calls, warning,
):
    """SPEC-ADDON-050 I2: true and empty, not pending forever."""
    caplog.set_level(logging.WARNING)
    path = builder(tmp_path / f"{case}.png")
    seed_indexed_file(search_db, "pic1", "image/png", path, **ALL_DONE)
    seed_ocr_rows(search_db, "pic1", ("stale",))
    harness = install(monkeypatch, lines=[] if case == "no-lines" else ["never reached"])
    harness.engine.error = engine_error

    run_worker(harness, "pic1")

    assert bool(search_db.flag("pic1", "ocr_indexed")) is True
    assert search_db.ocr_embeddings("pic1") == []
    assert search_db.ocr_fts("pic1") == []
    assert len(harness.engine.seen) == engine_calls
    if warning:
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        named = [message for message in messages if "pic1" in message]
        assert named, f"no WARNING named pic1; saw {messages}"
        if engine_error is not None:
            assert any("RuntimeError" in message for message in named)
        elif case == "undecodable":
            assert any(re.search(r"[A-Za-z_]+Error", message) for message in named)


@pytest.mark.parametrize(
    "builder",
    [picture, _garbage, huge_pixel_png, oversize_png],
    ids=["no-lines", "undecodable", "over-pixel-cap", "over-byte-cap"],
)
async def test_no_sweep_queues_a_picture_that_ended_true(
    search_db, tmp_path, monkeypatch, builder
):
    """SPEC-ADDON-050 I2: a finished picture is not picked up again."""
    from app.indexer import TaskType

    patch_features(monkeypatch, image_ocr="on_index")
    path = builder(tmp_path / "done.png")
    seed_indexed_file(search_db, "pic1", "image/png", path, **ALL_DONE)
    harness = install(monkeypatch, lines=[])

    run_worker(harness, "pic1")

    manager = make_manager()
    await manager._resume_incomplete()

    assert bool(search_db.flag("pic1", "ocr_indexed")) is True
    assert manager._queues[TaskType.OCR].qsize() == 0


# ---------------------------------------------------------------------------
# I3 - failures that keep the picture pending
# ---------------------------------------------------------------------------


def _unreachable_path(tmp_path: Path, case: str) -> Path:
    if case == "missing":
        return tmp_path / "gone.png"
    directory = tmp_path / "a-directory.png"
    directory.mkdir()
    return directory


@pytest.mark.parametrize("case", ["missing", "directory", "embedding"])
async def test_failures_keep_rows_and_the_picture_pending(
    search_db, tmp_path, monkeypatch, caplog, case
):
    """SPEC-ADDON-050 I3: nothing written, false, and the sweep retries it."""
    from app.indexer import TaskType

    caplog.set_level(logging.WARNING)
    patch_features(monkeypatch, image_ocr="on_index")
    if case == "embedding":
        path = picture(tmp_path / "pic.png")
        seed_indexed_file(search_db, "pic1", "image/png", path, **ALL_DONE)
        harness = install(
            monkeypatch, lines=["text"], embed_error=RuntimeError("embed down")
        )
    else:
        path = _unreachable_path(tmp_path, case)
        seed_indexed_file(search_db, "pic1", "image/png", path, **ALL_DONE)
        harness = install(monkeypatch, lines=["text"])
    seed_ocr_rows(search_db, "pic1", ("kept",))

    run_worker(harness, "pic1")

    assert bool(search_db.flag("pic1", "ocr_indexed")) is False
    assert search_db.ocr_fts("pic1") == [(0, "3", "kept")]
    assert [row[2] for row in search_db.ocr_embeddings("pic1")] == ["kept"]
    if case != "embedding":
        assert any(
            "pic1" in record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    manager = make_manager()
    await manager._resume_incomplete()

    taken = await manager._collect_batch(TaskType.OCR, 5)
    assert [task.file_id for task in taken] == ["pic1"]
    assert taken[0].priority == -10


def test_a_picture_that_leaves_the_index_while_being_read_writes_nothing(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-050 I3: the write transaction finds no active row and stops."""
    path = picture(tmp_path / "pic.png")
    seed_indexed_file(search_db, "pic1", "image/png", path, **ALL_DONE)
    seed_ocr_rows(search_db, "pic1", ("kept",))
    engine = FakeEngine(("late text",))
    engine.on_call = lambda image: search_db.execute(
        "UPDATE indexed_files SET active = 0 WHERE file_id = 'pic1'"
    )
    harness = install(monkeypatch, engine=engine)

    run_worker(harness, "pic1")

    assert search_db.flag("pic1", "active") == 0
    assert search_db.ocr_fts("pic1") == [(0, "3", "kept")]
    assert [row[2] for row in search_db.ocr_embeddings("pic1")] == ["kept"]
    assert bool(search_db.flag("pic1", "ocr_indexed")) is False


async def test_a_missing_engine_drops_ocr_tasks_and_other_indexing_continues(
    search_db, tmp_path, monkeypatch, caplog
):
    """SPEC-ADDON-050 I3: one ERROR, no flag change, other queues keep going."""
    caplog.set_level(logging.WARNING)
    from app.indexer import IndexTask, TaskType

    patch_features(monkeypatch, image_ocr="on_index")
    install(monkeypatch, engine_error=RuntimeError("models missing"))
    for file_id in ("pic1", "pic2"):
        seed_indexed_file(
            search_db, file_id, "image/png", picture(tmp_path / f"{file_id}.png"),
            **ALL_DONE,
        )
    seed_indexed_file(
        search_db, "doc1", "text/plain", tmp_path / "doc1.txt",
        **{**ALL_DONE, "metadata_indexed": False},
    )

    metadata_done: list[str] = []

    def fake_metadata(file_ids: list[str]) -> int:
        for file_id in file_ids:
            search_db.set_flag(file_id, "metadata_indexed", True)
            metadata_done.append(file_id)
        return len(file_ids)

    monkeypatch.setattr("app.indexer.index_metadata_batch", fake_metadata)
    manager = make_manager()
    for file_id in ("pic1", "pic2"):
        await manager._enqueue(IndexTask(file_id=file_id, task_type=TaskType.OCR))
    await manager._enqueue(IndexTask(file_id="doc1", task_type=TaskType.METADATA))

    ocr_worker = asyncio.create_task(manager._ocr_worker())
    metadata_worker = asyncio.create_task(manager._metadata_worker())
    try:
        await wait_for(
            lambda: metadata_done == ["doc1"]
            and manager._queues[TaskType.OCR].qsize() == 0
        )
    finally:
        for task in (ocr_worker, metadata_worker):
            task.cancel()
        await asyncio.gather(ocr_worker, metadata_worker, return_exceptions=True)

    assert metadata_done == ["doc1"]
    for file_id in ("pic1", "pic2"):
        assert bool(search_db.flag(file_id, "ocr_indexed")) is False
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1, [record.getMessage() for record in errors]


def test_a_new_process_asks_for_the_engine_again(search_db, tmp_path, monkeypatch):
    """SPEC-ADDON-050 I3: a failed creation is not remembered across starts."""
    path = picture(tmp_path / "pic.png")
    seed_indexed_file(search_db, "pic1", "image/png", path, **ALL_DONE)

    failed = install(monkeypatch, engine_error=RuntimeError("models missing"))
    run_worker(failed, "pic1")
    assert bool(search_db.flag("pic1", "ocr_indexed")) is False

    retried = install(monkeypatch, lines=["read now"])
    run_worker(retried, "pic1")

    assert search_db.ocr_fts("pic1") == [(0, "", "read now")]


# ---------------------------------------------------------------------------
# I4 - formats read the way the spec says
# ---------------------------------------------------------------------------


def _is_dark(pixel) -> bool:
    return pixel is not None and max(pixel) < 60


@pytest.mark.parametrize(
    "case,builder,mime,suffix,expected_size,pixel_check",
    [
        ("heic", heic_picture, "image/heic", ".heic", (32, 16), None),
        ("png", picture, "image/png", ".png", (40, 20),
         lambda pixel: pixel == (250, 250, 250)),
        ("rgba-flattened", rgba_picture, "image/png", ".png", (20, 10),
         lambda pixel: pixel == (255, 255, 255)),
        ("gif-first-frame", animated_gif, "image/gif", ".gif", (30, 10),
         lambda pixel: pixel == (220, 0, 0)),
        ("jpeg-exif-6", jpeg_with_exif_orientation_6, "image/jpeg", ".jpg",
         (20, 40), _is_dark),
    ],
    ids=["heic", "png", "rgba-flattened", "gif-first-frame", "jpeg-exif-6"],
)
def test_reads_every_picture_format_after_orientation_and_flattening(
    search_db, tmp_path, monkeypatch, case, builder, mime, suffix, expected_size,
    pixel_check,
):
    """SPEC-ADDON-050 I4: first frame, RGB, EXIF applied."""
    path = builder(tmp_path / f"{case}{suffix}")
    seed_indexed_file(search_db, "pic1", mime, path, **ALL_DONE)
    harness = install(monkeypatch, lines=["text"])

    run_worker(harness, "pic1")

    assert harness.engine.seen, "the engine never saw the picture"
    seen = harness.engine.seen[0]
    assert seen.size == expected_size
    assert seen.mode == "RGB"
    if pixel_check is not None:
        assert pixel_check(seen.pixel)
    assert search_db.ocr_fts("pic1") == [(0, "", "text")]


# ---------------------------------------------------------------------------
# I5, I7 - concurrency, threads, pause, no network
# ---------------------------------------------------------------------------


async def test_processes_one_picture_at_a_time(search_db, tmp_path, monkeypatch):
    """SPEC-ADDON-050 I5: a single picture is being read at any moment."""
    from app.indexer import IndexTask, TaskType

    patch_features(monkeypatch, image_ocr="on_index")
    engine = FakeEngine(("line",))
    engine.on_call = lambda image: time.sleep(0.05)
    harness = install(monkeypatch, engine=engine)
    for file_id in ("pic1", "pic2"):
        seed_indexed_file(
            search_db, file_id, "image/png", picture(tmp_path / f"{file_id}.png"),
            **ALL_DONE,
        )

    manager = make_manager()
    for file_id in ("pic1", "pic2"):
        await manager._enqueue(IndexTask(file_id=file_id, task_type=TaskType.OCR))
    worker = asyncio.create_task(manager._ocr_worker())
    try:
        await wait_for(
            lambda: manager._queues[TaskType.OCR].qsize() == 0
            and len(engine.seen) == 2
        )
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    assert len(engine.seen) == 2
    assert engine.max_concurrent == 1
    assert harness.factory.threads == 2


async def test_takes_no_task_while_the_queue_is_paused(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-050 I5: the pause holds OCR like every other queue."""
    from app.indexer import IndexTask, TaskType

    patch_features(monkeypatch, image_ocr="on_index")
    harness = install(monkeypatch, lines=["paused"])
    seed_indexed_file(
        search_db, "pic1", "image/png", picture(tmp_path / "pic.png"), **ALL_DONE
    )
    manager = make_manager()
    manager.pause()
    await manager._enqueue(IndexTask(file_id="pic1", task_type=TaskType.OCR))

    worker = asyncio.create_task(manager._ocr_worker())
    try:
        await asyncio.sleep(0.2)
        assert harness.engine.seen == []
        assert manager._queues[TaskType.OCR].qsize() == 1

        manager.resume()
        await wait_for(lambda: search_db.ocr_fts("pic1") != [])
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    assert search_db.ocr_fts("pic1") == [(0, "", "paused")]


def test_reads_the_picture_without_holding_the_search_db_lock(
    search_db, tmp_path, monkeypatch
):
    """SPEC-ADDON-050 I5: decoding and the engine run outside the write lock."""
    path = picture(tmp_path / "pic.png")
    seed_indexed_file(search_db, "pic1", "image/png", path, **ALL_DONE)
    engine_harness = FakeEngine(("text",))
    engine_harness.on_call = lambda image: probe_write_outside_the_lock(search_db)
    harness = install(monkeypatch, engine=engine_harness)

    run_worker(harness, "pic1")

    assert search_db.scalar("SELECT COUNT(*) FROM ocr_lock_probe") == 1
    assert search_db.ocr_fts("pic1") == [(0, "", "text")]


def test_sends_nothing_over_the_network(search_db, tmp_path, monkeypatch):
    """SPEC-ADDON-050 I7: the engine runs in the container and fetches nothing."""
    path = picture(tmp_path / "pic.png")
    seed_indexed_file(search_db, "pic1", "image/png", path, **ALL_DONE)
    harness = install(monkeypatch, lines=["offline"])

    def blocked(*args, **kwargs):
        raise AssertionError("the OCR path opened a socket")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)

    run_worker(harness, "pic1")

    assert search_db.ocr_fts("pic1") == [(0, "", "offline")]
