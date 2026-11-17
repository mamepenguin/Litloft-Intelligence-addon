"""SPEC-ADDON-122: what a frame-OCR task does with a read stored under the old
rule, and what a re-read that does not complete leaves behind.

Item 1 step 8, §2, §3, I6, I7, I8. The task's work is ``index_frame_ocr``, the
reader SPEC-ADDON-066 names. The engine here is never one that answers: every
case below ends before a frame's text could be stored, so the outcome does not
depend on which engine steps the reader calls.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from app.models import VideoFrameText  # noqa: F401 - registers the table
from tests.ocr_fixtures import (
    build_search_db,
    patch_features,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    seed_indexed_file,
    seed_ocr_rows,
)
from tests.test_frame_page_ocr_readers import media  # noqa: F401 - fixture

pytestmark = pytest.mark.usefixtures("pillow_real")

EARLIER = (
    (0, 0.0, 10.0, "EARLIER ONE"),
    (1, 10.0, 13.0, "EARLIER TWO"),
)


class EngineDown(RuntimeError):
    pass


class RaisingEngine:
    """Raises on any call, through any attribute path, and counts the calls."""

    def __init__(self, counter: list | None = None) -> None:
        self._counter = counter if counter is not None else []

    @property
    def calls(self) -> int:
        return len(self._counter)

    def __getattr__(self, _name):
        return RaisingEngine(self._counter)

    def __call__(self, *_args, **_kwargs):
        self._counter.append(1)
        raise EngineDown("engine call failed")


class EngineSource:
    """The reader's ``_get_engine``: counts how often it is asked."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.asked = 0

    def __call__(self):
        self.asked += 1
        return self.engine


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    return build_search_db(tmp_path, monkeypatch)


def wire_reader(search_db, monkeypatch):
    """The frame reader over ``search_db``, with an engine that raises on every call."""
    import app.workers.frame_ocr as module

    @contextmanager
    def _write():
        session = search_db.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def _read():
        session = search_db.Session()
        try:
            yield session
        finally:
            session.rollback()
            session.close()

    monkeypatch.setattr(module, "get_search_db", _write)
    monkeypatch.setattr(module, "get_search_db_read", _read)
    monkeypatch.setattr(module, "embed_passages", lambda texts: [
        np.zeros(4, dtype="float32") for _ in texts
    ])
    patch_features(monkeypatch, video_frame_ocr="on_index")
    source = EngineSource(RaisingEngine())
    monkeypatch.setattr(module, "_get_engine", source)
    return module, source


@pytest.fixture()
def reader(search_db, monkeypatch):
    return wire_reader(search_db, monkeypatch)


def _seed_earlier_read(db, path: Path, file_id: str = "vid1", **flags) -> None:
    values = dict(frame_ocr_read=True, frame_ocr_indexed=True)
    values.update(flags)
    seed_indexed_file(db, file_id, "video/mp4", path, file_type="video", **values)
    for chunk, start, end, line in EARLIER:
        db.execute(
            "INSERT INTO video_frame_texts"
            "(file_id, chunk_index, start_time, end_time, text) "
            "VALUES(:f, :c, :s, :e, :t)",
            f=file_id, c=chunk, s=start, e=end, t=line,
        )
    seed_ocr_rows(db, file_id, tuple(line for *_, line in EARLIER))


def _snapshot(db, file_id: str = "vid1") -> dict:
    return {
        "frames": db.rows(
            "SELECT chunk_index, start_time, end_time, text FROM video_frame_texts "
            "WHERE file_id = :f ORDER BY chunk_index", f=file_id,
        ),
        "embeddings": db.ocr_embeddings(file_id),
        "fts": db.ocr_fts(file_id),
        "read": bool(db.flag(file_id, "frame_ocr_read")),
        "indexed": bool(db.flag(file_id, "frame_ocr_indexed")),
    }


async def _queued_again() -> list[tuple[str, int]]:
    from app.indexer import IndexManager, TaskType

    manager = IndexManager()
    await manager._resume_incomplete()
    return [
        (t.file_id, t.priority)
        for t in await manager._collect_batch(TaskType.FRAME_OCR, 10)
    ]


class _NoShowinfoProcess:
    """An ffmpeg run whose frame carries no time (``times_failed``)."""

    returncode = 0

    def __init__(self, frame_bytes: int):
        self.stdout = tempfile.TemporaryFile()
        self.stderr = tempfile.TemporaryFile()
        self.stdout.write(b"\x00" * frame_bytes)
        self.stdout.seek(0)
        self.stderr.write(b"[Parsed_showinfo_0 @ 0x1] n:0 pts:0 pts_time:N/A\n")
        self.stderr.seek(0)

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        return None


def _make_times_fail(module, monkeypatch, path: Path) -> None:
    width, height = module._probe(str(path))
    real_popen = module.subprocess.Popen

    def fake_popen(command, *args, **kwargs):
        if command and "ffprobe" in str(command[0]):
            return real_popen(command, *args, **kwargs)
        return _NoShowinfoProcess(width * height * 3)

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)


# ---------------------------------------------------------------------------
# §2, I8 - a re-read that is not complete keeps the earlier read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["unreadable", "times-failed"])
def test_spec_addon_122_i8_an_incomplete_reread_keeps_the_earlier_read_and_sets_rule_two(
    reader, search_db, media, monkeypatch, outcome,  # noqa: F811
):
    module, _source = reader
    clip = {
        "unreadable": media["corrupt"],
        "times-failed": media["large_trim"],
    }[outcome]
    _seed_earlier_read(search_db, clip)
    if outcome == "times-failed":
        _make_times_fail(module, monkeypatch, clip)
    before = _snapshot(search_db)

    module.index_frame_ocr("vid1")

    assert _snapshot(search_db) == before
    assert before["frames"] == [tuple(row) for row in EARLIER]
    assert search_db.flag("vid1", "frame_ocr_rule") == 2


async def test_spec_addon_122_i8_an_incomplete_reread_is_not_queued_again(
    reader, search_db, media,  # noqa: F811
):
    module, _source = reader
    _seed_earlier_read(search_db, media["corrupt"])
    assert await _queued_again() == [("vid1", -10)]

    module.index_frame_ocr("vid1")

    assert await _queued_again() == []
    assert search_db.flag("vid1", "frame_ocr_rule") == 2


def test_spec_addon_122_section_8_an_incomplete_reread_logs_one_warning_and_no_first_read_line(
    reader, search_db, media, caplog,  # noqa: F811
):
    module, _source = reader
    _seed_earlier_read(search_db, media["corrupt"])

    with caplog.at_level("INFO"):
        module.index_frame_ocr("vid1")

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert len(warnings) == 1, warnings
    assert "closing it as read" not in logged
    assert "keeping the frames read" not in logged


def test_spec_addon_122_i8_an_engine_raising_on_every_frame_was_asked_and_still_keeps_the_read(
    reader, search_db, media,  # noqa: F811
):
    module, source = reader
    _seed_earlier_read(search_db, media["landscape"])
    before = _snapshot(search_db)

    module.index_frame_ocr("vid1")

    assert source.asked >= 1
    assert source.engine.calls >= 1
    assert _snapshot(search_db) == before
    assert search_db.flag("vid1", "frame_ocr_rule") == 2


# ---------------------------------------------------------------------------
# §2, I8 - a re-read that changes nothing and is tried again
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["unreachable", "no-engine"])
async def test_spec_addon_122_i8_a_reread_that_cannot_start_changes_nothing_and_is_tried_again(
    reader, search_db, media, tmp_path, monkeypatch, outcome,  # noqa: F811
):
    module, _source = reader
    if outcome == "unreachable":
        _seed_earlier_read(search_db, tmp_path / "gone.mp4")
    else:
        _seed_earlier_read(search_db, media["landscape"])
        monkeypatch.setattr(module, "_get_engine", lambda: None)
    before = _snapshot(search_db)

    module.index_frame_ocr("vid1")

    assert _snapshot(search_db) == before
    assert search_db.flag("vid1", "frame_ocr_rule") is None
    assert await _queued_again() == [("vid1", -10)]


# ---------------------------------------------------------------------------
# Item 1 step 8 - a read done under the new rule is closed without reading
# ---------------------------------------------------------------------------


def test_spec_addon_122_item_8_a_rule_two_read_is_closed_without_asking_for_the_engine(
    reader, search_db, media,  # noqa: F811
):
    module, source = reader
    _seed_earlier_read(search_db, media["landscape"], frame_ocr_rule=2)
    before = _snapshot(search_db)

    module.index_frame_ocr("vid1")

    assert source.asked == 0
    assert source.engine.calls == 0
    assert _snapshot(search_db) == before
    assert search_db.flag("vid1", "frame_ocr_rule") == 2


# ---------------------------------------------------------------------------
# Item 1 step 8, §3, I6, I7 - rebuild only, whatever the rule holds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", [None, 2])
def test_spec_addon_122_i7_a_rebuild_calls_no_engine_and_leaves_the_rule(
    reader, search_db, media, rule,  # noqa: F811
):
    module, source = reader
    flags = dict(frame_ocr_indexed=False)
    if rule is not None:
        flags["frame_ocr_rule"] = rule
    _seed_earlier_read(search_db, media["landscape"], **flags)

    module.index_frame_ocr("vid1")

    assert source.engine.calls == 0
    assert search_db.flag("vid1", "frame_ocr_rule") == rule
    assert bool(search_db.flag("vid1", "frame_ocr_read")) is True
    assert bool(search_db.flag("vid1", "frame_ocr_indexed")) is True
    assert [row[3] for row in search_db.rows(
        "SELECT chunk_index, start_time, end_time, text FROM video_frame_texts "
        "WHERE file_id = 'vid1' ORDER BY chunk_index"
    )] == [line for *_, line in EARLIER]
