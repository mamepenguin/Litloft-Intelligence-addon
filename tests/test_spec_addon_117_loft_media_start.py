"""SPEC-ADDON-117 item 1.1, §2 (restart during a fetch), I10: what a start does
before the first sweep can queue a fetch.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.config import FeaturesConfig
from tests.ocr_fixtures import (
    SearchDB,
    build_search_db,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    seed_indexed_file,
)
from tests.search_endpoint_db import patch_everywhere

pytestmark = pytest.mark.usefixtures("pillow_real")

LOFT_MIME = "application/vnd.litloft.loft+json"
CLOSED = dict(frame_scan_indexed=True, frame_ocr_read=True, frame_ocr_indexed=True)
FLAGS = tuple(CLOSED)


@pytest.fixture()
def db(tmp_path, monkeypatch) -> SearchDB:
    return build_search_db(tmp_path, monkeypatch)


@pytest.fixture()
def settings(monkeypatch, make_settings):
    value = make_settings(features=FeaturesConfig(video_frame_ocr="on_index"))  # type: ignore[call-arg]
    patch_everywhere(monkeypatch, "settings", value)
    return value


def _loft(db, tmp_path, file_id: str, source: str | None) -> None:
    path = tmp_path / f"{file_id}.loft"
    path.write_text("{}")
    seed_indexed_file(db, file_id, LOFT_MIME, path, file_type="video", **CLOSED)
    db.execute(
        "UPDATE indexed_files SET loft_source = :s WHERE file_id = :f", s=source, f=file_id,
    )


def _flags(db, file_id: str) -> tuple:
    return tuple(bool(db.flag(file_id, f)) for f in FLAGS)


async def _start(manager):
    stop = asyncio.Event()

    async def idle(*_a, **_k):
        await stop.wait()

    for name in dir(type(manager)):
        if name.endswith("_worker") and inspect.iscoroutinefunction(getattr(type(manager), name)):
            setattr(manager, name, idle)
    manager.reconcile = AsyncMock(return_value={})
    await asyncio.wait_for(manager.start(), timeout=5)
    return stop


async def test_spec_addon_117_i10_a_start_reopens_unclassified_lofts_with_nothing_stored(
    db, settings, tmp_path,
):
    """Revised SPEC-ADDON-146 I8: every YouTube or unclassified `.loft` whose
    candidates are not settled is queued for the backfill fetch too."""
    from app.indexer import IndexManager, TaskType

    _loft(db, tmp_path, "fresh", None)
    _loft(db, tmp_path, "scanned", None)
    db.execute(
        "INSERT INTO video_frame_scans(file_id, version, seconds, hashes, sizes, created_at) "
        "VALUES ('scanned', 1, 1, :h, :s, :now)",
        h=b"\0" * 32, s=b"\0\0", now=datetime.now(UTC),
    )
    _loft(db, tmp_path, "read", None)
    db.execute(
        "INSERT INTO video_frame_texts(file_id, chunk_index, start_time, end_time, text) "
        "VALUES ('read', 0, 0.0, 10.0, 'kept')"
    )
    _loft(db, tmp_path, "youtube", "youtube")
    _loft(db, tmp_path, "other", "other")
    seed_indexed_file(
        db, "native", "video/mp4", tmp_path / "native.mp4", file_type="video", **CLOSED,
    )
    manager = IndexManager()

    stop = await _start(manager)
    try:
        reopened = _flags(db, "fresh")
        untouched = {f: _flags(db, f) for f in ("scanned", "read", "youtube", "other", "native")}
        await manager._resume_incomplete()
        queued = [
            (t.file_id, t.priority)
            for t in await manager._collect_batch(TaskType("loft_media"), 10)
        ]
    finally:
        stop.set()
        await manager.stop()

    assert reopened == (False, False, False)
    assert untouched == {f: (True, True, True) for f in untouched}
    assert queued == [
        ("fresh", -10), ("scanned", -10), ("read", -10), ("youtube", -10),
    ]


async def test_spec_addon_117_i2_a_start_empties_the_fetch_directory(db, settings, tmp_path):
    from app.indexer import IndexManager

    fetch_dir = Path(settings.intelligence_data_dir) / "loft_media"
    fetch_dir.mkdir(parents=True)
    (fetch_dir / "abc.part").write_bytes(b"left by a killed fetch")
    (fetch_dir / "def.mkv").write_bytes(b"left by a killed fetch")
    kept = Path(settings.intelligence_data_dir) / "frames"
    kept.mkdir(parents=True)
    (kept / "keep.webp").write_bytes(b"x")
    manager = IndexManager()

    stop = await _start(manager)
    try:
        left = sorted(p.name for p in fetch_dir.iterdir()) if fetch_dir.exists() else []
    finally:
        stop.set()
        await manager.stop()

    assert left == []
    assert (kept / "keep.webp").exists()
