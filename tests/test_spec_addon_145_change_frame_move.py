"""SPEC-ADDON-145 item 3-4, §2, I3 and I4: at start-up, stored change frames move into
the scene candidates of their file, keeping their ids, so no file's selection pool
changes.

The move is driven through the addon's lifespan, as a start runs it.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time

import numpy as np
import pytest
from sqlalchemy import text

from tests.scene_sections_db import Video, basis
from tests.search_endpoint_db import DRIVE, api, patch_everywhere  # noqa: F401 - fixture
from tests.vec_partition_db import search_index  # noqa: F401 - fixture

LOFT_MIME = "application/vnd.litloft.loft+json"
GIB = 1024 ** 3

# vid: pool candidates at 0, 100 and 200 s. 96.9 is 3.1 s from one, 203 exactly 3 s.
MOVED = {"chg_vid_50": 50.0, "chg_vid_96": 96.9, "chg_vid_300": 300.0, "chg_vid_401": 401.0,
         "chg_loft_40": 40.0}
DELETED = {"chg_vid_1", "chg_vid_102", "chg_vid_203"}


async def _until(predicate, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return predicate()


@pytest.fixture()
def lifespan(search_index, monkeypatch):  # noqa: F811
    from app import main

    monkeypatch.setattr(main, "init_litloft_db", lambda: None)
    return main


class CacheClears:
    def __init__(self, monkeypatch):
        self.count = 0
        patch_everywhere(monkeypatch, "invalidate_similar_cache", self)

    def __call__(self, *_args, **_kwargs):
        self.count += 1


def _seed(index) -> dict[str, np.ndarray]:
    vid = Video(index, "vid")
    dim = vid.clip_dim
    vid.clip(0.0, basis(dim, 0)).clip(100.0, basis(dim, 1)).clip(200.0, basis(dim, 2))
    vectors = {
        "chg_vid_50": basis(dim, 10), "chg_vid_96": basis(dim, 11),
        "chg_vid_300": basis(dim, 12), "chg_vid_401": basis(dim, 13),
        "chg_vid_102": basis(dim, 14), "chg_vid_203": basis(dim, 15),
        "chg_loft_40": basis(dim, 16), "chg_vid_1": basis(dim, 17),
    }
    for emb_id, t in MOVED.items():
        if emb_id.startswith("chg_vid_"):
            vid.change(t, vectors[emb_id])
    vid.change(1.0, vectors["chg_vid_1"])
    vid.change(102.0, vectors["chg_vid_102"])
    vid.change(203.0, vectors["chg_vid_203"])
    vid.commit()

    loft = Video(index, "loft", mime=LOFT_MIME)
    loft.clip(10.0, basis(dim, 3)).change(40.0, vectors["chg_loft_40"]).commit()
    index.sql("UPDATE indexed_files SET loft_source = 'youtube' WHERE file_id = 'loft'")

    untouched = Video(index, "plain")
    untouched.clip(5.0, basis(dim, 4)).commit()
    return vectors


def _rows(index, file_id: str) -> list[tuple]:
    with index.engine.connect() as conn:
        return [tuple(r) for r in conn.execute(text(
            "SELECT id, embedding_type, vector_table, timestamp_start, timestamp_end, "
            "content_preview FROM embeddings WHERE file_id = :f ORDER BY id"
        ), {"f": file_id})]


def _change_rows(index) -> list[str]:
    with index.engine.connect() as conn:
        return [r[0] for r in conn.execute(text(
            "SELECT id FROM embeddings WHERE embedding_type = 'clip_change' ORDER BY id"
        ))]


def _pool_view(api, file_id: str):  # noqa: F811
    response = api(f"/files/{file_id}/scenes", {}, drive=DRIVE)
    assert response.status_code == 200, response.text
    body = response.json()
    return (
        body.get("candidate_count"),
        [
            {k: v for k, v in scene.items() if "type" not in k}
            for scene in body["scenes"]
        ],
    )


async def test_spec_addon_145_i3_far_change_frames_become_scene_candidates_and_the_rest_go(
    lifespan, search_index, api, monkeypatch, caplog,  # noqa: F811
):
    vectors = _seed(search_index)
    before = {f: _pool_view(api, f) for f in ("vid", "loft", "plain")}
    plain_before = _rows(search_index, "plain")
    clears = CacheClears(monkeypatch)
    caplog.set_level(logging.INFO)

    async with lifespan.lifespan(lifespan.app):
        moved = await _until(lambda: _change_rows(search_index) == [])
        await asyncio.sleep(0.5)

    assert moved, f"change frames left after start-up: {_change_rows(search_index)}"
    clip_rows = {
        r[0]: r for f in ("vid", "loft") for r in _rows(search_index, f) if r[1] == "clip"
    }
    for emb_id, t in MOVED.items():
        assert clip_rows[emb_id][2:] == ("vec_clip", pytest.approx(t),
                                         pytest.approx(t + 30), "x"), emb_id
    assert not DELETED & {r[0] for f in ("vid",) for r in _rows(search_index, f)}
    assert search_index.vectors("vec_clip_change") == {}
    keyed = search_index.keyed("vec_clip")
    for emb_id in MOVED:
        assert keyed[emb_id] == ("clip", vectors[emb_id].astype(np.float32).tobytes())
        assert search_index.knn("vec_clip", vectors[emb_id], 1, "clip") == [emb_id]
    assert _rows(search_index, "plain") == plain_before
    assert {f: _pool_view(api, f) for f in ("vid", "loft", "plain")} == before
    assert clears.count == 1

    counts = [{int(n) for n in re.findall(r"\d+", r.getMessage())} for r in caplog.records]
    assert any({2, 5, 3} <= found for found in counts), (
        "no line carries the files moved (2), frames moved (5) and frames deleted (3)"
    )

    async with lifespan.lifespan(lifespan.app):
        await asyncio.sleep(1.0)

    assert clears.count == 1


async def test_spec_addon_145_section_2_a_file_whose_move_fails_keeps_everything_and_the_next_start_moves_it(
    lifespan, search_index, api, monkeypatch, caplog,  # noqa: F811
):
    _seed(search_index)
    bad = Video(search_index, "bad")
    bad.clip(0.0, basis(bad.clip_dim, 5)).change(60.0, basis(bad.clip_dim, 6))
    bad.change(1.0, basis(bad.clip_dim, 7)).commit()
    bad_before = _rows(search_index, "bad")
    bad_vectors = {
        k: v for k, v in search_index.keyed("vec_clip_change").items() if k.startswith("chg_bad")
    }
    for event in ("UPDATE", "DELETE", "INSERT"):
        row = "NEW" if event != "DELETE" else "OLD"
        search_index.sql(
            f"CREATE TRIGGER refuse_bad_{event.lower()} BEFORE {event} ON embeddings "
            f"WHEN {row}.file_id = 'bad' BEGIN SELECT RAISE(ABORT, 'refused'); END"
        )
    caplog.set_level(logging.INFO)

    async with lifespan.lifespan(lifespan.app):
        moved_others = await _until(
            lambda: _change_rows(search_index) == ["chg_bad_1", "chg_bad_60"]
        )
        logged = await _until(lambda: any(
            "bad" in r.getMessage() and "IntegrityError" in r.getMessage()
            for r in caplog.records
        ))

    assert moved_others, f"change frames after the first start: {_change_rows(search_index)}"
    assert logged, "no line names the failed file and its exception class"
    assert _rows(search_index, "bad") == bad_before
    assert {
        k: v for k, v in search_index.keyed("vec_clip_change").items() if k.startswith("chg_bad")
    } == bad_vectors

    for event in ("update", "delete", "insert"):
        search_index.sql(f"DROP TRIGGER refuse_bad_{event}")
    async with lifespan.lifespan(lifespan.app):
        retried = await _until(lambda: _change_rows(search_index) == [])

    assert retried, f"change frames after the second start: {_change_rows(search_index)}"
    bad_after = {r[0]: r[1] for r in _rows(search_index, "bad")}
    assert bad_after == {"clip_bad_1": "clip", "chg_bad_60": "clip"}


async def test_spec_addon_145_i4_a_start_short_of_space_moves_nothing_and_clears_no_cache(
    lifespan, search_index, monkeypatch,  # noqa: F811
):
    _seed(search_index)
    rows_before = {f: _rows(search_index, f) for f in ("vid", "loft")}
    usage = shutil.disk_usage("/")
    clears = CacheClears(monkeypatch)

    asked: list[str] = []

    def no_space(path):
        asked.append(str(path))
        return type(usage)(usage.total, usage.total, 0)

    monkeypatch.setattr(shutil, "disk_usage", no_space)
    async with lifespan.lifespan(lifespan.app):
        await _until(lambda: asked, 10.0)
        await asyncio.sleep(0.5)

    assert {f: _rows(search_index, f) for f in ("vid", "loft")} == rows_before
    assert clears.count == 0

    monkeypatch.setattr(
        shutil, "disk_usage", lambda _p: type(usage)(usage.total, 0, 100 * GIB),
    )
    async with lifespan.lifespan(lifespan.app):
        moved = await _until(lambda: _change_rows(search_index) == [])

    assert moved
    assert clears.count == 1
