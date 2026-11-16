"""SPEC-ADDON-117 item 1.6, I9 (SPEC-ADDON-079 I1 revised): ``scene-tile`` for a
``.loft`` with no storyboard tile at that time serves the ``small`` frame cache
entry kept by its fetch.
"""

from __future__ import annotations

import subprocess

import httpx

from app.frame_cache import frame_cache_path
from tests.test_spec_addon_079_scene_tiles import (
    TILE_SUBDIR,
    new_id,
    scene_tile,
    tiles_dir,
    tree,
)
from tests.youtube_frame_doubles import frame_api  # noqa: F401 - fixture


def _small(file_id: str, t: float, data: bytes) -> None:
    path = frame_cache_path(file_id, t)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _tile(file_id: str, ms: int, data: bytes) -> None:
    tiles_dir(file_id).mkdir(parents=True, exist_ok=True)
    (tiles_dir(file_id) / f"{ms}.webp").write_bytes(data)


async def test_spec_addon_117_i9_a_loft_with_no_tile_serves_the_small_picture_at_that_time(
    frame_api,
):
    loft = new_id()
    frame_api.add_loft(loft)
    _small(loft, 10.0, b"SMALL-AT-10000")
    native = new_id()
    path = frame_api.drive_dir / f"{native}.mp4"
    path.write_bytes(b"")
    frame_api.add_row(native, path, mime="video/mp4")
    _small(native, 10.0, b"NATIVE-SMALL")

    loft_response = await scene_tile(frame_api, loft, 10.0)
    native_response = await scene_tile(frame_api, native, 10.0)
    other_time = await scene_tile(frame_api, loft, 11.0)

    assert loft_response.status_code == 200
    assert loft_response.headers["content-type"].startswith("image/webp")
    assert loft_response.content == b"SMALL-AT-10000"
    assert native_response.status_code == 404
    assert other_time.status_code == 404


async def test_spec_addon_117_i9_a_storyboard_tile_wins_over_the_small_picture(frame_api):
    loft = new_id()
    frame_api.add_loft(loft)
    _tile(loft, 10000, b"TILE-AT-10000")
    _small(loft, 10.0, b"SMALL-AT-10000")
    _small(loft, 20.0, b"SMALL-AT-20000")

    at_tile = await scene_tile(frame_api, loft, 10.0)
    beside_tile = await scene_tile(frame_api, loft, 20.0)

    assert at_tile.content == b"TILE-AT-10000"
    assert beside_tile.status_code == 200
    assert beside_tile.content == b"SMALL-AT-20000"


async def test_spec_addon_117_i9_the_small_fallback_never_calls_out_runs_a_process_or_writes(
    frame_api, monkeypatch,
):
    loft = new_id()
    frame_api.add_loft(loft)
    _small(loft, 10.0, b"SMALL")
    (tiles_dir(loft).parent / TILE_SUBDIR).mkdir(parents=True, exist_ok=True)

    def refuse(*args, **kwargs):
        raise AssertionError("the scene-tile route reached outside its files")

    for module, name in (
        (httpx, "get"), (httpx, "request"), (httpx, "post"),
        (subprocess, "run"), (subprocess, "Popen"), (subprocess, "check_output"),
    ):
        monkeypatch.setattr(module, name, refuse)
    before = tree(frame_api.data_dir)

    response = await scene_tile(frame_api, loft, 10.0)
    missing = await scene_tile(frame_api, loft, 30.0)

    assert response.status_code == 200
    assert missing.status_code == 404
    assert tree(frame_api.data_dir) == before

