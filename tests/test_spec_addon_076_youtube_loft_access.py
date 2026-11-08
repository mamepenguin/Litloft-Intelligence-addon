"""A ``.loft`` the addon may not or cannot read, on the frame and scenes routes.

The frame route answers as the native one does when the path is outside the
drive (404 ``Video file not accessible``) and 400 ``No frame source`` when the
file cannot be read; the scenes route answers 400 ``Not a video file`` for both,
as it did for every ``.loft`` before.
"""

from __future__ import annotations

import uuid

import pytest

from tests.search_endpoint_db import patch_everywhere
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    DRIVE,
    describe,
    frame_api,
    needs_ffmpeg,
    youtube,
)

pytestmark = needs_ffmpeg


def new_id() -> str:
    return "ac" + uuid.uuid4().hex[:10]


async def _scenes(frame_api, file_id):
    return await frame_api.client.get(
        f"/files/{file_id}/scenes", headers={"X-Lit-Drive": DRIVE},
    )


async def test_a_loft_outside_the_allowed_directories_is_not_read(
    youtube, frame_api, monkeypatch,
):
    """Frame: 404 like a native video whose path fails validation; no lookup runs."""
    youtube.set_ytdlp(describe=describe([youtube.format("f720", 720)]))
    fid = new_id()
    frame_api.add_loft(fid)
    patch_everywhere(monkeypatch, "validate_file_path", lambda _path: False)

    response = await frame_api.frame(fid, 1.0)

    assert response.status_code == 404
    assert response.json()["detail"] == "Video file not accessible"
    assert youtube.lookups() == []
    assert frame_api.cached(fid) == []


async def test_scenes_of_a_loft_outside_the_allowed_directories_is_not_a_video_file(
    youtube, frame_api, monkeypatch,
):
    fid = new_id()
    frame_api.add_loft(fid)
    patch_everywhere(monkeypatch, "validate_file_path", lambda _path: False)

    response = await _scenes(frame_api, fid)

    assert response.status_code == 400
    assert response.json()["detail"] == "Not a video file"


@pytest.mark.parametrize("how", ["missing", "not-json"])
async def test_scenes_of_an_unreadable_loft_is_not_a_video_file(how, youtube, frame_api):
    fid = new_id()
    path = frame_api.add_loft(fid, "not json at all" if how == "not-json" else None)
    if how == "missing":
        path.unlink()

    response = await _scenes(frame_api, fid)

    assert response.status_code == 400
    assert response.json()["detail"] == "Not a video file"
