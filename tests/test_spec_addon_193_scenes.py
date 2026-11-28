"""SPEC-ADDON-193 I10: ``GET /files/{id}/scenes`` for a web reference.

A ``web`` `.loft` is served as a YouTube one; an `other` `.loft` and an
unclassified reference of another provider are still not video files.
"""

from __future__ import annotations

import uuid

import pytest

from app.models import IndexedFile
from app.workers.video_visual_selection import SelectionInputs
from tests.search_endpoint_db import patch_everywhere
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    DRIVE,
    YOUTUBE_URL,
    frame_api,
    youtube,
)

WEB_BODY = {
    "provider": "example-video",
    "url": "https://videos.example-video.test/watch/abcdef123456",
}


@pytest.fixture(autouse=True)
def selection(monkeypatch):
    patch_everywhere(
        monkeypatch, "load_selection_inputs",
        lambda file_id: SelectionInputs(duration=None),
    )
    patch_everywhere(monkeypatch, "_select_transcript_excerpt", lambda *a: "")


def new_id() -> str:
    return "sc" + uuid.uuid4().hex[:10]


def _source(frame_api, file_id: str, source: str | None) -> None:
    with frame_api.Session() as session:
        session.query(IndexedFile).filter_by(file_id=file_id).update(
            {"loft_source": source}
        )
        session.commit()


async def _scenes(frame_api, file_id):
    return await frame_api.client.get(
        f"/files/{file_id}/scenes", headers={"X-Lit-Drive": DRIVE},
    )


async def test_spec_addon_193_i10_scenes_of_a_web_loft_are_served_without_a_network_call(
    frame_api, youtube,
):
    file_id = new_id()
    frame_api.add_loft(file_id, WEB_BODY, duration=600.0)
    _source(frame_api, file_id, "web")

    response = await _scenes(frame_api, file_id)

    assert response.status_code == 200, response.text
    assert response.json()["file_id"] == file_id
    assert response.json()["scenes"] == []
    assert youtube.lookups() == []
    assert youtube.fetched("/") == []
    assert youtube.ffmpeg_events() == []


async def test_spec_addon_193_i10_scenes_of_an_other_loft_are_not_a_video_file(
    frame_api, youtube,
):
    file_id = new_id()
    frame_api.add_loft(file_id, WEB_BODY)
    _source(frame_api, file_id, "other")

    response = await _scenes(frame_api, file_id)

    assert response.status_code == 400
    assert response.json()["detail"] == "Not a video file"


async def test_spec_addon_193_i10_scenes_of_an_unclassified_web_reference_are_not_a_video_file(
    frame_api, youtube,
):
    file_id = new_id()
    frame_api.add_loft(file_id, WEB_BODY)

    response = await _scenes(frame_api, file_id)

    assert response.status_code == 400
    assert response.json()["detail"] == "Not a video file"


async def test_spec_addon_193_i10_scenes_of_an_unclassified_youtube_reference_are_served(
    frame_api, youtube,
):
    file_id = new_id()
    frame_api.add_loft(file_id, {"provider": "youtube", "url": YOUTUBE_URL})

    response = await _scenes(frame_api, file_id)

    assert response.status_code == 200, response.text
    assert response.json()["scenes"] == []
