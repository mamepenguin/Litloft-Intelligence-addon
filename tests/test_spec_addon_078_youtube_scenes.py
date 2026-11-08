"""SPEC-ADDON-078 item 7 and I9: ``GET /files/{file_id}/scenes`` for a YouTube ``.loft``."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from sqlalchemy import func, select

from app.models import Embedding, IndexedFile
from app.workers.video_visual_selection import Candidate
from tests.search_endpoint_db import patch_everywhere
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    DRIVE,
    frame_api,
    youtube,
)

DURATION = 600.0


def _unit_vec(seed: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _candidates(file_id: str) -> list[Candidate]:
    return [Candidate(f"{file_id}-{i}", i * 15.0, _unit_vec(i)) for i in range(40)]


def new_id() -> str:
    return "sc" + uuid.uuid4().hex[:10]


@pytest.fixture()
def rows(monkeypatch):
    served: dict[str, list[Candidate]] = {}

    def _load_candidates(file_id):
        return list(served.get(file_id, [])), DURATION

    patch_everywhere(monkeypatch, "_load_candidates", _load_candidates)
    patch_everywhere(monkeypatch, "_select_transcript_excerpt", lambda *a: "")
    return served


async def _scenes(frame_api, file_id, params=None, drive=DRIVE):
    return await frame_api.client.get(
        f"/files/{file_id}/scenes", params=params or {}, headers={"X-Lit-Drive": drive},
    )


def _comparable(body: dict) -> dict:
    return {k: v for k, v in body.items() if k != "file_id"}


class TestAYoutubeLoftIsServed:
    async def test_the_selection_is_the_one_a_native_video_gets_from_the_same_rows(
        self, frame_api, rows, youtube,
    ):
        """SPEC-ADDON-078 I9: the same selection over the same rows as a native video."""
        native, loft = new_id(), new_id()
        frame_api.add_row(native, frame_api.drive_dir / "n.mp4", mime="video/mp4", duration=DURATION)
        frame_api.add_loft(loft, duration=DURATION)
        rows[native] = _candidates("n")
        rows[loft] = _candidates("n")

        from_native = await _scenes(frame_api, native)
        from_loft = await _scenes(frame_api, loft)

        assert from_native.status_code == 200, from_native.text
        assert from_loft.status_code == 200, from_loft.text
        assert from_loft.json()["file_id"] == loft
        assert _comparable(from_loft.json()) == _comparable(from_native.json())
        assert from_loft.json()["scenes"]

    @pytest.mark.parametrize("params", [{"start": 150}, {"end": 300}, {"start": 150, "end": 450}])
    async def test_a_window_is_applied_as_for_a_native_video(
        self, params, frame_api, rows, youtube,
    ):
        """SPEC-ADDON-078 I9: the window selection is the native one."""
        native, loft = new_id(), new_id()
        frame_api.add_row(native, frame_api.drive_dir / "n.mp4", mime="video/mp4", duration=DURATION)
        frame_api.add_loft(loft, duration=DURATION)
        rows[native] = _candidates("n")
        rows[loft] = _candidates("n")

        from_native = await _scenes(frame_api, native, params)
        from_loft = await _scenes(frame_api, loft, params)

        assert from_loft.status_code == 200, from_loft.text
        assert _comparable(from_loft.json()) == _comparable(from_native.json())

    async def test_a_loft_without_scene_rows_has_an_empty_list(self, frame_api, rows, youtube):
        """SPEC-ADDON-078 failure table: ``scenes: []`` and ``candidate_count: 0``."""
        loft = new_id()
        frame_api.add_loft(loft, duration=DURATION)

        response = await _scenes(frame_api, loft)

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["scenes"] == []
        assert body["candidate_count"] == 0

    async def test_the_request_reaches_no_network_and_writes_nothing(
        self, frame_api, rows, youtube,
    ):
        """SPEC-ADDON-078 I9: no yt-dlp, no ffmpeg, no row written."""
        loft = new_id()
        frame_api.add_loft(loft, duration=DURATION)
        rows[loft] = _candidates("n")

        def counts():
            with frame_api.Session() as s:
                return (
                    s.execute(select(func.count()).select_from(Embedding)).scalar(),
                    s.execute(select(func.count()).select_from(IndexedFile)).scalar(),
                    s.query(IndexedFile).filter_by(file_id=loft).one().clip_indexed,
                )

        before = counts()
        response = await _scenes(frame_api, loft)

        assert response.status_code == 200, response.text
        assert youtube.lookups() == []
        assert youtube.ffmpeg_events() == []
        assert youtube.fetched("/") == []
        assert counts() == before


class TestOtherLofts:
    @pytest.mark.parametrize(
        "body",
        [
            {"provider": "vimeo", "url": "https://vimeo.com/1"},
            {"provider": "generic", "url": "https://example.com/v"},
        ],
        ids=["vimeo", "generic"],
    )
    async def test_a_loft_of_another_provider_is_not_a_video_file(
        self, body, frame_api, rows, youtube,
    ):
        """SPEC-ADDON-078 I9: 400 ``Not a video file`` for any other provider."""
        other, loft = new_id(), new_id()
        frame_api.add_loft(other, body)
        frame_api.add_loft(loft)
        rows[loft] = _candidates("n")

        refused = await _scenes(frame_api, other)
        served = await _scenes(frame_api, loft)

        assert served.status_code == 200, served.text
        assert refused.status_code == 400
        assert refused.json()["detail"] == "Not a video file"
