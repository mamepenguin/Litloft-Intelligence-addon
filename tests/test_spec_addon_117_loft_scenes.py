"""SPEC-ADDON-117 item 1.5: a YouTube ``.loft``'s change frames join the
representative-frame selection, read through ``GET /files/{id}/scenes``.
"""

from __future__ import annotations

from tests.scene_sections_db import Video, basis
from tests.search_endpoint_db import DRIVE, api  # noqa: F401 - fixture
from tests.vec_partition_db import search_index  # noqa: F401 - fixture

LOFT_MIME = "application/vnd.litloft.loft+json"


def _sections(api, file_id):
    response = api(f"/files/{file_id}/scenes", {}, drive=DRIVE)
    assert response.status_code == 200, response.text
    return [(s["start"], s["end"], s.get("frame_time")) for s in response.json()["scenes"]]


def test_spec_addon_117_a_youtube_lofts_change_frames_give_the_sections_a_native_videos_do(
    api, search_index,  # noqa: F811
):
    for file_id, mime in (("native", "video/mp4"), ("ytloft", LOFT_MIME)):
        video = Video(search_index, file_id, mime=mime)
        for n, t in enumerate([100.0, 300.0, 500.0]):
            video.change(t, basis(video.clip_dim, n))
        video.commit()
    search_index.sql("UPDATE indexed_files SET loft_source = 'youtube' WHERE file_id = 'ytloft'")

    native = _sections(api, "native")

    assert native == [(0.0, 300.0, 100.0), (300.0, 500.0, 300.0), (500.0, 600.0, 500.0)]
    assert _sections(api, "ytloft") == native
