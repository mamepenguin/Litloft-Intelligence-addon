"""SPEC-ADDON-113 item 5, SPEC-ADDON-108 I3 as revised: a question about a video
with no text sends the frame times of the selection's Ask list.

The selection reads the stored rows; nothing between them and the frames sent is
replaced.
"""

from __future__ import annotations

from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    addon,
    ask,
    ask_slots,
    by_kind,
    call,
    db,
    llm,
    vision_client,
)
from tests.scene_sections_db import Video, basis, mix
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import FFmpeg, ms, seed_text_less_video, sent_keys

DROPPED = "vid000000113"
KEPT = "vid000000114"


def _video(db, search_index, tmp_path, file_id, third_cos):
    seed_text_less_video(db, file_id, path=tmp_path / f"{file_id}.mp4", duration=80.0)
    video = Video(search_index, file_id, duration=80.0)
    video.clip(10.0, basis(video.clip_dim, 0))
    video.clip(40.0, basis(video.clip_dim, 1))
    video.clip(70.0, mix(video.clip_dim, third_cos, 0, 2))
    video.commit(with_file=False)


async def test_spec_addon_113_the_ask_list_drops_a_section_showing_what_another_shows(
    db, search_index, addon, llm, call, tmp_path, monkeypatch,
):
    """SPEC-ADDON-112 item 9, SPEC-ADDON-108 I3: 0.95 against the sections kept."""
    _video(db, search_index, tmp_path, DROPPED, 0.96)
    _video(db, search_index, tmp_path, KEPT, 0.94)
    FFmpeg().install(monkeypatch)
    _, wire = vision_client(llm)

    dropped = await ask(call, DROPPED)
    kept = await ask(call, KEPT)

    assert by_kind(dropped)["meta"]["image"] == "attached"
    assert by_kind(kept)["meta"]["image"] == "attached"
    keys = sent_keys(wire)
    assert keys[:8] == [ms(t) for t in [5.0, 10.0, 15.0, 25.0, 35.0, 40.0, 45.0, 55.0]]
    assert keys[8:] == [ms(t) for t in [5.0, 10.0, 15.0, 25.0, 35.0, 40.0, 45.0, 70.0]]
