"""SPEC-ADDON-122 §3: the frame read of a YouTube ``.loft``'s download is a stored
read like a native one, so it writes rule 2 and the ``.loft`` is never outdated.
"""

from __future__ import annotations

import pytest

from tests.frame_box_doubles import SimpleBoxEngine, install
from tests.loft_media_world import LoftWorld, fetch
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.test_spec_addon_119_loft_fetch_counting import (  # noqa: F401 - fixtures
    VIDEO_SECONDS,
    fresh_worker_state,
    gradient,
)
from tests.vec_partition_db import search_index  # noqa: F401 - fixture
from tests.youtube_frame_doubles import needs_ffmpeg, youtube  # noqa: F401 - fixtures

pytestmark = [needs_ffmpeg, pytest.mark.usefixtures("pillow_real")]


@pytest.fixture()
def world(search_index, youtube, monkeypatch, tmp_path, make_settings):  # noqa: F811
    return LoftWorld(search_index, youtube, monkeypatch, tmp_path, make_settings)


def test_spec_addon_122_section_3_a_loft_read_is_stored_under_rule_two(
    world, gradient, monkeypatch,  # noqa: F811
):
    world.configure(frame_ocr="on_index")
    fake = SimpleBoxEngine()
    install(monkeypatch, fake)
    world.serve(gradient, duration=VIDEO_SECONDS)
    world.add_loft("ytloft000001", frame_scan_indexed=True,
                   frame_ocr_read=False, frame_ocr_indexed=False)

    fetch("ytloft000001")

    assert fake.recognises >= 1
    assert world.frame_texts("ytloft000001") != []
    assert world.flags("ytloft000001") == (True, True, True)
    assert world.value("ytloft000001", "frame_ocr_rule") == 2
