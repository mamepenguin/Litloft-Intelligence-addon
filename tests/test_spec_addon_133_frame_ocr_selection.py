"""SPEC-ADDON-133 item 1 step 2, I1: a later frame is read only 3 seconds or
more after the last read frame.

The picture flips between a left-to-right and a right-to-left ramp every second,
so every flip moves the difference hash far past both thresholds and only the
gap decides. The engine raises on every call, which makes one engine call per
read frame whatever steps the reader asks it for; a frame whose engine call
raised is still the last read frame.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.models import VideoFrameText  # noqa: F401 - registers the table
from tests.ocr_fixtures import (
    build_search_db,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    real_pillow,
    seed_indexed_file,
)
from tests.test_spec_addon_133_frame_ocr_reread import wire_reader

pytestmark = pytest.mark.usefixtures("pillow_real")

WIDTH, HEIGHT = 320, 240


def _encode_flips(path: Path, seconds: int) -> Path:
    real_pillow()
    from PIL import Image

    ramp = np.tile(np.linspace(0, 255, WIDTH, dtype=np.uint8), (HEIGHT, 1))
    pictures = [
        Image.fromarray(np.stack([ramp] * 3, axis=-1)),
        Image.fromarray(np.stack([ramp[:, ::-1]] * 3, axis=-1)),
    ]
    fps = 5
    process = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps),
            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-g", "5", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for index in range(seconds * fps):
        process.stdin.write(pictures[(index // fps) % 2].tobytes())
    process.stdin.close()
    assert process.wait() == 0
    return path


@pytest.fixture(scope="module")
def flips(tmp_path_factory) -> dict[int, Path]:
    saved = {
        name: module for name, module in sys.modules.items()
        if name == "PIL" or name.startswith("PIL.")
    }
    root = tmp_path_factory.mktemp("frame-ocr-flips")
    try:
        clips = {seconds: _encode_flips(root / f"flips-{seconds}.mp4", seconds)
                 for seconds in (3, 4, 7)}
    finally:
        for name in [n for n in sys.modules if n == "PIL" or n.startswith("PIL.")]:
            del sys.modules[name]
        sys.modules.update(saved)
    return clips


@pytest.fixture()
def reader(tmp_path, monkeypatch):
    db = build_search_db(tmp_path, monkeypatch)
    module, source = wire_reader(db, monkeypatch)
    return module, db, source.engine


def test_spec_addon_133_i1_a_frame_is_read_three_seconds_after_the_last_read_and_not_sooner(
    reader, flips,
):
    module, db, engine = reader
    reads = {}
    for seconds in (3, 4, 7):
        file_id = f"vid{seconds}"
        seed_indexed_file(db, file_id, "video/mp4", flips[seconds], file_type="video")
        before = engine.calls
        module.index_frame_ocr(file_id)
        reads[seconds] = engine.calls - before

    assert reads == {3: 1, 4: 2, 7: 3}
