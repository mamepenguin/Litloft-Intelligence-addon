"""SPEC-ADDON-133 I1: the frame reader's 8-bit rule for a later frame.

The clip's one change moves the difference hash by exactly 8 bits, the
smallest difference this reader reads. The frame scan's 16 bits, or 9, would
read the first frame only.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.models import VideoFrameText  # noqa: F401 - registers the table
from tests.ocr_fixtures import pillow_real, real_pillow  # noqa: F401 - usefixtures
from tests.test_frame_page_ocr_readers import (  # noqa: F401 - fixtures and helpers
    _frame_rows,
    _video,
    frame_reader,
    search_db,
)

pytestmark = pytest.mark.usefixtures("pillow_real")

WIDTH, HEIGHT = 340, 320
SECONDS = 9
CHANGE_SECOND = 4
BLOCK = (60, 120, 60, 140)
MOVED_BLOCK = (100, 160, 60, 140)
BOX = ((50, 50), (170, 50), (170, 150), (50, 150))


def _encode(path: Path) -> Path:
    """A black clip with a white block, which moves 40 px to the right at 4 s."""
    real_pillow()
    from PIL import Image, ImageDraw

    fps = 5
    process = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps),
            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-g", "10", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for index in range(SECONDS * fps):
        left, right, top, bottom = (
            BLOCK if index / fps < CHANGE_SECOND else MOVED_BLOCK
        )
        frame = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
        ImageDraw.Draw(frame).rectangle(
            (left, top, right - 1, bottom - 1), fill=(255, 255, 255),
        )
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    assert process.wait() == 0
    return path


def _hash_distance(module, clip: Path) -> int:
    """The reader's own difference hash of the first frame and the changed one."""
    from PIL import Image

    frame_size = WIDTH * HEIGHT * 3
    decoded = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(clip), "-map", "0:v:0",
            "-vf", "fps=1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ],
        check=True,
        capture_output=True,
    ).stdout

    def hashed(second: int):
        start = second * frame_size
        return module._dhash(Image.frombytes(
            "RGB", (WIDTH, HEIGHT), decoded[start:start + frame_size],
        ))

    return int((hashed(0) != hashed(CHANGE_SECOND)).sum())


def _block_side(image) -> str:
    grey = image.convert("L")
    if grey.getpixel((70, 100)) > 128:
        return "left"
    if grey.getpixel((110, 100)) > 128:
        return "right"
    return "none"


class BlockBoxes:
    """One box over the block's path, whose line names where the block sits."""

    def __init__(self) -> None:
        self.reads: list[str] = []

    def detect_boxes(self, _engine, image):
        self.reads.append(_block_side(image))
        return [np.asarray(BOX, dtype=np.float32)]

    def recognise_boxes(self, _engine, image, _boxes):
        side = _block_side(image)
        return [side.upper() if side != "none" else None]


def test_spec_addon_133_i1_a_picture_that_moved_by_eight_hash_bits_is_read(
    frame_reader, search_db, monkeypatch, tmp_path,
):
    module, _engine = frame_reader
    boxes = BlockBoxes()
    monkeypatch.setattr(module, "detect_boxes", boxes.detect_boxes)
    monkeypatch.setattr(module, "recognise_boxes", boxes.recognise_boxes)
    clip = _encode(tmp_path / "eight-bits.mp4")
    # 8 is the smallest difference this reader reads; 9 and 16 both skip it.
    assert _hash_distance(module, clip) == 8
    _video(search_db, tmp_path, clip)

    assert module.index_frame_ocr("vid1") is True

    assert boxes.reads == ["left", "right"]
    assert _frame_rows(search_db) == [
        (0, 0.0, 4.0, "LEFT"),
        (1, 4.0, 9.0, "RIGHT"),
    ]
