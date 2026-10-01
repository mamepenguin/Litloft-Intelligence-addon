from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app.workers.clip import _extract_scene_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _make_video(path: Path, title: bytes) -> None:
    # Red then blue: one hard cut, so the scene filter selects at least one frame.
    subprocess.run(
        [
            b"ffmpeg", b"-y", b"-loglevel", b"error",
            b"-f", b"lavfi", b"-i", b"color=red:s=64x64:d=1:r=10",
            b"-f", b"lavfi", b"-i", b"color=blue:s=64x64:d=1:r=10",
            b"-filter_complex", b"[0:v][1:v]concat=n=2:v=1[v]",
            b"-map", b"[v]",
            b"-metadata", b"title=" + title,
            os.fsencode(path),
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "title",
    [
        "鋼の錬金術師 予告".encode(),
        "鋼の錬金術師 予告".encode("cp932"),
    ],
    ids=["utf-8", "cp932"],
)
def test_extracts_frames_whatever_the_title_encoding(tmp_path, title):
    video = tmp_path / "in.mkv"
    _make_video(video, title)
    out = tmp_path / "frames"
    out.mkdir()

    frames = _extract_scene_frames(str(video), out, threshold=0.3)

    assert len(frames) == 1
    assert all(p.exists() for _, p in frames)
    assert frames[0][0] == pytest.approx(1.0, abs=0.15)
