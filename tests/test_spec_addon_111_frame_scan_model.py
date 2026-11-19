"""SPEC-ADDON-111 §2 Failure cases, I7 as revised by SPEC-ADDON-145: a frame scan
runs, stores its hashes and sets its flag without the CLIP model.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

import app.database as database
from app.models import IndexedFile
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.search_endpoint_db import patch_everywhere, unreadable_policy
from tests.vec_partition_db import search_index  # noqa: F401 - fixture

pytestmark = pytest.mark.usefixtures("pillow_real")

PREVIOUS_CHANGE = "chg_vid1_5"


def _video(path: Path, seconds: int = 6) -> Path:
    """Six seconds of a moving block: enough seconds and one change to embed."""
    fps = 5
    process = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "160x120", "-r", str(fps), "-i", "-", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-g", "10", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for index in range(seconds * fps):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        left = (index * 4) % 150
        frame[:, left:left + 10] = 255
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    assert process.wait() == 0
    return path


def _seed(search_index, path: Path) -> None:
    with database.get_search_db() as session:
        session.add(IndexedFile(
            file_id="vid1", drive="family", filename=path.name,
            file_path=str(path), file_type="video", mime_type="video/mp4",
            file_size=1, active=True, frame_scan_indexed=False,
        ))


def _seed_previous_scan(search_index) -> None:
    dim = search_index.dim("vec_clip_change")
    search_index.add_embedding(
        PREVIOUS_CHANGE, "vid1", "clip_change", "vec_clip_change",
    )
    search_index.put_vector(
        "vec_clip_change", PREVIOUS_CHANGE, np.ones(dim, dtype=np.float32),
        partition="clip_change",
    )
    search_index.sql(
        "INSERT INTO video_frame_scans(file_id, version, seconds, hashes, sizes, "
        "created_at) VALUES ('vid1', 1, 1, :h, :s, CURRENT_TIMESTAMP)",
        h=b"\0" * 32, s=b"\0\0",
    )


@pytest.mark.parametrize("previous", [False, True], ids=["no-previous", "rescan"])
def test_spec_addon_111_revised_i7_the_scan_runs_without_the_clip_model(
    search_index, tmp_path, monkeypatch, previous,
):
    """SPEC-ADDON-145 item 1, revised SPEC-ADDON-111 I7."""
    video = _video(tmp_path / "vid1.mp4")
    _seed(search_index, video)
    if previous:
        _seed_previous_scan(search_index)

    from app.workers import frame_scan

    attempts: list[int] = []

    def _cannot_load() -> None:
        attempts.append(1)
        raise RuntimeError("the CLIP weights are not on disk")

    def _no_embedding(_image):
        raise AssertionError("no frame may be embedded")

    patch_everywhere(monkeypatch, "load_image_model", _cannot_load)
    patch_everywhere(monkeypatch, "embed_image", _no_embedding)
    patch_everywhere(monkeypatch, "is_feature_enabled", unreadable_policy)

    assert frame_scan.index_frame_scan("vid1") is True

    assert attempts == []
    with search_index.engine.connect() as conn:
        assert conn.execute(text(
            "SELECT frame_scan_indexed FROM indexed_files WHERE file_id = 'vid1'"
        )).scalar() == 1
    assert [
        e[0] for e in search_index.embeddings("vid1") if e[1] == "clip_change"
    ] == []
    assert search_index.vectors("vec_clip_change") == {}
    with search_index.engine.connect() as conn:
        assert conn.execute(text(
            "SELECT COUNT(*) FROM video_frame_scans WHERE file_id = 'vid1'"
        )).scalar() == 1
