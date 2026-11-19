"""SPEC-ADDON-111 items 2-8, the failure table and I7 as revised by
SPEC-ADDON-145: ``index_frame_scan`` stores its hashes and writes no change frame.

The test video alternates two horizontal gradients that run in opposite
directions, so every difference hash bit flips at each switch.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

import app.database as database
from app.config import FrameExtractionConfig, IndexingConfig
from app.models import IndexedFile
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.search_endpoint_db import patch_everywhere, unreadable_policy
from tests.vec_partition_db import search_index  # noqa: F401 - fixture

pytestmark = pytest.mark.usefixtures("pillow_real")

LOFT_MIME = "application/vnd.litloft.loft+json"
# Gradient A for [0, 5), B for [5, 12), A for [12, 25), B for [25, 40).
SEGMENTS = [(0, 5, "A"), (5, 12, "B"), (12, 25, "A"), (25, 40, "B")]
CHANGE_SECONDS = [0, 10, 20, 30]


def _frame(kind: str, width: int = 320, height: int = 240) -> bytes:
    ramp = np.linspace(0, 255, width).astype(np.uint8)
    if kind == "B":
        ramp = ramp[::-1]
    grey = np.tile(ramp, (height, 1))
    return np.repeat(grey[:, :, None], 3, axis=2).tobytes()


def _encode(path: Path) -> Path:
    fps = 5
    process = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "320x240", "-r", str(fps), "-i", "-", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-g", "10", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    frames = {kind: _frame(kind) for kind in "AB"}
    for start, end, kind in SEGMENTS:
        for _ in range((end - start) * fps):
            process.stdin.write(frames[kind])
    process.stdin.close()
    assert process.wait() == 0
    return path


def _tail_cut(source: Path, target: Path, keep: float) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-c", "copy",
         "-movflags", "+faststart", str(target)],
        check=True,
    )
    data = target.read_bytes()
    target.write_bytes(data[: int(len(data) * keep)])
    return target


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("frame-scan-media")
    video = _encode(root / "switches.mp4")
    cut = _tail_cut(video, root / "cut.mp4", keep=0.5)
    corrupt = root / "corrupt.mp4"
    corrupt.write_bytes(b"this is not a video")
    return {"video": video, "cut": cut, "corrupt": corrupt}


class Scan:
    def __init__(self, index, monkeypatch, tmp_path, make_settings):
        self.index = index
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        self.make_settings = make_settings
        self.dim = index.dim("vec_clip")
        self.embedded = 0
        self.on_embed = None
        self.policy = None
        self.max_frames = 500

    def video(self, source: Path, file_id: str = "vid1", *, mime: str = "video/mp4",
              active: bool = True, scanned: bool = False) -> Path:
        path = self.tmp_path / f"{file_id}{source.suffix}"
        if source.exists():
            path.write_bytes(source.read_bytes())
        with database.get_search_db() as s:
            s.add(IndexedFile(
                file_id=file_id, drive="family", filename=path.name, file_path=str(path),
                file_type="video", mime_type=mime, file_size=1, active=active,
                frame_scan_indexed=scanned,
            ))
        return path

    def _embed(self, *_args, **_kwargs):
        self.embedded += 1
        if self.on_embed is not None:
            self.on_embed(self.embedded)
        return np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)

    def run(self, file_id: str = "vid1", **kwargs):
        import app.workers.clip as clip
        from app.workers import frame_scan

        settings = self.make_settings(
            allowed_base_dirs=(str(self.tmp_path) + os.sep,),
            indexing=IndexingConfig(
                frame_extraction=FrameExtractionConfig(max_frames=self.max_frames),
            ),
        )
        patch_everywhere(self.monkeypatch, "settings", settings)
        self.monkeypatch.setattr(clip, "embed_image", self._embed)
        patch_everywhere(self.monkeypatch, "embed_image", self._embed)
        policy = self.policy or _allow
        self.monkeypatch.setattr("app.policy_client.is_feature_enabled", policy)
        patch_everywhere(self.monkeypatch, "is_feature_enabled", policy)
        return frame_scan.index_frame_scan(file_id, **kwargs)

    # -- reading back ------------------------------------------------------

    def flag(self, file_id: str = "vid1"):
        return self._scalar(
            "SELECT frame_scan_indexed FROM indexed_files WHERE file_id = :f", f=file_id,
        )

    def changes(self, file_id: str = "vid1") -> list[tuple[str, float, str]]:
        with self.index.engine.connect() as conn:
            return [tuple(r) for r in conn.execute(text(
                "SELECT id, timestamp_start, vector_table FROM embeddings "
                "WHERE file_id = :f AND embedding_type = 'clip_change' "
                "ORDER BY timestamp_start"
            ), {"f": file_id})]

    def scan_row(self, file_id: str = "vid1"):
        with self.index.engine.connect() as conn:
            return conn.execute(text(
                "SELECT version, seconds, hashes, sizes FROM video_frame_scans "
                "WHERE file_id = :f"
            ), {"f": file_id}).fetchone()

    def _scalar(self, sql, **params):
        with self.index.engine.connect() as conn:
            return conn.execute(text(sql), params).scalar()

    def nothing_written(self, file_id: str = "vid1") -> bool:
        return (
            self.changes(file_id) == []
            and self.scan_row(file_id) is None
            and self.index.vectors("vec_clip_change") == {}
        )


async def _allow(*_args, **_kwargs) -> bool:
    return True


async def _index_off(*args, **kwargs) -> bool:
    return "index" not in list(args) + list(kwargs.values())


@pytest.fixture()
def scan(search_index, monkeypatch, tmp_path, make_settings):  # noqa: F811
    return Scan(search_index, monkeypatch, tmp_path, make_settings)


def _bits(hashes: bytes, second: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(hashes[second * 32:(second + 1) * 32], dtype=np.uint8))


# ---------------------------------------------------------------------------
# Normal flow - items 3 to 8
# ---------------------------------------------------------------------------


def test_spec_addon_111_revised_i7_the_scan_stores_its_hashes_and_writes_no_change_frame(
    scan, media,
):
    """SPEC-ADDON-145 item 1, revised SPEC-ADDON-111 I7."""
    scan.video(media["video"])

    assert scan.run() is True

    assert scan.changes() == []
    assert scan.index.vectors("vec_clip_change") == {}
    assert scan.index.vectors("vec_clip") == {}
    assert scan.embedded == 0
    assert scan.flag() == 1
    version, seconds, hashes, sizes = scan.scan_row()
    assert len(hashes) == seconds * 32
    assert len(sizes) == seconds * 2


def test_spec_addon_111_the_scan_row_holds_a_hash_and_a_size_per_second(scan, media):
    scan.video(media["video"])

    scan.run()

    version, seconds, hashes, sizes = scan.scan_row()
    assert version == 1
    assert 39 <= seconds <= 41
    assert len(hashes) == seconds * 32
    assert len(sizes) == seconds * 2
    assert int(np.sum(_bits(hashes, 0) != _bits(hashes, 7))) >= 16
    assert int(np.sum(_bits(hashes, 0) != _bits(hashes, 3))) < 16
    assert all(v > 0 for v in np.frombuffer(sizes, dtype="<u2"))


def test_spec_addon_111_revised_i7_max_frames_does_not_thin_the_scan(scan, media):
    """SPEC-ADDON-145 item 1, revised SPEC-ADDON-111 I7: max_frames caps frames, not hashes."""
    scan.max_frames = 2
    scan.video(media["video"])

    assert scan.run() is True

    assert scan.scan_row()[1] >= 39
    assert scan.changes() == []
    assert scan.embedded == 0
    assert scan.flag() == 1


def test_spec_addon_111_revised_i7_a_rescan_replaces_the_scan_and_removes_the_change_frames(
    scan, media,
):
    """SPEC-ADDON-145 item 1, revised SPEC-ADDON-111 I7."""
    scan.video(media["video"])
    scan.index.add_embedding("chg_vid1_99", "vid1", "clip_change", "vec_clip_change")
    scan.index.put_vector("vec_clip_change", "chg_vid1_99", np.ones(scan.dim),
                          partition="clip_change")
    scan.index.sql(
        "INSERT INTO video_frame_scans(file_id, version, seconds, hashes, sizes, "
        "created_at) VALUES ('vid1', 1, 1, :h, :s, :now)",
        h=b"\0" * 32, s=b"\0\0", now=datetime.now(UTC),
    )

    assert scan.run() is True

    assert scan.changes() == []
    assert "chg_vid1_99" not in scan.index.vectors("vec_clip_change")
    assert scan.scan_row()[1] >= 39


def test_spec_addon_111_the_scan_waits_while_the_indexer_is_paused(scan, media):
    scan.video(media["video"])
    state = {"paused": True}
    result = {}

    def work():
        result["value"] = scan.run(is_paused=lambda: state["paused"])

    thread = threading.Thread(target=work)
    thread.start()
    time.sleep(1.0)
    embedded_while_paused = scan.embedded
    state["paused"] = False
    thread.join(timeout=60)

    assert embedded_while_paused == 0
    assert not thread.is_alive()
    assert result["value"] is True


# ---------------------------------------------------------------------------
# Item 2 - start checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param({"scanned": True}, id="flag-already-true"),
        pytest.param({"active": False}, id="inactive-row"),
        pytest.param({"mime": LOFT_MIME}, id="loft-is-not-native"),
        pytest.param({"mime": "image/jpeg"}, id="picture-is-not-native"),
    ],
)
def test_spec_addon_111_a_task_that_finds_nothing_to_scan_does_nothing(
    scan, media, setup,
):
    scan.video(media["video"], **setup)

    assert scan.run() is False

    assert scan.embedded == 0
    assert scan.nothing_written()


def test_spec_addon_111_a_task_for_a_gone_row_does_nothing(scan, media):
    assert scan.run("ghost") is False
    assert scan.embedded == 0


def test_spec_addon_111_index_off_writes_nothing_and_leaves_the_flag(scan, media):
    scan.video(media["video"])
    scan.policy = _index_off

    assert scan.run() is False

    assert scan.embedded == 0
    assert scan.nothing_written()
    assert scan.flag() == 0


def test_spec_addon_111_an_unreadable_policy_counts_as_on(scan, media):
    scan.video(media["video"])
    scan.policy = unreadable_policy

    assert scan.run() is True
    assert scan.flag() == 1


# ---------------------------------------------------------------------------
# Failure table, I7
# ---------------------------------------------------------------------------


def test_spec_addon_111_an_unreachable_file_writes_nothing_and_is_logged(
    scan, media, caplog,
):
    path = scan.video(media["video"])
    path.unlink()

    with caplog.at_level("WARNING"):
        assert scan.run() is False

    assert scan.nothing_written()
    assert scan.flag() == 0
    assert any(
        r.levelname == "WARNING" and "vid1" in r.getMessage() for r in caplog.records
    )


def test_spec_addon_111_a_file_gone_after_extraction_writes_nothing(scan, media):
    scan.video(media["video"])
    from app.workers import frame_scan

    original = frame_scan._decode

    def decode_then_remove(source, *, paused):
        decoded = original(source, paused=paused)
        Path(source).unlink()
        return decoded

    scan.monkeypatch.setattr(frame_scan, "_decode", decode_then_remove)

    assert scan.run() is False

    assert scan.nothing_written()
    assert scan.flag() == 0


def test_spec_addon_111_a_file_trashed_during_the_scan_writes_nothing(scan, media):
    scan.video(media["video"])
    from app.workers import frame_scan

    original = frame_scan._decode

    def decode_then_trash(source, *, paused):
        decoded = original(source, paused=paused)
        scan.index.sql("UPDATE indexed_files SET active = 0 WHERE file_id = 'vid1'")
        return decoded

    scan.monkeypatch.setattr(frame_scan, "_decode", decode_then_trash)

    assert scan.run() is False

    assert scan.nothing_written()
    assert scan.flag() == 0


def test_spec_addon_111_revised_i7_the_scan_embeds_no_frame(scan, media):
    """SPEC-ADDON-145 item 1, revised SPEC-ADDON-111 I7."""
    scan.video(media["video"])

    def fail_second(n):
        raise RuntimeError("no frame may be embedded")

    scan.on_embed = fail_second

    assert scan.run() is True

    assert scan.embedded == 0
    assert scan.changes() == []
    assert scan.flag() == 1


def test_spec_addon_111_an_early_ffmpeg_end_keeps_the_seconds_read(scan, media):
    scan.video(media["cut"])

    assert scan.run() is True

    version, seconds, hashes, sizes = scan.scan_row()
    assert 0 < seconds < 39
    assert len(hashes) == seconds * 32
    assert scan.changes() == []
    assert scan.flag() == 1


def test_spec_addon_111_no_second_read_deletes_the_previous_scan_and_sets_the_flag(
    scan, media,
):
    scan.video(media["corrupt"])
    scan.index.add_embedding("chg_vid1_3", "vid1", "clip_change", "vec_clip_change")
    scan.index.put_vector("vec_clip_change", "chg_vid1_3", np.ones(scan.dim),
                          partition="clip_change")
    scan.index.sql(
        "INSERT INTO video_frame_scans(file_id, version, seconds, hashes, sizes, "
        "created_at) VALUES ('vid1', 1, 1, :h, :s, :now)",
        h=b"\0" * 32, s=b"\0\0", now=datetime.now(UTC),
    )

    assert scan.run() is True

    assert scan.nothing_written()
    assert scan.flag() == 1
    assert scan.embedded == 0
