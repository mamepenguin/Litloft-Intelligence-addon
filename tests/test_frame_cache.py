"""Tests for the on-disk CLIP frame cache.

Covers the pure helpers in :mod:`app.frame_cache` (extracted from
``app.routers.files`` so the video-visual worker can share the same
cache implementation):

* ``frame_cache_path`` quantises the timestamp to milliseconds so two
  identical floats land on the same cache file.
* ``extract_frame_to_cache`` writes via a ``.tmp`` sibling and atomic
  rename, removes the temp file on ffmpeg failure / timeout, and
  surfaces an HTTPException on non-zero exit.
* ``ensure_frame_cached`` short-circuits on a cache hit and delegates
  to ``extract_frame_to_cache`` on a miss.
* ``purge_frame_cache`` removes the per-file directory and tolerates
  a missing one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from fastapi import HTTPException  # noqa: E402

from app.frame_cache import (  # noqa: E402
    ensure_frame_cached,
    extract_frame_to_cache,
    frame_cache_dir,
    frame_cache_path,
    purge_frame_cache,
)


@pytest.fixture()
def patched_data_dir(tmp_path, monkeypatch):
    """Point ``settings.intelligence_data_dir`` at a per-test tmp dir.

    The cache helpers read ``settings.intelligence_data_dir`` lazily on
    every call (fresh ``from app.config import settings`` inside the
    function body), so monkeypatching the live module attribute is enough.
    """
    import app.config as config

    original = config.settings
    new_settings = type(original).__new__(type(original))
    object.__setattr__(new_settings, "__dict__", dict(original.__dict__))
    object.__setattr__(new_settings, "intelligence_data_dir", tmp_path)
    monkeypatch.setattr(config, "settings", new_settings)
    return tmp_path


# ---------------------------------------------------------------------------
# frame_cache_path
# ---------------------------------------------------------------------------


class TestFrameCachePath:
    def test_path_is_per_file_dir_with_ms_filename(self, patched_data_dir):
        path = frame_cache_path("abc123", 12.345)
        assert path == patched_data_dir / "frames" / "abc123" / "12345.webp"

    def test_quantises_to_milliseconds(self, patched_data_dir):
        # Same millisecond → same file name. Defends against floating
        # point drift between repeated requests.
        a = frame_cache_path("f", 1.2349999)
        b = frame_cache_path("f", 1.235)
        assert a == b

    def test_different_files_get_separate_dirs(self, patched_data_dir):
        a = frame_cache_path("file_a", 5.0)
        b = frame_cache_path("file_b", 5.0)
        assert a.parent != b.parent
        assert a.name == b.name == "5000.webp"


# ---------------------------------------------------------------------------
# extract_frame_to_cache
# ---------------------------------------------------------------------------


class TestExtractFrameToCache:
    def test_writes_via_tmp_then_atomic_rename(self, tmp_path):
        cache_path = tmp_path / "f" / "1000.webp"

        def fake_run(cmd, capture_output, timeout):
            # ffmpeg in the real flow writes to the last positional arg
            target = Path(cmd[-1])
            target.write_bytes(b"FAKE_WEBP")
            return MagicMock(returncode=0)

        with patch("app.frame_cache.subprocess.run", side_effect=fake_run):
            extract_frame_to_cache("/tmp/video.mp4", cache_path, 1.0)

        assert cache_path.exists()
        assert cache_path.read_bytes() == b"FAKE_WEBP"
        # tmp sibling must have been moved away
        assert not cache_path.with_suffix(cache_path.suffix + ".tmp").exists()

    def test_creates_parent_dirs(self, tmp_path):
        cache_path = tmp_path / "deeply" / "nested" / "f" / "1.webp"

        def fake_run(cmd, capture_output, timeout):
            Path(cmd[-1]).write_bytes(b"X")
            return MagicMock(returncode=0)

        with patch("app.frame_cache.subprocess.run", side_effect=fake_run):
            extract_frame_to_cache("/tmp/video.mp4", cache_path, 0.001)

        assert cache_path.exists()

    def test_ffmpeg_nonzero_raises_500_and_cleans_tmp(self, tmp_path):
        cache_path = tmp_path / "f" / "1000.webp"

        def fake_run(cmd, capture_output, timeout):
            # Simulate ffmpeg writing a partial file then failing.
            Path(cmd[-1]).write_bytes(b"PARTIAL")
            return MagicMock(returncode=1, stderr=b"boom")

        with patch("app.frame_cache.subprocess.run", side_effect=fake_run):
            with pytest.raises(HTTPException) as excinfo:
                extract_frame_to_cache("/tmp/video.mp4", cache_path, 1.0)

        assert excinfo.value.status_code == 500
        assert not cache_path.exists()
        # Temp file must be cleaned up so retries don't see stale bytes.
        assert not cache_path.with_suffix(cache_path.suffix + ".tmp").exists()

    def test_ffmpeg_empty_output_raises_500(self, tmp_path):
        cache_path = tmp_path / "f" / "1000.webp"

        def fake_run(cmd, capture_output, timeout):
            # Touch an empty file to mimic ffmpeg silently producing 0 bytes.
            Path(cmd[-1]).write_bytes(b"")
            return MagicMock(returncode=0)

        with patch("app.frame_cache.subprocess.run", side_effect=fake_run):
            with pytest.raises(HTTPException) as excinfo:
                extract_frame_to_cache("/tmp/video.mp4", cache_path, 1.0)

        assert excinfo.value.status_code == 500

    def test_timeout_raises_504_and_cleans_tmp(self, tmp_path):
        cache_path = tmp_path / "f" / "1000.webp"
        tmp_sibling = cache_path.with_suffix(cache_path.suffix + ".tmp")

        def fake_run(cmd, capture_output, timeout):
            # Simulate ffmpeg starting a write before being killed.
            tmp_sibling.parent.mkdir(parents=True, exist_ok=True)
            tmp_sibling.write_bytes(b"WIP")
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        with patch("app.frame_cache.subprocess.run", side_effect=fake_run):
            with pytest.raises(HTTPException) as excinfo:
                extract_frame_to_cache("/tmp/video.mp4", cache_path, 1.0)

        assert excinfo.value.status_code == 504
        assert not tmp_sibling.exists()


# ---------------------------------------------------------------------------
# ensure_frame_cached
# ---------------------------------------------------------------------------


class TestEnsureFrameCached:
    def test_cache_hit_skips_extraction(self, patched_data_dir):
        cache_path = frame_cache_path("f_hit", 2.0)
        cache_path.parent.mkdir(parents=True)
        cache_path.write_bytes(b"CACHED")

        with patch("app.frame_cache.extract_frame_to_cache") as extract:
            result = ensure_frame_cached("f_hit", 2.0, "/tmp/video.mp4")

        extract.assert_not_called()
        assert result == cache_path

    def test_cache_miss_extracts(self, patched_data_dir):
        cache_path = frame_cache_path("f_miss", 3.0)

        def fake_extract(abs_path, path, ts):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"NEW")

        with patch(
            "app.frame_cache.extract_frame_to_cache", side_effect=fake_extract
        ) as extract:
            result = ensure_frame_cached("f_miss", 3.0, "/tmp/video.mp4")

        extract.assert_called_once_with("/tmp/video.mp4", cache_path, 3.0)
        assert result == cache_path
        assert result.read_bytes() == b"NEW"

    def test_zero_byte_cache_file_is_treated_as_miss(self, patched_data_dir):
        cache_path = frame_cache_path("f_empty", 4.0)
        cache_path.parent.mkdir(parents=True)
        cache_path.touch()  # 0 bytes — a prior failed/interrupted write

        with patch("app.frame_cache.extract_frame_to_cache") as extract:
            ensure_frame_cached("f_empty", 4.0, "/tmp/video.mp4")

        extract.assert_called_once()


# ---------------------------------------------------------------------------
# purge_frame_cache
# ---------------------------------------------------------------------------


class TestPurgeFrameCache:
    def test_removes_per_file_directory(self, patched_data_dir):
        cache_dir = frame_cache_dir("file_x")
        cache_dir.mkdir(parents=True)
        (cache_dir / "1000.webp").write_bytes(b"X")
        (cache_dir / "5000.webp").write_bytes(b"Y")

        purge_frame_cache("file_x")

        assert not cache_dir.exists()

    def test_missing_directory_is_noop(self, patched_data_dir):
        # Should not raise; absence of the dir means nothing to do.
        purge_frame_cache("never_indexed_file")
