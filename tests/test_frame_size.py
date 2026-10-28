"""SPEC-ADDON-030: the frame endpoint's ``size`` and the size-aware frame cache."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base

from app.frame_cache import (
    ensure_frame_cached,
    extract_frame_to_cache,
    frame_cache_dir,
    frame_cache_path,
    purge_frame_cache,
)
from app.models import IndexedFile

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg not installed",
)


@pytest.fixture()
def patched_data_dir(tmp_path, monkeypatch):
    import app.config as config

    original = config.settings
    new_settings = type(original).__new__(type(original))
    object.__setattr__(new_settings, "__dict__", dict(original.__dict__))
    object.__setattr__(new_settings, "intelligence_data_dir", tmp_path / "data")
    monkeypatch.setattr(config, "settings", new_settings)
    return tmp_path / "data"


def _make_video(path: Path, width: int, height: int) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate=10:duration=2",
            "-pix_fmt", "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p",
            str(path),
        ],
        check=True,
    )


def _dimensions(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(path),
        ],
        check=True, capture_output=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    return stream["width"], stream["height"]


def _record_ffmpeg(cmds: list[list[str]]):
    def fake_run(cmd, capture_output, timeout):
        cmds.append([str(a) for a in cmd])
        Path(cmd[-1]).write_bytes(b"FAKE_WEBP")
        return MagicMock(returncode=0)

    return fake_run


class TestFrameCachePath:
    def test_spec_addon_030_large_has_its_own_file_in_the_same_directory(self, patched_data_dir):
        small = frame_cache_path("abc123", 12.345)
        large = frame_cache_path("abc123", 12.345, "large")
        assert small == patched_data_dir / "frames" / "abc123" / "12345.webp"
        assert large == patched_data_dir / "frames" / "abc123" / "12345.large.webp"

    def test_spec_addon_030_explicit_small_is_the_default_path(self, patched_data_dir):
        assert frame_cache_path("abc123", 1.5, "small") == frame_cache_path("abc123", 1.5)
        assert frame_cache_path("abc123", 1.5, size="small") == frame_cache_path("abc123", 1.5)


class TestExtractArguments:
    def test_spec_addon_030_explicit_small_runs_the_default_ffmpeg_command(self, tmp_path):
        cmds: list[list[str]] = []
        cache_path = tmp_path / "f" / "1000.webp"
        with patch("app.frame_cache.subprocess.run", side_effect=_record_ffmpeg(cmds)):
            extract_frame_to_cache("/tmp/video.mp4", cache_path, 1.0)
            extract_frame_to_cache("/tmp/video.mp4", cache_path, 1.0, "small")
            extract_frame_to_cache("/tmp/video.mp4", cache_path, 1.0, size="small")
        # The output is a uuid-named temp file beside the cache path, so it
        # differs per call; every argument before it must not.
        assert cmds[0][:-1] == cmds[1][:-1] == cmds[2][:-1]
        assert all(Path(c[-1]).name.startswith(cache_path.name) for c in cmds)
        joined = " ".join(cmds[0])
        assert "libwebp" in joined
        assert "-q:v 70" in joined

    def test_spec_addon_030_large_uses_q80_and_writes_the_output_last(self, tmp_path):
        cmds: list[list[str]] = []
        cache_path = tmp_path / "f" / "1000.large.webp"
        with patch("app.frame_cache.subprocess.run", side_effect=_record_ffmpeg(cmds)):
            extract_frame_to_cache("/tmp/video.mp4", cache_path, 1.0, size="large")
        joined = " ".join(cmds[0])
        assert "libwebp" in joined
        assert "-q:v 80" in joined
        assert "-q:v 70" not in joined
        assert Path(cmds[0][-1]).parent == cache_path.parent
        assert cache_path.read_bytes() == b"FAKE_WEBP"


@needs_ffmpeg
class TestLargeDimensions:
    @pytest.mark.parametrize(
        "source",
        [
            pytest.param((1920, 1080), id="landscape-1080p"),
            pytest.param((1080, 1920), id="portrait-1080p"),
            pytest.param((2000, 3000), id="portrait-tall"),
            pytest.param((1281, 721), id="just-over-1280"),
            pytest.param((640, 360), id="smaller-than-large"),
            pytest.param((1280, 720), id="exactly-1280"),
        ],
    )
    def test_spec_addon_030_large_fits_1280_keeps_aspect_and_never_upscales(
        self, tmp_path, source,
    ):
        src_w, src_h = source
        video = tmp_path / "in.mp4"
        _make_video(video, src_w, src_h)
        out = tmp_path / "f" / "1000.large.webp"

        extract_frame_to_cache(str(video), out, 1.0, size="large")

        w, h = _dimensions(out)
        assert max(w, h) <= 1280
        assert max(w, h) <= max(src_w, src_h)
        assert max(w, h) == min(1280, max(src_w, src_h))
        assert abs(h - w * src_h / src_w) <= 1
        assert abs(w - h * src_w / src_h) <= 1

    def test_spec_addon_030_small_and_large_at_the_same_time_do_not_overwrite(
        self, tmp_path, patched_data_dir,
    ):
        video = tmp_path / "in.mp4"
        _make_video(video, 1920, 1080)
        small = frame_cache_path("vid", 1.0)
        large = frame_cache_path("vid", 1.0, "large")

        extract_frame_to_cache(str(video), small, 1.0)
        extract_frame_to_cache(str(video), large, 1.0, size="large")

        assert small != large
        assert _dimensions(small)[0] == 320
        assert _dimensions(large) == (1280, 720)


class TestCacheLifecycle:
    def test_spec_addon_030_ensure_frame_cached_serves_small_even_when_large_is_cached(
        self, patched_data_dir,
    ):
        large = frame_cache_path("f_only_large", 2.0, "large")
        large.parent.mkdir(parents=True)
        large.write_bytes(b"LARGE")

        def fake_extract(*args):
            args[1].write_bytes(b"SMALL")

        with patch(
            "app.frame_cache.extract_frame_to_cache", side_effect=fake_extract,
        ) as extract:
            result = ensure_frame_cached("f_only_large", 2.0, "/tmp/video.mp4")

        extract.assert_called_once_with(
            "/tmp/video.mp4", frame_cache_path("f_only_large", 2.0), 2.0,
        )
        assert result.read_bytes() == b"SMALL"
        assert large.read_bytes() == b"LARGE"

    def test_spec_addon_030_purge_removes_both_sizes(self, patched_data_dir):
        small = frame_cache_path("file_x", 1.0)
        large = frame_cache_path("file_x", 1.0, "large")
        small.parent.mkdir(parents=True)
        small.write_bytes(b"S")
        large.write_bytes(b"L")

        purge_frame_cache("file_x")

        assert not frame_cache_dir("file_x").exists()


def _patch_everywhere(monkeypatch, name, value) -> None:
    for mod_name, module in list(sys.modules.items()):
        if module is None or not (mod_name == "app" or mod_name.startswith("app.")):
            continue
        if name in getattr(module, "__dict__", {}):
            monkeypatch.setattr(module, name, value)


@pytest.fixture()
def frame_api(tmp_path, monkeypatch, make_settings):
    from app.main import app

    video = tmp_path / "drives" / "family" / "v.mp4"
    video.parent.mkdir(parents=True)
    if shutil.which("ffmpeg"):
        _make_video(video, 1920, 1080)

    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as s:
        s.add(IndexedFile(
            file_id="vid", drive="family", filename="v.mp4", file_path=str(video),
            file_type="video", mime_type="video/mp4", file_size=1, active=True,
            duration=2.0,
        ))
        s.commit()

    @contextmanager
    def _session():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    settings = make_settings(allowed_base_dirs=(str(tmp_path / "drives") + "/",))
    _patch_everywhere(monkeypatch, "get_search_db_read", _session)
    _patch_everywhere(monkeypatch, "get_search_db", _session)
    _patch_everywhere(monkeypatch, "settings", settings)
    _patch_everywhere(monkeypatch, "resolve_file_path", lambda _drive, path: path)
    _patch_everywhere(monkeypatch, "validate_file_path", lambda _path: True)

    client = TestClient(app)

    def get(params, drive="family"):
        headers = {"X-Lit-Drive": drive} if drive else {}
        return client.get("/files/vid/frame", params=params, headers=headers)

    return get, settings.intelligence_data_dir / "frames" / "vid"


def _webp_dimensions(content: bytes, tmp_path: Path) -> tuple[int, int]:
    path = tmp_path / "response.webp"
    path.write_bytes(content)
    return _dimensions(path)


class TestEndpointSize:
    @pytest.mark.parametrize("size", ["medium", "LARGE", "", "1280"])
    def test_spec_addon_030_unknown_size_is_422(self, frame_api, size):
        get, _ = frame_api
        response = get({"t": 1.0, "size": size})
        assert response.status_code == 422

    def test_spec_addon_030_cached_frames_are_served_by_size(self, frame_api):
        get, cache_dir = frame_api
        cache_dir.mkdir(parents=True)
        (cache_dir / "1000.webp").write_bytes(b"SMALL")
        (cache_dir / "1000.large.webp").write_bytes(b"LARGE")

        for params, drive, expected in [
            ({"t": 1.0}, "family", b"SMALL"),
            ({"t": 1.0, "size": "small"}, "family", b"SMALL"),
            ({"t": 1.0, "size": "large"}, "family", b"LARGE"),
            ({"t": 1.0, "size": "large"}, None, b"LARGE"),
        ]:
            response = get(params, drive=drive)
            assert response.status_code == 200, (params, drive, response.text)
            assert response.content == expected, (params, drive)

    @needs_ffmpeg
    def test_spec_addon_030_large_is_extracted_into_its_own_cache_file(self, frame_api, tmp_path):
        get, cache_dir = frame_api

        small = get({"t": 1.0})
        large = get({"t": 1.0, "size": "large"})

        assert small.status_code == 200, small.text
        assert large.status_code == 200, large.text
        assert large.headers["content-type"].startswith("image/webp")
        assert _webp_dimensions(small.content, tmp_path)[0] == 320
        assert _webp_dimensions(large.content, tmp_path) == (1280, 720)
        assert (cache_dir / "1000.webp").read_bytes() == small.content
        assert (cache_dir / "1000.large.webp").read_bytes() == large.content
        assert get({"t": 1.0}).content == small.content

    @needs_ffmpeg
    def test_spec_addon_030_large_for_another_accessible_drive_is_404_and_caches_nothing(
        self, frame_api,
    ):
        get, cache_dir = frame_api

        assert get({"t": 1.0, "size": "large"}).status_code == 200
        assert (cache_dir / "1000.large.webp").exists()
        other = get({"t": 1.5, "size": "large"}, drive="other")

        assert other.status_code == 404
        assert "family" not in other.text
        assert not (cache_dir / "1500.large.webp").exists()
        assert not (cache_dir / "1500.webp").exists()
