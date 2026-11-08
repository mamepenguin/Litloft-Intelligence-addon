"""Doubles for the YouTube frame and storyboard tests.

``yt-dlp`` and ``ffmpeg`` are replaced by executables placed first (and only) on
``PATH``, so the tests hold whatever way the addon starts them. The fake
``yt-dlp`` prints a recorded ``-J`` description; the fake ``ffmpeg`` waits as
told and then becomes the real ffmpeg. The "stream" and the storyboard sheets
are served by a local HTTP server, so a frame is a real decode of a real file
and every address the addon uses can be seen in the server's request log.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import IndexedFile
from tests.search_endpoint_db import patch_everywhere

LOFT_MIME = "application/vnd.litloft.loft+json"
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
SIGNED_TOKEN = "SIGNED-TOKEN-9f3a"
HEADER_TOKEN = "HDR-SECRET-77c1"
DRIVE = "family"

REAL_FFMPEG = shutil.which("ffmpeg")
REAL_FFPROBE = shutil.which("ffprobe")

needs_ffmpeg = pytest.mark.skipif(
    REAL_FFMPEG is None or REAL_FFPROBE is None, reason="ffmpeg not installed",
)


def real_pil_modules() -> dict:
    """The real Pillow modules, which ``conftest`` replaces with mocks."""
    saved = {k: v for k, v in sys.modules.items() if k == "PIL" or k.startswith("PIL.")}
    for key in saved:
        del sys.modules[key]
    try:
        try:
            import PIL.Image
            import PIL.ImageDraw  # noqa: F401
        except ImportError:
            pytest.skip("Pillow not installed")
        PIL.Image.init()
        return {
            k: v for k, v in sys.modules.items() if k == "PIL" or k.startswith("PIL.")
        }
    finally:
        for key in [k for k in sys.modules if k == "PIL" or k.startswith("PIL.")]:
            del sys.modules[key]
        sys.modules.update(saved)


_YTDLP_SCRIPT = """#!{python}
import json, os, sys, time
d = os.environ["FAKE_YT_DIR"]
cfg = json.load(open(d + "/ytdlp.json"))
with open(d + "/ytdlp.log", "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "t": time.time()}}) + "\\n")
time.sleep(cfg.get("delay", 0))
if cfg.get("fail"):
    sys.stderr.write(cfg["fail"] + "\\n")
    sys.exit(1)
json.dump(cfg["describe"], sys.stdout)
"""

_FFMPEG_SCRIPT = """#!{python}
import json, os, sys, time
d = os.environ["FAKE_YT_DIR"]
cfg = json.load(open(d + "/ffmpeg.json"))
def note(kind):
    with open(d + "/ffmpeg.log", "a") as f:
        f.write(json.dumps({{"kind": kind, "pid": os.getpid(), "t": time.time(),
                            "argv": sys.argv[1:]}}) + "\\n")
note("start")
time.sleep(cfg.get("hold", 0))
note("end")
os.execv("{real}", ["{real}"] + sys.argv[1:])
"""


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        server = self.server
        parts = urlsplit(self.path)
        with server.lock:
            server.requests.append({
                "path": parts.path,
                "query": parts.query,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "t": time.time(),
            })
            override = dict(server.overrides.get(parts.path, {}))
        if override.get("delay"):
            time.sleep(override["delay"])
        if override.get("status"):
            self.send_response(override["status"])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = server.files.get(parts.path)
        if body is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end, status = 0, len(body) - 1, 200
        spec = self.headers.get("Range")
        if spec and spec.startswith("bytes="):
            lo, _, hi = spec[6:].partition("-")
            start = int(lo) if lo else 0
            end = int(hi) if hi else len(body) - 1
            end = min(end, len(body) - 1)
            status = 206
        chunk = body[start:end + 1]
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(chunk)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
        self.end_headers()
        try:
            self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


class FakeYouTube:
    def __init__(self, root: Path, videos_dir: Path):
        self.root = root
        self.videos_dir = videos_dir
        self.state = root / "state"
        self.state.mkdir()
        self.bin = root / "bin"
        self.bin.mkdir()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.daemon_threads = True
        self.server.lock = threading.Lock()
        self.server.requests = []
        self.server.overrides = {}
        self.server.files = {}
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        python = sys.executable
        (self.bin / "yt-dlp").write_text(_YTDLP_SCRIPT.format(python=python))
        (self.bin / "ffmpeg").write_text(
            _FFMPEG_SCRIPT.format(python=python, real=REAL_FFMPEG)
        )
        if REAL_FFPROBE:
            (self.bin / "ffprobe").symlink_to(REAL_FFPROBE)
        for name in ("yt-dlp", "ffmpeg"):
            (self.bin / name).chmod(0o755)
        self.set_ytdlp(describe=describe([]))
        self.set_ffmpeg(hold=0)

    # -- configuration -------------------------------------------------

    def set_ytdlp(self, *, describe=None, fail=None, delay=0.0) -> None:
        (self.state / "ytdlp.json").write_text(json.dumps(
            {"describe": describe, "fail": fail, "delay": delay}
        ))

    def set_ffmpeg(self, *, hold: float) -> None:
        (self.state / "ffmpeg.json").write_text(json.dumps({"hold": hold}))

    def remove_ytdlp(self) -> None:
        (self.bin / "yt-dlp").unlink()

    def serve_video(self, name: str, width: int, height: int) -> str:
        path = f"/v/{name}"
        self.server.files[path] = self._video_bytes(width, height)
        return path

    def format(self, fid: str, height: int, **kwargs) -> dict:
        """A video format whose stream is served here, at ``height`` pixels."""
        fmt = video_format(fid, height, **kwargs)
        if kwargs.get("url") is None:
            fmt["url"] = self.base + fmt["url"]
            self.serve_video(fid, fmt["width"], fmt["height"])
        return fmt

    def serve_sheet(self, name: str, data: bytes) -> str:
        path = f"/sb/{name}"
        self.server.files[path] = data
        return path

    def override(self, path: str, **kwargs) -> None:
        with self.server.lock:
            self.server.overrides[path] = kwargs

    def _video_bytes(self, width: int, height: int) -> bytes:
        out = self.videos_dir / f"{width}x{height}.mp4"
        if not out.exists():
            subprocess.run(
                [
                    REAL_FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"testsrc=size={width}x{height}:rate=10:duration=70",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
                ],
                check=True,
            )
        return out.read_bytes()

    # -- observation ---------------------------------------------------

    def lookups(self) -> list[dict]:
        return self._log("ytdlp.log")

    def ffmpeg_events(self) -> list[dict]:
        return self._log("ffmpeg.log")

    def _log(self, name: str) -> list[dict]:
        path = self.state / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def fetched(self, prefix: str) -> list[dict]:
        with self.server.lock:
            return [r for r in self.server.requests if r["path"].startswith(prefix)]

    def peak_ffmpeg_concurrency(self) -> int:
        events = sorted(self.ffmpeg_events(), key=lambda e: e["t"])
        live, peak = set(), 0
        for event in events:
            if event["kind"] == "start":
                live.add(event["pid"])
                peak = max(peak, len(live))
            else:
                live.discard(event["pid"])
        return peak

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def video_format(
    fid: str, height: int, *, vcodec: str = "avc1.4d401f", ext: str = "mp4",
    protocol: str = "https", acodec: str = "none", url: str | None = None,
    headers: dict | None = None,
) -> dict:
    width = height * 16 // 9 // 2 * 2
    return {
        "format_id": fid, "ext": ext, "protocol": protocol, "vcodec": vcodec,
        "acodec": acodec, "height": height, "width": width,
        "url": url if url is not None else f"/v/{fid}?sig={SIGNED_TOKEN}",
        "http_headers": headers if headers is not None else {"X-Fake-Auth": HEADER_TOKEN},
    }


def storyboard_format(
    base: str, sheets: list[tuple[str, float]], *, rows: int = 3, columns: int = 3,
    tile: tuple[int, int] = (320, 180), fps: float | None = None,
) -> dict:
    fmt = {
        "format_id": "sb0", "format_note": "storyboard", "ext": "mhtml",
        "protocol": "mhtml", "vcodec": "none", "acodec": "none",
        "width": tile[0], "height": tile[1], "rows": rows, "columns": columns,
        "url": f"{base}/sb/{sheets[0][0]}" if sheets else "",
        "fragments": [{"url": f"{base}/sb/{n}", "duration": d} for n, d in sheets],
    }
    if fps is not None:
        fmt["fps"] = fps
    return fmt


def describe(formats: list[dict], *, duration: float | None = None) -> dict:
    out: dict = {"id": "dQw4w9WgXcQ", "title": "fixture", "formats": formats}
    if duration is not None:
        out["duration"] = duration
    return out


@pytest.fixture()
def youtube(tmp_path, tmp_path_factory, monkeypatch):
    """The fake yt-dlp and ffmpeg on ``PATH`` and a local server for what they name."""
    videos = tmp_path_factory.getbasetemp() / "yt-videos"
    videos.mkdir(exist_ok=True)
    (tmp_path / "yt").mkdir()
    fake = FakeYouTube(tmp_path / "yt", videos)
    # The kept stream descriptions belong to the process, not to a test: one
    # left by an earlier test would name that test's server.
    from app import youtube_source

    youtube_source._descriptions.clear()
    youtube_source._lookups.clear()
    monkeypatch.setenv("PATH", str(fake.bin))
    monkeypatch.setenv("FAKE_YT_DIR", str(fake.state))
    yield fake
    fake.stop()


@contextmanager
def _session_factory(Session):
    session = Session()
    try:
        yield session
    finally:
        session.close()


class FrameApi:
    """The frame, scenes and ask routes over a real search DB and a tmp drive."""

    def __init__(self, Session, drive_dir: Path, data_dir: Path, client):
        self.Session = Session
        self.drive_dir = drive_dir
        self.data_dir = data_dir
        self.client = client

    def add_loft(
        self, file_id: str, body: str | dict | None = None, *, duration=None,
        drive: str = DRIVE, file_type: str = "video",
    ) -> Path:
        path = self.drive_dir / f"{file_id}.loft"
        if body is None:
            body = {"provider": "youtube", "url": YOUTUBE_URL}
        path.write_text(body if isinstance(body, str) else json.dumps(body))
        self.add_row(
            file_id, path, mime=LOFT_MIME, duration=duration, drive=drive,
            file_type=file_type,
        )
        return path

    def add_row(
        self, file_id: str, path: Path, *, mime: str, duration=None,
        drive: str = DRIVE, file_type: str = "video",
    ) -> None:
        with self.Session() as s:
            s.add(IndexedFile(
                file_id=file_id, drive=drive, filename=path.name,
                file_path=str(path), file_type=file_type, mime_type=mime,
                file_size=1, active=True, duration=duration,
            ))
            s.commit()

    def cache_dir(self, file_id: str) -> Path:
        return self.data_dir / "frames" / file_id

    def cached(self, file_id: str) -> list[str]:
        d = self.cache_dir(file_id)
        return sorted(p.name for p in d.iterdir()) if d.exists() else []

    async def frame(self, file_id: str, t: float, size: str | None = None, drive=DRIVE):
        params: dict = {"t": t}
        if size is not None:
            params["size"] = size
        headers = {"X-Lit-Drive": drive} if drive else {}
        return await self.client.get(
            f"/files/{file_id}/frame", params=params, headers=headers,
        )


@pytest.fixture()
async def frame_api(tmp_path, monkeypatch, make_settings):
    from app.main import app

    drive_dir = tmp_path / "drives" / DRIVE
    drive_dir.mkdir(parents=True)
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _read():
        session = Session()
        try:
            yield session
        finally:
            session.rollback()
            session.close()

    @contextmanager
    def _write():
        session = Session()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    settings = make_settings(allowed_base_dirs=(str(tmp_path / "drives") + "/",))
    patch_everywhere(monkeypatch, "get_search_db_read", _read)
    patch_everywhere(monkeypatch, "get_search_db", _write)
    patch_everywhere(monkeypatch, "settings", settings)
    patch_everywhere(monkeypatch, "resolve_file_path", lambda _drive, path: path)
    patch_everywhere(monkeypatch, "validate_file_path", lambda _path: True)

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://intelligence") as client:
        yield FrameApi(Session, drive_dir, Path(settings.intelligence_data_dir), client)


class Clock:
    def __init__(self):
        self.offset = 0.0

    def advance(self, seconds: float) -> None:
        self.offset += seconds


@pytest.fixture()
def clock(monkeypatch):
    real_time, real_monotonic = time.time, time.monotonic
    c = Clock()
    monkeypatch.setattr(time, "time", lambda: real_time() + c.offset)
    monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + c.offset)
    return c


def webp_size(content: bytes, tmp_path: Path) -> tuple[int, int]:
    path = tmp_path / "response.webp"
    path.write_bytes(content)
    out = subprocess.run(
        [
            REAL_FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(path),
        ],
        check=True, capture_output=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    return stream["width"], stream["height"]


def everything_logged(caplog) -> str:
    return "\n".join(
        f"{r.name} {r.getMessage()} {r.args!r}" for r in caplog.records
    )


def log_everything(caplog) -> None:
    caplog.set_level(logging.DEBUG)
