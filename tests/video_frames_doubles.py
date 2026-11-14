"""Seeding and probes for the questions about a video with no text.

A frame's bytes name its millisecond key (``FRAME@<ms>``), so the images a
provider received can be read back as the times they were taken at, in order.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

import app.frame_cache as frame_cache
from app.workers.video_visual_selection import Candidate
from tests.file_question_vision_doubles import DRIVE
from tests.search_endpoint_db import patch_everywhere

MARKER = b"FRAME@"


def ms(seconds: float) -> int:
    return round(seconds * 1000)


def frame_bytes(key_ms: int) -> bytes:
    return MARKER + str(key_ms).encode()


def seed_text_less_video(
    db, file_id: str, *, path: Path, duration: float | None,
    drive: str = DRIVE, mime: str = "video/mp4", active: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"VIDEO-BYTES")
    db.file(
        file_id, drive=drive, mime=mime, file_path=str(path),
        file_size=11, duration=duration, active=active,
    )


def scenes(monkeypatch, times: list[float], *, duration: float | None) -> list[str]:
    """Make ``times`` the selected scene timestamps; return the files read."""
    candidates = [
        Candidate(f"clip{i:03d}", t, np.ones(4, dtype=np.float32))
        for i, t in enumerate(times)
    ]
    loaded: list[str] = []

    def _load(file_id):
        loaded.append(file_id)
        return list(candidates), duration

    def _select(given, *_args, **_kwargs):
        return sorted(given, key=lambda c: c.timestamp)

    patch_everywhere(monkeypatch, "_load_candidates", _load)
    patch_everywhere(monkeypatch, "select_candidates", _select)
    return loaded


def key_of(command) -> int:
    """The millisecond key of the cache entry an ffmpeg command writes."""
    return int(Path(command[-1]).name.split(".")[0])


class FFmpeg:
    """Stands in for ``subprocess.run`` in the frame cache.

    ``fail`` holds keys that end with a non-zero exit and ``hang`` keys whose
    run blocks until ``release`` (or ``hang_all``). Every start is recorded with
    the time it entered, and the peak number running at once is kept.
    """

    def __init__(self, *, fail=(), timeout=(), hang=(), hang_all=False, hold=0.0):
        self.fail = set(fail)
        self.timeout = set(timeout)
        self.hang = set(hang)
        self.hang_all = hang_all
        self.hold = hold
        self.starts: list[int] = []
        self.start_times: list[float] = []
        self.live = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.gates: dict[int, threading.Event] = {}
        self.entered = threading.Event()
        self.release_all = threading.Event()

    def gate(self, key: int) -> threading.Event:
        with self.lock:
            return self.gates.setdefault(key, threading.Event())

    def release(self, key: int) -> None:
        self.gate(key).set()

    def run(self, command, **kwargs):
        key = key_of(command)
        with self.lock:
            self.starts.append(key)
            self.start_times.append(time.monotonic())
            self.live += 1
            self.peak = max(self.peak, self.live)
        self.entered.set()
        try:
            if self.hang_all or key in self.hang:
                gate = self.gate(key)
                deadline = time.monotonic() + 60
                while not (gate.is_set() or self.release_all.is_set()):
                    if time.monotonic() > deadline:
                        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=60)
                    time.sleep(0.02)
            elif self.hold:
                time.sleep(self.hold)
            if key in self.timeout:
                raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=15)
            if key in self.fail:
                return subprocess.CompletedProcess(command, 1)
            output = Path(command[-1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(frame_bytes(key))
            return subprocess.CompletedProcess(command, 0)
        finally:
            with self.lock:
                self.live -= 1

    def install(self, monkeypatch) -> "FFmpeg":
        monkeypatch.setattr(frame_cache.subprocess, "run", self.run)
        return self


def decoded(payload: str) -> bytes:
    if payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


def sent_keys(wire) -> list[int]:
    """The millisecond keys of the images the provider received, in order."""
    out: list[int] = []
    for payload in wire.images:
        raw = decoded(payload)
        assert raw.startswith(MARKER), raw[:40]
        out.append(int(raw[len(MARKER):]))
    return out


def user_prompt(wire, index: int = -1) -> str:
    """The text of the last user message of one request, without the system prompt."""
    body = wire.requests[index]
    users = [m for m in body.get("messages", []) if m.get("role") == "user"]
    content = users[-1].get("content") if users else ""
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
    return content or ""


def listed_in_order(prompt: str, stamps: list[str]) -> bool:
    """True when ``[1]`` .. ``[m]`` each appear followed by their own stamp, in order."""
    cursor = 0
    for number, stamp in enumerate(stamps, start=1):
        marker = prompt.find(f"[{number}]", cursor)
        if marker == -1:
            return False
        at = prompt.find(stamp, marker)
        following = prompt.find(f"[{number + 1}]", marker)
        if at == -1 or (following != -1 and at > following):
            return False
        cursor = at
    return True


async def timed_post(app, *, path: str, body: dict, drive: str = DRIVE):
    """POST over raw ASGI and return ``(started, [(monotonic, bytes)], status)``."""
    queue: asyncio.Queue[dict] = asyncio.Queue()
    await queue.put({
        "type": "http.request", "body": json.dumps(body).encode(), "more_body": False,
    })
    received: list[tuple[float, bytes]] = []
    status: list[int] = []

    async def receive() -> dict:
        return await queue.get()

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            status.append(message["status"])
        elif message["type"] == "http.response.body":
            received.append((time.monotonic(), message.get("body", b"")))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"x-lit-drive", drive.encode()),
        ],
        "server": ("intelligence", 8100),
        "client": ("test", 1),
    }
    started = time.monotonic()
    await asyncio.wait_for(app(scope, receive, send), timeout=120)
    return started, received, (status[0] if status else None)
