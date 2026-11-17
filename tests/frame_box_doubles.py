"""Scripted videos and a box-level OCR engine for the frame reader (SPEC-ADDON-123).

A scene is a list of states, each held for three seconds, so the reader reads one
frame per state: the lower band flips between two ramps on every state, which
moves the difference hash far past the threshold, and the 3-second gap does the
rest. The clip is lossless (FFV1, planar RGB) and every painted pixel is grey,
so the grayscale the reader compares is exactly the value painted.

The engine is faked at ``app.workers.ocr.detect_boxes`` / ``recognise_boxes``.
Detection answers the boxes scripted for the state it sees, recognition answers
``<name>#<state>`` unless the script says otherwise, so a reused result shows the
state it was recognised in. A ticker box, last in detection order, changes its
pixels on every state; no two read frames therefore share a text.
"""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tests.search_endpoint_db import patch_everywhere

WIDTH, HEIGHT = 480, 400
BACKGROUND = 128
SECONDS_PER_STATE = 3
MARKER = (460, 0, 480, 8)
TICKER = (300, 260, 380, 280)
FLIP_BAND = (0, 320, WIDTH, HEIGHT)

Rect = tuple[int, int, int, int]


def quad(rect: Rect) -> list[list[int]]:
    x0, y0, x1, y1 = rect
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _rect_key(box: Any) -> Rect:
    points = np.asarray(box, dtype=float).reshape(-1, 2)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))


@dataclass
class State:
    """One three-second state: grey rectangles painted, boxes detected, text overrides."""

    paint: list[tuple[Rect, int]] = field(default_factory=list)
    boxes: list[tuple[str, Rect]] = field(default_factory=list)
    texts: dict[str, str | None] = field(default_factory=dict)


def _picture(states: list[State], k: int) -> np.ndarray:
    grey = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.uint8)
    for (x0, y0, x1, y1), value in states[k].paint:
        grey[y0:y1, x0:x1] = value
    x0, y0, x1, y1 = TICKER
    grey[y0 - 8:y1 + 8, x0 - 8:x1 + 8] = 60 if k % 2 == 0 else 190
    x0, y0, x1, y1 = MARKER
    grey[y0:y1, x0:x1] = 30 + 20 * k
    x0, y0, x1, y1 = FLIP_BAND
    ramp = np.linspace(0, 255, x1 - x0).astype(np.uint8)
    grey[y0:y1, x0:x1] = ramp if k % 2 == 0 else ramp[::-1]
    return np.stack([grey] * 3, axis=-1)


def encode_scene(path: Path, states: list[State]) -> Path:
    process = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", "1",
            "-i", "-", "-c:v", "ffv1", "-pix_fmt", "gbrp", "-g", "1", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for k in range(len(states)):
        frame = _picture(states, k).tobytes()
        for _ in range(SECONDS_PER_STATE):
            process.stdin.write(frame)
    process.stdin.close()
    assert process.wait() == 0
    return path


def grey_of(image: Any) -> np.ndarray:
    if hasattr(image, "convert"):
        return np.asarray(image.convert("L"))
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    rgb = array[..., :3].astype(np.float64)
    return np.asarray(
        rgb[..., 0] * 299 / 1000 + rgb[..., 1] * 587 / 1000 + rgb[..., 2] * 114 / 1000,
    ).astype(np.uint8)


def state_of(image: Any) -> int:
    x0, y0, x1, y1 = MARKER
    value = int(grey_of(image)[(y0 + y1) // 2, (x0 + x1) // 2])
    return round((value - 30) / 20)


class EngineDown(RuntimeError):
    pass


class DirectEngine:
    """What ``_get_engine`` hands out: the reader must not call it directly."""

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("the frame reader called the engine outside detect/recognise")


class BoxEngine:
    """Fakes ``detect_boxes`` and ``recognise_boxes`` from a scene script."""

    def __init__(self, states: list[State]) -> None:
        self.states = states
        self.log: list[tuple] = []
        self.detect_raises: set[int] = set()
        self.recognise_raises: set[int] = set()
        self.on_detect: Callable[[int], None] | None = None
        self.on_recognise: Callable[[int], None] | None = None
        self.seen_grey: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def detect_boxes(self, _engine, image):
        k = state_of(image)
        with self._lock:
            self.log.append(("detect", k))
            self.seen_grey[k] = grey_of(image).copy()
        if self.on_detect is not None:
            self.on_detect(k)
        if k in self.detect_raises:
            raise EngineDown(f"detection failed on state {k}")
        boxes = list(self.states[k].boxes) + [("tick", TICKER)]
        return [np.asarray(quad(rect), dtype=np.float32) for _name, rect in boxes]

    def recognise_boxes(self, _engine, image, boxes):
        k = state_of(image)
        by_rect = {rect: name for name, rect in self.states[k].boxes}
        by_rect[TICKER] = "tick"
        names = [by_rect[_rect_key(box)] for box in boxes]
        with self._lock:
            self.log.append(("recognise", k, names))
        if self.on_recognise is not None:
            self.on_recognise(k)
        if k in self.recognise_raises:
            raise EngineDown(f"recognition failed on state {k}")
        texts = self.states[k].texts
        return [texts[name] if name in texts else f"{name}#{k}" for name in names]

    def detects(self) -> list[int]:
        return [entry[1] for entry in self.log if entry[0] == "detect"]

    def recognised(self) -> dict[int, list[list[str]]]:
        calls: dict[int, list[list[str]]] = {}
        for entry in self.log:
            if entry[0] == "recognise":
                calls.setdefault(entry[1], []).append(entry[2])
        return calls


class SimpleBoxEngine:
    """One box over the whole picture, named after its brightness band."""

    def __init__(self) -> None:
        self.detects = 0
        self.recognises = 0
        self.recognise_raises_on: int | None = None

    def detect_boxes(self, _engine, image):
        self.detects += 1
        height, width = grey_of(image).shape
        return [np.asarray(quad((0, 0, width - 1, height - 1)), dtype=np.float32)]

    def recognise_boxes(self, _engine, image, boxes):
        self.recognises += 1
        if self.recognise_raises_on is not None and self.recognises == self.recognise_raises_on:
            raise EngineDown("recognition failed")
        rgb = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
        red, blue = float(rgb[..., 0].mean()), float(rgb[..., 2].mean())
        band = "WARM" if red > blue else "COOL"
        return [f"{band}{int(grey_of(image).mean()) // 32}" for _ in boxes]


def install(monkeypatch, fake) -> None:
    import app.workers.ocr as ocr

    for name in ("detect_boxes", "recognise_boxes"):
        monkeypatch.setattr(ocr, name, getattr(fake, name), raising=False)
        patch_everywhere(monkeypatch, name, getattr(fake, name))


def frame_texts(db, file_id: str = "vid1") -> list[str]:
    return [
        row[0] for row in db.rows(
            "SELECT text FROM video_frame_texts WHERE file_id = :f ORDER BY chunk_index",
            f=file_id,
        )
    ]
