"""Helpers for the stored lines of a frame read (SPEC-ADDON-134).

``ShapedEngine`` is the box engine of ``tests.frame_box_doubles`` with each box's
points scripted as given, so a test can hand the reader a quad in any corner
order, a skewed one, one past the frame's edges, or one that is not four points.
Recognition finds a box by its points rather than by its rectangle.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any

import numpy as np

from tests.frame_box_doubles import TICKER, grey_of, quad, state_of

Points = list[list[float]]


def norm(points: Points, width: int, height: int) -> list[float]:
    flat: list[float] = []
    for x, y in points:
        flat.append(round(min(max(x / width, 0.0), 1.0), 4))
        flat.append(round(min(max(y / height, 0.0), 1.0), 4))
    return flat


class ShapedEngine:
    """``states[k]`` lists ``(name, points)`` detected in state ``k``; the ticker is
    appended unless ``tick`` is False. Recognition answers ``texts[k][name]`` when
    scripted (``None`` drops the box), ``<name>#<k>`` otherwise."""

    def __init__(
        self,
        states: list[list[tuple[str, Points]]],
        texts: list[dict[str, str | None]] | None = None,
        *,
        tick: bool = True,
    ) -> None:
        self.states = states
        self.texts = texts or [{} for _ in states]
        self.tick = tick
        self.seen_shape: dict[int, tuple[int, int]] = {}
        self.recognised: dict[int, list[list[str]]] = {}
        self._lock = threading.Lock()

    def _boxes(self, k: int) -> list[tuple[str, Points]]:
        boxes = list(self.states[k])
        if self.tick:
            boxes.append(("tick", quad(TICKER)))
        return boxes

    def detect_boxes(self, _engine, image):
        k = state_of(image)
        with self._lock:
            self.seen_shape[k] = grey_of(image).shape
        return [np.asarray(points, dtype=np.float32) for _name, points in self._boxes(k)]

    def _name_of(self, k: int, box: Any) -> str:
        given = np.asarray(box, dtype=np.float64).reshape(-1)
        for name, points in self._boxes(k):
            scripted = np.asarray(points, dtype=np.float32).astype(np.float64).reshape(-1)
            if scripted.shape == given.shape and np.allclose(scripted, given, equal_nan=True):
                return name
        raise AssertionError(f"state {k}: the reader asked about a box it was not given: {box!r}")

    def recognise_boxes(self, _engine, image, boxes):
        k = state_of(image)
        names = [self._name_of(k, box) for box in boxes]
        with self._lock:
            self.recognised.setdefault(k, []).append(names)
        texts = self.texts[k]
        return [texts[name] if name in texts else f"{name}#{k}" for name in names]


def line_rows(db, file_id: str = "vid1") -> list[tuple]:
    return [
        tuple(row) for row in db.rows(
            "SELECT segment_index, start_time, end_time, lines_json FROM video_frame_lines "
            "WHERE file_id = :f ORDER BY segment_index",
            f=file_id,
        )
    ]


def all_line_rows(db) -> list[tuple]:
    return [tuple(row) for row in db.rows("SELECT * FROM video_frame_lines ORDER BY id")]


def decoded(lines_json: str) -> list[dict]:
    return json.loads(lines_json)


def texts_of(lines_json: str) -> list[str]:
    return [line["text"] for line in decoded(lines_json)]


def quads_of(lines_json: str) -> dict[str, list[float]]:
    return {line["text"]: line["quad"] for line in decoded(lines_json)}


def chunk_rows(db, file_id: str = "vid1") -> list[tuple]:
    return [
        tuple(row) for row in db.rows(
            "SELECT chunk_index, start_time, end_time, text FROM video_frame_texts "
            "WHERE file_id = :f ORDER BY chunk_index",
            f=file_id,
        )
    ]


def seed_line_row(
    db, file_id: str = "vid1", segment_index: int = 0, *,
    start: float = 0.0, end: float = 10.0, lines_json: str = '[{"text":"OLD","quad":[0,0,1,0,1,1,0,1]}]',
) -> None:
    db.execute(
        "INSERT INTO video_frame_lines(file_id, segment_index, start_time, end_time, lines_json) "
        "VALUES(:f, :i, :s, :e, :j)",
        f=file_id, i=segment_index, s=start, e=end, j=lines_json,
    )


def numbers_in(message: str) -> list[float]:
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", message)]
