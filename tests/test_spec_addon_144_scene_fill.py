"""SPEC-ADDON-144: a native video's scene candidates are its scene changes plus a fill
that leaves no gap longer than ``min_interval``, thinned evenly over the whole video
when they number more than ``max_frames``.

The videos are flat colours joined by hard cuts, so ffmpeg's scene pass finds a frame
at each cut and nowhere else. The CLIP model is a stand-in.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

import app.database as database
from app.config import FrameExtractionConfig, IndexingConfig
from app.models import IndexedFile
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.search_endpoint_db import patch_everywhere
from tests.test_spec_addon_111_frame_scan_task import _encode
from tests.vec_partition_db import search_index  # noqa: F401 - fixture

pytestmark = [
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed"),
    pytest.mark.usefixtures("pillow_real"),
]

# The gradient video of the frame-scan tests: cuts at 5, 12 and 25 s, 40 s long.
GRADIENT_SECONDS = 40.0
# Anchors 0, 5, 12, 25; with I = 10 the gaps 12-25 and 25-40 each get one fill time.
GRADIENT_TIMES = [0.0, 5.0, 12.0, 18.5, 25.0, 32.5]
GRADIENT_FILL = [0.0, 18.5, 32.5]


def _colours(path: Path, parts: list[tuple[str, float]]) -> Path:
    inputs: list[str] = []
    for colour, seconds in parts:
        inputs += ["-f", "lavfi", "-i", f"color={colour}:s=64x48:d={seconds}:r=10"]
    joined = "".join(f"[{n}:v]" for n in range(len(parts)))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs,
         "-filter_complex", f"{joined}concat=n={len(parts)}:v=1[v]",
         "-map", "[v]", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("scene-fill-media")
    return {
        "gradient": _encode(root / "gradient.mp4"),
        "early-cut": _colours(root / "early.mp4", [("red", 0.5), ("blue", 20.0)]),
        "still-10": _colours(root / "still10.mp4", [("red", 10.0)]),
        "still-30": _colours(root / "still30.mp4", [("red", 30.0)]),
        "still-31": _colours(root / "still31.mp4", [("red", 31.0)]),
    }


async def _allow(*_args, **_kwargs) -> bool:
    return True


def _seconds(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class Leg:
    """The CLIP leg of one native video over a real search DB."""

    def __init__(self, index, monkeypatch, tmp_path, make_settings):
        self.index = index
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        self.make_settings = make_settings
        self.dim = index.dim("vec_clip")
        self.embedded = 0
        self.fail_embed_at: int | None = None
        self.extracted: list[float] = []
        self.fail_extract_at: float | None = None
        self.path: Path | None = None
        self._original_extract = None

    def video(self, source: Path, duration: float | None, file_id: str = "vid1") -> None:
        self.path = self.tmp_path / f"{file_id}{source.suffix}"
        shutil.copyfile(source, self.path)
        with database.get_search_db() as s:
            s.add(IndexedFile(
                file_id=file_id, drive="family", filename=self.path.name,
                file_path=str(self.path), file_type="video", mime_type="video/mp4",
                file_size=1, active=True, duration=duration,
            ))

    def _embed(self, *_args, **_kwargs):
        self.embedded += 1
        if self.fail_embed_at == self.embedded:
            raise RuntimeError("one frame could not be embedded")
        return np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=np.float32)

    def _extract(self, original):
        def wrapped(*args, **kwargs):
            values = list(args) + list(kwargs.values())
            at = next((s for s in map(_seconds, values) if s is not None), None)
            if at is not None:
                self.extracted.append(at)
            if (
                at is not None and self.fail_extract_at is not None
                and math.isclose(at, self.fail_extract_at, abs_tol=0.01)
            ):
                gone = str(self.tmp_path / "no-such-video.mp4")
                args = tuple(
                    gone if isinstance(a, (str, Path)) and str(a) == str(self.path) else a
                    for a in args
                )
                kwargs = {
                    k: gone if isinstance(v, (str, Path)) and str(v) == str(self.path) else v
                    for k, v in kwargs.items()
                }
            return original(*args, **kwargs)

        return wrapped

    def run(self, file_id: str = "vid1", *, min_interval: float | None = 10,
            max_frames: int = 500):
        import app.workers.clip as clip

        frame_extraction = (
            FrameExtractionConfig(max_frames=max_frames)
            if min_interval is None
            else FrameExtractionConfig(min_interval=min_interval, max_frames=max_frames)
        )
        settings = self.make_settings(
            allowed_base_dirs=(str(self.tmp_path) + os.sep,),
            indexing=IndexingConfig(frame_extraction=frame_extraction),
        )
        patch_everywhere(self.monkeypatch, "settings", settings)
        patch_everywhere(self.monkeypatch, "embed_image", self._embed)
        patch_everywhere(self.monkeypatch, "is_feature_enabled", _allow)
        self.monkeypatch.setattr("app.policy_client.is_feature_enabled", _allow)
        self.monkeypatch.setattr(
            clip, "_generate_blip_caption_if_needed", lambda *a, **k: None,
        )
        if self._original_extract is None:
            self._original_extract = clip.extract_frame
        patch_everywhere(
            self.monkeypatch, "extract_frame", self._extract(self._original_extract),
        )
        return clip._index_clip_sync(file_id)

    def rows(self, file_id: str = "vid1") -> list[tuple[str, float, float]]:
        with self.index.engine.connect() as conn:
            return [tuple(r) for r in conn.execute(text(
                "SELECT id, timestamp_start, timestamp_end FROM embeddings "
                "WHERE file_id = :f AND embedding_type = 'clip' ORDER BY timestamp_start"
            ), {"f": file_id})]

    def times(self, file_id: str = "vid1") -> list[float]:
        return [t for _, t, _ in self.rows(file_id)]


@pytest.fixture()
def leg(search_index, monkeypatch, tmp_path, make_settings):  # noqa: F811
    return Leg(search_index, monkeypatch, tmp_path, make_settings)


def _approx(times):
    return pytest.approx(times, abs=0.15)


# ---------------------------------------------------------------------------
# Item 1 steps 1-2, I1: scene frames plus a fill with no gap longer than I
# ---------------------------------------------------------------------------


def test_spec_addon_144_i1_the_fill_splits_each_long_gap_evenly(leg, media):
    leg.video(media["gradient"], GRADIENT_SECONDS)

    assert leg.run(min_interval=10) is True

    assert leg.times() == _approx(GRADIENT_TIMES)
    assert sorted(leg.extracted) == _approx(GRADIENT_FILL)
    for _, start, end in leg.rows():
        assert end == pytest.approx(start + 30)
    assert set(r[0] for r in leg.rows()) <= set(leg.index.vectors("vec_clip"))


BOUNDARIES = {
    "still-10": (10.0, [0.0]),
    "still-30": (30.0, [0.0, 10.0, 20.0]),
    "still-31": (31.0, [0.0, 7.75, 15.5, 23.25]),
    "early-cut": (20.5, [0.5, 10.5]),
}


def test_spec_addon_144_i1_the_fill_times_follow_the_anchors_and_the_duration(leg, media):
    """A gap of exactly I gets no fill; ceil(gap / I) - 1 times split a longer one; a
    scene frame below 1 s replaces the anchor 0 s."""
    for name, (duration, _) in BOUNDARIES.items():
        leg.video(media[name], duration, file_id=name)

    for name in BOUNDARIES:
        assert leg.run(name, min_interval=10) is True

    got = {name: leg.times(name) for name in BOUNDARIES}
    assert got == {name: _approx(expected) for name, (_, expected) in BOUNDARIES.items()}
    for name, (duration, _) in BOUNDARIES.items():
        times = got[name]
        assert times[0] < 1.0
        assert all(b - a <= 10.0 + 0.15 for a, b in zip(times, times[1:]))
        assert duration - times[-1] <= 10.0 + 0.15


# ---------------------------------------------------------------------------
# Item 1 steps 3-4, I2: thinning on times, at most M fill extractions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "max_frames, expected",
    [
        pytest.param(1, [0.0], id="M-1"),
        pytest.param(2, [0.0, 18.5], id="M-2"),
        pytest.param(3, [0.0, 12.0, 25.0], id="M-3"),
        pytest.param(4, [0.0, 5.0, 18.5, 25.0], id="M-4"),
        pytest.param(5, [0.0, 5.0, 12.0, 18.5, 25.0], id="M-one-below-count"),
        pytest.param(6, GRADIENT_TIMES, id="M-equal-to-count"),
        pytest.param(7, GRADIENT_TIMES, id="M-above-count"),
    ],
)
def test_spec_addon_144_i2_more_times_than_max_frames_are_thinned_over_the_whole_video(
    leg, media, max_frames, expected,
):
    leg.video(media["gradient"], GRADIENT_SECONDS)

    assert leg.run(min_interval=10, max_frames=max_frames) is True

    assert leg.times() == _approx(expected)
    kept_fill = [t for t in expected if t in GRADIENT_FILL]
    assert sorted(leg.extracted) == _approx(kept_fill)
    assert len(leg.extracted) <= max_frames


# ---------------------------------------------------------------------------
# §2: a frame that fails is left out and not replaced
# ---------------------------------------------------------------------------


def test_spec_addon_144_a_fill_time_whose_extraction_fails_is_left_out_and_not_replaced(
    leg, media,
):
    leg.video(media["gradient"], GRADIENT_SECONDS)
    leg.fail_extract_at = 18.5

    assert leg.run(min_interval=10, max_frames=4) is True

    assert leg.times() == _approx([0.0, 5.0, 25.0])
    assert sorted(leg.extracted) == _approx([0.0, 18.5])


def test_spec_addon_144_a_frame_that_fails_to_embed_is_left_out(leg, media):
    leg.video(media["gradient"], GRADIENT_SECONDS)
    leg.fail_embed_at = 2

    assert leg.run(min_interval=10) is True

    times = leg.times()
    assert len(times) == len(GRADIENT_TIMES) - 1
    assert all(any(abs(t - e) <= 0.15 for e in GRADIENT_TIMES) for t in times)


# ---------------------------------------------------------------------------
# I13: the default I is 10; a configured value is used as it is
# ---------------------------------------------------------------------------


def test_spec_addon_144_i13_without_a_configured_min_interval_the_fill_uses_10_s(leg, media):
    leg.video(media["gradient"], GRADIENT_SECONDS)

    assert leg.run(min_interval=None) is True

    assert leg.times() == _approx(GRADIENT_TIMES)


def _load(tmp_path, monkeypatch, config: Path):
    from app.config import load_settings

    monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HOMEVAULT_DB_PATH", str(tmp_path / "hv.db"))
    monkeypatch.setenv("SEARCH_CONFIG_PATH", str(config))
    return load_settings().indexing.frame_extraction.min_interval


def test_spec_addon_144_i13_the_configured_value_wins_and_its_absence_is_10(
    tmp_path, monkeypatch,
):
    absent = tmp_path / "absent.yml"
    absent.write_text("indexing:\n  frame_extraction:\n    max_frames: 500\n")
    thirty = tmp_path / "thirty.yml"
    thirty.write_text("indexing:\n  frame_extraction:\n    min_interval: 30\n")
    example = Path(__file__).resolve().parent.parent / "search-config.yml.example"

    assert _load(tmp_path, monkeypatch, absent) == 10
    assert _load(tmp_path, monkeypatch, thirty) == 30
    assert _load(tmp_path, monkeypatch, example) == 10
