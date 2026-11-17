"""SPEC-ADDON-121: a YouTube fetch reads the storyboard a reference is missing.

The fetch is the ``LoftWorld`` of SPEC-ADDON-117 (real search DB, the gradient
rendition served over HTTP); its lookup answer also names an ``sb0`` storyboard
whose sheets are the flat-colour PNGs of the SPEC-ADDON-078 tests, so a tile is
told from a scan frame by its colour.
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from sqlalchemy import text

import app.database as database
from app.config import FeaturesConfig, LLMConfig
from app.models import Embedding
from tests.loft_media_world import GIB, FreeSpace, LoftWorld, fetch
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.search_endpoint_db import patch_everywhere
from tests.test_spec_addon_078_youtube_storyboard import (
    TILE_MARK,
    serve_storyboard,
    storyboard_fps,
)
from tests.test_spec_addon_079_scene_tiles import leftover_dirs, tile_bytes, tile_names
from tests.test_spec_addon_111_frame_scan_task import _encode
from tests.vec_partition_db import search_index  # noqa: F401 - fixture
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    describe,
    needs_ffmpeg,
    video_format,
    youtube,
)

pytestmark = [needs_ffmpeg, pytest.mark.usefixtures("pillow_real")]

VIDEO_SECONDS = 40.0
KEPT_SECONDS = [0.0, 10.0, 20.0, 30.0]
LOOKUP_FAILURE = "ERROR: [youtube] x: Video unavailable"


def new_id() -> str:
    return "sb" + uuid.uuid4().hex[:10]


@pytest.fixture(scope="module")
def gradient(tmp_path_factory):
    root = tmp_path_factory.mktemp("loft-storyboard")
    return _encode(root / "switches.mp4")


@pytest.fixture(autouse=True)
def fresh_worker_state():
    from app.workers import loft_media

    loft_media.reset_worker_state()
    yield
    loft_media.reset_worker_state()


class TileEmbedder:
    """Stands in for CLIP; a storyboard tile is a flat colour whose blue is ``TILE_MARK``."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.tiles: list[str] = []
        self.frames = 0
        self.fail_tiles = False
        self.fail_in_thread: str | None = None
        self.failed = 0

    @staticmethod
    def is_tile(image) -> bool:
        red, green, blue = image.convert("RGB").getpixel(
            (image.size[0] // 2, image.size[1] // 2)
        )
        return blue == TILE_MARK and not (red == green == blue)

    def __call__(self, image, *_args, **_kwargs):
        if self.is_tile(image):
            thread = threading.current_thread().name
            if self.fail_tiles or self.fail_in_thread == thread:
                self.failed += 1
                raise RuntimeError("CLIP failed")
            self.tiles.append(thread)
        else:
            self.frames += 1
        return np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=np.float32)


class World:
    def __init__(self, loft: LoftWorld, gradient) -> None:
        self.loft = loft
        self.youtube = loft.youtube
        self.monkeypatch = loft.monkeypatch
        self.gradient = gradient
        self.free = FreeSpace(100 * GIB)
        self.monkeypatch.setattr(shutil, "disk_usage", self.free)
        self.image = sys.modules["PIL.Image"]
        patch_everywhere(self.monkeypatch, "Image", self.image)
        self.embedder = TileEmbedder(loft.index.dim("vec_clip"))
        patch_everywhere(self.monkeypatch, "embed_image", self.embedder)
        from app.workers import clip as clip_worker

        self.monkeypatch.setattr(
            clip_worker, "_generate_blip_caption_if_needed", lambda *a, **k: None,
        )
        self.made = 0

    def configure(self, **kwargs) -> None:
        self.loft.configure(**kwargs)
        patch_everywhere(self.monkeypatch, "Image", self.image)
        patch_everywhere(self.monkeypatch, "embed_image", self.embedder)
        self.monkeypatch.setattr(shutil, "disk_usage", self.free)

    def answer(self, *, storyboard: bool = True, duration: float | None = VIDEO_SECONDS,
               fps: float | None = None, rendition: bool = True) -> dict | None:
        """The lookup answer every lookup from now on gets: the rendition and ``sb0``."""
        formats = []
        if rendition:
            path = "/v/rendition"
            self.youtube.server.files[path] = self.gradient.read_bytes()
            formats.append(video_format(
                "rendition", 720, url=f"{self.youtube.base}{path}?sig=x",
            ))
        board = None
        if storyboard:
            board, _ = serve_storyboard(
                self.youtube, self.image, tiles=5, rows=2, columns=2,
                fps=fps if fps is not None else storyboard_fps(VIDEO_SECONDS),
            )
            formats.append(board)
        self.youtube.set_ytdlp(describe=describe(formats, duration=duration))
        return board

    def add(self, file_id: str | None = None, **flags) -> str:
        self.made += 1
        file_id = file_id or new_id()
        url = f"https://www.youtube.com/watch?v=v{self.made:010d}"
        self.loft.add_loft(file_id, {"provider": "youtube", "url": url}, **flags)
        return file_id

    def clip_seconds(self, file_id: str, kind: str = "clip") -> list[float]:
        with self.loft.index.engine.connect() as conn:
            return [r[0] for r in conn.execute(text(
                "SELECT timestamp_start FROM embeddings WHERE file_id = :f "
                "AND embedding_type = :k ORDER BY timestamp_start, id"
            ), {"f": file_id, "k": kind})]

    def clip_ids(self, file_id: str) -> list[str]:
        with self.loft.index.engine.connect() as conn:
            return sorted(r[0] for r in conn.execute(text(
                "SELECT id FROM embeddings WHERE file_id = :f AND embedding_type = 'clip'"
            ), {"f": file_id}))

    def vectors(self) -> set[str]:
        return set(self.loft.index.vectors("vec_clip"))

    def other_rows(self, file_id: str) -> set[str]:
        with self.loft.index.engine.connect() as conn:
            return {r[0] for r in conn.execute(text(
                "SELECT id FROM embeddings WHERE file_id = :f AND embedding_type != 'clip'"
            ), {"f": file_id})}

    def clip_indexed(self, file_id: str) -> bool:
        return bool(self.loft.value(file_id, "clip_indexed"))

    def add_row(self, file_id: str, kind: str, seconds: float | None = None) -> str:
        row_id = f"old_{kind}_{file_id}"
        with database.get_search_db() as s:
            s.add(Embedding(
                id=row_id, file_id=file_id, embedding_type=kind,
                timestamp_start=seconds, vector_table="vec_clip",
            ))
        self.loft.index.put_vector(
            "vec_clip", row_id,
            np.ones(self.loft.index.dim("vec_clip"), dtype=np.float32),
            partition="clip",
        )
        return row_id

    def sheets(self) -> list[dict]:
        return self.youtube.fetched("/sb/")

    def leg(self, file_id: str):
        from app.workers import clip as clip_worker

        return clip_worker._index_clip_sync(file_id)


@pytest.fixture()
def world(search_index, youtube, monkeypatch, tmp_path, make_settings, gradient):  # noqa: F811
    return World(
        LoftWorld(search_index, youtube, monkeypatch, tmp_path, make_settings), gradient,
    )


# ---------------------------------------------------------------------------
# Item 1.1-1.2, I1: the step writes what the leg writes, from the fetch's lookup
# ---------------------------------------------------------------------------


def test_spec_addon_121_i1_a_fetch_writes_the_candidates_and_tiles_the_leg_writes(world):
    world.answer()
    fetched, by_leg = world.add(), world.add()

    fetch(fetched)

    assert world.loft.lookups() == 1
    assert world.clip_seconds(fetched) == pytest.approx(KEPT_SECONDS)
    assert world.clip_indexed(fetched) is True
    assert world.loft.flags(fetched)[0] is True

    assert world.leg(by_leg) is True
    assert world.clip_seconds(fetched) == world.clip_seconds(by_leg)
    assert tile_names(fetched) == tile_names(by_leg)
    assert len(tile_names(fetched)) == len(KEPT_SECONDS)
    assert tile_bytes(fetched) == tile_bytes(by_leg)
    assert leftover_dirs(fetched) == []
    vectors = world.vectors()
    assert set(world.clip_ids(fetched)) <= vectors
    assert set(world.clip_ids(by_leg)) <= vectors
    assert len(world.clip_ids(fetched)) == len(world.clip_ids(by_leg))


def test_spec_addon_121_i1_the_sheets_are_read_before_the_download(world):
    world.answer()
    file_id = world.add()

    fetch(file_id)

    sheets, downloads = world.sheets(), world.loft.downloads()
    assert sheets and downloads
    assert max(r["t"] for r in sheets) <= min(r["t"] for r in downloads)


def test_spec_addon_121_revised_117_i3_a_fetch_stopped_by_the_disk_guard_keeps_its_candidates(
    world,
):
    world.answer()
    file_id = world.add()
    world.loft.set(file_id, loft_fetch_failures=1)
    world.free.free = 1 * GIB

    fetch(file_id)

    assert world.loft.downloads() == []
    assert world.loft.lookups() == 1
    assert world.clip_seconds(file_id) == pytest.approx(KEPT_SECONDS)
    assert len(tile_names(file_id)) == len(KEPT_SECONDS)
    assert world.clip_indexed(file_id) is True
    assert world.loft.flags(file_id) == (False, True, True)
    assert world.loft.failures(file_id) == 1


# ---------------------------------------------------------------------------
# Item 1.2: the leg's once-per-process retry gate is neither consulted nor recorded
# ---------------------------------------------------------------------------


def test_spec_addon_121_item_1_2_a_fetch_runs_the_step_after_the_leg_failed_in_this_process(
    world,
):
    file_id = world.add()
    world.youtube.set_ytdlp(fail=LOOKUP_FAILURE)
    assert world.leg(file_id) is False
    assert world.leg(file_id) is False

    world.answer()
    fetch(file_id)

    assert world.loft.flags(file_id)[0] is True
    assert world.clip_seconds(file_id) == pytest.approx(KEPT_SECONDS)
    assert world.clip_indexed(file_id) is True


def test_spec_addon_121_item_1_2_a_failed_step_does_not_stop_the_leg_retrying(world):
    world.answer()
    file_id = world.add()
    world.youtube.override("/sb/sheet1.png", status=500)

    fetch(file_id)

    assert world.sheets(), "the fetch read no storyboard"
    assert world.clip_seconds(file_id) == []
    lookups = world.loft.lookups()

    world.youtube.override("/sb/sheet1.png")
    assert world.leg(file_id) is True

    assert world.loft.lookups() == lookups + 1
    assert world.clip_seconds(file_id) == pytest.approx(KEPT_SECONDS)


# ---------------------------------------------------------------------------
# Item 1.3, I5: the step and the leg never run for one file at once
# ---------------------------------------------------------------------------


def _sheet_runs(requests: list[dict]) -> list[list[str]]:
    paths = [r["path"].rsplit("/", 1)[1] for r in sorted(requests, key=lambda r: r["t"])]
    return [paths[i:i + 2] for i in range(0, len(paths), 2)]


def _wait_until(condition, seconds: float = 10.0) -> bool:
    limit = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > limit:
            return False
        time.sleep(0.02)
    return True


def test_spec_addon_121_i5_a_leg_started_during_the_fetchs_step_waits_for_it(world):
    world.answer()
    file_id = world.add()
    world.free.free = 1 * GIB
    world.youtube.override("/sb/sheet0.png", delay=2.5)

    fetcher = threading.Thread(target=fetch, args=(file_id,), name="fetch")
    fetcher.start()
    assert _wait_until(lambda: world.sheets()), "the fetch read no storyboard"
    leg_started = time.time()
    leg = threading.Thread(target=world.leg, args=(file_id,), name="leg")
    leg.start()
    fetcher.join(30)
    leg.join(30)

    sheets = world.sheets()
    assert any(r["t"] < leg_started for r in sheets)
    assert all(run == ["sheet0.png", "sheet1.png"] for run in _sheet_runs(sheets))
    assert world.clip_seconds(file_id) == pytest.approx(KEPT_SECONDS)
    assert len(tile_names(file_id)) == len(KEPT_SECONDS)
    assert leftover_dirs(file_id) == []


def test_spec_addon_121_i5_a_fetch_started_during_the_leg_waits_and_then_runs_its_step(world):
    """The leg's tiles fail, so the reference still has no candidates when the fetch goes on."""
    world.answer()
    file_id = world.add()
    world.free.free = 1 * GIB
    world.embedder.fail_in_thread = "leg"
    world.youtube.override("/sb/sheet0.png", delay=2.5)

    leg = threading.Thread(target=world.leg, args=(file_id,), name="leg")
    leg.start()
    assert _wait_until(lambda: world.sheets()), "the leg read no storyboard"
    fetcher = threading.Thread(target=fetch, args=(file_id,), name="fetch")
    fetcher.start()
    leg.join(30)
    fetcher.join(30)

    assert world.embedder.failed >= 1, "the leg's tiles did not fail"
    sheets = world.sheets()
    assert len(sheets) == 4
    assert all(run == ["sheet0.png", "sheet1.png"] for run in _sheet_runs(sheets))
    assert world.clip_seconds(file_id) == pytest.approx(KEPT_SECONDS)
    assert len(tile_names(file_id)) == len(KEPT_SECONDS)
    assert leftover_dirs(file_id) == []


# ---------------------------------------------------------------------------
# Item 1.4: the scan that sets its flag leaves an eligible reference
# ---------------------------------------------------------------------------


def _visual_index(monkeypatch, make_settings, use_llm) -> None:
    for name in ("app.workers.video_visual", "app.routers.video_visual"):
        monkeypatch.setattr(f"{name}.get_search_db", database.get_search_db, raising=False)
        monkeypatch.setattr(f"{name}.get_search_db_read", database.get_search_db_read)
    settings = make_settings(
        features=FeaturesConfig(video_visual_index="manual"),  # type: ignore[call-arg]
        llm=LLMConfig(
            provider="openai_compatible", base_url="http://test",
            model="gemma2:27b", vision_model="llava:13b",
        ),
    )
    for name in ("app.config", "app.workers.video_visual", "app.routers.video_visual"):
        monkeypatch.setattr(f"{name}.settings", settings)
    monkeypatch.setattr(
        "app.workers.video_visual.is_file_feature_enabled",
        AsyncMock(return_value=True), raising=False,
    )
    monkeypatch.setattr(
        "app.routers.video_visual.is_feature_enabled", AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "app.workers.video_visual.emit_video_visual_event", AsyncMock(return_value=None),
    )
    index_manager = MagicMock()
    index_manager.prioritize = AsyncMock(return_value=True)
    monkeypatch.setattr("app.dependencies.get_index_manager", lambda: index_manager)
    use_llm(MagicMock())


async def test_spec_addon_121_item_1_4_a_fetched_reference_is_eligible_for_a_visual_index(
    world, make_settings, use_llm,
):
    import asyncio

    from app.workers.video_visual import VideoVisualWorker

    world.answer()
    file_id = world.add()
    await asyncio.to_thread(fetch, file_id)
    assert world.loft.flags(file_id)[0] is True

    _visual_index(world.monkeypatch, make_settings, use_llm)
    result = await VideoVisualWorker().enqueue(file_id, requested_by="manual")

    assert result["accepted"] is True, result


# ---------------------------------------------------------------------------
# §3, I2: a reference with candidates gets no storyboard work
# ---------------------------------------------------------------------------


def test_spec_addon_121_i2_only_a_reference_without_clip_rows_gets_the_step(world):
    world.answer()
    has_candidates = world.add()
    world.add_row(has_candidates, "clip", 5.0)
    thumbnail_only = world.add()
    world.add_row(thumbnail_only, "clip_thumbnail")
    vectors = world.vectors()

    fetch(has_candidates)

    assert world.sheets() == []
    assert world.vectors() - vectors <= world.other_rows(has_candidates)
    assert world.clip_ids(has_candidates) == [f"old_clip_{has_candidates}"]
    assert world.clip_indexed(has_candidates) is False
    assert tile_names(has_candidates) == []
    assert world.loft.flags(has_candidates)[0] is True

    fetch(thumbnail_only)

    assert world.sheets() != []
    assert world.clip_seconds(thumbnail_only) == pytest.approx(KEPT_SECONDS)
    assert world.clip_seconds(thumbnail_only, "clip_thumbnail") == [None]


# ---------------------------------------------------------------------------
# §2: no storyboard or no duration is a definite result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(dict(storyboard=False), id="no-sb0"),
        pytest.param(dict(fps=0.0), id="sb0-without-fps"),
        pytest.param(dict(duration=None), id="no-duration"),
    ],
)
def test_spec_addon_121_section_2_no_storyboard_sets_clip_indexed_and_writes_nothing(
    world, answer,
):
    world.answer(**answer)
    file_id = world.add()

    fetch(file_id)

    assert world.loft.lookups() == 1
    assert world.sheets() == []
    assert world.clip_seconds(file_id) == []
    assert tile_names(file_id) == []
    assert world.clip_indexed(file_id) is True
    assert world.loft.flags(file_id)[0] is True


def _storyboard_lines(caplog, file_id: str) -> list[str]:
    return [
        r.getMessage().replace(file_id, "<file>")
        for r in caplog.records
        if file_id in r.getMessage() and "storyboard" in r.getMessage().lower()
    ]


def test_spec_addon_121_section_2_each_fetch_of_a_reference_without_a_storyboard_logs_the_absence(
    world, caplog,
):
    world.answer(storyboard=False)
    control = world.add()
    with caplog.at_level("DEBUG"):
        assert world.leg(control) is True
    expected = _storyboard_lines(caplog, control)
    assert expected, "the leg logged no storyboard line for the control reference"

    file_id = world.add()
    lookups = world.loft.lookups()
    caplog.clear()
    with caplog.at_level("DEBUG"):
        fetch(file_id)
    first = _storyboard_lines(caplog, file_id)
    world.loft.set(file_id, frame_scan_indexed=False)
    caplog.clear()
    with caplog.at_level("DEBUG"):
        fetch(file_id)
    second = _storyboard_lines(caplog, file_id)

    assert world.loft.lookups() == lookups + 2
    assert sorted(first) == sorted(expected)
    assert sorted(second) == sorted(expected)
    assert world.clip_seconds(file_id) == []
    assert world.clip_indexed(file_id) is True


# ---------------------------------------------------------------------------
# §2, I3: a failed step changes nothing and the fetch goes on
# ---------------------------------------------------------------------------


def _break_the_step(world, how: str) -> None:
    if how == "sheet-500":
        world.youtube.override("/sb/sheet1.png", status=500)
    elif how == "sheet-garbage":
        world.youtube.serve_sheet("sheet1.png", b"not an image at all")
    elif how == "embedding-fails":
        world.embedder.fail_tiles = True
    else:
        raise AssertionError(how)


def _step_ran(world, how: str) -> bool:
    if how == "embedding-fails":
        return len(world.sheets()) == 2
    return any(r["path"].endswith("sheet1.png") for r in world.sheets())


STEP_FAILURES = ["sheet-500", "sheet-garbage", "embedding-fails"]


@pytest.mark.parametrize("how", STEP_FAILURES)
def test_spec_addon_121_i3_a_failed_step_writes_nothing_and_the_fetch_still_downloads_and_scans(
    world, how,
):
    world.answer()
    file_id = world.add()
    _break_the_step(world, how)
    vectors = world.vectors()

    fetch(file_id)

    assert _step_ran(world, how), "the fetch did not try the storyboard"
    assert world.clip_seconds(file_id) == []
    assert world.vectors() - vectors <= world.other_rows(file_id)
    assert tile_names(file_id) == []
    assert leftover_dirs(file_id) == []
    assert world.clip_indexed(file_id) is False
    assert world.loft.downloads() != []
    assert world.loft.flags(file_id)[0] is True
    assert world.loft.failures(file_id) == 0
    assert world.loft.lookups() == 1


BACKING_OFF = "loft_media backing off"


@pytest.mark.parametrize("how", STEP_FAILURES)
def test_spec_addon_121_i3_a_failed_step_changes_no_flag_count_or_youtube_result(
    world, how, caplog,
):
    """The middle fetch ends at the disk guard, which leaves no YouTube result of
    its own (SPEC-ADDON-119), so the run of lookup failures around it is unbroken."""
    def failed_lookup() -> None:
        world.youtube.set_ytdlp(fail=LOOKUP_FAILURE)
        with caplog.at_level("INFO"):
            fetch(world.add())

    failed_lookup()
    failed_lookup()

    world.answer()
    file_id = world.add()
    world.loft.set(file_id, loft_fetch_failures=1)
    _break_the_step(world, how)
    world.free.free = 1 * GIB
    vectors = world.vectors()
    with caplog.at_level("INFO"):
        fetch(file_id)

    assert _step_ran(world, how), "the fetch did not try the storyboard"
    assert world.clip_seconds(file_id) == []
    assert world.vectors() == vectors
    assert world.clip_indexed(file_id) is False
    assert world.loft.flags(file_id) == (False, True, True)
    assert world.loft.failures(file_id) == 1

    world.free.free = 100 * GIB
    failed_lookup()

    assert [
        r.getMessage() for r in caplog.records if BACKING_OFF in r.getMessage()
    ] == ["loft_media backing off 30 min after 3 failures in a row"]


# ---------------------------------------------------------------------------
# §2: the CLIP image model when only OCR is wanted
# ---------------------------------------------------------------------------


def _ocr_only(world) -> str:
    world.configure(frame_ocr="on_index")
    world.answer()
    return world.add(frame_scan_indexed=True, frame_ocr_read=False, frame_ocr_indexed=False)


def test_spec_addon_121_section_2_with_only_ocr_wanted_the_step_loads_the_model_and_runs(world):
    file_id = _ocr_only(world)

    fetch(file_id)

    assert world.clip_seconds(file_id) == pytest.approx(KEPT_SECONDS)
    assert world.clip_indexed(file_id) is True
    assert world.loft.flags(file_id) == (True, True, True)


def test_spec_addon_121_section_2_with_only_ocr_wanted_a_model_that_cannot_load_skips_the_step(
    world,
):
    file_id = _ocr_only(world)
    attempts = []

    def cannot_load(*_args, **_kwargs):
        attempts.append(1)
        raise RuntimeError("the CLIP weights are not on disk")

    patch_everywhere(world.monkeypatch, "load_image_model", cannot_load)

    fetch(file_id)

    assert attempts, "the fetch did not try to load the model for the step"
    assert world.sheets() == []
    assert world.clip_seconds(file_id) == []
    assert world.clip_indexed(file_id) is False
    assert world.loft.flags(file_id) == (True, True, True)
    assert world.loft.failures(file_id) == 0
