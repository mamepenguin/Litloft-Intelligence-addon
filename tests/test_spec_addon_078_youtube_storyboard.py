"""SPEC-ADDON-078: the ``clip`` task of a YouTube ``.loft`` indexes its storyboard.

The sheets are real PNGs whose tiles each have one flat colour naming the tile,
so what CLIP is given is read back from the image it receives. Pillow is the
suite's mock in every other test; here the real package stands in for it.
"""

from __future__ import annotations

import logging
import math
import sys
import uuid
from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import FrameExtractionConfig, IndexingConfig
from app.models import Base, Embedding, IndexedFile
from app.workers import clip as clip_worker
from tests.search_endpoint_db import patch_everywhere
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    LOFT_MIME,
    YOUTUBE_URL,
    describe,
    everything_logged,
    needs_ffmpeg,
    real_pil_modules,
    storyboard_format,
    youtube,
)

TILE_MARK = 9
THUMBNAIL_MARK = 200
SHEET_TILES = 9
SHEET_SECONDS = 90.0


def new_id() -> str:
    return "sb" + uuid.uuid4().hex[:10]


@pytest.fixture()
def pil(monkeypatch):
    real = real_pil_modules()
    for name, module in real.items():
        monkeypatch.setitem(sys.modules, name, module)
    patch_everywhere(monkeypatch, "Image", real["PIL.Image"])
    return real["PIL.Image"]


class Embedder:
    """Stands in for CLIP: records which flat-colour image it was given."""

    def __init__(self):
        self.tiles: list[int] = []
        self.thumbnails = 0
        self.fail_on_tile: int | None = None

    def __call__(self, image):
        red, green, blue = image.convert("RGB").getpixel(
            (image.size[0] // 2, image.size[1] // 2)
        )
        vector = np.zeros(512, dtype=np.float32)
        if blue == THUMBNAIL_MARK:
            self.thumbnails += 1
            vector[511] = 1.0
            return vector
        assert blue == TILE_MARK, f"not a tile or the thumbnail: {(red, green, blue)}"
        index = ((green << 8) | red) - 1
        if self.fail_on_tile is not None and len(self.tiles) + 1 >= self.fail_on_tile:
            raise RuntimeError("CLIP failed")
        self.tiles.append(index)
        vector[index % 500] = 1.0
        return vector


@pytest.fixture()
def world(monkeypatch, tmp_path, make_settings, pil):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE vec_clip (embedding_id TEXT PRIMARY KEY, embedding_type TEXT, vector BLOB)"
        ))
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _read():
        s = Session()
        try:
            yield s
        finally:
            s.rollback()
            s.close()

    @contextmanager
    def _write():
        s = Session()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    mount = tmp_path / "thumbs"
    (mount / "d").mkdir(parents=True)
    monkeypatch.setenv("HOMEVAULT_THUMBNAILS_DIR", str(mount))

    embedder = Embedder()
    patch_everywhere(monkeypatch, "get_search_db", _write)
    patch_everywhere(monkeypatch, "get_search_db_read", _read)
    patch_everywhere(monkeypatch, "embed_image", embedder)
    patch_everywhere(monkeypatch, "validate_file_path", lambda p: True)
    patch_everywhere(monkeypatch, "resolve_file_path", lambda _drive, path: path)
    monkeypatch.setattr(
        clip_worker, "_resolve_thumbnail_abspath", lambda p: str(mount / p),
    )
    monkeypatch.setattr(
        clip_worker, "_generate_blip_caption_if_needed", lambda *a, **k: None,
    )

    class World:
        pass

    w = World()
    w.Session, w.embedder, w.mount, w.tmp = Session, embedder, mount, tmp_path
    w.pil = pil
    w.make_settings = make_settings
    w.monkeypatch = monkeypatch

    def configure(min_interval=30, max_frames=500):
        settings = make_settings(
            allowed_base_dirs=(str(tmp_path) + "/",),
            indexing=replace(
                IndexingConfig(),
                frame_extraction=FrameExtractionConfig(
                    min_interval=min_interval, max_frames=max_frames,
                ),
            ),
        )
        patch_everywhere(monkeypatch, "settings", settings)

    configure()
    w.configure = configure
    return w


def _png(pil, size, paint) -> bytes:
    image = pil.new("RGB", size, (0, 0, 0))
    paint(image)
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def serve_storyboard(
    youtube, pil, *, tiles: int, rows: int = 3, columns: int = 3,
    tile: tuple[int, int] = (320, 180), seconds_per_tile: float = 10.0,
    fps: float | None = None, last_sheet_seconds: float | None = None,
    last_sheet_rows: int | None = None, last_sheet_cells: int | None = None,
) -> tuple[dict, float]:
    """Sheets that hold ``tiles`` tiles in all, and the ``sb0`` format naming them.

    The last sheet can be shorter than the declared grid (a smaller image and
    fewer painted cells), as YouTube's last sheet usually is.
    """
    per_sheet = rows * columns
    count = math.ceil(tiles / per_sheet)
    sheets = []
    for sheet in range(count):
        last = sheet == count - 1
        image_rows = last_sheet_rows if last and last_sheet_rows else rows
        painted = last_sheet_cells if last and last_sheet_cells else per_sheet

        def paint(image, sheet=sheet, image_rows=image_rows, painted=painted):
            for cell in range(painted):
                index = sheet * per_sheet + cell
                row, col = divmod(cell, columns)
                if row >= image_rows:
                    continue
                colour = ((index + 1) & 255, (index + 1) >> 8, TILE_MARK)
                image.paste(colour, (
                    col * tile[0], row * tile[1],
                    (col + 1) * tile[0], (row + 1) * tile[1],
                ))

        name = f"sheet{sheet}.png"
        youtube.serve_sheet(
            name, _png(pil, (columns * tile[0], image_rows * tile[1]), paint),
        )
        seconds = seconds_per_tile * per_sheet
        if last and last_sheet_seconds is not None:
            seconds = last_sheet_seconds
        sheets.append((name, seconds))
    return (
        storyboard_format(
            youtube.base, sheets, rows=rows, columns=columns, tile=tile, fps=fps,
        ),
        seconds_per_tile,
    )


def add_loft(w, file_id, *, duration=None, body=None, thumbnail=False, **flags):
    path = w.tmp / f"{file_id}.loft"
    if body is not False:
        import json

        path.write_text(json.dumps(
            body if body is not None else {"provider": "youtube", "url": YOUTUBE_URL}
        ))
    thumbnail_path = None
    if thumbnail:
        thumbnail_path = f"d/{file_id}.jpg"
        w.pil.new("RGB", (64, 36), (1, 2, THUMBNAIL_MARK)).save(
            w.mount / thumbnail_path, format="PNG",
        )
    with w.Session() as s:
        s.add(IndexedFile(
            file_id=file_id, drive="family", filename=path.name, file_path=str(path),
            file_type="video", mime_type=LOFT_MIME, file_size=1, active=True,
            duration=duration, thumbnail_path=thumbnail_path,
            clip_indexed=flags.get("clip_indexed", False),
            clip_thumbnail_indexed=flags.get("clip_thumbnail_indexed", False),
        ))
        s.commit()
    return path


def add_old_rows(w, file_id, kinds=("clip", "clip_thumbnail"), count=2):
    with w.Session() as s:
        for kind in kinds:
            for n in range(count if kind == "clip" else 1):
                row_id = f"old_{kind}_{file_id}_{n}"
                s.add(Embedding(
                    id=row_id, file_id=file_id, embedding_type=kind,
                    vector_table="vec_clip", content_preview="old",
                    timestamp_start=float(n) if kind == "clip" else None,
                ))
                s.execute(
                    text("INSERT INTO vec_clip(embedding_id, embedding_type, vector) VALUES (:i, :t, :v)"),
                    {"i": row_id, "t": kind, "v": np.ones(512, dtype=np.float32).tobytes()},
                )
        s.commit()


def rows_of(w, file_id, kind="clip"):
    with w.Session() as s:
        return [
            (e.id, e.timestamp_start)
            for e in s.query(Embedding)
            .filter_by(file_id=file_id, embedding_type=kind)
            .order_by(Embedding.timestamp_start, Embedding.id)
            .all()
        ]


def vec_ids(w, file_id):
    with w.Session() as s:
        return sorted(
            r[0] for r in s.execute(
                text("SELECT embedding_id FROM vec_clip WHERE embedding_id IN "
                     "(SELECT id FROM embeddings WHERE file_id = :f)"),
                {"f": file_id},
            )
        )


def flags_of(w, file_id):
    with w.Session() as s:
        row = s.query(IndexedFile).filter_by(file_id=file_id).one()
        return row.clip_indexed, row.clip_thumbnail_indexed


def expected_ticks(timestamps, duration, max_frames):
    spacing = duration / max(1, max_frames)
    kept: list[float] = []
    for ts in timestamps:
        if ts >= duration:
            continue
        if kept and ts - kept[-1] < spacing - 1e-9:
            continue
        kept.append(ts)
    return kept


def storyboard_fps(duration: float, interval: float = 10.0) -> float:
    """The ``fps`` yt-dlp reports for a storyboard of whole-second ``interval``."""
    return (math.ceil(duration / interval) + 1) / duration


def recovered_interval(duration: float, count: int) -> float:
    if count <= 1:
        return 1.0
    for seconds in range(1, math.ceil(duration) + 1):
        if math.ceil(duration / seconds) + 1 == count:
            return float(seconds)
    return duration / (count - 1)


def timed_tiles(duration: float, count: int, cells: int) -> list[float]:
    """The timestamps of the tiles below ``duration``, at ``k`` times the interval."""
    if count <= 1:
        return [0.0]
    interval = recovered_interval(duration, count)
    return [
        k * interval for k in range(min(count, cells)) if k * interval < duration
    ]


def run(file_id):
    return clip_worker._index_clip_sync(file_id)


pytestmark = needs_ffmpeg


class TestWhichTilesAreKept:
    @pytest.mark.parametrize(
        ("duration", "min_interval", "max_frames", "tiles"),
        [
            (300.0, 30, 500, 36),
            (3600.0, 30, 20, 360),
            (3600.0, 30, 500, 360),
            (3600.0, 30, 7, 360),
            (600.0, 60, 500, 63),
            (95.0, 10, 500, 18),
            (95.0, 30, 500, 18),
            (3600.0, 600, 20, 360),
            (90.0, 10, 500, 18),
            (89.5, 10, 500, 18),
        ],
        ids=[
            "short-video-keeps-every-tile", "long-video-spread-by-cap",
            "long-video-keeps-every-tile", "long-video-spread-with-a-tight-cap",
            "wide-min-interval-does-not-thin", "last-sheet-empty-cells-dropped",
            "min-interval-above-tile-spacing-does-not-thin",
            "min-interval-does-not-truncate-a-long-video",
            "tile-at-duration-dropped", "duration-inside-a-tile-interval",
        ],
    )
    def test_kept_tiles_are_scene_rows_in_time_order(
        self, duration, min_interval, max_frames, tiles, world, youtube,
    ):
        """The first tile is kept, then only one at least ``duration / max_frames`` later."""
        world.configure(min_interval, max_frames)
        sb0, per_tile = serve_storyboard(
            youtube, world.pil, tiles=tiles, fps=storyboard_fps(duration),
        )
        youtube.set_ytdlp(describe=describe([sb0], duration=duration))
        fid = new_id()
        add_loft(world, fid, duration=None)

        assert run(fid) is True

        timestamps = [n * per_tile for n in range(tiles)]
        expected = expected_ticks(timestamps, duration, max_frames)
        rows = rows_of(world, fid)
        assert [ts for _, ts in rows] == pytest.approx(expected)
        assert rows[0][1] == pytest.approx(0.0)
        assert len(rows) <= max_frames
        assert world.embedder.tiles == [round(ts / per_tile) for ts in expected]
        assert all(ts < duration for _, ts in rows)
        assert sorted(vec_ids(world, fid)) == sorted(i for i, _ in rows)
        assert flags_of(world, fid)[0] is True

    @pytest.mark.parametrize(
        ("rows", "columns", "tile", "tiles", "per_tile"),
        [
            (3, 3, (320, 180), 27, 10.0),
            (5, 5, (160, 90), 50, 10.0),
            (2, 4, (200, 100), 24, 5.0),
        ],
        ids=["3x3-320", "5x5-160", "2x4-200-5s"],
    )
    def test_the_declared_grid_and_tile_size_cut_the_sheets(
        self, rows, columns, tile, tiles, per_tile, world, youtube,
    ):
        """Tiles are cut at the declared grid and sized by the declared tile size."""
        world.configure(min_interval=30, max_frames=500)
        sb0, _ = serve_storyboard(
            youtube, world.pil, tiles=tiles, rows=rows, columns=columns,
            tile=tile, seconds_per_tile=per_tile,
            fps=storyboard_fps(tiles * per_tile, per_tile),
        )
        duration = tiles * per_tile
        youtube.set_ytdlp(describe=describe([sb0], duration=duration))
        fid = new_id()
        add_loft(world, fid)

        assert run(fid) is True

        expected = expected_ticks(
            [n * per_tile for n in range(tiles)], duration, 500,
        )
        assert [ts for _, ts in rows_of(world, fid)] == pytest.approx(expected)
        assert world.embedder.tiles == [round(ts / per_tile) for ts in expected]


class TestTheTileInterval:
    @pytest.mark.parametrize(
        ("duration", "count", "sheets", "interval"),
        [
            (1120.0, 113, 13, 10.0),
            (486.0, 99, 11, 5.0),
            (56.0, 57, 7, 1.0),
            (6980.0, 699, 78, 10.0),
            (10.0, 7, 1, 10.0 / 6),
            (10.0, 1, 1, None),
        ],
        ids=[
            "1120s-113-tiles-10s",
            "486s-99-tiles-5s",
            "56s-57-tiles-1s",
            "6980s-699-tiles-10s",
            "no-whole-second-falls-back",
            "a-single-tile",
        ],
    )
    def test_each_tile_is_numbered_across_sheets_and_timed_by_the_recovered_interval(
        self, duration, count, sheets, interval, world, youtube,
    ):
        """SPEC-ADDON-078 I7: tile ``k`` is at ``k`` times the recovered interval.

        ``count`` is the frame count YouTube declares, so ``1 / fps`` is only an
        average of it and the padding cell at the end carries no scene.
        """
        world.configure(min_interval=30, max_frames=1000)
        cells = sheets * SHEET_TILES
        sb0, _ = serve_storyboard(
            youtube, world.pil, tiles=count, tile=(16, 9),
            seconds_per_tile=interval or 10.0, fps=count / duration,
        )
        youtube.set_ytdlp(describe=describe([sb0], duration=duration))
        fid = new_id()
        add_loft(world, fid)

        assert run(fid) is True

        expected = timed_tiles(duration, count, cells)
        rows = rows_of(world, fid)
        assert [ts for _, ts in rows] == pytest.approx(expected)
        assert world.embedder.tiles == list(range(len(expected)))
        if interval is not None:
            assert len(rows) > 1
            assert (rows[1][1] - rows[0][1]) == pytest.approx(interval)


class TestAShortLastSheet:
    @pytest.mark.parametrize(
        ("duration", "count", "interval", "kept", "last"),
        [
            (1120.0, 113, 10.0, 112, 1110.0),
            (486.0, 99, 5.0, 98, 485.0),
            (440.0, 45, 10.0, 44, 430.0),
        ],
        ids=[
            "last-sheet-half-as-long-10s",
            "last-sheet-half-as-long-5s",
            "cells-past-the-video-duration",
        ],
    )
    def test_the_short_last_sheet_keeps_the_tile_interval(
        self, duration, count, interval, kept, last, world, youtube,
    ):
        """A sheet's own length does not change the interval, and its cells past the end go."""
        world.configure(min_interval=30, max_frames=1000)
        sb0, _ = serve_storyboard(
            youtube, world.pil, tiles=count, tile=(16, 9),
            seconds_per_tile=interval, last_sheet_seconds=50.0,
            fps=count / duration,
        )
        youtube.set_ytdlp(describe=describe([sb0], duration=duration))
        fid = new_id()
        add_loft(world, fid)

        assert run(fid) is True

        rows = rows_of(world, fid)
        assert len(rows) == kept
        assert [ts for _, ts in rows] == pytest.approx(
            [k * interval for k in range(kept)]
        )
        assert rows[-1][1] == pytest.approx(last)
        assert world.embedder.tiles == list(range(kept))


class TestTheVideosDuration:
    def test_the_lookups_duration_wins_over_the_indexed_one(self, world, youtube):
        """SPEC-ADDON-078 step 3: the lookup's ``duration``, else the indexed one."""
        world.configure(min_interval=10)
        sb0, _ = serve_storyboard(
            youtube, world.pil, tiles=36, fps=storyboard_fps(90.0),
        )
        youtube.set_ytdlp(describe=describe([sb0], duration=90.0))
        fid = new_id()
        add_loft(world, fid, duration=600.0)

        run(fid)

        assert max(ts for _, ts in rows_of(world, fid)) < 90.0
        assert len(rows_of(world, fid)) == 9

    def test_the_indexed_duration_is_used_when_the_lookup_has_none(self, world, youtube):
        """SPEC-ADDON-078 step 3: media_import copies yt-dlp's duration into the row."""
        world.configure(min_interval=10)
        sb0, _ = serve_storyboard(
            youtube, world.pil, tiles=36, fps=storyboard_fps(60.0),
        )
        youtube.set_ytdlp(describe=describe([sb0]))
        fid = new_id()
        add_loft(world, fid, duration=60.0)

        assert run(fid) is True

        assert len(rows_of(world, fid)) == 6

    @pytest.mark.parametrize("indexed", [None, 0.0, -5.0])
    def test_without_a_known_duration_the_video_has_no_storyboard(
        self, indexed, world, youtube,
    ):
        """SPEC-ADDON-078 step 3/6: no rows, ``clip_indexed`` true, no sheet downloaded."""
        sb0, _ = serve_storyboard(
            youtube, world.pil, tiles=36, fps=storyboard_fps(90.0),
        )
        youtube.set_ytdlp(describe=describe([sb0]))
        fid = new_id()
        add_loft(world, fid, duration=indexed)

        assert run(fid) is True

        assert len(youtube.lookups()) == 1
        assert youtube.fetched("/sb/") == []
        assert rows_of(world, fid) == []
        assert flags_of(world, fid)[0] is True


class TestDefiniteResultsWithoutRows:
    def test_no_storyboard_format_is_a_finished_task(self, world, youtube):
        """SPEC-ADDON-078 step 6: no ``sb0`` closes the task with no rows."""
        youtube.set_ytdlp(describe=describe([youtube.format("f720", 720)], duration=300.0))
        fid = new_id()
        add_loft(world, fid)

        assert run(fid) is True

        assert len(youtube.lookups()) == 1
        assert rows_of(world, fid) == []
        assert flags_of(world, fid)[0] is True

    @pytest.mark.parametrize("fps", [None, 0.0])
    def test_a_storyboard_without_fps_is_a_finished_task(self, fps, world, youtube):
        """SPEC-ADDON-078 I7: no positive ``fps`` is no storyboard, with no sheet fetched."""
        sb0, _ = serve_storyboard(youtube, world.pil, tiles=18, fps=fps)
        youtube.set_ytdlp(describe=describe([sb0], duration=180.0))
        fid = new_id()
        add_loft(world, fid)

        assert run(fid) is True

        assert len(youtube.lookups()) == 1
        assert youtube.fetched("/sb/") == []
        assert rows_of(world, fid) == []
        assert flags_of(world, fid)[0] is True

    def test_only_a_youtube_loft_is_looked_up(self, world, youtube):
        """SPEC-ADDON-078 step 6: another provider finishes at once, with no network call."""
        sb0, _ = serve_storyboard(
            youtube, world.pil, tiles=18, fps=storyboard_fps(120.0),
        )
        youtube.set_ytdlp(describe=describe([sb0], duration=120.0))
        yt, vimeo = new_id(), new_id()
        add_loft(world, yt)
        add_loft(world, vimeo, body={"provider": "vimeo", "url": "https://vimeo.com/1"})

        assert run(vimeo) is True
        assert youtube.lookups() == []
        assert run(yt) is True

        assert len(youtube.lookups()) == 1
        assert rows_of(world, vimeo) == []
        assert flags_of(world, vimeo)[0] is True
        assert rows_of(world, yt)


class TestTheRowsAreReplacedWhole:
    def test_old_scene_rows_are_replaced_and_nothing_else_is_touched(self, world, youtube):
        """SPEC-ADDON-078 I7: own ``clip`` rows replaced, no ``clip_thumbnail`` or other file row touched."""
        world.configure(min_interval=30)
        sb0, _ = serve_storyboard(
            youtube, world.pil, tiles=18, fps=storyboard_fps(180.0),
        )
        youtube.set_ytdlp(describe=describe([sb0], duration=180.0))
        fid, other = new_id(), new_id()
        add_loft(world, fid, clip_thumbnail_indexed=True)
        add_loft(world, other, clip_thumbnail_indexed=True)
        add_old_rows(world, fid)
        add_old_rows(world, other)
        before_other = rows_of(world, other) + rows_of(world, other, "clip_thumbnail")
        before_thumb = rows_of(world, fid, "clip_thumbnail")

        assert run(fid) is True

        rows = rows_of(world, fid)
        assert [ts for _, ts in rows] == pytest.approx([n * 10.0 for n in range(18)])
        assert not any(i.startswith("old_") for i, _ in rows)
        assert rows_of(world, fid, "clip_thumbnail") == before_thumb
        assert rows_of(world, other) + rows_of(world, other, "clip_thumbnail") == before_other
        assert sorted(vec_ids(world, fid)) == sorted(
            [i for i, _ in rows] + [i for i, _ in before_thumb]
        )

    def test_a_repeat_run_leaves_one_set_of_rows_and_one_thumbnail_row(self, world, youtube):
        """SPEC-ADDON-078 I8: no duplicates after the task is reset and run again."""
        world.configure(min_interval=30)
        sb0, _ = serve_storyboard(
            youtube, world.pil, tiles=18, fps=storyboard_fps(180.0),
        )
        youtube.set_ytdlp(describe=describe([sb0], duration=180.0))
        fid = new_id()
        add_loft(world, fid, thumbnail=True)

        assert run(fid) is True
        first = rows_of(world, fid)
        with world.Session() as s:
            s.query(IndexedFile).filter_by(file_id=fid).update({"clip_indexed": False})
            s.commit()
        assert run(fid) is True

        assert len(first) == 18
        assert [ts for _, ts in rows_of(world, fid)] == [ts for _, ts in first]
        assert len(rows_of(world, fid, "clip_thumbnail")) == 1
        assert world.embedder.thumbnails == 1


class TestTheThumbnailLeg:
    def test_the_storyboard_leg_runs_after_the_thumbnail_leg(self, world, youtube):
        """SPEC-ADDON-078 I7: a ``.loft`` with a thumbnail gets both its row and its scene rows."""
        world.configure(min_interval=30)
        sb0, _ = serve_storyboard(
            youtube, world.pil, tiles=18, fps=storyboard_fps(180.0),
        )
        youtube.set_ytdlp(describe=describe([sb0], duration=180.0))
        fid = new_id()
        add_loft(world, fid, thumbnail=True)

        assert run(fid) is True

        assert len(rows_of(world, fid, "clip_thumbnail")) == 1
        assert len(rows_of(world, fid)) == 18
        assert flags_of(world, fid) == (True, True)

    def test_a_finished_thumbnail_leg_is_not_embedded_again(self, world, youtube):
        """SPEC-ADDON-078 item 1 step 1: skipped when ``clip_thumbnail_indexed`` is true."""
        world.configure(min_interval=30)
        sb0, _ = serve_storyboard(
            youtube, world.pil, tiles=18, fps=storyboard_fps(180.0),
        )
        youtube.set_ytdlp(describe=describe([sb0], duration=180.0))
        fid = new_id()
        add_loft(world, fid, thumbnail=True, clip_thumbnail_indexed=True)

        assert run(fid) is True

        assert world.embedder.thumbnails == 0
        assert rows_of(world, fid, "clip_thumbnail") == []
        assert len(rows_of(world, fid)) == 18


class TestTransientFailures:
    @pytest.mark.parametrize(
        "how", ["lookup", "sheet-500", "sheet-garbage", "clip-fails", "loft-missing"],
    )
    def test_nothing_is_written_and_the_task_stays_open_and_is_not_retried_in_process(
        self, how, world, youtube,
    ):
        """SPEC-ADDON-078 I8: False, rows as they were, ``clip_indexed`` false, no second attempt."""
        world.configure(min_interval=30)
        sb0, _ = serve_storyboard(
            youtube, world.pil, tiles=18, fps=storyboard_fps(180.0),
        )
        youtube.set_ytdlp(describe=describe([sb0], duration=180.0))
        fid = new_id()
        path = add_loft(world, fid, clip_thumbnail_indexed=True)
        add_old_rows(world, fid, kinds=("clip",))
        before = rows_of(world, fid)
        if how == "lookup":
            youtube.set_ytdlp(fail="ERROR: [youtube] x: Video unavailable")
        elif how == "sheet-500":
            youtube.override("/sb/sheet1.png", status=500)
        elif how == "sheet-garbage":
            youtube.serve_sheet("sheet1.png", b"not an image at all")
        elif how == "clip-fails":
            world.embedder.fail_on_tile = 4
        else:
            path.unlink()

        assert run(fid) is False

        assert rows_of(world, fid) == before
        assert vec_ids(world, fid) == sorted(i for i, _ in before)
        assert flags_of(world, fid)[0] is False
        if how != "loft-missing":
            assert len(youtube.lookups()) == 1

        world.embedder.fail_on_tile = None
        youtube.override("/sb/sheet1.png")
        youtube.set_ytdlp(describe=describe([sb0], duration=180.0))
        lookups, fetched = len(youtube.lookups()), len(youtube.fetched("/sb/"))

        assert run(fid) is False

        assert len(youtube.lookups()) == lookups
        assert len(youtube.fetched("/sb/")) == fetched
        assert rows_of(world, fid) == before
        assert flags_of(world, fid)[0] is False

    def test_a_failure_of_one_file_does_not_stop_another(self, world, youtube):
        """SPEC-ADDON-078 I8: only the file that failed is skipped."""
        world.configure(min_interval=30)
        sb0, _ = serve_storyboard(
            youtube, world.pil, tiles=18, fps=storyboard_fps(180.0),
        )
        youtube.set_ytdlp(fail="ERROR: [youtube] x: Video unavailable")
        failing, fine = new_id(), new_id()
        add_loft(world, failing)
        add_loft(world, fine)
        assert run(failing) is False

        youtube.set_ytdlp(describe=describe([sb0], duration=180.0))

        assert run(fine) is True
        assert len(rows_of(world, fine)) == 18


SHEET_QUERY = "?sqp=SQP-LEAK-3f9a&sigh=SIGH-LEAK-4b2e"


class TestTheSheetFetchIsNotLogged:
    def test_no_record_from_any_logger_carries_a_sheet_address(
        self, world, youtube, caplog,
    ):
        """SPEC-ADDON-078 I5: no logger prints a sheet URL or any part of its signed query."""
        world.configure(min_interval=30)
        sb0, _ = serve_storyboard(
            youtube, world.pil, tiles=18, fps=storyboard_fps(180.0),
        )
        for fragment in sb0["fragments"]:
            fragment["url"] += SHEET_QUERY
        youtube.set_ytdlp(describe=describe([sb0], duration=180.0))
        fid = new_id()
        add_loft(world, fid)
        caplog.set_level(logging.INFO)

        assert run(fid) is True

        assert len(youtube.fetched("/sb/")) == 2
        logged = everything_logged(caplog)
        for secret in ("sheet0.png", "sheet1.png", SHEET_QUERY, "sqp=", "sigh="):
            assert secret not in logged
