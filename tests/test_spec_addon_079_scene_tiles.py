"""SPEC-ADDON-079: the scene-tile store, its route, and the storyboard swap.

The storyboard leg is driven through the real ``yt-dlp``/sheet doubles of
``tests.youtube_frame_doubles`` and the real Pillow of
``tests.test_spec_addon_078_youtube_storyboard``; a tile is read back from
the WebP the leg wrote.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.frame_cache import frame_cache_dir
from app.models import IndexedFile
from app.workers import clip as clip_worker
from tests.search_endpoint_db import patch_everywhere
from tests.test_spec_addon_078_youtube_storyboard import (  # noqa: F401 - fixtures and helpers
    add_loft,
    new_id,
    pil,
    rows_of,
    run,
    serve_storyboard,
    storyboard_fps,
    world,
)
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    DRIVE,
    YOUTUBE_URL,
    describe,
    everything_logged,
    frame_api,
    youtube,
)

TILE_SUBDIR = "tiles"
CACHE_CONTROL = "public, max-age=86400, immutable"


def tiles_dir(file_id: str) -> Path:
    return frame_cache_dir(file_id) / TILE_SUBDIR


def tile_names(file_id: str) -> list[str]:
    directory = tiles_dir(file_id)
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.iterdir())


def tile_bytes(file_id: str) -> dict[str, bytes]:
    directory = tiles_dir(file_id)
    if not directory.exists():
        return {}
    return {
        p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()
    }


def leftover_dirs(file_id: str) -> list[str]:
    directory = frame_cache_dir(file_id)
    if not directory.exists():
        return []
    return sorted(
        p.name for p in directory.iterdir() if p.is_dir() and p.name.startswith("tiles.")
    )


def frame_dir_files(file_id: str) -> list[str]:
    directory = frame_cache_dir(file_id)
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.iterdir() if p.is_file())


def tree(root: Path) -> dict[str, bytes | None]:
    if not root.exists():
        return {}
    out: dict[str, bytes | None] = {}
    for path in sorted(root.rglob("*")):
        out[str(path.relative_to(root))] = path.read_bytes() if path.is_file() else None
    return out


def reopen_scene_leg(world, file_id: str) -> None:
    with world.Session() as session:
        session.query(IndexedFile).filter_by(file_id=file_id).update(
            {"clip_indexed": False}
        )
        session.commit()


def successful_leg(world, youtube, *, tiles: int = 18, tile_size=(320, 180)):
    world.configure(min_interval=10, max_frames=500)
    seconds = tiles * 10.0
    board, per_tile = serve_storyboard(
        youtube, world.pil, tiles=tiles, tile=tile_size,
        fps=storyboard_fps(seconds),
    )
    youtube.set_ytdlp(describe=describe([board], duration=seconds))
    file_id = new_id()
    add_loft(world, file_id)
    assert run(file_id) is True
    return file_id, per_tile


async def scene_tile(frame_api, file_id: str, t=None, *, drive=DRIVE, **params):
    query = dict(params)
    if t is not None:
        query["t"] = t
    headers = {"X-Lit-Drive": drive} if drive else {}
    return await frame_api.client.get(
        f"/files/{file_id}/scene-tile", params=query, headers=headers,
    )


class TestTheSceneTileRoute:
    async def test_a_stored_tile_is_returned_as_webp_with_the_frame_routes_immutable_cache_header(
        self, frame_api,
    ):
        file_id = new_id()
        frame_api.add_loft(file_id)
        (frame_cache_dir(file_id) / TILE_SUBDIR).mkdir(parents=True)
        (frame_cache_dir(file_id) / TILE_SUBDIR / "2000.webp").write_bytes(
            b"TILE-AT-2000"
        )
        (frame_cache_dir(file_id) / "2000.webp").write_bytes(b"FRAME-AT-2000")
        frame_response = await frame_api.frame(file_id, 2.0)

        response = await scene_tile(frame_api, file_id, 2.0)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/webp")
        assert response.content == b"TILE-AT-2000"
        assert response.headers["cache-control"] == CACHE_CONTROL
        assert response.headers["cache-control"] == frame_response.headers["cache-control"]

    @pytest.mark.parametrize(
        "how",
        [
            "not-a-row-time",
            "no-tiles",
            "native-video",
            "inactive",
            "unindexed",
            "another-drive",
        ],
    )
    async def test_a_request_with_no_stored_tile_at_that_time_is_404(self, how, frame_api):
        file_id = new_id()
        drive = DRIVE
        if how != "unindexed":
            if how == "native-video":
                path = frame_api.drive_dir / f"{file_id}.mp4"
                path.write_bytes(b"")
                frame_api.add_row(file_id, path, mime="video/mp4")
            else:
                frame_api.add_loft(file_id)
            if how != "no-tiles":
                (frame_cache_dir(file_id) / TILE_SUBDIR).mkdir(parents=True)
                (frame_cache_dir(file_id) / TILE_SUBDIR / "2000.webp").write_bytes(
                    b"T"
                )
        if how == "inactive":
            with frame_api.Session() as session:
                session.query(IndexedFile).filter_by(file_id=file_id).update(
                    {"active": False}
                )
                session.commit()
        if how == "another-drive":
            drive = "elsewhere"
        t = 3.0 if how == "not-a-row-time" else 2.0

        response = await scene_tile(frame_api, file_id, t, drive=drive)

        assert response.status_code == 404
        detail = response.json().get("detail")
        if how in ("inactive", "unindexed"):
            assert detail == "File not indexed"
        elif how == "another-drive":
            assert detail == "File not found"
        else:
            # The proxy answers an unknown path with the framework default;
            # these must come from the route's own tile check.
            assert detail != "Not Found"

    async def test_a_non_video_file_answers_400_not_a_video_file(self, frame_api):
        file_id = new_id()
        path = frame_api.drive_dir / f"{file_id}.pdf"
        path.write_bytes(b"%PDF")
        frame_api.add_row(file_id, path, mime="application/pdf", file_type="document")
        (frame_cache_dir(file_id) / TILE_SUBDIR).mkdir(parents=True)
        (frame_cache_dir(file_id) / TILE_SUBDIR / "2000.webp").write_bytes(b"T")

        response = await scene_tile(frame_api, file_id, 2.0)

        assert response.status_code == 400
        assert response.json()["detail"] == "Not a video file"

    @pytest.mark.parametrize(
        "query", [{}, {"t": -1}, {"t": "not-a-number"}], ids=["missing", "negative", "text"],
    )
    async def test_t_is_validated_as_the_frame_route_validates_it(self, query, frame_api):
        file_id = new_id()
        frame_api.add_loft(file_id)
        (frame_cache_dir(file_id) / TILE_SUBDIR).mkdir(parents=True)
        (frame_cache_dir(file_id) / TILE_SUBDIR / "2000.webp").write_bytes(b"T")

        response = await frame_api.client.get(
            f"/files/{file_id}/scene-tile", params=query,
            headers={"X-Lit-Drive": DRIVE},
        )

        assert response.status_code == 422

    async def test_the_route_never_calls_out_runs_a_process_or_writes_a_file(
        self, frame_api, monkeypatch,
    ):
        file_id = new_id()
        frame_api.add_loft(file_id)
        (frame_cache_dir(file_id) / TILE_SUBDIR).mkdir(parents=True)
        (frame_cache_dir(file_id) / TILE_SUBDIR / "2000.webp").write_bytes(b"TILE")

        def refuse(*args, **kwargs):
            raise AssertionError("the scene-tile route reached outside its files")

        for module, name in (
            (httpx, "get"), (httpx, "request"), (httpx, "post"),
            (subprocess, "run"), (subprocess, "Popen"), (subprocess, "check_output"),
        ):
            monkeypatch.setattr(module, name, refuse)
        before = tree(frame_api.data_dir)

        response = await scene_tile(frame_api, file_id, 2.0)

        assert response.status_code == 200
        assert tree(frame_api.data_dir) == before


def _manifest_routes() -> dict[str, dict]:
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "manifest.json").read_text(encoding="utf-8")
    )
    return {route["path"]: route for route in manifest["proxy"]["routes"]}


class TestTheManifestEntry:
    def test_the_scene_tile_route_is_proxied_as_the_frame_route_is(self):
        routes = _manifest_routes()
        entry = routes.get("/files/{file_id}/scene-tile")

        assert entry is not None, "the host proxy cannot reach a route it does not know"
        assert entry["methods"] == ["GET"]
        assert entry.get("drive_optional") is True
        assert entry.get("pre_check") == {"type": "file_access", "param": "file_id"}

        frame = routes["/files/{file_id}/frame"]
        assert {k: v for k, v in entry.items() if k != "path"} == {
            k: v for k, v in frame.items() if k != "path"
        }


class TestTheTilesAfterASuccessfulSwap:
    @pytest.mark.parametrize(
        ("tile", "tiles"),
        [((320, 180), 18), ((200, 100), 12)],
        ids=["320x180", "200x100"],
    )
    def test_each_tile_is_named_by_its_rows_millisecond_and_kept_at_the_tiles_own_size(
        self, tile, tiles, world, youtube,
    ):
        file_id, per_tile = successful_leg(world, youtube, tiles=tiles, tile_size=tile)

        rows = rows_of(world, file_id)
        expected = {f"{int(round(ts * 1000))}.webp" for _, ts in rows}
        assert len(rows) == tiles
        assert tile_names(file_id) == sorted(expected)

        step_ms = int(round(per_tile * 1000))
        for name in tile_names(file_id):
            path = tiles_dir(file_id) / name
            with world.pil.open(path) as image:
                assert image.size == tile
                red, green, blue = image.convert("RGB").getpixel(
                    (image.size[0] // 2, image.size[1] // 2)
                )
            index = int(path.stem) // step_ms
            assert abs(red - ((index + 1) & 255)) <= 8, name
            assert abs(green - ((index + 1) >> 8)) <= 8, name
            assert abs(blue - 9) <= 8, name

        assert leftover_dirs(file_id) == []
        assert frame_dir_files(file_id) == []

    def test_a_later_leg_replaces_the_old_tile_set_with_its_own(self, world, youtube):
        file_id, _ = successful_leg(world, youtube, tiles=18)
        reopen_scene_leg(world, file_id)
        board, per_tile = serve_storyboard(
            youtube, world.pil, tiles=9, fps=storyboard_fps(90.0),
        )
        youtube.set_ytdlp(describe=describe([board], duration=9 * per_tile))

        assert run(file_id) is True

        rows = rows_of(world, file_id)
        assert len(rows) == 9
        assert tile_names(file_id) == sorted(
            f"{int(round(ts * 1000))}.webp" for _, ts in rows
        )
        assert leftover_dirs(file_id) == []


class TestAFailureBeforeTheCommit:
    @pytest.mark.parametrize(
        "how",
        ["sheet-500", "sheet-garbage", "embed-fails", "tile-write-fails"],
    )
    def test_the_old_rows_and_tiles_stay_whole_and_no_temporary_directory_remains(
        self, how, world, youtube, monkeypatch, caplog,
    ):
        file_id, per_tile = successful_leg(world, youtube)
        before_rows = rows_of(world, file_id)
        before_tiles = tile_bytes(file_id)
        reopen_scene_leg(world, file_id)
        board, _ = serve_storyboard(
            youtube, world.pil, tiles=18, fps=storyboard_fps(18 * per_tile),
        )
        youtube.set_ytdlp(describe=describe([board], duration=18 * per_tile))
        if how == "sheet-500":
            youtube.override("/sb/sheet1.png", status=500)
        elif how == "sheet-garbage":
            youtube.serve_sheet("sheet1.png", b"not an image at all")
        elif how == "embed-fails":
            world.embedder.fail_on_tile = 4
        else:
            real_save = world.pil.Image.save

            def refuse_webp(image, fp, *args, **kwargs):
                fmt = kwargs.get("format") or (args[0] if args else None)
                if (isinstance(fmt, str) and fmt.upper() == "WEBP") or str(
                    fp
                ).lower().endswith(".webp"):
                    raise OSError("no space left on device")
                return real_save(image, fp, *args, **kwargs)

            monkeypatch.setattr(world.pil.Image, "save", refuse_webp)

        with caplog.at_level(logging.DEBUG):
            assert run(file_id) is False

        assert rows_of(world, file_id) == before_rows
        assert tile_bytes(file_id) == before_tiles
        assert leftover_dirs(file_id) == []
        assert YOUTUBE_URL not in everything_logged(caplog)
        if how == "tile-write-fails":
            warnings = [r for r in caplog.records if r.levelname == "WARNING"]
            assert any(
                "tile_write_failed" in r.getMessage()
                and file_id in r.getMessage()
                for r in warnings
            )


class TestAFailureWhileWritingRows:
    def test_it_leaves_the_old_rows_and_tiles_and_no_temporary_directory(
        self, world, youtube, monkeypatch,
    ):
        file_id, per_tile = successful_leg(world, youtube)
        before_rows = rows_of(world, file_id)
        before_tiles = tile_bytes(file_id)
        reopen_scene_leg(world, file_id)
        board, per_tile = serve_storyboard(
            youtube, world.pil, tiles=9, fps=storyboard_fps(90.0),
        )
        youtube.set_ytdlp(describe=describe([board], duration=9 * per_tile))

        def refuse(*args, **kwargs):
            raise RuntimeError("the row write failed")

        monkeypatch.setattr(clip_worker, "_store_clip_embedding", refuse)

        with pytest.raises(RuntimeError):
            run(file_id)

        assert rows_of(world, file_id) == before_rows
        assert tile_bytes(file_id) == before_tiles
        assert leftover_dirs(file_id) == []


class TestALegWithNoStoryboard:
    @pytest.mark.parametrize("how", ["no-sb0", "no-duration", "not-youtube"])
    def test_it_leaves_the_rows_the_tiles_and_a_leftover_temporary_directory_alone(
        self, how, world, youtube, monkeypatch,
    ):
        file_id, per_tile = successful_leg(world, youtube)
        leftover = frame_cache_dir(file_id) / "tiles.left-by-an-earlier-run"
        leftover.mkdir(parents=True, exist_ok=True)
        (leftover / "999.webp").write_bytes(b"OLD LEFTOVER")
        before_rows = rows_of(world, file_id)
        before_tiles = tile_bytes(file_id)
        before_tree = tree(frame_cache_dir(file_id))
        reopen_scene_leg(world, file_id)
        if how == "no-sb0":
            youtube.set_ytdlp(
                describe=describe([youtube.format("f720", 720)], duration=18 * per_tile)
            )
        elif how == "no-duration":
            board, _ = serve_storyboard(
                youtube, world.pil, tiles=18, fps=storyboard_fps(18 * per_tile),
            )
            youtube.set_ytdlp(describe=describe([board]))
        else:
            loft = world.tmp / f"{file_id}.loft"
            loft.write_text(json.dumps(
                {"provider": "vimeo", "url": "https://vimeo.com/1"}
            ))

        assert run(file_id) is True

        assert rows_of(world, file_id) == before_rows
        assert tile_bytes(file_id) == before_tiles
        assert leftover_dirs(file_id) == ["tiles.left-by-an-earlier-run"]
        assert tree(frame_cache_dir(file_id)) == before_tree


class TestAFailedSwap:
    def _refuse_the_directory_swap(self, monkeypatch, file_id: str, state: dict):
        cache = frame_cache_dir(file_id)
        real_rename, real_replace = os.rename, os.replace

        def is_swap(src, dst) -> bool:
            source, target = Path(src), Path(dst)
            return (
                source.parent == cache
                and target.parent == cache
                and ("tiles" in {source.name, target.name})
            )

        def guarded(real):
            def call(src, dst, *args, **kwargs):
                if state["fail"] and is_swap(src, dst):
                    raise OSError("a simulated swap failure")
                return real(src, dst, *args, **kwargs)

            return call

        monkeypatch.setattr(os, "rename", guarded(real_rename))
        monkeypatch.setattr(os, "replace", guarded(real_replace))

    def test_the_committed_rows_and_flag_stay_and_a_warning_names_the_file(
        self, world, youtube, monkeypatch, caplog,
    ):
        file_id, _ = successful_leg(world, youtube)
        old_tiles = tile_bytes(file_id)
        reopen_scene_leg(world, file_id)
        board, per_tile = serve_storyboard(
            youtube, world.pil, tiles=9, fps=storyboard_fps(90.0),
        )
        youtube.set_ytdlp(describe=describe([board], duration=9 * per_tile))
        state = {"fail": True}
        self._refuse_the_directory_swap(monkeypatch, file_id, state)

        with caplog.at_level(logging.DEBUG):
            assert run(file_id) is True

        rows = rows_of(world, file_id)
        assert [ts for _, ts in rows] == pytest.approx([n * per_tile for n in range(9)])
        with world.Session() as session:
            assert session.query(IndexedFile).filter_by(file_id=file_id).one().clip_indexed is True
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any(file_id in r.getMessage() for r in warnings)
        logged = everything_logged(caplog)
        assert YOUTUBE_URL not in logged
        assert "dQw4w9WgXcQ" not in logged.replace(file_id, "")
        assert tile_bytes(file_id) == old_tiles

    def test_the_next_successful_leg_removes_every_leftover_and_writes_its_own_set(
        self, world, youtube, monkeypatch,
    ):
        file_id, _ = successful_leg(world, youtube)
        reopen_scene_leg(world, file_id)
        board, per_tile = serve_storyboard(
            youtube, world.pil, tiles=9, fps=storyboard_fps(90.0),
        )
        youtube.set_ytdlp(describe=describe([board], duration=9 * per_tile))
        state = {"fail": True}
        self._refuse_the_directory_swap(monkeypatch, file_id, state)
        assert run(file_id) is True
        state["fail"] = False
        for name in ("tiles.stale-one", "tiles.stale-two"):
            (frame_cache_dir(file_id) / name).mkdir(parents=True)
            (frame_cache_dir(file_id) / name / "1.webp").write_bytes(b"STALE")
        reopen_scene_leg(world, file_id)
        board, per_tile = serve_storyboard(
            youtube, world.pil, tiles=12, fps=storyboard_fps(120.0),
        )
        youtube.set_ytdlp(describe=describe([board], duration=12 * per_tile))

        assert run(file_id) is True

        rows = rows_of(world, file_id)
        assert len(rows) == 12
        assert leftover_dirs(file_id) == []
        assert tile_names(file_id) == sorted(
            f"{int(round(ts * 1000))}.webp" for _, ts in rows
        )
        assert frame_dir_files(file_id) == []


class _EmptyQuery:
    def filter(self, *args, **kwargs):
        return self

    def filter_by(self, *args, **kwargs):
        return self

    def distinct(self):
        return self

    def all(self):
        return []

    def first(self):
        return None

    def delete(self, *args, **kwargs):
        return 0


class _EmptyResult:
    def fetchone(self):
        return None

    def fetchall(self):
        return []


class _EmptySession:
    def query(self, *args, **kwargs):
        return _EmptyQuery()

    def execute(self, *args, **kwargs):
        return _EmptyResult()

    def delete(self, obj):
        pass


@contextmanager
def _empty_db():
    yield _EmptySession()


@contextmanager
def _drive_db(file_id: str):
    class _Query:
        def filter(self, *args, **kwargs):
            return self

        def all(self):
            return [SimpleNamespace(file_id=file_id)]

    class _Session:
        def query(self, *args, **kwargs):
            return _Query()

    yield _Session()


def _seed_cache(file_id: str) -> None:
    (frame_cache_dir(file_id) / TILE_SUBDIR).mkdir(parents=True)
    (frame_cache_dir(file_id) / TILE_SUBDIR / "0.webp").write_bytes(b"TILE")
    (frame_cache_dir(file_id) / "0.webp").write_bytes(b"FRAME")


class TestPurge:
    def test_removing_a_file_from_the_index_removes_its_tiles_with_its_frame_cache(
        self, monkeypatch, make_settings,
    ):
        patch_everywhere(monkeypatch, "settings", make_settings())
        purged, other = "pu" + uuid.uuid4().hex[:10], "pu" + uuid.uuid4().hex[:10]
        _seed_cache(purged)
        _seed_cache(other)
        monkeypatch.setattr("app.indexer.get_search_db", _empty_db)

        from app.indexer import _purge_file

        _purge_file(purged)

        assert not frame_cache_dir(purged).exists()
        assert tile_bytes(other) == {"0.webp": b"TILE"}

    def test_the_policy_off_drive_purge_removes_tiles_too(self, monkeypatch, make_settings):
        patch_everywhere(monkeypatch, "settings", make_settings())
        file_id = "pu" + uuid.uuid4().hex[:10]
        _seed_cache(file_id)
        monkeypatch.setattr("app.database.get_search_db", lambda: _drive_db(file_id))
        monkeypatch.setattr("app.indexer.get_search_db", _empty_db)
        monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)

        from app.purge import purge_drive

        assert purge_drive("family") == 1

        assert not frame_cache_dir(file_id).exists()


class TestTheFrameRoute:
    async def test_a_tile_in_the_tiles_directory_is_never_served_as_a_frame(
        self, frame_api, monkeypatch,
    ):
        def no_frame_source(*args, **kwargs):
            raise RuntimeError("this test gives the frame route no source")

        monkeypatch.setattr("app.youtube_source.ensure_youtube_frame", no_frame_source)
        file_id = new_id()
        frame_api.add_loft(file_id)
        tiles_dir(file_id).mkdir(parents=True)
        (tiles_dir(file_id) / "2000.webp").write_bytes(b"A STORYBOARD TILE")

        response = await frame_api.frame(file_id, 2.0)

        assert response.content != b"A STORYBOARD TILE"
        assert not (frame_cache_dir(file_id) / "2000.webp").exists()
