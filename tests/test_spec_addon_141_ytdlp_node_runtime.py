"""SPEC-ADDON-141: every yt-dlp process the addon starts may use Node and only Node."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.loft_media_world import GIB, FreeSpace, LoftWorld, fetch
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.test_spec_addon_078_youtube_storyboard import (  # noqa: F401 - fixtures
    add_loft as add_storyboard_loft,
    new_id as new_storyboard_id,
    pil,
    run as run_storyboard_index,
    serve_storyboard,
    storyboard_fps,
    world,
)
from tests.test_spec_addon_111_frame_scan_task import _encode
from tests.test_spec_addon_119_loft_fetch_counting import BACKING_OFF, Fetches
from tests.vec_partition_db import search_index  # noqa: F401 - fixture
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    YOUTUBE_URL,
    describe,
    frame_api,
    needs_ffmpeg,
    youtube,
)

pytestmark = needs_ffmpeg

VIDEO_SECONDS = 40.0
MISSING_RUNTIME_WARNING = (
    "WARNING: [youtube] dQw4w9WgXcQ: No supported JavaScript runtime could be found. "
    "Only deno is enabled by default; to use another runtime add --js-runtimes "
    "RUNTIME[:PATH] to your command/config. YouTube extraction without a JS runtime "
    "has been deprecated, and some formats may be missing."
)


def assert_node_only(argv: list[str], url: str) -> None:
    assert "--no-js-runtimes" in argv, argv
    assert "--js-runtimes=node" in argv, argv
    assert argv.index("--no-js-runtimes") < argv.index("--js-runtimes=node"), argv
    assert [a for a in argv if a.startswith("--js-runtimes")] == ["--js-runtimes=node"], argv
    assert not any(a.startswith("--remote-components") for a in argv), argv
    assert argv[-1] == url, argv


def assert_lookup_argv(argv: list[str], url: str) -> None:
    assert_node_only(argv, url)
    assert all(a.startswith("-") for a in argv[:-1]), argv


@pytest.fixture(scope="module")
def gradient(tmp_path_factory):
    return _encode(tmp_path_factory.mktemp("node-runtime") / "switches.mp4")


@pytest.fixture()
def loft(search_index, youtube, monkeypatch, tmp_path, make_settings):  # noqa: F811
    from app.workers import loft_media

    loft_media.reset_worker_state()
    yield LoftWorld(search_index, youtube, monkeypatch, tmp_path, make_settings)
    loft_media.reset_worker_state()


async def test_spec_addon_141_i2_the_frame_reads_lookup_allows_node_only(youtube, frame_api):  # noqa: F811
    fmt = youtube.format("f720", 720)
    youtube.set_ytdlp(describe=describe([fmt]))
    frame_api.add_loft("ytnode000001")

    response = await frame_api.frame("ytnode000001", 1.0)

    assert response.status_code == 200
    (lookup,) = youtube.lookups()
    assert_lookup_argv(lookup["argv"], YOUTUBE_URL)


def test_spec_addon_141_i2_the_storyboard_indexs_lookup_allows_node_only(world, youtube):  # noqa: F811
    world.configure(min_interval=10)
    sb0, _ = serve_storyboard(youtube, world.pil, tiles=18, fps=storyboard_fps(120.0))
    youtube.set_ytdlp(describe=describe([sb0], duration=120.0))
    fid = new_storyboard_id()
    add_storyboard_loft(world, fid)

    assert run_storyboard_index(fid) is True

    (lookup,) = youtube.lookups()
    assert_lookup_argv(lookup["argv"], YOUTUBE_URL)


def _fetch_one(loft, gradient) -> None:
    loft.monkeypatch.setattr(shutil, "disk_usage", FreeSpace(100 * GIB))
    loft.serve(gradient, duration=VIDEO_SECONDS)
    loft.add_loft("ytnode000002")
    fetch("ytnode000002")
    assert loft.flags("ytnode000002")[0] is True


@pytest.mark.usefixtures("pillow_real")
def test_spec_addon_141_i2_a_loft_fetchs_lookup_allows_node_only(loft, gradient):
    _fetch_one(loft, gradient)

    (lookup,) = loft.youtube.lookups()
    assert_lookup_argv(lookup["argv"], YOUTUBE_URL)


@pytest.mark.usefixtures("pillow_real")
def test_spec_addon_141_i2_a_loft_fetchs_download_allows_node_only(loft, gradient):
    _fetch_one(loft, gradient)

    (download,) = loft.youtube.download_commands()
    assert_node_only(download["argv"], YOUTUBE_URL)


@pytest.mark.usefixtures("pillow_real")
def test_spec_addon_141_i5_a_lookup_that_warns_of_no_runtime_is_a_success(
    loft, gradient, caplog,
):
    fetches = Fetches(loft, gradient, caplog)
    fetches.run("success")
    warned = fetches.loft(failures=2)
    fetches.prepare("success")
    fmt = loft.serve(gradient, duration=VIDEO_SECONDS, name="ok")
    loft.youtube.set_ytdlp(
        describe=describe([fmt], duration=VIDEO_SECONDS), stderr=MISSING_RUNTIME_WARNING,
    )

    with caplog.at_level("INFO"):
        fetch(warned)

    assert loft.flags(warned)[0] is True
    assert loft.failures(warned) <= 2
    assert not any(BACKING_OFF in r.getMessage() for r in caplog.records)

    after = fetches.run("lookup")

    assert loft.failures(after) == 1


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".")[:3])


def test_spec_addon_141_item_1_the_test_image_has_node_24_and_yt_dlps_ejs_component():
    from importlib.metadata import version

    node = shutil.which("node")
    assert node is not None, "no node on PATH"
    out = subprocess.run([node, "--version"], capture_output=True, text=True, check=True)
    assert out.stdout.strip().startswith("v24."), out.stdout
    assert _version(version("yt-dlp")) >= (2026, 10, 26)
    assert version("yt-dlp-ejs")
