"""SPEC-ADDON-146 items 2, 4-6, 9-10, §2, §8, I5, I6, I8-I10, and SPEC-ADDON-117 I6 /
I11 and SPEC-ADDON-078 I7 as revised: a YouTube ``.loft``'s scene candidates are built
from its downloaded video.

The fetch is the SPEC-ADDON-121 world: the rendition is the gradient video (cuts at 5,
12 and 25 s, 40 s long) and the lookup answer also names an ``sb0`` storyboard whose
tiles fall at 0, 10, 20 and 30 s.
"""

from __future__ import annotations

import importlib

import pytest
from sqlalchemy import text

from app.frame_cache import frame_cache_path
from tests.loft_media_world import GIB, index_off
from tests.ocr_fixtures import pillow_real  # noqa: F401 - the usefixtures marker resolves it
from tests.search_endpoint_db import patch_everywhere
from tests.test_spec_addon_079_scene_tiles import frame_dir_files, tile_names, tiles_dir
from tests.test_spec_addon_121_loft_fetch_storyboard import (  # noqa: F401 - fixtures
    KEPT_SECONDS,
    LOOKUP_FAILURE,
    fresh_worker_state,
    gradient,
    world,
)
from tests.vec_partition_db import search_index  # noqa: F401 - fixture
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    describe,
    everything_logged,
    needs_ffmpeg,
    youtube,
)

pytestmark = [needs_ffmpeg, pytest.mark.usefixtures("pillow_real", "fresh_worker_state")]

FLAG = "loft_candidates_done"
VIDEO_SECONDS = 40.0
# SPEC-ADDON-144 over the gradient video with I = 10: the cuts, anchor 0 and one fill
# time inside each of the gaps 12-25 and 25-40.
CANDIDATE_SECONDS = [0.0, 5.0, 12.0, 18.5, 25.0, 32.5]


def _approx(times):
    return pytest.approx(times, abs=0.15)


class CacheClears:
    def __init__(self, monkeypatch):
        self.count = 0
        # patch_everywhere only reaches modules already imported, and the
        # code under test imports app.search lazily.
        importlib.import_module("app.search")
        patch_everywhere(monkeypatch, "invalidate_similar_cache", self)

    def __call__(self, *_args, **_kwargs):
        self.count += 1


def _flag(world, file_id: str) -> bool:
    return bool(world.loft.value(file_id, FLAG))


def _candidates_only(world, *, failures: int = 0) -> tuple[str, str]:
    """A YouTube ``.loft`` already scanned and read, holding one storyboard row."""
    file_id = world.add(frame_scan_indexed=True)
    world.loft.set(file_id, loft_source="youtube", loft_fetch_failures=failures)
    world.loft.set(file_id, **{FLAG: 0})
    return file_id, world.add_row(file_id, "clip", 5.0)


def _warm(world) -> None:
    """A successful fetch, so the next lookup or download failure directly follows a success."""
    world.answer()
    warm = world.add()
    fetch(warm)
    assert world.loft.flags(warm)[0] is True


def fetch(file_id: str):
    from tests.loft_media_world import fetch as run

    return run(file_id)


# ---------------------------------------------------------------------------
# Item 5, I5: the stage replaces the storyboard with the video's own frames
# ---------------------------------------------------------------------------


def test_spec_addon_146_i5_a_fetch_replaces_the_storyboard_with_the_videos_scene_candidates(
    world, monkeypatch,
):
    world.answer()
    file_id = world.add()
    kept = frame_cache_path(file_id, 5.0)
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_bytes(b"KEPT")
    clears = CacheClears(monkeypatch)

    fetch(file_id)

    assert world.clip_seconds(file_id) == _approx(CANDIDATE_SECONDS)
    assert world.clip_seconds(file_id, "clip_change") == []
    assert set(world.clip_ids(file_id)) <= world.vectors()
    assert _flag(world, file_id) is True
    assert world.clip_indexed(file_id) is True
    assert tile_names(file_id) == []
    assert not tiles_dir(file_id).exists()
    assert kept.read_bytes() == b"KEPT"
    for t in world.clip_seconds(file_id):
        assert frame_cache_path(file_id, t).exists(), f"no small entry at {t}s"
    assert clears.count >= 1
    assert world.loft.fetch_dir_files() == []


def test_spec_addon_146_i8_an_already_fetched_loft_downloads_once_and_runs_only_the_candidates_stage(
    world,
):
    """Item 9, SPEC-ADDON-117 I6 as revised: a candidates-only fetch keeps the count."""
    world.answer()
    file_id, old_row = _candidates_only(world, failures=2)

    fetch(file_id)

    assert world.loft.lookups() == 1
    assert len({r["path"] for r in world.loft.downloads()}) == 1
    assert world.loft.scan_rows(file_id) == 0
    assert old_row not in world.clip_ids(file_id)
    assert world.clip_seconds(file_id) == _approx(CANDIDATE_SECONDS)
    assert _flag(world, file_id) is True
    assert world.loft.failures(file_id) == 2


def _cannot_load(*_args, **_kwargs):
    raise RuntimeError("the CLIP weights are not on disk")


def test_spec_addon_146_revised_117_i11_a_scan_alone_writes_no_frame_cache_entry_and_needs_no_model(
    world,
):
    """Revised SPEC-ADDON-117 I3: the model check follows the candidates stage, not the scan."""
    world.answer()
    file_id = world.add()
    world.add_row(file_id, "clip", 5.0)
    world.loft.set(file_id, loft_source="youtube", **{FLAG: 1})
    patch_everywhere(world.monkeypatch, "load_image_model", _cannot_load)

    fetch(file_id)

    assert world.loft.flags(file_id)[0] is True
    assert world.loft.scan_rows(file_id) == 1
    assert frame_dir_files(file_id) == []


# ---------------------------------------------------------------------------
# Item 6, revised SPEC-ADDON-078 I7: the storyboard leg checks the gate
# ---------------------------------------------------------------------------


def test_spec_addon_146_item_6_the_storyboard_leg_makes_no_lookup_while_the_gate_holds(world):
    world.answer()
    built, _ = _candidates_only(world)
    fetch(built)
    flag_without_rows = world.add()
    world.loft.set(flag_without_rows, loft_source="youtube", **{FLAG: 1})
    lookups = world.loft.lookups()
    candidates = world.clip_seconds(built)
    world.loft.set(built, clip_indexed=0)

    assert world.leg(built) is True

    assert world.loft.lookups() == lookups
    assert world.clip_seconds(built) == candidates
    assert tile_names(built) == []
    assert world.clip_indexed(built) is True

    assert world.leg(flag_without_rows) is True

    assert world.clip_seconds(flag_without_rows) == _approx(KEPT_SECONDS)


# ---------------------------------------------------------------------------
# Item 10, I9: a candidates-only fetch's outcome
# ---------------------------------------------------------------------------


def _prepare(world, kind: str) -> None:
    if kind == "lookup":
        world.youtube.set_ytdlp(fail=LOOKUP_FAILURE)
    elif kind == "download-403":
        world.answer()
        world.youtube.override("/v/rendition", status=403)
    elif kind == "no-rendition":
        world.youtube.set_ytdlp(describe=describe([], duration=VIDEO_SECONDS))
    elif kind == "short":
        world.answer(duration=100.0)
    elif kind == "disk-guard":
        world.answer()
        world.free.free = 1 * GIB
    elif kind == "index-off":
        world.answer()
        world.configure(policy=index_off)
    elif kind == "no-yt-dlp":
        world.answer()
        executable = world.youtube.bin / "yt-dlp"
        world.saved_ytdlp = (executable.read_bytes(), executable.stat().st_mode)
        world.youtube.remove_ytdlp()
    else:
        raise AssertionError(kind)


def _restore(world) -> None:
    """Undo ``_prepare``: the next fetch gets a usable answer."""
    world.configure()
    world.free.free = 100 * GIB
    saved = getattr(world, "saved_ytdlp", None)
    if saved is not None:
        executable = world.youtube.bin / "yt-dlp"
        executable.write_bytes(saved[0])
        executable.chmod(saved[1])
    world.answer()


@pytest.mark.parametrize(
    "kind, warm, closed",
    [
        pytest.param("lookup", True, True, id="lookup-failure-after-a-success"),
        pytest.param("download-403", True, True, id="download-failure-after-a-success"),
        pytest.param("no-rendition", False, True, id="no-rendition"),
        pytest.param("short", True, True, id="short-download"),
        pytest.param("lookup", False, False, id="lookup-failure-with-no-success-before"),
        pytest.param("disk-guard", True, False, id="disk-guard"),
        pytest.param("index-off", True, False, id="index-off"),
        pytest.param("no-yt-dlp", True, False, id="missing-yt-dlp"),
    ],
)
def test_spec_addon_146_i9_a_failed_candidates_only_fetch_closes_the_flag_only_when_counted(
    world, kind, warm, closed,
):
    if warm:
        _warm(world)
    file_id, old_row = _candidates_only(world, failures=1)
    _prepare(world, kind)

    fetch(file_id)

    assert _flag(world, file_id) is closed
    assert world.loft.failures(file_id) == 1
    assert world.clip_ids(file_id) == [old_row]
    if closed:
        return

    _restore(world)
    fetch(file_id)

    assert world.clip_seconds(file_id) == _approx(CANDIDATE_SECONDS)
    assert _flag(world, file_id) is True


def test_spec_addon_146_i10_the_third_counted_failure_of_a_scan_fetch_closes_the_flag(world):
    _warm(world)
    file_id = world.add()
    old_row = world.add_row(file_id, "clip", 5.0)
    world.loft.set(file_id, loft_source="youtube", loft_fetch_failures=2)
    world.youtube.set_ytdlp(fail=LOOKUP_FAILURE)

    fetch(file_id)

    assert world.loft.failures(file_id) == 3
    assert _flag(world, file_id) is True
    assert world.clip_ids(file_id) == [old_row]


# ---------------------------------------------------------------------------
# §2, §8, I6: a stage that fails changes nothing it did not own
# ---------------------------------------------------------------------------


def test_spec_addon_146_i6_a_stage_whose_write_fails_keeps_the_rows_tiles_and_cache_and_is_counted(
    world, caplog,
):
    _warm(world)
    file_id = world.add()
    old_row = world.add_row(file_id, "clip", 5.0)
    world.loft.set(file_id, loft_source="youtube")
    kept = frame_cache_path(file_id, 5.0)
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_bytes(b"KEPT")
    tiles_dir(file_id).mkdir(parents=True, exist_ok=True)
    (tiles_dir(file_id) / "0.webp").write_bytes(b"TILE")
    for event in ("INSERT", "UPDATE"):
        world.loft.index.sql(
            f"CREATE TRIGGER refuse_{event.lower()} BEFORE {event} ON embeddings "
            f"WHEN NEW.file_id = '{file_id}' AND NEW.embedding_type = 'clip' "
            "BEGIN SELECT RAISE(ABORT, 'refused'); END"
        )
    world.answer()

    with caplog.at_level("INFO"):
        fetch(file_id)

    assert world.clip_ids(file_id) == [old_row]
    assert _flag(world, file_id) is False
    assert world.loft.failures(file_id) == 1
    assert kept.read_bytes() == b"KEPT"
    assert frame_dir_files(file_id) == [kept.name]
    assert tile_names(file_id) == ["0.webp"]
    assert (
        f"loft_media candidates failed for {file_id} (IntegrityError)"
        in everything_logged(caplog)
    )


def test_spec_addon_146_i6_a_loft_trashed_during_the_stage_keeps_its_rows_and_is_not_counted(
    world,
):
    world.answer()
    file_id, old_row = _candidates_only(world, failures=1)
    embedder = world.embedder

    def trash_then_embed(image, *args, **kwargs):
        world.loft.set(file_id, active=0)
        return embedder(image, *args, **kwargs)

    patch_everywhere(world.monkeypatch, "embed_image", trash_then_embed)

    fetch(file_id)

    assert world.clip_ids(file_id) == [old_row]
    assert _flag(world, file_id) is False
    assert world.loft.failures(file_id) == 1
    assert frame_dir_files(file_id) == []

    world.loft.set(file_id, active=1)
    patch_everywhere(world.monkeypatch, "embed_image", embedder)
    fetch(file_id)

    assert world.clip_seconds(file_id) == _approx(CANDIDATE_SECONDS)


# ---------------------------------------------------------------------------
# Item 2: classification as source other closes the flag
# ---------------------------------------------------------------------------


def test_spec_addon_146_item_2_a_loft_classified_as_other_has_the_flag_set(world):
    world.answer()
    file_id = world.add()
    path = world.loft.drive_dir / f"{file_id}.loft"
    path.write_text('{"provider": "vimeo", "url": "https://vimeo.com/1"}')

    fetch(file_id)

    assert world.loft.source(file_id) == "other"
    assert _flag(world, file_id) is True


def _clip_rows(world, file_id: str) -> list[tuple]:
    with world.loft.index.engine.connect() as conn:
        return [tuple(r) for r in conn.execute(text(
            "SELECT id, timestamp_start, timestamp_end FROM embeddings "
            "WHERE file_id = :f AND embedding_type = 'clip' ORDER BY timestamp_start"
        ), {"f": file_id})]


def test_spec_addon_146_i5_the_stage_rows_end_30_s_after_they_start(world):
    world.answer()
    file_id, _ = _candidates_only(world)

    fetch(file_id)

    rows = _clip_rows(world, file_id)
    assert [r[1] for r in rows] == _approx(CANDIDATE_SECONDS)
    for _, start, end in rows:
        assert end == pytest.approx(start + 30)


def test_spec_addon_146_item_4_a_candidates_fetch_without_the_model_ends_before_the_lookup(world):
    world.answer()
    file_id, old_row = _candidates_only(world, failures=1)
    patch_everywhere(world.monkeypatch, "load_image_model", _cannot_load)

    fetch(file_id)

    assert world.loft.lookups() == 0
    assert world.clip_ids(file_id) == [old_row]
    assert _flag(world, file_id) is False
    assert world.loft.failures(file_id) == 1

    world.configure()
    fetch(file_id)

    assert world.clip_seconds(file_id) == _approx(CANDIDATE_SECONDS)
