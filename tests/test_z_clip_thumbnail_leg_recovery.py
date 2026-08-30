"""The thumbnail CLIP leg must recover on its own, and cost only itself.

``_index_clip_sync`` closes ``clip_thumbnail_indexed`` whenever the
representative JPEG cannot be reached, so the queue stops re-picking a
file it cannot finish. Two things then have to hold, or the leg stays
empty for the life of the library:

- something reopens the leg once the JPEG does arrive
  (``reset_falsely_completed_clip_thumbnail``), and
- reopening it must not re-extract every frame of a video whose scenes
  are already indexed, or the recovery costs more than the feature.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import database as db_mod  # noqa: E402
from app import indexer as indexer_mod  # noqa: E402
from app.models import Base, Embedding, IndexedFile  # noqa: E402
from app.workers import clip as clip_worker  # noqa: E402

VIDEO_MIME = "video/mp4"
IMAGE_MIME = "image/jpeg"

#: Filled by the autouse fixture; ``_seed`` writes its sources under it.
_MOUNT: list = []


@pytest.fixture()
def search_engine(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE vec_clip (embedding_id TEXT PRIMARY KEY, vector BLOB)"
        ))
    return engine


@pytest.fixture()
def Session(monkeypatch, search_engine):
    maker = sessionmaker(bind=search_engine, expire_on_commit=False)

    @contextmanager
    def _get_search_db():
        s = maker()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    for target in (db_mod, clip_worker, indexer_mod):
        monkeypatch.setattr(target, "get_search_db", _get_search_db)
    return maker


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch, tmp_path):
    # The reset gates on this directory existing, so give the default
    # case a real one. Tests about the missing mount override it.
    mount = tmp_path / "thumbs"
    (mount / "d").mkdir(parents=True)
    monkeypatch.setenv("HOMEVAULT_THUMBNAILS_DIR", str(mount))
    _MOUNT.append(mount)
    monkeypatch.setattr(
        clip_worker, "embed_image", lambda image: np.zeros(512, dtype=np.float32),
    )
    monkeypatch.setattr(
        clip_worker, "_generate_blip_caption_if_needed", lambda *a, **k: None,
    )
    fake = MagicMock()
    fake.convert.return_value = fake
    monkeypatch.setattr(clip_worker, "Image", MagicMock(open=lambda p: fake))
    monkeypatch.setattr(clip_worker, "validate_file_path", lambda p: True)
    monkeypatch.setattr(
        clip_worker, "_resolve_thumbnail_abspath", lambda p: f"/data/thumbnails/{p}",
    )


def _seed(
    Session, file_id, *, mime_type=VIDEO_MIME, thumbnail_path=None,
    clip_indexed=False, clip_thumbnail_indexed=False, active=True,
    with_thumb_embedding=False,
):
    mount = _MOUNT[-1] if _MOUNT else None
    if mount is not None and thumbnail_path:
        target = mount / thumbnail_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"jpeg")
    if mount is not None and mime_type in ("image/jpeg", "image/png"):
        source = mount / f"{file_id}.bin"
        source.write_bytes(b"jpeg")

    s = Session()
    try:
        s.add(IndexedFile(
            file_id=file_id,
            drive="default",
            filename=f"{file_id}.bin",
            file_path=str(mount / f"{file_id}.bin") if mount else f"/drives/{file_id}.bin",
            file_type="video" if mime_type.startswith("video/") else "image",
            mime_type=mime_type,
            file_size=64,
            thumbnail_path=thumbnail_path,
            clip_indexed=clip_indexed,
            clip_thumbnail_indexed=clip_thumbnail_indexed,
            active=active,
        ))
        if with_thumb_embedding:
            s.add(Embedding(
                id=f"clipt_{file_id}",
                file_id=file_id,
                embedding_type="clip_thumbnail",
                vector_table="vec_clip",
                content_preview="Thumbnail: x",
            ))
        s.commit()
    finally:
        s.close()


def _thumb_flag(Session, file_id) -> bool:
    s = Session()
    try:
        return bool(
            s.query(IndexedFile).filter_by(file_id=file_id).first()
            .clip_thumbnail_indexed
        )
    finally:
        s.close()


# ---------------------------------------------------------------------------
# The scene leg is not re-run to fill the thumbnail leg
# ---------------------------------------------------------------------------

def test_scene_leg_skipped_when_already_indexed(Session, monkeypatch):
    """A thumbnail-only retry must not re-extract the video's frames."""
    _seed(Session, "vid1", thumbnail_path="d/vid1.jpg", clip_indexed=True)
    scene = MagicMock(return_value=True)
    monkeypatch.setattr(clip_worker, "_index_clip_video", scene)

    assert clip_worker._index_clip_sync("vid1") is True

    scene.assert_not_called()
    assert _thumb_flag(Session, "vid1") is True


def test_scene_leg_runs_when_not_indexed(Session, monkeypatch):
    """A genuinely unindexed video still gets its scene pass."""
    _seed(Session, "vid2", thumbnail_path="d/vid2.jpg", clip_indexed=False)
    scene = MagicMock(return_value=True)
    monkeypatch.setattr(clip_worker, "_index_clip_video", scene)

    clip_worker._index_clip_sync("vid2")

    scene.assert_called_once()


# ---------------------------------------------------------------------------
# Reopening the leg
# ---------------------------------------------------------------------------

def test_video_with_thumbnail_but_no_embedding_is_reopened(Session):
    _seed(
        Session, "vid3", thumbnail_path="d/vid3.jpg",
        clip_indexed=True, clip_thumbnail_indexed=True,
    )
    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == ["vid3"]
    assert _thumb_flag(Session, "vid3") is False


def test_video_without_thumbnail_path_is_left_closed(Session):
    """No JPEG to reach means reopening would only close again next pass."""
    _seed(
        Session, "vid4", thumbnail_path=None,
        clip_indexed=True, clip_thumbnail_indexed=True,
    )
    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == []
    assert _thumb_flag(Session, "vid4") is True


def test_image_is_reopened_without_a_thumbnail_path(Session):
    """An image embeds its own file, so it needs no projected path."""
    _seed(
        Session, "img1", mime_type=IMAGE_MIME, thumbnail_path=None,
        clip_indexed=True, clip_thumbnail_indexed=True,
    )
    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == ["img1"]
    assert _thumb_flag(Session, "img1") is False


def test_existing_thumbnail_embedding_is_left_alone(Session):
    _seed(
        Session, "vid5", thumbnail_path="d/vid5.jpg",
        clip_indexed=True, clip_thumbnail_indexed=True,
        with_thumb_embedding=True,
    )
    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == []
    assert _thumb_flag(Session, "vid5") is True


def test_inactive_file_is_left_alone(Session):
    _seed(
        Session, "vid6", thumbnail_path="d/vid6.jpg",
        clip_indexed=True, clip_thumbnail_indexed=True, active=False,
    )
    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == []


def test_scene_rows_do_not_hide_an_empty_thumbnail_leg(Session):
    """The regression: ``clip`` rows satisfied the old completion check."""
    _seed(
        Session, "vid7", thumbnail_path="d/vid7.jpg",
        clip_indexed=True, clip_thumbnail_indexed=True,
    )
    s = Session()
    try:
        s.add(Embedding(
            id="clip_vid7_0", file_id="vid7", embedding_type="clip",
            vector_table="vec_clip", content_preview="Frame 0",
        ))
        s.commit()
    finally:
        s.close()

    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == ["vid7"]
    assert _thumb_flag(Session, "vid7") is False


# ---------------------------------------------------------------------------
# The mount itself
# ---------------------------------------------------------------------------

def test_a_missing_thumbnail_mount_is_reported(monkeypatch, caplog, tmp_path):
    """A stale override closes every video's leg without saying anything."""
    monkeypatch.setenv("HOMEVAULT_THUMBNAILS_DIR", str(tmp_path / "absent"))

    with caplog.at_level("WARNING"):
        assert clip_worker.warn_if_thumbnails_unreachable() is False

    assert any("not readable" in r.message for r in caplog.records)


def test_a_present_thumbnail_mount_is_silent(monkeypatch, caplog, tmp_path):
    present = tmp_path / "thumbnails"
    present.mkdir()
    monkeypatch.setenv("HOMEVAULT_THUMBNAILS_DIR", str(present))

    with caplog.at_level("WARNING"):
        assert clip_worker.warn_if_thumbnails_unreachable() is True

    assert not caplog.records


# ---------------------------------------------------------------------------
# Review findings (codex, PR #24)
# ---------------------------------------------------------------------------

def test_a_new_thumbnail_embedding_drops_the_similar_cache(Session, monkeypatch):
    """The cache has no TTL, and recovery writes arrive outside webhooks."""
    from app import search as search_mod

    _seed(Session, "vid8", thumbnail_path="d/vid8.jpg", clip_indexed=True)
    search_mod._similar_cache["vid8:6:default"] = object()

    assert clip_worker._index_clip_thumbnail(
        "vid8", "/data/thumbnails/d/vid8.jpg", "vid8.bin",
    ) is True

    assert search_mod._similar_cache == {}


@pytest.mark.asyncio
async def test_a_move_that_grants_a_thumbnail_path_reopens_the_leg(
    Session, monkeypatch,
):
    """Reconcile cannot notice: the path it would compare was just written."""
    _seed(
        Session, "vid9", thumbnail_path=None,
        clip_indexed=True, clip_thumbnail_indexed=True,
    )

    monkeypatch.setattr(
        indexer_mod, "_get_litloft_files_by_ids",
        lambda ids: {"vid9": {
            "drive": "default", "file_path": "moved/vid9.mp4",
            "filename": "vid9.mp4", "title": "",
            "thumbnail_path": "default/moved/vid9.jpg",
            "file_type": "video", "mime_type": VIDEO_MIME,
        }},
    )
    # The move's thumbnail is on disk; that is what makes the leg
    # reopenable rather than a terminal failure.
    moved = _MOUNT[-1] / "default" / "moved"
    moved.mkdir(parents=True, exist_ok=True)
    (moved / "vid9.jpg").write_bytes(b"jpeg")

    monkeypatch.setattr(indexer_mod, "resolve_file_path", lambda d, p: f"/drives/{p}")
    monkeypatch.setattr(indexer_mod, "delete_fts_file", lambda *a, **k: None)
    monkeypatch.setattr(indexer_mod, "upsert_fts_file", lambda *a, **k: None)

    import app.policy_client as policy
    async def _enabled(drive, feature):
        return True
    monkeypatch.setattr(policy, "is_feature_enabled", _enabled)

    queued = []
    async def _enqueue(task):
        queued.append(task.file_id)

    manager = indexer_mod.IndexManager.__new__(indexer_mod.IndexManager)
    manager._enqueue = _enqueue
    await indexer_mod.IndexManager.handle_files_moved(manager, ["vid9"])

    assert _thumb_flag(Session, "vid9") is False
    # The next reconcile is an hour away by default.
    assert queued == ["vid9"]


def test_a_missing_mount_stops_the_reopen_churn(Session, monkeypatch, tmp_path):
    """Reopening what cannot succeed queues, fails, and closes — every restart."""
    monkeypatch.setenv("HOMEVAULT_THUMBNAILS_DIR", str(tmp_path / "absent"))
    _seed(
        Session, "vid10", thumbnail_path="d/vid10.jpg",
        clip_indexed=True, clip_thumbnail_indexed=True,
    )
    _seed(
        Session, "img2", mime_type=IMAGE_MIME,
        clip_indexed=True, clip_thumbnail_indexed=True,
    )

    # The image route reads the file itself, so it is unaffected.
    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == ["img2"]
    assert _thumb_flag(Session, "vid10") is True


def test_the_reopen_can_be_scoped_to_named_files(Session):
    """files-moved knows exactly which rows it touched."""
    _seed(
        Session, "vid11", thumbnail_path="d/vid11.jpg",
        clip_indexed=True, clip_thumbnail_indexed=True,
    )
    _seed(
        Session, "vid12", thumbnail_path="d/vid12.jpg",
        clip_indexed=True, clip_thumbnail_indexed=True,
    )

    assert indexer_mod.reset_falsely_completed_clip_thumbnail(["vid11"]) == ["vid11"]
    assert _thumb_flag(Session, "vid12") is True
    assert indexer_mod.reset_falsely_completed_clip_thumbnail([]) == []


def test_a_thumbnail_that_is_gone_stays_closed(Session, tmp_path):
    """Closed-and-failed must not be read as closed-and-not-yet-rendered."""
    _seed(
        Session, "vid13", thumbnail_path="d/vid13.jpg",
        clip_indexed=True, clip_thumbnail_indexed=True,
    )
    (_MOUNT[-1] / "d" / "vid13.jpg").unlink()

    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == []
    assert _thumb_flag(Session, "vid13") is True


def test_a_path_escaping_the_mount_stays_closed(Session):
    """The dispatcher rejects it too, so reopening only repeats the failure."""
    _seed(
        Session, "vid14", thumbnail_path="../../etc/passwd",
        clip_indexed=True, clip_thumbnail_indexed=True,
    )

    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == []
    assert _thumb_flag(Session, "vid14") is True
