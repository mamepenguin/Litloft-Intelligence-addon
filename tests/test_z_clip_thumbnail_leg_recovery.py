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
def _stub_embed(monkeypatch):
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
    s = Session()
    try:
        s.add(IndexedFile(
            file_id=file_id,
            drive="default",
            filename=f"{file_id}.bin",
            file_path=f"/drives/{file_id}.bin",
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
    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == 1
    assert _thumb_flag(Session, "vid3") is False


def test_video_without_thumbnail_path_is_left_closed(Session):
    """No JPEG to reach means reopening would only close again next pass."""
    _seed(
        Session, "vid4", thumbnail_path=None,
        clip_indexed=True, clip_thumbnail_indexed=True,
    )
    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == 0
    assert _thumb_flag(Session, "vid4") is True


def test_image_is_reopened_without_a_thumbnail_path(Session):
    """An image embeds its own file, so it needs no projected path."""
    _seed(
        Session, "img1", mime_type=IMAGE_MIME, thumbnail_path=None,
        clip_indexed=True, clip_thumbnail_indexed=True,
    )
    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == 1
    assert _thumb_flag(Session, "img1") is False


def test_existing_thumbnail_embedding_is_left_alone(Session):
    _seed(
        Session, "vid5", thumbnail_path="d/vid5.jpg",
        clip_indexed=True, clip_thumbnail_indexed=True,
        with_thumb_embedding=True,
    )
    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == 0
    assert _thumb_flag(Session, "vid5") is True


def test_inactive_file_is_left_alone(Session):
    _seed(
        Session, "vid6", thumbnail_path="d/vid6.jpg",
        clip_indexed=True, clip_thumbnail_indexed=True, active=False,
    )
    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == 0


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

    assert indexer_mod.reset_falsely_completed_clip_thumbnail() == 1
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
