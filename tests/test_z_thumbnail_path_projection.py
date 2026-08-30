"""The addon's copy of ``thumbnail_path`` has to follow core's.

Core renders a video's thumbnail after the addon has already indexed
the file, so the row is written with no path. Reconcile's drift check
compares path, name and classification — none of which changed — so
nothing ever revisits it, and the thumbnail CLIP route has nothing to
open for the life of the file.

A move invalidates the value the same way: core derives the thumbnail's
own path from the file path.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

for _mod in ("PIL", "PIL.Image", "open_clip", "torch", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import database as db_mod  # noqa: E402
from app import indexer as indexer_mod  # noqa: E402
from app.models import Base, IndexedFile  # noqa: E402


@pytest.fixture()
def Session(monkeypatch, tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _get_search_db():
        s = maker()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    for target in (db_mod, indexer_mod):
        monkeypatch.setattr(target, "get_search_db", _get_search_db)
    return maker


def _seed(Session, file_id, thumbnail_path):
    s = Session()
    try:
        s.add(IndexedFile(
            file_id=file_id,
            drive="default",
            filename=f"{file_id}.mp4",
            file_path=f"/drives/{file_id}.mp4",
            file_type="video",
            mime_type="video/mp4",
            file_size=64,
            thumbnail_path=thumbnail_path,
            active=True,
        ))
        s.commit()
    finally:
        s.close()


def _stored(Session, file_id):
    s = Session()
    try:
        return s.query(IndexedFile).filter_by(file_id=file_id).first().thumbnail_path
    finally:
        s.close()


def test_a_thumbnail_rendered_after_indexing_is_picked_up(Session):
    _seed(Session, "vid1", None)

    synced = indexer_mod._sync_thumbnail_paths(
        {"vid1": {"thumbnail_path": "d/vid1.jpg"}},
        {"vid1": {"thumbnail_path": None}},
    )

    assert synced == 1
    assert _stored(Session, "vid1") == "d/vid1.jpg"


def test_an_unchanged_path_is_not_rewritten(Session):
    """Rewriting every row on every scan is what used to stall the loop."""
    _seed(Session, "vid2", "d/vid2.jpg")

    synced = indexer_mod._sync_thumbnail_paths(
        {"vid2": {"thumbnail_path": "d/vid2.jpg"}},
        {"vid2": {"thumbnail_path": "d/vid2.jpg"}},
    )

    assert synced == 0


def test_a_moved_files_new_path_replaces_the_old_one(Session):
    _seed(Session, "vid3", "old/vid3.jpg")

    synced = indexer_mod._sync_thumbnail_paths(
        {"vid3": {"thumbnail_path": "new/vid3.jpg"}},
        {"vid3": {"thumbnail_path": "old/vid3.jpg"}},
    )

    assert synced == 1
    assert _stored(Session, "vid3") == "new/vid3.jpg"


def test_a_cleared_path_propagates(Session):
    """Core dropping the thumbnail must not leave a dangling pointer."""
    _seed(Session, "vid4", "d/vid4.jpg")

    synced = indexer_mod._sync_thumbnail_paths(
        {"vid4": {"thumbnail_path": None}},
        {"vid4": {"thumbnail_path": "d/vid4.jpg"}},
    )

    assert synced == 1
    assert _stored(Session, "vid4") is None


def test_a_file_core_knows_but_the_index_does_not_is_left_to_the_insert_path(
    Session,
):
    synced = indexer_mod._sync_thumbnail_paths(
        {"unknown": {"thumbnail_path": "d/x.jpg"}},
        {},
    )

    assert synced == 0


def test_the_snapshot_carries_thumbnail_path(Session, monkeypatch):
    """``_sync_thumbnail_paths`` reads its "before" from this snapshot."""
    _seed(Session, "vid5", "d/vid5.jpg")

    @contextmanager
    def _read():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    monkeypatch.setattr(indexer_mod, "get_search_db_read", _read)

    meta = indexer_mod._get_indexed_metadata()

    assert meta["vid5"]["thumbnail_path"] == "d/vid5.jpg"
