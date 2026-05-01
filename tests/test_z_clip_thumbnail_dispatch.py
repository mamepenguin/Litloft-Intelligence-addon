"""Dispatch + indexing tests for the ``clip_thumbnail`` route.

Spec: ``2026-05-02-thumbnail-clip-default-shallow-search.md``.

The CLIP worker now has three branches:

- IMAGE_TYPES → ``_index_clip_thumbnail`` only (image *is* the thumbnail).
- VIDEO_TYPES → ``_index_clip_video`` (scene frames) +
  ``_index_clip_thumbnail`` (representative frame from
  ``data/thumbnails/<id>.jpg``).
- THUMBNAIL_FALLBACK_TYPES (.loft, HEIC) → ``_index_clip_thumbnail`` only,
  using ``IndexedFile.thumbnail_path``.

These tests verify the dispatch picks the right route and that
``embedding_type="clip_thumbnail"`` rows land in the embeddings table
with the right flags flipped.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Stub heavy ML deps. ``conftest.py`` already covers most, but the
# CLIP module also reaches into ``open_clip`` lazily; keep the stubs
# present so ``import app.workers.clip`` does not blow up.
for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import database as db_mod  # noqa: E402
from app.models import Base, Embedding, IndexedFile  # noqa: E402
from app.workers import clip as clip_worker  # noqa: E402




@pytest.fixture()
def search_engine(tmp_path):
    """In-memory-ish search DB with the schema the worker expects."""
    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    # vec_clip is a virtual table normally, but for these tests a plain
    # table that accepts the same INSERT shape is sufficient — we never
    # query it back.
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE vec_clip ("
            "  embedding_id TEXT PRIMARY KEY,"
            "  vector BLOB"
            ")"
        ))
    return engine


@pytest.fixture()
def patched_db(monkeypatch, search_engine):
    """Wire ``get_search_db`` to the test engine."""
    Session = sessionmaker(bind=search_engine, expire_on_commit=False)

    @contextmanager
    def _get_search_db():
        s = Session()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    monkeypatch.setattr(db_mod, "get_search_db", _get_search_db)
    monkeypatch.setattr(clip_worker, "get_search_db", _get_search_db)
    return Session


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch):
    """Replace ``embed_image`` and BLIP caption gen with deterministic fakes."""
    monkeypatch.setattr(
        clip_worker, "embed_image",
        lambda image: np.zeros(512, dtype=np.float32),
    )
    # BLIP caption is a side-effect; just make it a no-op.
    monkeypatch.setattr(
        clip_worker, "_generate_blip_caption_if_needed",
        lambda *args, **kwargs: None,
    )
    # ``Image.open`` is stubbed by conftest, but we call ``.convert`` on
    # its return; make a chainable mock to avoid AttributeError.
    fake_image = MagicMock()
    fake_image.convert.return_value = fake_image
    monkeypatch.setattr(clip_worker, "Image", MagicMock(open=lambda p: fake_image))
    # ``validate_file_path`` is unrelated — say yes.
    monkeypatch.setattr(clip_worker, "validate_file_path", lambda p: True)


def _seed_indexed_file(
    Session, file_id: str, *, mime_type: str, file_path: str = "/drives/x.bin",
    thumbnail_path: str | None = None,
) -> None:
    s = Session()
    try:
        f = IndexedFile(
            file_id=file_id,
            drive="default",
            filename=f"{file_id}.bin",
            file_path=file_path,
            file_type="video" if mime_type.startswith("video/") else "image",
            mime_type=mime_type,
            file_size=64,
            thumbnail_path=thumbnail_path,
            active=True,
        )
        s.add(f)
        s.commit()
    finally:
        s.close()


def _emb_types(Session, file_id: str) -> list[str]:
    s = Session()
    try:
        return [
            row.embedding_type
            for row in s.query(Embedding).filter_by(file_id=file_id).all()
        ]
    finally:
        s.close()


def _flags(Session, file_id: str) -> tuple[bool, bool]:
    s = Session()
    try:
        f = s.query(IndexedFile).filter_by(file_id=file_id).first()
        return (bool(f.clip_indexed), bool(f.clip_thumbnail_indexed))
    finally:
        s.close()


# ---------------------------------------------------------------------------
# IMAGE route: only writes clip_thumbnail
# ---------------------------------------------------------------------------


def test_image_writes_clip_thumbnail_only(patched_db):
    Session = patched_db
    _seed_indexed_file(Session, "img-1", mime_type="image/jpeg")

    ok = clip_worker._index_clip_sync("img-1")
    assert ok is True

    types = _emb_types(Session, "img-1")
    assert types == ["clip_thumbnail"]

    clip_done, thumb_done = _flags(Session, "img-1")
    assert clip_done is True   # image leg closes both flags
    assert thumb_done is True


# ---------------------------------------------------------------------------
# THUMBNAIL_FALLBACK route: .loft / HEIC
# ---------------------------------------------------------------------------


def test_loft_with_thumbnail_writes_clip_thumbnail(monkeypatch, patched_db, tmp_path):
    """`.loft` files use ``thumbnail_path`` from IndexedFile."""
    Session = patched_db
    # Create a real image file so PIL.Image.open (stubbed) does not 404
    thumb_dir = tmp_path / "thumbnails"
    thumb_dir.mkdir()
    thumb_rel = "default/clip.jpg"
    (thumb_dir / "default").mkdir()
    (thumb_dir / thumb_rel).write_bytes(b"\x00" * 8)

    monkeypatch.setenv("HOMEVAULT_THUMBNAILS_DIR", str(thumb_dir))

    _seed_indexed_file(
        Session, "loft-1",
        mime_type="application/vnd.litloft.loft+json",
        thumbnail_path=thumb_rel,
    )

    ok = clip_worker._index_clip_sync("loft-1")
    assert ok is True

    types = _emb_types(Session, "loft-1")
    assert types == ["clip_thumbnail"]

    clip_done, thumb_done = _flags(Session, "loft-1")
    assert thumb_done is True


def test_loft_without_thumbnail_closes_flags_without_emb(patched_db):
    """Legacy `.loft` rows with no thumbnail_path are skipped cleanly."""
    Session = patched_db
    _seed_indexed_file(
        Session, "loft-2",
        mime_type="application/vnd.litloft.loft+json",
        thumbnail_path=None,
    )

    ok = clip_worker._index_clip_sync("loft-2")
    assert ok is True

    types = _emb_types(Session, "loft-2")
    assert types == []

    clip_done, thumb_done = _flags(Session, "loft-2")
    assert clip_done is True
    assert thumb_done is True


def test_heic_routes_to_thumbnail_fallback(monkeypatch, patched_db, tmp_path):
    """HEIC uses the same path as `.loft` (avoids Pillow on raw HEIC)."""
    Session = patched_db
    thumb_dir = tmp_path / "thumbnails"
    thumb_dir.mkdir()
    (thumb_dir / "default").mkdir()
    (thumb_dir / "default/heic.jpg").write_bytes(b"\x00" * 8)

    monkeypatch.setenv("HOMEVAULT_THUMBNAILS_DIR", str(thumb_dir))

    _seed_indexed_file(
        Session, "heic-1",
        mime_type="image/heic",
        thumbnail_path="default/heic.jpg",
    )

    ok = clip_worker._index_clip_sync("heic-1")
    assert ok is True

    assert _emb_types(Session, "heic-1") == ["clip_thumbnail"]


# ---------------------------------------------------------------------------
# VIDEO route: writes both scene clip AND thumbnail (when available)
# ---------------------------------------------------------------------------


def test_video_with_thumbnail_writes_both_routes(monkeypatch, patched_db, tmp_path):
    """Video gets scene clips (existing) + clip_thumbnail (new)."""
    Session = patched_db
    thumb_dir = tmp_path / "thumbnails"
    thumb_dir.mkdir()
    (thumb_dir / "default").mkdir()
    (thumb_dir / "default/video.jpg").write_bytes(b"\x00" * 8)

    monkeypatch.setenv("HOMEVAULT_THUMBNAILS_DIR", str(thumb_dir))

    _seed_indexed_file(
        Session, "vid-1",
        mime_type="video/mp4",
        file_path="/drives/default/v.mp4",
        thumbnail_path="default/video.jpg",
    )

    # _index_clip_video is heavy (ffmpeg); stub it to write a fake "clip"
    # embedding so the dispatch contract is verifiable.
    def _fake_video(file_id, file_path, duration):
        with patched_db().__enter__() if False else clip_worker.get_search_db() as s:
            from app.models import Embedding
            emb = Embedding(
                id=f"clip_{file_id}_fake",
                file_id=file_id,
                embedding_type="clip",
                vector_table="vec_clip",
                content_preview="fake",
            )
            s.add(emb)
            s.flush()
            s.execute(
                text("INSERT INTO vec_clip(embedding_id, vector) VALUES(:id, :v)"),
                {"id": emb.id, "v": b"\x00" * 8},
            )
            f = s.query(IndexedFile).filter_by(file_id=file_id).first()
            if f is not None:
                f.clip_indexed = True
        return True

    monkeypatch.setattr(clip_worker, "_index_clip_video", _fake_video)

    ok = clip_worker._index_clip_sync("vid-1")
    assert ok is True

    types = sorted(_emb_types(Session, "vid-1"))
    assert types == ["clip", "clip_thumbnail"]

    clip_done, thumb_done = _flags(Session, "vid-1")
    assert clip_done is True
    assert thumb_done is True


def test_video_without_thumbnail_writes_only_scene_clip(monkeypatch, patched_db):
    """If core has not written a thumbnail yet, only scene CLIP runs."""
    Session = patched_db
    _seed_indexed_file(
        Session, "vid-2",
        mime_type="video/mp4",
        file_path="/drives/default/no_thumb.mp4",
        thumbnail_path=None,
    )

    def _fake_video(file_id, file_path, duration):
        with clip_worker.get_search_db() as s:
            from app.models import Embedding
            emb = Embedding(
                id=f"clip_{file_id}_fake",
                file_id=file_id,
                embedding_type="clip",
                vector_table="vec_clip",
                content_preview="fake",
            )
            s.add(emb)
            s.flush()
            s.execute(
                text("INSERT INTO vec_clip(embedding_id, vector) VALUES(:id, :v)"),
                {"id": emb.id, "v": b"\x00" * 8},
            )
            f = s.query(IndexedFile).filter_by(file_id=file_id).first()
            if f is not None:
                f.clip_indexed = True
        return True

    monkeypatch.setattr(clip_worker, "_index_clip_video", _fake_video)

    ok = clip_worker._index_clip_sync("vid-2")
    assert ok is True

    types = sorted(_emb_types(Session, "vid-2"))
    assert types == ["clip"]

    clip_done, thumb_done = _flags(Session, "vid-2")
    assert clip_done is True
    assert thumb_done is True  # closed despite no thumbnail emb


# ---------------------------------------------------------------------------
# Unsupported mime: closes both flags, no embedding written
# ---------------------------------------------------------------------------


def test_unsupported_mime_closes_flags(patched_db):
    Session = patched_db
    _seed_indexed_file(Session, "txt-1", mime_type="text/plain")

    ok = clip_worker._index_clip_sync("txt-1")
    assert ok is True

    assert _emb_types(Session, "txt-1") == []
    clip_done, thumb_done = _flags(Session, "txt-1")
    assert clip_done is True
    assert thumb_done is True


# ---------------------------------------------------------------------------
# _resolve_thumbnail_abspath: containment defense-in-depth
# ---------------------------------------------------------------------------


def test_resolve_thumbnail_abspath_relative_in_base(monkeypatch, tmp_path):
    base = tmp_path / "thumbnails"
    base.mkdir()
    monkeypatch.setenv("HOMEVAULT_THUMBNAILS_DIR", str(base))

    out = clip_worker._resolve_thumbnail_abspath("default/clip.jpg")
    assert out == str(base / "default/clip.jpg")


def test_resolve_thumbnail_abspath_rejects_dotdot_traversal(monkeypatch, tmp_path):
    base = tmp_path / "thumbnails"
    base.mkdir()
    (tmp_path / "secret.txt").write_text("x")
    monkeypatch.setenv("HOMEVAULT_THUMBNAILS_DIR", str(base))

    with pytest.raises(ValueError, match="escapes mount root"):
        clip_worker._resolve_thumbnail_abspath("../secret.txt")


def test_resolve_thumbnail_abspath_rejects_absolute_outside_base(
    monkeypatch, tmp_path,
):
    base = tmp_path / "thumbnails"
    base.mkdir()
    monkeypatch.setenv("HOMEVAULT_THUMBNAILS_DIR", str(base))

    with pytest.raises(ValueError, match="escapes mount root"):
        clip_worker._resolve_thumbnail_abspath("/etc/passwd")


def test_resolve_thumbnail_abspath_accepts_absolute_inside_base(
    monkeypatch, tmp_path,
):
    base = tmp_path / "thumbnails"
    base.mkdir()
    monkeypatch.setenv("HOMEVAULT_THUMBNAILS_DIR", str(base))
    real_inside = base / "ok.jpg"
    real_inside.write_bytes(b"\x00")

    out = clip_worker._resolve_thumbnail_abspath(str(real_inside))
    assert out == str(real_inside)


def test_loft_with_traversing_thumbnail_path_skips_cleanly(
    monkeypatch, patched_db, tmp_path,
):
    """Defense-in-depth: a poisoned thumbnail_path must not crash the worker.

    The dispatcher swallows the ``ValueError`` from
    ``_resolve_thumbnail_abspath`` and closes the flags so the queue
    does not spin on the poisoned row.
    """
    Session = patched_db
    base = tmp_path / "thumbnails"
    base.mkdir()
    monkeypatch.setenv("HOMEVAULT_THUMBNAILS_DIR", str(base))

    _seed_indexed_file(
        Session, "loft-evil",
        mime_type="application/vnd.litloft.loft+json",
        thumbnail_path="../../etc/passwd",
    )

    ok = clip_worker._index_clip_sync("loft-evil")
    assert ok is True
    assert _emb_types(Session, "loft-evil") == []
    clip_done, thumb_done = _flags(Session, "loft-evil")
    assert clip_done is True
    assert thumb_done is True
