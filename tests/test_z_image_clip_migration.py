"""Tests for the one-shot image ``clip`` → ``clip_thumbnail`` migration.

Spec ``2026-05-02-thumbnail-clip-default-shallow-search.md`` §3.6:

* Existing image embeddings written before this rollout had
  ``embedding_type='clip'``. The new dispatch reserves ``"clip"`` for
  scene-detected video frames and tags single-image embeddings as
  ``"clip_thumbnail"``. ``_migrate_image_clip_to_clip_thumbnail``
  rebrands them in chunks.

* Video rows must be left alone — their ``clip`` rows are scene
  frames that keep their meaning.

* Migration must be idempotent and self-heal across restarts.

This file is named ``test_z_*`` for the same pytest-asyncio +
legacy-event-loop ordering reason as
``test_z_clip_thumbnail_dispatch.py``.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# Stub heavy ML deps before importing app.database (transitive imports).
for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine, text  # noqa: E402


def _make_search_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE indexed_files ("
            "  file_id TEXT PRIMARY KEY,"
            "  drive TEXT NOT NULL,"
            "  filename TEXT NOT NULL,"
            "  file_path TEXT NOT NULL,"
            "  file_type TEXT NOT NULL,"
            "  mime_type TEXT NOT NULL,"
            "  file_size INTEGER NOT NULL,"
            "  thumbnail_path TEXT,"
            "  active BOOLEAN NOT NULL DEFAULT 1,"
            "  metadata_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  clip_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  clip_thumbnail_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  whisper_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  text_indexed BOOLEAN NOT NULL DEFAULT 0,"
            "  title TEXT NOT NULL DEFAULT '',"
            "  description TEXT NOT NULL DEFAULT '',"
            "  tags_text TEXT NOT NULL DEFAULT '',"
            "  indexed_at TEXT,"
            "  updated_at TEXT"
            ")"
        ))
        conn.execute(text(
            "CREATE TABLE embeddings ("
            "  id TEXT PRIMARY KEY,"
            "  file_id TEXT NOT NULL,"
            "  embedding_type TEXT NOT NULL,"
            "  timestamp_start REAL,"
            "  timestamp_end REAL,"
            "  content_preview TEXT NOT NULL DEFAULT '',"
            "  vector_table TEXT NOT NULL,"
            "  created_at TEXT"
            ")"
        ))
    return engine


def _seed_indexed_file(engine, file_id, mime_type, *, clip_indexed=False):
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO indexed_files "
                "(file_id, drive, filename, file_path, file_type, "
                "mime_type, file_size, clip_indexed) "
                "VALUES (:fid, 'd', :fn, :fp, 'image', :m, 1, :ci)"
            ),
            {
                "fid": file_id, "fn": file_id, "fp": f"/x/{file_id}",
                "m": mime_type, "ci": int(clip_indexed),
            },
        )


def _seed_embedding(engine, emb_id, file_id, embedding_type):
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO embeddings "
                "(id, file_id, embedding_type, vector_table) "
                "VALUES (:i, :f, :t, 'vec_clip')"
            ),
            {"i": emb_id, "f": file_id, "t": embedding_type},
        )


def _embedding_types_for(engine, file_id) -> list[str]:
    with engine.connect() as conn:
        return [
            row[0]
            for row in conn.execute(
                text(
                    "SELECT embedding_type FROM embeddings "
                    "WHERE file_id = :f ORDER BY id"
                ),
                {"f": file_id},
            ).fetchall()
        ]


def _flag(engine, file_id) -> tuple[int, int]:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT clip_indexed, clip_thumbnail_indexed "
                "FROM indexed_files WHERE file_id = :f"
            ),
            {"f": file_id},
        ).fetchone()
        return (row[0], row[1])


@pytest.fixture()
def patched_engine(monkeypatch, tmp_path):
    """Wire ``app.database._search_engine`` to an in-memory test DB."""
    from app import database as db_mod

    engine = _make_search_db(tmp_path)
    monkeypatch.setattr(db_mod, "_search_engine", engine)
    return engine


# ---------------------------------------------------------------------------
# Happy path: image rows get rebranded
# ---------------------------------------------------------------------------


def test_image_clip_rows_rebranded_to_clip_thumbnail(patched_engine):
    from app.database import _migrate_image_clip_to_clip_thumbnail

    _seed_indexed_file(patched_engine, "img-1", "image/jpeg")
    _seed_embedding(patched_engine, "e1", "img-1", "clip")

    _migrate_image_clip_to_clip_thumbnail()

    assert _embedding_types_for(patched_engine, "img-1") == ["clip_thumbnail"]


def test_image_migration_sets_clip_thumbnail_indexed(patched_engine):
    from app.database import _migrate_image_clip_to_clip_thumbnail

    _seed_indexed_file(patched_engine, "img-1", "image/png", clip_indexed=True)
    _seed_embedding(patched_engine, "e1", "img-1", "clip")

    _migrate_image_clip_to_clip_thumbnail()

    clip_done, thumb_done = _flag(patched_engine, "img-1")
    assert thumb_done == 1


# ---------------------------------------------------------------------------
# Video rows must be left alone — scene clip retains semantic meaning
# ---------------------------------------------------------------------------


def test_video_clip_rows_unchanged(patched_engine):
    from app.database import _migrate_image_clip_to_clip_thumbnail

    with patched_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO indexed_files "
                "(file_id, drive, filename, file_path, file_type, "
                "mime_type, file_size, clip_indexed) "
                "VALUES ('vid-1', 'd', 'v.mp4', '/x/v.mp4', 'video', "
                "'video/mp4', 100, 1)"
            )
        )
    # Multiple scene-frame embeddings.
    for i in range(3):
        _seed_embedding(patched_engine, f"v{i}", "vid-1", "clip")

    _migrate_image_clip_to_clip_thumbnail()

    assert _embedding_types_for(patched_engine, "vid-1") == ["clip"] * 3


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_migration_is_idempotent(patched_engine):
    from app.database import _migrate_image_clip_to_clip_thumbnail

    _seed_indexed_file(patched_engine, "img-1", "image/jpeg")
    _seed_embedding(patched_engine, "e1", "img-1", "clip")

    _migrate_image_clip_to_clip_thumbnail()
    # Second run: no rows left to migrate, must not raise or duplicate.
    _migrate_image_clip_to_clip_thumbnail()

    assert _embedding_types_for(patched_engine, "img-1") == ["clip_thumbnail"]


def test_migration_noop_when_no_image_clip_rows(patched_engine):
    from app.database import _migrate_image_clip_to_clip_thumbnail

    # Empty DB: must not raise.
    _migrate_image_clip_to_clip_thumbnail()


# ---------------------------------------------------------------------------
# Chunked write: a long list crosses _IMAGE_MIGRATION_CHUNK boundary
# ---------------------------------------------------------------------------


def test_chunked_writes_handle_more_than_one_batch(monkeypatch, patched_engine):
    """Force a tiny chunk size and seed enough rows to exercise paging."""
    from app import database as db_mod
    from app.database import _migrate_image_clip_to_clip_thumbnail

    monkeypatch.setattr(db_mod, "_IMAGE_MIGRATION_CHUNK", 3)

    n = 10
    for i in range(n):
        fid = f"img-{i:02d}"
        _seed_indexed_file(patched_engine, fid, "image/jpeg")
        _seed_embedding(patched_engine, f"e{i}", fid, "clip")

    _migrate_image_clip_to_clip_thumbnail()

    for i in range(n):
        fid = f"img-{i:02d}"
        assert _embedding_types_for(patched_engine, fid) == ["clip_thumbnail"]
        clip_done, thumb_done = _flag(patched_engine, fid)
        assert thumb_done == 1
