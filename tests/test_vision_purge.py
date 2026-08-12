"""Per-drive purge tests for the vision_describe feature (RED phase).

When the host flips ``addons.intelligence.vision_describe`` to ``false``
(or the umbrella ``addons.intelligence`` flag goes off) for a drive, the
existing vision artefacts for that drive must be wiped:

1. ``file_summaries.visual_description*`` columns NULL'd or row removed
2. ``embeddings`` rows with ``embedding_type = "vision_description"``
   for any file in that drive deleted

Follows the shape of ``test_purge.py`` — the whole-drive purge path is
reused from ``app.purge.purge_drive`` (per-file ``_purge_file``), so the
only extra surface tested here is:

* ``_purge_file`` wipes vision columns alongside the rest of
  ``file_summaries`` and removes vision_description embeddings.
* A per-feature purge helper (``purge_vision_for_drive`` or
  equivalent — spec-level name) clears only vision columns/embeddings
  when the umbrella index stays on but vision_describe goes off.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

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
from sqlalchemy.orm import sessionmaker  # noqa: E402


def _build_engine(tmp_path):
    from app.database import (
        Base,
        _create_file_summaries_table,
        _create_suggested_chapters_table,
        _create_suggested_tags_table,
    )

    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _create_file_summaries_table(conn)
        _create_suggested_chapters_table(conn)
        _create_suggested_tags_table(conn)
        # FTS mirrors (purge touches these).
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_files "
            "USING fts5(file_id, filename, title, description, "
            "tags_text, tokenize='trigram')"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_transcripts "
            "USING fts5(file_id, chunk_index, text, tokenize='trigram')"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content "
            "USING fts5(file_id, chunk_index, page, text, tokenize='trigram')"
        ))
        # Phase 3 dual-index parallel tables.
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_files_word "
            "USING fts5(file_id, filename, title, description, "
            "tags_text, tokenize=\"unicode61 remove_diacritics 2\")"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_transcripts_word "
            "USING fts5(file_id, chunk_index, text, "
            "tokenize=\"unicode61 remove_diacritics 2\")"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content_word "
            "USING fts5(file_id, chunk_index, page, text, "
            "tokenize=\"unicode61 remove_diacritics 2\")"
        ))
        # Seed a minimal real ``vec_text`` table so ``_purge_file``'s
        # unconditional DELETE-from-vec_text path succeeds without
        # needing sqlite-vec loaded. A plain relational table with the
        # same column ``embedding_id`` is enough for the DELETE
        # statements used in tests.
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS vec_text ("
            "embedding_id TEXT PRIMARY KEY, vector BLOB)"
        ))
    return engine


def _seed_image_file(engine, *, file_id="img-1", drive="work"):
    from app.models import Embedding, IndexedFile

    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()
    try:
        session.add(
            IndexedFile(
                file_id=file_id,
                drive=drive,
                filename="photo.jpg",
                file_path=f"/drives/{drive}/photo.jpg",
                file_type="image",
                mime_type="image/jpeg",
                file_size=5000,
                active=True,
            )
        )
        session.add(
            Embedding(
                id=f"vd_{file_id}",
                file_id=file_id,
                embedding_type="vision_description",
                content_preview="A red apple on a wooden table.",
                vector_table="vec_text",
            )
        )
        session.commit()
    finally:
        session.close()

    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO file_summaries "
                "(file_id, short_summary, long_summary, model, "
                "context_type, context_chars, was_truncated, status, "
                "created_at, visual_description, visual_description_status, "
                "visual_description_model, visual_description_generated_at) "
                "VALUES (:fid, '', '', '', 'image', 0, 0, 'hidden', "
                ":now, 'A red apple.', 'success', 'llava:13b', :now)"
            ),
            {"fid": file_id, "now": now},
        )

    return Session


def _patch_get_search_db(monkeypatch, Session):
    @contextmanager
    def _get_search_db():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    monkeypatch.setattr("app.indexer.get_search_db", _get_search_db)


# ---------------------------------------------------------------------------
# Drive-level purge (umbrella index off)
# ---------------------------------------------------------------------------


def test_purge_file_removes_vision_columns_and_embedding(tmp_path, monkeypatch):
    """``_purge_file`` must wipe visual_description* AND the embedding row.

    Since the vision columns live on ``file_summaries`` (which
    ``_purge_file`` already deletes wholesale), this test mainly guards
    against a future refactor that splits vision into its own table.
    """
    engine = _build_engine(tmp_path)
    Session = _seed_image_file(engine, file_id="img-1", drive="work")
    _patch_get_search_db(monkeypatch, Session)
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)

    from app.indexer import _purge_file

    _purge_file("img-1")

    with engine.connect() as conn:
        fs_row = conn.execute(
            text("SELECT 1 FROM file_summaries WHERE file_id = :fid"),
            {"fid": "img-1"},
        ).fetchone()
        emb_row = conn.execute(
            text(
                "SELECT 1 FROM embeddings "
                "WHERE file_id = :fid AND embedding_type = 'vision_description'"
            ),
            {"fid": "img-1"},
        ).fetchone()

    assert fs_row is None
    assert emb_row is None


# ---------------------------------------------------------------------------
# Per-feature purge (umbrella stays on, vision_describe flips off)
# ---------------------------------------------------------------------------


def test_purge_vision_for_drive_clears_only_vision_artefacts(
    tmp_path, monkeypatch,
):
    """Umbrella index stays on; vision_describe flips off → clear vision only.

    The drive's other intelligence data (transcripts, clip embeddings,
    short/long summaries) must survive. Only the vision columns and
    ``vision_description`` embeddings are cleared.
    """
    # Implementation is expected to expose a purge helper — either
    # ``app.purge.purge_vision_for_drive`` or a generic
    # ``purge_feature_for_drive(drive, "vision_describe")``. We try
    # both and skip if neither is present (RED phase).
    try:
        from app.purge import purge_vision_for_drive as _purge_vision
    except ImportError:
        try:
            from app.purge import purge_feature_for_drive

            def _purge_vision(drive: str) -> int:  # type: ignore[misc]
                return purge_feature_for_drive(drive, "vision_describe")
        except ImportError:
            pytest.skip("vision purge helper not implemented (RED phase)")

    engine = _build_engine(tmp_path)
    Session = _seed_image_file(engine, file_id="img-1", drive="work")

    # Seed short/long summary fields on the SAME row so we can verify
    # they survive the vision-only purge.
    now = datetime.now(UTC).isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE file_summaries "
                "SET short_summary = 'short', long_summary = 'long', "
                "    model = 'gemma', status = 'generated', "
                "    context_chars = 100 "
                "WHERE file_id = :fid"
            ),
            {"fid": "img-1"},
        )

    _patch_get_search_db(monkeypatch, Session)
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)

    purged = _purge_vision("work")
    assert purged >= 1

    with engine.connect() as conn:
        fs = conn.execute(
            text(
                "SELECT short_summary, long_summary, "
                "visual_description, visual_description_status, "
                "visual_description_model "
                "FROM file_summaries WHERE file_id = :fid"
            ),
            {"fid": "img-1"},
        ).fetchone()
        vision_emb = conn.execute(
            text(
                "SELECT 1 FROM embeddings "
                "WHERE file_id = :fid "
                "AND embedding_type = 'vision_description'"
            ),
            {"fid": "img-1"},
        ).fetchone()

    # Short/long preserved.
    assert fs is not None
    assert fs[0] == "short"
    assert fs[1] == "long"
    # Vision columns cleared.
    assert fs[2] is None
    assert fs[3] is None
    assert fs[4] is None
    # Vision embedding removed.
    assert vision_emb is None


def test_purge_vision_leaves_other_drives_untouched(tmp_path, monkeypatch):
    """A purge scoped to drive=work must not touch drive=family's data."""
    try:
        from app.purge import purge_vision_for_drive as _purge_vision
    except ImportError:
        try:
            from app.purge import purge_feature_for_drive

            def _purge_vision(drive: str) -> int:  # type: ignore[misc]
                return purge_feature_for_drive(drive, "vision_describe")
        except ImportError:
            pytest.skip("vision purge helper not implemented (RED phase)")

    engine = _build_engine(tmp_path)
    Session = _seed_image_file(engine, file_id="img-work", drive="work")
    _seed_image_file(engine, file_id="img-family", drive="family")
    _patch_get_search_db(monkeypatch, Session)
    monkeypatch.setattr("app.search.invalidate_similar_cache", lambda: 0)

    _purge_vision("work")

    # work side wiped.
    with engine.connect() as conn:
        work_row = conn.execute(
            text(
                "SELECT visual_description FROM file_summaries "
                "WHERE file_id = 'img-work'"
            )
        ).fetchone()
        family_row = conn.execute(
            text(
                "SELECT visual_description FROM file_summaries "
                "WHERE file_id = 'img-family'"
            )
        ).fetchone()
        family_emb = conn.execute(
            text(
                "SELECT 1 FROM embeddings "
                "WHERE file_id = 'img-family' "
                "AND embedding_type = 'vision_description'"
            )
        ).fetchone()

    assert work_row is not None and work_row[0] is None
    assert family_row is not None and family_row[0] == "A red apple."
    assert family_emb is not None


@pytest.mark.asyncio
async def test_purge_disabled_drives_triggers_vision_purge(
    tmp_path, monkeypatch,
):
    """Drives whose vision_describe policy is off get vision-purged.

    Mirrors ``test_purge.test_purge_disabled_drives_only_purges_off`` but
    exercises the vision feature code path.
    """
    try:
        from app import policy_client
        from app.purge import purge_disabled_vision_drives  # noqa: F401
    except ImportError:
        pytest.skip(
            "purge_disabled_vision_drives not implemented (RED phase)"
        )

    from app import purge
    from types import SimpleNamespace

    # Patch the drive enumerator with in-memory data.
    drives = {"work": ["img-work"], "family": ["img-family"]}

    class _Query:
        def __init__(self_inner, payload):
            self_inner._payload = payload
            self_inner._drive_filter = None
            self_inner._distinct = False

        def filter(self_inner, *args, **kwargs):
            for arg in args:
                rhs = getattr(arg, "right", None)
                value = getattr(rhs, "value", None) if rhs is not None else None
                if isinstance(value, str):
                    self_inner._drive_filter = value
            return self_inner

        def distinct(self_inner):
            self_inner._distinct = True
            return self_inner

        def all(self_inner):
            if self_inner._distinct:
                return [SimpleNamespace(drive=d) for d in self_inner._payload]
            ids = self_inner._payload.get(self_inner._drive_filter, [])
            return [SimpleNamespace(file_id=fid) for fid in ids]

    class _Session:
        def query(self_inner, _col):
            return _Query(drives)

    @contextmanager
    def _get_search_db():
        yield _Session()

    monkeypatch.setattr("app.database.get_search_db", _get_search_db)
    monkeypatch.setattr(purge, "get_search_db", _get_search_db, raising=False)

    async def fake_policy(drive, feature):
        # work OFF for vision_describe; family ON.
        if feature == "vision_describe":
            return drive != "work"
        return True

    policy_client.reset_cache()
    monkeypatch.setattr(policy_client, "is_feature_enabled", fake_policy)

    purged_drives: list[str] = []
    monkeypatch.setattr(
        "app.purge.purge_vision_for_drive",
        lambda drive: purged_drives.append(drive) or 1,
        raising=False,
    )

    from app.purge import purge_disabled_vision_drives

    result = await purge_disabled_vision_drives()

    assert result == {"work": 1}
    assert purged_drives == ["work"]
