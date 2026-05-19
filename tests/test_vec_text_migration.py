"""Migration tests for ``_migrate_vec_text_if_needed`` (Phase 2).

RED-phase tests for spec ``2026-05-20-gui-text-embedding-model.md``
invariants §2.1-1 / §2.1-2 / §2.1-5.

The rebuild trigger keys on the **recorded model NAME**
(``index_meta['text_embedding_model']``) versus the configured
``settings.models.text_embedding`` — NOT on the vector dimension.
Dimension-only keying (the latent bug in the existing
``_migrate_vec_clip_if_needed``) would wrongly skip a same-dimension
model swap (e5-base 768 ↔ ruri-130m 768), leaving a vector space
that is non-comparable under cosine distance. Test (b) is the
regression-critical case that locks this in.

Branches under test (plan Phase 2):
  (a) recorded != configured, different dim   -> DROP + purge + reindex
  (b) recorded != configured, SAME dim        -> DROP + purge + reindex
  (c) recorded absent, vec_text absent        -> record only, no purge
  (d) recorded absent, vec_text present, dim
      matches configured (upgrade migration)  -> seed index_meta only,
                                                 NO purge
  (e) recorded == configured                  -> no-op

The bare test env has no sqlite-vec extension, so ``vec_text`` is
created as a plain table whose ``CREATE`` SQL still carries the
``float[N]`` signature the dimension probe reads from
``sqlite_master.sql`` (mirrors how the production virtual table
records its schema).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# Stub heavy ML deps before importing app.database (mirrors the other
# intelligence migration tests; conftest also does this but be explicit
# so the file is runnable in isolation).
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

from app.config import ModelConfig  # noqa: E402


# Model ids drawn from app.workers.embedder._MODEL_DIMS.
_E5_SMALL = "intfloat/multilingual-e5-small"      # 384
_E5_BASE = "intfloat/multilingual-e5-base"        # 768
_E5_LARGE = "intfloat/multilingual-e5-large"      # 1024
_RURI_130 = "cl-nagoya/ruri-v3-130m"              # 768  (same dim as e5-base)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_db(tmp_path, *, vec_text_dim: int | None, recorded_model: str | None):
    """Create a search.db skeleton for the migration under test.

    * ``vec_text_dim`` None -> do NOT create vec_text (fresh path).
    * ``recorded_model`` None -> do NOT seed index_meta (recorded absent).
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE indexed_files ("
            "  file_id TEXT PRIMARY KEY,"
            "  drive TEXT NOT NULL,"
            "  text_indexed BOOLEAN NOT NULL DEFAULT 0"
            ")"
        ))
        conn.execute(text(
            "CREATE TABLE embeddings ("
            "  embedding_id TEXT PRIMARY KEY,"
            "  file_id TEXT NOT NULL,"
            "  embedding_type TEXT NOT NULL"
            ")"
        ))
        conn.execute(text(
            "CREATE TABLE detailed_summary_citations ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  file_id TEXT NOT NULL,"
            "  section_path TEXT NOT NULL"
            ")"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE fts_text_content "
            "USING fts5(file_id, chunk_index, page, text, "
            "tokenize='trigram')"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE fts_text_content_word "
            "USING fts5(file_id, chunk_index, page, text, "
            "tokenize=\"unicode61 remove_diacritics 2\")"
        ))
        conn.execute(text(
            "CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT)"
        ))
        if vec_text_dim is not None:
            # Plain table carrying the float[N] signature so the
            # dimension probe (sqlite_master.sql regex) still works
            # without the sqlite-vec extension loaded.
            conn.execute(text(
                f"CREATE TABLE vec_text "
                f"(embedding_id TEXT PRIMARY KEY, vector float[{vec_text_dim}])"
            ))
        if recorded_model is not None:
            conn.execute(
                text(
                    "INSERT INTO index_meta(key, value) "
                    "VALUES('text_embedding_model', :m)"
                ),
                {"m": recorded_model},
            )
    return engine


def _seed_indexed_text(engine) -> None:
    """Insert one text-indexed file with text_content embeddings,
    FTS rows, and a citation row so a purge is observable."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO indexed_files(file_id, drive, text_indexed) "
                "VALUES('f1', 'default', 1)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO embeddings(embedding_id, file_id, embedding_type) "
                "VALUES('txt_f1_0', 'f1', 'text_content')"
            )
        )
        # A non-text embedding that must SURVIVE the purge.
        conn.execute(
            text(
                "INSERT INTO embeddings(embedding_id, file_id, embedding_type) "
                "VALUES('clip_f1_0', 'f1', 'clip')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO fts_text_content(file_id, chunk_index, page, text) "
                "VALUES('f1', '0', '1', 'hello world')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO fts_text_content_word"
                "(file_id, chunk_index, page, text) "
                "VALUES('f1', '0', '1', 'hello world')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO detailed_summary_citations(file_id, section_path) "
                "VALUES('f1', 'sec/0')"
            )
        )


def _patch_configured_model(monkeypatch, model_id: str) -> None:
    """Point ``app.config.settings.models.text_embedding`` at ``model_id``.

    ``_get_text_embedding_dim()`` and the migration both read
    ``app.config.settings`` lazily, so rebinding the module attribute
    is sufficient.
    """
    import app.config as cfg

    new_models = ModelConfig(text_embedding=model_id)
    new_settings = cfg.Settings(
        intelligence_data_dir=cfg.settings.intelligence_data_dir,
        litloft_db_path=cfg.settings.litloft_db_path,
        model_cache_dir=cfg.settings.model_cache_dir,
        search_db_path=cfg.settings.search_db_path,
        allowed_base_dirs=cfg.settings.allowed_base_dirs,
        drive_mounts=cfg.settings.drive_mounts,
        models=new_models,
        search=cfg.settings.search,
        indexing=cfg.settings.indexing,
        workers=cfg.settings.workers,
        memory=cfg.settings.memory,
        features=cfg.settings.features,
        llm=cfg.settings.llm,
    )
    monkeypatch.setattr(cfg, "settings", new_settings)


def _read_meta(engine) -> str | None:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT value FROM index_meta "
                "WHERE key='text_embedding_model'"
            )
        ).fetchone()
    return row[0] if row else None


def _vec_text_exists(engine) -> bool:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='vec_text'"
            )
        ).fetchone()
    return row is not None


def _scalar(engine, sql: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(sql)).fetchone()[0]


def _run_migration(engine) -> None:
    from app.database import _migrate_vec_text_if_needed

    with engine.connect() as conn:
        _migrate_vec_text_if_needed(conn)
        conn.commit()


# ---------------------------------------------------------------------------
# (a) recorded != configured, DIFFERENT dim -> drop + purge + reindex
# ---------------------------------------------------------------------------


def test_a_model_name_change_diff_dim_drops_and_purges(
    tmp_path, monkeypatch
) -> None:
    # recorded e5-small (384), configured e5-large (1024).
    engine = _build_db(tmp_path, vec_text_dim=384, recorded_model=_E5_SMALL)
    _seed_indexed_text(engine)
    _patch_configured_model(monkeypatch, _E5_LARGE)

    _run_migration(engine)

    assert not _vec_text_exists(engine), "vec_text must be dropped"
    assert _scalar(
        engine,
        "SELECT COUNT(*) FROM embeddings WHERE embedding_type='text_content'",
    ) == 0
    # Non-text embeddings untouched.
    assert _scalar(
        engine, "SELECT COUNT(*) FROM embeddings WHERE embedding_type='clip'"
    ) == 1
    assert _scalar(
        engine, "SELECT COUNT(*) FROM indexed_files WHERE text_indexed=1"
    ) == 0
    assert _scalar(engine, "SELECT COUNT(*) FROM fts_text_content") == 0
    assert _scalar(engine, "SELECT COUNT(*) FROM fts_text_content_word") == 0
    assert _scalar(
        engine, "SELECT COUNT(*) FROM detailed_summary_citations"
    ) == 0
    assert _read_meta(engine) == _E5_LARGE


# ---------------------------------------------------------------------------
# (b) recorded != configured, SAME dim -> STILL fires (regression-critical)
# ---------------------------------------------------------------------------


def test_b_same_dimension_different_model_still_fires(
    tmp_path, monkeypatch
) -> None:
    """e5-base (768) -> ruri-130m (768): identical dimension but a
    different vector space. Dimension-only keying would WRONGLY skip
    this; model-name keying must rebuild."""
    engine = _build_db(tmp_path, vec_text_dim=768, recorded_model=_E5_BASE)
    _seed_indexed_text(engine)
    _patch_configured_model(monkeypatch, _RURI_130)

    _run_migration(engine)

    assert not _vec_text_exists(engine), (
        "same-dim model swap MUST rebuild vec_text "
        "(dimension-only keying is the latent vec_clip bug)"
    )
    assert _scalar(
        engine,
        "SELECT COUNT(*) FROM embeddings WHERE embedding_type='text_content'",
    ) == 0
    assert _scalar(
        engine, "SELECT COUNT(*) FROM indexed_files WHERE text_indexed=1"
    ) == 0
    assert _scalar(engine, "SELECT COUNT(*) FROM fts_text_content") == 0
    assert _scalar(engine, "SELECT COUNT(*) FROM fts_text_content_word") == 0
    assert _scalar(
        engine, "SELECT COUNT(*) FROM detailed_summary_citations"
    ) == 0
    assert _read_meta(engine) == _RURI_130


# ---------------------------------------------------------------------------
# (c) fresh DB, vec_text absent -> no purge, record current model
# ---------------------------------------------------------------------------


def test_c_fresh_db_records_model_no_purge(tmp_path, monkeypatch) -> None:
    engine = _build_db(tmp_path, vec_text_dim=None, recorded_model=None)
    # A pre-existing text-indexed row (e.g. created later by the indexer)
    # — nothing exists yet at migration time, but assert no destructive
    # action regardless.
    _patch_configured_model(monkeypatch, _E5_SMALL)

    _run_migration(engine)

    assert _read_meta(engine) == _E5_SMALL
    # vec_text was absent and stays absent (created later by
    # _create_vec_tables, out of scope here).
    assert not _vec_text_exists(engine)


# ---------------------------------------------------------------------------
# (d) upgrade path: recorded absent, vec_text present, dim matches
#     configured -> seed index_meta only, NO purge
# ---------------------------------------------------------------------------


def test_d_upgrade_path_seeds_meta_without_purge(
    tmp_path, monkeypatch
) -> None:
    """Existing user upgrading into this feature: index_meta is absent
    but vec_text already exists at the configured model's dimension.
    Treat it as "already built at baseline" — seed the record only.
    Must NOT re-embed everything (invariant §2.1-5)."""
    engine = _build_db(tmp_path, vec_text_dim=768, recorded_model=None)
    _seed_indexed_text(engine)
    # Configured model is e5-base (768) == vec_text's existing dim.
    _patch_configured_model(monkeypatch, _E5_BASE)

    _run_migration(engine)

    assert _read_meta(engine) == _E5_BASE
    assert _vec_text_exists(engine), "vec_text must NOT be dropped"
    assert _scalar(
        engine,
        "SELECT COUNT(*) FROM embeddings WHERE embedding_type='text_content'",
    ) == 1, "text_content embeddings must be preserved"
    assert _scalar(
        engine, "SELECT COUNT(*) FROM indexed_files WHERE text_indexed=1"
    ) == 1, "text_indexed must NOT be reset"
    assert _scalar(engine, "SELECT COUNT(*) FROM fts_text_content") == 1
    assert _scalar(
        engine, "SELECT COUNT(*) FROM detailed_summary_citations"
    ) == 1


# ---------------------------------------------------------------------------
# (e) recorded == configured -> no-op
# ---------------------------------------------------------------------------


def test_e_recorded_equals_configured_is_noop(
    tmp_path, monkeypatch
) -> None:
    engine = _build_db(tmp_path, vec_text_dim=384, recorded_model=_E5_SMALL)
    _seed_indexed_text(engine)
    _patch_configured_model(monkeypatch, _E5_SMALL)

    _run_migration(engine)

    assert _vec_text_exists(engine), "no-op must not drop vec_text"
    assert _scalar(
        engine,
        "SELECT COUNT(*) FROM embeddings WHERE embedding_type='text_content'",
    ) == 1
    assert _scalar(
        engine, "SELECT COUNT(*) FROM indexed_files WHERE text_indexed=1"
    ) == 1
    assert _scalar(engine, "SELECT COUNT(*) FROM fts_text_content") == 1
    assert _scalar(
        engine, "SELECT COUNT(*) FROM detailed_summary_citations"
    ) == 1
    assert _read_meta(engine) == _E5_SMALL


# ---------------------------------------------------------------------------
# index_meta table DDL is created next to _create_vec_tables
# ---------------------------------------------------------------------------


def test_index_meta_table_created_by_init_path(tmp_path, monkeypatch) -> None:
    """`index_meta(key TEXT PRIMARY KEY, value TEXT)` must be created
    alongside the vec tables so `_migrate_vec_text_if_needed` can read
    it on a brand-new DB."""
    from app.database import _create_vec_tables

    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    sqlite_vec = sys.modules.get("sqlite_vec")
    if isinstance(sqlite_vec, MagicMock):
        pytest.skip(
            "sqlite-vec extension unavailable in bare env; "
            "covered by the Docker test image"
        )
    with engine.begin() as conn:
        _create_vec_tables(conn)

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='index_meta'"
            )
        ).fetchone()
        cols = {
            r[1]
            for r in conn.execute(
                text("PRAGMA table_info(index_meta)")
            ).fetchall()
        }
    assert row is not None, "index_meta table must exist"
    assert cols == {"key", "value"}
