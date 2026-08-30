"""``embeddings.chunk_index`` migration.

A document chunk's full text lives in ``fts_text_content``, keyed by
``(file_id, chunk_index)``. The embedding row had no chunk index to join
on, so the column is added and the existing ``text_content`` rows are
purged to force a re-extract + re-embed.

The purge is deliberately narrow. ``whisper`` embeddings already carry
``timestamp_start`` / ``timestamp_end``, which is a natural key into
``transcript_chunks``, so they need neither the column nor a rebuild —
and resetting ``whisper_indexed`` would re-run the transcription
provider over the whole library, not merely the embedding step.

``chunk_index`` is the key that turns an embedding row back into the
text in ``fts_text_content`` it was built from.
"""

from sqlalchemy import create_engine, text


def _columns(engine) -> set[str]:
    with engine.connect() as conn:
        return {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(embeddings)")).fetchall()
        }


def _seed(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE indexed_files ("
            "file_id TEXT PRIMARY KEY, text_indexed INTEGER NOT NULL, "
            "whisper_indexed INTEGER NOT NULL)"
        ))
        conn.execute(text(
            "INSERT INTO indexed_files VALUES ('f1', 1, 1), ('f2', 1, 1)"
        ))
        conn.execute(text(
            "CREATE TABLE embeddings ("
            "id TEXT PRIMARY KEY, file_id TEXT NOT NULL, "
            "embedding_type TEXT NOT NULL, content_preview TEXT NOT NULL, "
            "vector_table TEXT NOT NULL)"
        ))
        conn.execute(text(
            "INSERT INTO embeddings VALUES "
            "('txt_f1_0_aaaa', 'f1', 'text_content', 'a chunk', 'vec_text'),"
            "('wh_f2_0_bbbb', 'f2', 'whisper', 'a transcript chunk', 'vec_text'),"
            "('meta_f1_cccc', 'f1', 'metadata', 'doc.pdf', 'vec_text')"
        ))
        conn.execute(text(
            "CREATE TABLE vec_text ("
            "embedding_id TEXT PRIMARY KEY, vector BLOB)"
        ))
        conn.execute(text(
            "INSERT INTO vec_text VALUES "
            "('txt_f1_0_aaaa', x'00'), ('wh_f2_0_bbbb', x'00'), "
            "('meta_f1_cccc', x'00')"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE fts_text_content "
            "USING fts5(file_id, chunk_index, page, text, tokenize='trigram')"
        ))
        conn.execute(text(
            "INSERT INTO fts_text_content(file_id, chunk_index, page, text) "
            "VALUES ('f1', '0', '', 'a chunk, in full')"
        ))


def test_migration_adds_column_and_purges_only_text_content(tmp_path):
    from app.database import _migrate_embeddings_chunk_index

    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    _seed(engine)

    with engine.begin() as conn:
        _migrate_embeddings_chunk_index(conn)

    assert "chunk_index" in _columns(engine)

    with engine.connect() as conn:
        remaining = {
            row[0] for row in conn.execute(text("SELECT id FROM embeddings"))
        }
        vectors = {
            row[0] for row in conn.execute(text(
                "SELECT embedding_id FROM vec_text"
            ))
        }
        flags = conn.execute(text(
            "SELECT text_indexed, whisper_indexed FROM indexed_files "
            "ORDER BY file_id"
        )).fetchall()
        fts_rows = conn.execute(text(
            "SELECT COUNT(*) FROM fts_text_content"
        )).scalar()

    # The text_content row and its vector are gone; nothing else is.
    assert remaining == {"wh_f2_0_bbbb", "meta_f1_cccc"}
    assert vectors == {"wh_f2_0_bbbb", "meta_f1_cccc"}

    # Only the files that actually had text_content embeddings are asked
    # to run again (f1), and only the document path: resetting
    # whisper_indexed would re-transcribe, not re-embed.
    assert [tuple(row) for row in flags] == [(0, 1), (1, 1)]

    # Keyword search keeps working during the re-index window:
    # upsert_fts_text_content replaces a file's rows as it is re-indexed.
    assert fts_rows == 1


def test_migration_is_idempotent_and_does_not_purge_twice(tmp_path):
    from app.database import _migrate_embeddings_chunk_index

    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    _seed(engine)

    with engine.begin() as conn:
        _migrate_embeddings_chunk_index(conn)

    # A file re-indexed after the migration writes rows that already
    # carry the column. A second run must leave them alone.
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO embeddings "
            "(id, file_id, embedding_type, content_preview, vector_table, "
            "chunk_index) "
            "VALUES ('txt_f1_0_dddd', 'f1', 'text_content', 'a chunk', "
            "'vec_text', 0)"
        ))
        conn.execute(text(
            "UPDATE indexed_files SET text_indexed = 1 WHERE file_id = 'f1'"
        ))

    with engine.begin() as conn:
        _migrate_embeddings_chunk_index(conn)

    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT chunk_index FROM embeddings WHERE id = 'txt_f1_0_dddd'"
        )).one_or_none()
        text_indexed = conn.execute(text(
            "SELECT text_indexed FROM indexed_files WHERE file_id = 'f1'"
        )).scalar()

    assert row is not None and row[0] == 0
    assert text_indexed == 1


def test_eval_snapshot_keeps_its_vectors(tmp_path, monkeypatch):
    """A frozen snapshot gets the column but never loses rows.

    The eval harness runs ``init_search_db`` against a checked-in
    snapshot and starts no indexer, so a purge there would delete
    vectors nothing can rebuild and silently move the baseline.
    """
    from app.database import _migrate_embeddings_chunk_index

    snapshot = tmp_path / "snapshot.db"
    engine = create_engine(f"sqlite:///{snapshot}")
    _seed(engine)
    monkeypatch.setenv("INTELLIGENCE_SEARCH_DB_PATH", str(snapshot))

    with engine.begin() as conn:
        _migrate_embeddings_chunk_index(conn)

    assert "chunk_index" in _columns(engine)
    with engine.connect() as conn:
        embeddings = conn.execute(text("SELECT COUNT(*) FROM embeddings")).scalar()
        vectors = conn.execute(text("SELECT COUNT(*) FROM vec_text")).scalar()
        text_indexed = conn.execute(text(
            "SELECT text_indexed FROM indexed_files WHERE file_id = 'f1'"
        )).scalar()

    assert embeddings == 3
    assert vectors == 3
    assert text_indexed == 1
