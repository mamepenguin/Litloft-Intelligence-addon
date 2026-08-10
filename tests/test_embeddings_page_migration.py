from sqlalchemy import create_engine, text


def columns(engine) -> set[str]:
    with engine.connect() as conn:
        return {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(embeddings)")).fetchall()
        }


def test_embeddings_page_migration_preserves_rows_and_is_idempotent(tmp_path):
    from app.database import _migrate_embeddings_page_column

    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE indexed_files ("
            "file_id TEXT PRIMARY KEY, mime_type TEXT NOT NULL)"
        ))
        conn.execute(text(
            "INSERT INTO indexed_files VALUES "
            "('f1', 'text/plain'), ('f2', 'application/pdf')"
        ))
        conn.execute(text(
            "CREATE TABLE embeddings ("
            "id TEXT PRIMARY KEY, file_id TEXT NOT NULL, "
            "embedding_type TEXT NOT NULL, content_preview TEXT NOT NULL, "
            "vector_table TEXT NOT NULL)"
        ))
        conn.execute(text(
            "INSERT INTO embeddings VALUES "
            "('e1', 'f1', 'text_content', 'source excerpt', 'vec_text')"
        ))
        conn.execute(text(
            "INSERT INTO embeddings VALUES "
            "('e2', 'f2', 'text_content', 'PDF excerpt (page 7)', 'vec_text')"
        ))
        _migrate_embeddings_page_column(conn)
        _migrate_embeddings_page_column(conn)

    assert "page" in columns(engine)
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT content_preview, page FROM embeddings WHERE id = 'e1'"
        )).one()
    assert tuple(row) == ("source excerpt", None)
    with engine.connect() as conn:
        migrated = conn.execute(text(
            "SELECT content_preview, page FROM embeddings WHERE id = 'e2'"
        )).one()
    assert tuple(migrated) == ("PDF excerpt", 7)
