"""Vector tables partitioned by ``embedding_type``.

``vec_text`` and ``vec_clip`` carry the embedding type as a sqlite-vec partition
key, so a ``MATCH`` constrained to one type scans only that type's rows. These
helpers are the one place that knows the schema: the DDL, how a row is written
with its partition value, the one-time migration of an unpartitioned table, and
the repair of rows stored without a partition value.
"""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import time
from pathlib import Path

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection, Engine

logger = logging.getLogger(__name__)

VEC_TABLES = ("vec_clip", "vec_text")

_PARTITION_RE = re.compile(r"embedding_type\s+TEXT\s+PARTITION\s+KEY", re.IGNORECASE)
_DIM_RE = re.compile(r"float\[(\d+)\]")

#: Free space the migration needs, as a multiple of the vector bytes it copies.
#: Measured on 200,000 768-dim rows: the WAL peaked at 2.3x the vectors before
#: commit, and the checkpoint then grew the file by 1.3x while the WAL still held
#: its pages.
_FREE_SPACE_FACTOR = 4

_REPAIR_BATCH = 500


def vec_table_ddl(table: str, dim: int, *, if_not_exists: bool = False) -> str:
    guard = "IF NOT EXISTS " if if_not_exists else ""
    return (
        f"CREATE VIRTUAL TABLE {guard}{table} USING vec0("
        f"embedding_id TEXT PRIMARY KEY, "
        f"embedding_type TEXT PARTITION KEY, "
        f"vector float[{dim}])"
    )


def insert_vector_sql(table: str) -> str:
    """The INSERT every writer uses, binding ``:id``, ``:type`` and ``:vec``."""
    return (
        f"INSERT INTO {table}(embedding_id, embedding_type, vector) "
        f"VALUES(:id, :type, :vec)"
    )


def is_partitioned(conn: Connection, table: str) -> bool:
    row = conn.execute(
        text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = :n"),
        {"n": table},
    ).fetchone()
    return row is not None and _PARTITION_RE.search(row[0] or "") is not None


def rekey_vector(conn: Connection, table: str, embedding_id: str, embedding_type: str) -> bool:
    """Move one vec row to ``embedding_type``'s partition, keeping its bytes.

    sqlite-vec rejects an UPDATE of a partition key column, so the row is
    deleted and inserted again. Returns False when the row does not exist.
    """
    row = conn.execute(
        text(f"SELECT vector FROM {table} WHERE embedding_id = :id"),
        {"id": embedding_id},
    ).fetchone()
    if row is None:
        return False
    conn.execute(text(f"DELETE FROM {table} WHERE embedding_id = :id"), {"id": embedding_id})
    conn.execute(
        text(insert_vector_sql(table)),
        {"id": embedding_id, "type": embedding_type, "vec": row[0]},
    )
    return True


# ---------------------------------------------------------------------------
# One-time migration of unpartitioned tables
# ---------------------------------------------------------------------------


def migrate_if_needed(engine: Engine, db_path: str) -> None:
    """Rebuild every unpartitioned vec table as a partitioned one, atomically.

    Runs on a raw connection in autocommit mode with an explicit BEGIN, because
    pysqlite commits DDL on its own when it manages the transaction, and a DROP
    committed alone would lose the table's vectors.
    """
    raw = engine.raw_connection()
    try:
        dbapi: sqlite3.Connection = raw.driver_connection
        previous = dbapi.isolation_level
        dbapi.isolation_level = None
        try:
            _migrate(dbapi, db_path)
        finally:
            dbapi.isolation_level = previous
    finally:
        raw.close()


def _pending_tables(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    pending = []
    for table in VEC_TABLES:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,),
        ).fetchone()
        if row is None or _PARTITION_RE.search(row[0] or ""):
            continue
        match = _DIM_RE.search(row[0] or "")
        if match is None:
            raise RuntimeError(f"vec partition migration: no float[N] column in {table}")
        pending.append((table, int(match.group(1))))
    return pending


def _migrate(conn: sqlite3.Connection, db_path: str) -> None:
    pending = _pending_tables(conn)
    if not pending:
        return

    _checkpoint(conn)
    rows = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t, _ in pending}
    estimated = sum(rows[t] * dim * 4 for t, dim in pending)
    required = _FREE_SPACE_FACTOR * estimated
    free = shutil.disk_usage(Path(db_path).parent).free
    if free < required:
        logger.error(
            "vec partition migration: not enough free disk: free=%d required=%d",
            free, required,
        )
        raise RuntimeError("Not enough free disk to partition the vector tables")

    logger.info(
        "vec partition migration: start tables=%s estimated_bytes=%d",
        ",".join(t for t, _ in pending), estimated,
    )
    started = time.monotonic()
    copied: dict[str, int] = {}
    conn.execute("BEGIN IMMEDIATE")
    try:
        for table, dim in pending:
            copied[table] = _rebuild(conn, table, dim)
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        _checkpoint(conn)
        raise
    _checkpoint(conn)

    logger.info(
        "vec partition migration: done %s elapsed=%.1fs",
        " ".join(
            f"{t} copied={copied[t]} orphans_dropped={rows[t] - copied[t]}"
            for t, _ in pending
        ),
        time.monotonic() - started,
    )


def _rebuild(conn: sqlite3.Connection, table: str, dim: int) -> int:
    # ALTER TABLE RENAME on a vec0 table leaves its shadow tables under the old
    # name, so the new table cannot be created beside it: copy through a plain
    # table instead.
    staging = f"{table}_migrate"
    conn.execute(
        f"CREATE TABLE {staging} (embedding_id TEXT PRIMARY KEY, "
        f"embedding_type TEXT NOT NULL, vector BLOB NOT NULL)"
    )
    conn.execute(
        f"INSERT INTO {staging} SELECT v.embedding_id, e.embedding_type, v.vector "
        f"FROM {table} v JOIN embeddings e ON e.id = v.embedding_id"
    )
    count = conn.execute(f"SELECT COUNT(*) FROM {staging}").fetchone()[0]
    conn.execute(f"DROP TABLE {table}")
    conn.execute(vec_table_ddl(table, dim))
    conn.execute(
        f"INSERT INTO {table}(embedding_id, embedding_type, vector) "
        f"SELECT embedding_id, embedding_type, vector FROM {staging}"
    )
    conn.execute(f"DROP TABLE {staging}")
    return count


def _checkpoint(conn: sqlite3.Connection) -> None:
    result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if result is not None and result[0]:
        logger.warning(
            "vec partition migration: WAL checkpoint was busy (result=%s); "
            "a later checkpoint truncates it",
            tuple(result),
        )


# ---------------------------------------------------------------------------
# Repair of rows stored without a partition value
# ---------------------------------------------------------------------------


def repair_unkeyed_rows(engine: Engine) -> dict[str, int]:
    """Give every NULL-partition vec row its embedding's type, or drop it.

    Addon code older than the partitioned schema inserts without the partition
    column, and sqlite-vec stores NULL there, which no type-constrained KNN
    returns. Each row's delete and re-insert commit together.
    """
    rekeyed = dropped = 0
    for table in VEC_TABLES:
        with engine.connect() as conn:
            if not is_partitioned(conn, table):
                continue
            ids = [
                r[0] for r in conn.execute(
                    text(f"SELECT embedding_id FROM {table} WHERE embedding_type IS NULL")
                )
            ]
        for start in range(0, len(ids), _REPAIR_BATCH):
            batch = ids[start:start + _REPAIR_BATCH]
            with engine.begin() as conn:
                types = dict(conn.execute(
                    text("SELECT id, embedding_type FROM embeddings WHERE id IN :ids")
                    .bindparams(bindparam("ids", expanding=True)),
                    {"ids": batch},
                ).fetchall())
                for embedding_id in batch:
                    embedding_type = types.get(embedding_id)
                    if embedding_type is None:
                        conn.execute(
                            text(f"DELETE FROM {table} WHERE embedding_id = :id"),
                            {"id": embedding_id},
                        )
                        dropped += 1
                    elif rekey_vector(conn, table, embedding_id, embedding_type):
                        rekeyed += 1

    if rekeyed or dropped:
        logger.info(
            "vec partition repair: rekeyed=%d dropped=%d", rekeyed, dropped,
        )
    return {"rekeyed": rekeyed, "dropped": dropped}
