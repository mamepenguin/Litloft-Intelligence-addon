"""A search DB built by ``init_search_db`` itself, with sqlite-vec loaded, for the
SPEC-ADDON-022 / SPEC-ADDON-023 tests.

The test image builds the same ``vec0`` extension as the service image, and
``init_search_db`` loads it, so these tests run real partitioned ``vec0`` tables and
real ``MATCH`` scans: a plain-table stand-in has no partition semantics.
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import event, text
from sqlalchemy.engine import Engine

import app.database as database
import app.models  # noqa: F401  (registers the ORM tables init_search_db creates)
from app.models import Embedding, IndexedFile

VEC_TABLES = ("vec_clip", "vec_text")
MIGRATION_LOG = "vec partition migration"

_DIM = re.compile(r"float\[(\d+)\]")


def unit(dim: int, axis: int = 0) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[axis % dim] = 1.0
    return v


def at_distance(dim: int, distance: float, axis: int) -> np.ndarray:
    """A unit vector whose L2 distance from ``unit(dim, 0)`` is ``distance``.

    ``axis`` picks the orthogonal direction, so two vectors at the same distance
    from the source can still differ from each other.
    """
    cos = 1.0 - distance * distance / 2.0
    sin = float(np.sqrt(max(0.0, 1.0 - cos * cos)))
    v = np.zeros(dim, dtype=np.float32)
    v[0] = cos
    v[1 + (axis % (dim - 1))] = sin
    return v


def random_vector(dim: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class SearchIndex:
    def __init__(self, path: Path):
        self.path = path

    @property
    def engine(self) -> Engine:
        return database._search_engine

    # -- schema ---------------------------------------------------------------

    def create_sql(self, table: str) -> str | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT sql FROM sqlite_master WHERE name = :n"), {"n": table},
            ).fetchone()
        return None if row is None else row[0]

    def is_partitioned(self, table: str) -> bool:
        sql = self.create_sql(table) or ""
        return re.search(
            r"embedding_type\s+text\s+partition\s+key", sql, re.IGNORECASE,
        ) is not None

    def dim(self, table: str) -> int:
        match = _DIM.search(self.create_sql(table) or "")
        assert match, f"{table} has no float[N] column"
        return int(match.group(1))

    def table_names(self) -> set[str]:
        with self.engine.connect() as conn:
            return {
                r[0] for r in conn.execute(text("SELECT name FROM sqlite_master"))
            }

    def downgrade(self, table: str) -> None:
        """Replace ``table`` with the pre-change, unpartitioned vec0 table."""
        dim = self.dim(table)
        with self.engine.begin() as conn:
            conn.execute(text(f"DROP TABLE {table}"))
            conn.execute(text(
                f"CREATE VIRTUAL TABLE {table} USING "
                f"vec0(embedding_id TEXT PRIMARY KEY, vector float[{dim}])"
            ))

    # -- rows -----------------------------------------------------------------

    def add_file(
        self, file_id: str, *, drive: str = "home", active: bool = True,
        mime_type: str = "video/mp4", file_path: str | None = None,
        duration: float | None = None, thumbnail_path: str | None = None,
    ) -> None:
        file_type = mime_type.split("/")[0]
        if file_type not in ("video", "image", "audio"):
            file_type = "document"
        with database.get_search_db() as s:
            s.add(IndexedFile(
                file_id=file_id, drive=drive,
                filename=os.path.basename(file_path or f"{file_id}.bin"),
                file_path=file_path or f"/drives/{drive}/{file_id}.bin",
                file_type=file_type, mime_type=mime_type, file_size=1,
                duration=duration, thumbnail_path=thumbnail_path, active=active,
            ))

    def add_embedding(self, emb_id: str, file_id: str, embedding_type: str, table: str) -> None:
        with database.get_search_db() as s:
            s.add(Embedding(
                id=emb_id, file_id=file_id, embedding_type=embedding_type,
                vector_table=table, content_preview="x",
            ))

    def put_vector(
        self, table: str, emb_id: str, vector: np.ndarray, partition: str | None = "omit",
    ) -> None:
        """Insert a vec row. ``partition="omit"`` leaves the partition column out,
        as code that predates the column does."""
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        with self.engine.begin() as conn:
            if partition == "omit":
                conn.execute(
                    text(f"INSERT INTO {table}(embedding_id, vector) VALUES (:i, :v)"),
                    {"i": emb_id, "v": blob},
                )
            else:
                conn.execute(
                    text(
                        f"INSERT INTO {table}(embedding_id, embedding_type, vector) "
                        "VALUES (:i, :t, :v)"
                    ),
                    {"i": emb_id, "t": partition, "v": blob},
                )

    def add_many(self, rows) -> None:
        """Bulk ``(file_id, drive, active, emb_id, embedding_type, table, vector)``
        rows, one file per distinct file_id, every vec row keyed by its type."""
        files = {}
        for file_id, drive, active, *_ in rows:
            files.setdefault(file_id, (drive, active))
        with database.get_search_db() as s:
            s.add_all([
                IndexedFile(
                    file_id=f, drive=d, filename=f"{f}.mp4", file_path=f"/drives/{d}/{f}.mp4",
                    file_type="video", mime_type="video/mp4", file_size=1, active=a,
                )
                for f, (d, a) in files.items()
            ])
            s.add_all([
                Embedding(
                    id=e, file_id=f, embedding_type=t, vector_table=tb, content_preview="x",
                )
                for f, _, _, e, t, tb, _ in rows
            ])
        with self.engine.begin() as conn:
            for table in {r[5] for r in rows}:
                conn.execute(
                    text(
                        f"INSERT INTO {table}(embedding_id, embedding_type, vector) "
                        "VALUES (:i, :t, :v)"
                    ),
                    [
                        {"i": e, "t": t, "v": np.asarray(v, dtype=np.float32).tobytes()}
                        for _, _, _, e, t, tb, v in rows if tb == table
                    ],
                )

    def vectors(self, table: str) -> dict[str, bytes]:
        with self.engine.connect() as conn:
            return {
                r[0]: bytes(r[1]) for r in conn.execute(
                    text(f"SELECT embedding_id, vector FROM {table}")
                )
            }

    def keyed(self, table: str) -> dict[str, tuple[str | None, bytes]]:
        """embedding_id -> (partition value, vector bytes)."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT embedding_id, embedding_type, vector FROM {table}")
            ).fetchall()
        assert len(rows) == len({r[0] for r in rows}), f"{table} holds a duplicate id"
        return {r[0]: (r[1], bytes(r[2])) for r in rows}

    def knn(self, table: str, vector: np.ndarray, k: int, embedding_type: str) -> list[str]:
        with self.engine.connect() as conn:
            return [
                r[0] for r in conn.execute(
                    text(
                        f"SELECT embedding_id FROM {table} WHERE vector MATCH :v "
                        "AND k = :k AND embedding_type = :t ORDER BY distance"
                    ),
                    {"v": np.asarray(vector, dtype=np.float32).tobytes(), "k": k,
                     "t": embedding_type},
                )
            ]

    def embeddings(self, file_id: str | None = None) -> list[tuple[str, str, str]]:
        """(id, embedding_type, vector_table) rows."""
        sql = "SELECT id, embedding_type, vector_table FROM embeddings"
        params = {}
        if file_id is not None:
            sql += " WHERE file_id = :f"
            params["f"] = file_id
        with self.engine.connect() as conn:
            return [tuple(r) for r in conn.execute(text(sql + " ORDER BY id"), params)]

    def sql(self, statement: str, **params) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(statement), params)

    def wal_bytes(self) -> int:
        wal = Path(f"{self.path}-wal")
        return wal.stat().st_size if wal.exists() else 0

    def assert_partition_matches_type(self, file_id: str) -> list[str]:
        """Every embedding of ``file_id`` has exactly one vec row, keyed by its type.

        Returns the embedding types seen, so a caller can require the one it wrote.
        """
        rows = self.embeddings(file_id)
        assert rows, f"no embedding was written for {file_id}"
        by_table = {t: self.keyed(t) for t in VEC_TABLES}
        for emb_id, etype, table in rows:
            assert emb_id in by_table[table], f"{emb_id} has no {table} row"
            assert by_table[table][emb_id][0] == etype, (
                f"{table} row {emb_id} is keyed {by_table[table][emb_id][0]!r}, "
                f"its embedding is {etype!r}"
            )
        return sorted({etype for _, etype, _ in rows})


def start() -> None:
    """One start of the addon's search DB: ``init_search_db`` as the lifespan runs it."""
    database.init_search_db()


def run_partition_migration() -> None:
    database._migrate_vec_partition_if_needed()


@pytest.fixture()
def search_index(tmp_path, monkeypatch):
    """A fresh search DB created by ``init_search_db``, with the module's own
    engine and sessions pointed at it, and the module restored afterwards."""
    saved = dict(vars(database))
    path = tmp_path / "search.db"
    monkeypatch.setattr(database, "_resolve_search_db_path", lambda: path)
    index = SearchIndex(path)
    start()
    try:
        yield index
    finally:
        engine = vars(database).get("_search_engine")
        if engine is not None:
            engine.dispose()
        for key in list(vars(database)):
            if key not in saved:
                delattr(database, key)
        for key, value in saved.items():
            if vars(database).get(key) is not value:
                setattr(database, key, value)


class Authorizer:
    """Installs a SQLite authorizer on every new connection of every engine.

    ``deny_create_vtable`` names a table whose ``CREATE VIRTUAL TABLE`` fails; with
    ``exit_code`` set the process dies there instead, as a killed container does.
    """

    def __init__(self):
        self.deny_create_vtable: str | None = None
        self.exit_code: int | None = None

    def _authorize(self, action, arg1, arg2, dbname, source):
        if (
            action == sqlite3.SQLITE_CREATE_VTABLE
            and self.deny_create_vtable is not None
            and arg1 == self.deny_create_vtable
        ):
            if self.exit_code is not None:
                os._exit(self.exit_code)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def _on_connect(self, dbapi_conn, _record):
        dbapi_conn.set_authorizer(self._authorize)

    def install(self) -> None:
        event.listen(Engine, "connect", self._on_connect)

    def remove(self) -> None:
        event.remove(Engine, "connect", self._on_connect)


@pytest.fixture()
def authorizer(search_index):
    auth = Authorizer()
    auth.install()
    search_index.engine.dispose()
    try:
        yield auth
    finally:
        auth.remove()
        search_index.engine.dispose()
