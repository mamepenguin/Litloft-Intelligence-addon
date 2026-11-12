"""A core DB and a search DB on disk, shaped like production, for Pickup.

The core tables carry the columns Pickup reads; the vector tables are plain
tables with the same two columns sqlite-vec exposes, so every read path runs
unpatched.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.models import Base

DIM = 8
LOFT_MIME = "application/vnd.litloft.loft+json"


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def ts(value: datetime) -> str:
    return value.isoformat(sep=" ")


def unit(*components: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for i, c in enumerate(components):
        v[i] = c
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def at_angle(degrees: float, axis_a: int = 0, axis_b: int = 1) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[axis_a] = math.cos(math.radians(degrees))
    v[axis_b] = math.sin(math.radians(degrees))
    return v


_CORE_DDL = (
    "CREATE TABLE files ("
    " id VARCHAR(12) PRIMARY KEY, filename VARCHAR NOT NULL DEFAULT 'f',"
    " title VARCHAR NOT NULL DEFAULT 'f', drive VARCHAR NOT NULL,"
    " folder_path VARCHAR NOT NULL DEFAULT '', file_path VARCHAR NOT NULL,"
    " file_size INTEGER NOT NULL DEFAULT 1,"
    " file_type VARCHAR NOT NULL DEFAULT 'other',"
    " mime_type VARCHAR NOT NULL DEFAULT 'application/octet-stream',"
    " duration FLOAT, liked_at DATETIME,"
    " is_favorite BOOLEAN NOT NULL DEFAULT 0,"
    " created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL,"
    " deleted_at DATETIME, missing_since DATETIME)",
    "CREATE TABLE watch_history ("
    " viewer_id VARCHAR(16) NOT NULL, file_id VARCHAR(12) NOT NULL,"
    " playback_position FLOAT NOT NULL DEFAULT 0,"
    " duration FLOAT NOT NULL DEFAULT 0,"
    " last_played_at DATETIME NOT NULL, dismissed_at DATETIME,"
    " PRIMARY KEY (viewer_id, file_id))",
    "CREATE TABLE played_segments ("
    " viewer_id VARCHAR(16) NOT NULL, file_id VARCHAR(12) NOT NULL,"
    " segment_index INTEGER NOT NULL, played_seconds FLOAT NOT NULL,"
    " PRIMARY KEY (viewer_id, file_id, segment_index))",
)

_KIND_MIME = {
    "video": "video/mp4",
    "document": "application/pdf",
    "image": "image/jpeg",
    "audio": "audio/mpeg",
}


class CoreDB:
    """The Litloft DB as the core writes it."""

    def __init__(self, path: Path):
        self.path = path
        self.engine = create_engine(
            f"sqlite:///{path}", connect_args={"check_same_thread": False},
        )
        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.engine.begin() as conn:
            for ddl in _CORE_DDL:
                conn.execute(text(ddl))

    def add_file(
        self,
        file_id: str,
        *,
        drive: str = "a",
        kind: str = "video",
        mime: str | None = None,
        liked: bool = False,
        favourite: bool = False,
        deleted: bool = False,
        missing: bool = False,
        created_at: datetime | str | None = None,
    ) -> None:
        created = created_at if created_at is not None else utcnow() - timedelta(days=400)
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO files (id, drive, file_path, file_type, mime_type,"
                    " liked_at, is_favorite, created_at, updated_at, deleted_at,"
                    " missing_since) VALUES (:id, :drive, :path, :kind, :mime,"
                    " :liked, :fav, :created, :created, :deleted, :missing)"
                ),
                {
                    "id": file_id, "drive": drive, "path": f"/{file_id}",
                    "kind": kind, "mime": mime or _KIND_MIME.get(kind, "application/octet-stream"),
                    "liked": ts(utcnow()) if liked else None,
                    "fav": 1 if favourite else 0,
                    "created": created if isinstance(created, str) else ts(created),
                    "deleted": ts(utcnow()) if deleted else None,
                    "missing": ts(utcnow()) if missing else None,
                },
            )

    def watch(
        self,
        file_id: str,
        *,
        viewer: str = "v1",
        played_at: datetime | str,
        position: float = 100.0,
        duration: float = 100.0,
        dismissed_at: datetime | str | None = None,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO watch_history (viewer_id, file_id,"
                    " playback_position, duration, last_played_at, dismissed_at)"
                    " VALUES (:v, :f, :p, :d, :t, :dis)"
                ),
                {
                    "v": viewer, "f": file_id, "p": position, "d": duration,
                    "t": played_at if isinstance(played_at, str) else ts(played_at),
                    "dis": (
                        dismissed_at if dismissed_at is None or isinstance(dismissed_at, str)
                        else ts(dismissed_at)
                    ),
                },
            )

    def segments(self, file_id: str, seconds: list[float], *, viewer: str = "v1") -> None:
        with self.engine.begin() as conn:
            start = conn.execute(
                text(
                    "SELECT COUNT(*) FROM played_segments"
                    " WHERE viewer_id = :v AND file_id = :f"
                ),
                {"v": viewer, "f": file_id},
            ).scalar()
            for i, s in enumerate(seconds):
                conn.execute(
                    text("INSERT INTO played_segments VALUES (:v, :f, :i, :s)"),
                    {"v": viewer, "f": file_id, "i": start + i, "s": s},
                )

    def execute(self, sql: str, params: dict | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(sql), params or {})


class SearchDB:
    """The Intelligence DB: index rows, embeddings and plain vector tables."""

    def __init__(self, path: Path, *, create_models: bool = True):
        self.path = path
        self.engine = create_engine(
            f"sqlite:///{path}", connect_args={"check_same_thread": False},
        )
        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False)
        if create_models:
            Base.metadata.create_all(self.engine)
        with self.engine.begin() as conn:
            for table in ("vec_clip", "vec_text"):
                conn.execute(text(
                    f"CREATE TABLE IF NOT EXISTS {table} "
                    "(embedding_id TEXT PRIMARY KEY, vector BLOB)"
                ))

    def index(
        self,
        file_id: str,
        vectors: list[np.ndarray] | np.ndarray,
        *,
        drive: str = "a",
        channel: str = "clip_thumbnail",
        kind: str = "video",
    ) -> None:
        from app.models import Embedding, IndexedFile

        if isinstance(vectors, np.ndarray):
            vectors = [vectors]
        table = "vec_clip" if channel == "clip_thumbnail" else "vec_text"
        session = self.sessions()
        try:
            if not session.get(IndexedFile, file_id):
                session.add(IndexedFile(
                    file_id=file_id, drive=drive, filename=f"{file_id}.bin",
                    file_path=f"/drives/{drive}/{file_id}.bin", file_type=kind,
                    mime_type=_KIND_MIME.get(kind, "application/octet-stream"),
                    file_size=1, active=True,
                ))
            for i in range(len(vectors)):
                session.add(Embedding(
                    id=f"e_{file_id}_{channel}_{i}", file_id=file_id,
                    embedding_type=channel, vector_table=table,
                    content_preview="x",
                ))
            session.commit()
        finally:
            session.close()
        with self.engine.begin() as conn:
            for i, vec in enumerate(vectors):
                conn.execute(
                    text(f"INSERT INTO {table} VALUES (:i, :v)"),
                    {"i": f"e_{file_id}_{channel}_{i}",
                     "v": np.asarray(vec, dtype=np.float32).tobytes()},
                )

    def query(self, sql: str, params: dict | None = None) -> list[tuple]:
        with self.engine.begin() as conn:
            return [tuple(r) for r in conn.execute(text(sql), params or {})]

    def execute(self, sql: str, params: dict | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(sql), params or {})

    def create_feed_log(self) -> None:
        self.execute(
            "CREATE TABLE IF NOT EXISTS pickup_feed_log ("
            " drive_id TEXT NOT NULL, viewer_id VARCHAR(16) NOT NULL,"
            " computed_at DATETIME NOT NULL, profile_version INTEGER NOT NULL,"
            " file_ids TEXT NOT NULL,"
            " PRIMARY KEY (drive_id, viewer_id, computed_at))"
        )

    def add_feed_log(
        self, *, drive: str = "a", viewer: str = "v1",
        computed_at: datetime, version: int = 3, file_ids: list[str],
    ) -> None:
        self.execute(
            "INSERT INTO pickup_feed_log VALUES (:d, :v, :c, :p, :f)",
            {"d": drive, "v": viewer, "c": ts(computed_at), "p": version,
             "f": json.dumps(file_ids)},
        )

    def drop_feed_log(self) -> None:
        self.execute("DROP TABLE IF EXISTS pickup_feed_log")


def wire(monkeypatch, core: CoreDB, search: SearchDB) -> None:
    """Point the addon's session getters at the two test databases."""
    import app.database as database

    monkeypatch.setattr(database, "_HomevaultSession", core.sessions)
    monkeypatch.setattr(database, "_litloft_engine", core.engine)
    monkeypatch.setattr(database, "_SearchSession", search.sessions)
    monkeypatch.setattr(database, "_search_engine", search.engine)
