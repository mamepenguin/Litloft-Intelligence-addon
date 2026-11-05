"""Hosts for the extraction-correction acceptance tests.

Every test here drives the addon through the routes and workers the spec
names, not through helpers invented for the test. The search DB is a real
SQLite file with the production FTS/vector skeleton, and the FastAPI app is
built from the real routers, so a route that does not exist answers 404
instead of failing the module at import time.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

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

from app.database import Base  # noqa: E402
from app.models import (  # noqa: E402
    Embedding,
    IndexedFile,
    JobRecord,
    TranscriptChunk,
    TranscriptWord,
)

DRIVE = "drive1"
OTHER_DRIVE = "drive2"
HEADERS = {"X-Lit-Drive": DRIVE}

_FTS_DDL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS fts_files USING fts5("
    "file_id, filename, title, description, tags_text, tokenize='trigram')",
    "CREATE VIRTUAL TABLE IF NOT EXISTS fts_files_word USING fts5("
    "file_id, filename, title, description, tags_text, "
    "tokenize=\"unicode61 remove_diacritics 2\")",
    "CREATE VIRTUAL TABLE IF NOT EXISTS fts_transcripts USING fts5("
    "file_id, chunk_index, text, tokenize='trigram')",
    "CREATE VIRTUAL TABLE IF NOT EXISTS fts_transcripts_word USING fts5("
    "file_id, chunk_index, text, tokenize=\"unicode61 remove_diacritics 2\")",
    "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content USING fts5("
    "file_id, chunk_index, page, text, tokenize='trigram')",
    "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content_word USING fts5("
    "file_id, chunk_index, page, text, "
    "tokenize=\"unicode61 remove_diacritics 2\")",
)

_MODULES = (
    "app.database",
    "app.routers.files",
    "app.routers.refine",
    "app.workers.whisper",
    "app.workers.ocr",
    "app.workers.refine",
    "app.workers.metadata",
    "app.indexer",
    "app.rag.tools.get_file_chunks",
)


@dataclass
class SearchDB:
    engine: Any
    path: Path
    Session: Any = field(repr=False)

    def add(self, row: Any) -> None:
        with self.Session() as session:
            session.add(row)
            session.commit()

    def execute(self, sql: str, **params: Any) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(sql), params)

    def rows(self, sql: str, **params: Any) -> list[tuple]:
        with self.engine.connect() as conn:
            return list(conn.execute(text(sql), params).fetchall())

    def scalar(self, sql: str, **params: Any) -> Any:
        result = self.rows(sql, **params)
        return result[0][0] if result else None

    def flag(self, file_id: str, name: str) -> Any:
        return self.scalar(
            f"SELECT {name} FROM indexed_files WHERE file_id = :f", f=file_id
        )

    def set_flag(self, file_id: str, name: str, value: Any) -> None:
        self.execute(
            f"UPDATE indexed_files SET {name} = :v WHERE file_id = :f",
            v=1 if value else 0,
            f=file_id,
        )

    def chunks(self, file_id: str) -> list[tuple]:
        return self.rows(
            "SELECT chunk_index, text, timestamp_start, timestamp_end, language, "
            "speaker_id FROM transcript_chunks WHERE file_id = :f "
            "ORDER BY chunk_index",
            f=file_id,
        )

    def words(self, file_id: str) -> list[tuple]:
        return self.rows(
            "SELECT text, timestamp_start, timestamp_end, language, speaker_id "
            "FROM transcript_words WHERE file_id = :f "
            "ORDER BY timestamp_start, id",
            f=file_id,
        )

    def fts(self, table: str, file_id: str) -> list[tuple]:
        return self.rows(
            f"SELECT chunk_index, text FROM {table} WHERE file_id = :f "
            "ORDER BY CAST(chunk_index AS INTEGER)",
            f=file_id,
        )

    def embeddings(self, file_id: str, kind: str) -> list[tuple]:
        return self.rows(
            "SELECT id, chunk_index, content_preview FROM embeddings "
            "WHERE file_id = :f AND embedding_type = :k ORDER BY id",
            f=file_id,
            k=kind,
        )


def build_search_db(tmp_path: Path, monkeypatch) -> SearchDB:
    """A real search.db skeleton wired into every module a path touches."""
    for name in _MODULES:
        importlib.import_module(name)

    path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for ddl in _FTS_DDL:
            conn.execute(text(ddl))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS vec_text ("
            "embedding_id TEXT PRIMARY KEY, embedding_type TEXT, vector BLOB)"
        ))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS suggested_tags (file_id TEXT)"
        ))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS suggested_chapters (file_id TEXT)"
        ))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS file_summaries (file_id TEXT)"
        ))

    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _write():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def _read():
        session = Session()
        try:
            yield session
        finally:
            session.expunge_all()
            session.rollback()
            session.close()

    for name in _MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        for attr, factory in (("get_search_db", _write), ("get_search_db_read", _read)):
            if hasattr(module, attr):
                monkeypatch.setattr(module, attr, factory)

    db = SearchDB(engine=engine, path=path, Session=Session)
    _install_settings(monkeypatch)
    return db


def _install_settings(monkeypatch) -> None:
    """A settings object whose feature modes are all safe for tests."""
    import dataclasses

    import app.config as config

    features = dataclasses.replace(
        config.settings.features,
        image_ocr="on_index",
        transcript_refine="manual",
        detailed_summaries="manual",
    )
    settings = dataclasses.replace(config.settings, features=features)
    for name in (
        "app.config",
        "app.indexer",
        "app.routers.files",
        "app.routers.refine",
        "app.routers.admin",
        "app.workers.whisper",
        "app.workers.ocr",
        "app.workers.refine",
        "app.main",
    ):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "settings"):
            monkeypatch.setattr(module, "settings", settings)


class FakeIndexManager:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str]] = []
        self.queued_pairs: set[tuple[str, str]] = set()

    def is_queued(self, file_id: str, task: str) -> bool:
        return (file_id, task) in self.queued_pairs

    async def enqueue_task_for_file(self, file_id: str, task: str) -> None:
        self.enqueued.append((file_id, task))


def build_api(
    monkeypatch,
    *,
    raise_server_exceptions: bool = True,
) -> tuple[TestClient, FakeIndexManager]:
    from app.routers import files, refine

    app = FastAPI()
    app.include_router(files.router)
    app.include_router(refine.router)
    manager = FakeIndexManager()
    app.dependency_overrides[files.get_index_manager] = lambda: manager
    client = TestClient(app, raise_server_exceptions=raise_server_exceptions)
    return client, manager


@pytest.fixture()
def search_db(tmp_path, monkeypatch) -> SearchDB:
    return build_search_db(tmp_path, monkeypatch)


@pytest.fixture()
def api(search_db, monkeypatch) -> tuple[TestClient, FakeIndexManager]:
    return build_api(monkeypatch)


def seed_file(
    db: SearchDB,
    file_id: str,
    *,
    drive: str = DRIVE,
    file_type: str = "video",
    mime_type: str = "video/mp4",
    file_path: str | None = None,
    active: bool = True,
    **flags: Any,
) -> None:
    db.add(IndexedFile(
        file_id=file_id,
        drive=drive,
        filename=Path(file_path or f"{file_id}.bin").name,
        file_path=file_path or f"/drives/{drive}/{file_id}.bin",
        file_type=file_type,
        mime_type=mime_type,
        file_size=1,
        active=active,
        metadata_indexed=flags.pop("metadata_indexed", True),
        clip_indexed=flags.pop("clip_indexed", True),
        clip_thumbnail_indexed=flags.pop("clip_thumbnail_indexed", True),
        whisper_indexed=flags.pop("whisper_indexed", True),
        text_indexed=flags.pop("text_indexed", True),
        tfidf_keywords_indexed=flags.pop("tfidf_keywords_indexed", True),
        idf_terms_indexed=flags.pop("idf_terms_indexed", True),
        idf_terms_gen=flags.pop("idf_terms_gen", 1),
        **flags,
    ))


def seed_transcript(
    db: SearchDB,
    file_id: str,
    chunks: list[dict[str, Any]],
    *,
    words: list[dict[str, Any]] | None = None,
    drive: str = DRIVE,
    language: str = "en",
    flags: dict[str, Any] | None = None,
) -> None:
    """Seed chunk rows, their word rows, the FTS mirrors and embeddings."""
    seed_flags = dict(flags or {})
    active = seed_flags.pop("active", True)
    seed_file(db, file_id, drive=drive, active=active, **seed_flags)
    with db.Session() as session:
        for chunk in chunks:
            session.add(TranscriptChunk(
                file_id=file_id,
                chunk_index=chunk["index"],
                text=chunk["text"],
                language=chunk.get("language", language),
                timestamp_start=chunk["start"],
                timestamp_end=chunk["end"],
                speaker_id=chunk.get("speaker"),
                text_refined_at=chunk.get("refined_at"),
            ))
        for word in words or []:
            session.add(TranscriptWord(
                file_id=file_id,
                text=word["text"],
                language=word.get("language", language),
                timestamp_start=word["start"],
                timestamp_end=word["end"],
                speaker_id=word.get("speaker"),
            ))
        session.commit()

    for chunk in chunks:
        if not chunk["text"].strip():
            continue
        for table in ("fts_transcripts", "fts_transcripts_word"):
            db.execute(
                f"INSERT INTO {table}(file_id, chunk_index, text) "
                "VALUES(:f, :i, :t)",
                f=file_id,
                i=str(chunk["index"]),
                t=chunk["text"],
            )
        _seed_embedding(
            db,
            f"wh_{file_id}_{chunk['index']}_deadbeef",
            file_id,
            "whisper",
            chunk["text"],
            start=chunk["start"],
            end=chunk["end"],
            chunk_index=chunk["index"],
        )
    _seed_embedding(
        db,
        f"tk_{file_id}_deadbeef",
        file_id,
        "tfidf_keywords",
        "keywords from the transcript",
    )


def seed_picture(
    db: SearchDB,
    file_id: str,
    chunks: list[str],
    *,
    drive: str = DRIVE,
    mime_type: str = "image/png",
    ocr_indexed: bool = True,
    active: bool = True,
    page: int | None = None,
) -> None:
    """Seed the rows a picture read before the stored-text change has."""
    seed_file(
        db,
        file_id,
        drive=drive,
        file_type="image",
        mime_type=mime_type,
        active=active,
        ocr_indexed=ocr_indexed,
    )
    for index, chunk in enumerate(chunks):
        page_value = "" if page is None else str(page)
        for table in ("fts_text_content", "fts_text_content_word"):
            db.execute(
                f"INSERT INTO {table}(file_id, chunk_index, page, text) "
                "VALUES(:f, :i, :p, :t)",
                f=file_id,
                i=str(index),
                p=page_value,
                t=chunk,
            )
        _seed_embedding(
            db,
            f"ocr_{file_id}_{index}_deadbeef",
            file_id,
            "text_content",
            chunk,
            page=page,
            chunk_index=index,
        )


def _seed_embedding(
    db: SearchDB,
    embedding_id: str,
    file_id: str,
    kind: str,
    preview: str,
    *,
    start: float | None = None,
    end: float | None = None,
    page: int | None = None,
    chunk_index: int | None = None,
) -> None:
    with db.Session() as session:
        session.add(Embedding(
            id=embedding_id,
            file_id=file_id,
            embedding_type=kind,
            vector_table="vec_text",
            timestamp_start=start,
            timestamp_end=end,
            page=page,
            chunk_index=chunk_index,
            content_preview=preview,
            created_at=datetime.now(UTC),
        ))
        session.commit()
    db.execute(
        "INSERT INTO vec_text(embedding_id, embedding_type, vector) "
        "VALUES(:id, :k, :v)",
        id=embedding_id,
        k=kind,
        v=b"\x00" * 16,
    )


def run_whisper(monkeypatch, file_id: str, provider, *, keyword_calls=None) -> bool:
    """Run ``index_whisper`` against a fake provider and the test DB."""
    from unittest.mock import AsyncMock, patch

    from app.workers import whisper

    def _embed(texts):
        return [np.zeros(4, dtype=np.float32) for _ in texts]

    calls = keyword_calls if keyword_calls is not None else []
    monkeypatch.setattr(whisper, "embed_passages", _embed)
    monkeypatch.setattr(
        whisper, "_index_tfidf_keywords",
        lambda fid, texts: calls.append((fid, list(texts))),
    )
    monkeypatch.setattr(whisper, "_emit_ws_event", AsyncMock())
    with (
        patch.object(whisper, "get_provider", return_value=provider),
        patch.object(whisper, "validate_file_path", return_value=True),
        patch(
            "app.policy_client.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
    ):
        return asyncio.run(whisper.index_whisper(file_id))


def add_running_job(db: SearchDB, file_id: str, *, kind: str = "transcription") -> None:
    db.add(JobRecord(
        file_id=file_id,
        job_kind=kind,
        provider="whisper_local",
        status="running",
    ))


def coded_detail(response) -> Any:
    body = response.json()
    return (body or {}).get("detail")
