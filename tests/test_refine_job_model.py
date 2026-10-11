"""The refine job records the model of the profile it was resolved with."""

from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

for _mod in ("torch", "sentence_transformers", "faster_whisper", "sqlite_vec"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import Base  # noqa: E402
from app.models import IndexedFile, TranscriptChunk  # noqa: E402
from app.workers import refine  # noqa: E402
from tests.llm_helpers import resolved_with  # noqa: E402


@pytest.mark.asyncio
async def test_refine_job_stamps_the_text_model_on_every_write(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for table in ("fts_transcripts", "fts_transcripts_word"):
            conn.execute(text(
                f"CREATE VIRTUAL TABLE {table} USING fts5(file_id, chunk_index, text)"
            ))
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as s:
        s.add(IndexedFile(
            file_id="f00000000001", drive="d", filename="talk.mp4",
            file_path="/nowhere/talk.mp4", file_type="video", mime_type="video/mp4",
            file_size=1, active=True,
        ))
        s.add(TranscriptChunk(
            id=1, file_id="f00000000001", chunk_index=0, text="hello world",
            language="en", timestamp_start=0.0, timestamp_end=1.0,
            created_at=datetime.now(UTC),
        ))
        s.commit()

    @contextmanager
    def _db():
        session = Session()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    rechunked: list = []

    def _rechunk(session, file_id, *, refined_model):
        rechunked.append(refined_model)
        return []

    monkeypatch.setattr(refine, "get_search_db", _db)
    monkeypatch.setattr(refine, "_emit_ws_event", AsyncMock())
    monkeypatch.setattr(refine, "realign_words_for_chunk", lambda *a, **k: 0)
    monkeypatch.setattr(refine, "recompute_chunk_embeddings", AsyncMock())
    monkeypatch.setattr(refine, "rechunk_from_words", _rechunk)
    monkeypatch.setattr(refine.aligner, "job_scope", nullcontext)
    client = MagicMock()
    client.generate_json = AsyncMock(
        return_value=[{"id": 1, "text_refined": "Hello, world."}]
    )

    await refine._run_refine_job(
        "f00000000001", "job1", [1],
        resolved_with(client, model="text-model", vision_model="vision-model"),
    )

    with Session() as s:
        chunk = s.get(TranscriptChunk, 1)
        assert (chunk.text, chunk.refined_model) == ("Hello, world.", "text-model")
    assert rechunked == ["text-model"]
