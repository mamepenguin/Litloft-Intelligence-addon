"""Every addon event names the drive of the file it concerns.

The core broadcasts an addon event whose payload has no top-level
``drive`` to every connected viewer, including viewers who cannot open
the file's drive. The refine, detailed-summary and vision events used to
carry no drive, so those viewers received them too. SPEC-ADDON-057 I14
makes the drive mandatory and drops a post that has none.

The HTTP call is intercepted at ``httpx``'s transport, so the assertion is
on the wire body rather than on a helper's arguments.
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
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

from app.database import Base, _create_file_summaries_table  # noqa: E402
from app.llm import VisionGeneration  # noqa: E402
from app.models import IndexedFile, TranscriptChunk  # noqa: E402
from app.workers import refine, summaries, vision  # noqa: E402
from tests.llm_helpers import resolved_with  # noqa: E402

DRIVE = "private-drive"

_REAL_ASYNC_CLIENT = httpx.AsyncClient
_REAL_CLIENT = httpx.Client


def _capture_posts(monkeypatch, captured: list[dict[str, Any]]) -> None:
    """Record every posted JSON body and answer 200."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        captured.append(body)
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)

    def _async_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return _REAL_ASYNC_CLIENT(transport=transport, **kwargs)

    def _sync_factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs.pop("transport", None)
        return _REAL_CLIENT(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _async_factory)
    monkeypatch.setattr(httpx, "Client", _sync_factory)


# ---------------------------------------------------------------------------
# The poster
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_post_without_a_drive_is_not_sent(
    monkeypatch, caplog,
) -> None:
    from app.addon_events import post_addon_event

    captured: list[dict[str, Any]] = []
    _capture_posts(monkeypatch, captured)

    with caplog.at_level(logging.WARNING, logger="app.addon_events"):
        posted = await post_addon_event(
            "intelligence.test.no_drive", {"file_id": "f-1"}
        )

    assert posted is False
    assert captured == []
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert any("intelligence.test.no_drive" in message for message in warnings)


def test_sync_post_without_a_drive_is_not_sent(monkeypatch, caplog) -> None:
    from app.addon_events import post_addon_event_sync

    captured: list[dict[str, Any]] = []
    _capture_posts(monkeypatch, captured)

    with caplog.at_level(logging.WARNING, logger="app.addon_events"):
        posted = post_addon_event_sync(
            "intelligence.test.no_drive_sync", {"file_id": "f-1"}
        )

    assert posted is False
    assert captured == []
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert any(
        "intelligence.test.no_drive_sync" in message for message in warnings
    )


@pytest.mark.asyncio
async def test_a_drive_inside_data_is_named_at_the_top_level(monkeypatch) -> None:
    from app.addon_events import post_addon_event

    captured: list[dict[str, Any]] = []
    _capture_posts(monkeypatch, captured)

    posted = await post_addon_event(
        "intelligence.test.scoped", {"file_id": "f-1", "drive": DRIVE}
    )

    assert posted is True
    assert captured[0]["drive"] == DRIVE


# ---------------------------------------------------------------------------
# Refine
# ---------------------------------------------------------------------------


@pytest.fixture()
def refine_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for table in ("fts_transcripts", "fts_transcripts_word"):
            conn.execute(text(
                f"CREATE VIRTUAL TABLE {table} USING fts5(file_id, chunk_index, text)"
            ))
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as session:
        session.add(IndexedFile(
            file_id="f-refine",
            drive=DRIVE,
            filename="talk.mp4",
            file_path="/nowhere/talk.mp4",
            file_type="video",
            mime_type="video/mp4",
            file_size=1,
            active=True,
        ))
        session.add(TranscriptChunk(
            id=1,
            file_id="f-refine",
            chunk_index=0,
            text="hello world",
            language="en",
            timestamp_start=0.0,
            timestamp_end=1.0,
            created_at=datetime.now(UTC),
        ))
        session.commit()

    @contextmanager
    def _db():
        session = Session()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    monkeypatch.setattr(refine, "get_search_db", _db)
    return engine


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "expected_events"),
    [
        (
            "success",
            {
                "intelligence.refine.started",
                "intelligence.refine.progress",
                "intelligence.refine.completed",
            },
        ),
        (
            "failure",
            {"intelligence.refine.started", "intelligence.refine.failed"},
        ),
    ],
    ids=["success", "failure"],
)
async def test_refine_events_name_the_files_drive(
    monkeypatch, refine_db, scenario, expected_events,
) -> None:
    captured: list[dict[str, Any]] = []
    _capture_posts(monkeypatch, captured)

    monkeypatch.setattr(refine, "recompute_chunk_embeddings", AsyncMock())
    monkeypatch.setattr(refine, "realign_words_for_chunk", lambda *a, **k: 0)
    monkeypatch.setattr(refine, "rechunk_from_words", lambda *a, **k: [])
    monkeypatch.setattr(refine.aligner, "job_scope", nullcontext)
    if scenario == "failure":
        def _boom(*args, **kwargs):
            raise RuntimeError("reindex failed")

        monkeypatch.setattr(refine, "_reindex_keywords", _boom)

    client = MagicMock()
    client.generate_json = AsyncMock(
        return_value=[{"id": 1, "text_refined": "Hello, world."}]
    )

    await refine._run_refine_job(
        "f-refine", "job1", [1], resolved_with(client, model="text-model")
    )

    posted = {body["event"] for body in captured}
    assert expected_events <= posted
    assert all(body.get("drive") == DRIVE for body in captured), captured


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summaries_citations_event_names_the_files_drive(
    monkeypatch,
) -> None:
    captured: list[dict[str, Any]] = []
    _capture_posts(monkeypatch, captured)

    import app.citations as citations

    monkeypatch.setattr(
        citations, "calculate_and_store", lambda file_id, text: (2, 1)
    )

    await summaries._recalculate_citations("f-summary", "body text", DRIVE)

    assert [body["event"] for body in captured] == [
        "intelligence.detailed_summary.citations_ready"
    ]
    assert captured[0]["drive"] == DRIVE


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------


@pytest.fixture()
def vision_db(tmp_path, monkeypatch, make_settings):
    engine = create_engine(f"sqlite:///{tmp_path / 'search.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _create_file_summaries_table(conn)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as session:
        session.add(IndexedFile(
            file_id="img-1",
            drive=DRIVE,
            filename="cat.jpg",
            file_path="/nowhere/cat.jpg",
            file_type="image",
            mime_type="image/jpeg",
            file_size=1000,
            active=True,
        ))
        session.commit()

    @contextmanager
    def _db():
        session = Session()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    monkeypatch.setattr(vision, "get_search_db", _db)
    monkeypatch.setattr(vision, "settings", make_settings())
    # The success path re-embeds metadata; that is not this test's subject.
    import app.workers.metadata as metadata_worker

    monkeypatch.setattr(
        metadata_worker, "index_metadata_batch", lambda file_ids: None
    )
    return engine


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "expected_events"),
    [
        (
            "success",
            {
                "intelligence.vision_describe.started",
                "intelligence.vision_describe.succeeded",
            },
        ),
        (
            "load_failure",
            {
                "intelligence.vision_describe.started",
                "intelligence.vision_describe.failed",
            },
        ),
    ],
    ids=["success", "load_failure"],
)
async def test_vision_events_name_the_files_drive(
    use_llm, monkeypatch, vision_db, scenario, expected_events,
) -> None:
    captured: list[dict[str, Any]] = []
    _capture_posts(monkeypatch, captured)

    llm_stub = MagicMock()
    llm_stub.enabled = True
    llm_stub.generate_vision = AsyncMock(
        return_value=VisionGeneration("A red apple.", None)
    )
    use_llm(llm_stub, vision_model="llava:13b")

    monkeypatch.setattr(
        vision, "_preprocess_image", lambda raw, mime: (raw, mime)
    )
    monkeypatch.setattr(vision, "_embed_and_store", lambda file_id, text: None)
    monkeypatch.setattr(
        vision,
        "_load_image_bytes",
        lambda file_id: None
        if scenario == "load_failure"
        else (b"\xff\xd8fake", "image/jpeg"),
    )

    await vision.VisionDescribeWorker()._process_file("img-1")

    posted = {body["event"] for body in captured}
    assert expected_events <= posted
    assert all(body.get("drive") == DRIVE for body in captured), captured
