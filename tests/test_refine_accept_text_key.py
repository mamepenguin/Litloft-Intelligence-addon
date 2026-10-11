"""SPEC-ADDON-020: refine accepts the refined text under the input's key."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import IndexedFile, TranscriptChunk
from app.workers import refine
from app.workers.refine import WINDOW_SIZE, refine_chunks
from tests.llm_helpers import resolved_with


def _chunk(cid: int, body: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=cid,
        file_id="fileabc",
        chunk_index=cid - 1,
        text=body,
        text_refined_at=None,
        refined_model=None,
        timestamp_start=float(cid - 1),
        timestamp_end=float(cid),
        language="en",
    )


def _llm(*responses):
    client = MagicMock()
    client.enabled = True
    client.generate_json = AsyncMock(side_effect=list(responses))
    return client


async def _refine(chunks, *responses):
    return await refine_chunks(MagicMock(), _llm(*responses), chunks, model="routed-model")


def _state(chunks):
    return [(c.text, c.text_refined_at is not None, c.refined_model) for c in chunks]


# I2 / Normal flow 2: the measured local-model shape, under both accepted wrappers.
@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [lambda items: {"items": items}, lambda items: items],
                         ids=["items_wrapper", "bare_list"])
async def test_spec_addon_020_items_under_text_are_applied(wrap):
    chunks = [_chunk(1, "hello world"), _chunk(2, "how are you")]

    result = await _refine(chunks, wrap([
        {"id": 1, "text": "Hello, world."},
        {"id": 2, "text": "How are you?"},
    ]))

    assert (result.refined_count, result.skipped_count) == (2, 0)
    assert _state(chunks) == [
        ("Hello, world.", True, "routed-model"),
        ("How are you?", True, "routed-model"),
    ]


# Boundary: a window of one chunk.
@pytest.mark.asyncio
async def test_spec_addon_020_single_item_under_text_is_applied():
    chunks = [_chunk(1, "only one")]

    result = await _refine(chunks, {"items": [{"id": 1, "text": "Only one."}]})

    assert (result.refined_count, result.skipped_count) == (1, 0)
    assert _state(chunks) == [("Only one.", True, "routed-model")]


# Normal flow 3: id is coerced to int whichever key carries the text.
@pytest.mark.asyncio
async def test_spec_addon_020_string_id_with_text_key_is_coerced():
    chunks = [_chunk(1, "one"), _chunk(2, "two")]

    result = await _refine(chunks, {"items": [
        {"id": "1", "text": "One."},
        {"id": "2", "text": "Two."},
    ]})

    assert (result.refined_count, result.skipped_count) == (2, 0)
    assert [c.text for c in chunks] == ["One.", "Two."]


# I1 / Normal flow 2: text_refined wins when both are present. The sibling item under
# text makes the window depend on the text key being read at all.
@pytest.mark.asyncio
async def test_spec_addon_020_text_refined_wins_over_text():
    chunks = [_chunk(1, "hello world"), _chunk(2, "two")]

    result = await _refine(chunks, {"items": [
        {"id": 1, "text": "echoed input", "text_refined": "Hello, world."},
        {"id": 2, "text": "Two."},
    ]})

    assert (result.refined_count, result.skipped_count) == (2, 0)
    assert _state(chunks) == [
        ("Hello, world.", True, "routed-model"),
        ("Two.", True, "routed-model"),
    ]


# I5: a window mixing the two keys is accepted.
@pytest.mark.asyncio
async def test_spec_addon_020_mixed_keys_in_one_window_are_accepted():
    chunks = [_chunk(1, "one"), _chunk(2, "two"), _chunk(3, "three")]

    result = await _refine(chunks, {"items": [
        {"id": 1, "text": "One."},
        {"id": 2, "text_refined": "Two."},
        {"id": 3, "text": "ignored", "text_refined": "Three."},
    ]})

    assert (result.refined_count, result.skipped_count) == (3, 0)
    assert [c.text for c in chunks] == ["One.", "Two.", "Three."]


# Failure case: echoed unchanged under text is stored and stamped as refined.
@pytest.mark.asyncio
async def test_spec_addon_020_unchanged_echo_under_text_is_stamped_refined():
    chunks = [_chunk(1, "hello world")]

    result = await _refine(chunks, {"items": [{"id": 1, "text": "hello world"}]})

    assert (result.refined_count, result.skipped_count) == (1, 0)
    assert _state(chunks) == [("hello world", True, "routed-model")]


def _two_windows():
    """WINDOW_SIZE chunks in the first window and one in the second."""
    return [_chunk(i, f"t{i}") for i in range(1, WINDOW_SIZE + 2)]


def _second_window_under_text(chunks):
    return {"items": [{"id": chunks[WINDOW_SIZE].id, "text": "Last."}]}


def _assert_first_rejected_second_applied(result, chunks):
    assert (result.refined_count, result.skipped_count) == (1, WINDOW_SIZE)
    assert _state(chunks[:WINDOW_SIZE]) == [(f"t{c.id}", False, None) for c in chunks[:WINDOW_SIZE]]
    assert _state(chunks[WINDOW_SIZE:]) == [("Last.", True, "routed-model")]


# I3: each unusable item rejects its whole window; a following window under text is applied.
@pytest.mark.asyncio
@pytest.mark.parametrize("bad_item", [
    {},
    {"text_refined": None, "text": "Good."},
    {"text_refined": 7, "text": "Good."},
    {"text_refined": ["Good."]},
    {"text": None},
    {"text": 7},
    {"text": {"value": "Good."}},
], ids=[
    "neither_key",
    "text_refined_null_with_string_text",
    "text_refined_int_with_string_text",
    "text_refined_list",
    "text_null",
    "text_int",
    "text_dict",
])
async def test_spec_addon_020_window_with_unusable_item_is_rejected(bad_item):
    chunks = _two_windows()
    first = [{"id": c.id, "text": f"R{c.id}"} for c in chunks[:WINDOW_SIZE]]
    first[-1] = {"id": chunks[WINDOW_SIZE - 1].id, **bad_item}

    result = await _refine(chunks, {"items": first}, _second_window_under_text(chunks))

    _assert_first_rejected_second_applied(result, chunks)


# I4: a missing or an extra id rejects the window, whichever key the items use.
@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["text", "text_refined"])
@pytest.mark.parametrize("mismatch", ["missing_id", "extra_id"])
async def test_spec_addon_020_id_mismatch_is_rejected_for_either_key(key, mismatch):
    chunks = _two_windows()
    ids = [c.id for c in chunks[:WINDOW_SIZE]]
    ids = ids[:-1] if mismatch == "missing_id" else ids + [999]

    result = await _refine(
        chunks,
        {"items": [{"id": i, key: f"R{i}"} for i in ids]},
        _second_window_under_text(chunks),
    )

    _assert_first_rejected_second_applied(result, chunks)


# I6: the job with the measured response shape and no word rows.
FILE = "f00000000001"


def _db_factory(Session):
    @contextmanager
    def _db():
        session = Session()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    return _db


def _event_payloads(events, name):
    found = []

    def _walk(value):
        if isinstance(value, dict):
            if "refined_count" in value:
                found.append(value)
            for v in value.values():
                _walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                _walk(v)

    for call in events.call_args_list:
        if name in repr(call):
            _walk(list(call.args) + list(call.kwargs.values()))
    return found


@pytest.mark.asyncio
async def test_spec_addon_020_job_applies_measured_text_key_response(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for table in ("fts_transcripts", "fts_transcripts_word"):
            conn.execute(text(
                f"CREATE VIRTUAL TABLE {table} USING fts5(file_id, chunk_index, text)"
            ))
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    originals = ["we deployed it on cooper netties", "the cluster had three nodes",
                 "and it worked fine"]
    with Session() as s:
        s.add(IndexedFile(
            file_id=FILE, drive="d", filename="talk.mp4", file_path="/nowhere/talk.mp4",
            file_type="video", mime_type="video/mp4", file_size=1, active=True,
        ))
        for n, body in enumerate(originals):
            s.add(TranscriptChunk(
                id=n + 1, file_id=FILE, chunk_index=n, text=body, language="en",
                timestamp_start=n * 4.0, timestamp_end=(n + 1) * 4.0,
                created_at=datetime.now(UTC),
            ))
        s.commit()

    events = AsyncMock()
    monkeypatch.setattr(refine, "get_search_db", _db_factory(Session))
    monkeypatch.setattr(refine, "_emit_ws_event", events)
    monkeypatch.setattr(refine, "realign_words_for_chunk", lambda *a, **k: 0)
    monkeypatch.setattr(refine, "recompute_chunk_embeddings", AsyncMock())
    monkeypatch.setattr(refine.aligner, "job_scope", nullcontext)
    refined = ["We deployed it on Kubernetes.", "The cluster had 3 nodes.",
               "And it worked fine."]
    client = MagicMock()
    client.enabled = True
    client.generate_json = AsyncMock(return_value={
        "items": [{"id": n + 1, "text": body} for n, body in enumerate(refined)]
    })

    await refine._run_refine_job(
        FILE, "job1", [1, 2, 3], resolved_with(client, model="profile-model")
    )

    with Session() as s:
        rows = [
            (r.text, r.text_refined_at is not None, r.refined_model)
            for r in s.query(TranscriptChunk)
            .filter(TranscriptChunk.file_id == FILE)
            .order_by(TranscriptChunk.chunk_index)
        ]
    assert rows == [(body, True, "profile-model") for body in refined]
    completed = _event_payloads(events, "intelligence.refine.completed")
    assert completed, events.call_args_list
    assert [p["refined_count"] for p in completed] == [3]
