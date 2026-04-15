"""RED-phase tests for the LLM batch refine core.

Spec: docs/superpowers/specs/2026-04-15-intelligence-transcript-refine.md

The refine worker sends ``transcript_chunks`` to the LLM in windows of
N chunks at a time and expects back a JSON array of
``{id, text_refined}`` objects. Malformed responses, id mismatches and
LLM exceptions must each skip the failing window without corrupting
chunks in other windows.

Target module: ``app.workers.refine`` with:

* ``WINDOW_SIZE`` (int, default 10)
* ``async refine_chunks(session, llm, chunks)`` -> ``RefineResult``
  where ``RefineResult.refined_count`` and ``.skipped_count`` are ints.

This module does not exist yet — tests import-guard accordingly.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import is EXPECTED to fail during RED phase — the module does not
# yet exist. Collection failure here counts as a failing test file.
from app.workers.refine import WINDOW_SIZE, refine_chunks  # noqa: E402


def _chunk(cid: int, text: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(
        id=cid,
        file_id="fileabc",
        chunk_index=cid - 1,
        text=text,
        text_refined_at=None,
        timestamp_start=start,
        timestamp_end=end,
        language="ja",
    )


@pytest.fixture()
def session():
    """SQLAlchemy-session-shaped mock; individual tests assert on calls."""
    return MagicMock()


@pytest.fixture()
def llm_stub():
    """LLM client stub with generate_json as AsyncMock."""
    stub = MagicMock()
    stub.enabled = True
    stub.generate_json = AsyncMock()
    return stub


@pytest.mark.asyncio
class TestRefineChunksHappyPath:
    async def test_updates_chunks_and_stamps_refined_at(self, session, llm_stub):
        chunks = [
            _chunk(1, "origin one", 0.0, 5.0),
            _chunk(2, "origin two", 5.0, 10.0),
        ]
        llm_stub.generate_json.return_value = [
            {"id": 1, "text_refined": "refined one"},
            {"id": 2, "text_refined": "refined two"},
        ]

        result = await refine_chunks(session, llm_stub, chunks)

        assert result.refined_count == 2
        assert result.skipped_count == 0
        # Text is overwritten and refined_at stamped. Originals are NOT
        # preserved — refine downstream re-chunks the transcript, which
        # would invalidate per-chunk originals anyway.
        assert chunks[0].text == "refined one"
        assert chunks[0].text_refined_at is not None
        assert chunks[1].text == "refined two"
        assert chunks[1].text_refined_at is not None


@pytest.mark.asyncio
class TestRefineChunksMalformedResponse:
    async def test_malformed_json_preserves_original(self, session, llm_stub):
        """generate_json returns None → preserve all chunks in that window."""
        chunks = [_chunk(1, "origin", 0.0, 5.0)]
        llm_stub.generate_json.return_value = None

        result = await refine_chunks(session, llm_stub, chunks)

        assert result.refined_count == 0
        assert result.skipped_count == 1
        assert chunks[0].text == "origin"
        assert chunks[0].text_refined_at is None

    async def test_non_list_response_preserves_original(self, session, llm_stub):
        chunks = [_chunk(1, "origin", 0.0, 5.0)]
        llm_stub.generate_json.return_value = {"unexpected": "dict"}

        result = await refine_chunks(session, llm_stub, chunks)
        assert result.refined_count == 0
        assert result.skipped_count == 1
        assert chunks[0].text == "origin"


@pytest.mark.asyncio
class TestRefineChunksIdMismatch:
    async def test_missing_id_skips_window(self, session, llm_stub):
        """A window is considered invalid when the returned id set
        doesn't match the requested id set. Spec requires the whole
        window be preserved, not partially applied — partial application
        could desync words alignment.
        """
        chunks = [
            _chunk(1, "a", 0.0, 5.0),
            _chunk(2, "b", 5.0, 10.0),
        ]
        # Missing id=2
        llm_stub.generate_json.return_value = [
            {"id": 1, "text_refined": "A"},
        ]

        result = await refine_chunks(session, llm_stub, chunks)
        assert result.refined_count == 0
        assert result.skipped_count == 2
        assert chunks[0].text == "a"
        assert chunks[1].text == "b"

    async def test_extra_id_skips_window(self, session, llm_stub):
        chunks = [_chunk(1, "a", 0.0, 5.0)]
        llm_stub.generate_json.return_value = [
            {"id": 1, "text_refined": "A"},
            {"id": 999, "text_refined": "hallucinated"},
        ]

        result = await refine_chunks(session, llm_stub, chunks)
        assert result.refined_count == 0
        assert result.skipped_count == 1
        assert chunks[0].text == "a"


@pytest.mark.asyncio
class TestRefineChunksExceptionHandling:
    async def test_llm_exception_skips_window_continues_next(
        self, session, llm_stub
    ):
        """When WINDOW_SIZE spans multiple batches, an exception in the
        first window must not abort the second.
        """
        # Build 2 * WINDOW_SIZE chunks so we get exactly 2 windows.
        total = WINDOW_SIZE * 2
        chunks = [_chunk(i + 1, f"txt{i}", i * 1.0, (i + 1) * 1.0) for i in range(total)]

        # First call raises, second call returns valid payload for its window.
        second_payload = [
            {"id": c.id, "text_refined": f"R{c.id}"}
            for c in chunks[WINDOW_SIZE:]
        ]
        llm_stub.generate_json.side_effect = [
            Exception("LLM exploded"),
            second_payload,
        ]

        result = await refine_chunks(session, llm_stub, chunks)

        # Window 1 (WINDOW_SIZE chunks) skipped, window 2 refined.
        assert result.refined_count == WINDOW_SIZE
        assert result.skipped_count == WINDOW_SIZE
        # Window 1 preserved verbatim (LLM raised, no mutations).
        for c in chunks[:WINDOW_SIZE]:
            assert c.text.startswith("txt")
            assert c.text_refined_at is None
        # Window 2 refined.
        for c in chunks[WINDOW_SIZE:]:
            assert c.text.startswith("R")
            assert c.text_refined_at is not None


@pytest.mark.asyncio
class TestRefineChunksBatching:
    async def test_respects_window_size_boundary(self, session, llm_stub):
        """With 2*WINDOW_SIZE + 3 chunks we expect 3 LLM calls with the
        last window short (the remainder). Each request must only carry
        the ids of its own window.
        """
        total = WINDOW_SIZE * 2 + 3
        chunks = [_chunk(i + 1, f"t{i}", i * 1.0, (i + 1) * 1.0) for i in range(total)]

        # Return a valid payload shaped to whatever was asked for each call.
        async def _respond(system_prompt: str, user_prompt: str, **_kw):
            # Parse the JSON payload to extract exact ids (avoid substring
            # matching bugs where "id": 1 matches as a prefix of "id": 11).
            parsed = json.loads(user_prompt)
            parsed_ids = [int(entry["id"]) for entry in parsed]
            return [{"id": i, "text_refined": f"R{i}"} for i in parsed_ids]

        llm_stub.generate_json.side_effect = _respond

        result = await refine_chunks(session, llm_stub, chunks)

        assert llm_stub.generate_json.call_count == 3
        assert result.refined_count == total
        assert result.skipped_count == 0
