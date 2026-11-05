"""Invariants I1-I15 of the correction/original change.

SPEC-ID: SPEC-ADDON-065
SPEC-ID: SPEC-ADDON-067
SPEC-ID: SPEC-ADDON-068

Each class names the invariant it holds. Cases the spec leaves to the
follow-up worker are represented by their persisted or route-visible part,
because the spec does not name that worker's entry point.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.workers.transcription.base import (
    ProviderCapabilities,
    TranscriptionSegment,
    WordToken,
)
from tests.extraction_fixtures import (
    api,
    search_db,
    DRIVE,
    HEADERS,
    OTHER_DRIVE,
    add_running_job,
    build_api,
    seed_picture,
    seed_transcript,
)
from tests.llm_helpers import resolved_with

FILE = "invariant001"
PICTURE = "invariant002"


def _chunks() -> list[dict]:
    return [
        {"index": 0, "text": "alpha one", "start": 0.0, "end": 5.0, "speaker": "s0"},
        {"index": 1, "text": "bravo two", "start": 5.0, "end": 10.0, "speaker": "s1"},
        {"index": 2, "text": "charlie three", "start": 10.0, "end": 15.0},
    ]


def _words() -> list[dict]:
    return [
        {"text": "alpha", "start": 0.2, "end": 1.0},
        {"text": "one", "start": 4.5, "end": 5.0},
        {"text": "|", "start": 5.0, "end": 5.0},
        {"text": "bravo", "start": 5.2, "end": 6.0},
        {"text": "two", "start": 9.0, "end": 9.9},
        {"text": "charlie", "start": 10.2, "end": 11.0},
    ]


def _correct(client, index: int, text: str, expected: str, start: float):
    return client.put(
        f"/files/{FILE}/transcript/utterances/{index}",
        json={"text": text, "expected_text": expected, "expected_start": start},
        headers=HEADERS,
    )


def _capture_posts(monkeypatch, captured: list[dict[str, Any]]) -> None:
    real_async = httpx.AsyncClient
    real_sync = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content) if request.content else {})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)

    def async_factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_async(transport=transport, **kwargs)

    def sync_factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_sync(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", async_factory)
    monkeypatch.setattr(httpx, "Client", sync_factory)


class TestI1OnlyTheChosenUtteranceChanges:
    @pytest.mark.parametrize(
        "index,old_text,new_text,start",
        [
            (0, "alpha one", "Alpha, fixed.", 0.0),
            (1, "bravo two", "Bravo, fixed.", 5.0),
            (2, "charlie three", "Charlie, fixed.", 10.0),
        ],
    )
    def test_every_other_row_and_word_is_unchanged(
        self, search_db, api, index, old_text, new_text, start
    ):
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        before = {row[0]: row for row in search_db.chunks(FILE)}
        before_words = set(search_db.words(FILE))
        boundary = ("|", 5.0, 5.0, "en", None)

        assert _correct(client, index, new_text, old_text, start).status_code == 200

        after = {row[0]: row for row in search_db.chunks(FILE)}
        assert after[index][1] == new_text
        for other in before:
            if other != index:
                assert after[other] == before[other]
        assert boundary in search_db.words(FILE)
        # Only words inside the edited row's span leave; the boundary stays.
        for word in before_words:
            if word == boundary:
                continue
            inside = before[index][2] <= word[1] and word[2] <= before[index][3]
            if not inside:
                assert word in search_db.words(FILE)


class TestI2SearchFollowsTheText:
    @pytest.mark.parametrize("edit", ["correction", "restore"])
    def test_the_replaced_term_leaves_and_the_new_term_arrives(
        self, search_db, api, edit
    ):
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        if edit == "correction":
            assert _correct(client, 1, "Kubernetes", "bravo two", 5.0).status_code == 200
        else:
            assert _correct(client, 1, "Kubernetes", "bravo two", 5.0).status_code == 200
            assert client.post(
                f"/files/{FILE}/transcript/restore", headers=HEADERS
            ).status_code == 200

        rows = search_db.fts("fts_transcripts", FILE)
        text = "\n".join(row[1] for row in rows)
        if edit == "correction":
            assert "Kubernetes" in text
            assert "bravo two" not in text
        else:
            assert "bravo two" in text
            assert "Kubernetes" not in text

    def test_a_stale_item_is_not_what_makes_a_later_correction_stale(
        self, search_db, api
    ):
        """I2: the persisted pending mark survives a fresh read."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client, 1, "Bravo, fixed.", "bravo two", 5.0).status_code == 200

        first = client.get(f"/files/{FILE}/transcript", headers=HEADERS).json()
        second = client.get(f"/files/{FILE}/transcript", headers=HEADERS).json()

        assert first["followups_pending"] is True
        assert second["followups_pending"] is True
        assert second["chunks"][1]["wordsPending"] is True


class TestI3Subtitles:
    def test_a_corrected_no_word_utterance_is_one_cue_and_no_cache(
        self, search_db, api
    ):
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client, 1, "Bravo, fixed.", "bravo two", 5.0).status_code == 200

        response = client.get(f"/files/{FILE}/subtitles.vtt", headers=HEADERS)

        assert response.status_code == 200
        assert response.headers["cache-control"] == "private, no-cache"
        assert response.text.count("Bravo, fixed.") == 1
        assert "00:00:05.000 --> 00:00:10.000" in response.text


class TestI4TheOriginalIsWrittenOnce:
    def test_a_second_correction_does_not_replace_the_original(self, search_db, api):
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        machine = search_db.chunks(FILE)
        assert _correct(client, 1, "first", "bravo two", 5.0).status_code == 200
        assert _correct(client, 1, "second", "first", 5.0).status_code == 200

        assert client.post(
            f"/files/{FILE}/transcript/restore", headers=HEADERS
        ).status_code == 200
        assert search_db.chunks(FILE) == machine


class TestI5RestoreIsAllOrNothing:
    def test_a_restore_that_raises_leaves_the_corrected_state(self, search_db, monkeypatch):
        client, _ = build_api(monkeypatch, raise_server_exceptions=False)
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client, 1, "first", "bravo two", 5.0).status_code == 200
        before = search_db.chunks(FILE)
        search_db.execute(
            "CREATE TRIGGER fail_i5_insert BEFORE INSERT ON transcript_chunks "
            "BEGIN SELECT RAISE(ABORT, 'no'); END"
        )
        search_db.execute(
            "CREATE TRIGGER fail_i5_update BEFORE UPDATE ON transcript_chunks "
            "BEGIN SELECT RAISE(ABORT, 'no'); END"
        )

        assert client.post(
            f"/files/{FILE}/transcript/restore", headers=HEADERS
        ).status_code == 500
        assert search_db.chunks(FILE) == before
        assert client.get(
            f"/files/{FILE}/transcript", headers=HEADERS
        ).json()["has_original"] is True


class TestI10Purge:
    @pytest.mark.parametrize("kind", ["transcript", "picture"])
    def test_every_row_of_a_purged_file_is_gone(self, search_db, api, kind):
        client, _ = api
        if kind == "transcript":
            seed_transcript(search_db, FILE, _chunks(), words=_words())
            assert _correct(client, 1, "fixed", "bravo two", 5.0).status_code == 200
            file_id = FILE
        else:
            seed_picture(search_db, PICTURE, ["old text"])
            assert client.get(
                f"/files/{PICTURE}/picture-text", headers=HEADERS
            ).status_code == 200
            assert client.put(
                f"/files/{PICTURE}/picture-text",
                json={"text": "fixed", "expected_text": "old text"},
                headers=HEADERS,
            ).status_code == 200
            file_id = PICTURE

        from app.indexer import _purge_file

        _purge_file(file_id)

        with search_db.engine.connect() as conn:
            tables = [
                row[0]
                for row in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            ]
            for table in tables:
                if table.startswith("sqlite_"):
                    continue
                columns = {
                    row[1]
                    for row in conn.exec_driver_sql(
                        f"PRAGMA table_info({table})"
                    )
                }
                if "file_id" not in columns:
                    continue
                remaining = conn.exec_driver_sql(
                    f"SELECT COUNT(*) FROM {table} WHERE file_id = ?",
                    (file_id,),
                ).scalar()
                assert remaining == 0, table


class TestI12GapsKeepChunkIndexes:
    def test_the_panel_tool_and_excerpt_neighbours_use_the_index(
        self, search_db, api, monkeypatch
    ):
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client, 1, "", "bravo two", 5.0).status_code == 200

        from app.rag.tools.context import ToolContext
        from app.rag.tools import get_file_chunks as tool

        async def _allowed(file_ids, credential=None, drive=None):
            return set(file_ids)

        monkeypatch.setattr(tool, "ensure_access", _allowed)
        result = asyncio.run(tool.get_file_chunks(
            context=ToolContext(drive=DRIVE),
            file_id=FILE,
            type="transcript",
            mode="summary",
        ))

        assert [row["chunk_id"] for row in result.payload["chunks"]] == [0, 2]

        excerpt = client.get(
            f"/files/{FILE}/chunks/transcript:2/excerpt", headers=HEADERS
        )
        assert excerpt.status_code == 200, excerpt.text
        assert "alpha one" in excerpt.json()["prefix"]


class TestI13MarksAndIdempotence:
    def test_a_no_op_and_the_last_utterance_write_nothing(self, search_db, api):
        client, _ = api
        seed_transcript(
            search_db, FILE,
            [{"index": 0, "text": "only", "start": 0.0, "end": 1.0}],
        )
        before = search_db.chunks(FILE)

        assert _correct(client, 0, "only", "only", 0.0).status_code == 200
        last = _correct(client, 0, "", "only", 0.0)
        assert last.status_code == 422
        assert search_db.chunks(FILE) == before

    def test_a_new_correction_clears_redo_ended_without_result(
        self, search_db, api, monkeypatch
    ):
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client, 1, "first", "bravo two", 5.0).status_code == 200
        assert client.post(
            f"/files/{FILE}/reindex",
            json={"tasks": ["whisper"], "overwrite_corrections": True},
            headers=HEADERS,
        ).status_code in (200, 202)

        from tests.extraction_fixtures import run_whisper
        from app.workers.transcription.errors import FatalError

        provider = MagicMock()
        provider.name = "fake"
        provider.capabilities = ProviderCapabilities(
            sends_audio_offhost=False, supports_diarization=False,
            supports_hotwords=False, supports_word_timestamps=True,
            handles_own_retry=False,
        )
        provider.transcribe = AsyncMock(side_effect=FatalError("boom"))
        assert run_whisper(monkeypatch, FILE, provider) is False
        details = client.get(f"/files/{FILE}/index-details", headers=HEADERS).json()
        assert details["redo_ended_without_result"]["transcript"] is True

        assert _correct(client, 1, "second", "first", 5.0).status_code == 200
        details = client.get(f"/files/{FILE}/index-details", headers=HEADERS).json()
        assert details["redo_ended_without_result"]["transcript"] is False


class TestI14RefineAlignmentFailureKeepsWords:
    @pytest.mark.parametrize(
        "outcome",
        [None, []],
        ids=["none", "empty"],
    )
    def test_a_non_success_outcome_leaves_the_old_words(
        self, search_db, outcome, monkeypatch
    ):
        from app.workers import aligner, refine

        seed_transcript(
            search_db, FILE,
            [{"index": 0, "text": "alpha one", "start": 0.0, "end": 5.0}],
            words=[{"text": "alpha", "start": 0.2, "end": 1.0}],
        )
        before = search_db.words(FILE)
        monkeypatch.setattr(aligner, "align_segment", lambda **kwargs: outcome)

        with search_db.Session() as session:
            changed = refine.realign_words_for_chunk(
                session, FILE, 0.0, 5.0, "alpha one", waveform=object()
            )
            session.commit()

        assert changed == 0
        assert search_db.words(FILE) == before

    def test_an_unsupported_language_does_not_estimate_times(self, search_db, monkeypatch):
        from app.workers import aligner, refine

        seed_transcript(
            search_db, FILE,
            [{"index": 0, "text": "alpha one", "start": 0.0, "end": 5.0}],
            words=[{"text": "alpha", "start": 0.2, "end": 1.0}],
        )
        before = search_db.words(FILE)
        monkeypatch.setattr(aligner, "_load_align_model", lambda language: None)

        with search_db.Session() as session:
            changed = refine.realign_words_for_chunk(
                session, FILE, 0.0, 5.0, "alpha one", waveform=object(),
                language_override="xx",
            )
            session.commit()

        assert changed == 0
        assert search_db.words(FILE) == before


class TestI15EventReasons:
    def test_a_completed_refine_posts_replaced_for_the_drive(
        self, search_db, api, monkeypatch
    ):
        captured: list[dict[str, Any]] = []
        _capture_posts(monkeypatch, captured)
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())

        from app.workers import refine

        async def _refine_chunks(session, llm, chunks, *, model):
            for chunk in chunks:
                chunk.text = chunk.text + " refined"
                chunk.text_refined_at = datetime.now(UTC)
                chunk.refined_model = model
            return SimpleNamespace(refined_count=len(chunks), skipped_count=0)

        monkeypatch.setattr(refine, "refine_chunks", _refine_chunks)
        monkeypatch.setattr(refine, "_emit_ws_event", AsyncMock())
        monkeypatch.setattr(refine, "realign_words_for_chunk", lambda *a, **k: 0)
        monkeypatch.setattr(refine, "recompute_chunk_embeddings", AsyncMock())
        monkeypatch.setattr(refine, "rechunk_from_words", lambda *a, **k: [])
        monkeypatch.setattr(refine.aligner, "job_scope", nullcontext)

        asyncio.run(refine._run_refine_job(
            FILE, "job1", [], resolved_with(MagicMock(), model="m")
        ))

        events = [
            body for body in captured
            if body.get("event") == "intelligence.transcript.updated"
        ]
        assert events, captured
        assert all(body["drive"] == DRIVE for body in events)
        assert events[-1]["data"]["reason"] == "replaced"

    def test_a_restore_names_restored_once(self, search_db, api, monkeypatch):
        captured: list[dict[str, Any]] = []
        _capture_posts(monkeypatch, captured)
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client, 1, "fixed", "bravo two", 5.0).status_code == 200

        assert client.post(
            f"/files/{FILE}/transcript/restore", headers=HEADERS
        ).status_code == 200

        restored = [
            body for body in captured
            if body.get("event") == "intelligence.transcript.updated"
            and body.get("data", {}).get("reason") == "restored"
        ]
        assert len(restored) == 1
