"""Protection, the explicit redo and a confirmed Refine (SPEC-ADDON-068).

SPEC-ID: SPEC-ADDON-068

Every case names an automatic path or a redo route the spec lists, and reads
the outcome from the rows and the routes the panel uses.
"""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.workers.transcription.base import (
    ProviderCapabilities,
    TranscriptionSegment,
    WordToken,
)
from app.workers.transcription.errors import FatalError
from tests.extraction_fixtures import (
    api,
    search_db,
    DRIVE,
    HEADERS,
    OTHER_DRIVE,
    add_running_job,
    build_api,
    coded_detail,
    run_whisper,
    seed_file,
    seed_picture,
    seed_transcript,
)
from tests.llm_helpers import resolved_with

FILE = "protected001"
PICTURE = "protected002"
LOFT = "protected003"


def _chunks() -> list[dict]:
    return [
        {"index": 0, "text": "alpha one", "start": 0.0, "end": 5.0},
        {"index": 1, "text": "bravo two", "start": 5.0, "end": 10.0},
    ]


def _words() -> list[dict]:
    return [
        {"text": "alpha", "start": 0.2, "end": 1.0},
        {"text": "one", "start": 4.5, "end": 5.0},
        {"text": "bravo", "start": 5.2, "end": 6.0},
        {"text": "two", "start": 9.0, "end": 9.9},
    ]


def _correct(client, file_id: str = FILE, index: int = 1, text: str = "Bravo, fixed."):
    return client.put(
        f"/files/{file_id}/transcript/utterances/{index}",
        json={"text": text, "expected_text": "bravo two", "expected_start": 5.0},
        headers=HEADERS,
    )


def _get(client, file_id: str = FILE):
    return client.get(f"/files/{file_id}/transcript", headers=HEADERS)


def _confirmed(client, file_id: str = FILE, tasks=("whisper",), **body):
    return client.post(
        f"/files/{file_id}/reindex",
        json={"tasks": list(tasks), "overwrite_corrections": True, **body},
        headers=HEADERS,
    )


def _provider(segments=None, error: Exception | None = None):
    provider = MagicMock()
    provider.name = "fake"
    provider.capabilities = ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
        handles_own_retry=False,
    )
    if error is not None:
        provider.transcribe = AsyncMock(side_effect=error)
    else:
        provider.transcribe = AsyncMock(return_value=segments or [])
    return provider


def _speech():
    words = "the quick brown fox".split()
    return _provider(segments=[TranscriptionSegment(
        text=" ".join(words),
        start=0.0,
        end=4.0,
        language="en",
        words=[
            WordToken(text=word, start=float(index), end=float(index) + 0.9)
            for index, word in enumerate(words)
        ],
    )])


def _fake_refine(monkeypatch, *, apply: bool = True, captured: list[int] | None = None):
    from app.workers import refine

    seen = captured if captured is not None else []

    async def _refine_chunks(session, llm, chunks, *, model):
        seen.extend(chunk.id for chunk in chunks)
        if apply:
            for chunk in chunks:
                chunk.text = chunk.text + " refined"
                chunk.text_refined_at = datetime.now(UTC)
                chunk.refined_model = model
            return SimpleNamespace(refined_count=len(chunks), skipped_count=0)
        return SimpleNamespace(refined_count=0, skipped_count=len(chunks))

    monkeypatch.setattr(refine, "refine_chunks", _refine_chunks)
    monkeypatch.setattr(refine, "_emit_ws_event", AsyncMock())
    monkeypatch.setattr(refine, "realign_words_for_chunk", lambda *a, **k: 0)
    monkeypatch.setattr(refine, "recompute_chunk_embeddings", AsyncMock())
    monkeypatch.setattr(refine, "rechunk_from_words", lambda *a, **k: [])
    monkeypatch.setattr(refine.aligner, "job_scope", nullcontext)
    return seen


async def _run_refine(file_id: str = FILE) -> None:
    from app.workers import refine

    client = MagicMock()
    await refine._run_refine_job(
        file_id, "job1", [], resolved_with(client, model="profile-model")
    )


class TestAutomaticPathsDoNotTouchACorrectedFile:
    @pytest.mark.asyncio
    async def test_resume_incomplete_does_not_queue_a_corrected_transcript(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C6, I6."""
        from app.indexer import IndexManager

        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        search_db.set_flag(FILE, "whisper_indexed", False)
        search_db.set_flag(FILE, "tfidf_keywords_indexed", False)
        before = search_db.chunks(FILE)

        manager = IndexManager()
        queued: list[tuple[str, str]] = []

        async def _enqueue(task, *, force: bool = False) -> None:
            queued.append((task.file_id, task.task_type.value))

        monkeypatch.setattr(manager, "_enqueue", _enqueue)
        await manager._resume_incomplete()

        assert ("whisper" not in [task for _, task in queued])
        assert bool(search_db.flag(FILE, "tfidf_keywords_indexed")) is False
        assert search_db.chunks(FILE) == before

    @pytest.mark.asyncio
    async def test_resume_incomplete_does_not_queue_a_corrected_picture(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C6: every features.image_ocr mode."""
        from app.indexer import IndexManager

        client, _ = api
        seed_picture(search_db, PICTURE, ["old text"], ocr_indexed=True)
        assert client.get(
            f"/files/{PICTURE}/picture-text", headers=HEADERS
        ).status_code == 200
        assert client.put(
            f"/files/{PICTURE}/picture-text",
            json={"text": "fixed", "expected_text": "old text"},
            headers=HEADERS,
        ).status_code == 200
        search_db.set_flag(PICTURE, "ocr_indexed", False)

        manager = IndexManager()
        queued: list[tuple[str, str]] = []

        async def _enqueue(task, *, force: bool = False) -> None:
            queued.append((task.file_id, task.task_type.value))

        monkeypatch.setattr(manager, "_enqueue", _enqueue)
        await manager._resume_incomplete()

        assert ("ocr" not in [task for _, task in queued])

    def test_the_loft_reset_ignores_a_corrected_reference(
        self, search_db, api, tmp_path
    ):
        """SPEC-ADDON-068 C6: a temp audio file does not win over a correction."""
        from app.indexer import IndexManager
        from app.workers.whisper import LOFT_MIME

        client, _ = api
        folder = tmp_path / "loft"
        folder.mkdir()
        loft = folder / "Talk.loft"
        loft.write_text("{}", encoding="utf-8")
        (folder / "Talk.vtt").write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nold captions\n",
            encoding="utf-8",
        )
        (folder / "Talk.stt_temp.m4a").write_bytes(b"audio")

        seed_transcript(search_db, LOFT, _chunks(), words=_words())
        search_db.execute(
            "UPDATE indexed_files SET mime_type = :m, file_path = :p, "
            "whisper_indexed = 1 WHERE file_id = :f",
            m=LOFT_MIME, p=str(loft), f=LOFT,
        )
        assert _correct(client, file_id=LOFT).status_code == 200
        before = search_db.chunks(LOFT)

        IndexManager.__new__(IndexManager)._reset_loft_refs_with_new_vtt()

        assert bool(search_db.flag(LOFT, "whisper_indexed")) is True
        assert search_db.chunks(LOFT) == before

    def test_the_orphan_cleanup_leaves_a_corrected_transcript(
        self, search_db, api
    ):
        """SPEC-ADDON-068 C6."""
        from app.workers.whisper import fail_orphaned_running_jobs

        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        add_running_job(search_db, FILE)

        fail_orphaned_running_jobs()

        assert len(search_db.chunks(FILE)) == 2
        state = _get(client).json()
        assert state["corrected"] is True
        assert state["has_original"] is True

    def test_the_transcription_task_hands_a_corrected_file_to_the_rebuild(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C7, I6."""
        from unittest.mock import patch

        from app.workers import whisper

        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        search_db.set_flag(FILE, "whisper_indexed", False)
        before = search_db.chunks(FILE)
        provider = _provider()

        with patch.object(whisper, "get_provider", return_value=provider):
            asyncio.run(whisper.index_whisper(FILE))

        provider.transcribe.assert_not_called()
        assert search_db.chunks(FILE) == before
        assert bool(search_db.flag(FILE, "whisper_indexed")) is False
        assert _get(client).json()["followups_pending"] is True

    def test_persist_transcript_discards_a_result_that_races_a_correction(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C7: the write-time re-check."""
        from app.workers import whisper

        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        add_running_job(search_db, FILE)
        job_id = search_db.scalar(
            "SELECT id FROM job_records WHERE file_id = :f ORDER BY id DESC LIMIT 1",
            f=FILE,
        )
        before = search_db.chunks(FILE)
        keyword_calls: list = []
        monkeypatch.setattr(
            whisper, "_index_tfidf_keywords",
            lambda fid, texts: keyword_calls.append((fid, list(texts))),
        )

        whisper._persist_transcript(
            FILE, "fake",
            [{
                "text": "the result that lost the race",
                "start": 0.0,
                "end": 1.0,
                "language": "en",
                "words": [{"text": "the", "start": 0.0, "end": 1.0}],
            }],
            job_id,
        )

        assert search_db.chunks(FILE) == before
        job = search_db.rows(
            "SELECT status, error_class FROM job_records WHERE id = :i",
            i=job_id,
        )[0]
        assert job == ("skipped", "Corrected")
        assert keyword_calls == []

    def test_the_zero_segment_path_also_rechecks_before_marking(
        self, search_db, api
    ):
        """SPEC-ADDON-068 C7."""
        from app.workers import whisper

        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        search_db.set_flag(FILE, "whisper_indexed", False)
        add_running_job(search_db, FILE)
        job_id = search_db.scalar(
            "SELECT id FROM job_records WHERE file_id = :f ORDER BY id DESC LIMIT 1",
            f=FILE,
        )

        whisper._persist_transcript(FILE, "fake", [], job_id)

        job = search_db.rows(
            "SELECT status, error_class FROM job_records WHERE id = :i",
            i=job_id,
        )[0]
        assert job == ("skipped", "Corrected")
        assert bool(search_db.flag(FILE, "whisper_indexed")) is False

    def test_ocr_write_discards_a_result_that_races_a_correction(
        self, search_db, api
    ):
        """SPEC-ADDON-068 C7."""
        from app.workers import ocr

        client, _ = api
        seed_picture(search_db, PICTURE, ["old text"])
        assert client.get(
            f"/files/{PICTURE}/picture-text", headers=HEADERS
        ).status_code == 200
        assert client.put(
            f"/files/{PICTURE}/picture-text",
            json={"text": "fixed", "expected_text": "old text"},
            headers=HEADERS,
        ).status_code == 200
        before = search_db.fts("fts_text_content", PICTURE)

        written = ocr._write(PICTURE, ["the result that lost the race"], [b"\x00" * 16])

        assert written is False
        assert search_db.fts("fts_text_content", PICTURE) == before
        assert client.get(
            f"/files/{PICTURE}/picture-text", headers=HEADERS
        ).json()["text"] == "fixed"


class TestExplicitRedoChecks:
    def test_reindex_refuses_a_corrected_kind_without_the_flag(
        self, search_db, api
    ):
        """SPEC-ADDON-068 C8, I7."""
        client, manager = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        search_db.set_flag(FILE, "whisper_indexed", True)

        response = client.post(
            f"/files/{FILE}/reindex",
            json={"tasks": ["whisper", "clip"]},
            headers=HEADERS,
        )

        assert response.status_code == 409
        assert coded_detail(response) == {"code": "corrected"}
        assert manager.enqueued == []
        assert bool(search_db.flag(FILE, "whisper_indexed")) is True
        assert bool(search_db.flag(FILE, "clip_indexed")) is True

    def test_the_flag_on_an_uncorrected_kind_is_ignored(self, search_db, api):
        """SPEC-ADDON-068 C8: a kind that is not corrected keeps today's path."""
        client, manager = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200

        response = client.post(
            f"/files/{FILE}/reindex",
            json={"tasks": ["clip"], "overwrite_corrections": True},
            headers=HEADERS,
        )

        assert response.status_code == 200, response.text
        assert bool(search_db.flag(FILE, "clip_indexed")) is False
        assert manager.enqueued == [(FILE, "clip")]

    def test_a_confirmed_reindex_marks_the_file_before_the_queued_split(
        self, search_db, api
    ):
        """SPEC-ADDON-068 C8: an already queued task still runs as the redo."""
        client, manager = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        manager.queued_pairs.add((FILE, "whisper"))

        response = _confirmed(client)

        assert response.status_code == 202
        assert response.json()["status"] == "already_queued"
        assert _get(client).json()["busy"] is True
        # Busy now refuses a correction.
        assert client.put(
            f"/files/{FILE}/transcript/utterances/1",
            json={
                "text": "another",
                "expected_text": "Bravo, fixed.",
                "expected_start": 5.0,
            },
            headers=HEADERS,
        ).status_code == 409

    def test_reindex_checks_the_body_and_the_picture_before_corrected(
        self, search_db, api
    ):
        """SPEC-ADDON-068 C8."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200

        empty = client.post(
            f"/files/{FILE}/reindex", json={"tasks": []}, headers=HEADERS
        )
        assert empty.status_code == 422

        missing = client.post(
            f"/files/missing00001/reindex", json={"tasks": ["whisper"]},
            headers=HEADERS,
        )
        assert missing.status_code == 404

        ocr = client.post(
            f"/files/{FILE}/reindex", json={"tasks": ["ocr"]}, headers=HEADERS
        )
        assert ocr.status_code == 422
        assert coded_detail(ocr) is not None or "ocr" in str(ocr.json()).lower()

    def test_refine_refuses_a_corrected_file_before_busy(self, search_db, api, monkeypatch):
        """SPEC-ADDON-068 C8/C10."""
        from unittest.mock import AsyncMock

        from app.routers import refine as refine_router

        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        monkeypatch.setattr(
            refine_router, "is_feature_enabled", AsyncMock(return_value=True)
        )
        monkeypatch.setattr(
            refine_router, "require_llm", AsyncMock(return_value=object())
        )
        started: list[str] = []
        monkeypatch.setattr(
            refine_router, "start_refine_job",
            lambda fid, ids, resolved: started.append(fid) or "job",
        )

        refused = client.post(
            f"/refine/files/{FILE}", json={}, headers=HEADERS
        )
        assert refused.status_code == 409
        assert coded_detail(refused) == {"code": "corrected"}
        assert started == []

        busy = client.post(
            f"/refine/files/{FILE}",
            json={"overwrite_corrections": True},
            headers=HEADERS,
        )
        assert busy.status_code == 409
        assert coded_detail(busy) == {"code": "busy"}
        assert started == []

    def test_folder_refine_skips_corrected_files_and_lists_them(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C6/C8, I7."""
        from unittest.mock import AsyncMock

        from app.routers import refine as refine_router

        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        seed_transcript(
            search_db, "uncorrected1",
            [{"index": 0, "text": "normal", "start": 0.0, "end": 1.0}],
        )
        assert _correct(client).status_code == 200
        monkeypatch.setattr(
            refine_router, "is_feature_enabled", AsyncMock(return_value=True)
        )
        monkeypatch.setattr(
            refine_router, "require_llm", AsyncMock(return_value=object())
        )
        monkeypatch.setattr(
            refine_router, "filter_transcript_file_ids",
            lambda session, drive, ids: list(ids),
        )
        monkeypatch.setattr(
            refine_router, "refine_chunk_ids", lambda session, fid: [1]
        )
        started: list[str] = []
        monkeypatch.setattr(
            refine_router, "start_refine_job",
            lambda fid, ids, resolved: started.append(fid) or f"job-{fid}",
        )

        response = client.post(
            "/refine/folders",
            json={"drive": DRIVE, "file_ids": [FILE, "uncorrected1"]},
            headers=HEADERS,
        )

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["skipped_corrected"] == [FILE]
        assert started == ["uncorrected1"]


class TestConfirmedRegenerate:
    def test_a_confirmed_transcription_replaces_the_correction(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C9, I8."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        assert _confirmed(client).status_code in (200, 202)

        assert run_whisper(monkeypatch, FILE, _speech()) is True

        state = _get(client).json()
        assert state["corrected"] is False
        assert state["has_original"] is False
        assert all(chunk["text"] != "Bravo, fixed." for chunk in state["chunks"])
        assert state["chunks"][0]["text"] == "the quick brown fox"

    def test_a_failed_confirmed_transcription_keeps_the_correction(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C9."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        assert _confirmed(client).status_code in (200, 202)

        assert run_whisper(monkeypatch, FILE, _provider(error=FatalError("boom"))) is False

        state = _get(client).json()
        assert state["corrected"] is True
        assert state["has_original"] is True
        details = client.get(f"/files/{FILE}/index-details", headers=HEADERS).json()
        assert details["redo_ended_without_result"]["transcript"] is True
        # The confirmation is gone, so a plain reindex is refused again.
        assert client.post(
            f"/files/{FILE}/reindex", json={"tasks": ["whisper"]}, headers=HEADERS,
        ).status_code == 409

    def test_a_confirmed_transcription_that_finds_nothing_keeps_the_correction(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C9."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        assert _confirmed(client).status_code in (200, 202)

        assert run_whisper(monkeypatch, FILE, _provider(segments=[])) is True

        state = _get(client).json()
        assert state["corrected"] is True
        assert state["has_original"] is True
        details = client.get(f"/files/{FILE}/index-details", headers=HEADERS).json()
        assert details["redo_ended_without_result"]["transcript"] is True

    def test_a_confirmed_ocr_run_replaces_the_picture_text(
        self, search_db, api, monkeypatch, tmp_path
    ):
        """SPEC-ADDON-068 C9."""
        from tests.ocr_fixtures import install, picture

        client, _ = api
        path = picture(tmp_path / "slide.png")
        seed_picture(search_db, PICTURE, ["old text"])
        search_db.execute(
            "UPDATE indexed_files SET file_path = :p WHERE file_id = :f",
            p=str(path), f=PICTURE,
        )
        assert client.get(
            f"/files/{PICTURE}/picture-text", headers=HEADERS
        ).status_code == 200
        assert client.put(
            f"/files/{PICTURE}/picture-text",
            json={"text": "fixed", "expected_text": "old text"},
            headers=HEADERS,
        ).status_code == 200
        assert _confirmed(client, PICTURE, ("ocr",)).status_code in (200, 202)
        harness = install(monkeypatch, lines=["engine text"])

        harness.module.index_ocr(PICTURE)

        payload = client.get(
            f"/files/{PICTURE}/picture-text", headers=HEADERS
        ).json()
        assert payload["text"] == "engine text"
        assert payload["corrected"] is False
        assert payload["has_original"] is False

    @pytest.mark.parametrize(
        "engine_error,lines,expected_text",
        [
            (RuntimeError("engine failed"), ["never"], "fixed"),
            (None, [], "fixed"),
        ],
        ids=["engine-fails", "no-line"],
    )
    def test_a_failed_confirmed_ocr_run_keeps_the_picture_text(
        self, search_db, api, monkeypatch, tmp_path, engine_error, lines, expected_text
    ):
        """SPEC-ADDON-068 C9."""
        from tests.ocr_fixtures import install, picture

        client, _ = api
        path = picture(tmp_path / "slide.png")
        seed_picture(search_db, PICTURE, ["old text"])
        search_db.execute(
            "UPDATE indexed_files SET file_path = :p WHERE file_id = :f",
            p=str(path), f=PICTURE,
        )
        assert client.get(
            f"/files/{PICTURE}/picture-text", headers=HEADERS
        ).status_code == 200
        assert client.put(
            f"/files/{PICTURE}/picture-text",
            json={"text": "fixed", "expected_text": "old text"},
            headers=HEADERS,
        ).status_code == 200
        assert _confirmed(client, PICTURE, ("ocr",)).status_code in (200, 202)
        harness = install(monkeypatch, lines=lines, engine_error=engine_error)

        harness.module.index_ocr(PICTURE)

        payload = client.get(
            f"/files/{PICTURE}/picture-text", headers=HEADERS
        ).json()
        assert payload["text"] == expected_text
        assert payload["corrected"] is True
        assert payload["has_original"] is True
        details = client.get(
            f"/files/{PICTURE}/index-details", headers=HEADERS
        ).json()
        assert details["redo_ended_without_result"]["picture_text"] is True


class TestConfirmedRefine:
    def test_refine_stores_the_original_of_a_machine_transcript(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C11, I4."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        _fake_refine(monkeypatch, apply=True)

        asyncio.run(_run_refine(FILE))

        state = _get(client).json()
        assert state["has_original"] is True
        assert state["corrected"] is False

    def test_a_refine_that_applies_no_window_stores_nothing(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C11."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        _fake_refine(monkeypatch, apply=False)

        asyncio.run(_run_refine(FILE))

        assert _get(client).json()["has_original"] is False

    def test_refine_leaves_a_corrected_utterance_without_words_out_of_the_window(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C10, I9."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        corrected_id = search_db.scalar(
            "SELECT id FROM transcript_chunks WHERE file_id = :f AND chunk_index = 1",
            f=FILE,
        )
        before = search_db.chunks(FILE)
        captured: list[int] = []
        _fake_refine(monkeypatch, apply=True, captured=captured)

        asyncio.run(_run_refine(FILE))

        assert corrected_id not in captured
        after = search_db.chunks(FILE)
        corrected = [row for row in after if row[0] == 1][0]
        original = [row for row in before if row[0] == 1][0]
        assert corrected[1:4] == original[1:4]
        assert _get(client).json()["has_original"] is True

    def test_rechunking_keeps_a_corrected_utterance_without_words(
        self, search_db, api
    ):
        """SPEC-ADDON-068 C10."""
        from app.workers import refine

        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200

        with search_db.Session() as session:
            refine.rechunk_from_words(session, FILE, refined_model="m")
            session.commit()

        rows = search_db.chunks(FILE)
        assert any(row[1] == "Bravo, fixed." for row in rows)
        kept = [row for row in rows if row[1] == "Bravo, fixed."][0]
        assert kept[2:4] == (5.0, 10.0)
        state = _get(client).json()
        assert [chunk["index"] for chunk in state["chunks"]] == list(range(len(rows)))


class TestStoredResultEndsAnOriginal:
    def test_a_stored_transcription_deletes_a_refined_file_s_original(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C12, I4."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        _fake_refine(monkeypatch, apply=True)
        asyncio.run(_run_refine(FILE))
        assert _get(client).json()["has_original"] is True

        assert run_whisper(monkeypatch, FILE, _speech()) is True

        state = _get(client).json()
        assert state["has_original"] is False
        assert state["corrected"] is False
        assert state["chunks"][0]["text"] == "the quick brown fox"

    def test_the_zero_segment_success_deletes_nothing(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C12."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        _fake_refine(monkeypatch, apply=True)
        asyncio.run(_run_refine(FILE))
        assert _get(client).json()["has_original"] is True

        assert run_whisper(monkeypatch, FILE, _provider(segments=[])) is True

        assert _get(client).json()["has_original"] is True
