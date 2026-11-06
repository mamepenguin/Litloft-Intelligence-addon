"""Startup recovery and ordinary failures around a correction.

SPEC-ID: SPEC-ADDON-068
SPEC-ID: SPEC-ADDON-065

The tests drive the startup helpers and the workers directly, so they
see the marks and the queued work rather than only the HTTP answer.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

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
    HEADERS,
    add_running_job,
    run_whisper,
    seed_picture,
    seed_transcript,
)
from tests.ocr_fixtures import install, pillow_real  # noqa: F401
from tests.ocr_fixtures import picture

FILE = "startup001"
PICTURE = "startup002"


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


def _correct(client, file_id: str = FILE, index: int = 1):
    return client.put(
        f"/files/{file_id}/transcript/utterances/{index}",
        json={
            "text": "Bravo, fixed.",
            "expected_text": "bravo two",
            "expected_start": 5.0,
        },
        headers=HEADERS,
    )


def _confirmed(client, file_id: str, tasks):
    return client.post(
        f"/files/{file_id}/reindex",
        json={"tasks": list(tasks), "overwrite_corrections": True},
        headers=HEADERS,
    )


def _correct_picture(client):
    current = client.get(
        f"/files/{PICTURE}/picture-text", headers=HEADERS
    ).json()["text"]
    return client.put(
        f"/files/{PICTURE}/picture-text",
        json={"text": "fixed", "expected_text": current},
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


class TestStartupClearsConfirmedRedo:
    def test_every_redo_mark_ends_at_start_and_the_file_is_editable(
        self, search_db, api
    ):
        """SPEC-ADDON-068 C9/I8: a restart ends a confirmed run."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        seed_picture(search_db, PICTURE, ["old text"])
        assert _correct(client).status_code == 200
        assert client.get(
            f"/files/{PICTURE}/picture-text", headers=HEADERS
        ).status_code == 200
        assert _correct_picture(client).status_code == 200
        assert _confirmed(client, FILE, ("whisper",)).status_code in (200, 202)
        assert _confirmed(client, PICTURE, ("ocr",)).status_code in (200, 202)
        assert bool(search_db.flag(FILE, "transcript_redo")) is True
        assert bool(search_db.flag(PICTURE, "picture_text_redo")) is True

        from app import extraction

        extraction.clear_redo_marks_on_start()

        assert bool(search_db.flag(FILE, "transcript_redo")) is False
        assert bool(search_db.flag(FILE, "transcript_redo_ended")) is True
        assert bool(search_db.flag(PICTURE, "picture_text_redo")) is False
        assert bool(search_db.flag(PICTURE, "picture_text_redo_ended")) is True

        details = client.get(
            f"/files/{FILE}/index-details", headers=HEADERS
        ).json()
        assert details["redo_ended_without_result"]["transcript"] is True
        state = client.get(f"/files/{FILE}/transcript", headers=HEADERS).json()
        assert state["corrected"] is True
        assert state["has_original"] is True
        assert state["followups_pending"] is True
        assert client.put(
            f"/files/{FILE}/transcript/utterances/1",
            json={
                "text": "Bravo, again.",
                "expected_text": "Bravo, fixed.",
                "expected_start": 5.0,
            },
            headers=HEADERS,
        ).status_code == 200

        picture = client.get(
            f"/files/{PICTURE}/picture-text", headers=HEADERS
        ).json()
        assert picture["corrected"] is True
        assert picture["has_original"] is True


class TestOrdinaryFailuresLeaveNoRedoTrace:
    def test_a_failed_transcription_of_an_uncorrected_file_queues_nothing(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C9: only a confirmed run gets the ended mark."""
        client, _ = api
        seed_transcript(
            search_db, FILE, _chunks(), words=_words(),
            flags={"whisper_indexed": False},
        )
        provider = _provider(error=FatalError("boom"))

        assert run_whisper(monkeypatch, FILE, provider) is False

        assert bool(search_db.flag(FILE, "whisper_indexed")) is False
        assert bool(search_db.flag(FILE, "transcript_redo_ended")) is False
        with search_db.Session() as session:
            from app.models import FollowupItem

            assert session.query(FollowupItem).filter_by(file_id=FILE).count() == 0

    @pytest.mark.usefixtures("pillow_real")
    def test_a_missing_ocr_engine_leaves_no_redo_trace(
        self, search_db, tmp_path, monkeypatch
    ):
        """SPEC-ADDON-068 C9: an engine that cannot load is not a redo ending."""
        path = picture(tmp_path / "sign.png")
        from tests.extraction_fixtures import seed_file

        seed_file(
            search_db, PICTURE, file_type="image", mime_type="image/png",
            file_path=str(path), ocr_indexed=False,
        )
        harness = install(monkeypatch, engine_error=RuntimeError("missing"))

        assert harness.module.index_ocr(PICTURE) is False

        assert bool(search_db.flag(PICTURE, "ocr_indexed")) is False
        assert bool(search_db.flag(PICTURE, "picture_text_redo_ended")) is False
        with search_db.Session() as session:
            from app.models import FollowupItem

            assert session.query(FollowupItem).filter_by(file_id=PICTURE).count() == 0

    @pytest.mark.usefixtures("pillow_real")
    def test_a_failed_picture_embedding_leaves_no_redo_trace(
        self, search_db, tmp_path, monkeypatch
    ):
        """SPEC-ADDON-068 C9: a failed embedding is not a redo ending."""
        path = picture(tmp_path / "sign.png")
        from tests.extraction_fixtures import seed_file

        seed_file(
            search_db, PICTURE, file_type="image", mime_type="image/png",
            file_path=str(path), ocr_indexed=False,
        )
        harness = install(
            monkeypatch, lines=["engine text"],
            embed_error=RuntimeError("embed down"),
        )

        assert harness.module.index_ocr(PICTURE) is False

        assert bool(search_db.flag(PICTURE, "ocr_indexed")) is False
        assert bool(search_db.flag(PICTURE, "picture_text_redo_ended")) is False
        with search_db.Session() as session:
            from app.models import FollowupItem

            assert session.query(FollowupItem).filter_by(file_id=PICTURE).count() == 0


class TestEmbeddingModelRebuildRestoresExtractedText:
    def test_the_startup_sweep_rebuilds_corrected_search_rows(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C7/I6, decision 18."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        seed_picture(search_db, PICTURE, ["old text"])
        assert _correct(client).status_code == 200
        assert client.get(
            f"/files/{PICTURE}/picture-text", headers=HEADERS
        ).status_code == 200
        assert _correct_picture(client).status_code == 200
        chunks_before = search_db.chunks(FILE)
        words_before = search_db.words(FILE)
        picture_before = client.get(
            f"/files/{PICTURE}/picture-text", headers=HEADERS
        ).json()["text"]

        # The purge the text-embedding-model rebuild performs.
        search_db.execute(
            "DELETE FROM embeddings WHERE file_id IN (:t, :p) "
            "AND embedding_type IN ('whisper', 'text_content', 'tfidf_keywords')",
            t=FILE, p=PICTURE,
        )
        search_db.execute(
            "DELETE FROM fts_text_content WHERE file_id = :p", p=PICTURE
        )
        search_db.set_flag(FILE, "whisper_indexed", False)
        search_db.set_flag(FILE, "tfidf_keywords_indexed", False)
        search_db.set_flag(PICTURE, "ocr_indexed", False)

        from app import database
        from app.workers import followup

        database.text_embedding_model_rebuilt = True

        def _embed(texts):
            import numpy as np

            return [np.zeros(4, dtype=np.float32) for _ in texts]

        monkeypatch.setattr("app.workers.embedder.embed_passages", _embed)

        def _align(file_path, start, end, text, language):
            words = [
                {
                    "text": row[0],
                    "timestamp_start": row[1],
                    "timestamp_end": row[2],
                }
                for row in search_db.words(FILE)
                if row[1] >= start and row[2] <= end
            ]
            return "aligned", words

        monkeypatch.setattr(followup.aligner, "align_span", _align)
        monkeypatch.setattr(followup, "post_addon_event_sync", lambda *a, **k: True)
        calls: list[str] = []
        monkeypatch.setattr(
            "app.workers.whisper._index_tfidf_keywords",
            lambda fid, texts: calls.append(fid),
        )

        provider = _provider()
        with patch("app.workers.whisper.get_provider", return_value=provider):
            followup.startup_sweep()
            while (item_id := followup.next_due_item_id()) is not None:
                followup.process_item(item_id)

        provider.transcribe.assert_not_called()
        assert calls.count(FILE) >= 1
        assert bool(search_db.flag(FILE, "whisper_indexed")) is True
        assert bool(search_db.flag(PICTURE, "ocr_indexed")) is True
        assert search_db.chunks(FILE) == chunks_before
        assert search_db.words(FILE) == words_before
        assert client.get(
            f"/files/{PICTURE}/picture-text", headers=HEADERS
        ).json()["text"] == picture_before
        assert search_db.embeddings(FILE, "whisper")
        assert search_db.fts("fts_text_content", PICTURE)


class TestZeroSegmentSuccessClosesItsJob:
    def test_a_silent_file_marks_its_job_succeeded(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C12: the zero-segment path writes no transcript."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())

        assert run_whisper(monkeypatch, FILE, _provider(segments=[])) is True

        row = search_db.rows(
            "SELECT status, completed_at FROM job_records "
            "WHERE file_id = :f ORDER BY id DESC LIMIT 1",
            f=FILE,
        )[0]
        assert row[0] == "succeeded"
        assert row[1] is not None
