"""The one kept original and restore (SPEC-ADDON-068).

SPEC-ID: SPEC-ADDON-068

Restore is exercised through
``POST /files/{id}/transcript/restore`` and
``POST /files/{id}/picture-text/restore``; what the original held is read
back by restoring and by the answers the panel gets.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from tests.extraction_fixtures import (
    api,
    search_db,
    DRIVE,
    HEADERS,
    OTHER_DRIVE,
    add_running_job,
    build_api,
    coded_detail,
    seed_file,
    seed_picture,
    seed_transcript,
)

FILE = "original0001"
PICTURE = "original0002"


def _transcript() -> list[dict]:
    return [
        {"index": 0, "text": "alpha one", "start": 0.0, "end": 5.0, "speaker": "spk_0"},
        {"index": 1, "text": "bravo two", "start": 5.0, "end": 10.0, "speaker": "spk_1"},
        {"index": 2, "text": "charlie three", "start": 10.0, "end": 15.0},
    ]


def _words() -> list[dict]:
    return [
        {"text": "alpha", "start": 0.2, "end": 1.0, "speaker": "spk_0"},
        {"text": "one", "start": 4.5, "end": 5.0, "speaker": "spk_0"},
        {"text": "bravo", "start": 5.2, "end": 6.0, "speaker": "spk_1"},
        {"text": "two", "start": 9.0, "end": 9.9, "speaker": "spk_1"},
        {"text": "charlie", "start": 10.2, "end": 11.0},
    ]


def _correct(client, file_id: str = FILE, index: int = 1, text: str = "Bravo, fixed."):
    expected_text = "bravo two" if index == 1 else "charlie three"
    expected_start = 5.0 if index == 1 else 10.0
    return client.put(
        f"/files/{file_id}/transcript/utterances/{index}",
        json={
            "text": text,
            "expected_text": expected_text,
            "expected_start": expected_start,
        },
        headers=HEADERS,
    )


def _restore(client, file_id: str = FILE):
    return client.post(f"/files/{file_id}/transcript/restore", headers=HEADERS)


def _get(client, file_id: str = FILE):
    return client.get(f"/files/{file_id}/transcript", headers=HEADERS)


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


class TestOriginalStorage:
    def test_the_original_is_the_state_immediately_before_the_first_write(
        self, search_db, api
    ):
        """SPEC-ADDON-068 C1, I4."""
        client, _ = api
        seed_transcript(search_db, FILE, _transcript(), words=_words())
        before_chunks = search_db.chunks(FILE)
        before_words = search_db.words(FILE)

        assert _correct(client).status_code == 200

        state = _get(client).json()
        assert state["has_original"] is True
        assert state["corrected"] is True

        second = client.put(
            f"/files/{FILE}/transcript/utterances/1",
            json={
                "text": "Bravo, fixed again.",
                "expected_text": "Bravo, fixed.",
                "expected_start": 5.0,
            },
            headers=HEADERS,
        )
        assert second.status_code == 200, second.text
        assert _get(client).json()["has_original"] is True

        restored = _restore(client)
        assert restored.status_code == 200, restored.text
        assert search_db.chunks(FILE) == before_chunks
        assert search_db.words(FILE) == before_words

    def test_restore_clears_the_correction_and_the_original(self, search_db, api):
        """SPEC-ADDON-068 C4/C5, I5."""
        client, _ = api
        seed_transcript(search_db, FILE, _transcript(), words=_words())
        before_chunks = search_db.chunks(FILE)
        before_words = search_db.words(FILE)
        assert _correct(client).status_code == 200

        response = _restore(client)

        assert response.status_code == 200, response.text
        assert search_db.chunks(FILE) == before_chunks
        assert search_db.words(FILE) == before_words

        state = _get(client).json()
        assert state["corrected"] is False
        assert state["has_original"] is False
        for chunk in state["chunks"]:
            assert chunk["correctedAt"] is None
            assert chunk["wordsPending"] is False

    def test_restore_rewrites_the_keyword_rows_and_removes_vectors(self, search_db, api):
        """SPEC-ADDON-068 C4, I2/I5."""
        client, _ = api
        seed_transcript(search_db, FILE, _transcript(), words=_words())
        assert _correct(client).status_code == 200

        assert _restore(client).status_code == 200

        for table in ("fts_transcripts", "fts_transcripts_word"):
            rows = search_db.fts(table, FILE)
            assert ("1", "bravo two") in rows
            assert all(text != "Bravo, fixed." for _, text in rows)
        assert search_db.embeddings(FILE, "whisper") == []
        assert search_db.embeddings(FILE, "tfidf_keywords") == []
        assert bool(search_db.flag(FILE, "tfidf_keywords_indexed")) is False
        assert bool(search_db.flag(FILE, "idf_terms_indexed")) is False

    def test_a_correction_after_restore_stores_a_new_original(self, search_db, api):
        """SPEC-ADDON-068 C5: the file is in the machine state again."""
        client, _ = api
        seed_transcript(search_db, FILE, _transcript(), words=_words())
        machine = search_db.chunks(FILE)
        assert _correct(client).status_code == 200
        assert _restore(client).status_code == 200

        assert _correct(client, index=2, text="Charlie, fixed.").status_code == 200
        assert _get(client).json()["has_original"] is True
        assert _restore(client).status_code == 200
        assert search_db.chunks(FILE) == machine

    def test_restore_emits_the_transcript_event_for_the_drive(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-068 C4, I15."""
        captured: list[dict[str, Any]] = []
        _capture_posts(monkeypatch, captured)
        client, _ = api
        seed_transcript(search_db, FILE, _transcript(), words=_words())
        assert _correct(client).status_code == 200

        assert _restore(client).status_code == 200

        events = [
            body for body in captured
            if body.get("event") == "intelligence.transcript.updated"
        ]
        assert events, captured
        assert events[-1]["drive"] == DRIVE
        assert events[-1]["data"]["file_id"] == FILE
        assert events[-1]["data"]["reason"] == "restored"


class TestRestoreChecks:
    @pytest.mark.parametrize(
        "case",
        ["missing", "inactive", "other-drive", "no-original"],
    )
    def test_a_missing_file_or_original_answers_404(self, search_db, api, case):
        client, _ = api
        if case == "inactive":
            seed_transcript(search_db, FILE, _transcript(), flags={"active": False})
        elif case == "other-drive":
            seed_transcript(search_db, FILE, _transcript(), drive=OTHER_DRIVE)
        elif case == "no-original":
            seed_transcript(search_db, FILE, _transcript())

        response = _restore(client)

        assert response.status_code == 404

    def test_a_busy_file_answers_busy(self, search_db, api):
        """SPEC-ADDON-068 C3."""
        client, _ = api
        seed_transcript(search_db, FILE, _transcript(), words=_words())
        assert _correct(client).status_code == 200
        assert client.post(
            f"/files/{FILE}/reindex",
            json={"tasks": ["whisper"], "overwrite_corrections": True},
            headers=HEADERS,
        ).status_code in (200, 202)

        response = _restore(client)

        assert response.status_code == 409
        assert coded_detail(response) == {"code": "busy"}
        assert _get(client).json()["has_original"] is True

    def test_a_restore_that_raises_changes_nothing(self, search_db, monkeypatch):
        """SPEC-ADDON-068 failure table C4/C6."""
        client, _ = build_api(monkeypatch, raise_server_exceptions=False)
        seed_transcript(search_db, FILE, _transcript(), words=_words())
        assert _correct(client).status_code == 200
        corrected = search_db.chunks(FILE)
        search_db.execute(
            "CREATE TRIGGER fail_restore_insert BEFORE INSERT ON transcript_chunks "
            "BEGIN SELECT RAISE(ABORT, 'insert failed'); END"
        )
        search_db.execute(
            "CREATE TRIGGER fail_restore_update BEFORE UPDATE ON transcript_chunks "
            "BEGIN SELECT RAISE(ABORT, 'update failed'); END"
        )

        response = _restore(client)

        assert response.status_code == 500
        assert search_db.chunks(FILE) == corrected
        state = _get(client).json()
        assert state["has_original"] is True
        assert state["corrected"] is True


class TestPictureRestore:
    def _get_picture(self, client):
        return client.get(f"/files/{PICTURE}/picture-text", headers=HEADERS)

    def _correct_picture(self, client, text: str = "fixed"):
        current = self._get_picture(client).json()["text"]
        return client.put(
            f"/files/{PICTURE}/picture-text",
            json={"text": text, "expected_text": current},
            headers=HEADERS,
        )

    def test_picture_restore_puts_the_original_text_back(self, search_db, api):
        """SPEC-ADDON-068 C4."""
        client, _ = api
        seed_picture(search_db, PICTURE, ["the machine text"])
        assert self._get_picture(client).json()["text"] == "the machine text"
        assert self._correct_picture(client, "fixed").status_code == 200

        response = client.post(
            f"/files/{PICTURE}/picture-text/restore", headers=HEADERS
        )

        assert response.status_code == 200, response.text
        payload = self._get_picture(client).json()
        assert payload["text"] == "the machine text"
        assert payload["has_original"] is False
        assert payload["corrected"] is False

    def test_a_picture_with_an_empty_current_text_stays_restorable(self, search_db, api):
        """SPEC-ADDON-068 C2/B7."""
        client, _ = api
        seed_picture(search_db, PICTURE, ["the machine text"])
        assert self._get_picture(client).status_code == 200
        assert self._correct_picture(client, "").status_code == 200

        visible = self._get_picture(client)
        assert visible.status_code == 200
        assert visible.json()["text"] == ""
        assert visible.json()["has_original"] is True

        assert client.post(
            f"/files/{PICTURE}/picture-text/restore", headers=HEADERS
        ).status_code == 200
        assert self._get_picture(client).json()["text"] == "the machine text"

    def test_picture_restore_emits_the_picture_event(self, search_db, api, monkeypatch):
        """SPEC-ADDON-068 C4, I15."""
        captured: list[dict[str, Any]] = []
        _capture_posts(monkeypatch, captured)
        client, _ = api
        seed_picture(search_db, PICTURE, ["the machine text"])
        assert self._get_picture(client).status_code == 200
        assert self._correct_picture(client, "fixed").status_code == 200

        assert client.post(
            f"/files/{PICTURE}/picture-text/restore", headers=HEADERS
        ).status_code == 200

        events = [
            body for body in captured
            if body.get("event") == "intelligence.picture_text.updated"
        ]
        assert events, captured
        assert events[-1]["drive"] == DRIVE
        assert events[-1]["data"]["reason"] == "restored"


class TestIndexDetailsMarks:
    def test_index_details_reports_corrected_per_kind(self, search_db, api):
        """SPEC-ADDON-068 C8/D."""
        client, _ = api
        seed_transcript(search_db, FILE, _transcript(), words=_words())
        seed_picture(search_db, PICTURE, ["picture text"])
        assert _correct(client).status_code == 200

        payload = client.get(
            f"/files/{FILE}/index-details", headers=HEADERS
        ).json()

        assert payload["corrected"] == {
            "transcript": True,
            "picture_text": False,
        }
        assert payload["redo_ended_without_result"] == {
            "transcript": False,
            "picture_text": False,
        }
