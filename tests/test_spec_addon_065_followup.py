"""What the follow-up worker does with a saved correction.

SPEC-ID: SPEC-ADDON-065
SPEC-ID: SPEC-ADDON-068

The worker is the only thing that embeds a corrected utterance and
realigns its words, so these tests drain its stored items directly.
"""

from __future__ import annotations

import pytest

from tests.extraction_fixtures import (
    api,
    search_db,
    HEADERS,
    seed_transcript,
)

FILE = "followup001"


def _chunks() -> list[dict]:
    return [
        {"index": 0, "text": "alpha one", "start": 0.0, "end": 5.0},
        {"index": 1, "text": "bravo two", "start": 5.0, "end": 10.0},
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


def _drain() -> None:
    from app.workers import followup

    while (item_id := followup.next_due_item_id()) is not None:
        followup.process_item(item_id)


def _stub_work(monkeypatch, db, units: list[dict]):
    from app.workers import followup

    def _embed(texts):
        import numpy as np

        return [np.zeros(4, dtype=np.float32) for _ in texts]

    monkeypatch.setattr("app.workers.embedder.embed_passages", _embed)
    calls: list[str] = []
    keyword_calls: list[list[str]] = []

    def _keywords(file_id, texts):
        keyword_calls.append(list(texts))
        db.set_flag(file_id, "tfidf_keywords_indexed", True)

    monkeypatch.setattr(
        "app.workers.whisper._index_tfidf_keywords", _keywords
    )

    def _align(file_path, start, end, text, language):
        calls.append(text)
        return "aligned", units

    monkeypatch.setattr(followup.aligner, "align_span", _align)
    monkeypatch.setattr(followup, "post_addon_event_sync", lambda *a, **k: True)
    return calls, keyword_calls


class TestFollowupAppliesTheCorrection:
    def test_it_embeds_and_realigns_the_saved_utterance(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-065 A8/A9, SPEC-ADDON-068 C10."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client, 1, "Bravo, fixed.", "bravo two", 5.0).status_code == 200
        calls, keyword_calls = _stub_work(monkeypatch, search_db, [
            {"text": "Bravo,", "timestamp_start": 5.1, "timestamp_end": 5.6},
            {"text": "fixed", "timestamp_start": 5.7, "timestamp_end": 6.2},
        ])

        _drain()

        whisper = search_db.embeddings(FILE, "whisper")
        assert [(row[1], row[2]) for row in whisper if row[1] == 1] == [
            (1, "Bravo, fixed.")
        ]
        words = search_db.words(FILE)
        assert ("Bravo,", 5.1, 5.6, "en", None) in words
        assert ("fixed", 5.7, 6.2, "en", None) in words
        assert bool(search_db.flag(FILE, "tfidf_keywords_indexed")) is True
        assert keyword_calls and "Bravo, fixed." in keyword_calls[0]
        state = client.get(f"/files/{FILE}/transcript", headers=HEADERS).json()
        assert state["followups_pending"] is False
        assert state["chunks"][1]["wordsPending"] is False
        assert calls == ["Bravo, fixed."]

    def test_the_follow_up_keeps_the_neighbours_boundary_word(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-065 I1: the save keeps it, so its follow-up must too."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client, 1, "Bravo, fixed.", "bravo two", 5.0).status_code == 200
        _stub_work(monkeypatch, search_db, [
            {"text": "Bravo,", "timestamp_start": 5.1, "timestamp_end": 5.6},
        ])

        _drain()

        assert ("|", 5.0, 5.0, "en", None) in search_db.words(FILE)

    def test_two_saves_of_one_utterance_leave_only_the_latest(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-065 I2: a stale item writes nothing."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client, 1, "first text", "bravo two", 5.0).status_code == 200
        assert _correct(client, 1, "second text", "first text", 5.0).status_code == 200
        _calls, keyword_calls = _stub_work(monkeypatch, search_db, [])

        _drain()

        rows = search_db.embeddings(FILE, "whisper")
        assert [row[2] for row in rows if row[1] == 1] == ["second text"]

    def test_two_corrected_utterances_both_get_their_embedding(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-065 A8: another utterance's save does not make an item stale."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client, 0, "Alpha, fixed.", "alpha one", 0.0).status_code == 200
        assert _correct(
            client, 2, "Charlie, fixed.", "charlie three", 10.0
        ).status_code == 200
        _calls, keyword_calls = _stub_work(monkeypatch, search_db, [])

        _drain()

        rows = search_db.embeddings(FILE, "whisper")
        by_index = {row[1]: row[2] for row in rows}
        assert by_index[0] == "Alpha, fixed."
        assert by_index[2] == "Charlie, fixed."

    def test_an_empty_save_rebuilds_the_keyword_vector(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-065 A6/A8: an empty save still queues the keyword vector."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client, 1, "", "bravo two", 5.0).status_code == 200
        _calls, keyword_calls = _stub_work(monkeypatch, search_db, [])

        _drain()

        assert bool(search_db.flag(FILE, "tfidf_keywords_indexed")) is True
        assert keyword_calls
