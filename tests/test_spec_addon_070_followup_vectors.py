"""The follow-up worker's embeddings, retries and parked items.

SPEC-ID: SPEC-ADDON-070
SPEC-ID: SPEC-ADDON-068
"""

from __future__ import annotations

import numpy as np
import pytest

from app.models import FollowupItem, PictureText
from tests.extraction_fixtures import (
    api,
    search_db,
    HEADERS,
    seed_picture,
    seed_transcript,
)

FILE = "vectors001"
OTHER = "vectors002"
PICTURE = "vectors003"


def _chunks() -> list[dict]:
    return [
        {"index": 0, "text": "alpha one", "start": 0.0, "end": 5.0},
        {"index": 1, "text": "bravo two", "start": 5.0, "end": 10.0},
    ]


def _words() -> list[dict]:
    return [
        {"text": "alpha", "start": 0.2, "end": 1.0},
        {"text": "bravo", "start": 5.2, "end": 6.0},
    ]


def _stub(monkeypatch, db, outcome="aligned"):
    from app.workers import followup

    def _embed(texts):
        return [np.zeros(4, dtype=np.float32) for _ in texts]

    monkeypatch.setattr("app.workers.embedder.embed_passages", _embed)
    monkeypatch.setattr(
        "app.workers.whisper._index_tfidf_keywords",
        lambda fid, texts: db.set_flag(fid, "tfidf_keywords_indexed", True),
    )
    monkeypatch.setattr(
        followup.aligner,
        "align_span",
        lambda *a: (
            ("aligned", [{"text": "fixed", "timestamp_start": 5.1, "timestamp_end": 5.6}])
            if outcome == "aligned"
            else (outcome, None)
        ),
    )
    monkeypatch.setattr(followup, "post_addon_event_sync", lambda *a, **k: True)


def _drain():
    from app.workers import followup

    while (item_id := followup.next_due_item_id()) is not None:
        followup.process_item(item_id)


def _correct(client, file_id: str = FILE):
    return client.put(
        f"/files/{file_id}/transcript/utterances/1",
        json={"text": "fixed", "expected_text": "bravo two", "expected_start": 5.0},
        headers=HEADERS,
    )


class TestPictureVector:
    def test_a_picture_follow_up_leaves_its_embeddings(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-070 B7: the corrected text is embedded."""
        client, _ = api
        seed_picture(search_db, PICTURE, ["old text"])
        assert client.get(
            f"/files/{PICTURE}/picture-text", headers=HEADERS
        ).status_code == 200
        assert client.put(
            f"/files/{PICTURE}/picture-text",
            json={"text": "new words", "expected_text": "old text"},
            headers=HEADERS,
        ).status_code == 200
        _stub(monkeypatch, search_db)

        _drain()

        rows = search_db.embeddings(PICTURE, "text_content")
        assert [row[2] for row in rows] == ["new words"]


class TestRestoreVector:
    def test_a_restore_follow_up_leaves_the_transcript_vectors(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-070 C4: restore re-embeds the restored text."""
        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        assert client.post(
            f"/files/{FILE}/transcript/restore", headers=HEADERS
        ).status_code == 200
        _stub(monkeypatch, search_db)

        _drain()

        assert search_db.embeddings(FILE, "whisper")
        assert bool(search_db.flag(FILE, "tfidf_keywords_indexed")) is True

    def test_a_restore_follow_up_leaves_the_picture_vectors(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-070 C4 (picture)."""
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
        assert client.post(
            f"/files/{PICTURE}/picture-text/restore", headers=HEADERS
        ).status_code == 200
        _stub(monkeypatch, search_db)

        _drain()

        assert search_db.embeddings(PICTURE, "text_content")


class TestStartupSweepPictures:
    def _seed_uncorrected_picture(self, search_db) -> None:
        seed_picture(search_db, PICTURE, ["machine text"], ocr_indexed=False)
        with search_db.Session() as session:
            session.add(PictureText(file_id=PICTURE, text="machine text"))
            session.commit()
        search_db.execute(
            "DELETE FROM embeddings WHERE file_id = :f", f=PICTURE
        )
        search_db.execute(
            "DELETE FROM fts_text_content WHERE file_id = :f", f=PICTURE
        )

    def test_the_start_after_a_model_rebuild_rebuilds_a_picture(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-070 C7, decision 18."""
        from app import database
        from app.workers import followup

        self._seed_uncorrected_picture(search_db)
        database.text_embedding_model_rebuilt = True
        _stub(monkeypatch, search_db)

        followup.startup_sweep()
        _drain()

        assert bool(search_db.flag(PICTURE, "ocr_indexed")) is True
        assert search_db.embeddings(PICTURE, "text_content")
        assert search_db.fts("fts_text_content", PICTURE)

    def test_without_the_rebuild_a_reset_picture_is_not_swept(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-070 C7: a reindex reset is not silently rebuilt."""
        from app import database
        from app.workers import followup

        self._seed_uncorrected_picture(search_db)
        database.text_embedding_model_rebuilt = False

        followup.startup_sweep()

        with search_db.Session() as session:
            assert (
                session.query(FollowupItem)
                .filter_by(file_id=PICTURE, item_type="rebuild")
                .count()
                == 0
            )


class TestAttempts:
    def test_a_load_failure_gives_up_after_three_tries(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-070 A10: the third failed attempt ends with no words."""
        from app.workers import followup

        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        _stub(monkeypatch, search_db, "load_failed")

        for _ in range(3):
            search_db.execute(
                "UPDATE followup_items SET last_attempt_at = NULL"
            )
            item = followup.next_due_item_id()
            assert item is not None
            followup.process_item(item)

        assert search_db.rows(
            "SELECT count(*) FROM followup_items WHERE file_id = :f", f=FILE
        )[0][0] == 0
        row = search_db.rows(
            "SELECT words_pending, words_pending_attempts "
            "FROM transcript_chunks WHERE file_id = :f AND chunk_index = 1",
            f=FILE,
        )[0]
        assert row == (0, 0)


class TestInactiveFile:
    def test_an_inactive_file_does_not_block_the_other_items(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-070 A8: both corrections get their work done."""
        from app.workers import followup

        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        seed_transcript(search_db, OTHER, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        assert _correct(client, OTHER).status_code == 200
        _stub(monkeypatch, search_db)
        search_db.execute(
            "UPDATE indexed_files SET active = 0 WHERE file_id = :f", f=FILE
        )

        seen: list[int] = []
        for _ in range(3):
            item = followup.next_due_item_id()
            if item is None:
                break
            seen.append(item)
            followup.process_item(item)

        assert len(set(seen)) > 1
        assert search_db.embeddings(OTHER, "whisper")

        # A recovered file's parked item is still finished.
        search_db.execute(
            "UPDATE indexed_files SET active = 1 WHERE file_id = :f", f=FILE
        )
        search_db.execute(
            "UPDATE followup_items SET last_attempt_at = NULL WHERE file_id = :f",
            f=FILE,
        )
        while (item := followup.next_due_item_id()) is not None:
            followup.process_item(item)
        assert search_db.embeddings(FILE, "whisper")
        with search_db.Session() as session:
            assert (
                session.query(FollowupItem).filter_by(file_id=FILE).count() == 0
            )

    def test_a_retry_rebuilds_the_keywords_but_does_not_announce(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-070 A10/I15: a retry is not an ending."""
        from app.workers import followup

        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        events: list[dict] = []
        _stub(monkeypatch, search_db, "load_failed")
        monkeypatch.setattr(
            followup,
            "post_addon_event_sync",
            lambda event, data: events.append({"event": event, **data}),
        )

        item = followup.next_due_item_id()
        followup.process_item(item)

        assert bool(search_db.flag(FILE, "tfidf_keywords_indexed")) is True
        assert events == []
        with search_db.Session() as session:
            assert (
                session.query(FollowupItem).filter_by(file_id=FILE).count() == 1
            )

    def test_an_aligned_follow_up_announces_once(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-070 I15: one corrected event after the words exist."""
        from app.workers import followup

        client, _ = api
        seed_transcript(search_db, FILE, _chunks(), words=_words())
        assert _correct(client).status_code == 200
        events: list[dict] = []
        _stub(monkeypatch, search_db, "aligned")
        monkeypatch.setattr(
            followup,
            "post_addon_event_sync",
            lambda event, data: events.append({"event": event, **data}),
        )

        item = followup.next_due_item_id()
        followup.process_item(item)

        assert [e["reason"] for e in events if e["event"].endswith(".updated")] == [
            "corrected"
        ]
        assert ("fixed", 5.1, 5.6, "en", None) in search_db.words(FILE)
