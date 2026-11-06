"""Correct one transcript utterance (SPEC-ADDON-070).

SPEC-ID: SPEC-ADDON-070

The checks are made through ``PUT /files/{id}/transcript/utterances/{index}``
and the answers the panel reads, so the order of the checks, the coded error
bodies and the transaction's effects are all visible from outside.
"""

from __future__ import annotations

import pytest

from tests.extraction_fixtures import (
    api,
    search_db,
    DRIVE,
    HEADERS,
    OTHER_DRIVE,
    coded_detail,
    seed_file,
    seed_transcript,
)

FILE = "transcript01"
OTHER = "transcript02"


def _two_chunks() -> list[dict]:
    return [
        {"index": 0, "text": "alpha one", "start": 0.0, "end": 5.0, "speaker": "spk_0"},
        {"index": 1, "text": "bravo two", "start": 5.0, "end": 10.0, "speaker": "spk_1"},
        {"index": 2, "text": "charlie three", "start": 10.0, "end": 15.0},
    ]


def _words() -> list[dict]:
    return [
        {"text": "alpha", "start": 0.2, "end": 1.0},
        {"text": "one", "start": 4.5, "end": 5.0},
        # A zero-duration word exactly on the chunk 0 / chunk 1 boundary.
        {"text": "|", "start": 5.0, "end": 5.0},
        {"text": "bravo", "start": 5.2, "end": 6.0},
        {"text": "two", "start": 9.0, "end": 9.9},
        {"text": "charlie", "start": 10.2, "end": 11.0},
        {"text": "three", "start": 14.0, "end": 14.9},
    ]


def _url(file_id: str = FILE, index: int = 1) -> str:
    return f"/files/{file_id}/transcript/utterances/{index}"


def _body(text: str, expected_text: str = "bravo two", expected_start: float = 5.0) -> dict:
    return {
        "text": text,
        "expected_text": expected_text,
        "expected_start": expected_start,
    }


def _get(client, file_id: str = FILE, drive: str = DRIVE):
    return client.get(f"/files/{file_id}/transcript", headers={"X-Lit-Drive": drive})


class TestSaveChecks:
    @pytest.mark.parametrize(
        "seed,file_id,index,drive",
        [
            ("missing", "ghost0000001", 0, DRIVE),
            ("inactive", FILE, 1, DRIVE),
            ("other-drive", OTHER, 1, DRIVE),
            ("no-transcript", OTHER, 0, DRIVE),
        ],
        ids=["missing", "inactive", "other-drive", "no-transcript"],
    )
    def test_a_file_outside_the_request_writes_nothing(
        self, search_db, api, seed, file_id, index, drive
    ):
        client, _ = api
        if seed == "inactive":
            seed_file(search_db, FILE, active=False)
        elif seed == "other-drive":
            seed_transcript(search_db, OTHER, _two_chunks(), drive=OTHER_DRIVE)
        elif seed == "no-transcript":
            seed_file(search_db, OTHER)
        else:
            seed_transcript(search_db, FILE, _two_chunks())

        before = search_db.chunks(FILE) if seed in ("inactive", "missing") else []
        response = client.put(
            _url(file_id, index), json=_body("changed"), headers={"X-Lit-Drive": drive}
        )

        assert response.status_code == 404
        if before:
            assert search_db.chunks(FILE) == before

    def test_a_busy_file_answers_busy_before_checking_the_text(self, search_db, api):
        """SPEC-ADDON-070 A4: busy is checked before changed."""
        client, _ = api
        seed_transcript(search_db, FILE, _two_chunks())

        assert client.put(_url(), json=_body("fixed"), headers=HEADERS).status_code == 200
        confirmed = client.post(
            f"/files/{FILE}/reindex",
            json={"tasks": ["whisper"], "overwrite_corrections": True},
            headers=HEADERS,
        )
        assert confirmed.status_code in (200, 202)

        response = client.put(
            _url(), json=_body("different", expected_text="not the stored text"),
            headers=HEADERS,
        )

        assert response.status_code == 409
        assert coded_detail(response) == {"code": "busy"}
        assert _get(client).json()["busy"] is True

    @pytest.mark.parametrize(
        "expected_text,expected_start",
        [("wrong text", 5.0), ("bravo two", 7.0)],
        ids=["text", "start"],
    )
    def test_changed_text_or_start_writes_nothing(
        self, search_db, api, expected_text, expected_start
    ):
        client, _ = api
        seed_transcript(search_db, FILE, _two_chunks())
        before = search_db.chunks(FILE)
        before_fts = search_db.fts("fts_transcripts", FILE)

        response = client.put(
            _url(), json=_body("fixed", expected_text, expected_start), headers=HEADERS
        )

        assert response.status_code == 409
        assert coded_detail(response) == {"code": "changed"}
        assert search_db.chunks(FILE) == before
        assert search_db.fts("fts_transcripts", FILE) == before_fts
        assert _get(client).json()["has_original"] is False

    def test_a_line_break_is_rejected_before_the_length_cap(self, search_db, api):
        client, _ = api
        seed_transcript(search_db, FILE, _two_chunks())

        response = client.put(
            _url(), json=_body("line one\nline two" + "x" * 3000), headers=HEADERS
        )

        assert response.status_code == 422
        assert coded_detail(response) == {"code": "line_break"}
        assert search_db.chunks(FILE)[1][1] == "bravo two"

    def test_a_text_over_2000_code_points_is_too_long(self, search_db, api):
        client, _ = api
        seed_transcript(search_db, FILE, _two_chunks())

        response = client.put(_url(), json=_body("x" * 2001), headers=HEADERS)

        assert response.status_code == 422
        assert coded_detail(response) == {"code": "too_long"}
        assert search_db.chunks(FILE)[1][1] == "bravo two"

    def test_emptying_the_only_utterance_is_refused(self, search_db, api):
        client, _ = api
        seed_transcript(
            search_db,
            FILE,
            [{"index": 0, "text": "only", "start": 0.0, "end": 1.0}],
        )

        response = client.put(
            _url(index=0), json=_body("", expected_text="only", expected_start=0.0),
            headers=HEADERS,
        )

        assert response.status_code == 422
        assert coded_detail(response) == {"code": "last_utterance"}
        assert search_db.chunks(FILE) == [(0, "only", 0.0, 1.0, "en", None)]

    @pytest.mark.parametrize(
        "case,expected",
        [
            ("missing-before-text", 404),
            ("busy-before-changed", 409),
            ("changed-before-line-break", 409),
            ("line-break-before-too-long", 422),
            ("too-long-before-last-utterance", 422),
        ],
    )
    def test_the_checks_run_in_the_order_the_spec_fixes(self, search_db, api, case, expected):
        client, _ = api
        if case != "missing-before-text":
            seed_transcript(search_db, FILE, _two_chunks())
        if case == "busy-before-changed":
            client.put(_url(), json=_body("fixed"), headers=HEADERS)
            client.post(
                f"/files/{FILE}/reindex",
                json={"tasks": ["whisper"], "overwrite_corrections": True},
                headers=HEADERS,
            )
        if case == "too-long-before-last-utterance":
            search_db.execute(
                "DELETE FROM transcript_chunks WHERE file_id = :f AND chunk_index != 1",
                f=FILE,
            )
            search_db.execute(
                "DELETE FROM transcript_words WHERE file_id = :f", f=FILE
            )

        bad = {
            "missing-before-text": ("changed\ntext" + "x" * 3000, "nope", 0.0),
            "busy-before-changed": ("changed\ntext", "nope", 0.0),
            "changed-before-line-break": ("changed\ntext", "nope", 5.0),
            "line-break-before-too-long": ("changed\ntext" + "x" * 3000, "bravo two", 5.0),
            "too-long-before-last-utterance": (
                "x" * 3000, "bravo two", 5.0,
            ),
        }[case]
        response = client.put(
            _url(), json=_body(bad[0], bad[1], bad[2]), headers=HEADERS
        )

        assert response.status_code == expected, response.text


class TestNormalisation:
    def test_surrounding_whitespace_is_trimmed_before_the_no_op_comparison(
        self, search_db, api
    ):
        """SPEC-ADDON-070 A3/A5: the normalised text decides the no-op."""
        client, _ = api
        seed_transcript(search_db, FILE, _two_chunks())

        response = client.put(_url(), json=_body("  bravo two  "), headers=HEADERS)

        assert response.status_code == 200
        state = _get(client).json()
        assert state["has_original"] is False
        assert state["corrected"] is False
        assert search_db.chunks(FILE)[1][1] == "bravo two"

    def test_the_cap_is_counted_after_trimming_in_code_points(self, search_db, api):
        """SPEC-ADDON-070 A4: 2,000 code points fit, surrounding space does not count."""
        client, _ = api
        seed_transcript(search_db, FILE, _two_chunks())

        exactly = "  " + "界" * 2000 + "  "
        accepted = client.put(_url(), json=_body(exactly), headers=HEADERS)
        assert accepted.status_code == 200, accepted.text

        rejected = client.put(
            _url(), json=_body("界" * 2001, exactly), headers=HEADERS
        )
        assert rejected.status_code == 422
        assert coded_detail(rejected) == {"code": "too_long"}


class TestSaveEffects:
    def test_only_the_utterance_and_its_words_change(self, search_db, api):
        """SPEC-ADDON-070 A6/A7, I1."""
        client, _ = api
        seed_transcript(search_db, FILE, _two_chunks(), words=_words())
        before = search_db.chunks(FILE)
        before_words = search_db.words(FILE)

        response = client.put(_url(), json=_body("Bravo, fixed."), headers=HEADERS)

        assert response.status_code == 200, response.text
        updated = response.json()
        assert updated["index"] == 1
        assert updated["text"] == "Bravo, fixed."
        assert updated["correctedAt"]
        assert updated["wordsPending"] is True

        after = search_db.chunks(FILE)
        assert after[0] == before[0]
        assert after[2] == before[2]
        assert after[1][0] == 1
        assert after[1][1] == "Bravo, fixed."
        assert after[1][2:4] == (5.0, 10.0)
        assert after[1][4:] == (before[1][4], before[1][5])

        words = search_db.words(FILE)
        assert ("bravo", 5.2, 6.0, "en", None) not in words
        assert ("two", 9.0, 9.9, "en", None) not in words
        # The zero-duration boundary word belongs to the previous utterance.
        assert ("|", 5.0, 5.0, "en", None) in words
        assert before_words[0] in words
        assert before_words[-1] in words

        state = _get(client).json()
        assert state["has_original"] is True
        assert state["corrected"] is True
        assert state["followups_pending"] is True

    def test_the_keyword_rows_embeddings_and_term_set_follow_the_save(
        self, search_db, api
    ):
        """SPEC-ADDON-070 A6, I2."""
        client, _ = api
        seed_transcript(search_db, FILE, _two_chunks(), words=_words())

        assert client.put(_url(), json=_body("Bravo, fixed."), headers=HEADERS).status_code == 200

        for table in ("fts_transcripts", "fts_transcripts_word"):
            rows = search_db.fts(table, FILE)
            assert ("1", "Bravo, fixed.") in rows
            assert all(text != "bravo two" for _, text in rows)
            assert ("0", "alpha one") in rows
            assert ("2", "charlie three") in rows

        whisper = search_db.embeddings(FILE, "whisper")
        assert all(row[1] != 1 for row in whisper)
        assert {row[1] for row in whisper} == {0, 2}
        assert search_db.embeddings(FILE, "tfidf_keywords") == []
        assert bool(search_db.flag(FILE, "tfidf_keywords_indexed")) is False
        assert bool(search_db.flag(FILE, "idf_terms_indexed")) is False

    def test_an_empty_text_removes_the_row_and_leaves_the_gap(self, search_db, api):
        """SPEC-ADDON-070 A6: indexes keep their values, so the gap is visible."""
        client, _ = api
        seed_transcript(search_db, FILE, _two_chunks(), words=_words())

        response = client.put(_url(), json=_body("   "), headers=HEADERS)

        assert response.status_code == 200, response.text
        assert [row[0] for row in search_db.chunks(FILE)] == [0, 2]
        assert all(idx != "1" for idx, _ in search_db.fts("fts_transcripts", FILE))
        words = search_db.words(FILE)
        assert ("bravo", 5.2, 6.0, "en", None) not in words
        assert ("|", 5.0, 5.0, "en", None) in words

        state = _get(client).json()
        assert state["has_original"] is True
        assert state["corrected"] is True
        # The empty save still queues the keyword-vector item.
        assert state["followups_pending"] is True

    def test_saving_the_same_text_writes_nothing(self, search_db, api):
        """SPEC-ADDON-070 A5."""
        client, _ = api
        seed_transcript(search_db, FILE, _two_chunks(), words=_words())
        before = search_db.chunks(FILE)
        before_fts = search_db.fts("fts_transcripts", FILE)
        before_embeddings = search_db.embeddings(FILE, "whisper")

        response = client.put(_url(), json=_body("bravo two"), headers=HEADERS)

        assert response.status_code == 200
        assert response.json()["text"] == "bravo two"
        assert search_db.chunks(FILE) == before
        assert search_db.fts("fts_transcripts", FILE) == before_fts
        assert search_db.embeddings(FILE, "whisper") == before_embeddings
        state = _get(client).json()
        assert state["has_original"] is False
        assert state["corrected"] is False
        assert state["followups_pending"] is False


class TestTranscriptAnswer:
    def test_the_panel_receives_the_fields_it_branches_on(self, search_db, api):
        """SPEC-ADDON-070 D."""
        client, _ = api
        seed_transcript(search_db, FILE, _two_chunks(), words=_words())

        before = _get(client)
        assert before.status_code == 200
        payload = before.json()
        for field in ("has_original", "corrected", "busy", "followups_pending"):
            assert field in payload
        for chunk in payload["chunks"]:
            assert "correctedAt" in chunk
            assert "wordsPending" in chunk

        client.put(_url(), json=_body("Bravo, fixed."), headers=HEADERS)
        after = _get(client).json()
        row = after["chunks"][1]
        assert row["correctedAt"]
        assert row["wordsPending"] is True


class TestSubtitlesForACorrectedUtterance:
    def test_a_corrected_utterance_without_words_is_one_cue(
        self, search_db, api
    ):
        """SPEC-ADDON-070 A12, I3."""
        client, _ = api
        seed_transcript(search_db, FILE, _two_chunks(), words=_words())
        client.put(_url(), json=_body("Bravo, fixed."), headers=HEADERS)

        response = client.get(f"/files/{FILE}/subtitles.vtt", headers=HEADERS)

        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "private, no-cache"
        assert "00:00:05.000 --> 00:00:10.000" in response.text
        assert "Bravo, fixed." in response.text
        assert "bravo" not in response.text

    def test_a_file_with_no_word_rows_keeps_answering_404(self, search_db, api):
        """SPEC-ADDON-070 A12: a VTT-based or pre-word file is unchanged."""
        client, _ = api
        seed_transcript(search_db, FILE, _two_chunks(), words=[])
        client.put(_url(), json=_body("Bravo, fixed."), headers=HEADERS)

        response = client.get(f"/files/{FILE}/subtitles.vtt", headers=HEADERS)

        assert response.status_code == 404

    def test_an_uncorrected_utterance_without_words_gets_no_cue(
        self, search_db, api
    ):
        """SPEC-ADDON-070 A12: only a corrected empty-words utterance falls back."""
        client, _ = api
        seed_transcript(
            search_db,
            FILE,
            _two_chunks(),
            words=[
                {"text": "alpha", "start": 0.2, "end": 1.0},
                {"text": "one", "start": 4.5, "end": 5.0},
            ],
        )

        response = client.get(f"/files/{FILE}/subtitles.vtt", headers=HEADERS)

        assert response.status_code == 200
        assert "alpha" in response.text
        assert "one" in response.text
        assert "bravo" not in response.text
