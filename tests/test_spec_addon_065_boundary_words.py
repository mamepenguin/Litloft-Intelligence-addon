"""A saved correction removes its own words even on the chunk boundary.

SPEC-ID: SPEC-ADDON-065

`_build_chunks_from_words` makes a chunk start at its first word's start
and end at its last word's end, so these boundary timestamps are the
normal shape, not a contrived fixture.
"""

from __future__ import annotations

from tests.extraction_fixtures import api, search_db, HEADERS, seed_transcript

FILE = "boundary01"


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
        # The neighbour's boundary word, kept by the save.
        {"text": "|", "start": 5.0, "end": 5.0},
        # The utterance's own first and last words, exactly on its bounds.
        {"text": "bravo", "start": 5.0, "end": 6.0},
        {"text": "two", "start": 9.0, "end": 10.0},
        {"text": "charlie", "start": 10.2, "end": 11.0},
    ]


def test_the_utterances_boundary_words_are_removed(search_db, api):
    """SPEC-ADDON-065 A6, I1."""
    client, _ = api
    seed_transcript(search_db, FILE, _chunks(), words=_words())

    response = client.put(
        f"/files/{FILE}/transcript/utterances/1",
        json={
            "text": "Bravo, fixed.",
            "expected_text": "bravo two",
            "expected_start": 5.0,
        },
        headers=HEADERS,
    )

    assert response.status_code == 200, response.text
    words = search_db.words(FILE)
    assert ("bravo", 5.0, 6.0, "en", None) not in words
    assert ("two", 9.0, 10.0, "en", None) not in words
    assert ("|", 5.0, 5.0, "en", None) in words

    vtt = client.get(f"/files/{FILE}/subtitles.vtt", headers=HEADERS).text
    assert vtt.count("Bravo, fixed.") == 1
    assert "00:00:05.000 --> 00:00:10.000" in vtt
