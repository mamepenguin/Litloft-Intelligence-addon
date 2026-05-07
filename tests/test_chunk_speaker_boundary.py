"""Phase 1C tests: speaker change as a chunk boundary signal.

Spec ``2026-05-07-cloud-transcription-providers.md`` §"Speaker_id の
chunking" adds a 4th boundary condition to ``_build_chunks_from_words``:

1. Sentence-final punctuation (``.。!?！？…``)
2. Silence gap > 400 ms
3. min_duration / max_duration constraints
4. **NEW**: ``speaker_id`` differs from the previous word, when both
   are non-NULL

When all words have ``speaker_id=None`` (whisper_local /
openai_compatible) the new condition never fires and the legacy
behaviour is preserved.

Each emitted chunk also carries a ``speaker_id`` field — the speaker
of the words inside the chunk, picked by majority when a chunk
straddles speakers despite our boundary logic (defensive: should not
happen in practice but the build_chunks contract should not silently
discard the information).
"""

from __future__ import annotations

from app.workers.whisper import _build_chunks_from_words


def _word(
    text: str,
    start: float,
    end: float,
    *,
    language: str = "en",
    speaker_id: str | None = None,
) -> dict:
    return {
        "text": text,
        "start": start,
        "end": end,
        "language": language,
        "speaker_id": speaker_id,
    }


def test_speaker_change_creates_chunk_boundary_past_min() -> None:
    """A speaker change past min_duration must flush a chunk.

    Without the new R4 rule, this 11-word run-on (no punctuation, no
    gap) at duration 11s would stay a single chunk. With R4 the
    speaker change at index 5 forces a flush.
    """
    words = [
        _word("alpha", 0.0, 1.0, speaker_id="0"),
        _word("beta", 1.0, 2.0, speaker_id="0"),
        _word("gamma", 2.0, 3.0, speaker_id="0"),
        _word("delta", 3.0, 4.0, speaker_id="0"),
        _word("epsilon", 4.0, 11.0, speaker_id="0"),
        # Speaker change here, past min_duration=10.
        _word("zeta", 11.0, 12.0, speaker_id="1"),
        _word("eta", 12.0, 13.0, speaker_id="1"),
    ]
    chunks = _build_chunks_from_words(words, min_duration=10, max_duration=30)
    assert len(chunks) >= 2
    # The first chunk must end at the last "0" speaker word.
    assert chunks[0]["end"] == 11.0
    assert chunks[0]["speaker_id"] == "0"
    # The next chunk starts at the new speaker.
    assert chunks[1]["start"] == 11.0
    assert chunks[1]["speaker_id"] == "1"


def test_speaker_change_does_not_flush_below_min() -> None:
    """Below min_duration the chunk should not flush even on speaker change.

    The min_duration floor exists to keep chunks searchable; honouring
    it for sentence boundaries but ignoring it for speaker changes
    would produce 1-2 word chunks every time a question / answer
    alternates rapidly.
    """
    words = [
        _word("a", 0.0, 0.5, speaker_id="0"),
        _word("b", 0.5, 1.0, speaker_id="1"),
        _word("c", 1.0, 1.5, speaker_id="0"),
    ]
    chunks = _build_chunks_from_words(words, min_duration=10, max_duration=30)
    # All three words fold into a single chunk (duration 1.5s < min 10s).
    assert len(chunks) == 1


def test_all_speaker_id_none_preserves_legacy_behaviour() -> None:
    """When every word has speaker_id=None, R4 must not fire."""
    words = [
        _word("alpha", 0.0, 1.0),
        _word("beta", 1.0, 2.0),
        _word("gamma", 2.0, 11.0),  # past min_duration
        _word("delta", 11.0, 12.0),
        _word("epsilon", 12.0, 13.0),
    ]
    chunks = _build_chunks_from_words(words, min_duration=10, max_duration=30)
    # No punctuation, no gap, no speaker info → exactly one chunk.
    assert len(chunks) == 1
    # ``speaker_id`` field is present but None when nothing is tagged.
    assert chunks[0].get("speaker_id") is None


def test_first_word_speaker_id_does_not_trigger_boundary() -> None:
    """The very first word has no "previous" — speaker_id is just stored.

    Defensive: the loop must compare against the prior word, not raise
    or treat None→speaker_0 as a transition.
    """
    words = [
        _word("first", 0.0, 1.0, speaker_id="0"),
        _word("second", 1.0, 2.0, speaker_id="0"),
    ]
    chunks = _build_chunks_from_words(words, min_duration=10, max_duration=30)
    assert len(chunks) == 1
    assert chunks[0]["speaker_id"] == "0"


def test_chunk_speaker_id_is_majority_when_mixed() -> None:
    """Defensive: if a chunk somehow contains mixed speakers (e.g. a
    speaker switch under min_duration that did NOT trigger a flush),
    the chunk reports the majority speaker rather than dropping the
    field entirely.
    """
    # Three words with speaker 0, one with speaker 1, all under min_duration.
    words = [
        _word("a", 0.0, 0.5, speaker_id="0"),
        _word("b", 0.5, 1.0, speaker_id="0"),
        _word("c", 1.0, 1.5, speaker_id="0"),
        _word("d", 1.5, 2.0, speaker_id="1"),
    ]
    chunks = _build_chunks_from_words(words, min_duration=10, max_duration=30)
    assert len(chunks) == 1
    assert chunks[0]["speaker_id"] == "0"


def test_speaker_change_with_null_neighbour_does_not_flush() -> None:
    """A None→"0" transition must NOT count as a speaker change.

    Mixed-NULL streams should fall back to the legacy pre-1C path —
    the new condition only fires when both neighbours have speaker
    labels.
    """
    words = [
        _word("a", 0.0, 1.0, speaker_id=None),
        _word("b", 1.0, 2.0, speaker_id=None),
        _word("c", 2.0, 11.0, speaker_id=None),
        # Becomes labelled mid-stream — must NOT trigger a boundary.
        _word("d", 11.0, 12.0, speaker_id="0"),
    ]
    chunks = _build_chunks_from_words(words, min_duration=10, max_duration=30)
    assert len(chunks) == 1


def test_speaker_change_chunk_boundaries_are_contiguous() -> None:
    """When a speaker change flushes, the next chunk must start exactly
    where the previous ended (no gap, no overlap)."""
    words = [
        _word("alpha", 0.0, 1.0, speaker_id="0"),
        _word("beta", 1.0, 2.0, speaker_id="0"),
        _word("gamma", 2.0, 11.0, speaker_id="0"),  # past min
        _word("delta", 11.0, 12.0, speaker_id="1"),
        _word("epsilon", 12.0, 13.0, speaker_id="1"),
    ]
    chunks = _build_chunks_from_words(words, min_duration=10, max_duration=30)
    assert len(chunks) >= 2
    # Contiguous: previous chunk end == next chunk start.
    assert chunks[0]["end"] == chunks[1]["start"]
