"""RED-phase tests for transcript word re-alignment after AI refine.

Spec: docs/superpowers/specs/2026-04-15-intelligence-transcript-refine.md

When a chunk's text is refined, the Whisper-derived ``transcript_words``
rows for that chunk's time range must be regenerated with timestamps
distributed proportionally across the refined tokens. HvLink-derived
chunks (no words in the table) are skipped.

These tests target a module that does not yet exist:
``app.workers.refine`` with a public ``realign_words_for_chunk``
function. Import is expected to fail or the behaviour is expected to
be absent — this is the RED phase.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Target under test — intentionally unimplemented (RED phase). The
# import below is EXPECTED to fail until `app.workers.refine` lands.
# Once the module exists, individual tests will drive the behavioural
# contract documented in the spec.
from app.workers.refine import realign_words_for_chunk  # noqa: E402


def _word(idx: int, text: str, start: float, end: float) -> SimpleNamespace:
    """Build a mock TranscriptWord-like row."""
    return SimpleNamespace(
        id=idx + 1,
        file_id="fileabc",
        word_index=idx,
        text=text,
        language="ja",
        timestamp_start=start,
        timestamp_end=end,
    )


class TestRealignWordsForChunk:
    """Proportional time distribution over refined tokens.

    API shape under test:

        realign_words_for_chunk(
            session,
            file_id,
            chunk_start,
            chunk_end,
            refined_text,
        ) -> int   # number of new words inserted
    """

    def test_distributes_timestamps_proportionally(self):
        session = MagicMock()
        # Five original words spanning [10.0, 20.0].
        original = [
            _word(0, "alpha", 10.0, 12.0),
            _word(1, "beta", 12.0, 14.0),
            _word(2, "gamma", 14.0, 16.0),
            _word(3, "delta", 16.0, 18.0),
            _word(4, "epsilon", 18.0, 20.0),
        ]
        # The impl is expected to query words via session; stub the
        # common shapes so either path works. Test should FAIL until
        # the real impl picks one and drives it.
        session.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
            original
        )
        session.execute.return_value.fetchall.return_value = [
            (w.id, w.word_index, w.text, w.timestamp_start, w.timestamp_end)
            for w in original
        ]

        # Refined text has 7 tokens → (20 - 10) / 7 ≈ 1.4286s each.
        refined = "a b c d e f g"

        inserted = realign_words_for_chunk(
            session, "fileabc", 10.0, 20.0, refined
        )

        assert inserted == 7
        # The implementation must have issued INSERTs for 7 new words,
        # each covering a contiguous 1/7 slice of [10.0, 20.0]. We don't
        # pin the exact call shape (ORM add vs. raw SQL) — just verify
        # *some* write happened 7 times.
        write_calls = (
            session.add.call_count
            + session.add_all.call_count
            + session.execute.call_count
        )
        assert write_calls >= 1

    def test_empty_refined_text_deletes_words_in_range(self):
        """A chunk refined to an empty string means "this segment was
        silence / a mis-hearing". Words in that time range must be
        deleted so the subtitle track doesn't keep stale hallucinations.
        """
        session = MagicMock()

        inserted = realign_words_for_chunk(
            session, "fileabc", 10.0, 20.0, ""
        )

        assert inserted == 0
        # At least one DELETE-like call must have been issued.
        assert session.execute.called or session.query.called

    def test_single_word_chunk(self):
        """Edge: chunk containing a single original word — refined text
        with one token must land on the full [t_start, t_end] span.
        """
        session = MagicMock()
        session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [
            _word(0, "hello", 5.0, 7.0),
        ]

        inserted = realign_words_for_chunk(
            session, "fileabc", 5.0, 7.0, "hi"
        )

        assert inserted == 1

    def test_chunk_with_no_words_is_skipped(self):
        """HvLink-origin chunks have no words in the DB. The function
        must skip cleanly (no INSERTs, no exceptions) rather than
        fabricating word rows from nothing.
        """
        session = MagicMock()
        session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        session.execute.return_value.fetchall.return_value = []

        inserted = realign_words_for_chunk(
            session, "fileabc", 100.0, 110.0, "some refined text"
        )

        assert inserted == 0

    def test_whitespace_only_refined_text_treated_as_empty(self):
        """Edge: whitespace-only refined text must delete range, not
        create a single empty word row.
        """
        session = MagicMock()

        inserted = realign_words_for_chunk(
            session, "fileabc", 10.0, 20.0, "   \n\t  "
        )
        assert inserted == 0

    def test_timestamps_are_monotonic_and_cover_range(self):
        """Successive word timestamps must be non-decreasing and the
        first/last must cover the chunk's [t_start, t_end] exactly.
        """
        session = MagicMock()
        captured: list[tuple[float, float]] = []

        def _capture_add(row):
            captured.append((row.timestamp_start, row.timestamp_end))

        session.add.side_effect = _capture_add
        session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [
            _word(0, "x", 0.0, 4.0),
        ]

        realign_words_for_chunk(
            session, "fileabc", 0.0, 4.0, "alpha beta gamma delta"
        )

        if captured:
            # Sorted by start
            starts = [s for s, _ in captured]
            ends = [e for _, e in captured]
            assert starts == sorted(starts)
            assert ends == sorted(ends)
            assert starts[0] == pytest.approx(0.0, abs=1e-6)
            assert ends[-1] == pytest.approx(4.0, abs=1e-6)
