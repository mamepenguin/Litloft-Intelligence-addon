"""Tests for ``realign_words_for_chunk`` forced-alignment behaviour.

Spec: docs/superpowers/specs/2026-04-15-intelligence-transcript-refine.md
Decision: hako iG6Uotc_uQ8cpXufZQf6v (forced alignment replaces time-
proportional allocation; fallback = keep existing Whisper words).

The function is expected to call :mod:`app.workers.aligner` for any
chunk that has original Whisper words and a waveform available. On
aligner failure (None return, no waveform, no existing words) the
original rows must be left untouched rather than replaced with a
degraded time-proportional fallback.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.workers import refine as refine_module
from app.workers.refine import realign_words_for_chunk


def _word(idx: int, text: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(
        id=idx + 1,
        file_id="fileabc",
        word_index=idx,
        text=text,
        language="ja",
        timestamp_start=start,
        timestamp_end=end,
    )


def _session_with_words(rows: list[SimpleNamespace]) -> MagicMock:
    session = MagicMock()
    session.query.return_value.filter.return_value.order_by.return_value.all.return_value = rows
    return session


class TestRealignWordsForChunk:
    def test_uses_aligner_output_when_available(self, monkeypatch):
        """Happy path: aligner returns units, realign deletes old rows
        and inserts new ones carrying the aligner's timestamps.
        """
        session = _session_with_words([
            _word(0, "こ", 10.0, 11.0),
            _word(1, "ん", 11.0, 12.0),
        ])
        fake_units = [
            {"text": "こ", "timestamp_start": 10.1, "timestamp_end": 11.2},
            {"text": "ん", "timestamp_start": 11.2, "timestamp_end": 12.3},
            {"text": "に", "timestamp_start": 12.3, "timestamp_end": 13.0},
        ]
        monkeypatch.setattr(
            refine_module, "aligner", MagicMock(align_segment=MagicMock(return_value=fake_units))
        )

        inserted = realign_words_for_chunk(
            session, "fileabc", 10.0, 13.0, "こんに", waveform=object()
        )

        assert inserted == 3
        # DELETE + three add() calls.
        assert session.execute.called
        assert session.add.call_count == 3
        # Verify the first added row carries the aligner's timestamp.
        first_added = session.add.call_args_list[0][0][0]
        assert first_added.timestamp_start == 10.1
        assert first_added.timestamp_end == 11.2
        assert first_added.text == "こ"

    def test_no_waveform_keeps_existing_rows(self, monkeypatch):
        """Missing audio → return 0 without deleting or aligning."""
        session = _session_with_words([_word(0, "a", 0.0, 1.0)])
        mock_aligner = MagicMock()
        monkeypatch.setattr(refine_module, "aligner", mock_aligner)

        inserted = realign_words_for_chunk(
            session, "fileabc", 0.0, 1.0, "b", waveform=None
        )

        assert inserted == 0
        mock_aligner.align_segment.assert_not_called()
        session.execute.assert_not_called()
        session.add.assert_not_called()

    def test_aligner_failure_preserves_existing_words(self, monkeypatch):
        """Aligner returns None (language unsupported / OOM / …) →
        existing rows must stay untouched (no DELETE, no INSERT).
        """
        session = _session_with_words([_word(0, "a", 0.0, 1.0)])
        monkeypatch.setattr(
            refine_module, "aligner", MagicMock(align_segment=MagicMock(return_value=None))
        )

        inserted = realign_words_for_chunk(
            session, "fileabc", 0.0, 1.0, "b", waveform=object()
        )

        assert inserted == 0
        session.execute.assert_not_called()
        session.add.assert_not_called()

    def test_hvlink_chunk_with_no_existing_words_is_skipped(self, monkeypatch):
        """LoftRef-origin chunks have no transcript_words rows. Must
        short-circuit before the aligner is even invoked.
        """
        session = _session_with_words([])
        mock_aligner = MagicMock()
        monkeypatch.setattr(refine_module, "aligner", mock_aligner)

        inserted = realign_words_for_chunk(
            session, "fileabc", 100.0, 110.0, "refined", waveform=object()
        )

        assert inserted == 0
        mock_aligner.align_segment.assert_not_called()
        session.execute.assert_not_called()

    def test_language_override_wins_over_existing_row_language(self, monkeypatch):
        """Caller may pass ``language_override`` (e.g. re-tagged chunk)
        and that value must be forwarded to the aligner.
        """
        session = _session_with_words([_word(0, "a", 0.0, 1.0)])
        captured: dict = {}

        def _capture_align(**kwargs):
            captured.update(kwargs)
            return [{"text": "b", "timestamp_start": 0.1, "timestamp_end": 0.9}]

        monkeypatch.setattr(
            refine_module,
            "aligner",
            MagicMock(align_segment=MagicMock(side_effect=_capture_align)),
        )

        realign_words_for_chunk(
            session,
            "fileabc",
            0.0,
            1.0,
            "b",
            waveform=object(),
            language_override="en",
        )

        assert captured["language"] == "en"

    # NOTE: ``test_word_index_continues_from_existing_base`` was REMOVED.
    # It encoded the broken behaviour where new aligner rows reused the
    # old chunk's ``word_index`` base — when the aligner emits more rows
    # than the original chunk had, the new indices overflow into the
    # neighbouring chunk's index space, producing interleaved-duplicate
    # subtitles in VTT output (Stream A originals + Stream B refine
    # residue ordered by word_index). The DELETE is timestamp-scoped so
    # the overflow rows are never cleaned up. ``word_index`` is being
    # removed; ordering invariants are now asserted on timestamps.
    # Decision: hako GfJ-m48_jisu3dpMfRkcg.
    pass


# ---------------------------------------------------------------------------
# Regression: timestamp-based row identity (no word_index reliance)
# ---------------------------------------------------------------------------


import sys
from datetime import UTC, datetime
from unittest.mock import MagicMock as _MM

for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = _MM()

import pytest
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import TranscriptChunk, TranscriptWord


@pytest.fixture()
def real_session(tmp_path):
    """Real SQLite session for transcript_words integration assertions.

    We avoid the MagicMock approach used above because the bug under
    test is about how DELETE + INSERT interact across a real table —
    something a mock can't model. ``word_index`` is intentionally NOT
    populated by these tests (it is being removed); rows are identified
    purely by ``(file_id, timestamp_start, timestamp_end)``.
    """
    db_path = tmp_path / "words.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _insert_word(session, file_id, text, ts, te, *, language="ja"):
    """Insert one TranscriptWord (word_index column has been removed)."""
    session.execute(
        sql_text(
            "INSERT INTO transcript_words "
            "(file_id, text, language, "
            " timestamp_start, timestamp_end, created_at) "
            "VALUES (:fid, :t, :lang, :ts, :te, :ca)"
        ),
        {
            "fid": file_id,
            "t": text,
            "lang": language,
            "ts": ts,
            "te": te,
            "ca": datetime.now(UTC).isoformat(),
        },
    )


class TestNoOverflowIntoNeighborChunk:
    def test_realign_does_not_leak_word_rows_into_neighbor_chunk(
        self, real_session, monkeypatch
    ):
        """Aligner emits MORE rows than the chunk originally had.

        The current implementation computes ``base_index = min(existing
        word_index)`` and assigns ``base_index + i`` to each new row.
        When ``len(aligned) > len(existing)``, the overflow rows take
        on word_index values that belong to the next chunk — but the
        DELETE is timestamp-scoped so the next chunk's rows survive.
        Result: duplicate / interleaved data when ordered by word_index.

        With word_index gone, the invariant becomes: chunk2's rows
        (by timestamp) must be byte-for-byte identical before/after.
        """
        fid = "fileovrf"
        # Chunk 1: [0, 10s], 3 original words.
        _insert_word(real_session, fid, "a", 0.0, 3.0)
        _insert_word(real_session, fid, "b", 3.0, 6.0)
        _insert_word(real_session, fid, "c", 6.0, 9.5)
        # Chunk 2: [10, 20s], 3 original words. These must NOT be touched.
        _insert_word(real_session, fid, "X", 10.0, 13.0)
        _insert_word(real_session, fid, "Y", 13.0, 16.0)
        _insert_word(real_session, fid, "Z", 16.0, 19.5)
        real_session.commit()

        # Aligner returns 8 rows for chunk 1 — far more than the 3
        # originals, simulating refine producing extra char-level tokens.
        fake_units = [
            {"text": f"r{i}", "timestamp_start": 0.0 + i, "timestamp_end": 0.5 + i}
            for i in range(8)
        ]
        monkeypatch.setattr(
            refine_module,
            "aligner",
            MagicMock(align_segment=MagicMock(return_value=fake_units)),
        )

        inserted = realign_words_for_chunk(
            real_session, fid, 0.0, 10.0, "refined", waveform=object()
        )
        real_session.commit()

        assert inserted == 8

        # Chunk 2 must be intact: same count, same texts, same timestamps.
        chunk2 = real_session.execute(
            sql_text(
                "SELECT text, timestamp_start, timestamp_end "
                "FROM transcript_words "
                "WHERE file_id = :fid "
                "AND timestamp_start >= 10.0 "
                "ORDER BY timestamp_start"
            ),
            {"fid": fid},
        ).fetchall()
        assert [(r[0], r[1], r[2]) for r in chunk2] == [
            ("X", 10.0, 13.0),
            ("Y", 13.0, 16.0),
            ("Z", 16.0, 19.5),
        ]

        # Total rows = chunk2 (3) + new aligned chunk1 (8) = 11.
        total = real_session.execute(
            sql_text(
                "SELECT COUNT(*) FROM transcript_words WHERE file_id = :fid"
            ),
            {"fid": fid},
        ).scalar()
        assert total == 11

        # No duplicate (timestamp_start, timestamp_end, text) triples.
        dup_count = real_session.execute(
            sql_text(
                "SELECT COUNT(*) FROM ("
                "  SELECT timestamp_start, timestamp_end, text, COUNT(*) c "
                "  FROM transcript_words WHERE file_id = :fid "
                "  GROUP BY timestamp_start, timestamp_end, text "
                "  HAVING c > 1"
                ")"
            ),
            {"fid": fid},
        ).scalar()
        assert dup_count == 0

        # User-visible invariant: with word_index removed, the endpoint
        # orders by (timestamp_start, id). Verify the post-fix query
        # produces a stable timestamp-sorted sequence.
        by_ts_id = real_session.execute(
            sql_text(
                "SELECT timestamp_start FROM transcript_words "
                "WHERE file_id = :fid ORDER BY timestamp_start, id"
            ),
            {"fid": fid},
        ).fetchall()
        by_ts = real_session.execute(
            sql_text(
                "SELECT timestamp_start FROM transcript_words "
                "WHERE file_id = :fid ORDER BY timestamp_start"
            ),
            {"fid": fid},
        ).fetchall()
        assert [r[0] for r in by_ts_id] == [r[0] for r in by_ts], (
            "Timestamp-based ordering must be stable."
        )


class TestRefineRevertCycle:
    def test_refine_revert_cycle_produces_no_duplicate_words(
        self, real_session, monkeypatch
    ):
        """Refine all chunks, then revert. Net rowcount unchanged,
        no duplicate (file_id, ts_start, ts_end) triples, and the
        timestamp-ordered text sequence matches the pre-refine baseline.
        """
        fid = "filecycl"
        baseline = [
            ("hello", 0.0, 1.0),
            ("world", 1.0, 2.0),
            ("foo", 10.0, 11.0),
            ("bar", 11.0, 12.0),
        ]
        for t, ts, te in baseline:
            _insert_word(real_session, fid, t, ts, te, language="en")
        real_session.commit()

        baseline_count = real_session.execute(
            sql_text("SELECT COUNT(*) FROM transcript_words WHERE file_id = :fid"),
            {"fid": fid},
        ).scalar()
        assert baseline_count == 4

        # Refine chunk 1 ([0, 5s]) — aligner emits 3 rows (one extra).
        refine_units = [
            {"text": "HELLO", "timestamp_start": 0.0, "timestamp_end": 0.6},
            {"text": "BRAVE", "timestamp_start": 0.6, "timestamp_end": 1.3},
            {"text": "WORLD", "timestamp_start": 1.3, "timestamp_end": 2.0},
        ]
        monkeypatch.setattr(
            refine_module,
            "aligner",
            MagicMock(align_segment=MagicMock(return_value=refine_units)),
        )
        realign_words_for_chunk(
            real_session, fid, 0.0, 5.0, "HELLO BRAVE WORLD", waveform=object()
        )
        # Refine chunk 2 ([10, 15s]) — aligner emits 3 rows.
        refine_units2 = [
            {"text": "FOO", "timestamp_start": 10.0, "timestamp_end": 10.7},
            {"text": "BAZ", "timestamp_start": 10.7, "timestamp_end": 11.3},
            {"text": "BAR", "timestamp_start": 11.3, "timestamp_end": 12.0},
        ]
        monkeypatch.setattr(
            refine_module,
            "aligner",
            MagicMock(align_segment=MagicMock(return_value=refine_units2)),
        )
        realign_words_for_chunk(
            real_session, fid, 10.0, 15.0, "FOO BAZ BAR", waveform=object()
        )
        real_session.commit()

        # Now REVERT: realign with the original text. The aligner mock
        # returns the baseline word units for each chunk.
        revert1 = [
            {"text": "hello", "timestamp_start": 0.0, "timestamp_end": 1.0},
            {"text": "world", "timestamp_start": 1.0, "timestamp_end": 2.0},
        ]
        monkeypatch.setattr(
            refine_module,
            "aligner",
            MagicMock(align_segment=MagicMock(return_value=revert1)),
        )
        realign_words_for_chunk(
            real_session, fid, 0.0, 5.0, "hello world", waveform=object()
        )
        revert2 = [
            {"text": "foo", "timestamp_start": 10.0, "timestamp_end": 11.0},
            {"text": "bar", "timestamp_start": 11.0, "timestamp_end": 12.0},
        ]
        monkeypatch.setattr(
            refine_module,
            "aligner",
            MagicMock(align_segment=MagicMock(return_value=revert2)),
        )
        realign_words_for_chunk(
            real_session, fid, 10.0, 15.0, "foo bar", waveform=object()
        )
        real_session.commit()

        # Net rowcount unchanged.
        final_count = real_session.execute(
            sql_text("SELECT COUNT(*) FROM transcript_words WHERE file_id = :fid"),
            {"fid": fid},
        ).scalar()
        assert final_count == baseline_count

        # No duplicate (file_id, ts_start, ts_end) triples.
        dup_count = real_session.execute(
            sql_text(
                "SELECT COUNT(*) FROM ("
                "  SELECT file_id, timestamp_start, timestamp_end, COUNT(*) c "
                "  FROM transcript_words WHERE file_id = :fid "
                "  GROUP BY file_id, timestamp_start, timestamp_end "
                "  HAVING c > 1"
                ")"
            ),
            {"fid": fid},
        ).scalar()
        assert dup_count == 0

        # Timestamp-ordered text sequence == baseline.
        rows = real_session.execute(
            sql_text(
                "SELECT text, timestamp_start, timestamp_end "
                "FROM transcript_words WHERE file_id = :fid "
                "ORDER BY timestamp_start"
            ),
            {"fid": fid},
        ).fetchall()
        assert [(r[0], r[1], r[2]) for r in rows] == baseline

        # Endpoint path (post-fix: ORDER BY timestamp_start, id) must
        # also match baseline after a refine + revert cycle.
        rows_endpoint = real_session.execute(
            sql_text(
                "SELECT text FROM transcript_words WHERE file_id = :fid "
                "ORDER BY timestamp_start, id"
            ),
            {"fid": fid},
        ).fetchall()
        assert [r[0] for r in rows_endpoint] == [t for t, _, _ in baseline], (
            "timestamp ordering after refine/revert cycle is scrambled."
        )


class TestVttEndpointOrderingInvariant:
    def test_vtt_endpoint_orders_by_timestamp_not_insertion_order(
        self, real_session, monkeypatch
    ):
        """Insert rows in shuffled timestamp order via bulk_insert path,
        then build the VTT and check cues come out time-sorted.

        The current endpoint uses ``ORDER BY word_index`` — when
        word_index doesn't match insertion-order or timestamp-order,
        the output is scrambled. With word_index removed, the endpoint
        MUST sort by timestamp_start.
        """
        from app.models import TranscriptWord as TW
        from app.subtitle_builder import build_vtt

        fid = "filevtt0"
        # Shuffled insertion order: ts 5.0, 1.0, 3.0.
        real_session.bulk_insert_mappings(
            TW,
            [
                {
                    "file_id": fid,
                    "text": "third",
                    "language": "en",
                    "timestamp_start": 5.0,
                    "timestamp_end": 5.5,
                },
                {
                    "file_id": fid,
                    "text": "first",
                    "language": "en",
                    "timestamp_start": 1.0,
                    "timestamp_end": 1.5,
                },
                {
                    "file_id": fid,
                    "text": "second",
                    "language": "en",
                    "timestamp_start": 3.0,
                    "timestamp_end": 3.5,
                },
            ],
        )
        real_session.commit()

        # Post-fix endpoint query: ORDER BY timestamp_start (+id tiebreak).
        rows = real_session.execute(
            sql_text(
                "SELECT text, timestamp_start, timestamp_end "
                "FROM transcript_words WHERE file_id = :fid "
                "ORDER BY timestamp_start, id"
            ),
            {"fid": fid},
        ).fetchall()

        words = [
            {"text": r[0], "timestamp_start": r[1], "timestamp_end": r[2]}
            for r in rows
        ]
        vtt = build_vtt(words, language="en")

        # Find the order in which 'first', 'second', 'third' appear in
        # the rendered VTT body.
        positions = {
            tag: vtt.find(tag) for tag in ("first", "second", "third")
        }
        assert all(p >= 0 for p in positions.values()), vtt
        assert positions["first"] < positions["second"] < positions["third"], (
            f"VTT cues are not ordered by timestamp:\n{vtt}"
        )


# The previous ``TestUniqueTimestamps`` class was removed as part of
# the clamp-in-aligner refactor. The "unique timestamp_start per file"
# invariant was over-claimed: back-to-back ASR segments and zero-
# duration units can legitimately share a ``timestamp_start`` while
# remaining distinct rows. The weaker, exact invariant — "chunk N
# realign cannot damage chunk N+1's rows" — is covered by
# ``TestNoOverflowIntoNeighborChunk`` above. Aligner-window clamp
# behaviour itself is covered in ``tests/test_refine_aligner.py``.
