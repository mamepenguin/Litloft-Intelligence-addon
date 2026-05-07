"""Phase 1C tests: refine clears speaker_id on regenerated rows.

Spec ``2026-05-07-cloud-transcription-providers.md`` §"WhisperX 接続":
``transcript_refine`` deletes original ``TranscriptWord`` rows and
re-inserts new ones via the WhisperX forced aligner. The original
↔ new mapping is N:M and not provider-aware, so the spec mandates
that **refine drops speaker_id back to NULL** on every refreshed
row (both words and chunks).

Phase 2 may decide to either inherit speaker_id by time-overlap or
expose "edited (no speaker info)" in a future speaker UI. For now,
NULL is the contract.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

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
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import Base  # noqa: E402
from app.models import IndexedFile, TranscriptChunk, TranscriptWord  # noqa: E402


@pytest.fixture()
def engine(tmp_path):
    db_path = tmp_path / "search.db"
    eng = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session(engine):
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    s = Session()
    s.add(IndexedFile(
        file_id="f00000000001",
        drive="d1",
        filename="x.mp4",
        file_path="/drives/d1/x.mp4",
        file_type="video",
        mime_type="video/mp4",
        file_size=1,
        active=True,
    ))
    s.commit()
    yield s
    s.close()


def _seed_speaker_words(session) -> None:
    """Pre-existing diarized words (from a Deepgram run)."""
    for i, sid in enumerate(["spk_0", "spk_0", "spk_1", "spk_1"]):
        session.add(TranscriptWord(
            file_id="f00000000001",
            text=f"w{i}",
            language="en",
            timestamp_start=float(i),
            timestamp_end=float(i + 1),
            speaker_id=sid,
        ))
    session.commit()


def test_realign_words_drops_speaker_id_when_aligner_succeeds(
    session, monkeypatch
) -> None:
    """After ``realign_words_for_chunk`` rewrites words, all new rows
    must have ``speaker_id=None`` regardless of what the original
    rows carried."""
    from app.workers import refine

    _seed_speaker_words(session)

    # Stub aligner so we don't need WhisperX during the test.
    fake_units = [
        {"text": "alpha", "timestamp_start": 0.1, "timestamp_end": 0.5},
        {"text": "beta", "timestamp_start": 0.5, "timestamp_end": 1.0},
    ]
    monkeypatch.setattr(
        refine.aligner, "align_segment", lambda **kw: fake_units
    )

    n = refine.realign_words_for_chunk(
        session,
        file_id="f00000000001",
        chunk_start=0.0,
        chunk_end=2.0,
        refined_text="alpha beta",
        waveform=object(),  # any non-None
    )
    assert n == 2

    # The two re-aligned words must replace the seeded "spk_0" pair.
    survivors = (
        session.query(TranscriptWord)
        .filter(
            TranscriptWord.file_id == "f00000000001",
            TranscriptWord.timestamp_start <= 1.0,
        )
        .all()
    )
    assert {w.text for w in survivors} == {"alpha", "beta"}
    for w in survivors:
        assert w.speaker_id is None, (
            f"refine produced word {w.text!r} with speaker_id={w.speaker_id!r}; "
            "expected None"
        )


def test_realign_words_preserves_speaker_id_when_aligner_fails(
    session, monkeypatch
) -> None:
    """When the aligner returns nothing, refine MUST NOT delete words.

    The pre-existing speaker_id values therefore stay intact (this is
    the legacy "preserve original on aligner failure" contract from
    hako iG6Uotc_uQ8cpXufZQf6v).
    """
    from app.workers import refine

    _seed_speaker_words(session)

    monkeypatch.setattr(refine.aligner, "align_segment", lambda **kw: [])

    n = refine.realign_words_for_chunk(
        session,
        file_id="f00000000001",
        chunk_start=0.0,
        chunk_end=2.0,
        refined_text="alpha beta",
        waveform=object(),
    )
    assert n == 0

    untouched = (
        session.query(TranscriptWord)
        .filter(TranscriptWord.file_id == "f00000000001")
        .order_by(TranscriptWord.timestamp_start)
        .all()
    )
    # Original speaker_id values still present on the rows we wanted
    # to leave alone.
    assert untouched[0].speaker_id == "spk_0"
    assert untouched[2].speaker_id == "spk_1"


def test_rechunk_drops_speaker_id_on_new_chunks(session) -> None:
    """``rechunk_from_words`` regenerates chunks from current word rows.

    Even if those words still have speaker_id set (refine hasn't
    realigned them yet, or they survived an aligner failure), the new
    chunks must be NULL — refine deliberately decouples chunks from
    diarisation per the spec.
    """
    from app.workers import refine

    # Seed words with speaker labels such that they fold into a single
    # chunk under the default min/max settings.
    _seed_speaker_words(session)

    new_ids = refine.rechunk_from_words(session, file_id="f00000000001")
    assert new_ids, "rechunk must produce at least one new chunk"

    chunks = (
        session.query(TranscriptChunk)
        .filter(TranscriptChunk.file_id == "f00000000001")
        .all()
    )
    assert chunks, "rechunk must persist new TranscriptChunk rows"
    for c in chunks:
        assert c.speaker_id is None, (
            f"rechunk produced TranscriptChunk(speaker_id={c.speaker_id!r}); "
            "expected None"
        )
