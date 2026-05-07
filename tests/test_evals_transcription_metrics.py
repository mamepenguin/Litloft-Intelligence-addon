"""Tests for WER / CER / sa-WER scoring."""

from __future__ import annotations

import pytest

from app.evals_transcription.metrics import (
    SpeakerSegment,
    score_speaker_attributed_wer,
    score_text,
)
from app.workers.transcription.base import WordToken


# ---------------------------------------------------------------------------
# score_text
# ---------------------------------------------------------------------------


def test_english_perfect_match_returns_zero_wer_and_cer() -> None:
    wer, cer = score_text("Hello world", "Hello world", "en")
    assert wer == 0.0
    assert cer == 0.0


def test_english_word_substitution_increases_wer() -> None:
    wer, cer = score_text("hello world", "hello earth", "en")
    assert wer > 0
    assert cer > 0
    # 1 of 2 words wrong → WER == 0.5
    assert wer == pytest.approx(0.5)


def test_english_punctuation_does_not_count_as_error() -> None:
    """The pre-normaliser strips punctuation; jiwer must not re-add it."""
    wer, cer = score_text("hello, world.", "hello world", "en")
    assert wer == 0.0
    assert cer == 0.0


def test_japanese_returns_none_for_wer_and_meaningful_cer() -> None:
    """ja has no whitespace word boundaries → WER would be 0/1 binary."""
    wer, cer = score_text("こんにちは、世界", "こんにちは、世界", "ja")
    assert wer is None
    assert cer == 0.0


def test_japanese_one_char_substitution_yields_proportional_cer() -> None:
    wer, cer = score_text("こんにちは", "こんはちは", "ja")
    assert wer is None
    # 1 of 5 chars wrong
    assert 0 < cer < 1
    assert cer == pytest.approx(0.2)


def test_japanese_katakana_hiragana_tolerance() -> None:
    """Whisper-family vs cloud providers differ on script; fold should
    erase the difference."""
    wer, cer = score_text("カタカナ", "かたかな", "ja")
    assert wer is None
    assert cer == 0.0


def test_english_case_insensitive() -> None:
    wer, cer = score_text("Hello", "hello", "en")
    assert wer == 0.0
    assert cer == 0.0


def test_empty_hypothesis_against_nonempty_reference() -> None:
    """Total deletion: every word / char should count as error."""
    wer, cer = score_text("hello world", "", "en")
    assert wer == pytest.approx(1.0)
    assert cer == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_speaker_attributed_wer
# ---------------------------------------------------------------------------


def test_sa_wer_returns_none_when_no_speakers() -> None:
    """No GT speakers → no diarization to score."""
    words = [WordToken(text="x", start=0, end=1, speaker_id="0")]
    assert score_speaker_attributed_wer([], words) is None


def test_sa_wer_returns_none_when_provider_has_no_speaker_ids() -> None:
    """Provider does not diarise → words have speaker_id=None →
    sa-WER is N/A."""
    speakers = [SpeakerSegment("A", 0.0, 5.0)]
    words = [WordToken(text="x", start=0, end=1, speaker_id=None)]
    assert score_speaker_attributed_wer(speakers, words) is None


def test_sa_wer_perfect_assignment_yields_zero() -> None:
    """Provider speaker IDs map cleanly to GT IDs by majority."""
    speakers = [
        SpeakerSegment("A", 0.0, 2.0),
        SpeakerSegment("B", 2.0, 4.0),
    ]
    words = [
        WordToken(text="hi", start=0.5, end=1.0, speaker_id="0"),
        WordToken(text="bye", start=2.5, end=3.0, speaker_id="1"),
    ]
    assert score_speaker_attributed_wer(speakers, words) == 0.0


def test_sa_wer_mismatch_increases_score() -> None:
    """Word at t=2.5s belongs to GT B but provider tagged speaker 0
    (which the majority assignment maps to GT A)."""
    speakers = [
        SpeakerSegment("A", 0.0, 2.0),
        SpeakerSegment("B", 2.0, 4.0),
    ]
    # Two words clearly tagged "0" → A side, two clearly tagged "1" → B
    # side, plus one mistaken "0" within B's window.
    words = [
        WordToken(text="hi", start=0.5, end=1.0, speaker_id="0"),
        WordToken(text="hello", start=1.0, end=1.5, speaker_id="0"),
        WordToken(text="bye", start=2.5, end=3.0, speaker_id="1"),
        WordToken(text="see", start=3.0, end=3.5, speaker_id="1"),
        WordToken(text="oops", start=3.6, end=3.9, speaker_id="0"),
    ]
    rate = score_speaker_attributed_wer(speakers, words)
    assert rate is not None
    # 1 mismatched word out of 5
    assert rate == pytest.approx(0.2)


def test_sa_wer_handles_alphabet_speaker_ids_from_assemblyai() -> None:
    """AssemblyAI emits ``"A"``, ``"B"``; majority-vote mapping should
    still work."""
    speakers = [
        SpeakerSegment("speaker_0", 0.0, 2.0),
        SpeakerSegment("speaker_1", 2.0, 4.0),
    ]
    words = [
        WordToken(text="hi", start=0.5, end=1.0, speaker_id="A"),
        WordToken(text="bye", start=2.5, end=3.0, speaker_id="B"),
    ]
    assert score_speaker_attributed_wer(speakers, words) == 0.0
