"""WER / CER / sa-WER computation with explicit jiwer transforms.

Phase 2C: jiwer 3.x's default transforms strip punctuation, lowercase,
and split on whitespace. We pre-normalize the strings ourselves
(see :mod:`.normalize`), so feeding them straight into ``jiwer.wer``
double-processes the input — and for Japanese the default whitespace
tokenizer collapses the entire sentence into one "word". To avoid
this, both metric calls install identity-style transforms.

For Japanese specifically WER is meaningless (no whitespace word
boundaries) so :func:`score_text` returns ``None`` for WER and CER as
the primary text metric. English and other whitespace languages get
both.

The diarization column in the final report is labelled ``sa-WER``
(speaker-attributed WER) — NOT industry-standard DER (NIST). Canonical
DER is frame-based and accounts for false-alarm + missed-speech +
speaker-confusion separately. We compute a simpler word-level
confusion rate so single multi-speaker case can pin a number without
the operator owning a frame-level VAD reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer

from app.evals_transcription.normalize import normalize
from app.workers.transcription.base import WordToken


@dataclass(frozen=True)
class SpeakerSegment:
    """One time-range belonging to a single speaker in the GT.

    ``segments`` from the case YAML are converted into a flat list of
    these by the loader.
    """

    speaker_id: str
    start: float
    end: float


# Identity-ish transforms: jiwer requires its final step to be one of
# ``ReduceToListOfListOfWords`` (for word edit-distance) or
# ``ReduceToListOfListOfChars`` (for character edit-distance). They
# both run on the already-normalized strings without further
# tokenisation behaviour.
_WORD_TRANSFORM = jiwer.Compose([jiwer.ReduceToListOfListOfWords()])
_CHAR_TRANSFORM = jiwer.Compose([jiwer.ReduceToListOfListOfChars()])


def score_text(
    reference: str,
    hypothesis: str,
    language: str,
) -> tuple[float | None, float]:
    """Return ``(WER_or_None, CER)`` after language-aware normalisation.

    For ``ja`` we return ``None`` for WER because Japanese has no
    whitespace word boundaries — jiwer would treat the entire
    sentence as a single word and report WER ∈ {0, 1}, which carries
    no signal. CER is meaningful for both languages and is the
    primary text metric for ``ja``.

    jiwer's ``wer()`` accepts ``truth_transform`` / ``hypothesis_transform``,
    but ``cer()`` does not (it routes through ``process_characters``).
    For CER we therefore explicitly call the underlying
    ``process_characters`` with our identity transform so the
    pre-normalised strings are not double-processed.
    """
    ref_n = normalize(reference, language)
    hyp_n = normalize(hypothesis, language)

    char_output = jiwer.process_characters(
        ref_n,
        hyp_n,
        reference_transform=_CHAR_TRANSFORM,
        hypothesis_transform=_CHAR_TRANSFORM,
    )
    cer_score = float(char_output.cer)

    code = language.lower().split("-")[0]
    if code == "ja":
        return None, cer_score

    wer_score = float(
        jiwer.wer(
            ref_n,
            hyp_n,
            reference_transform=_WORD_TRANSFORM,
            hypothesis_transform=_WORD_TRANSFORM,
        )
    )
    return wer_score, cer_score


def score_speaker_attributed_wer(
    case_speakers: list[SpeakerSegment],
    hypothesis_words: list[WordToken],
) -> float | None:
    """Token-level speaker-attribution error rate.

    For each hypothesis word with a non-None ``speaker_id``, look up
    the GT speaker by word midpoint timestamp and count mismatches.
    Words without a ``speaker_id`` are excluded from the denominator
    so providers that don't diarise (or synthetic word splits from a
    Gemini-style backend) don't contribute noise.

    Returns ``None`` when no hypothesis word carries a ``speaker_id``
    so the report shows N/A instead of 0%. Returns 0.0 when every
    word matches.
    """
    if not case_speakers:
        return None
    diarised = [w for w in hypothesis_words if w.speaker_id is not None]
    if not diarised:
        return None

    sorted_segments = sorted(case_speakers, key=lambda s: s.start)

    # Build a hypothesis_speaker_id → gt_speaker_id mapping by majority
    # vote: providers assign their own speaker labels (``"0"`` /
    # ``"speaker_0"`` / ``"A"``) and we don't know which corresponds
    # to which GT label. We therefore tally ``(hyp_speaker, gt_speaker)``
    # co-occurrences and use the dominant gt speaker for each hyp
    # speaker. This is a Hungarian-lite assignment that handles the
    # ASR side label noise.
    cooccurrence: dict[tuple[str, str], int] = {}
    for w in diarised:
        midpoint = (w.start + w.end) / 2
        gt = _gt_speaker_at(sorted_segments, midpoint)
        if gt is None:
            continue
        key = (w.speaker_id, gt)
        cooccurrence[key] = cooccurrence.get(key, 0) + 1

    if not cooccurrence:
        return None

    hyp_to_gt: dict[str, str] = {}
    for hyp_speaker in {h for h, _ in cooccurrence}:
        candidates = [
            (gt, count)
            for (h, gt), count in cooccurrence.items()
            if h == hyp_speaker
        ]
        best_gt, _ = max(candidates, key=lambda c: c[1])
        hyp_to_gt[hyp_speaker] = best_gt

    errors = 0
    total = 0
    for w in diarised:
        midpoint = (w.start + w.end) / 2
        gt = _gt_speaker_at(sorted_segments, midpoint)
        if gt is None:
            continue
        total += 1
        predicted_gt = hyp_to_gt.get(w.speaker_id)
        if predicted_gt != gt:
            errors += 1
    if total == 0:
        return None
    return errors / total


def _gt_speaker_at(
    sorted_segments: list[SpeakerSegment],
    t: float,
) -> str | None:
    """Linear scan over GT segments. Falls back to the closest segment
    when ``t`` is in a gap (typical at the very start / end)."""
    for seg in sorted_segments:
        if seg.start <= t < seg.end:
            return seg.speaker_id
    if not sorted_segments:
        return None
    # Closest by absolute distance to either endpoint.
    closest = min(
        sorted_segments,
        key=lambda s: min(abs(s.start - t), abs(s.end - t)),
    )
    return closest.speaker_id
