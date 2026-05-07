"""ffmpeg-based audio chunking for long-form transcription.

Phase 2B helper: when a cloud provider's hard byte cap is exceeded,
this module normalizes the input to 16 kHz mono FLAC, detects silence
boundaries with ffmpeg ``silencedetect``, and slices the normalized
file into chunks small enough for the provider while biasing cuts
towards natural pauses.

Spec: ``2026-05-08-transcription-providers-phase-2b.md``.

Design notes:

* The normalisation step exists primarily to make ``bytes/sec``
  predictable. Cloud STT providers vary wildly on what container /
  codec / sample-rate combinations they accept; routing everything
  through ``-ac 1 -ar 16000 -c:a flac`` collapses the input space and
  also dodges Phase 1F bugs like the "M4A inside .mp4 wrapper" trap
  (hako ``4t5FWrH4IpLUlGDXxh7cO``).
* ``_choose_split_points`` looks within ±25% of each target boundary
  for a silence midpoint and falls back to the exact boundary if no
  silence is in window. This keeps cuts away from word boundaries
  whenever possible while guaranteeing progress on continuous-speech
  inputs.
* Slice ordering ``-ss start -i src -t duration`` is deliberate:
  putting both ``-ss`` and ``-to`` *before* ``-i`` causes ffmpeg to
  treat ``-to`` as input read duration (cuts from 0 to ``end``),
  not as output end-time. Input-side ``-ss`` keeps demuxer-seek speed;
  output-side ``-t`` measures duration relative to the post-seek
  position. FLAC streams are frame-seekable, so chunks land within
  ~256 ms of the requested boundary (FLAC frame ≈ 4096 samples at
  16 kHz). True sample-accuracy would require re-encoding (CPU cost
  outweighs the precision gain for search/chunking workloads).
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass

from app.workers.transcription.base import (
    TranscriptionSegment,
    WordToken,
)
from app.workers.transcription.errors import FatalError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NORMALIZED_SAMPLE_RATE = 16000
NORMALIZED_CHANNELS = 1
NORMALIZED_FORMAT = "flac"

# 16 kHz mono 16-bit raw = 32 KB/s. FLAC compresses to roughly 0.5-0.7
# of the raw rate for natural speech, but we keep the conservative
# upper bound as a fallback when measurement isn't possible. The
# preferred path measures the actual normalized file's bytes/sec.
FLAC_BYTES_PER_SECOND_ESTIMATE = 32 * 1024

# Leave headroom on the provider cap so VBR variation in FLAC output
# does not push individual chunks past the limit.
SPLIT_SAFETY_FACTOR = 0.8

# silencedetect filter parameters: -30 dB for 0.5 s tracks
# sentence-break level pauses without latching onto sub-syllabic noise.
SILENCE_NOISE_DB = "-30dB"
SILENCE_MIN_DURATION_S = 0.5

# Range around each target boundary in which we accept a silence
# midpoint as a clean cut. Tighter than ±target_duration so multi-hour
# files do not produce wildly uneven chunks.
TARGET_WINDOW_TOLERANCE = 0.25

# Lower bound on chunk duration: providers occasionally reject
# pathologically short clips and we don't want to bill the user for a
# 1-second tail chunk when merging it back into the previous chunk
# costs nothing.
MIN_CHUNK_DURATION_S = 5.0

TMPDIR_PREFIX = "litloft-split-"


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioChunk:
    """One sliced chunk of the normalized FLAC.

    ``path`` is absolute and lives inside the tmpdir returned alongside
    by :func:`split_audio`; the caller is responsible for cleanup of
    the entire tmpdir (which removes both the chunks and the
    normalized parent).
    """

    path: str
    start_offset_s: float
    duration_s: float


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


async def split_audio(
    src_path: str,
    cap_bytes: int,
) -> tuple[list[AudioChunk], str]:
    """Normalize, detect silences, and slice into cap-respecting chunks.

    Returns ``(chunks, tmpdir)``. ``tmpdir`` is a fresh ``tempfile.mkdtemp``
    directory containing all generated files; callers must
    ``shutil.rmtree(tmpdir, ignore_errors=True)`` once done. On failure
    the tmpdir is removed before the exception propagates.

    The single-chunk pass-through case (file is already small enough
    after normalisation) returns one ``AudioChunk`` whose ``path`` is
    the normalized file itself.
    """
    tmpdir = tempfile.mkdtemp(prefix=TMPDIR_PREFIX)
    try:
        normalized = os.path.join(tmpdir, "normalized.flac")
        await _ffmpeg_normalize(src_path, normalized)

        duration = await _ffprobe_duration(normalized)
        bytes_per_second = (
            _measure_bytes_per_second(normalized, duration)
            or FLAC_BYTES_PER_SECOND_ESTIMATE
        )
        target_duration = (
            cap_bytes * SPLIT_SAFETY_FACTOR / bytes_per_second
        )

        if duration <= target_duration:
            return (
                [AudioChunk(
                    path=normalized,
                    start_offset_s=0.0,
                    duration_s=duration,
                )],
                tmpdir,
            )

        silences = await _detect_silences(normalized)
        split_points = _choose_split_points(
            duration, target_duration, silences
        )
        chunks: list[AudioChunk] = []
        prev = 0.0
        for i, end in enumerate(split_points + [duration]):
            chunk_duration = end - prev
            # Note (R1 spec M-R1-1): when ``split_points`` is empty we
            # have already returned via the early-return pass-through
            # above; the ``and chunks`` guard below is therefore
            # defensive — a single sub-MIN chunk can never reach this
            # loop.
            if chunk_duration < MIN_CHUNK_DURATION_S and chunks:
                # Fold the too-short tail into the previous chunk by
                # re-slicing in place (overwriting the previous chunk
                # file with the wider span).
                last = chunks.pop()
                prev = last.start_offset_s
                chunk_duration = end - prev
                await _ffmpeg_slice(normalized, prev, end, last.path)
                chunks.append(
                    AudioChunk(
                        path=last.path,
                        start_offset_s=prev,
                        duration_s=chunk_duration,
                    )
                )
                prev = end
                continue
            chunk_path = os.path.join(tmpdir, f"chunk_{i:03d}.flac")
            await _ffmpeg_slice(normalized, prev, end, chunk_path)
            chunks.append(
                AudioChunk(
                    path=chunk_path,
                    start_offset_s=prev,
                    duration_s=chunk_duration,
                )
            )
            prev = end
        return chunks, tmpdir
    except Exception:
        # Failure path cleanup. The success path lets the caller own
        # tmpdir lifetime so chunk files remain accessible during
        # transcription.
        with contextlib.suppress(OSError):
            shutil.rmtree(tmpdir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe wrappers
# ---------------------------------------------------------------------------


async def _ffmpeg_normalize(src: str, dst: str) -> None:
    """Re-encode ``src`` to 16 kHz mono FLAC at ``dst``."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src,
        "-vn",                              # discard any video stream
        "-ac", str(NORMALIZED_CHANNELS),
        "-ar", str(NORMALIZED_SAMPLE_RATE),
        "-c:a", NORMALIZED_FORMAT,
        dst,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise FatalError(
            f"ffmpeg normalize failed: "
            f"{stderr.decode('utf-8', 'replace')[:500]}"
        )


async def _ffprobe_duration(path: str) -> float:
    """Return the duration of ``path`` in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise FatalError(
            f"ffprobe failed: {stderr.decode('utf-8', 'replace')[:500]}"
        )
    raw = stdout.decode().strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise FatalError(
            f"ffprobe returned non-numeric duration: {raw!r}"
        ) from exc


def _measure_bytes_per_second(path: str, duration: float) -> float | None:
    """Read the actual file size and divide by duration.

    Returns ``None`` when measurement fails (file vanished, duration
    is 0) so the caller can fall back to the conservative constant
    bytes/sec estimate.
    """
    if duration <= 0:
        return None
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    return size / duration


# Number pattern tolerant of negative starts (silencedetect emits
# ``-0.000123`` near zero in some FFmpeg builds) and scientific
# notation, but tight enough to reject ``1.2.3``-style false positives
# (R1 spec L-R1-2).
_NUMBER_PATTERN = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_SILENCE_START_RE = re.compile(
    rf"silence_start:\s*({_NUMBER_PATTERN})"
)
_SILENCE_END_RE = re.compile(rf"silence_end:\s*({_NUMBER_PATTERN})")


async def _detect_silences(path: str) -> list[tuple[float, float]]:
    """Return ``[(silence_start_s, silence_end_s), ...]``.

    Forces ``-loglevel info`` so the silencedetect filter's stderr
    lines are present regardless of any global ffmpeg log-level
    setting in the container. Failures here are FatalError because
    silencedetect on the same just-normalized file will fail again
    on retry; transient classification would build up retry storms.
    """
    cmd = [
        "ffmpeg", "-loglevel", "info", "-i", path,
        "-af",
        f"silencedetect=noise={SILENCE_NOISE_DB}:d={SILENCE_MIN_DURATION_S}",
        "-f", "null", "-",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise FatalError(
            f"silencedetect failed: "
            f"{stderr.decode('utf-8', 'replace')[:500]}"
        )

    silences: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in stderr.decode("utf-8", "replace").splitlines():
        m_start = _SILENCE_START_RE.search(line)
        if m_start:
            pending_start = float(m_start.group(1))
            continue
        m_end = _SILENCE_END_RE.search(line)
        if m_end and pending_start is not None:
            silences.append((pending_start, float(m_end.group(1))))
            pending_start = None
    return silences


def _choose_split_points(
    duration: float,
    target_duration: float,
    silences: list[tuple[float, float]],
) -> list[float]:
    """Choose split points biased toward target_duration spacing.

    For each target boundary ``k * target_duration`` (k=1..N-1), look
    for a silence whose midpoint lies within
    ``TARGET_WINDOW_TOLERANCE`` of that boundary. Prefer the longest
    silence in the window. If no silence is found, fall back to the
    exact target time (fail-soft for continuous-speech inputs).
    """
    if duration <= target_duration:
        return []
    n = max(1, math.ceil(duration / target_duration) - 1)
    points: list[float] = []
    for k in range(1, n + 1):
        boundary = k * target_duration
        if boundary >= duration:
            break
        window_low = boundary * (1 - TARGET_WINDOW_TOLERANCE)
        window_high = boundary * (1 + TARGET_WINDOW_TOLERANCE)
        candidates = [
            (s, e)
            for s, e in silences
            if window_low <= (s + e) / 2 <= window_high
        ]
        if candidates:
            best = max(candidates, key=lambda se: se[1] - se[0])
            points.append((best[0] + best[1]) / 2)
        else:
            points.append(boundary)

    # Defensive: ensure strictly increasing and within (0, duration).
    seen: set[float] = set()
    out: list[float] = []
    for p in points:
        if 0.0 < p < duration and p not in seen:
            seen.add(p)
            out.append(p)
    return sorted(out)


async def _ffmpeg_slice(src: str, start: float, end: float, dst: str) -> None:
    """Slice ``[start, end)`` (seconds) from ``src`` into ``dst``.

    Uses input-side ``-ss`` plus output-side ``-t`` so the seek-then-
    duration semantics are correct. Putting both ``-ss`` and ``-to``
    *before* ``-i`` makes ffmpeg interpret ``-to`` as the input-read
    duration (cuts from 0 to ``end`` in absolute time, not from
    ``start`` to ``end``); putting them after ``-i`` slows down on
    long inputs because the demuxer reads everything before the cut
    point. The chosen ordering is FLAC frame-accurate (~256 ms drift
    from ideal at the boundary) and as fast as ``-c:a copy`` can
    deliver.
    """
    duration = max(0.0, end - start)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", src,
        "-t", f"{duration:.3f}",
        "-c:a", "copy",
        dst,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise FatalError(
            f"ffmpeg slice {start}-{end} failed: "
            f"{stderr.decode('utf-8', 'replace')[:500]}"
        )


# ---------------------------------------------------------------------------
# Segment helpers used by SplittingTranscriber
# ---------------------------------------------------------------------------


def offset_segment(seg: TranscriptionSegment, offset: float) -> TranscriptionSegment:
    """Return a new segment with ``offset`` added to every timestamp."""
    return TranscriptionSegment(
        text=seg.text,
        start=seg.start + offset,
        end=seg.end + offset,
        language=seg.language,
        words=[
            WordToken(
                text=w.text,
                start=w.start + offset,
                end=w.end + offset,
                speaker_id=w.speaker_id,
            )
            for w in seg.words
        ],
    )


def segments_tail_text(
    segments: list[TranscriptionSegment],
    max_chars: int,
) -> str:
    """Return the trailing ``max_chars`` of concatenated segment text.

    Used to seed ``initial_prompt`` for the next chunk. Bounded short
    so the Whisper input prompt stays inside the 224-token limit
    (CJK ≈ 1 token per char; 150 chars leaves room for bos/lang
    special tokens). Per Phase 2B precedence rule for whisper_local,
    the caller-supplied initial_prompt **replaces** the language
    default for chunk N>0, so the 150 chars do not concatenate with
    anything else. Limits Whisper vocabulary bleed risk per hako
    ``jqiF9yOk9VxEhU-H8sLbL``.
    """
    if not segments:
        return ""
    joined = " ".join(s.text.strip() for s in segments if s.text)
    return joined[-max_chars:] if len(joined) > max_chars else joined
