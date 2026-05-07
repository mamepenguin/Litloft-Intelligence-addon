"""Tests for the ffmpeg-based audio splitter.

The full ffmpeg pipeline is exercised against real binaries inside the
intelligence-test image (which already ships ffmpeg / ffprobe for
faster-whisper). Only the silencedetect parser and split-point chooser
are pure-Python; everything else is integration-tested with synthetic
WAV inputs to keep tests fast and self-contained.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import struct
import wave
from unittest.mock import patch

import pytest

from app.workers.transcription.errors import FatalError
from app.workers.transcription.base import (
    TranscriptionSegment,
    WordToken,
)
from app.workers.transcription.splitter import (
    AudioChunk,
    MIN_CHUNK_DURATION_S,
    _choose_split_points,
    _detect_silences,
    _ffmpeg_normalize,
    _ffmpeg_slice,
    _ffprobe_duration,
    _measure_bytes_per_second,
    offset_segment,
    segments_tail_text,
    split_audio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wav(
    path: str,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    *,
    silent_ranges: list[tuple[float, float]] | None = None,
) -> None:
    """Write a 16-bit PCM mono WAV with optional silent regions.

    Active regions are filled with a 440 Hz square wave at half
    amplitude so silencedetect has something to bracket against.
    """
    silent_ranges = silent_ranges or []
    n_frames = int(sample_rate * duration_s)
    samples = []
    for i in range(n_frames):
        t = i / sample_rate
        in_silence = any(s <= t < e for s, e in silent_ranges)
        if in_silence:
            samples.append(0)
        else:
            # Square wave at 440 Hz.
            samples.append(8000 if (int(t * 880)) % 2 == 0 else -8000)

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f"<{n_frames}h", *samples))


# ---------------------------------------------------------------------------
# offset_segment / segments_tail_text — pure helpers
# ---------------------------------------------------------------------------


def test_offset_segment_shifts_all_timestamps() -> None:
    seg = TranscriptionSegment(
        text="hello",
        start=1.0,
        end=2.0,
        language="en",
        words=[
            WordToken(text="hello", start=1.2, end=1.8, speaker_id="A"),
        ],
    )
    shifted = offset_segment(seg, 10.0)
    assert shifted.start == 11.0
    assert shifted.end == 12.0
    assert shifted.words[0].start == 11.2
    assert shifted.words[0].end == 11.8
    assert shifted.words[0].speaker_id == "A"
    assert shifted.text == "hello"
    assert shifted.language == "en"


def test_offset_segment_does_not_mutate_input() -> None:
    seg = TranscriptionSegment(
        text="x",
        start=0.0,
        end=1.0,
        language="en",
        words=[WordToken(text="x", start=0.0, end=1.0)],
    )
    offset_segment(seg, 5.0)
    # Frozen dataclass is enforced; verify nothing changed.
    assert seg.start == 0.0
    assert seg.words[0].start == 0.0


def test_segments_tail_text_empty() -> None:
    assert segments_tail_text([], max_chars=100) == ""


def test_segments_tail_text_short_returns_full() -> None:
    seg = TranscriptionSegment(
        text="hello world",
        start=0,
        end=1,
        language="en",
        words=[],
    )
    assert segments_tail_text([seg], max_chars=100) == "hello world"


def test_segments_tail_text_long_truncates_to_tail() -> None:
    seg = TranscriptionSegment(
        text="abcdefghij" * 30,  # 300 chars
        start=0,
        end=1,
        language="en",
        words=[],
    )
    out = segments_tail_text([seg], max_chars=50)
    assert len(out) == 50
    assert out.endswith("abcdefghij")


# ---------------------------------------------------------------------------
# _choose_split_points — pure helper
# ---------------------------------------------------------------------------


def test_choose_split_points_short_duration_returns_empty() -> None:
    assert _choose_split_points(duration=10.0, target_duration=20.0, silences=[]) == []


def test_choose_split_points_no_silences_falls_back_to_target() -> None:
    points = _choose_split_points(
        duration=30.0, target_duration=10.0, silences=[]
    )
    assert points == [10.0, 20.0]


def test_choose_split_points_picks_silence_midpoint() -> None:
    """Silence at 9.5-10.0 (midpoint 9.75) is inside the ±25% window of 10s."""
    points = _choose_split_points(
        duration=30.0,
        target_duration=10.0,
        silences=[(9.5, 10.0)],
    )
    assert points[0] == pytest.approx(9.75)


def test_choose_split_points_prefers_longest_silence_in_window() -> None:
    points = _choose_split_points(
        duration=30.0,
        target_duration=10.0,
        # both inside [7.5, 12.5] window; second is longer
        silences=[(8.0, 8.5), (10.0, 11.5)],
    )
    assert points[0] == pytest.approx(10.75)


def test_choose_split_points_dedups_and_sorts() -> None:
    """Pathological silence list should still produce strictly-increasing
    in-bounds points."""
    points = _choose_split_points(
        duration=20.0,
        target_duration=10.0,
        silences=[(9.0, 11.0)],
    )
    # midpoint = 10.0, only one boundary expected
    assert points == [10.0]
    assert all(0 < p < 20.0 for p in points)


# ---------------------------------------------------------------------------
# _measure_bytes_per_second
# ---------------------------------------------------------------------------


def test_measure_bytes_per_second_returns_none_for_zero_duration() -> None:
    assert _measure_bytes_per_second("/tmp/anything", 0.0) is None


def test_measure_bytes_per_second_returns_none_for_missing_file() -> None:
    assert _measure_bytes_per_second("/tmp/no-such-file-zzz.flac", 1.0) is None


def test_measure_bytes_per_second_divides_size_by_duration(tmp_path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"\x00" * 1000)
    assert _measure_bytes_per_second(str(p), 2.0) == 500.0


# ---------------------------------------------------------------------------
# Full integration: real ffmpeg pipeline against a tiny WAV
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_pass_through_when_under_cap(tmp_path) -> None:
    """A 1-second tone fits under any reasonable cap and emits a single
    chunk whose path is the normalized FLAC."""
    src = tmp_path / "tone.wav"
    _write_wav(str(src), duration_s=1.0)

    chunks, tmpdir = await split_audio(str(src), cap_bytes=100 * 1024 * 1024)
    try:
        assert len(chunks) == 1
        assert chunks[0].start_offset_s == 0.0
        # Normalized FLAC sits inside tmpdir
        assert chunks[0].path.startswith(tmpdir)
        assert os.path.basename(chunks[0].path) == "normalized.flac"
        assert chunks[0].duration_s == pytest.approx(1.0, abs=0.05)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_full_pipeline_splits_when_over_cap(tmp_path) -> None:
    """A 30-second source with a 0.6-second silence near the 15-second
    midpoint should produce two FLAC chunks (one per side)."""
    src = tmp_path / "long.wav"
    _write_wav(
        str(src),
        duration_s=30.0,
        silent_ranges=[(14.5, 15.5)],
    )

    # Force splitting by setting an aggressive cap that forces target
    # duration to ~15 seconds (32 KB/s × 0.8 ≈ 26 KB/s, 400 KB cap →
    # 15.5 s target).
    chunks, tmpdir = await split_audio(str(src), cap_bytes=400 * 1024)
    try:
        assert len(chunks) >= 2
        # All chunks contiguous
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt.start_offset_s == pytest.approx(
                prev.start_offset_s + prev.duration_s, abs=0.5
            )
        # First chunk starts at zero
        assert chunks[0].start_offset_s == 0.0
        # Sum of durations ≈ source duration
        total = sum(c.duration_s for c in chunks)
        assert total == pytest.approx(30.0, abs=1.0)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_full_pipeline_invalid_input_is_fatal(tmp_path) -> None:
    """A non-audio file should fail at the normalize step → FatalError."""
    bogus = tmp_path / "garbage.wav"
    bogus.write_bytes(b"\x00\x01\x02\x03")

    with pytest.raises(FatalError):
        await split_audio(str(bogus), cap_bytes=100 * 1024 * 1024)


# ---------------------------------------------------------------------------
# silencedetect parser — no real ffmpeg needed
# ---------------------------------------------------------------------------


_SAMPLE_SILENCEDETECT_STDERR = b"""\
[silencedetect @ 0x10] silence_start: 0.500000
[silencedetect @ 0x10] silence_end: 1.000000 | silence_duration: 0.500000
[silencedetect @ 0x10] silence_start: 5.250000
[silencedetect @ 0x10] silence_end: 6.0 | silence_duration: 0.750000
[silencedetect @ 0x10] silence_start: -0.000123
[silencedetect @ 0x10] silence_end: 0.001 | silence_duration: 0.001
"""


@pytest.mark.asyncio
async def test_detect_silences_parses_ffmpeg_output(tmp_path, monkeypatch) -> None:
    """The parser must extract every (start, end) pair and accept the
    negative-zero-ish starts FFmpeg emits in the wild."""

    class _FakeProc:
        returncode = 0

        async def communicate(self):
            return b"", _SAMPLE_SILENCEDETECT_STDERR

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FakeProc()

    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", fake_create_subprocess_exec
    )

    silences = await _detect_silences("ignored.flac")
    assert silences == [
        (0.5, 1.0),
        (5.25, 6.0),
        (-0.000123, 0.001),
    ]


@pytest.mark.asyncio
async def test_detect_silences_failure_is_fatal(monkeypatch) -> None:
    class _FakeProc:
        returncode = 1

        async def communicate(self):
            return b"", b"ffmpeg: failed"

    async def fake(*args, **kwargs):
        return _FakeProc()

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake)

    with pytest.raises(FatalError, match="silencedetect failed"):
        await _detect_silences("ignored.flac")


@pytest.mark.asyncio
async def test_detect_silences_empty_when_no_matches(monkeypatch) -> None:
    class _FakeProc:
        returncode = 0

        async def communicate(self):
            return b"", b"some unrelated ffmpeg log line"

    async def fake(*args, **kwargs):
        return _FakeProc()

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake)

    assert await _detect_silences("ignored.flac") == []


# ---------------------------------------------------------------------------
# Min chunk merge guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_split_audio_merges_too_short_tail(tmp_path) -> None:
    """A 17-second source with a forced 16-second target produces two
    boundaries; the tail (1 second) must be merged into the previous
    chunk so no chunk drops below ``MIN_CHUNK_DURATION_S``."""
    src = tmp_path / "long.wav"
    _write_wav(str(src), duration_s=17.0)

    # cap = 16s × 32KB/s / 0.8 ≈ 640 KB. The tail at 16-17s would be 1s
    # which is below the MIN guard; expect a single merged chunk
    # spanning roughly the full duration.
    chunks, tmpdir = await split_audio(
        str(src), cap_bytes=int(16 * 32 * 1024 / 0.8)
    )
    try:
        for c in chunks:
            assert c.duration_s >= MIN_CHUNK_DURATION_S - 0.5
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
