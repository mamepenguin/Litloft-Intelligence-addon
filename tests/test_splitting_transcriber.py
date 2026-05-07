"""Tests for the SplittingTranscriber decorator.

Covered surface:

* Pass-through path: provider with ``max_input_bytes=None`` is never
  split; provider with a cap but file ≤ cap is also pass-through.
* Split path: file > cap triggers ffmpeg pipeline (mocked here),
  per-chunk transcribe is called, segments are stitched with
  timestamp offsets, prior tails are forwarded only when the inner
  provider declares ``accepts_initial_prompt=True``.
* Per-chunk retry: a transient on chunk N retries chunk N alone;
  segments from chunks 0..N-1 are not lost.
* Lifecycle: tmpdir cleanup runs on success and on failure.
* ``capabilities.handles_own_retry`` is True regardless of inner.
* ``name`` proxies the inner provider's name.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.workers.transcription.base import (
    ProviderCapabilities,
    TranscriptionSegment,
    WordToken,
)
from app.workers.transcription.errors import (
    FatalError,
    TransientError,
)
from app.workers.transcription.splitter import AudioChunk
from app.workers.transcription.splitting_transcriber import (
    SplittingTranscriber,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _caps(
    *,
    max_input_bytes: int | None = 100,
    accepts_initial_prompt: bool = True,
) -> ProviderCapabilities:
    return ProviderCapabilities(
        sends_audio_offhost=True,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
        max_input_bytes=max_input_bytes,
        accepts_initial_prompt=accepts_initial_prompt,
        handles_own_retry=False,
    )


def _segment(text: str, start: float, end: float) -> TranscriptionSegment:
    return TranscriptionSegment(
        text=text,
        start=start,
        end=end,
        language="en",
        words=[WordToken(text=text, start=start, end=end)],
    )


@pytest.fixture()
def small_file(tmp_path):
    p = tmp_path / "small.wav"
    p.write_bytes(b"\x00" * 50)  # 50 bytes < cap 100
    return str(p)


@pytest.fixture()
def big_file(tmp_path):
    p = tmp_path / "big.wav"
    p.write_bytes(b"\x00" * 10_000)  # 10 KB > cap 100
    return str(p)


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_capabilities_overrides_handles_own_retry() -> None:
    inner = MagicMock()
    inner.capabilities = _caps()
    inner.name = "inner-name"
    wrapper = SplittingTranscriber(inner)
    assert wrapper.capabilities.handles_own_retry is True
    # Other fields are inherited verbatim
    assert wrapper.capabilities.max_input_bytes == 100
    assert wrapper.capabilities.accepts_initial_prompt is True


def test_name_proxies_inner_provider() -> None:
    inner = MagicMock()
    inner.capabilities = _caps()
    inner.name = "openai_compatible"
    wrapper = SplittingTranscriber(inner)
    assert wrapper.name == "openai_compatible"


# ---------------------------------------------------------------------------
# Pass-through paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_cap_passes_through_directly(tmp_path) -> None:
    p = tmp_path / "x.wav"
    p.write_bytes(b"x" * 1024)

    inner = MagicMock()
    inner.capabilities = _caps(max_input_bytes=None)
    inner.name = "uncapped"
    inner.transcribe = AsyncMock(return_value=[_segment("hi", 0, 1)])

    wrapper = SplittingTranscriber(inner)
    result = await wrapper.transcribe(str(p))

    assert result == [_segment("hi", 0, 1)]
    inner.transcribe.assert_awaited_once()
    args, kwargs = inner.transcribe.await_args
    assert kwargs.get("initial_prompt") is None


@pytest.mark.asyncio
async def test_under_cap_passes_through_directly(small_file) -> None:
    inner = MagicMock()
    inner.capabilities = _caps(max_input_bytes=100)
    inner.name = "capped"
    inner.transcribe = AsyncMock(return_value=[_segment("hi", 0, 1)])

    wrapper = SplittingTranscriber(inner)
    result = await wrapper.transcribe(small_file, initial_prompt="prior")

    assert result == [_segment("hi", 0, 1)]
    args, kwargs = inner.transcribe.await_args
    assert kwargs["initial_prompt"] == "prior"


@pytest.mark.asyncio
async def test_missing_file_is_fatal(tmp_path) -> None:
    inner = MagicMock()
    inner.capabilities = _caps(max_input_bytes=100)

    wrapper = SplittingTranscriber(inner)
    with pytest.raises(FatalError, match="Cannot stat"):
        await wrapper.transcribe(str(tmp_path / "no-such-file.wav"))


# ---------------------------------------------------------------------------
# Split path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_split_path_calls_inner_per_chunk_and_offsets(big_file) -> None:
    inner = MagicMock()
    inner.capabilities = _caps(
        max_input_bytes=100, accepts_initial_prompt=False
    )
    inner.name = "deepgram"

    fake_chunks = [
        AudioChunk(path="/tmp/c0.flac", start_offset_s=0.0, duration_s=10.0),
        AudioChunk(path="/tmp/c1.flac", start_offset_s=10.0, duration_s=10.0),
    ]
    inner.transcribe = AsyncMock(
        side_effect=[
            [_segment("hello", 0.0, 1.0)],
            [_segment("world", 0.5, 1.5)],
        ]
    )

    with (
        patch(
            "app.workers.transcription.splitting_transcriber.split_audio",
            return_value=(fake_chunks, "/tmp/td"),
        ),
        patch(
            "app.workers.transcription.splitting_transcriber.shutil.rmtree"
        ) as rmtree_mock,
    ):
        wrapper = SplittingTranscriber(inner)
        result = await wrapper.transcribe(big_file)

    # Two segments, second offset by 10s
    assert len(result) == 2
    assert result[0].start == 0.0
    assert result[0].end == 1.0
    assert result[1].start == 10.5
    assert result[1].end == 11.5

    # tmpdir cleanup ran
    rmtree_mock.assert_called_once_with("/tmp/td", ignore_errors=True)


@pytest.mark.asyncio
async def test_split_path_passes_prior_tail_when_accepted(big_file) -> None:
    inner = MagicMock()
    inner.capabilities = _caps(
        max_input_bytes=100, accepts_initial_prompt=True
    )
    inner.name = "openai_compatible"

    fake_chunks = [
        AudioChunk(path="/tmp/c0.flac", start_offset_s=0.0, duration_s=10.0),
        AudioChunk(path="/tmp/c1.flac", start_offset_s=10.0, duration_s=10.0),
    ]
    inner.transcribe = AsyncMock(
        side_effect=[
            [_segment("hello world from chunk zero", 0, 1)],
            [_segment("chunk one continues", 0, 1)],
        ]
    )

    with patch(
        "app.workers.transcription.splitting_transcriber.split_audio",
        return_value=(fake_chunks, "/tmp/td"),
    ), patch(
        "app.workers.transcription.splitting_transcriber.shutil.rmtree"
    ):
        wrapper = SplittingTranscriber(inner)
        await wrapper.transcribe(big_file)

    # Chunk 0 gets the caller's initial_prompt (None here).
    first_call_kwargs = inner.transcribe.await_args_list[0].kwargs
    assert first_call_kwargs["initial_prompt"] is None
    # Chunk 1 gets the prior chunk's tail text.
    second_call_kwargs = inner.transcribe.await_args_list[1].kwargs
    assert second_call_kwargs["initial_prompt"] == (
        "hello world from chunk zero"
    )


@pytest.mark.asyncio
async def test_split_path_omits_prior_tail_when_not_accepted(big_file) -> None:
    inner = MagicMock()
    inner.capabilities = _caps(
        max_input_bytes=100, accepts_initial_prompt=False
    )
    inner.name = "deepgram"

    fake_chunks = [
        AudioChunk(path="/tmp/c0.flac", start_offset_s=0.0, duration_s=10.0),
        AudioChunk(path="/tmp/c1.flac", start_offset_s=10.0, duration_s=10.0),
    ]
    inner.transcribe = AsyncMock(
        side_effect=[
            [_segment("first chunk", 0, 1)],
            [_segment("second chunk", 0, 1)],
        ]
    )

    with patch(
        "app.workers.transcription.splitting_transcriber.split_audio",
        return_value=(fake_chunks, "/tmp/td"),
    ), patch(
        "app.workers.transcription.splitting_transcriber.shutil.rmtree"
    ):
        wrapper = SplittingTranscriber(inner)
        await wrapper.transcribe(big_file)

    for call in inner.transcribe.await_args_list:
        assert call.kwargs["initial_prompt"] is None


@pytest.mark.asyncio
async def test_split_path_cleans_up_tmpdir_on_failure(big_file) -> None:
    inner = MagicMock()
    inner.capabilities = _caps(max_input_bytes=100)
    inner.name = "openai_compatible"

    fake_chunks = [
        AudioChunk(path="/tmp/c0.flac", start_offset_s=0.0, duration_s=10.0),
    ]
    inner.transcribe = AsyncMock(side_effect=FatalError("api dead"))

    with patch(
        "app.workers.transcription.splitting_transcriber.split_audio",
        return_value=(fake_chunks, "/tmp/td"),
    ), patch(
        "app.workers.transcription.splitting_transcriber.shutil.rmtree"
    ) as rmtree_mock:
        wrapper = SplittingTranscriber(inner)
        with pytest.raises(FatalError):
            await wrapper.transcribe(big_file)

    rmtree_mock.assert_called_once_with("/tmp/td", ignore_errors=True)


# ---------------------------------------------------------------------------
# Per-chunk retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_chunk_retry_preserves_earlier_segments(big_file) -> None:
    """Chunk 0 succeeds → chunk 1 transient on first attempt → succeeds
    on retry. Chunk 0 segments must be present in the final output and
    chunk 0 transcribe must NOT have been called twice."""
    inner = MagicMock()
    inner.capabilities = _caps(
        max_input_bytes=100, accepts_initial_prompt=False
    )
    inner.name = "openai_compatible"

    fake_chunks = [
        AudioChunk(path="/tmp/c0.flac", start_offset_s=0.0, duration_s=10.0),
        AudioChunk(path="/tmp/c1.flac", start_offset_s=10.0, duration_s=10.0),
    ]

    call_log: list[str] = []

    async def transcribe_side_effect(file_path, **kwargs):
        call_log.append(file_path)
        if file_path == "/tmp/c0.flac":
            return [_segment("alpha", 0, 1)]
        if file_path == "/tmp/c1.flac":
            # Fail the first attempt; succeed on the retry.
            chunk1_calls = sum(1 for p in call_log if p == "/tmp/c1.flac")
            if chunk1_calls == 1:
                raise TransientError("flaky cloud")
            return [_segment("bravo", 0, 1)]
        raise AssertionError(f"unexpected path: {file_path}")

    inner.transcribe = AsyncMock(side_effect=transcribe_side_effect)

    async def instant_sleep(_):
        return None

    with patch(
        "app.workers.transcription.splitting_transcriber.split_audio",
        return_value=(fake_chunks, "/tmp/td"),
    ), patch(
        "app.workers.transcription.splitting_transcriber.shutil.rmtree"
    ), patch("asyncio.sleep", new=instant_sleep):
        wrapper = SplittingTranscriber(inner)
        result = await wrapper.transcribe(big_file)

    # Chunk 0 once, chunk 1 twice (one transient + one success)
    assert call_log.count("/tmp/c0.flac") == 1
    assert call_log.count("/tmp/c1.flac") == 2
    # Both segments present, with correct offsets
    assert len(result) == 2
    assert result[0].text == "alpha"
    assert result[1].start == 10.0
    assert result[1].text == "bravo"


@pytest.mark.asyncio
async def test_per_chunk_fatal_aborts_immediately(big_file) -> None:
    """A fatal on chunk 0 must propagate without invoking chunk 1."""
    inner = MagicMock()
    inner.capabilities = _caps(max_input_bytes=100)
    inner.name = "openai_compatible"

    fake_chunks = [
        AudioChunk(path="/tmp/c0.flac", start_offset_s=0.0, duration_s=10.0),
        AudioChunk(path="/tmp/c1.flac", start_offset_s=10.0, duration_s=10.0),
    ]

    inner.transcribe = AsyncMock(side_effect=FatalError("401"))

    with patch(
        "app.workers.transcription.splitting_transcriber.split_audio",
        return_value=(fake_chunks, "/tmp/td"),
    ), patch(
        "app.workers.transcription.splitting_transcriber.shutil.rmtree"
    ):
        wrapper = SplittingTranscriber(inner)
        with pytest.raises(FatalError):
            await wrapper.transcribe(big_file)

    # Only chunk 0 was attempted.
    assert inner.transcribe.await_count == 1
