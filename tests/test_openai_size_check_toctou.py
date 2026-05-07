"""TOCTOU contract for the OpenAI 25 MB pre-check.

The pre-check used to ``os.path.getsize(file_path)`` *before*
``open(file_path, "rb")``. A racy mutation between the two calls
(file replaced or grown to >25 MB) would let the SDK upload bytes
that did not match the size we vetted, and the user would see a
cryptic SDK 413 instead of our actionable error message.

The fix stats the open file descriptor (``os.fstat(audio.fileno())``)
*inside* the same ``with open(...)`` block, so the size we check and
the bytes the SDK reads come from the same fd — no race.

This test exercises the worst case directly: we patch
``os.path.getsize`` to lie about the size, while the fd's true size
is >25 MB. With the old (path-based) check, the call would proceed
to the SDK; with the fd-based check, ``FatalError`` fires.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.workers.transcription.errors import FatalError
from app.workers.transcription.openai_compatible import OpenAICompatibleProvider


@pytest.fixture()
def with_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    yield


@pytest.mark.asyncio
async def test_size_check_uses_open_fd_not_path(
    with_api_key, tmp_path, monkeypatch
) -> None:
    """``os.fstat(audio.fileno())`` is the source of truth, not ``getsize``.

    We simulate a TOCTOU race by patching ``os.path.getsize`` to
    return a small (under-cap) size, while writing 26 MB to disk.
    A correct implementation reads ``os.fstat(audio.fileno())`` and
    raises FatalError; an old implementation that trusted
    ``os.path.getsize`` would let the call continue.
    """
    big = tmp_path / "big.wav"
    big.write_bytes(b"\x00" * (26 * 1024 * 1024))

    # Lie about the path-based size — pretend it's tiny.
    real_getsize = os.path.getsize

    def lying_getsize(path):
        if str(path) == str(big):
            return 1024  # under the cap
        return real_getsize(path)

    monkeypatch.setattr(os.path, "getsize", lying_getsize)

    with patch(
        "app.workers.transcription.openai_compatible.config"
    ) as fake_config:
        fake_config.settings.transcription.openai_compatible.base_url = (
            "https://api.openai.com/v1"
        )
        fake_config.settings.transcription.openai_compatible.model = "whisper-1"
        fake_config.settings.transcription.openai_compatible.timeout_s = 600
        provider = OpenAICompatibleProvider()

        with pytest.raises(FatalError, match="25MB"):
            await provider.transcribe(str(big))


@pytest.mark.asyncio
async def test_pre_check_size_fd_directly(with_api_key, tmp_path) -> None:
    """Direct unit test for the fd-based helper.

    Bypasses ``transcribe()`` plumbing to exercise just the fd path.
    """
    big = tmp_path / "big.wav"
    big.write_bytes(b"\x00" * (26 * 1024 * 1024))

    with patch(
        "app.workers.transcription.openai_compatible.config"
    ) as fake_config:
        fake_config.settings.transcription.openai_compatible.base_url = (
            "https://api.openai.com/v1"
        )
        fake_config.settings.transcription.openai_compatible.model = "whisper-1"
        fake_config.settings.transcription.openai_compatible.timeout_s = 600
        provider = OpenAICompatibleProvider()

    with open(str(big), "rb") as fp:
        with pytest.raises(FatalError, match="25MB"):
            provider._pre_check_size_fd(fp.fileno(), str(big))


@pytest.mark.asyncio
async def test_pre_check_size_fd_passes_under_limit(
    with_api_key, tmp_path
) -> None:
    """Below-cap files must pass cleanly."""
    small = tmp_path / "small.wav"
    small.write_bytes(b"\x00" * 1024)

    with patch(
        "app.workers.transcription.openai_compatible.config"
    ) as fake_config:
        fake_config.settings.transcription.openai_compatible.base_url = (
            "https://api.openai.com/v1"
        )
        fake_config.settings.transcription.openai_compatible.model = "whisper-1"
        fake_config.settings.transcription.openai_compatible.timeout_s = 600
        provider = OpenAICompatibleProvider()

    with open(str(small), "rb") as fp:
        # Must not raise.
        provider._pre_check_size_fd(fp.fileno(), str(small))
