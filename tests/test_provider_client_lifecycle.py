"""Provider client lifecycle tests.

Cloud transcription providers used to store an ``httpx.AsyncClient`` /
``AsyncOpenAI`` instance on ``self`` in ``__init__``. Because
``get_provider()`` returns a fresh provider per request, those
clients were never ``aclose()``-d and leaked sockets / fds under
batch indexing.

This test file pins the new contract: each ``transcribe()`` call
constructs a short-lived client (or reuses an injected mock for
testing) and releases it before returning. We assert this by
counting how many clients are constructed across two back-to-back
calls — each call must build its own.

Hako pattern: ``W0F1YQspXF-lVYgaDb6V1``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from app.workers.transcription.deepgram import DeepgramProvider
from app.workers.transcription.elevenlabs_scribe import ElevenLabsScribeProvider


@pytest.fixture()
def deepgram_with_key(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    yield


@pytest.fixture()
def elevenlabs_with_key(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    yield


@pytest.fixture()
def fake_audio(tmp_path):
    p = tmp_path / "clip.wav"
    p.write_bytes(b"\x00" * 1024)
    return str(p)


def _ok_handler() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        # Both Deepgram and ElevenLabs accept "no words" as
        # succeeded-with-zero. We don't care about parse output here —
        # we only care that the call lifecycle is clean.
        return httpx.Response(
            200,
            json={
                "results": {"channels": [{"alternatives": [{"words": []}]}]},
                "metadata": {},
                "words": [],
            },
        )

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_deepgram_does_not_store_long_lived_client(
    deepgram_with_key, fake_audio
) -> None:
    """``__init__`` must NOT create an httpx.AsyncClient.

    The previous design built one in ``__init__`` and never closed it;
    each ``get_provider()`` call leaked a new socket pool.
    """
    provider = DeepgramProvider()
    # The new lifecycle uses ``_transport`` for test injection and
    # never stores a client on the instance.
    assert not hasattr(provider, "_client") or provider.__dict__.get("_client") is None


@pytest.mark.asyncio
async def test_elevenlabs_does_not_store_long_lived_client(
    elevenlabs_with_key, fake_audio
) -> None:
    provider = ElevenLabsScribeProvider()
    assert not hasattr(provider, "_client") or provider.__dict__.get("_client") is None


@pytest.mark.asyncio
async def test_deepgram_constructs_fresh_client_per_call(
    deepgram_with_key, fake_audio
) -> None:
    """Each ``transcribe()`` must build its own AsyncClient.

    We assert this by counting constructor calls across two back-to-
    back invocations on the same provider instance.
    """
    construct_count = 0
    real_client = httpx.AsyncClient

    transport = _ok_handler()

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        nonlocal construct_count
        construct_count += 1
        kwargs.pop("transport", None)
        return real_client(transport=transport, **kwargs)

    provider = DeepgramProvider()
    provider._transport = transport

    with patch.object(httpx, "AsyncClient", factory):
        await provider.transcribe(fake_audio)
        await provider.transcribe(fake_audio)

    assert construct_count == 2, (
        "Deepgram must build a new AsyncClient per transcribe() call so "
        "sockets are released between jobs"
    )


@pytest.mark.asyncio
async def test_elevenlabs_constructs_fresh_client_per_call(
    elevenlabs_with_key, fake_audio
) -> None:
    construct_count = 0
    real_client = httpx.AsyncClient

    transport = _ok_handler()

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        nonlocal construct_count
        construct_count += 1
        kwargs.pop("transport", None)
        return real_client(transport=transport, **kwargs)

    provider = ElevenLabsScribeProvider()
    provider._transport = transport

    with patch.object(httpx, "AsyncClient", factory):
        await provider.transcribe(fake_audio)
        await provider.transcribe(fake_audio)

    assert construct_count == 2
