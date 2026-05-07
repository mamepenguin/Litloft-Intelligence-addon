"""Phase 1A foundation tests for the policy_client cold-start grace period.

The intelligence container can warm up before backend has a healthy
HTTP listener. With ``default_on_failure=False`` (cloud transcription
gate, fail-closed), the first jobs would be permanently marked failed
even though the real cause is "infra not ready yet".

The cold-start grace period treats ``ConnectionError`` / network
failures as :class:`TransientError` for the first ``STARTUP_GRACE_S``
seconds AND until the policy backend has returned at least one 200.
After that, the grace flag clears and normal fail-closed semantics
apply.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import httpx
import pytest

# Stub heavy ML deps before importing app.
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


@pytest.fixture(autouse=True)
def reset_policy_client_state():
    """Reset module-level cache + grace state between tests.

    The grace clock is module-level so a successful 200 in one test
    must not leak healthy-state into the next.
    """
    from app import policy_client

    policy_client.reset_cache()
    policy_client._reset_grace_period_for_tests()
    yield
    policy_client.reset_cache()
    policy_client._reset_grace_period_for_tests()


def _async_client_factory(responses):
    """Build an httpx.AsyncClient stub that yields the given responses.

    Each call to ``client.get`` returns the next item; a ``BaseException``
    instance raises (used to simulate ConnectError), anything else is
    returned as the response. List is consumed FIFO.
    """
    iterator = iter(responses)

    class _StubClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            item = next(iterator)
            if isinstance(item, BaseException):
                raise item
            return item

    return _StubClient


def _ok_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value=payload)
    return resp


@pytest.mark.asyncio
async def test_grace_period_treats_connection_error_as_transient() -> None:
    """During the grace period a ConnectError must surface as TransientError
    so the worker can retry instead of marking the job failed."""
    from app import policy_client
    from app.workers.transcription.errors import TransientError

    client_factory = _async_client_factory([
        httpx.ConnectError("backend not up yet"),
    ])

    with patch.object(httpx, "AsyncClient", client_factory):
        with pytest.raises(TransientError):
            await policy_client.is_feature_enabled(
                "secret",
                "transcription_cloud",
                default_on_failure=False,
            )


@pytest.mark.asyncio
async def test_default_on_failure_true_keeps_legacy_fail_open() -> None:
    """Existing call sites pass ``default_on_failure=True`` (the default).
    A network error must continue to return True so legitimate work
    keeps flowing for non-cloud features (search, indexing)."""
    from app import policy_client

    client_factory = _async_client_factory([
        httpx.ConnectError("blip"),
    ])

    with patch.object(httpx, "AsyncClient", client_factory):
        result = await policy_client.is_feature_enabled(
            "drive",
            "index",
        )
    assert result is True


@pytest.mark.asyncio
async def test_grace_period_clears_after_first_200() -> None:
    """Once the policy backend returns a 200 response, the grace flag is
    cleared. Subsequent ConnectionErrors with default_on_failure=False
    fail closed (return False) instead of raising TransientError."""
    from app import policy_client

    client_factory = _async_client_factory([
        _ok_response({"default": True, "features": {"transcription_cloud": True}}),
        httpx.ConnectError("backend dropped"),
    ])

    with patch.object(httpx, "AsyncClient", client_factory):
        first = await policy_client.is_feature_enabled(
            "drive",
            "transcription_cloud",
            default_on_failure=False,
        )
        # Force a fresh lookup — drop cache so the second call hits the
        # network again. ``reset_cache`` only purges the cache, not the
        # "we've seen a 200" health flag.
        policy_client.reset_cache()
        second = await policy_client.is_feature_enabled(
            "drive2",
            "transcription_cloud",
            default_on_failure=False,
        )

    assert first is True
    assert second is False  # grace cleared, fail-closed engaged


@pytest.mark.asyncio
async def test_grace_period_expires_by_time() -> None:
    """After STARTUP_GRACE_S seconds even without a successful 200 the
    grace flag expires. Without a successful 200 the policy still
    cannot resolve, so ``default_on_failure=False`` returns False."""
    from app import policy_client

    # Pin clock to 1000 + grace + 1 so the grace window has expired.
    fake_now = [1000.0]
    policy_client._set_startup_for_tests(0.0)

    def now():
        return fake_now[0]

    client_factory = _async_client_factory([
        httpx.ConnectError("still down"),
    ])

    with patch.object(httpx, "AsyncClient", client_factory), \
         patch.object(policy_client.time, "monotonic", now):
        result = await policy_client.is_feature_enabled(
            "drive",
            "transcription_cloud",
            default_on_failure=False,
        )
    assert result is False


@pytest.mark.asyncio
async def test_default_on_failure_false_with_200_returns_value() -> None:
    """A normal 200 with default_on_failure=False just returns the
    feature value — no surprise behaviour from the new arg."""
    from app import policy_client

    client_factory = _async_client_factory([
        _ok_response({"default": False, "features": {"transcription_cloud": False}}),
    ])

    with patch.object(httpx, "AsyncClient", client_factory):
        result = await policy_client.is_feature_enabled(
            "secret",
            "transcription_cloud",
            default_on_failure=False,
        )
    assert result is False
