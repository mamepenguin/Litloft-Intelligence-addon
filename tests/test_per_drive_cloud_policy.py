"""Phase 1A foundation tests for per-drive cloud policy semantics.

The cloud-transcription policy is fail-CLOSED: when the policy
backend says "no" or cannot be reached after the grace period, we
must NOT send audio to a cloud provider. ``default_on_failure=False``
encodes that posture.

These tests pin the policy_client behaviour at the boundary between
"grace period over" and "policy explicitly disabled".
"""

from __future__ import annotations

import sys
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
def reset_state():
    from app import policy_client

    policy_client.reset_cache()
    policy_client._reset_grace_period_for_tests()
    yield
    policy_client.reset_cache()
    policy_client._reset_grace_period_for_tests()


def _async_client_factory(responses):
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


def _ok(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value=payload)
    return resp


@pytest.mark.asyncio
async def test_explicit_false_in_features_returns_false() -> None:
    """``transcription_cloud: false`` in the features dict must beat
    the default value, regardless of default_on_failure."""
    from app import policy_client

    client_factory = _async_client_factory([
        _ok({"default": True, "features": {"transcription_cloud": False}}),
    ])

    with patch.object(httpx, "AsyncClient", client_factory):
        result = await policy_client.is_feature_enabled(
            "secret",
            "transcription_cloud",
            default_on_failure=False,
        )
    assert result is False


@pytest.mark.asyncio
async def test_explicit_true_in_features_returns_true() -> None:
    from app import policy_client

    client_factory = _async_client_factory([
        _ok({"default": False, "features": {"transcription_cloud": True}}),
    ])

    with patch.object(httpx, "AsyncClient", client_factory):
        result = await policy_client.is_feature_enabled(
            "drive",
            "transcription_cloud",
            default_on_failure=False,
        )
    assert result is True


@pytest.mark.asyncio
async def test_404_with_default_false_returns_false() -> None:
    """A drive removed from drives.json must fail closed for cloud
    features instead of falling back to fail-open True."""
    from app import policy_client

    resp = MagicMock()
    resp.status_code = 404
    client_factory = _async_client_factory([resp])

    with patch.object(httpx, "AsyncClient", client_factory):
        result = await policy_client.is_feature_enabled(
            "removed_drive",
            "transcription_cloud",
            default_on_failure=False,
        )
    assert result is False


@pytest.mark.asyncio
async def test_500_after_grace_with_default_false_returns_false() -> None:
    """5xx after the grace period (and after a healthy 200 cleared the
    grace flag) fails closed under default_on_failure=False."""
    from app import policy_client

    err_resp = MagicMock()
    err_resp.status_code = 500
    client_factory = _async_client_factory([
        _ok({"default": True}),  # observe healthy first
        err_resp,                  # second call: 500
    ])

    with patch.object(httpx, "AsyncClient", client_factory):
        first = await policy_client.is_feature_enabled(
            "drive_a",
            "transcription_cloud",
            default_on_failure=False,
        )
        policy_client.reset_cache()
        second = await policy_client.is_feature_enabled(
            "drive_b",
            "transcription_cloud",
            default_on_failure=False,
        )

    assert first is True
    assert second is False


@pytest.mark.asyncio
async def test_500_with_default_true_returns_true() -> None:
    """Existing ``index`` callsites must keep their fail-open posture."""
    from app import policy_client

    err_resp = MagicMock()
    err_resp.status_code = 500
    client_factory = _async_client_factory([err_resp])

    with patch.object(httpx, "AsyncClient", client_factory):
        result = await policy_client.is_feature_enabled("drive", "index")

    assert result is True
