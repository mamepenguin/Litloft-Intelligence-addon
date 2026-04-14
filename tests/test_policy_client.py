"""Per-drive intelligence policy client behaviour."""

import pytest

from app import policy_client


@pytest.fixture(autouse=True)
def _reset_cache():
    policy_client.reset_cache()
    yield
    policy_client.reset_cache()


def test_evaluate_empty_means_enabled():
    assert policy_client._evaluate_response({}, "auto_tags") is True


def test_evaluate_all_shorthand_disables_everything():
    assert policy_client._evaluate_response({"_all": False}, "rag") is False
    assert policy_client._evaluate_response({"_all": True}, "rag") is True


def test_evaluate_per_feature_dict():
    payload = {"index": True, "auto_tags": False}
    assert policy_client._evaluate_response(payload, "index") is True
    assert policy_client._evaluate_response(payload, "auto_tags") is False
    # Unknown feature → default True (graceful degradation).
    assert policy_client._evaluate_response(payload, "rag") is True


@pytest.mark.asyncio
async def test_is_feature_enabled_caches_after_first_call(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        status_code = 200
        def json(self_inner):
            return {"auto_tags": False}

    class _Client:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self_inner): return self_inner
        async def __aexit__(self_inner, *a): return None
        async def get(self_inner, url, params=None):
            calls["n"] += 1
            return _Resp()

    monkeypatch.setattr(policy_client.httpx, "AsyncClient", _Client)

    a = await policy_client.is_feature_enabled("work", "auto_tags")
    b = await policy_client.is_feature_enabled("work", "auto_tags")
    assert a is False and b is False
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_is_feature_enabled_fails_open_on_network_error(monkeypatch):
    """Transient network failure must not silently disable real work."""

    class _BoomClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self_inner): return self_inner
        async def __aexit__(self_inner, *a): return None
        async def get(self_inner, url, params=None):
            import httpx
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(policy_client.httpx, "AsyncClient", _BoomClient)
    assert await policy_client.is_feature_enabled("any", "rag") is True


@pytest.mark.asyncio
async def test_is_feature_enabled_404_treated_as_disabled(monkeypatch):
    """An unknown drive (removed from drives.json) is treated as off."""

    class _Resp:
        status_code = 404
        def json(self_inner): return {}

    class _Client:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self_inner): return self_inner
        async def __aexit__(self_inner, *a): return None
        async def get(self_inner, url, params=None):
            return _Resp()

    monkeypatch.setattr(policy_client.httpx, "AsyncClient", _Client)
    assert await policy_client.is_feature_enabled("ghost", "index") is False
