"""Per-drive intelligence policy client behaviour."""

import pytest

from app import policy_client


@pytest.fixture(autouse=True)
def _reset_cache():
    policy_client.reset_cache()
    yield
    policy_client.reset_cache()


def test_evaluate_empty_means_enabled():
    assert policy_client._evaluate_response(
        {"default": True, "features": {}}, "auto_tags",
    ) is True


def test_evaluate_default_false_disables_everything():
    payload = {"default": False, "features": {}}
    assert policy_client._evaluate_response(payload, "rag") is False
    assert policy_client._evaluate_response(payload, "index") is False


def test_evaluate_per_feature_overrides_default():
    payload = {
        "default": True,
        "features": {"index": True, "auto_tags": False},
    }
    assert policy_client._evaluate_response(payload, "index") is True
    assert policy_client._evaluate_response(payload, "auto_tags") is False
    # Unknown feature falls back to the explicit default (True here).
    assert policy_client._evaluate_response(payload, "rag") is True


def test_evaluate_named_feature_can_override_default_false():
    payload = {
        "default": False,
        "features": {"index": True},
    }
    assert policy_client._evaluate_response(payload, "index") is True
    assert policy_client._evaluate_response(payload, "rag") is False


def test_evaluate_malformed_payload_fails_open():
    """Schema mismatch must not silently disable real work."""
    assert policy_client._evaluate_response({}, "rag") is True
    assert policy_client._evaluate_response({"features": "not-a-dict"}, "x") is True


@pytest.mark.asyncio
async def test_is_feature_enabled_caches_after_first_call(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        status_code = 200
        def json(self_inner):
            return {"default": True, "features": {"auto_tags": False}}

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


def _client_returning(outcome):
    class _Resp:
        def __init__(self, status, body):
            self.status_code = status
            self._body = body

        def json(self):
            if isinstance(self._body, Exception):
                raise self._body
            return self._body

    class _Client:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def get(self, url, params=None):
            if isinstance(outcome, Exception):
                raise outcome
            return _Resp(*outcome)

    return _Client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ((200, {"default": True, "features": {}}), "allowed"),
        ((200, {"default": True, "features": {"llm_cloud": False}}), "denied"),
        ((200, {"default": False, "features": {}}), "denied"),
        ((200, {"default": False, "features": {"llm_cloud": True}}), "allowed"),
        ((404, {}), "denied"),
        ((500, {}), "unknown"),
        ((200, {}), "unknown"),
        ((200, {"features": {}}), "unknown"),
        ((200, {"default": "yes", "features": {}}), "unknown"),
        ((200, ["not", "a", "dict"]), "unknown"),
        ((200, {"default": True, "features": {"llm_cloud": "yes"}}), "denied"),
        ((200, {"default": True, "features": "x"}), "unknown"),
        ((200, ValueError("not json")), "unknown"),
        (__import__("httpx").ConnectError("boom"), "unknown"),
    ],
)
async def test_lookup_feature_separates_denied_from_unknown(
    monkeypatch, outcome, expected
):
    monkeypatch.setattr(policy_client.httpx, "AsyncClient", _client_returning(outcome))
    assert await policy_client.lookup_feature("d", "llm_cloud") == expected


@pytest.mark.asyncio
async def test_lookup_feature_does_not_cache_unknown(monkeypatch):
    monkeypatch.setattr(
        policy_client.httpx, "AsyncClient", _client_returning((500, {}))
    )
    assert await policy_client.lookup_feature("d", "llm_cloud") == "unknown"
    monkeypatch.setattr(
        policy_client.httpx,
        "AsyncClient",
        _client_returning((200, {"default": True, "features": {}})),
    )
    assert await policy_client.lookup_feature("d", "llm_cloud") == "allowed"


@pytest.mark.asyncio
async def test_lookup_feature_does_not_cache_a_malformed_body(monkeypatch):
    monkeypatch.setattr(
        policy_client.httpx, "AsyncClient", _client_returning((200, {"features": {}}))
    )
    assert await policy_client.lookup_feature("d", "llm_cloud") == "unknown"
    monkeypatch.setattr(
        policy_client.httpx,
        "AsyncClient",
        _client_returning((200, {"default": False, "features": {}})),
    )
    assert await policy_client.lookup_feature("d", "llm_cloud") == "denied"


@pytest.mark.asyncio
async def test_lookup_feature_asks_about_the_given_drive(monkeypatch):
    asked = []

    class _Client:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def get(self, url, params=None):
            asked.append((url.rsplit("/", 1)[-1], params))
            raise __import__("httpx").ConnectError("boom")

    monkeypatch.setattr(policy_client.httpx, "AsyncClient", _Client)
    await policy_client.lookup_feature("private", "llm_cloud")

    assert asked == [("drive-policy", {"drive": "private", "addon": "intelligence"})]


@pytest.mark.asyncio
async def test_fail_open_reading_never_answers_a_strict_lookup(monkeypatch):
    monkeypatch.setattr(
        policy_client.httpx, "AsyncClient", _client_returning((200, {}))
    )
    assert await policy_client.is_feature_enabled("d", "llm_cloud") is True
    monkeypatch.setattr(
        policy_client.httpx, "AsyncClient", _client_returning((500, {}))
    )

    assert await policy_client.lookup_feature("d", "llm_cloud") == "unknown"
