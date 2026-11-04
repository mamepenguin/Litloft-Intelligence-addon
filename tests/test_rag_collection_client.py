"""Contract tests for the intelligence → core collection file-ids client.

Layer 1 pins the wire shape of
``GET /api/internal/collections/{collection_id}/file-ids``: method, URL,
the ``drive`` query parameter, the optional secret header, the timeout,
the response parse, and the exception each failure raises. Layer 2 holds
a hand-copied reference of the core's collection id pattern and response
key, checked against this client's behaviour so drift on either side
fails a test.
"""

from __future__ import annotations

import logging
import re

import httpx
import pytest

COLLECTION_ID = "AbCdEfGh1234"
DRIVE = "main"
FILE_ID_A = "aaaaaaaaaaaa"
FILE_ID_B = "bbbbbbbbbbbb"
FILE_ID_C = "cccccccccccc"

# Hand-copied from the core's ``schemas._NANOID_RE``. The core-side parity
# test points at this copy when the pattern changes.
_CORE_COLLECTION_ID_PATTERN = r"^[A-Za-z0-9_-]{12}$"
_CORE_FILE_IDS_KEY = "file_ids"
_CORE_ID_RE = re.compile(_CORE_COLLECTION_ID_PATTERN)


def _client():
    from app.rag import collection_client

    return collection_client


def _install_transport(monkeypatch, handler):
    """Send the client's httpx calls through a MockTransport.

    Returns a dict collecting each AsyncClient construction's kwargs, so a
    test can assert the configured timeout.
    """
    module = _client()
    seen: dict = {"client_kwargs": {}, "constructions": 0}
    orig_async_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        seen["client_kwargs"] = kwargs
        seen["constructions"] += 1
        kwargs["transport"] = httpx.MockTransport(handler)
        return orig_async_client(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", factory)
    return seen


def _ok_handler(requests, file_ids=None):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={_CORE_FILE_IDS_KEY: file_ids or []})

    return handler


def _core_accepts(candidate: str) -> bool:
    return _CORE_ID_RE.match(candidate) is not None


@pytest.fixture(autouse=True)
def _clear_internal_env(monkeypatch):
    monkeypatch.delenv("HOMEVAULT_INTERNAL_API_URL", raising=False)
    monkeypatch.delenv("CORE_INTERNAL_SECRET", raising=False)


class TestSpecAddon058I11:
    async def test_sends_get_to_the_collection_file_ids_endpoint(
        self, monkeypatch
    ):
        """SPEC-ADDON-058 I11: URL, method, drive as a query parameter."""
        requests: list[httpx.Request] = []
        _install_transport(
            monkeypatch, _ok_handler(requests, [FILE_ID_A])
        )

        result = await _client().fetch_collection_file_ids(
            collection_id=COLLECTION_ID, drive=DRIVE
        )

        assert result == [FILE_ID_A]
        assert len(requests) == 1
        request = requests[0]
        assert request.method == "GET"
        assert request.url.host == "backend"
        assert request.url.port == 8000
        assert request.url.path == (
            f"/api/internal/collections/{COLLECTION_ID}/file-ids"
        )
        assert dict(request.url.params) == {"drive": DRIVE}

    async def test_uses_the_configured_internal_api_base_url(
        self, monkeypatch
    ):
        """SPEC-ADDON-058 I11: the base URL comes from the environment."""
        requests: list[httpx.Request] = []
        _install_transport(monkeypatch, _ok_handler(requests))
        monkeypatch.setenv(
            "HOMEVAULT_INTERNAL_API_URL", "http://core:9000/api/internal"
        )

        await _client().fetch_collection_file_ids(
            collection_id=COLLECTION_ID, drive=DRIVE
        )

        assert len(requests) == 1
        request = requests[0]
        assert request.url.host == "core"
        assert request.url.port == 9000
        assert request.url.path == (
            f"/api/internal/collections/{COLLECTION_ID}/file-ids"
        )

    async def test_sends_secret_header_when_configured(self, monkeypatch):
        """SPEC-ADDON-058 I11: the secret header when the env is set."""
        requests: list[httpx.Request] = []
        _install_transport(monkeypatch, _ok_handler(requests))
        monkeypatch.setenv("CORE_INTERNAL_SECRET", "shared-secret")

        await _client().fetch_collection_file_ids(
            collection_id=COLLECTION_ID, drive=DRIVE
        )

        assert (
            requests[0].headers.get("x-internal-secret") == "shared-secret"
        )

    async def test_omits_secret_header_when_unset(self, monkeypatch):
        """SPEC-ADDON-058 I11: no secret header without the env."""
        requests: list[httpx.Request] = []
        _install_transport(monkeypatch, _ok_handler(requests))

        await _client().fetch_collection_file_ids(
            collection_id=COLLECTION_ID, drive=DRIVE
        )

        assert "x-internal-secret" not in requests[0].headers

    async def test_uses_a_five_second_timeout(self, monkeypatch):
        """SPEC-ADDON-058 I11: the request times out after 5 s."""
        requests: list[httpx.Request] = []
        seen = _install_transport(monkeypatch, _ok_handler(requests))

        await _client().fetch_collection_file_ids(
            collection_id=COLLECTION_ID, drive=DRIVE
        )

        assert seen["constructions"] == 1
        timeout = seen["client_kwargs"].get("timeout")
        assert timeout is not None
        if isinstance(timeout, httpx.Timeout):
            assert {
                timeout.connect,
                timeout.read,
                timeout.write,
                timeout.pool,
            } == {5.0}
        else:
            assert float(timeout) == 5.0

    async def test_returns_ids_in_the_core_order(self, monkeypatch):
        """SPEC-ADDON-058 I11: the core's order survives the parse."""
        requests: list[httpx.Request] = []
        core_order = [FILE_ID_C, FILE_ID_A, FILE_ID_B]
        _install_transport(
            monkeypatch, _ok_handler(requests, list(core_order))
        )

        result = await _client().fetch_collection_file_ids(
            collection_id=COLLECTION_ID, drive=DRIVE
        )

        assert isinstance(result, list)
        assert result == core_order

    async def test_returns_an_empty_list_for_an_empty_collection(
        self, monkeypatch
    ):
        """SPEC-ADDON-058 I11: [] is a successful empty answer."""
        requests: list[httpx.Request] = []
        _install_transport(monkeypatch, _ok_handler(requests, []))

        result = await _client().fetch_collection_file_ids(
            collection_id=COLLECTION_ID, drive=DRIVE
        )

        assert result == []

    @pytest.mark.parametrize(
        "malformed_id",
        [
            "abc",
            "AbCdEfGh123",
            "AbCdEfGh12345",
            "AbCdEfGh 234",
            "AbCdEfGh.234",
            "AbCdEfGh/234",
            "AbCdEfGh123!",
            "ああああああああああああ",
            "",
        ],
        ids=[
            "too-short",
            "eleven-chars",
            "thirteen-chars",
            "space",
            "dot",
            "slash",
            "bang",
            "non-ascii",
            "empty",
        ],
    )
    async def test_malformed_id_raises_not_found_without_a_request(
        self, monkeypatch, malformed_id
    ):
        """SPEC-ADDON-058 I11: a bad id never reaches the wire."""
        requests: list[httpx.Request] = []
        _install_transport(monkeypatch, _ok_handler(requests))

        with pytest.raises(_client().CollectionNotFound):
            await _client().fetch_collection_file_ids(
                collection_id=malformed_id, drive=DRIVE
            )

        assert requests == []

    @pytest.mark.parametrize("status", [400, 404])
    async def test_400_and_404_raise_collection_not_found(
        self, monkeypatch, status
    ):
        """SPEC-ADDON-058 I11: 400 and 404 are a not-found answer."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json={"detail": "nope"})

        _install_transport(monkeypatch, handler)

        with pytest.raises(_client().CollectionNotFound):
            await _client().fetch_collection_file_ids(
                collection_id=COLLECTION_ID, drive=DRIVE
            )

    @pytest.mark.parametrize(
        "status", [201, 401, 403, 409, 418, 422, 429, 500, 502, 503]
    )
    async def test_other_statuses_raise_collection_lookup_failed(
        self, monkeypatch, status
    ):
        """SPEC-ADDON-058 I11: only 200 is a success status."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status, json={_CORE_FILE_IDS_KEY: [FILE_ID_A]}
            )

        _install_transport(monkeypatch, handler)

        with pytest.raises(_client().CollectionLookupFailed):
            await _client().fetch_collection_file_ids(
                collection_id=COLLECTION_ID, drive=DRIVE
            )

    @pytest.mark.parametrize(
        "error",
        [
            httpx.ConnectError("connection refused"),
            httpx.ReadTimeout("read timed out"),
        ],
        ids=["network-error", "timeout"],
    )
    async def test_transport_errors_raise_collection_lookup_failed(
        self, monkeypatch, error
    ):
        """SPEC-ADDON-058 I11: a network error or timeout is a failure."""
        def handler(request: httpx.Request) -> httpx.Response:
            raise error

        _install_transport(monkeypatch, handler)

        with pytest.raises(_client().CollectionLookupFailed):
            await _client().fetch_collection_file_ids(
                collection_id=COLLECTION_ID, drive=DRIVE
            )

    @pytest.mark.parametrize(
        ("case", "response_kwargs"),
        [
            (
                "non-json",
                {
                    "content": b"<html>oops</html>",
                    "headers": {"content-type": "text/html"},
                },
            ),
            ("top-level-list", {"json": [FILE_ID_A]}),
            ("top-level-string", {"json": "nope"}),
            (
                "top-level-null",
                {
                    "content": b"null",
                    "headers": {"content-type": "application/json"},
                },
            ),
            ("missing-key", {"json": {"collections": []}}),
            ("null-value", {"json": {_CORE_FILE_IDS_KEY: None}}),
            ("string-value", {"json": {_CORE_FILE_IDS_KEY: FILE_ID_A}}),
            (
                "object-value",
                {"json": {_CORE_FILE_IDS_KEY: {"id": FILE_ID_A}}},
            ),
            (
                "integer-entry",
                {"json": {_CORE_FILE_IDS_KEY: [FILE_ID_A, 42]}},
            ),
            ("null-entry", {"json": {_CORE_FILE_IDS_KEY: [None]}}),
            (
                "list-entry",
                {"json": {_CORE_FILE_IDS_KEY: [[FILE_ID_A]]}},
            ),
            (
                "object-entry",
                {"json": {_CORE_FILE_IDS_KEY: [{"id": FILE_ID_A}]}},
            ),
        ],
        ids=[
            "non-json",
            "top-level-list",
            "top-level-string",
            "top-level-null",
            "missing-key",
            "null-value",
            "string-value",
            "object-value",
            "integer-entry",
            "null-entry",
            "list-entry",
            "object-entry",
        ],
    )
    async def test_malformed_bodies_raise_collection_lookup_failed(
        self, monkeypatch, case, response_kwargs
    ):
        """SPEC-ADDON-058 I11: a malformed 200 body is a failure."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, **response_kwargs)

        _install_transport(monkeypatch, handler)

        with pytest.raises(_client().CollectionLookupFailed):
            await _client().fetch_collection_file_ids(
                collection_id=COLLECTION_ID, drive=DRIVE
            )

    async def test_lookup_exceptions_share_a_base(self):
        """SPEC-ADDON-058 I11: one base type catches both failures."""
        module = _client()

        assert issubclass(
            module.CollectionNotFound, module.CollectionLookupError
        )
        assert issubclass(
            module.CollectionLookupFailed, module.CollectionLookupError
        )
        assert not issubclass(
            module.CollectionNotFound, module.CollectionLookupFailed
        )

    async def test_warning_log_carries_status_and_drive_but_no_file_id(
        self, monkeypatch, caplog
    ):
        """SPEC-ADDON-058 I11: a failing status logs drive + status only."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500,
                json={
                    "detail": "boom",
                    _CORE_FILE_IDS_KEY: [FILE_ID_A, FILE_ID_B],
                },
            )

        _install_transport(monkeypatch, handler)
        caplog.set_level(logging.WARNING)

        with pytest.raises(_client().CollectionLookupFailed):
            await _client().fetch_collection_file_ids(
                collection_id=COLLECTION_ID, drive=DRIVE
            )

        messages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ]
        joined = "\n".join(messages)
        assert messages
        assert "collection-client" in joined
        assert DRIVE in joined
        assert "500" in joined
        assert FILE_ID_A not in joined
        assert FILE_ID_B not in joined

    async def test_warning_log_carries_error_type_but_no_file_id(
        self, monkeypatch, caplog
    ):
        """SPEC-ADDON-058 I11: a transport error logs its type, not ids."""
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(
                f"connection refused while reading {FILE_ID_A}"
            )

        _install_transport(monkeypatch, handler)
        caplog.set_level(logging.WARNING)

        with pytest.raises(_client().CollectionLookupFailed):
            await _client().fetch_collection_file_ids(
                collection_id=COLLECTION_ID, drive=DRIVE
            )

        joined = "\n".join(
            record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.WARNING
        )
        assert "collection-client" in joined
        assert DRIVE in joined
        assert "ConnectError" in joined
        assert FILE_ID_A not in joined

    async def test_warning_log_omits_ids_of_a_rejected_entry(
        self, monkeypatch, caplog
    ):
        """SPEC-ADDON-058 I11: a non-string entry fails without logging ids."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={_CORE_FILE_IDS_KEY: [FILE_ID_A, FILE_ID_B, 42]}
            )

        _install_transport(monkeypatch, handler)
        caplog.set_level(logging.WARNING)

        with pytest.raises(_client().CollectionLookupFailed):
            await _client().fetch_collection_file_ids(
                collection_id=COLLECTION_ID, drive=DRIVE
            )

        joined = "\n".join(
            record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.WARNING
        )
        assert "collection-client" in joined
        assert FILE_ID_A not in joined
        assert FILE_ID_B not in joined

    async def test_not_found_logs_no_warning(self, monkeypatch, caplog):
        """SPEC-ADDON-058 I11: a not-found answer stays quiet."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "not found"})

        _install_transport(monkeypatch, handler)
        caplog.set_level(logging.WARNING)

        with pytest.raises(_client().CollectionNotFound):
            await _client().fetch_collection_file_ids(
                collection_id=COLLECTION_ID, drive=DRIVE
            )

        assert [
            record
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ] == []


class TestSpecAddon058I11Parity:
    @pytest.mark.parametrize(
        ("candidate", "core_accepts"),
        [
            ("AbCdEfGh1234", True),
            ("0123456789ab", True),
            ("abc-def_1234", True),
            ("abcdefghijkl", True),
            ("abcdefghij_l", True),
            ("abc", False),
            ("abcdefghijk", False),
            ("abcdefghijklm", False),
            ("abcdefghij l", False),
            ("abcdefghij.l", False),
            ("abcdefghij/l", False),
            ("ああああああああああああ", False),
            ("", False),
        ],
        ids=[
            "mixed-case",
            "digits",
            "hyphen-underscore",
            "lowercase",
            "underscore-inside",
            "too-short",
            "eleven-chars",
            "thirteen-chars",
            "space",
            "dot",
            "slash",
            "non-ascii",
            "empty",
        ],
    )
    async def test_id_pattern_parity_with_the_core_reference(
        self, monkeypatch, candidate, core_accepts
    ):
        """SPEC-ADDON-058 I11: the client's id gate matches the core's."""
        assert _core_accepts(candidate) is core_accepts
        module = _client()
        requests: list[httpx.Request] = []
        _install_transport(monkeypatch, _ok_handler(requests))

        if core_accepts:
            result = await module.fetch_collection_file_ids(
                collection_id=candidate, drive=DRIVE
            )
            assert result == []
            assert len(requests) == 1
            assert requests[0].url.path == (
                f"/api/internal/collections/{candidate}/file-ids"
            )
        else:
            with pytest.raises(module.CollectionNotFound):
                await module.fetch_collection_file_ids(
                    collection_id=candidate, drive=DRIVE
                )
            assert requests == []

    async def test_parses_the_core_response_key(self, monkeypatch):
        """SPEC-ADDON-058 I11: the reference response key is accepted."""
        requests: list[httpx.Request] = []
        _install_transport(
            monkeypatch, _ok_handler(requests, [FILE_ID_A, FILE_ID_B])
        )

        result = await _client().fetch_collection_file_ids(
            collection_id=COLLECTION_ID, drive=DRIVE
        )

        assert result == [FILE_ID_A, FILE_ID_B]

    @pytest.mark.parametrize(
        "body",
        [
            {"fileIds": [FILE_ID_A]},
            {"file_id": FILE_ID_A},
            {"ids": [FILE_ID_A]},
            {"data": {_CORE_FILE_IDS_KEY: [FILE_ID_A]}},
            {"files": [{"id": FILE_ID_A}]},
        ],
        ids=[
            "camel-case-key",
            "singular-key",
            "generic-ids-key",
            "nested-key",
            "files-key",
        ],
    )
    async def test_rejects_lookalike_response_keys(self, monkeypatch, body):
        """SPEC-ADDON-058 I11: only the core's response key parses."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body)

        _install_transport(monkeypatch, handler)

        with pytest.raises(_client().CollectionLookupFailed):
            await _client().fetch_collection_file_ids(
                collection_id=COLLECTION_ID, drive=DRIVE
            )
