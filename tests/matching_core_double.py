"""A transport-level double for the core's Internal API during filtered search.

The addon reaches the core through a client it builds per call. Patching the
transport classes rather than a client class catches whichever client the
implementation chooses (sync or async).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx

MATCHING_PATH = "/api/internal/files/matching"
BULK_PATH = "/api/internal/files/bulk"


@dataclass
class MatchingCore:
    """Answers ``POST /files/matching`` and ``POST /files/bulk``."""

    file_ids: list[str] = field(default_factory=list)
    status: int = 200
    payload: object | None = None
    raw: bytes | None = None
    raises: Exception | None = None
    matching_requests: list[httpx.Request] = field(default_factory=list)
    bulk_requests: list[httpx.Request] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith(MATCHING_PATH):
            self.matching_requests.append(request)
            if self.raises is not None:
                raise self.raises
            if self.raw is not None:
                return httpx.Response(200, content=self.raw)
            body = (
                self.payload
                if self.payload is not None
                else {"file_ids": list(self.file_ids)}
            )
            return httpx.Response(self.status, json=body)
        if path.endswith(BULK_PATH):
            self.bulk_requests.append(request)
            asked = json.loads(request.content or b"{}").get("file_ids", [])
            return httpx.Response(
                200,
                json={"files": [{"id": fid} for fid in asked], "not_found": []},
            )
        raise AssertionError(f"unexpected internal call: {request.method} {request.url}")


def install(monkeypatch, core: MatchingCore) -> MatchingCore:
    transport = httpx.MockTransport(core.handle)

    def _sync(_self, request: httpx.Request) -> httpx.Response:
        return transport.handle_request(request)

    async def _async(_self, request: httpx.Request) -> httpx.Response:
        return await transport.handle_async_request(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _sync)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _async)
    return core
