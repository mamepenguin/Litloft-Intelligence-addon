"""Caller credential forwarding helpers.

Addons accept browser cookies and non-browser Bearer tokens. When they
call back into core public APIs, they forward the same credential shape
the caller presented instead of normalizing everything to cookies.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request


@dataclass(frozen=True)
class CallerCredential:
    cookie: str | None = None
    bearer: str | None = None

    @classmethod
    def from_request(cls, request: Request) -> "CallerCredential":
        auth_header = request.headers.get("Authorization")
        bearer: str | None = None
        if auth_header:
            scheme, _, param = auth_header.partition(" ")
            if scheme.lower() == "bearer" and param:
                bearer = param
        return cls(cookie=request.headers.get("Cookie"), bearer=bearer)

    def headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.bearer:
            headers["Authorization"] = f"Bearer {self.bearer}"
        if self.cookie:
            headers["Cookie"] = self.cookie
        return headers
