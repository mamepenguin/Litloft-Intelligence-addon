"""Resolve the LLM for a request before any state changes."""

from __future__ import annotations

from fastapi import HTTPException

from app import llm_routing
from app.llm_routing import Defer, Resolved


async def require_llm(
    drive: str, feature: str, requested: str | None = None
) -> Resolved:
    result = await llm_routing.resolve(drive, feature, requested)
    if isinstance(result, Defer):
        raise HTTPException(
            status_code=503, detail="LLM policy is unavailable, try again shortly"
        )
    if not isinstance(result, Resolved):
        if requested is not None:
            raise HTTPException(status_code=400, detail="profile_unavailable")
        raise HTTPException(status_code=400, detail="LLM is not enabled")
    return result
