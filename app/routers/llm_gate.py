"""Resolve the LLM for a request before any state changes."""

from __future__ import annotations

from fastapi import HTTPException

from app import llm_routing
from app.llm_routing import Defer, Resolved


async def require_llm(drive: str, feature: str) -> Resolved:
    result = await llm_routing.resolve(drive, feature)
    if isinstance(result, Defer):
        raise HTTPException(
            status_code=503, detail="LLM policy is unavailable, try again shortly"
        )
    if not isinstance(result, Resolved):
        raise HTTPException(status_code=400, detail="LLM is not enabled")
    return result
