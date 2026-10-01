"""Which LLM profiles a viewer may choose for summaries, detailed summaries and Ask."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app import llm_routing
from app.drive_context import require_drive
from app.llm_routing import Defer
from app.schemas import LLMChoice, LLMChoicesResponse

router = APIRouter(tags=["llm"])


async def _choices(drive: str, feature: str) -> LLMChoicesResponse:
    result = await llm_routing.choices(drive, feature)
    if isinstance(result, Defer):
        raise HTTPException(
            status_code=503, detail="LLM policy is unavailable, try again shortly"
        )
    return LLMChoicesResponse(
        auto=result.auto.name if result.auto is not None else None,
        choices=[
            LLMChoice(name=p.name, model=p.config.model, offhost=p.offhost)
            for p in result.profiles
        ],
    )


# One literal path per feature: the proxy gates a route by one fixed policy feature.
@router.get("/llm/choices/summaries", response_model=LLMChoicesResponse)
async def summaries_choices(drive: str = Depends(require_drive)) -> LLMChoicesResponse:
    return await _choices(drive, "summaries")


@router.get("/llm/choices/detailed_summaries", response_model=LLMChoicesResponse)
async def detailed_summaries_choices(
    drive: str = Depends(require_drive),
) -> LLMChoicesResponse:
    return await _choices(drive, "detailed_summaries")


@router.get("/llm/choices/rag", response_model=LLMChoicesResponse)
async def rag_choices(drive: str = Depends(require_drive)) -> LLMChoicesResponse:
    return await _choices(drive, "rag")
