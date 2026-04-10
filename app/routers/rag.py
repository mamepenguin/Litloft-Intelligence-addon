"""RAG (question answering) endpoint.

Single public route: ``POST /ask``. The heavy lifting lives in
``app.rag.service.answer_question``; this module exists purely to
enforce feature gating + query validation and to translate the
service's internal dataclass into the Pydantic response model.

Gating layers (all must pass):

1. ``features.rag`` must be True (config toggle).
2. ``LLMClient.enabled`` must be True (provider configured).
3. ``body.query.strip()`` must be at least 3 characters.

The Pydantic ``AskRequest`` model already enforces ``1 <= len(query)
<= 1000`` and ``1 <= top_k <= 20``; the strict >=3 post-strip check
lives here so it runs after whitespace normalization.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Cookie, HTTPException

from app.config import settings
from app.dependencies import get_llm_client
from app.rag.service import AnswerResponse, answer_question
from app.schemas import (
    AnswerResponseModel,
    AskRequest,
    CitationModel,
    SourceModel,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])


def _require_rag_enabled() -> None:
    """Raise 4xx/5xx if the RAG feature or LLM is not available.

    * 400 when the feature is explicitly disabled in config.
    * 400 when the LLM provider is configured as "disabled".
    * 503 when the dependency container isn't initialized yet
      (startup race: the router is mounted before the lifespan
      handler finishes). This is a transient condition, not a
      permanent misconfiguration, so the distinct status code
      makes it easier to diagnose in logs vs. an everything-500.
    """
    if not settings.features.rag:
        raise HTTPException(
            status_code=400, detail="RAG feature is disabled"
        )
    try:
        client = get_llm_client()
    except RuntimeError:
        # Startup race: dependency injection not yet populated.
        raise HTTPException(
            status_code=503, detail="LLM client not initialized yet"
        )
    if not client.enabled:
        raise HTTPException(
            status_code=400, detail="LLM is not enabled"
        )


def _to_response_model(result: AnswerResponse) -> AnswerResponseModel:
    """Convert the service dataclass into the Pydantic response model."""
    return AnswerResponseModel(
        query=result.query,
        answer=result.answer,
        citations=[CitationModel(**c) for c in result.citations],
        sources=[SourceModel(**s) for s in result.sources],
        retrieved_count=result.retrieved_count,
        took_ms=result.took_ms,
    )


@router.post("/ask", response_model=AnswerResponseModel)
async def ask_endpoint(
    body: AskRequest,
    access_token: Annotated[str | None, Cookie()] = None,
) -> AnswerResponseModel:
    """Answer a natural-language question using retrieval-augmented generation.

    Security notes:

    * The caller's ``access_token`` cookie is forwarded to the retriever
      so drive access control runs BEFORE file content reaches the LLM.
      The host's Generic Addon Proxy passes browser cookies through to
      the intelligence service verbatim (see
      ``backend/app/routers/addon_proxy.py``), so we read the cookie
      directly from the request. The parameter name matches the cookie
      key used by ``get_unlocked_groups`` in ``backend/app/auth.py``.
    * The parser drops citations referencing file_ids that were not
      in the retrieved set (anti-hallucination).
    * Query length is clamped to 1000 characters and >= 3 non-whitespace
      characters to deter DoS-by-giant-prompt.
    """
    _require_rag_enabled()

    # Post-strip length check. Pydantic's min_length=1 only rejects the
    # empty string, but a 2-char query gives the LLM nothing to work
    # with and would waste an API call. "   " strips to "" which would
    # pass min_length=1 if it weren't for this check.
    if len(body.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short")

    try:
        result = await answer_question(
            query=body.query,
            hv_token=access_token,
            top_k=body.top_k,
            file_type=body.file_type,
            drive=body.drive,
        )
    except HTTPException:
        raise
    except Exception as e:
        # Log only the exception type to avoid leaking prompt / file
        # content in logs. The detailed traceback is still captured by
        # uvicorn's default error handling at DEBUG level.
        logger.error("RAG answer failed: %s", type(e).__name__)
        raise HTTPException(
            status_code=500, detail="Answer generation failed"
        ) from e

    return _to_response_model(result)
