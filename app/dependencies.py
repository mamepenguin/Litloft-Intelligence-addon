"""Shared state and dependency injection for FastAPI routers."""

import os

from fastapi import Header, HTTPException

from app.indexer import IndexManager
from app.llm import LLMClient
from app.workers.auto_tags import AutoTagsWorker
from app.workers.chapter_suggestions import ChapterSuggestionsWorker
from app.workers.pickup import PickupWorker
from app.workers.retrieval_keywords import RetrievalKeywordsWorker
from app.workers.summaries import SummariesWorker
from app.workers.vision import VisionDescribeWorker

# Module-level state (initialized during lifespan in main.py)
_index_manager: IndexManager | None = None
_auto_tags_worker: AutoTagsWorker | None = None
_chapter_suggestions_worker: ChapterSuggestionsWorker | None = None
_summaries_worker: SummariesWorker | None = None
_vision_worker: VisionDescribeWorker | None = None
_retrieval_keywords_worker: RetrievalKeywordsWorker | None = None
_pickup_worker: PickupWorker | None = None
_llm_client: LLMClient | None = None

_WEBHOOK_SECRET = os.environ.get("SEARCH_WEBHOOK_SECRET", "")


def get_index_manager() -> IndexManager:
    """Get the index manager instance.

    Raises:
        RuntimeError: If the manager is not initialized.
    """
    if _index_manager is None:
        raise RuntimeError("Index manager not initialized")
    return _index_manager


def get_auto_tags_worker() -> AutoTagsWorker:
    """Get the auto-tags worker instance.

    Raises:
        RuntimeError: If the worker is not initialized.
    """
    if _auto_tags_worker is None:
        raise RuntimeError("Auto-tags worker not initialized")
    return _auto_tags_worker


def get_chapter_suggestions_worker() -> ChapterSuggestionsWorker:
    if _chapter_suggestions_worker is None:
        raise RuntimeError("Chapter suggestions worker not initialized")
    return _chapter_suggestions_worker


def get_summaries_worker() -> SummariesWorker:
    """Get the summaries worker instance.

    Raises:
        RuntimeError: If the worker is not initialized.
    """
    if _summaries_worker is None:
        raise RuntimeError("Summaries worker not initialized")
    return _summaries_worker


def get_vision_worker() -> VisionDescribeWorker:
    """Get the vision_describe worker instance.

    Raises:
        RuntimeError: If the worker is not initialized.
    """
    if _vision_worker is None:
        raise RuntimeError("Vision describe worker not initialized")
    return _vision_worker


def get_retrieval_keywords_worker() -> RetrievalKeywordsWorker:
    """Get the retrieval_keywords worker instance.

    Raises:
        RuntimeError: If the worker is not initialized.
    """
    if _retrieval_keywords_worker is None:
        raise RuntimeError("Retrieval-keywords worker not initialized")
    return _retrieval_keywords_worker


def get_llm_client() -> LLMClient:
    """Get the LLM client instance.

    Raises:
        RuntimeError: If the client is not initialized.
    """
    if _llm_client is None:
        raise RuntimeError("LLM client not initialized")
    return _llm_client


async def verify_webhook_secret(
    x_webhook_secret: str = Header(default=""),
) -> None:
    """Verify webhook secret if configured."""
    if _WEBHOOK_SECRET and x_webhook_secret != _WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")
