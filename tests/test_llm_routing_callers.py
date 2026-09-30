"""Every LLM caller resolves its client per job, with the file's drive."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app import llm_routing
from app.config import FeaturesConfig, LLMConfig
from app.llm_routing import Defer, Skip

DRIVE = "private"


@pytest.fixture
def retries(monkeypatch: pytest.MonkeyPatch) -> list:
    scheduled: list = []
    monkeypatch.setattr(llm_routing, "retry_later", scheduled.append)
    return scheduled


def _indexed(fid: str, file_type: str = "video") -> dict:
    return {
        "file_id": fid,
        "drive": DRIVE,
        "filename": "talk.mp4",
        "file_type": file_type,
        "mime_type": "video/mp4",
        "title": "",
        "description": "",
        "tags_text": None,
    }


# ---------------------------------------------------------------------------
# auto_tags
# ---------------------------------------------------------------------------


def _auto_tags(monkeypatch, make_settings) -> dict:
    from app.workers import auto_tags as at
    from app.workers.auto_tags import TagCandidates

    monkeypatch.setattr(
        at, "settings", make_settings(features=FeaturesConfig(auto_tags="on_index"))
    )
    monkeypatch.setattr(at, "_has_suggested_tags", lambda fid: False)
    monkeypatch.setattr(at, "_get_indexed_file", _indexed)
    monkeypatch.setattr(at, "_get_existing_tags", lambda fid: [])
    monkeypatch.setattr(
        at, "_generate_candidates", lambda *a, **k: TagCandidates(clip=["料理"], tfidf=[])
    )
    monkeypatch.setattr(at, "_build_context", lambda *a, **k: "")
    saved: dict = {}
    monkeypatch.setattr(at, "_save_suggested_tags", lambda **kw: saved.update(kw))
    return saved


@pytest.mark.asyncio
async def test_auto_tags_defer_writes_nothing_and_retries(
    monkeypatch, make_settings, use_llm, retries
) -> None:
    from app.workers.auto_tags import AutoTagsWorker

    saved = _auto_tags(monkeypatch, make_settings)
    asked = use_llm(result=Defer("policy unavailable"))

    await AutoTagsWorker()._process_file("f1")

    assert asked == [(DRIVE, "auto_tags")]
    assert saved == {}
    assert len(retries) == 1


@pytest.mark.asyncio
async def test_auto_tags_skip_keeps_local_candidates_without_an_llm(
    monkeypatch, make_settings, use_llm, retries
) -> None:
    from app.workers.auto_tags import AutoTagsWorker

    saved = _auto_tags(monkeypatch, make_settings)
    use_llm(result=Skip("llm_cloud off"))

    await AutoTagsWorker()._process_file("f1")

    assert saved["model"] == "clip+tfidf"
    assert retries == []


@pytest.mark.asyncio
async def test_auto_tags_labels_the_resolved_model(
    monkeypatch, make_settings, use_llm
) -> None:
    from app.workers.auto_tags import AutoTagsWorker

    saved = _auto_tags(monkeypatch, make_settings)
    client = MagicMock()
    client.generate_json = AsyncMock(return_value={"tags": ["料理"]})
    use_llm(client, model="routed-model")

    await AutoTagsWorker()._process_file("f1")

    assert saved["model"] == "clip+tfidf+routed-model"


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------


def _summaries(monkeypatch, make_settings, **features) -> AsyncMock:
    from app.workers import summaries as sm

    settings = make_settings(
        features=FeaturesConfig(**features),
        llm=LLMConfig(provider="openai_compatible", model="global-model"),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr(sm, "settings", settings)
    monkeypatch.setattr(sm, "_has_summary", lambda fid: False)
    monkeypatch.setattr(sm, "_has_detailed_summary", lambda fid: False)
    monkeypatch.setattr(sm, "_get_indexed_file", _indexed)
    monkeypatch.setattr(sm, "_build_context", lambda f, t: "word " * 400)
    save = AsyncMock()
    monkeypatch.setattr(sm, "_save_summary", save)
    return save


@pytest.mark.asyncio
async def test_summaries_defer_writes_nothing_and_retries(
    monkeypatch, make_settings, use_llm, retries
) -> None:
    from app.workers.summaries import SummariesWorker

    save = _summaries(monkeypatch, make_settings, summaries="manual")
    asked = use_llm(result=Defer("policy unavailable"))

    await SummariesWorker()._process_file("f1")

    assert asked == [(DRIVE, "summaries")]
    save.assert_not_called()
    assert len(retries) == 1


@pytest.mark.asyncio
async def test_summaries_skip_writes_nothing(
    monkeypatch, make_settings, use_llm, retries
) -> None:
    from app.workers.summaries import SummariesWorker

    save = _summaries(monkeypatch, make_settings, summaries="manual")
    use_llm(result=Skip("llm_cloud off"))

    await SummariesWorker()._process_file("f1")

    save.assert_not_called()
    assert retries == []


@pytest.mark.asyncio
async def test_summaries_record_the_resolved_model(
    monkeypatch, make_settings, use_llm
) -> None:
    from app.workers.summaries import SummariesWorker

    save = _summaries(monkeypatch, make_settings, summaries="manual")
    client = MagicMock()
    client.generate_json = AsyncMock(return_value={"short": "s", "long": "l"})
    use_llm(client, model="routed-model")

    await SummariesWorker()._process_file("f1")

    assert save.call_args.kwargs["model"] == "routed-model"


@pytest.mark.asyncio
async def test_detailed_on_index_resolves_its_own_feature(
    monkeypatch, make_settings, use_llm, retries
) -> None:
    from app.workers import summaries as sm

    _summaries(
        monkeypatch, make_settings, summaries="false", detailed_summaries="on_index"
    )
    monkeypatch.setattr(
        "app.policy_client.is_file_feature_enabled", AsyncMock(return_value=True)
    )
    generate = AsyncMock()
    monkeypatch.setattr(sm, "generate_detailed_summary", generate)
    asked = use_llm(result=Skip("llm_cloud off"))

    await sm.SummariesWorker()._process_file("f1")

    assert asked == [(DRIVE, "detailed_summaries")]
    generate.assert_not_called()


# ---------------------------------------------------------------------------
# retrieval_keywords
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [Defer("unavailable"), Skip("off")])
async def test_retrieval_keywords_unresolved_writes_nothing(
    monkeypatch, use_llm, retries, result
) -> None:
    from app.workers import retrieval_keywords as rk

    settings = MagicMock()
    settings.features.retrieval_keywords = "on_index"
    monkeypatch.setattr(rk, "settings", settings)
    monkeypatch.setattr(rk, "_has_retrieval_keywords", lambda fid: False)
    monkeypatch.setattr(rk, "_get_indexed_file", _indexed)
    monkeypatch.setattr(rk, "_build_context", lambda f, t: "word " * 50)
    upsert = MagicMock()
    monkeypatch.setattr(rk, "upsert_retrieval_keywords", upsert)
    asked = use_llm(result=result)

    await rk.RetrievalKeywordsWorker()._process_file("f1")

    assert asked == [(DRIVE, "retrieval_keywords")]
    upsert.assert_not_called()
    assert len(retries) == (1 if isinstance(result, Defer) else 0)


# ---------------------------------------------------------------------------
# retry_later and the router gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_later_requeues_after_the_delay() -> None:
    requeued = asyncio.Event()

    async def _requeue() -> None:
        requeued.set()

    llm_routing.retry_later(_requeue, delay=0)

    await asyncio.wait_for(requeued.wait(), timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "status"),
    [(Defer("unavailable"), 503), (Skip("off"), 400)],
)
async def test_require_llm_rejects_an_unresolved_drive(use_llm, result, status) -> None:
    from app.routers.llm_gate import require_llm

    use_llm(result=result)

    with pytest.raises(HTTPException) as exc:
        await require_llm(DRIVE, "summaries")

    assert exc.value.status_code == status


@pytest.mark.asyncio
async def test_require_llm_returns_the_resolved_profile(use_llm) -> None:
    from app.routers.llm_gate import require_llm

    client = MagicMock()
    asked = use_llm(client)

    resolved = await require_llm(DRIVE, "transcript_refine")

    assert resolved.client is client
    assert asked == [(DRIVE, "transcript_refine")]


@pytest.mark.asyncio
async def test_regenerate_detailed_rejects_before_touching_the_stored_summary(
    monkeypatch, use_llm
) -> None:
    from app.routers import summaries as router_mod

    monkeypatch.setattr(
        router_mod, "settings", MagicMock(features=MagicMock(detailed_summaries="manual"))
    )
    touched = MagicMock()
    monkeypatch.setattr(router_mod, "_require_file_in_drive", touched)
    monkeypatch.setattr(router_mod, "get_search_db", touched)
    use_llm(result=Skip("llm_cloud off"))

    with pytest.raises(HTTPException):
        await router_mod.regenerate_detailed_summary(
            "f1", MagicMock(), None, DRIVE
        )

    touched.assert_not_called()

