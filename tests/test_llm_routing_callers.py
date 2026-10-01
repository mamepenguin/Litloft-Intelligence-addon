"""Every LLM caller resolves its client per job, with the file's drive."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app import llm_routing
from app.config import FeaturesConfig, LLMConfig
from app.llm_routing import Defer, Skip
from tests.llm_helpers import resolved_with

DRIVE = "private"


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
async def test_auto_tags_defer_writes_nothing(
    monkeypatch, make_settings, use_llm
) -> None:
    from app.workers.auto_tags import AutoTagsWorker

    saved = _auto_tags(monkeypatch, make_settings)
    asked = use_llm(result=Defer("policy unavailable"))

    worker = AutoTagsWorker()
    await worker._process_file("f1")

    assert asked == [(DRIVE, "auto_tags")]
    assert saved == {}
    assert worker.get_status()["waiting"] == 0


@pytest.mark.asyncio
async def test_auto_tags_skip_keeps_local_candidates_without_an_llm(
    monkeypatch, make_settings, use_llm
) -> None:
    from app.workers.auto_tags import AutoTagsWorker

    saved = _auto_tags(monkeypatch, make_settings)
    use_llm(result=Skip("llm_cloud off"))

    await AutoTagsWorker()._process_file("f1")

    assert saved["model"] == "clip+tfidf"


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
async def test_summaries_defer_writes_nothing(
    monkeypatch, make_settings, use_llm
) -> None:
    from app.workers.summaries import SummariesWorker

    save = _summaries(monkeypatch, make_settings, summaries="manual")
    asked = use_llm(result=Defer("policy unavailable"))

    worker = SummariesWorker()
    await worker._process_file("f1")

    assert asked == [(DRIVE, "summaries")]
    save.assert_not_called()
    assert worker.get_status()["waiting"] == 0


@pytest.mark.asyncio
async def test_summaries_skip_writes_nothing(
    monkeypatch, make_settings, use_llm
) -> None:
    from app.workers.summaries import SummariesWorker

    save = _summaries(monkeypatch, make_settings, summaries="manual")
    use_llm(result=Skip("llm_cloud off"))

    await SummariesWorker()._process_file("f1")

    save.assert_not_called()


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
async def test_a_chosen_summary_is_generated_by_the_profile_it_was_handed(
    monkeypatch, make_settings, use_llm
) -> None:
    from app.workers import summaries as sm

    save = _summaries(monkeypatch, make_settings, summaries="manual")
    asked = use_llm(MagicMock(), model="routed-model")
    client = MagicMock()
    client.generate_json = AsyncMock(return_value={"short": "s", "long": "l"})

    await sm.generate_summary("f1", resolved_with(client, model="big-model"))

    assert save.call_args.kwargs["model"] == "big-model"
    assert asked == []


@pytest.mark.asyncio
async def test_detailed_on_index_resolves_its_own_feature(
    monkeypatch, make_settings, use_llm
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
    monkeypatch, use_llm, result
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


# ---------------------------------------------------------------------------
# the router gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "result", "status", "detail"),
    [
        (None, Defer("unavailable"), 503, "LLM policy is unavailable, try again shortly"),
        (None, Skip("off"), 400, "LLM is not enabled"),
        ("big", Defer("unavailable"), 503, "LLM policy is unavailable, try again shortly"),
        ("big", Skip("off"), 400, "profile_unavailable"),
    ],
)
async def test_require_llm_rejects_an_unresolved_drive(
    use_llm, requested, result, status, detail
) -> None:
    from app.routers.llm_gate import require_llm

    asked = use_llm(result=result)

    with pytest.raises(HTTPException) as exc:
        await require_llm(DRIVE, "summaries", requested)

    assert (exc.value.status_code, exc.value.detail) == (status, detail)
    assert asked.requested == [requested]


@pytest.mark.asyncio
async def test_require_llm_returns_the_resolved_profile(use_llm) -> None:
    from app.routers.llm_gate import require_llm

    client = MagicMock()
    asked = use_llm(client)

    resolved = await require_llm(DRIVE, "transcript_refine")

    assert resolved.client is client
    assert asked == [(DRIVE, "transcript_refine")]


@pytest.mark.asyncio
async def test_detailed_summary_records_the_resolved_model(
    monkeypatch, make_settings
) -> None:
    from app.workers import summaries as sm

    _summaries(monkeypatch, make_settings, detailed_summaries="manual")
    statuses: list = []
    monkeypatch.setattr(sm, "_set_detailed_status", lambda *a, **k: statuses.append(k))
    saved: dict = {}
    monkeypatch.setattr(sm, "_save_detailed_summary", lambda **k: saved.update(k))
    monkeypatch.setattr(sm, "_recalculate_citations", AsyncMock())
    client = MagicMock()
    client.generate = AsyncMock(return_value="# Body")

    await sm.generate_detailed_summary("f1", resolved_with(client, model="routed-model"))

    assert statuses[0]["model"] == "routed-model"
    assert saved["model"] == "routed-model"


@pytest.mark.asyncio
async def test_retrieval_keywords_record_the_resolved_model(monkeypatch, use_llm) -> None:
    from app.workers import retrieval_keywords as rk

    settings = MagicMock()
    settings.features.retrieval_keywords = "on_index"
    settings.llm.model = "global-model"
    monkeypatch.setattr(rk, "settings", settings)
    monkeypatch.setattr(rk, "_has_retrieval_keywords", lambda fid: False)
    monkeypatch.setattr(rk, "_get_indexed_file", _indexed)
    monkeypatch.setattr(rk, "_build_context", lambda f, t: "word " * 50)
    monkeypatch.setattr(rk, "_post_filter", lambda raw: "alpha beta")
    upsert = MagicMock()
    monkeypatch.setattr(rk, "upsert_retrieval_keywords", upsert)
    monkeypatch.setattr(rk, "get_search_db", contextmanager(lambda: iter([MagicMock()])))
    client = MagicMock()
    client.generate_json = AsyncMock(return_value={"keywords": ["alpha"]})
    use_llm(client, model="routed-model")

    await rk.RetrievalKeywordsWorker()._process_file("f1")

    assert upsert.call_args.kwargs["model"] == "routed-model"


def _summary_router(monkeypatch) -> MagicMock:
    from app.routers import summaries as router_mod

    monkeypatch.setattr(
        router_mod,
        "settings",
        MagicMock(features=MagicMock(summaries="manual", detailed_summaries="manual")),
    )
    touched = MagicMock()
    for name in ("_require_file_in_drive", "get_search_db", "get_summaries_worker"):
        monkeypatch.setattr(router_mod, name, touched)
    return touched


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "feature"),
    [
        ("regenerate_summary", "summaries"),
        ("batch_summaries", "summaries"),
        ("start_detailed_summary", "detailed_summaries"),
        ("regenerate_detailed_summary", "detailed_summaries"),
    ],
)
async def test_summary_routes_resolve_their_feature_before_any_change(
    monkeypatch, use_llm, route, feature
) -> None:
    from app.routers import summaries as router_mod

    touched = _summary_router(monkeypatch)
    asked = use_llm(result=Skip("llm_cloud off"))
    args = {
        "regenerate_summary": ("f1", DRIVE),
        "batch_summaries": (MagicMock(file_ids=["f1"]), DRIVE),
        "start_detailed_summary": ("f1", MagicMock(), DRIVE),
        "regenerate_detailed_summary": ("f1", MagicMock(), None, DRIVE),
    }[route]

    with pytest.raises(HTTPException) as exc:
        await getattr(router_mod, route)(*args)

    assert exc.value.status_code == 400
    assert asked == [(DRIVE, feature)]
    assert asked.requested == [None]
    touched.assert_not_called()


def _choice_router(monkeypatch) -> MagicMock:
    from app.routers import summaries as router_mod

    touched = _summary_router(monkeypatch)
    for name in ("_reembed_metadata_after_summary_change", "_clear_knowledge_active_summary"):
        monkeypatch.setattr(router_mod, name, touched)
    monkeypatch.setattr("app.workers.summaries._set_detailed_status", touched)
    return touched


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "status"), [(Skip("llm_cloud off"), 400), (Defer("unavailable"), 503)]
)
@pytest.mark.parametrize(
    ("route", "feature"),
    [
        ("regenerate_summary", "summaries"),
        ("start_detailed_summary", "detailed_summaries"),
        ("regenerate_detailed_summary", "detailed_summaries"),
    ],
)
async def test_a_rejected_choice_changes_nothing(
    monkeypatch, use_llm, route, feature, result, status
) -> None:
    from fastapi import BackgroundTasks

    from app.routers import summaries as router_mod
    from app.schemas import DetailedSummaryRegenerateRequest, SummaryProfileRequest

    touched = _choice_router(monkeypatch)
    asked = use_llm(result=result)
    bg = BackgroundTasks()
    kwargs = {
        "regenerate_summary": {"body": SummaryProfileRequest(profile="big"), "background_tasks": bg},
        "start_detailed_summary": {"body": SummaryProfileRequest(profile="big"), "background_tasks": bg},
        "regenerate_detailed_summary": {
            "body": DetailedSummaryRegenerateRequest(force=True, profile="big"),
            "background_tasks": bg,
        },
    }[route]

    with pytest.raises(HTTPException) as exc:
        await getattr(router_mod, route)("f1", drive=DRIVE, **kwargs)

    assert exc.value.status_code == status
    assert asked == [(DRIVE, feature)]
    assert asked.requested == ["big"]
    touched.assert_not_called()
    assert bg.tasks == []


def _regenerate_router(monkeypatch) -> tuple[MagicMock, list]:
    from app.routers import summaries as router_mod

    monkeypatch.setattr(
        router_mod,
        "settings",
        MagicMock(features=MagicMock(summaries="manual", detailed_summaries="manual")),
    )
    monkeypatch.setattr(router_mod, "_require_file_in_drive", lambda fid, drive: None)
    monkeypatch.setattr(router_mod, "classify_missing_reason", lambda fid: "not_generated")
    executed: list = []

    @contextmanager
    def _db():
        session = MagicMock()
        session.execute.side_effect = lambda stmt, params=None: executed.append(
            (str(stmt), params)
        )
        yield session

    monkeypatch.setattr(router_mod, "get_search_db", _db)
    monkeypatch.setattr(router_mod, "_reembed_metadata_after_summary_change", AsyncMock())
    worker = MagicMock()
    worker.enqueue = AsyncMock()
    monkeypatch.setattr(router_mod, "get_summaries_worker", lambda: worker)
    return worker, executed


@pytest.mark.asyncio
async def test_summary_regenerate_without_a_choice_queues_through_routing(
    monkeypatch, use_llm
) -> None:
    from fastapi import BackgroundTasks

    from app.routers import summaries as router_mod

    worker, executed = _regenerate_router(monkeypatch)
    asked = use_llm(MagicMock())
    bg = BackgroundTasks()

    await router_mod.regenerate_summary("f1", drive=DRIVE, body=None, background_tasks=bg)

    assert asked.requested == [None]
    assert [sql for sql, _ in executed] == ["DELETE FROM file_summaries WHERE file_id = :fid"]
    worker.enqueue.assert_awaited_once_with("f1")
    assert bg.tasks == []


@pytest.mark.asyncio
async def test_summary_regenerate_with_a_choice_runs_it_beside_the_queue(
    monkeypatch, use_llm
) -> None:
    from fastapi import BackgroundTasks

    from app.routers import summaries as router_mod
    from app.schemas import SummaryProfileRequest
    from app.workers.summaries import generate_summary

    worker, executed = _regenerate_router(monkeypatch)
    client = MagicMock()
    asked = use_llm(client, model="big-model")
    bg = BackgroundTasks()

    await router_mod.regenerate_summary(
        "f1", drive=DRIVE, body=SummaryProfileRequest(profile="big"), background_tasks=bg
    )

    assert asked.requested == ["big"]
    assert [sql for sql, _ in executed] == ["DELETE FROM file_summaries WHERE file_id = :fid"]
    worker.enqueue.assert_not_called()
    [task] = bg.tasks
    assert task.func is generate_summary
    assert task.args[0] == "f1"
    assert task.args[1].client is client


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["start_detailed_summary", "regenerate_detailed_summary"])
@pytest.mark.parametrize("profile", [None, "big"])
async def test_detailed_routes_generate_with_the_profile_they_resolved(
    monkeypatch, use_llm, route, profile
) -> None:
    from fastapi import BackgroundTasks

    from app.routers import summaries as router_mod
    from app.schemas import DetailedSummaryRegenerateRequest, SummaryProfileRequest

    monkeypatch.setattr(
        router_mod,
        "settings",
        MagicMock(features=MagicMock(detailed_summaries="manual")),
    )
    monkeypatch.setattr(router_mod, "_require_file_in_drive", lambda fid, drive: None)
    monkeypatch.setattr(router_mod, "classify_detailed_missing_reason", lambda fid: "not_generated")
    monkeypatch.setattr(router_mod, "_has_detailed_summary", lambda fid: False)
    monkeypatch.setattr(router_mod, "get_search_db", contextmanager(lambda: iter([MagicMock()])))
    monkeypatch.setattr(router_mod, "_fetch_detailed_edit_state", lambda s, fid: None)
    monkeypatch.setattr(router_mod, "_clear_knowledge_active_summary", AsyncMock())
    monkeypatch.setattr("app.workers.summaries._set_detailed_status", MagicMock())
    client = MagicMock()
    asked = use_llm(client)
    body = (
        SummaryProfileRequest(profile=profile)
        if route == "start_detailed_summary"
        else DetailedSummaryRegenerateRequest(profile=profile)
    )
    bg = BackgroundTasks()

    await getattr(router_mod, route)("f1", bg, drive=DRIVE, body=body)

    assert asked.requested == [profile]
    [task] = bg.tasks
    assert task.func is router_mod.generate_detailed_summary
    assert task.args[1].client is client


@pytest.mark.asyncio
async def test_chapter_generate_resolves_before_reading_the_transcript(
    monkeypatch, use_llm
) -> None:
    from app.routers import chapter_suggestions as router_mod

    monkeypatch.setattr(
        router_mod, "settings", MagicMock(features=MagicMock(chapter_suggestions="manual"))
    )
    monkeypatch.setattr(router_mod, "_require_allowed", AsyncMock())
    touched = MagicMock()
    monkeypatch.setattr(router_mod, "get_search_db", touched)
    asked = use_llm(result=Skip("llm_cloud off"))

    with pytest.raises(HTTPException) as exc:
        await router_mod.generate_chapter_suggestions("f1", DRIVE)

    assert exc.value.status_code == 400
    assert asked == [(DRIVE, "chapter_suggestions")]
    touched.assert_not_called()


# ---------------------------------------------------------------------------
# Ask / Find
# ---------------------------------------------------------------------------


def _rag_on(monkeypatch) -> None:
    from app.routers import rag as rag_router

    monkeypatch.setattr(
        rag_router, "settings", MagicMock(features=MagicMock(rag=True))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "status"),
    [(Defer("unavailable"), 503), (Skip("llm_cloud off"), 400)],
)
@pytest.mark.parametrize("route", ["ask", "find"])
async def test_ask_and_find_refuse_an_unresolved_drive(
    monkeypatch, use_llm, route, result, status
) -> None:
    from app.routers import rag as rag_router
    from app.schemas import AskRequest, FindRequest

    _rag_on(monkeypatch)
    asked = use_llm(result=result)

    with pytest.raises(HTTPException) as exc:
        if route == "ask":
            await rag_router.ask_endpoint(
                AskRequest(query="what happened"), None, None, DRIVE, None
            )
        else:
            await rag_router.find_endpoint(
                FindRequest(question="what happened"), None, None, DRIVE, None
            )

    assert exc.value.status_code == status
    assert asked == [(DRIVE, "rag")]


@pytest.mark.asyncio
async def test_find_hands_the_resolved_profile_to_the_service(monkeypatch, use_llm) -> None:
    from app.routers import rag as rag_router
    from app.schemas import FindRequest

    _rag_on(monkeypatch)
    client = MagicMock()
    use_llm(client)
    seen: list = []

    async def _find(**kwargs):
        seen.append(kwargs["resolved"].client)
        return {}

    monkeypatch.setattr(rag_router, "find_files", _find)

    await rag_router.find_endpoint(
        FindRequest(question="what happened"), None, None, DRIVE, None
    )

    assert seen == [client]


@pytest.mark.asyncio
async def test_the_service_binds_its_profile_for_the_helpers_only_while_it_runs(
    monkeypatch,
) -> None:
    from app.rag import service

    client = MagicMock()
    seen: list = []

    async def _inner(*args, **kwargs):
        seen.append(llm_routing.bound_client())
        return "answer"

    monkeypatch.setattr(service, "_answer_question", _inner)

    result = await service.answer_question(
        "q", None, resolved=resolved_with(client)
    )

    assert (result, seen, llm_routing.bound_client()) == ("answer", [client], None)


@pytest.mark.parametrize(
    ("agentic", "mode", "expected"),
    [(True, "auto", True), (True, "off", False), (False, "auto", False)],
)
def test_the_agentic_gate_reads_the_bound_profile(agentic, mode, expected) -> None:
    from app.config import AgenticModelEntry
    from app.rag import service

    config = LLMConfig(
        provider="openai_compatible",
        base_url="http://llm.test/v1",
        model="m",
        agentic_mode=mode,
        agentic_models=(AgenticModelEntry(name="m"),),
    )
    client = MagicMock()
    client.enabled = True
    client.chat_with_tools = AsyncMock()

    with llm_routing.bound(resolved_with(client, agentic=agentic, config=config)):
        assert service._agentic_gate_open(force_legacy_rag=False) is expected


@pytest.mark.asyncio
async def test_the_stream_binds_its_profile_for_every_step(monkeypatch) -> None:
    from app.rag import service

    client = MagicMock()

    async def _inner(*args, **kwargs):
        yield llm_routing.bound_client()
        yield llm_routing.bound_client()

    monkeypatch.setattr(service, "_stream_answer", _inner)

    seen = [e async for e in service.stream_answer("q", None, resolved=resolved_with(client))]

    assert (seen, llm_routing.bound_client()) == ([client, client], None)


@pytest.mark.asyncio
async def test_a_stream_closed_from_another_context_does_not_raise(monkeypatch) -> None:
    import asyncio

    from app.rag import service

    closed: list = []

    async def _inner(*args, **kwargs):
        try:
            yield 1
            yield 2
        finally:
            closed.append(llm_routing.bound_client())

    monkeypatch.setattr(service, "_stream_answer", _inner)
    client = MagicMock()
    stream = service.stream_answer("q", None, resolved=resolved_with(client))

    first = await asyncio.create_task(anext(stream))
    await stream.aclose()

    assert (first, closed) == (1, [client])


@pytest.mark.asyncio
async def test_find_binds_its_profile_for_the_helpers(monkeypatch) -> None:
    from app.rag import service

    client = MagicMock()
    seen: list = []

    async def _inner(*args, **kwargs):
        seen.append(llm_routing.bound_client())
        return {}

    monkeypatch.setattr(service, "_find_files", _inner)

    await service.find_files("q", DRIVE, resolved=resolved_with(client))

    assert seen == [client]


@pytest.mark.asyncio
async def test_the_agentic_loop_is_given_the_asks_drive(monkeypatch) -> None:
    from app.rag import service

    loop = AsyncMock()
    monkeypatch.setattr(service, "run_agentic_loop", loop)

    with llm_routing.bound(resolved_with(MagicMock(), agentic=True)):
        await service._run_agentic(
            query="q", credential=None, drive=DRIVE, viewer_id=None, temperature=None
        )

    assert loop.await_args.kwargs["drive"] == DRIVE


@pytest.mark.asyncio
async def test_the_eval_runner_runs_every_stage_with_the_rag_profile_bound(
    monkeypatch,
) -> None:
    from app.evals import __main__ as runner

    client = MagicMock()
    monkeypatch.setattr(
        llm_routing, "resolve_without_ceiling", lambda feature: resolved_with(client)
    )
    seen: list = []

    async def _run(args):
        seen.append(llm_routing.bound_client())
        return 0

    monkeypatch.setattr(runner, "_run", _run)

    assert await runner._run_routed(MagicMock()) == 0
    assert seen == [client]


@pytest.mark.asyncio
async def test_ask_hands_the_resolved_profile_to_the_stream(monkeypatch, use_llm) -> None:
    from app.routers import rag as rag_router
    from app.schemas import AskRequest

    _rag_on(monkeypatch)
    client = MagicMock()
    use_llm(client)
    seen: list = []

    async def _stream(**kwargs):
        seen.append(kwargs["resolved"].client)
        return
        yield

    monkeypatch.setattr(rag_router, "stream_answer", _stream)

    response = await rag_router.ask_endpoint(
        AskRequest(query="what happened"), None, None, DRIVE, None
    )
    [chunk async for chunk in response.body_iterator]

    assert seen == [client]


def test_the_eval_entry_point_runs_bound_and_refuses_an_unrouted_rag(monkeypatch) -> None:
    from app.evals import __main__ as runner

    client = MagicMock()
    seen: list = []

    async def _run(args):
        seen.append(llm_routing.bound_client())
        return 0

    monkeypatch.setattr(runner, "_run", _run)
    monkeypatch.setattr(
        llm_routing, "resolve_without_ceiling", lambda feature: resolved_with(client)
    )
    monkeypatch.setattr(
        runner, "build_parser", lambda: MagicMock(parse_args=lambda argv: MagicMock())
    )

    assert runner.main([]) == 0
    assert seen == [client]

    monkeypatch.setattr(
        llm_routing, "resolve_without_ceiling", lambda feature: Skip("no profile")
    )
    assert runner.main([]) == 2
    assert seen == [client]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "status", "detail"),
    [
        (Skip("llm_cloud off"), 400, "profile_unavailable"),
        (Defer("unavailable"), 503, "LLM policy is unavailable, try again shortly"),
    ],
)
async def test_ask_with_a_rejected_choice_opens_no_stream(
    monkeypatch, use_llm, result, status, detail
) -> None:
    from app.routers import rag as rag_router
    from app.schemas import AskRequest

    _rag_on(monkeypatch)
    asked = use_llm(result=result)
    stream = MagicMock()
    monkeypatch.setattr(rag_router, "stream_answer", stream)

    with pytest.raises(HTTPException) as exc:
        await rag_router.ask_endpoint(
            AskRequest(query="what happened", profile="big"), None, None, DRIVE, None
        )

    assert (exc.value.status_code, exc.value.detail) == (status, detail)
    assert asked.requested == ["big"]
    stream.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [None, "big"])
async def test_every_step_of_an_ask_runs_on_the_profile_it_resolved(
    monkeypatch, use_llm, profile
) -> None:
    from app.rag import service
    from app.routers import rag as rag_router
    from app.schemas import AskRequest

    _rag_on(monkeypatch)
    client = MagicMock()
    asked = use_llm(client)
    seen: list = []

    async def _inner(*args, **kwargs):
        seen.append(llm_routing.bound_client())
        yield service.AnswerEvent(kind="answer_chunk", data={})
        seen.append(llm_routing.bound_client())

    monkeypatch.setattr(service, "_stream_answer", _inner)

    response = await rag_router.ask_endpoint(
        AskRequest(query="what happened", profile=profile), None, None, DRIVE, None
    )
    [chunk async for chunk in response.body_iterator]

    assert asked.requested == [profile]
    assert seen == [client, client]


@pytest.mark.parametrize(
    ("model", "extra"),
    [
        ("SummaryProfileRequest", {}),
        ("DetailedSummaryRegenerateRequest", {}),
        ("AskRequest", {"query": "what"}),
    ],
)
@pytest.mark.parametrize(
    ("profile", "valid"),
    [
        (None, True),
        ("big", True),
        ("local-2_b", True),
        ("", False),
        ("Big", False),
        ("-big", False),
        ("a" * 33, False),
        ("big/../x", False),
        ("big\n", False),
    ],
)
def test_a_choice_must_be_a_profile_name(model, extra, profile, valid) -> None:
    from pydantic import ValidationError

    from app import schemas

    make = getattr(schemas, model)
    if valid:
        assert make(profile=profile, **extra).profile == profile
    else:
        with pytest.raises(ValidationError):
            make(profile=profile, **extra)


def test_summary_regenerate_takes_an_optional_body_on_the_wire(monkeypatch, use_llm) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routers import summaries as router_mod

    worker, _ = _regenerate_router(monkeypatch)
    asked = use_llm(MagicMock())
    monkeypatch.setattr(router_mod, "generate_summary", AsyncMock())
    app = FastAPI()
    app.include_router(router_mod.router)
    client = TestClient(app)
    url = "/files/f1/summary/regenerate"
    headers = {"X-Lit-Drive": DRIVE}

    assert client.post(url, headers=headers).status_code == 200
    assert client.post(url, headers=headers, json={"profile": ""}).status_code == 422
    assert client.post(url, headers=headers, json={"profile": "big"}).status_code == 200

    assert asked.requested == [None, "big"]
    worker.enqueue.assert_awaited_once_with("f1")
    router_mod.generate_summary.assert_awaited_once()
