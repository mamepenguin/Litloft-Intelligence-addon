"""Tests for app.rag.service module.

answer_question is the full-pipeline orchestrator: retrieve ->
access filter -> contexts -> LLM -> parse -> AnswerResponse.

All collaborators (retriever, context builder, LLM) are mocked.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub out heavy dependencies before importing service + retriever.
for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "numpy",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.config import LLMConfig, RagConfig  # noqa: E402
from app.rag.context import ContextSnippet, FileContext  # noqa: E402
from app.rag.retriever import RetrievedFile  # noqa: E402
from app.rag.service import (  # noqa: E402
    AnswerEvent,
    AnswerResponse,
    answer_question,
    stream_answer,
)
from app.search import MatchInfo, SegmentGroup  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _retrieved(file_id: str, score: float = 0.9) -> RetrievedFile:
    match = MatchInfo(
        match_type="transcript",
        text="snippet",
        score=score,
        timestamp_start=0.0,
        timestamp_end=10.0,
    )
    return RetrievedFile(
        file_id=file_id,
        drive="Videos",
        filename=f"{file_id}.mp4",
        file_type="video",
        title=f"Title {file_id}",
        description=f"Description {file_id}",
        score=score,
        match_types=("transcript",),
        segments=(SegmentGroup(time_range=(0.0, 10.0), matches=(match,)),),
    )


def _context(file_id: str) -> FileContext:
    return FileContext(
        file_id=file_id,
        filename=f"{file_id}.mp4",
        drive="Videos",
        file_type="video",
        title=f"Title {file_id}",
        description=f"Description {file_id}",
        snippets=(
            ContextSnippet(
                source="transcript",
                text=f"sample text {file_id}",
                location="0:05",
            ),
        ),
        total_chars=len(f"sample text {file_id}"),
    )


def _make_llm_mock(response: dict | list | None):
    client = MagicMock()
    client.enabled = True
    client.generate_json = AsyncMock(return_value=response)
    return client


@pytest.fixture()
def patched_rag_enabled(monkeypatch, make_settings):
    """Settings with features.rag=True and LLM enabled."""
    settings = make_settings(
        features=type(make_settings().features)(
            indexing=True,
            search=True,
            auto_tags="false",
            summaries="false",
            rag=True,
        ),
        llm=LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="llama3",
        ),
        rag=RagConfig(),
    )
    monkeypatch.setattr("app.config.settings", settings)
    monkeypatch.setattr("app.rag.service.settings", settings)
    return settings


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestAnswerQuestionHappyPath:
    """T1: full pipeline with valid LLM response."""

    @pytest.mark.asyncio
    async def test_returns_answer_response_with_all_fields(
        self, monkeypatch, patched_rag_enabled
    ):
        candidates = [_retrieved("f1"), _retrieved("f2")]
        monkeypatch.setattr(
            "app.rag.service.retrieve_candidates",
            AsyncMock(return_value=candidates),
        )
        monkeypatch.setattr(
            "app.rag.service.assemble_contexts",
            lambda cands, cfg: [_context(c.file_id) for c in cands],
        )

        llm = _make_llm_mock({
            "answer": "The files cover topic X [1].",
            "citations": [
                {"file_id": "f1", "quote": "topic X explanation", "relevance": 0.9}
            ],
        })
        monkeypatch.setattr(
            "app.rag.service.get_llm_client", lambda: llm
        )

        result = await answer_question(
            query="What topic is covered?",
            hv_token="token",
        )

        assert isinstance(result, AnswerResponse)
        assert result.query == "What topic is covered?"
        assert result.answer == "The files cover topic X [1]."
        assert result.retrieved_count == 2
        assert len(result.sources) == 2
        # Citations were validated against allowed_file_ids and kept f1.
        assert len(result.citations) == 1
        cit = result.citations[0]
        assert cit["file_id"] == "f1"
        # sources contain enough info to render UI cards
        src_ids = {s["file_id"] for s in result.sources}
        assert src_ids == {"f1", "f2"}
        # took_ms is populated
        assert isinstance(result.took_ms, int)
        assert result.took_ms >= 0


# ---------------------------------------------------------------------------
# No candidates
# ---------------------------------------------------------------------------


class TestAnswerQuestionNoCandidates:
    """T2: retriever returns [] -> short-circuit with empty answer."""

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty_answer(
        self, monkeypatch, patched_rag_enabled
    ):
        monkeypatch.setattr(
            "app.rag.service.retrieve_candidates",
            AsyncMock(return_value=[]),
        )
        llm = _make_llm_mock({"answer": "should not be called", "citations": []})
        monkeypatch.setattr("app.rag.service.get_llm_client", lambda: llm)

        result = await answer_question(
            query="anything",
            hv_token=None,
        )

        assert result.answer is None
        assert result.citations == []
        assert result.sources == []
        assert result.retrieved_count == 0
        # The LLM must NOT be called when we have no candidates.
        llm.generate_json.assert_not_awaited()


# ---------------------------------------------------------------------------
# LLM returns None (parse failure)
# ---------------------------------------------------------------------------


class TestAnswerQuestionLLMFailure:
    """T3: LLM returns None -> answer=None but sources populated."""

    @pytest.mark.asyncio
    async def test_none_llm_response_returns_sources_without_answer(
        self, monkeypatch, patched_rag_enabled
    ):
        candidates = [_retrieved("f1")]
        monkeypatch.setattr(
            "app.rag.service.retrieve_candidates",
            AsyncMock(return_value=candidates),
        )
        monkeypatch.setattr(
            "app.rag.service.assemble_contexts",
            lambda cands, cfg: [_context(c.file_id) for c in cands],
        )
        # LLM returned unparseable output.
        llm = _make_llm_mock(None)
        monkeypatch.setattr("app.rag.service.get_llm_client", lambda: llm)

        result = await answer_question(
            query="q", hv_token="t"
        )

        assert result.answer is None
        assert result.citations == []
        # Sources ARE populated so user can see retrieval worked.
        assert len(result.sources) == 1
        assert result.sources[0]["file_id"] == "f1"
        assert result.retrieved_count == 1

    @pytest.mark.asyncio
    async def test_llm_returns_unparseable_shape(
        self, monkeypatch, patched_rag_enabled
    ):
        """Non-dict return from generate_json (parser rejects) -> same shape."""
        candidates = [_retrieved("f1")]
        monkeypatch.setattr(
            "app.rag.service.retrieve_candidates",
            AsyncMock(return_value=candidates),
        )
        monkeypatch.setattr(
            "app.rag.service.assemble_contexts",
            lambda cands, cfg: [_context(c.file_id) for c in cands],
        )
        llm = _make_llm_mock([{"wrong": "shape"}])
        monkeypatch.setattr("app.rag.service.get_llm_client", lambda: llm)

        result = await answer_question(query="q", hv_token="t")

        assert result.answer is None
        assert result.citations == []
        assert len(result.sources) == 1


# ---------------------------------------------------------------------------
# Citation filtering
# ---------------------------------------------------------------------------


class TestAnswerQuestionCitationFiltering:
    """T4: LLM citations with unknown file_ids are dropped."""

    @pytest.mark.asyncio
    async def test_drops_fabricated_file_id_citations(
        self, monkeypatch, patched_rag_enabled
    ):
        candidates = [_retrieved("real-1"), _retrieved("real-2")]
        monkeypatch.setattr(
            "app.rag.service.retrieve_candidates",
            AsyncMock(return_value=candidates),
        )
        monkeypatch.setattr(
            "app.rag.service.assemble_contexts",
            lambda cands, cfg: [_context(c.file_id) for c in cands],
        )

        llm = _make_llm_mock({
            "answer": "Covered across multiple files.",
            "citations": [
                {"file_id": "real-1", "quote": "ok", "relevance": 0.8},
                {"file_id": "hallucinated", "quote": "bad", "relevance": 0.9},
                {"file_id": "real-2", "quote": "ok2", "relevance": 0.7},
            ],
        })
        monkeypatch.setattr("app.rag.service.get_llm_client", lambda: llm)

        result = await answer_question(query="q", hv_token="t")

        assert result.answer == "Covered across multiple files."
        # Only the two real file_ids should appear in citations.
        cited_ids = {c["file_id"] for c in result.citations}
        assert cited_ids == {"real-1", "real-2"}

    @pytest.mark.asyncio
    async def test_all_citations_fabricated_keeps_answer_empty_citations(
        self, monkeypatch, patched_rag_enabled
    ):
        candidates = [_retrieved("real-1")]
        monkeypatch.setattr(
            "app.rag.service.retrieve_candidates",
            AsyncMock(return_value=candidates),
        )
        monkeypatch.setattr(
            "app.rag.service.assemble_contexts",
            lambda cands, cfg: [_context(c.file_id) for c in cands],
        )
        llm = _make_llm_mock({
            "answer": "Some answer",
            "citations": [
                {"file_id": "fake1", "quote": "x", "relevance": 0.9},
                {"file_id": "fake2", "quote": "y", "relevance": 0.9},
            ],
        })
        monkeypatch.setattr("app.rag.service.get_llm_client", lambda: llm)

        result = await answer_question(query="q", hv_token="t")

        assert result.answer == "Some answer"
        assert result.citations == []
        # sources still populated
        assert len(result.sources) == 1


# ---------------------------------------------------------------------------
# top_k forwarding
# ---------------------------------------------------------------------------


class TestAnswerQuestionTopK:
    """T5: default top_k comes from settings.rag.top_k."""

    @pytest.mark.asyncio
    async def test_uses_rag_config_top_k_by_default(
        self, monkeypatch, make_settings
    ):
        settings = make_settings(
            features=type(make_settings().features)(rag=True),
            llm=LLMConfig(
                provider="openai_compatible",
                base_url="http://localhost:11434/v1",
                model="llama3",
            ),
            rag=RagConfig(top_k=7),
        )
        monkeypatch.setattr("app.config.settings", settings)
        monkeypatch.setattr("app.rag.service.settings", settings)

        retrieve_spy = AsyncMock(return_value=[])
        monkeypatch.setattr(
            "app.rag.service.retrieve_candidates", retrieve_spy
        )
        monkeypatch.setattr(
            "app.rag.service.get_llm_client",
            lambda: _make_llm_mock({"answer": "x", "citations": []}),
        )

        await answer_question(query="q", hv_token=None)

        assert retrieve_spy.await_count == 1
        kwargs = retrieve_spy.call_args.kwargs
        assert kwargs.get("top_k") == 7

    @pytest.mark.asyncio
    async def test_explicit_top_k_overrides_default(
        self, monkeypatch, patched_rag_enabled
    ):
        retrieve_spy = AsyncMock(return_value=[])
        monkeypatch.setattr(
            "app.rag.service.retrieve_candidates", retrieve_spy
        )
        monkeypatch.setattr(
            "app.rag.service.get_llm_client",
            lambda: _make_llm_mock({"answer": "x", "citations": []}),
        )

        await answer_question(query="q", hv_token=None, top_k=3)

        kwargs = retrieve_spy.call_args.kwargs
        assert kwargs.get("top_k") == 3


# ---------------------------------------------------------------------------
# took_ms
# ---------------------------------------------------------------------------


class TestAnswerQuestionTookMs:
    """T6: took_ms should be a non-negative int."""

    @pytest.mark.asyncio
    async def test_took_ms_populated_empty_path(
        self, monkeypatch, patched_rag_enabled
    ):
        monkeypatch.setattr(
            "app.rag.service.retrieve_candidates",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(
            "app.rag.service.get_llm_client",
            lambda: _make_llm_mock({"answer": "x", "citations": []}),
        )

        result = await answer_question(query="q", hv_token=None)

        assert isinstance(result.took_ms, int)
        assert result.took_ms >= 0

    @pytest.mark.asyncio
    async def test_took_ms_populated_full_path(
        self, monkeypatch, patched_rag_enabled
    ):
        candidates = [_retrieved("f1")]
        monkeypatch.setattr(
            "app.rag.service.retrieve_candidates",
            AsyncMock(return_value=candidates),
        )
        monkeypatch.setattr(
            "app.rag.service.assemble_contexts",
            lambda cands, cfg: [_context(c.file_id) for c in cands],
        )
        monkeypatch.setattr(
            "app.rag.service.get_llm_client",
            lambda: _make_llm_mock(
                {
                    "answer": "Answer",
                    "citations": [
                        {"file_id": "f1", "quote": "q", "relevance": 0.9}
                    ],
                }
            ),
        )

        result = await answer_question(query="q", hv_token="t")

        assert isinstance(result.took_ms, int)
        assert result.took_ms >= 0


# ---------------------------------------------------------------------------
# Filter forwarding
# ---------------------------------------------------------------------------


class TestAnswerQuestionFilterForwarding:
    """file_type and drive filters should reach the retriever."""

    @pytest.mark.asyncio
    async def test_file_type_and_drive_forwarded(
        self, monkeypatch, patched_rag_enabled
    ):
        retrieve_spy = AsyncMock(return_value=[])
        monkeypatch.setattr(
            "app.rag.service.retrieve_candidates", retrieve_spy
        )
        monkeypatch.setattr(
            "app.rag.service.get_llm_client",
            lambda: _make_llm_mock({"answer": "x", "citations": []}),
        )

        await answer_question(
            query="q",
            hv_token=None,
            file_type="document",
            drive="Docs",
        )

        kwargs = retrieve_spy.call_args.kwargs
        assert kwargs.get("file_type") == "document"
        assert kwargs.get("drive") == "Docs"


# ---------------------------------------------------------------------------
# Streaming path: stream_answer
# ---------------------------------------------------------------------------


def _make_stream_llm_mock(deltas: list[str]):
    """Build a MagicMock LLM client whose generate_stream yields ``deltas``.

    Returns a tuple ``(client, deltas)`` so tests can assert on what
    was emitted without re-typing the sequence.
    """
    client = MagicMock()
    client.enabled = True

    async def _stream(*args, **kwargs):
        for delta in deltas:
            yield delta

    client.generate_stream = _stream
    return client


async def _collect(gen) -> list[AnswerEvent]:
    events: list[AnswerEvent] = []
    async for event in gen:
        events.append(event)
    return events


class TestStreamAnswerHappyPath:
    """stream_answer yields keywords -> sources -> chunks -> citations -> done."""

    @pytest.mark.asyncio
    async def test_emits_ordered_events(
        self, monkeypatch, patched_rag_enabled
    ):
        candidates = [_retrieved("f1"), _retrieved("f2")]
        monkeypatch.setattr(
            "app.rag.service.transform_query",
            AsyncMock(return_value="京都 紅葉"),
        )
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords",
            AsyncMock(return_value=candidates),
        )
        monkeypatch.setattr(
            "app.rag.service.assemble_contexts",
            lambda cands, cfg: [_context(c.file_id) for c in cands],
        )

        # Two-chunk stream that together reconstitutes valid JSON with
        # a [1] citation pointing at f1 (in the allowed candidate set).
        deltas = [
            '{"answer": "京都の紅葉 [1]", ',
            '"citations": [{"file_id": "f1", "quote": "紅葉", "relevance": 0.9}]}',
        ]
        llm = _make_stream_llm_mock(deltas)
        monkeypatch.setattr("app.rag.service.get_llm_client", lambda: llm)

        events = await _collect(stream_answer(
            query="京都の紅葉について",
            hv_token="tok",
        ))

        kinds = [e.kind for e in events]
        assert kinds == [
            "keywords",
            "sources",
            "answer_chunk",
            "answer_chunk",
            "citations",
            "done",
        ]

        assert events[0].data["keywords"] == "京都 紅葉"
        sources = events[1].data["sources"]
        assert {s["file_id"] for s in sources} == {"f1", "f2"}

        # The two answer_chunk events carry the raw deltas verbatim.
        assert events[2].data["delta"] == deltas[0]
        assert events[3].data["delta"] == deltas[1]

        # Citations are parsed from the full buffered JSON and the
        # f1 citation survives the allowed-id check.
        citations = events[4].data["citations"]
        assert len(citations) == 1
        assert citations[0]["file_id"] == "f1"

        # done carries timing metadata.
        done_payload = events[5].data
        assert done_payload.get("retrieved_count") == 2
        assert "took_ms" in done_payload


class TestStreamAnswerEmptyRetrieval:
    @pytest.mark.asyncio
    async def test_empty_candidates_short_circuits(
        self, monkeypatch, patched_rag_enabled
    ):
        monkeypatch.setattr(
            "app.rag.service.transform_query",
            AsyncMock(return_value="kw"),
        )
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords",
            AsyncMock(return_value=[]),
        )
        llm = _make_stream_llm_mock(["should", "not", "stream"])
        monkeypatch.setattr("app.rag.service.get_llm_client", lambda: llm)

        events = await _collect(stream_answer(query="anything", hv_token=None))

        kinds = [e.kind for e in events]
        # Keywords -> empty sources -> empty citations -> done.
        # Crucially, NO answer_chunk events: the LLM stream is never
        # opened when there are no candidates.
        assert "answer_chunk" not in kinds
        assert kinds[0] == "keywords"
        assert kinds[-1] == "done"
        # The empty sources event is emitted so the UI can say "nothing found".
        sources_events = [e for e in events if e.kind == "sources"]
        assert sources_events and sources_events[0].data["sources"] == []


class TestStreamAnswerHallucinationFilter:
    @pytest.mark.asyncio
    async def test_drops_citations_for_unknown_file_ids(
        self, monkeypatch, patched_rag_enabled
    ):
        candidates = [_retrieved("f1")]
        monkeypatch.setattr(
            "app.rag.service.transform_query",
            AsyncMock(return_value="kw"),
        )
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords",
            AsyncMock(return_value=candidates),
        )
        monkeypatch.setattr(
            "app.rag.service.assemble_contexts",
            lambda cands, cfg: [_context(c.file_id) for c in cands],
        )

        # The LLM claims a citation pointing at "f999" which was NOT
        # in the retrieved set. The anti-hallucination filter must
        # drop it; the "f1" citation survives.
        full_json = (
            '{"answer": "text [1][2]", "citations": ['
            '{"file_id": "f1", "quote": "q", "relevance": 0.9},'
            '{"file_id": "f999", "quote": "fake", "relevance": 0.5}'
            "]}"
        )
        llm = _make_stream_llm_mock([full_json])
        monkeypatch.setattr("app.rag.service.get_llm_client", lambda: llm)

        events = await _collect(stream_answer(query="q", hv_token="t"))

        citations_event = next(e for e in events if e.kind == "citations")
        kept = citations_event.data["citations"]
        assert len(kept) == 1
        assert kept[0]["file_id"] == "f1"


class TestStreamAnswerUnparseableJSON:
    @pytest.mark.asyncio
    async def test_unparseable_stream_yields_empty_citations(
        self, monkeypatch, patched_rag_enabled
    ):
        """Prose-only output still emits a clean terminal ``citations`` event."""
        candidates = [_retrieved("f1")]
        monkeypatch.setattr(
            "app.rag.service.transform_query",
            AsyncMock(return_value="kw"),
        )
        monkeypatch.setattr(
            "app.rag.service.retrieve_with_keywords",
            AsyncMock(return_value=candidates),
        )
        monkeypatch.setattr(
            "app.rag.service.assemble_contexts",
            lambda cands, cfg: [_context(c.file_id) for c in cands],
        )
        llm = _make_stream_llm_mock(["this is not json at all"])
        monkeypatch.setattr("app.rag.service.get_llm_client", lambda: llm)

        events = await _collect(stream_answer(query="q", hv_token="t"))

        citations_event = next(e for e in events if e.kind == "citations")
        assert citations_event.data["citations"] == []
        # The raw answer_chunk was still streamed — the user gets to
        # see whatever the model said even if we can't build citations.
        chunks = [e for e in events if e.kind == "answer_chunk"]
        assert chunks and chunks[0].data["delta"] == "this is not json at all"
