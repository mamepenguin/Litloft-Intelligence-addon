"""SPEC-ADDON-053: asking a question about one file, from that file's indexed text.

The routes are called through the addon's own FastAPI app, as the core proxy
calls them, so the validation and the stream are the ones a browser sees. What
was sent is read from the prompt the routed LLM records.
"""

from __future__ import annotations

import json
import logging
from urllib.parse import quote

import httpx
import pytest
from sqlalchemy import event

import app.database as database
from app.config import LLMConfig
from app.routers import rag as rag_router
from tests.file_question_doubles import (
    ASKABLE,
    DRIVE,
    EPUB,
    MEDIA,
    NOTE,
    OTHER,
    PDF,
    RecordingLLM,
    Seeder,
    chunk,
    drive_to_disconnect,
    frames,
    mentioned,
    numbered_in_order,
    settings_for_question,
    toward,
)
from tests.llm_helpers import resolved_with
from tests.search_endpoint_db import patch_everywhere
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(search_index):  # noqa: F811
    """The search DB, seeding a file's chunks and their stored vectors."""
    return Seeder(search_index)


@pytest.fixture()
def addon(monkeypatch, make_settings):
    """The addon's settings, applied to every addon module that reads them."""

    def _apply(**overrides):
        from app.main import app  # noqa: F401 - import every router before patching

        settings = settings_for_question(make_settings, **overrides)
        patch_everywhere(monkeypatch, "settings", settings)
        return settings

    _apply()
    return _apply


class Routing:
    """What ``llm_routing.resolve`` answers, and every call it was asked about."""

    def __init__(self):
        self.asked: list[tuple[str, str, str | None]] = []
        self.result = None

    def serve(self, client: RecordingLLM, *, output_language: str = "auto") -> RecordingLLM:
        self.result = resolved_with(
            client,
            config=LLMConfig(
                provider="openai_compatible", base_url="http://llm.test/v1",
                model="test-llm", output_language=output_language,
            ),
        )
        return client

    def skip(self) -> None:
        from app import llm_routing

        self.result = llm_routing.Skip("disabled")

    def defer(self) -> None:
        from app import llm_routing

        self.result = llm_routing.Defer("policy unreadable")


@pytest.fixture()
def llm(monkeypatch):
    """The routing the routes see; ``llm.serve()`` is the only LLM they can reach."""
    from app import llm_routing

    routing = Routing()

    async def _resolve(drive, feature, requested=None):
        routing.asked.append((drive, feature, requested))
        return routing.result

    monkeypatch.setattr(llm_routing, "resolve", _resolve)
    return routing


@pytest.fixture()
async def call(search_index):  # noqa: F811
    """The addon's HTTP app, called the way the core proxy calls it."""
    from app.main import app

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://intelligence") as client:

        async def _call(method: str, path: str, *, json=None, drive: str | None = DRIVE):
            headers = {} if drive is None else {"X-Lit-Drive": quote(drive)}
            return await client.request(method, path, json=json, headers=headers)

        yield _call


@pytest.fixture()
async def ask_slots(monkeypatch):
    """The Ask semaphore, reset so its capacity is known in this test's loop."""
    monkeypatch.setattr(rag_router, "_ask_semaphore", None)
    monkeypatch.setattr(rag_router, "_ask_semaphore_loop", None)
    return rag_router._get_ask_semaphore()


@pytest.fixture()
def statements(search_index):  # noqa: F811
    """Every statement the search engine runs, and the writes among them."""

    class Log:
        def __init__(self):
            self.statements: list[str] = []

        def _before(self, _conn, _cursor, statement, _params, _context, _many):
            self.statements.append(statement)

        def reset(self) -> None:
            self.statements.clear()

        def writes(self) -> list[str]:
            verbs = ("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "DROP", "ALTER")
            return [s for s in self.statements if s.lstrip().upper().startswith(verbs)]

    log = Log()
    engine = database._search_engine
    event.listen(engine, "before_cursor_execute", log._before)
    try:
        yield log
    finally:
        event.remove(engine, "before_cursor_execute", log._before)


async def question(call, file_id: str, text: str = "what is this") -> dict:
    """Ask, and return the SSE frames by event kind (the last of each)."""
    response = await call("POST", f"/files/{file_id}/ask", json={"question": text})
    assert response.status_code == 200, response.text
    return dict(frames(response.text))


def _toward_first_axis(db: Seeder):
    """A query embedding whose cosine to each seeded chunk is that seed's ``cos``.

    ``toward(dim, cos)`` seeds make the chunk vectors' similarity to the first
    axis the number each test ranks by, so the question's own embedding has to
    point down that axis for a rank to be observable at all.
    """

    def _embed_query(_text: str):
        return toward(db.db.dim, 1.0)

    return _embed_query


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestAvailability:
    async def test_available_for_an_active_indexed_file_with_a_chunk(self, db, addon, call):
        """SPEC-ADDON-053 I6: an indexed file of the request drive with one text chunk."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)

        response = await call("GET", f"/files/{MEDIA}/ask")

        assert response.status_code == 200
        assert response.json() == {"available": True}

    async def test_available_for_a_document_chunk(self, db, addon, call):
        """SPEC-ADDON-053 I6: a document's fts chunk is text too."""
        db.file(PDF, mime="application/pdf").document(chunk("one", 20), chunk_index=0)

        response = await call("GET", f"/files/{PDF}/ask")

        assert response.status_code == 200
        assert response.json() == {"available": True}

    @pytest.mark.parametrize(
        "case",
        ["another drive", "not indexed", "inactive", "no chunk"],
        ids=["another-drive", "not-indexed", "inactive", "no-chunk"],
    )
    async def test_unavailable_for_everything_but_an_askable_file(self, case, db, addon, call):
        """SPEC-ADDON-053 I6: anything else is ``available: false``, not an error."""
        if case == "another drive":
            db.file(MEDIA, drive=OTHER).transcript(chunk("one", 20), start=0)
        elif case == "inactive":
            db.file(MEDIA, active=False).transcript(chunk("one", 20), start=0)
        elif case == "no chunk":
            db.file(MEDIA)

        response = await call("GET", f"/files/{MEDIA}/ask")

        assert response.status_code == 200
        assert response.json() == {"available": False}

    async def test_unavailable_without_the_drive_header(self, db, addon, call):
        """SPEC-ADDON-053 6: the route is drive-scoped, so the header is required."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)

        response = await call("GET", f"/files/{MEDIA}/ask", drive=None)

        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Which file may be asked about
# ---------------------------------------------------------------------------


class TestTheRequestedFile:
    @pytest.mark.parametrize(
        "case",
        ["another drive", "not indexed", "inactive", "no chunk"],
        ids=["another-drive", "not-indexed", "inactive", "no-chunk"],
    )
    async def test_a_file_this_drive_cannot_ask_about_is_404(self, case, db, addon, llm, call):
        """SPEC-ADDON-053 2: nothing outside the request drive's askable files is answered."""
        if case == "another drive":
            db.file(MEDIA, drive=OTHER).transcript(chunk("one", 20), start=0)
        elif case == "inactive":
            db.file(MEDIA, active=False).transcript(chunk("one", 20), start=0)
        elif case == "no chunk":
            db.file(MEDIA)
        db.file(ASKABLE).transcript(chunk("good", 20), start=0)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        response = await call("POST", f"/files/{MEDIA}/ask", json={"question": "what is this"})
        askable = await call("POST", f"/files/{ASKABLE}/ask", json={"question": "what is this"})

        assert response.status_code == 404
        assert askable.status_code == 200
        assert len(recording.prompts) == 1

    async def test_only_the_requested_files_chunks_reach_the_llm(self, db, addon, llm, call):
        """SPEC-ADDON-053 I1: the LLM receives the asked file's chunks, and no other file's."""
        mine = chunk("mine", 40)
        theirs = chunk("theirs", 40)
        db.file(MEDIA).transcript(mine, start=0)
        db.file("vid000000002").transcript(theirs, start=0)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        response = await call("POST", f"/files/{MEDIA}/ask", json={"question": "what is this"})

        assert response.status_code == 200
        assert mine in recording.user_prompt
        assert theirs not in recording.user_prompt

    async def test_the_routing_entry_is_the_one_ask_uses(self, db, addon, llm, call):
        """SPEC-ADDON-053 I4: the client comes from ``resolve(drive, "rag")``."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.serve(RecordingLLM(deltas=("answer",)))

        await question(call, MEDIA)

        assert llm.asked == [(DRIVE, "rag", None)]

    async def test_a_chosen_profile_is_passed_through_unchanged(self, db, addon, llm, call):
        """SPEC-ADDON-053 I4: a profile the viewer chose is never replaced with another."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.serve(RecordingLLM(deltas=("answer",)))

        await call(
            "POST", f"/files/{MEDIA}/ask",
            json={"question": "what is this", "profile": "cloud"},
        )

        assert llm.asked == [(DRIVE, "rag", "cloud")]


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


class TestGates:
    async def test_rag_off_answers_as_ask_does(self, db, addon, llm, call):
        """SPEC-ADDON-053 I1: ``features.rag`` off closes the file question as ``/ask`` closes."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))
        addon(rag=False)

        file_response = await call("POST", f"/files/{MEDIA}/ask", json={"question": "what is this"})
        ask_response = await call("POST", "/ask", json={"query": "what is this"})

        assert file_response.status_code == 400
        assert file_response.json() == ask_response.json()
        assert recording.prompts == []

    async def test_no_llm_is_400(self, db, addon, llm, call):
        """SPEC-ADDON-053 5: no LLM configured answers 400 "LLM is not enabled"."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.skip()

        response = await call("POST", f"/files/{MEDIA}/ask", json={"question": "what is this"})

        assert response.status_code == 400
        assert response.json()["detail"] == "LLM is not enabled"

    async def test_an_unservable_profile_is_400(self, db, addon, llm, call):
        """SPEC-ADDON-053 5: a profile that cannot be served answers ``profile_unavailable``."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.skip()

        response = await call(
            "POST", f"/files/{MEDIA}/ask",
            json={"question": "what is this", "profile": "cloud"},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "profile_unavailable"

    async def test_an_unreadable_policy_is_503_and_sends_nothing(self, db, addon, llm, call):
        """SPEC-ADDON-053 I4: a policy that cannot be read sends nothing to any LLM."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.defer()

        response = await call("POST", f"/files/{MEDIA}/ask", json={"question": "what is this"})

        assert response.status_code == 503
        assert response.json()["detail"].startswith("LLM policy is unavailable")

    async def test_a_full_semaphore_is_503_with_asks_detail(self, db, addon, llm, call, ask_slots):
        """SPEC-ADDON-053 5: a full Ask semaphore answers 503 the way ``/ask`` does."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.serve(RecordingLLM(deltas=("answer",)))
        for _ in range(rag_router._MAX_CONCURRENT_ASK):
            await ask_slots.acquire()
        try:
            file_response = await call(
                "POST", f"/files/{MEDIA}/ask", json={"question": "what is this"},
            )
            ask_response = await call("POST", "/ask", json={"query": "what is this"})
        finally:
            for _ in range(rag_router._MAX_CONCURRENT_ASK):
                ask_slots.release()

        assert file_response.status_code == 503
        assert ask_response.status_code == 503
        assert file_response.json() == ask_response.json()

    @pytest.mark.parametrize(
        "text",
        ["", "   ", " ab ", "a" * 1001],
        ids=["empty", "spaces", "two-non-space", "too-long"],
    )
    async def test_a_question_outside_the_length_bounds_is_422(self, text, db, addon, llm, call):
        """SPEC-ADDON-053 2: under 3 non-space characters or over 1000 is 422."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.serve(RecordingLLM(deltas=("answer",)))

        response = await call("POST", f"/files/{MEDIA}/ask", json={"question": text})

        assert response.status_code == 422

    @pytest.mark.parametrize(
        "text", ["abc", "  abc  ", "a" * 1000], ids=["three", "padded", "exact-1000"],
    )
    async def test_a_question_inside_the_bounds_is_accepted(self, text, db, addon, llm, call):
        """SPEC-ADDON-053 2: three non-space characters and 1000 characters are allowed."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.serve(RecordingLLM(deltas=("answer",)))

        response = await call("POST", f"/files/{MEDIA}/ask", json={"question": text})

        assert response.status_code == 200

    async def test_a_missing_question_is_422(self, db, addon, llm, call):
        """SPEC-ADDON-053 2: the body must carry a question."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)

        response = await call("POST", f"/files/{MEDIA}/ask", json={})

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------


class TestTheStream:
    async def test_the_events_run_answer_chunks_then_citations_then_meta_then_done(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-053 7: ``text/event-stream`` with the four kinds in that order."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.serve(RecordingLLM(deltas=("The ", "answer ", "is [1].")))

        response = await call("POST", f"/files/{MEDIA}/ask", json={"question": "what is this"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = frames(response.text)
        answer = [data for kind, data in events if kind == "answer_chunk"]
        assert [kind for kind, _ in events] == ["answer_chunk"] * len(answer) + [
            "citations", "meta", "done",
        ]
        assert "".join(data["delta"] for data in answer) == "The answer is [1]."

    async def test_the_meta_event_reports_partial_and_done_carries_no_error(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-053 7: ``meta`` holds ``partial``; a clean end has no ``error``."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.serve(RecordingLLM(deltas=("answer",)))

        events = await question(call, MEDIA)

        assert events["meta"] == {"partial": False}
        assert "error" not in events["done"]

    async def test_citations_name_only_the_chunk_numbers_that_were_sent(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-053 I3: a citation naming a number that was not sent is dropped."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        db.on(MEDIA).transcript(chunk("two", 20), start=10)
        llm.serve(RecordingLLM(deltas=("From [1], [2] and [9].",)))

        events = await question(call, MEDIA)

        assert [citation["n"] for citation in events["citations"]] == [1, 2]
        assert all(set(citation) == {"n", "position"} for citation in events["citations"])

    async def test_a_mid_stream_failure_ends_with_done_carrying_error(self, db, addon, llm, call):
        """SPEC-ADDON-053 7: a failure after the stream started ends with ``done`` and ``error``."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.serve(RecordingLLM(deltas=("half an answer", " and then"), fail_after=1))

        response = await call("POST", f"/files/{MEDIA}/ask", json={"question": "what is this"})

        events = frames(response.text)
        assert events[-1][0] == "done"
        assert events[-1][1].get("error")
        assert "".join(
            data["delta"] for kind, data in events if kind == "answer_chunk"
        ) == "half an answer"

    async def test_the_warning_names_the_file_and_never_the_question(
        self, db, addon, llm, call, caplog,
    ):
        """SPEC-ADDON-053 I5: a failed question logs ``file question failed`` and not its text."""
        asked = "what does the secret annex say"
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.serve(RecordingLLM(deltas=("partial", " and"), fail_after=1))
        caplog.set_level(logging.DEBUG)

        await call("POST", f"/files/{MEDIA}/ask", json={"question": asked})

        warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
        assert any(
            "file question failed" in record.getMessage() and MEDIA in record.getMessage()
            for record in warnings
        )
        assert all(asked not in record.getMessage() for record in caplog.records)

    async def test_a_question_writes_nothing(self, db, addon, llm, call, statements):
        """SPEC-ADDON-053 I5: neither the question, the answer nor the citations are stored."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.serve(RecordingLLM(deltas=("answer [1]",)))
        statements.reset()

        await question(call, MEDIA)

        assert statements.writes() == []


# ---------------------------------------------------------------------------
# The Ask semaphore slot
# ---------------------------------------------------------------------------


class TestTheAskSlot:
    async def test_the_slot_returns_when_the_stream_ends(self, db, addon, llm, call, ask_slots):
        """SPEC-ADDON-053 5: the end of the stream frees the slot."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.serve(RecordingLLM(deltas=("answer",)))

        for _ in range(rag_router._MAX_CONCURRENT_ASK + 1):
            await question(call, MEDIA)

        assert ask_slots._value == rag_router._MAX_CONCURRENT_ASK

    async def test_the_slot_returns_after_a_mid_stream_failure(self, db, addon, llm, call, ask_slots):
        """SPEC-ADDON-053 5: a failure inside the stream frees the slot too."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.serve(RecordingLLM(deltas=("half",), fail_after=1))

        await question(call, MEDIA)

        assert ask_slots._value == rag_router._MAX_CONCURRENT_ASK

    async def test_the_slot_returns_when_the_stream_cannot_be_prepared(
        self, db, addon, llm, call, ask_slots, monkeypatch,
    ):
        """SPEC-ADDON-053 5: an error before the stream started frees the slot."""
        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        llm.serve(RecordingLLM(deltas=("answer",)))

        def _unreadable(*_args, **_kwargs):
            raise RuntimeError("search db unavailable")

        patch_everywhere(monkeypatch, "get_search_db_read", _unreadable)
        patch_everywhere(monkeypatch, "get_search_db", _unreadable)

        response = await call("POST", f"/files/{MEDIA}/ask", json={"question": "what is this"})

        ended_with_an_error = bool(dict(frames(response.text)).get("done", {}).get("error"))
        assert response.status_code >= 500 or ended_with_an_error
        assert ask_slots._value == rag_router._MAX_CONCURRENT_ASK

    async def test_the_slot_returns_when_the_client_disconnects(
        self, db, addon, llm, call, ask_slots,
    ):
        """SPEC-ADDON-053 5: a client disconnect frees the slot and closes the LLM stream."""
        from app.main import app

        db.file(MEDIA).transcript(chunk("one", 20), start=0)
        recording = llm.serve(RecordingLLM(deltas=("answer",), stall=True))

        await drive_to_disconnect(
            app,
            path=f"/files/{MEDIA}/ask",
            body=json.dumps({"question": "what is this"}).encode(),
            headers=[
                (b"x-lit-drive", DRIVE.encode()),
                (b"content-type", b"application/json"),
            ],
            first_body_frame=1,
        )

        assert recording.closed is True
        assert ask_slots._value == rag_router._MAX_CONCURRENT_ASK


# ---------------------------------------------------------------------------
# What the prompt holds
# ---------------------------------------------------------------------------


class TestWhenEveryChunkFits:
    async def test_all_chunks_are_sent_in_file_order(self, db, addon, llm, call):
        """SPEC-ADDON-053 I2: chunk texts totalling at most the budget are all sent."""
        addon(budget=100)
        texts = [chunk("first", 30), chunk("second", 30), chunk("third", 30)]
        db.file(MEDIA)
        for offset, text in zip((0, 10, 20), texts):
            db.transcript(text, start=offset)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        events = await question(call, MEDIA)

        assert mentioned(recording.user_prompt, texts) == texts
        assert events["meta"] == {"partial": False}

    async def test_the_question_is_not_embedded(self, db, addon, llm, call, monkeypatch):
        """SPEC-ADDON-053 I2: ranking runs only when the chunks do not fit."""
        addon(budget=100)
        db.file(MEDIA).transcript(chunk("only", 40), start=0)
        llm.serve(RecordingLLM(deltas=("answer",)))
        embedded: list[str] = []

        def _embed_query(text: str):
            embedded.append(text)
            return toward(db.db.dim, 1.0)

        patch_everywhere(monkeypatch, "embed_query", _embed_query)

        await question(call, MEDIA)

        assert embedded == []


class TestWhenOnlySomeChunksFit:
    async def test_the_ranked_chunks_are_sent_in_file_order_within_the_budget(
        self, db, addon, llm, call, monkeypatch,
    ):
        """SPEC-ADDON-053 I2: rank picks, overflows are skipped, file order sends."""
        addon(budget=100)
        texts = [chunk("c0", 40), chunk("c1", 60), chunk("c2", 60), chunk("c3", 30)]
        cosines = [0.1, 0.9, 0.5, 0.2]
        db.file(MEDIA)
        for offset, (text, cos) in enumerate(zip(texts, cosines)):
            db.transcript(text, start=offset * 10, cos=cos)
        patch_everywhere(monkeypatch, "embed_query", _toward_first_axis(db))
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        events = await question(call, MEDIA)

        # c1 (0.9) then c2 (0.5) would overflow, so c3 (0.2) fills; c0 (0.1) has no room.
        assert mentioned(recording.user_prompt, texts) == [texts[1], texts[3]]
        assert sum(len(text) for text in (texts[1], texts[3])) <= 100
        assert events["meta"] == {"partial": True}

    async def test_the_best_ranked_chunk_is_cut_when_none_fits(
        self, db, addon, llm, call, monkeypatch,
    ):
        """SPEC-ADDON-053 I2: when no chunk fits whole, the best-ranked one is cut to the budget."""
        addon(budget=10)
        texts = [chunk("c0", 50), chunk("c1", 50)]
        db.file(MEDIA)
        for offset, (text, cos) in enumerate(zip(texts, (0.4, 0.8))):
            db.transcript(text, start=offset * 10, cos=cos)
        patch_everywhere(monkeypatch, "embed_query", _toward_first_axis(db))
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        events = await question(call, MEDIA)

        assert texts[1][:10] in recording.user_prompt
        assert texts[1] not in recording.user_prompt
        assert texts[0] not in recording.user_prompt
        assert events["meta"] == {"partial": True}

    async def test_file_order_is_used_when_the_question_cannot_be_embedded(
        self, db, addon, llm, call, monkeypatch,
    ):
        """SPEC-ADDON-053 I2: an embedding that fails falls back to file order, same fitting."""
        addon(budget=100)
        texts = [chunk("c0", 60), chunk("c1", 60), chunk("c2", 30)]
        db.file(MEDIA)
        for offset, (text, cos) in enumerate(zip(texts, (0.1, 0.9, 0.5))):
            db.transcript(text, start=offset * 10, cos=cos)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        def _unembeddable(*_args, **_kwargs):
            raise RuntimeError("embedding model unavailable")

        patch_everywhere(monkeypatch, "embed_query", _unembeddable)
        patch_everywhere(monkeypatch, "embed_texts", _unembeddable)

        events = await question(call, MEDIA)

        # c0 fits, c1 does not, c2 fills the rest.
        assert mentioned(recording.user_prompt, texts) == [texts[0], texts[2]]
        assert events["meta"] == {"partial": True}

    async def test_chunks_without_an_embedding_rank_after_every_embedded_chunk(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-053 I2: an unembedded chunk ranks last, in file order among its peers."""
        addon(budget=60)
        texts = [chunk("c0", 20), chunk("c1", 20), chunk("c2", 40)]
        db.file(MEDIA)
        db.transcript(texts[0], start=0, cos=None)
        db.transcript(texts[1], start=10, cos=None)
        db.transcript(texts[2], start=20, cos=0.9)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        await question(call, MEDIA)

        # The embedded chunk goes first, then c0, the earlier of the two unembedded ones.
        assert mentioned(recording.user_prompt, texts) == [texts[0], texts[2]]
        assert texts[1] not in recording.user_prompt

    async def test_an_unembedded_chunk_ranks_below_a_weakly_similar_embedding(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-053 I2: every embedded chunk outranks a chunk with no vector."""
        addon(budget=50)
        texts = [chunk("c0", 20), chunk("c1", 40)]
        db.file(MEDIA)
        db.transcript(texts[0], start=0, cos=None)
        db.transcript(texts[1], start=10, cos=0.05)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        await question(call, MEDIA)

        assert texts[1] in recording.user_prompt
        assert texts[0] not in recording.user_prompt


# ---------------------------------------------------------------------------
# Numbering and positions
# ---------------------------------------------------------------------------


class TestNumbering:
    async def test_numbering_runs_across_transcript_then_document_chunks(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-053 3: timestamps first, then chunk indices compared as integers."""
        transcript_a, transcript_b = chunk("t-a", 30), chunk("t-b", 30)
        document_two, document_ten = chunk("d-two", 30), chunk("d-ten", 30)
        db.file(MEDIA)
        db.transcript(transcript_b, start=20)
        db.transcript(transcript_a, start=10)
        db.document(document_ten, chunk_index=10)
        db.document(document_two, chunk_index=2)
        recording = llm.serve(RecordingLLM(deltas=("See [1][2][3][4].",)))

        events = await question(call, MEDIA)

        ordered = [transcript_a, transcript_b, document_two, document_ten]
        assert mentioned(recording.user_prompt, ordered) == ordered
        assert numbered_in_order(recording.user_prompt, ordered)
        assert [citation["n"] for citation in events["citations"]] == [1, 2, 3, 4]


class TestPositions:
    @pytest.mark.parametrize(
        ("kind", "position"),
        [
            ("video", {"seconds": 1705.0}),
            ("pdf", {"page": 12}),
            ("epub", {"section": 3, "title": "Chapter Three"}),
            ("epub without a title", {"section": 4, "title": None}),
            ("markdown", None),
        ],
        ids=["seconds", "page", "section-title", "section-untitled", "other-kind"],
    )
    async def test_a_citation_carries_the_position_of_its_chunk(
        self, kind, position, db, addon, llm, call,
    ):
        """SPEC-ADDON-053 3 and I3: a citation names the chunk's own position inside the file."""
        file_id = MEDIA
        if kind == "video":
            db.file(MEDIA).transcript(chunk("one", 30), start=1705.0, end=1790.0)
        elif kind == "pdf":
            file_id = PDF
            db.file(PDF, mime="application/pdf")
            db.document(chunk("one", 30), chunk_index=0, page="12")
        elif kind == "epub":
            file_id = EPUB
            db.file(EPUB, mime="application/epub+zip")
            db.document(chunk("one", 30), chunk_index=0, page="3").section(3, "Chapter Three")
        elif kind == "epub without a title":
            file_id = EPUB
            db.file(EPUB, mime="application/epub+zip")
            db.document(chunk("one", 30), chunk_index=0, page="4")
        else:
            file_id = NOTE
            db.file(NOTE, mime="text/markdown")
            db.document(chunk("one", 30), chunk_index=0)
        llm.serve(RecordingLLM(deltas=("Only [1].",)))

        events = await question(call, file_id)

        assert events["citations"] == [{"n": 1, "position": position}]


class TestThePromptLanguage:
    @pytest.mark.parametrize(
        ("language", "word"),
        [("en", "English"), ("ja", "Japanese")],
        ids=["en", "ja"],
    )
    async def test_the_answer_is_requested_in_the_configured_language(
        self, language, word, db, addon, llm, call,
    ):
        """SPEC-ADDON-053 6: the prompt asks for an answer in ``llm.output_language``."""
        addon(output_language=language)
        db.file(MEDIA).transcript(chunk("one", 30), start=0)
        recording = llm.serve(RecordingLLM(deltas=("answer",)), output_language=language)

        await question(call, MEDIA)

        assert word in recording.system_prompt + recording.user_prompt
