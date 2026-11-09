"""SPEC-ADDON-081: a file question carrying the earlier turns of its conversation.

The route is called through the addon's own app, as the core proxy calls it.
What reached the model is read from the prompt the routed LLM records, and the
ranking text from the query the embedding stand-in was handed.
"""

from __future__ import annotations

import logging
import string
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest
from sqlalchemy import event

import app.config as config
import app.database as database
from app.config import LLMConfig
from app.routers import rag as rag_router
from tests.file_question_doubles import (
    DRIVE,
    MEDIA,
    RecordingLLM,
    Seeder,
    chunk,
    frames,
    mentioned,
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
    return Seeder(search_index)


@pytest.fixture()
def addon(monkeypatch, make_settings):
    def _apply(**overrides):
        from app.main import app  # noqa: F401 - import every router before patching

        settings = settings_for_question(make_settings, **overrides)
        patch_everywhere(monkeypatch, "settings", settings)
        return settings

    _apply()
    return _apply


class Routing:
    def __init__(self):
        self.asked: list[tuple[str, str, str | None]] = []
        self.result = None

    def serve(self, client: RecordingLLM) -> RecordingLLM:
        self.result = resolved_with(
            client,
            config=LLMConfig(
                provider="openai_compatible", base_url="http://llm.test/v1",
                model="test-llm", output_language="auto",
            ),
        )
        return client


@pytest.fixture()
def llm(monkeypatch):
    from app import llm_routing

    routing = Routing()

    async def _resolve(drive, feature, requested=None):
        routing.asked.append((drive, feature, requested))
        return routing.result

    monkeypatch.setattr(llm_routing, "resolve", _resolve)
    return routing


@pytest.fixture()
async def call(search_index):  # noqa: F811
    from app.main import app

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://intelligence") as client:

        async def _call(body: dict, *, file_id: str = MEDIA):
            return await client.request(
                "POST", f"/files/{file_id}/ask", json=body,
                headers={"X-Lit-Drive": quote(DRIVE)},
            )

        yield _call


@pytest.fixture()
async def ask_slots(monkeypatch):
    monkeypatch.setattr(rag_router, "_ask_semaphore", None)
    monkeypatch.setattr(rag_router, "_ask_semaphore_loop", None)
    return rag_router._get_ask_semaphore()


@pytest.fixture()
def embedded(monkeypatch, db):
    """Every text the question embedding was asked for."""
    texts: list[str] = []

    def _embed_query(text: str):
        texts.append(text)
        return toward(db.db.dim, 1.0)

    patch_everywhere(monkeypatch, "embed_query", _embed_query)
    return texts


@pytest.fixture()
def statements(search_index):  # noqa: F811
    class Log:
        def __init__(self):
            self.statements: list[str] = []

        def _before(self, _conn, _cursor, statement, _params, _context, _many):
            self.statements.append(statement)

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


def text(tag: str, size: int) -> str:
    """``size`` characters starting with ``tag``, with no run that repeats a character.

    A cut at any length is then a prefix that no longer prefix can be confused with.
    """
    filler = (string.ascii_lowercase * (size // 26 + 2))
    return (tag + filler)[:size]


def turn(question: str, answer: str) -> dict:
    return {"question": question, "answer": answer}


def body(question: str = "and then what", history: list[dict] | None = None) -> dict:
    out: dict = {"question": question}
    if history is not None:
        out["history"] = history
    return out


def one_chunk_file(db: Seeder, *, size: int = 20) -> str:
    passage = chunk("passage", size)
    db.file(MEDIA).transcript(passage, start=0)
    return passage


async def answered(call, payload: dict) -> dict:
    response = await call(payload)
    assert response.status_code == 200, response.text
    return dict(frames(response.text))


# ---------------------------------------------------------------------------
# The body
# ---------------------------------------------------------------------------


class TestTheHistoryLimits:
    async def test_ten_turns_reach_the_model_and_eleven_are_422(self, db, addon, llm, call):
        """SPEC-ADDON-081 I3: at most 10 turns."""
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))
        ten = [turn(f"QUESTION-{i:02d}", f"ANSWER-{i:02d}") for i in range(10)]

        accepted = await call(body(history=ten))
        rejected = await call(body(history=ten + [turn("QUESTION-10", "ANSWER-10")]))

        assert accepted.status_code == 200, accepted.text
        assert all(t["question"] in recording.user_prompt for t in ten)
        assert rejected.status_code == 422
        assert len(recording.prompts) == 1

    @pytest.mark.parametrize(
        ("question", "status"),
        [("Q", 200), ("Q" + "x" * 999, 200), ("", 422), ("Q" + "x" * 1000, 422)],
        ids=["one-char", "exact-1000", "empty", "1001"],
    )
    async def test_a_turn_question_is_1_to_1000_characters(
        self, question, status, db, addon, llm, call,
    ):
        """SPEC-ADDON-081 I3: a turn's question is 1 to 1,000 characters."""
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        response = await call(body(history=[turn(question, "EARLIER-ANSWER")]))

        assert response.status_code == status, response.text
        if status == 200:
            assert "EARLIER-ANSWER" in recording.user_prompt
        else:
            assert recording.prompts == []

    @pytest.mark.parametrize(
        ("answer", "status"),
        [("", 200), ("A" * 20_000, 200), ("A" * 20_001, 422)],
        ids=["empty", "exact-20000", "20001"],
    )
    async def test_a_turn_answer_is_0_to_20000_characters(
        self, answer, status, db, addon, llm, call,
    ):
        """SPEC-ADDON-081 I3: a turn's answer is 0 to 20,000 characters."""
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        response = await call(body(history=[turn("EARLIER-QUESTION", answer)]))

        assert response.status_code == status, response.text
        if status == 200:
            assert "EARLIER-QUESTION" in recording.user_prompt
        else:
            assert recording.prompts == []

    async def test_an_unknown_key_in_a_turn_is_422(self, db, addon, llm, call):
        """SPEC-ADDON-081 2 and I3: a turn holds ``question`` and ``answer`` only."""
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        plain = await call(body(history=[turn("EARLIER-QUESTION", "earlier")]))
        extra = await call(body(history=[{**turn("EARLIER-QUESTION", "earlier"), "citations": []}]))

        assert plain.status_code == 200, plain.text
        assert "EARLIER-QUESTION" in recording.user_prompt
        assert extra.status_code == 422
        assert len(recording.prompts) == 1

    async def test_a_bad_history_is_422_before_the_rag_gate(self, db, addon, llm, call):
        """SPEC-ADDON-081 I3: the body is rejected before any gate, LLM or slot."""
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))
        addon(rag=False)

        response = await call(body(history=[turn("", "earlier")]))

        assert response.status_code == 422
        assert llm.asked == []
        assert recording.prompts == []

    async def test_a_bad_history_is_422_with_every_slot_taken(
        self, db, addon, llm, call, ask_slots, embedded,
    ):
        """SPEC-ADDON-081 I3: no semaphore slot, embedding or model is used first."""
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))
        for _ in range(rag_router._MAX_CONCURRENT_ASK):
            await ask_slots.acquire()
        try:
            response = await call(body(history=[turn("Q", "A" * 20_001)]))
        finally:
            for _ in range(rag_router._MAX_CONCURRENT_ASK):
                ask_slots.release()

        assert response.status_code == 422
        assert recording.prompts == []
        assert embedded == []


# ---------------------------------------------------------------------------
# What the model is given
# ---------------------------------------------------------------------------


class TestTheHistoryBlock:
    async def test_the_turns_come_oldest_first_before_the_question_and_the_passages(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-081 3 and 6: kept turns oldest first, then the new question, then passages."""
        passage = one_chunk_file(db, size=40)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))
        history = [
            turn("FIRST-QUESTION", "FIRST-ANSWER"),
            turn("SECOND-QUESTION", "SECOND-ANSWER"),
        ]

        await answered(call, body("THE-NEW-QUESTION", history))

        ordered = [
            "FIRST-QUESTION", "FIRST-ANSWER", "SECOND-QUESTION", "SECOND-ANSWER",
            "THE-NEW-QUESTION", passage,
        ]
        assert mentioned(recording.user_prompt, ordered) == ordered

    async def test_every_marker_is_removed_from_an_earlier_answer(self, db, addon, llm, call):
        """SPEC-ADDON-081 3 and I2: ``[n]`` of 1 to 3 digits goes, code and nesting included."""
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))
        answer = "alpha[1]beta[23]gamma `delta[456]` epsilon[1[2]]zeta[1000]eta"

        await answered(call, body(history=[turn("EARLIER-QUESTION", answer)]))

        assert "alphabetagamma `delta` epsilonzeta[1000]eta" in recording.user_prompt
        for leftover in ("alpha[", "gamma[", "delta[", "epsilon["):
            assert leftover not in recording.user_prompt

    async def test_a_turn_with_nothing_left_of_its_answer_is_still_sent(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-081 failure cases: an answer empty after removal keeps its question."""
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        await answered(call, body(history=[turn("EARLIER-QUESTION", "[1][2][3]")]))

        assert "EARLIER-QUESTION" in recording.user_prompt

    async def test_a_turns_size_is_counted_after_marker_removal(self, db, addon, llm, call):
        """SPEC-ADDON-081 3: markers do not use up the history budget."""
        addon(budget=200)
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))
        question = text("Q-", 20)
        words = text("A-", 80)
        # 150 characters as sent, 100 once its markers are gone: exactly the budget.
        answer = words[:40] + "[123]" * 10 + words[40:]

        await answered(call, body(history=[turn(question, answer)]))

        assert question in recording.user_prompt
        assert words in recording.user_prompt

    async def test_the_first_turn_that_does_not_fit_and_every_older_one_are_left_out(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-081 3 and I2: newest first while they fit; a kept turn is never older."""
        addon(budget=200)
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))
        oldest = turn(text("T0Q-", 5), text("T0A-", 5))  # 10, would fit on its own
        too_big = turn(text("T1Q-", 30), text("T1A-", 30))  # 60
        middle = turn(text("T2Q-", 15), text("T2A-", 15))  # 30
        newest = turn(text("T3Q-", 20), text("T3A-", 20))  # 40

        await answered(call, body(history=[oldest, too_big, middle, newest]))

        prompt = recording.user_prompt
        for kept in (middle, newest):
            assert kept["question"] in prompt and kept["answer"] in prompt
        for dropped in (oldest, too_big):
            assert dropped["question"] not in prompt
            assert dropped["answer"] not in prompt

    @pytest.mark.parametrize(
        ("older_size", "older_kept"),
        [(40, True), (41, False)],
        ids=["fills-exactly", "one-over"],
    )
    async def test_the_history_budget_is_half_the_limit_rounded_down(
        self, older_size, older_kept, db, addon, llm, call,
    ):
        """SPEC-ADDON-081 3 and I2: at 201 the history budget is 100."""
        addon(budget=201)
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))
        older = turn(text("OQ-", 20), text("OA-", older_size - 20))
        newest = turn(text("NQ-", 30), text("NA-", 30))

        await answered(call, body(history=[older, newest]))

        assert (older["question"] in recording.user_prompt) is older_kept
        assert newest["answer"] in recording.user_prompt

    async def test_a_newest_turn_too_big_alone_keeps_its_question_and_the_start_of_its_answer(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-081 3: the answer is cut to what the question leaves."""
        addon(budget=200)
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))
        older = turn(text("OQ-", 5), text("OA-", 5))
        question = text("NQ-", 30)
        answer = text("NA-", 150)

        await answered(call, body(history=[older, turn(question, answer)]))

        prompt = recording.user_prompt
        assert question in prompt
        assert answer[:70] in prompt
        assert answer[:71] not in prompt
        assert older["question"] not in prompt

    async def test_a_newest_question_over_the_budget_is_cut_and_its_answer_dropped(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-081 3: the question is cut to the history budget first."""
        addon(budget=200)
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))
        question = text("NQ-", 150)
        answer = "ZZANSWERZZ" + text("NA-", 20)

        await answered(call, body(history=[turn(question, answer)]))

        prompt = recording.user_prompt
        assert question[:100] in prompt
        assert question[:101] not in prompt
        assert "ZZANSWERZZ" not in prompt


class TestThePassageBudget:
    @pytest.mark.parametrize(
        ("question_size", "answer_size", "chunk_sizes", "partial", "sent"),
        [
            (20, 30, (50, 50, 50), False, 3),
            (20, 31, (50, 50, 50), True, 2),
            (30, 300, (50, 50), False, 2),
            (30, 300, (50, 51), True, 1),
        ],
        ids=["fits-exactly", "one-over", "cut-turn-fits-exactly", "cut-turn-one-over"],
    )
    async def test_the_passages_get_what_the_history_leaves(
        self, question_size, answer_size, chunk_sizes, partial, sent,
        db, addon, llm, call, embedded,
    ):
        """SPEC-ADDON-081 4 and I2: passages get the limit minus the kept turns, after cutting."""
        addon(budget=200)
        texts = [chunk(f"c{i}", size) for i, size in enumerate(chunk_sizes)]
        db.file(MEDIA)
        for offset, (passage, cos) in enumerate(zip(texts, (0.9, 0.5, 0.1))):
            db.transcript(passage, start=offset * 10, cos=cos)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))
        history = [turn(text("HQ-", question_size), text("HA-", answer_size))]

        events = await answered(call, body(history=history))

        assert history[0]["question"] in recording.user_prompt
        assert events["meta"]["partial"] is partial
        assert len(mentioned(recording.user_prompt, texts)) == sent

    async def test_a_file_answered_whole_can_be_answered_from_part_of_it_as_a_follow_up(
        self, db, addon, llm, call, embedded,
    ):
        """SPEC-ADDON-081 4: the same file, first question whole, follow-up partial."""
        addon(budget=200)
        texts = [chunk(f"c{i}", 60) for i in range(3)]
        db.file(MEDIA)
        for offset, (passage, cos) in enumerate(zip(texts, (0.9, 0.5, 0.1))):
            db.transcript(passage, start=offset * 10, cos=cos)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        first = await answered(call, body("what is this"))
        follow_up = await answered(
            call, body(history=[turn(text("HQ-", 20), text("HA-", 20))]),
        )

        assert first["meta"]["partial"] is False
        assert follow_up["meta"]["partial"] is True
        sent = mentioned(recording.user_prompt, texts)
        assert sum(len(t) for t in sent) <= 200 - 40


class TestTheRankingText:
    async def test_ranking_embeds_the_last_turns_question_as_sent_then_the_new_one(
        self, db, addon, llm, call, embedded,
    ):
        """SPEC-ADDON-081 5 and I1: the last entry's question, uncut, a newline, the question."""
        addon(budget=200)
        db.file(MEDIA)
        for offset in range(4):
            db.transcript(chunk(f"c{offset}", 60), start=offset * 10, cos=0.5)
        llm.serve(RecordingLLM(deltas=("answer",)))
        last_question = text("LAST-", 150)

        await answered(call, body(
            "and why",
            [turn("EARLIER-QUESTION", "earlier"), turn(last_question, "x")],
        ))

        assert embedded == [last_question + "\n" + "and why"]

    async def test_without_history_the_new_question_alone_is_embedded(
        self, db, addon, llm, call, embedded,
    ):
        """SPEC-ADDON-081 5 and I1: no history, or an empty one, ranks by the question alone."""
        addon(budget=200)
        db.file(MEDIA)
        for offset in range(4):
            db.transcript(chunk(f"c{offset}", 60), start=offset * 10, cos=0.5)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        await answered(call, body("and why"))
        await answered(call, body("and why", []))
        await answered(call, body("and why", [turn("EARLIER-QUESTION", "x")]))

        assert embedded == ["and why", "and why", "EARLIER-QUESTION\nand why"]
        assert "EARLIER-QUESTION" in recording.user_prompt


class TestNoHistory:
    async def test_an_empty_history_gets_the_prompts_of_no_history_and_a_turn_changes_both(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-081 1 and I1; 6: history adds to the system prompt and the user prompt."""
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer",)))

        await answered(call, body("what is this"))
        await answered(call, body("what is this", []))
        await answered(call, body("what is this", [turn("EARLIER-QUESTION", "earlier")]))

        none, empty, with_turn = recording.prompts
        assert empty == none
        assert with_turn[0] != none[0]
        assert "EARLIER-QUESTION" in with_turn[1]
        assert "EARLIER-QUESTION" not in none[1]


class TestPictures:
    async def test_a_picture_with_no_text_takes_history_without_citation_rules(
        self, db, call, make_settings, monkeypatch, tmp_path,
    ):
        """SPEC-ADDON-081 6 and 7: the image is attached again; no citation rules appear."""
        from tests.file_question_vision_doubles import (
            seed_picture,
            vision_client,
            vision_settings,
        )

        settings = vision_settings(
            make_settings, allowed_base_dirs=(str(tmp_path) + "/",),
        )
        patch_everywhere(monkeypatch, "settings", settings)
        picture = "pic000000001"
        base = Path(config.settings.intelligence_data_dir).parent / "pictures"
        seed_picture(db, picture, text=None, path=base / f"{picture}.png")

        class VisionRouting(Routing):
            def serve(self, resolved):
                self.result = resolved
                return resolved

        routing = VisionRouting()
        from app import llm_routing

        async def _resolve(drive, feature, requested=None):
            return routing.result

        monkeypatch.setattr(llm_routing, "resolve", _resolve)
        _, wire = vision_client(routing)

        first = await call(body("what is this"), file_id=picture)
        second = await call(
            body("and the colour", [turn("EARLIER-QUESTION", "earlier answer")]),
            file_id=picture,
        )

        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert dict(frames(second.text))["meta"]["image"] == "attached"
        assert len(wire.image_bodies) == 2
        assert len(wire.images) == 2
        with_history = wire.image_bodies[1]
        assert "EARLIER-QUESTION" in wire.text_of(with_history)
        assert wire.system_of(with_history) != wire.system_of(wire.image_bodies[0])
        assert "cite" not in wire.text_of(with_history).lower()
        assert "cite" not in wire.system_of(with_history).lower()


class TestTheStream:
    async def test_citations_name_only_this_requests_passages(self, db, addon, llm, call):
        """SPEC-ADDON-081 8: an earlier answer's numbers never become citations."""
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("From [1] and [2].",)))

        events = await answered(
            call, body(history=[turn("EARLIER-QUESTION", "see [2] and [3]")]),
        )

        assert "EARLIER-QUESTION" in recording.user_prompt
        assert [c["n"] for c in events["citations"]] == [1]

    async def test_a_failed_follow_up_never_logs_the_history(
        self, db, addon, llm, call, caplog,
    ):
        """SPEC-ADDON-081; SPEC-ADDON-053 I5 as revised: history text is never logged."""
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("partial", " and"), fail_after=1))
        caplog.set_level(logging.DEBUG)

        await call(body(history=[turn("PRIVATE-EARLIER-QUESTION", "PRIVATE-EARLIER-ANSWER")]))

        assert "PRIVATE-EARLIER-QUESTION" in recording.user_prompt
        for record in caplog.records:
            assert "PRIVATE-EARLIER-QUESTION" not in record.getMessage()
            assert "PRIVATE-EARLIER-ANSWER" not in record.getMessage()

    async def test_a_follow_up_writes_nothing(self, db, addon, llm, call, statements):
        """SPEC-ADDON-053 I5 as revised: the history is not stored."""
        one_chunk_file(db)
        recording = llm.serve(RecordingLLM(deltas=("answer [1]",)))
        statements.statements.clear()

        await answered(call, body(history=[turn("EARLIER-QUESTION", "earlier")]))

        assert "EARLIER-QUESTION" in recording.user_prompt
        assert statements.writes() == []
