"""SPEC-ADDON-182/183: which questions take the document tool path, and what it asks.

Today's text request for the same question is read from a run with a profile
that cannot see, which keeps today's path.
"""

from __future__ import annotations

import re

import pytest

from tests.document_tool_doubles import (
    ANSWER,
    EPUB_BUDGET,
    EPUB_ID,
    EPUB_ROWS,
    EPUB_TITLES,
    GET_PAGE_IMAGES_DEFINITION,
    PDF_BUDGET,
    PDF_ID,
    PDF_MARKDOWN_PAGES,
    PDF_ROWS,
    PROVIDERS,
    QUESTION,
    ROW,
    WORD_ID,
    WORD_MIME,
    calls,
    epub_contents,
    expected_system_lines,
    get_text,
    is_final,
    offers_tool,
    pdf_contents,
    plays,
    rank_by_seeding,
    seed_epub,
    seed_pdf,
    seed_rows,
    system_of,
    tail_after_s181c,
    todays_body,
    tool_definitions,
    tool_names,
    user_prompt,
    wire_for,
)
from tests.file_context_doubles import assert_s179_and_s181
from tests.file_context_rules import block
from tests.file_question_doubles import chunk
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    TEXT_MODEL,
    VISION_MODEL,
    Reply,
    addon,
    ask,
    call,
    db,
    events,
    llm,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture

PDF_NOT_CUT_BUDGET = 16000
LIMITS = ("max_tokens", "max_completion_tokens")


def _seed(kind, db, addon, tmp_path, monkeypatch, *, cut: bool) -> str:
    if kind == "pdf":
        addon(budget=PDF_BUDGET if cut else PDF_NOT_CUT_BUDGET)
        seed_pdf(db, tmp_path, monkeypatch)
        return PDF_ID
    addon(budget=EPUB_BUDGET if cut else PDF_NOT_CUT_BUDGET)
    seed_epub(db, tmp_path, monkeypatch)
    return EPUB_ID


def _remainder(prompt: str, today_prompt: str) -> list[str]:
    """The lines the path adds after today's user prompt, with at least one empty line first."""
    head = today_prompt.rstrip("\n")
    assert prompt.startswith(head), prompt[:600]
    rest = prompt[len(head):]
    assert rest.startswith("\n\n"), repr(rest[:80])
    return [line for line in rest.split("\n") if line]


class TestWhichQuestionsTakeThePath:
    @pytest.mark.parametrize("provider", PROVIDERS)
    @pytest.mark.parametrize(
        ("kind", "cut", "offered"),
        [
            ("pdf", True, ["get_text", "get_page_images"]),
            ("pdf", False, ["get_page_images"]),
            ("epub", True, ["get_text"]),
        ],
        ids=["pdf-cut", "pdf-not-cut", "epub-cut"],
    )
    async def test_a_seeing_profile_is_offered_the_tools_in_order_by_the_vision_model(
        self, provider, kind, cut, offered, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 I1 §1.2 §1.6: the tools in order, on the vision model, with the tool limits."""
        file_id = _seed(kind, db, addon, tmp_path, monkeypatch, cut=cut)
        wire = wire_for(llm, provider, plays(then=Reply(text=ANSWER)))

        pairs = events(await ask(call, file_id, question=QUESTION))

        assert len(wire.requests) == 1
        body = wire.requests[0]
        assert offers_tool(wire, body)
        assert tool_names(body) == offered
        assert body["model"] == VISION_MODEL
        if provider == "ollama":
            assert body.get("think") is True
            assert body["options"]["num_predict"] == 8192
        else:
            assert [body[k] for k in LIMITS if k in body] == [16384]
        assert dict(pairs)["meta"]["partial"] is cut

    @pytest.mark.parametrize("provider", PROVIDERS)
    @pytest.mark.parametrize(
        "subject", ["blind-pdf", "blind-epub", "epub-not-cut", "word-cut"],
    )
    async def test_other_questions_send_todays_request_and_a_cut_pdf_gets_the_tools(
        self, subject, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 I1 §2: blind profiles, an uncut EPUB and other documents keep today's request."""
        if subject == "epub-not-cut":
            file_id = _seed("epub", db, addon, tmp_path, monkeypatch, cut=False)
        elif subject == "blind-epub":
            file_id = _seed("epub", db, addon, tmp_path, monkeypatch, cut=True)
        elif subject == "word-cut":
            addon(budget=PDF_BUDGET)
            path = tmp_path / "docs" / "w.docx"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"WORD")
            db.file(WORD_ID, mime=WORD_MIME, file_path=str(path), file_size=4)
            seed_rows(db, PDF_ROWS)
            rank_by_seeding(monkeypatch, db)
            file_id = WORD_ID
        else:
            file_id = _seed("pdf", db, addon, tmp_path, monkeypatch, cut=True)
        if subject == "blind-pdf":
            seed_epub(db, tmp_path, monkeypatch)
        blind = subject.startswith("blind")
        today = await todays_body(llm, call, provider, file_id)
        wire = wire_for(llm, provider, plays(then=Reply(text=ANSWER)), blind=blind)

        await ask(call, file_id, question=QUESTION)

        assert wire.requests == [today]
        assert "tools" not in today and today["model"] == TEXT_MODEL

        if subject == "blind-pdf":
            other = wire_for(llm, provider, plays(then=Reply(text=ANSWER)))
            addon(budget=EPUB_BUDGET)
            await ask(call, EPUB_ID, question=QUESTION)
            assert offers_tool(other, other.requests[0])
        else:
            other = wire_for(llm, provider, plays(then=Reply(text=ANSWER)))
            seed_pdf(db, tmp_path, monkeypatch)
            addon(budget=PDF_BUDGET)
            await ask(call, PDF_ID, question=QUESTION)
            assert offers_tool(other, other.requests[0])


class TestTheToolDefinitions:
    @pytest.mark.parametrize(
        ("kind", "unit"), [("pdf", "page"), ("epub", "section")], ids=["pdf", "epub"],
    )
    async def test_get_text_takes_a_unit_or_after_neither_required(
        self, kind, unit, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.5: the definition of get_text for the file's kind."""
        from tests.document_tool_doubles import get_text_definition

        file_id = _seed(kind, db, addon, tmp_path, monkeypatch, cut=True)
        wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

        await ask(call, file_id, question=QUESTION)

        definition = tool_definitions(wire.requests[0])["get_text"]
        expected = get_text_definition(unit)
        assert definition["description"] == expected["description"]
        parameters = definition["parameters"]
        assert parameters["type"] == "object"
        assert parameters["properties"] == expected["properties"]
        assert not parameters.get("required")

    async def test_get_page_images_takes_pages_and_passages_neither_required(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-183 §1.5: the definition of get_page_images."""
        _seed("pdf", db, addon, tmp_path, monkeypatch, cut=False)
        wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

        await ask(call, PDF_ID, question=QUESTION)

        definition = tool_definitions(wire.requests[0])["get_page_images"]
        assert definition["description"] == GET_PAGE_IMAGES_DEFINITION["description"]
        parameters = definition["parameters"]
        assert parameters["type"] == "object"
        assert parameters["properties"] == GET_PAGE_IMAGES_DEFINITION["properties"]
        assert not parameters.get("required")


class TestTheSystemPrompt:
    @pytest.mark.parametrize("history", [False, True], ids=["no-history", "history"])
    @pytest.mark.parametrize(
        ("kind", "cut", "text", "pages", "unit"),
        [
            ("pdf", True, True, True, "page"),
            ("pdf", False, False, True, "page"),
            ("epub", True, True, False, "section"),
        ],
        ids=["pdf-cut", "pdf-not-cut", "epub-cut"],
    )
    async def test_it_reads_line_for_line_as_the_spec_words_it(
        self, kind, cut, text, pages, unit, history, db, addon, llm, call, tmp_path,
        monkeypatch,
    ):
        """SPEC-ADDON-182 §1.4, SPEC-ADDON-181 I6 (revised): every line, in order."""
        file_id = _seed(kind, db, addon, tmp_path, monkeypatch, cut=cut)
        body = {"question": QUESTION}
        if history:
            body["history"] = [{"question": "EARLIER-QUESTION", "answer": "EARLIER-ANSWER"}]
        blind = wire_for(llm, "openai_compatible", plays(), blind=True)
        await call("POST", f"/files/{file_id}/ask", json=body)
        tail = tail_after_s181c(system_of(blind.requests[0]))
        wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

        await call("POST", f"/files/{file_id}/ask", json=body)

        system = system_of(wire.requests[0])
        all_lines = system.split("\n")
        assert [line for line in all_lines if line] == expected_system_lines(
            cut=cut, text=text, pages=pages, unit=unit, tail=tail, history=history,
        )
        assert all_lines[all_lines.index("Rules:") - 1] == ""
        assert all_lines[all_lines.index("More of the file:") - 1] == ""

    async def test_the_language_line_is_the_one_every_path_carries(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.4: the profile's language line, as on every path, before the tool section."""
        _seed("pdf", db, addon, tmp_path, monkeypatch, cut=True)
        today = await todays_body(llm, call, "openai_compatible", PDF_ID, output_language="ja")
        tail = tail_after_s181c(system_of(today))
        wire = wire_for(
            llm, "openai_compatible", plays(then=Reply(text=ANSWER)), output_language="ja",
        )

        await ask(call, PDF_ID, question=QUESTION)

        assert tail[-1] == "- Answer in Japanese"
        system = system_of(wire.requests[0])
        assert [line for line in system.split("\n") if line] == expected_system_lines(
            cut=True, text=True, pages=True, unit="page", tail=tail,
        )

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_every_request_of_the_path_has_the_file_block_and_the_rules_in_place(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-179 I1 I4, SPEC-ADDON-181 I6 on the document tool path, final request included."""
        _seed("pdf", db, addon, tmp_path, monkeypatch, cut=True)
        wire = wire_for(llm, provider, plays(
            calls(get_text(page=1)), calls(get_text(page=3)), calls(get_text(page=5)),
        ))

        await ask(call, PDF_ID, question=QUESTION)

        assert len(wire.requests) == 4
        assert offers_tool(wire, wire.requests[0])
        assert is_final(wire, wire.requests[3])
        expected_block = block(file=f"{PDF_ID}.bin", type_="application/pdf")
        for body in wire.requests:
            assert_s179_and_s181(system_of(body))
            text = user_prompt(body)
            assert text.startswith(expected_block + "Question: "), text[:300]


class TestTheUserPrompt:
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_a_cut_pdf_adds_its_page_count_and_contents_after_todays_prompt(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.3: today's prompt, an empty line, the page count, then one line per page."""
        _seed("pdf", db, addon, tmp_path, monkeypatch, cut=True)
        today = await todays_body(llm, call, provider, PDF_ID)
        wire = wire_for(llm, provider, plays(then=Reply(text=ANSWER)))

        await ask(call, PDF_ID, question=QUESTION)

        assert offers_tool(wire, wire.requests[0])
        added = _remainder(user_prompt(wire.requests[0]), user_prompt(today))
        assert added == [
            f"The document has {PDF_MARKDOWN_PAGES} pages.",
            "Contents, each with the start of what it says:",
            *pdf_contents(PDF_MARKDOWN_PAGES, PDF_ROWS),
        ]

    async def test_a_pdf_that_is_not_cut_adds_only_its_page_count(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.3: no contents when the material is whole."""
        _seed("pdf", db, addon, tmp_path, monkeypatch, cut=False)
        today = await todays_body(llm, call, "openai_compatible", PDF_ID)
        wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

        await ask(call, PDF_ID, question=QUESTION)

        added = _remainder(user_prompt(wire.requests[0]), user_prompt(today))
        assert added == [f"The document has {PDF_MARKDOWN_PAGES} pages."]

    async def test_a_cut_epub_adds_its_section_range_and_titled_contents(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.3: sections the rows have; the first section's title cut to 60."""
        _seed("epub", db, addon, tmp_path, monkeypatch, cut=True)
        today = await todays_body(llm, call, "openai_compatible", EPUB_ID)
        wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

        await ask(call, EPUB_ID, question=QUESTION)

        added = _remainder(user_prompt(wire.requests[0]), user_prompt(today))
        contents = epub_contents(EPUB_ROWS, EPUB_TITLES)
        assert added == [
            "The book has sections 3 to 6.",
            "Contents, each with the start of what it says:",
            *contents,
        ]
        assert contents[1].startswith(f"Section 4: {EPUB_TITLES[4][:60]} — ")
        assert contents[2].startswith("Section 5: EPUB-ROW-3")

    @pytest.mark.parametrize(
        ("markdown_pages", "largest_unit", "expected"),
        [(None, 4, 4), (2, 4, 4), (9, 4, 9)],
        ids=["no-markdown-row", "markdown-below-units", "markdown-above-units"],
    )
    async def test_the_page_count_is_the_larger_of_the_markdown_count_and_the_units(
        self, markdown_pages, largest_unit, expected, db, addon, llm, call, tmp_path,
        monkeypatch,
    ):
        """SPEC-ADDON-182 Terms (page count): a page with no row is the range alone."""
        rows = [
            (chunk(f"COUNT-ROW-{i}", ROW), str(i + 1), i == 0)
            for i in range(largest_unit)
        ]
        addon(budget=ROW + 50)
        seed_pdf(db, tmp_path, monkeypatch, rows, markdown_pages=markdown_pages)
        wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

        await ask(call, PDF_ID, question=QUESTION)

        prompt = user_prompt(wire.requests[0])
        assert f"The document has {expected} pages." in prompt.split("\n")
        assert prompt.rstrip("\n").split("\n")[-(expected):] == pdf_contents(expected, rows)
        if expected > largest_unit:
            assert f"Page {expected}" in prompt.split("\n")

    @pytest.mark.parametrize(
        ("pages", "group", "lines"),
        [(40, 1, 40), (41, 2, 21), (85, 3, 29), (120, 3, 40), (121, 4, 31)],
    )
    async def test_contents_group_pages_so_there_are_at_most_forty_lines(
        self, pages, group, lines, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.3 I5: groups of ceil(pages / 40), each ``Pages p–q`` or ``Page p``."""
        rows = [(chunk("GROUP-ROW-SENT", ROW), "1", True), (chunk("GROUP-ROW-B", ROW), "2", False)]
        addon(budget=ROW + 50)
        seed_pdf(db, tmp_path, monkeypatch, rows, markdown_pages=pages)
        wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

        await ask(call, PDF_ID, question=QUESTION)

        prompt = user_prompt(wire.requests[0]).rstrip("\n").split("\n")
        start = prompt.index("Contents, each with the start of what it says:") + 1
        contents = prompt[start:]
        assert contents == pdf_contents(pages, rows)
        assert len(contents) == lines
        first = f"Pages 1–{group}" if group > 1 else "Page 1"
        assert contents[0].startswith(f"{first}: GROUP-ROW-SENT")

    async def test_an_excerpt_drops_passage_numbers_collapses_whitespace_and_skips_rows_with_no_unit(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.3 I5: no ``[n]`` beside a page, at most 120 characters."""
        rows = [
            ("  Alpha [12]  beta\n\tgamma [1234]  ", "1", True),
            ("NOUNIT-MARKER row", "", False),
            ("Delta   [7]   epsilon " + "z" * 200, "1", False),
            ("Second page", "2", False),
        ]
        addon(budget=60)
        seed_pdf(db, tmp_path, monkeypatch, rows, markdown_pages=2)
        wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

        await ask(call, PDF_ID, question=QUESTION)

        prompt = user_prompt(wire.requests[0]).rstrip("\n").split("\n")
        start = prompt.index("Contents, each with the start of what it says:") + 1
        first = "Alpha beta gamma [1234] Delta epsilon " + "z" * 200
        assert prompt[start:] == [f"Page 1: {first[:120].rstrip()}", "Page 2: Second page"]
        assert len(prompt[start]) - len("Page 1: ") <= 120
        added = prompt[prompt.index("The document has 2 pages."):]
        assert not any("NOUNIT-MARKER" in line for line in added)
        unit_lines = [
            line for line in prompt if re.match(r"^(Page|Pages|Section|Sections) \d", line)
        ]
        assert unit_lines
        assert not any(re.search(r"\[\d{1,3}\]", line) for line in unit_lines)
