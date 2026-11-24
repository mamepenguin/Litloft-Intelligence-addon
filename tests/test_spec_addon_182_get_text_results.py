"""SPEC-ADDON-182: what a get_text result sends, in what order and shape.

The PDF fixture sends rows 3, 4 and 7 as passages ``[1]``–``[3]``; row 4 has
no unit. The EPUB fixture sends rows 1 and 4 as ``[1]`` and ``[2]``.
"""

from __future__ import annotations

import pytest

from tests.document_tool_doubles import (
    ANSWER,
    EPUB_BUDGET,
    EPUB_ID,
    EPUB_TEXT,
    EPUB_TITLES,
    FILE_ENDS,
    GET_TEXT_BAD_EPUB,
    GET_TEXT_BAD_PDF,
    LIMIT,
    PDF_BUDGET,
    PDF_ID,
    PDF_ROWS,
    PDF_TEXT,
    PROVIDERS,
    QUESTION,
    calls,
    cited,
    continues,
    get_page_images,
    get_text,
    lines,
    offers_tool,
    plays,
    requested,
    results,
    seed_epub,
    seed_pdf,
    wire_for,
)
from tests.file_question_doubles import chunk
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    Reply,
    ToolCall,
    addon,
    ask,
    call,
    db,
    events,
    llm,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture

R = PDF_TEXT
E = EPUB_TEXT


def _pdf(db, addon, tmp_path, monkeypatch, rows=PDF_ROWS, **kwargs):
    addon(budget=kwargs.pop("budget", PDF_BUDGET))
    seed_pdf(db, tmp_path, monkeypatch, rows, **kwargs)


def _epub(db, addon, tmp_path, monkeypatch):
    addon(budget=EPUB_BUDGET)
    seed_epub(db, tmp_path, monkeypatch)


async def _first_result(llm, call, *tool_calls, file_id=PDF_ID, provider="openai_compatible"):
    wire = wire_for(llm, provider, plays(calls(*tool_calls)))
    await ask(call, file_id, question=QUESTION)
    assert offers_tool(wire, wire.requests[0])
    assert len(wire.requests) >= 2
    return wire, results(wire.requests[1])


class TestTheWalk:
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_from_a_page_rows_come_in_file_order_with_markers_and_sent_rows_named(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.2 §1.8.3 §1.8.5 I3 I5: numbered on from [4]; a row with no unit has no marker."""
        _pdf(db, addon, tmp_path, monkeypatch)

        _wire, (result,) = await _first_result(llm, call, get_text(page=2), provider=provider)

        assert lines(result) == [
            "Page 2 begins.", *requested(4, R[2]),
            "already sent as [1]",
            "already sent as [2]",
            "Page 3 begins.", *requested(5, R[5]),
            "Page 4 begins.", *requested(6, R[6]),
            "already sent as [3]",
            "Page 5 begins.", "[7]", R[8],
            FILE_ENDS,
        ]

    async def test_from_the_first_page_the_walk_starts_at_the_files_first_row(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.2 §1.8.3: the file's first row is the first row of its unit."""
        _pdf(db, addon, tmp_path, monkeypatch)

        _wire, (result,) = await _first_result(llm, call, get_text(page=1))

        assert lines(result)[:7] == [
            "Page 1 begins.", *requested(4, R[0]), *requested(5, R[1]),
        ]
        assert lines(result)[-1] == FILE_ENDS

    async def test_after_a_passage_the_walk_starts_at_the_next_row(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.2 §1.8.3: a first row with no unit gets no marker and no Continuing line."""
        _pdf(db, addon, tmp_path, monkeypatch)

        _wire, (result,) = await _first_result(llm, call, get_text(after=1))

        assert lines(result) == [
            "already sent as [2]",
            "Page 3 begins.", *requested(4, R[5]),
            "Page 4 begins.", *requested(5, R[6]),
            "already sent as [3]",
            "Page 5 begins.", "[6]", R[8],
            FILE_ENDS,
        ]

    async def test_a_page_with_no_row_starts_the_walk_at_the_next_page_that_has_one(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.2: the first row whose unit is at least the named one."""
        rows = [
            (chunk("GAP-ROW-0", 100), "1", True),
            (chunk("GAP-ROW-1", 100), "", False),
            (chunk("GAP-ROW-2", 100), "4", False),
        ]
        _pdf(db, addon, tmp_path, monkeypatch, rows, budget=150, markdown_pages=4)

        _wire, (result,) = await _first_result(llm, call, get_text(page=2))

        assert lines(result) == ["Page 4 begins.", "[2]", rows[2][0], FILE_ENDS]

    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_a_walk_that_starts_inside_a_section_says_it_continues_it(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.3: ``Continuing section <u>.`` when the first row is not its unit's first."""
        _epub(db, addon, tmp_path, monkeypatch)

        _wire, (result,) = await _first_result(
            llm, call, get_text(after=2), file_id=EPUB_ID, provider=provider,
        )

        assert lines(result) == ["Continuing section 6.", "[3]", E[5], FILE_ENDS]

    async def test_an_epub_section_walk_marks_each_section(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.3 I5: ``Section <u> begins.``; a mid-section row has no marker."""
        _epub(db, addon, tmp_path, monkeypatch)

        _wire, (result,) = await _first_result(
            llm, call, get_text(section=4), file_id=EPUB_ID,
        )

        assert lines(result) == [
            "Section 4 begins.", *requested(3, E[2]),
            "Section 5 begins.", *requested(4, E[3]),
            "Section 6 begins.", "already sent as [2]",
            "[5]", E[5],
            FILE_ENDS,
        ]

    async def test_a_requested_passage_is_a_sent_row_for_the_rest_of_the_question(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.3 I3: a later walk names it; ``after`` takes a requested passage's number."""
        _pdf(db, addon, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(
            calls(get_text(page=1)), calls(get_text(after=4)),
        ))

        await ask(call, PDF_ID, question=QUESTION)

        assert offers_tool(wire, wire.requests[0])
        assert len(wire.requests) == 3
        second = results(wire.requests[2])[1]
        assert lines(second) == [
            "Continuing page 1.", "already sent as [5]",
            "Page 2 begins.", "already sent as [6]", "already sent as [1]",
            "already sent as [2]",
            "Page 3 begins.", "already sent as [7]",
            "Page 4 begins.", "already sent as [8]", "already sent as [3]",
            "Page 5 begins.", "already sent as [9]",
            FILE_ENDS,
        ]
        assert all(text not in second for text in R)


class TestWhereNothingFollows:
    async def test_a_page_past_every_row_but_within_the_count_has_nothing_written(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.2: ``Nothing is written from page <u> on.``"""
        _pdf(db, addon, tmp_path, monkeypatch, markdown_pages=7)

        _wire, (result,) = await _first_result(llm, call, get_text(page=6))

        assert result == "Nothing is written from page 6 on."

    async def test_after_the_last_row_the_file_ends_after_that_number(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.2: ``The file ends after [<n>].``"""
        _pdf(db, addon, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(
            calls(get_text(page=5)), calls(get_text(after=4)),
        ))

        await ask(call, PDF_ID, question=QUESTION)

        assert len(wire.requests) == 3
        first, second = results(wire.requests[2])
        assert lines(first) == ["Page 5 begins.", "[4]", R[8], FILE_ENDS]
        assert second == "The file ends after [4]."


class TestArguments:
    @pytest.mark.parametrize(
        "arguments",
        [
            {"page": "2"},
            {"page": " 2 "},
            {"page": 2.0},
            {"page": 2, "after": None},
            {"page": 2, "unknown": "x"},
        ],
        ids=["digit-string", "padded-string", "whole-float", "null-after", "unknown-key"],
    )
    async def test_an_integer_argument_may_be_written_several_ways(
        self, arguments, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 Terms (integer argument): each reads as page 2."""
        _pdf(db, addon, tmp_path, monkeypatch)

        _wire, (result,) = await _first_result(llm, call, ToolCall(arguments, name="get_text"))

        assert lines(result)[:3] == ["Page 2 begins.", "[4]", R[2]]

    @pytest.mark.parametrize(
        "arguments",
        [
            {},
            {"page": 2, "after": 1},
            {"page": 0},
            {"page": 6},
            {"after": 4},
            {"after": 0},
            {"page": "abc"},
            {"page": True},
            {"page": 2.5},
            {"page": "2.0"},
            {"section": 2},
            [2],
            "not json",
        ],
        ids=[
            "empty", "both", "page-0", "past-the-count", "after-not-given", "after-0",
            "not-a-number", "boolean", "fraction", "float-string", "epub-key", "a-list",
            "not-json",
        ],
    )
    async def test_bad_arguments_get_the_pdf_usage_line(
        self, arguments, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.1: exactly one key, an integer argument, in range or a given number."""
        _pdf(db, addon, tmp_path, monkeypatch)

        _wire, (result,) = await _first_result(llm, call, ToolCall(arguments, name="get_text"))

        assert result == GET_TEXT_BAD_PDF.format(n=5)

    @pytest.mark.parametrize(
        "arguments",
        [{"section": 2}, {"section": 7}, {"page": 4}, {"after": 3}],
        ids=["below-the-range", "above-the-range", "pdf-key", "after-not-given"],
    )
    async def test_bad_arguments_get_the_epub_usage_line(
        self, arguments, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.1: the section range runs from the smallest to the largest unit."""
        _epub(db, addon, tmp_path, monkeypatch)

        _wire, (result,) = await _first_result(
            llm, call, ToolCall(arguments, name="get_text"), file_id=EPUB_ID,
        )

        assert result == GET_TEXT_BAD_EPUB.format(a=3, b=6)

    async def test_a_page_image_number_is_not_a_given_number(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 Terms (given number): ``after`` a page image is a bad argument."""
        _pdf(db, addon, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(
            calls(get_page_images(pages=[1])), calls(get_text(after=4)),
        ))

        await ask(call, PDF_ID, question=QUESTION)

        assert len(wire.requests) == 3
        assert results(wire.requests[2])[1] == GET_TEXT_BAD_PDF.format(n=5)

    async def test_the_page_range_comes_from_the_page_count(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 Terms (page count) §1.8.1: ``pdf_markdown.page_count`` above the units widens it."""
        _pdf(db, addon, tmp_path, monkeypatch, markdown_pages=9)

        _wire, (result,) = await _first_result(llm, call, get_text(page=10))

        assert result == GET_TEXT_BAD_PDF.format(n=9)


class TestTheLimit:
    @staticmethod
    def _long(db, addon, tmp_path, monkeypatch):
        rows = [
            (chunk(f"LONG-{i}", 1000), str(i // 2 + 1), i == 5) for i in range(6)
        ]
        _pdf(db, addon, tmp_path, monkeypatch, rows, budget=1050, markdown_pages=3)
        return [text for text, _page, _sent in rows]

    async def test_a_result_stops_before_a_row_that_does_not_fit_and_after_continues_it(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.4 I3: whole rows, at most 3,000 characters, the note names the last number."""
        rows = self._long(db, addon, tmp_path, monkeypatch)
        wire = wire_for(llm, "openai_compatible", plays(
            calls(get_text(page=1)), calls(get_text(after=3)), calls(get_text(after=5)),
        ))

        await ask(call, PDF_ID, question=QUESTION)

        assert len(wire.requests) == 4
        first, second, third = results(wire.requests[3])
        assert lines(first) == [
            "Page 1 begins.", *requested(2, rows[0]), "[3]", rows[1], continues(3),
        ]
        assert lines(second) == [
            "Page 2 begins.", *requested(4, rows[2]), "[5]", rows[3], continues(5),
        ]
        assert lines(third) == [
            "Page 3 begins.", *requested(6, rows[4]), "already sent as [1]", FILE_ENDS,
        ]
        for result in (first, second, third):
            assert len(result) <= LIMIT

    @pytest.mark.parametrize(
        ("second_row", "fits"), [(1909, True), (1922, False)], ids=["fits", "does-not-fit"],
    )
    async def test_the_reserve_for_the_note_is_counted_before_a_row_is_written(
        self, second_row, fits, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.4 §1.8.6: the result so far, the row's lines and the reserve within 3,000."""
        rows = [
            (chunk("EDGE-0", 1000), "1", False),
            (chunk("EDGE-1", second_row), "1", False),
            (chunk("EDGE-SENT", 50), "1", True),
        ]
        _pdf(db, addon, tmp_path, monkeypatch, rows, budget=100, markdown_pages=1)

        _wire, (result,) = await _first_result(llm, call, get_text(page=1))

        if fits:
            assert lines(result) == [
                "Page 1 begins.", *requested(2, rows[0][0]), "[3]", rows[1][0],
                continues(3),
            ]
        else:
            assert lines(result) == [
                "Page 1 begins.", "[2]", rows[0][0], continues(2),
            ]
        assert len(result) <= LIMIT

    async def test_a_first_row_longer_than_the_limit_is_written_whole_and_nothing_after_it(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.8.4 I3: the walk's first row is always written, with its marker."""
        rows = [
            (chunk("HUGE-0", 3500), "1", False),
            (chunk("HUGE-1", 100), "1", False),
            (chunk("HUGE-SENT", 50), "2", True),
        ]
        _pdf(db, addon, tmp_path, monkeypatch, rows, budget=100, markdown_pages=2)

        _wire, (result,) = await _first_result(llm, call, get_text(page=1))

        assert lines(result) == ["Page 1 begins.", "[2]", rows[0][0], continues(2)]


class TestCitationsOfRequestedPassages:
    async def test_a_pdf_answer_cites_a_requested_passage_with_its_page(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.11 I4: a requested passage maps to its row's page; an unsent number is dropped."""
        _pdf(db, addon, tmp_path, monkeypatch)
        wire_for(llm, "openai_compatible", plays(
            calls(get_text(page=5)), then=Reply(text=f"{ANSWER} [4]. More [1]. None [9]."),
        ))

        pairs = events(await ask(call, PDF_ID, question=QUESTION))

        assert sorted(cited(pairs)) == [(1, {"page": 2}), (4, {"page": 5})]
        assert dict(pairs)["meta"] == {"partial": True, "image": "none"}

    async def test_an_epub_answer_cites_a_requested_passage_with_its_section_and_title(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.11 I4: ``{"section": s, "title": t}``, the title None when there is none."""
        _epub(db, addon, tmp_path, monkeypatch)
        wire_for(llm, "openai_compatible", plays(
            calls(get_text(section=4)), then=Reply(text=f"{ANSWER} [3]. Then [4]."),
        ))

        pairs = events(await ask(call, EPUB_ID, question=QUESTION))

        assert sorted(cited(pairs)) == [
            (3, {"section": 4, "title": EPUB_TITLES[4]}),
            (4, {"section": 5, "title": None}),
        ]
        assert dict(pairs)["meta"] == {"partial": True, "image": "none"}
