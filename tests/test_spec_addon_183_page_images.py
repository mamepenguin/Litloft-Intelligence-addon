"""SPEC-ADDON-183: get_page_images sends pictures of the requested PDF's pages.

The PDF on disk has 6 pages while its prompt says 5 (``pdf_markdown``), so a
page image is judged against the file's own count. Each page has its own
size, and an image names its page by its JPEG dimensions.
"""

from __future__ import annotations

import threading
import time

import pytest

from tests.document_tool_doubles import (
    ANSWER,
    GET_PAGES_BAD,
    PDF_BUDGET,
    PDF_ID,
    PDF_TEXT,
    PROVIDERS,
    QUESTION,
    as_json,
    calls,
    cited,
    get_page_images,
    get_text,
    images_follow_results,
    images_of,
    lines,
    offers_tool,
    page_images_head,
    plays,
    results,
    seed_pdf,
    wire_for,
)
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    Reply,
    ToolCall,
    addon,
    ask,
    call,
    db,
    events,
    llm,
    statements,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture

LONG_VALUE = "L" * 50


def _pdf(db, addon, tmp_path, monkeypatch, **kwargs):
    addon(budget=kwargs.pop("budget", PDF_BUDGET), **kwargs.pop("settings", {}))
    return seed_pdf(db, tmp_path, monkeypatch, **kwargs)


async def _after(llm, call, *replies, provider="openai_compatible", then=None):
    """Play ``replies``; return the wire and the request that follows the last of them."""
    wire = wire_for(llm, provider, plays(*replies, then=then or Reply(text=ANSWER)))
    pairs = events(await ask(call, PDF_ID, question=QUESTION))
    assert offers_tool(wire, wire.requests[0])
    assert len(wire.requests) == len(replies) + 1
    return wire, wire.requests[-1], pairs


class TestTakenPages:
    @pytest.mark.parametrize("provider", PROVIDERS)
    async def test_named_pages_follow_the_tool_results_labelled_and_numbered_on(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-183 §1.9.3 §1.9.4 §1.10 I7: JPEGs 1280 px on the long edge, ``[n] page p``."""
        _pdf(db, addon, tmp_path, monkeypatch)

        wire, body, pairs = await _after(
            llm, call, calls(get_page_images(pages=[1, 3])), provider=provider,
            then=Reply(text=f"{ANSWER} [5]."),
        )

        assert results(body) == [page_images_head(2)]
        assert images_of(wire, body) == [("[4] page 1", 1), ("[5] page 3", 3)]
        assert images_follow_results(wire, body)
        assert cited(pairs) == [(5, {"page": 3})]
        assert dict(pairs)["meta"] == {"partial": True, "image": "attached"}

    async def test_a_passage_stands_for_the_page_its_row_is_on(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-183 §1.9.2 I7: passages [1] and [3] are on pages 2 and 4."""
        _pdf(db, addon, tmp_path, monkeypatch)

        wire, body, _pairs = await _after(
            llm, call, calls(get_page_images(passages=[1, 3])),
        )

        assert results(body) == [page_images_head(2)]
        assert images_of(wire, body) == [("[4] page 2", 2), ("[5] page 4", 4)]

    async def test_a_requested_passage_names_its_page_too(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-183 §1.9.2: a requested passage is a given number."""
        _pdf(db, addon, tmp_path, monkeypatch)

        wire, body, _pairs = await _after(
            llm, call, calls(get_text(page=5)), calls(get_page_images(passages=[4])),
        )

        assert results(body)[1] == page_images_head(1)
        assert images_of(wire, body) == [("[5] page 5", 5)]

    async def test_each_value_is_decided_by_the_first_rule_that_applies(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-183 §1.9.2 §1.9.4: pages first, then passages; one note per value, in order."""
        _pdf(db, addon, tmp_path, monkeypatch)
        arguments = {
            "pages": [0, "x", True, 1.5, 7, 6, 2, 2, LONG_VALUE],
            "passages": [99, 2, 1],
        }

        wire, body, _pairs = await _after(
            llm, call, calls(ToolCall(arguments, name="get_page_images")),
        )

        (result,) = results(body)
        assert lines(result) == [
            page_images_head(2),
            "- page 0: past the end of the document",
            f"- {as_json('x')}: not a number I can read",
            "- true: not a number I can read",
            "- 1.5: not a number I can read",
            "- page 7: past the end of the document",
            "- page 2: named twice in this reply",
            f"- {as_json(LONG_VALUE)}: not a number I can read",
            "- [99]: not a passage you were given",
            "- [2]: its page is not known",
            "- page 2: named twice in this reply",
        ]
        assert images_of(wire, body) == [("[4] page 6", 6), ("[5] page 2", 2)]
        assert len(as_json(LONG_VALUE)) == 40

    async def test_a_fifth_page_of_one_call_is_over_the_limit(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-183 §1.9.2.7 I7: at most 4 page images per call."""
        _pdf(db, addon, tmp_path, monkeypatch)

        wire, body, _pairs = await _after(
            llm, call, calls(get_page_images(pages=[1, 2, 3, 4, 5, 6])),
        )

        assert lines(results(body)[0]) == [
            page_images_head(4),
            "- page 5: over the limit of 4 pages for one call",
            "- page 6: over the limit of 4 pages for one call",
        ]
        assert images_of(wire, body) == [
            ("[4] page 1", 1), ("[5] page 2", 2), ("[6] page 3", 3), ("[7] page 4", 4),
        ]

    async def test_a_page_already_sent_in_the_question_is_named_by_its_number(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-183 §1.9.2.5 I7: no page twice in one question."""
        _pdf(db, addon, tmp_path, monkeypatch)

        wire, body, _pairs = await _after(
            llm, call,
            calls(get_page_images(pages=[1])), calls(get_page_images(pages=[1, 2])),
        )

        assert results(body) == [
            page_images_head(1),
            "\n".join([page_images_head(1), "- page 1: already sent as [4]"]),
        ]
        assert images_of(wire, body) == [("[4] page 1", 1), ("[5] page 2", 2)]

    async def test_a_page_taken_by_an_earlier_call_of_the_reply_is_named_twice(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-183 §1.9.2.6: the same reply's calls share what they took."""
        _pdf(db, addon, tmp_path, monkeypatch)

        wire, body, _pairs = await _after(
            llm, call,
            calls(get_page_images(pages=[1]), get_page_images(pages=[1, 2])),
        )

        assert results(body) == [
            page_images_head(1),
            "\n".join([page_images_head(1), "- page 1: named twice in this reply"]),
        ]
        assert images_of(wire, body) == [("[4] page 1", 1), ("[5] page 2", 2)]

    async def test_requested_passages_take_numbers_before_page_images_of_the_same_reply(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-182 §1.10 I4: the page call comes first, its image still follows the passages."""
        _pdf(db, addon, tmp_path, monkeypatch)

        wire, body, pairs = await _after(
            llm, call,
            calls(get_page_images(pages=[1]), get_text(page=5)),
            then=Reply(text=f"{ANSWER} [5]. Also [4]."),
        )

        image_result, text_result = results(body)
        assert image_result == page_images_head(1)
        assert lines(text_result)[:3] == ["Page 5 begins.", "[4]", PDF_TEXT[8]]
        assert images_of(wire, body) == [("[5] page 1", 1)]
        assert sorted(cited(pairs)) == [(4, {"page": 5}), (5, {"page": 1})]
        assert dict(pairs)["meta"]["image"] == "attached"

    async def test_a_pdf_that_is_not_cut_answers_with_a_page_image(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-183 §1.2 §1.11: every PDF; ``partial`` stays the material's."""
        _pdf(db, addon, tmp_path, monkeypatch, budget=16000)

        wire, body, pairs = await _after(
            llm, call, calls(get_page_images(pages=[2])), then=Reply(text=f"{ANSWER} [10]."),
        )

        assert images_of(wire, body) == [("[10] page 2", 2)]
        assert cited(pairs) == [(10, {"page": 2})]
        assert dict(pairs)["meta"] == {"partial": False, "image": "attached"}

    async def test_a_relative_file_path_is_opened_under_the_drive_mount(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-183 §1.9.3: ``indexed_files.file_path`` joined to ``settings.drive_mounts[drive]``."""
        from tests.file_question_doubles import DRIVE

        addon(budget=PDF_BUDGET, drive_mounts={DRIVE: str(tmp_path)})
        seed_pdf(db, tmp_path, monkeypatch, relative_to=tmp_path)

        wire, body, _pairs = await _after(llm, call, calls(get_page_images(pages=[2])))

        assert images_of(wire, body) == [("[4] page 2", 2)]


class TestArguments:
    @pytest.mark.parametrize(
        "arguments",
        [{}, {"pages": 2}, {"passages": "1"}, {"pages": None}, [1], "not json"],
        ids=["empty", "pages-a-number", "passages-a-string", "only-null", "a-list", "not-json"],
    )
    async def test_arguments_without_a_list_get_the_usage_line(
        self, arguments, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-183 §1.9.1: an object holding pages, passages or both as lists."""
        _pdf(db, addon, tmp_path, monkeypatch)

        wire, body, _pairs = await _after(
            llm, call, calls(ToolCall(arguments, name="get_page_images")),
        )

        assert results(body) == [GET_PAGES_BAD]
        assert images_of(wire, body) == []

    async def test_a_null_key_is_absent_and_an_empty_list_takes_nothing(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-183 §1.9.1, SPEC-ADDON-182 Terms: ``null`` is absent; ``[]`` is a list."""
        _pdf(db, addon, tmp_path, monkeypatch)

        wire, body, _pairs = await _after(
            llm, call,
            calls(
                ToolCall({"pages": None, "passages": [1]}, name="get_page_images"),
                ToolCall({"pages": []}, name="get_page_images"),
            ),
        )

        assert results(body) == [page_images_head(1), page_images_head(0)]
        assert images_of(wire, body) == [("[4] page 2", 2)]


class TestPagesThatCannotBeRendered:
    async def test_a_pdf_that_cannot_be_opened_sends_no_image_and_no_number(
        self, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-183 §1.9.3 §2 §8 I7: ``not available``; past the end cannot be judged."""
        _pdf(db, addon, tmp_path, monkeypatch, file_pages=None)

        with caplog.at_level("DEBUG"):
            wire, body, pairs = await _after(
                llm, call,
                calls(ToolCall({"pages": [1, 99], "passages": [2]}, name="get_page_images")),
                then=Reply(text=f"{ANSWER} [4]."),
            )

        assert lines(results(body)[0]) == [
            page_images_head(0),
            "- page 1: not available",
            "- page 99: not available",
            "- [2]: its page is not known",
        ]
        assert images_of(wire, body) == []
        assert cited(pairs) == []
        assert dict(pairs)["meta"] == {"partial": True, "image": "unavailable"}
        failures = [
            line for line in caplog.text.splitlines() if "page_image_failed: " in line
        ]
        assert len(failures) == 2
        assert all(line.endswith(f"(file {PDF_ID})") for line in failures)

    async def test_a_page_that_fails_to_render_takes_no_number_and_the_others_close_up(
        self, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-183 §1.9.3 §1.10 §8 I7: the rendered pages are numbered without a gap."""
        import pymupdf

        _pdf(db, addon, tmp_path, monkeypatch)
        rendered = pymupdf.Page.get_pixmap

        def _render(self, *args, **kwargs):
            if self.number == 1:
                raise RuntimeError("render failed")
            return rendered(self, *args, **kwargs)

        monkeypatch.setattr(pymupdf.Page, "get_pixmap", _render)

        with caplog.at_level("DEBUG"):
            wire, body, pairs = await _after(
                llm, call, calls(get_page_images(pages=[1, 2, 3])),
                then=Reply(text=f"{ANSWER} [5]."),
            )

        assert lines(results(body)[0]) == [page_images_head(2), "- page 2: not available"]
        assert images_of(wire, body) == [("[4] page 1", 1), ("[5] page 3", 3)]
        assert cited(pairs) == [(5, {"page": 3})]
        assert dict(pairs)["meta"]["image"] == "attached"
        failures = [
            line for line in caplog.text.splitlines() if "page_image_failed: " in line
        ]
        assert len(failures) == 1 and failures[0].endswith(f"(file {PDF_ID})")

    async def test_every_pymupdf_call_for_a_page_image_holds_the_lock(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-183 I8: open and render run while ``PYMUPDF_LOCK`` is held."""
        import pymupdf

        from app.extractors.pdf import PYMUPDF_LOCK

        _pdf(db, addon, tmp_path, monkeypatch)
        held: list[tuple[str, bool]] = []
        opened = pymupdf.Document.__init__
        rendered = pymupdf.Page.get_pixmap

        def _open(self, *args, **kwargs):
            held.append(("open", PYMUPDF_LOCK.locked()))
            return opened(self, *args, **kwargs)

        def _render(self, *args, **kwargs):
            held.append(("render", PYMUPDF_LOCK.locked()))
            return rendered(self, *args, **kwargs)

        monkeypatch.setattr(pymupdf.Document, "__init__", _open)
        monkeypatch.setattr(pymupdf.Page, "get_pixmap", _render)

        wire, body, _pairs = await _after(llm, call, calls(get_page_images(pages=[1, 2])))

        assert images_of(wire, body) == [("[4] page 1", 1), ("[5] page 2", 2)]
        assert [kind for kind, _ in held].count("render") == 2
        assert "open" in [kind for kind, _ in held]
        assert all(locked for _, locked in held), held

    async def test_a_lock_held_past_thirty_seconds_makes_the_pages_not_available(
        self, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-183 §1.9.3 §2 §8 I7: the lock wait runs out at 30 s; the question goes on.

        Real time: the 30 s wait has no seam.
        """
        from app.extractors.pdf import PYMUPDF_LOCK

        _pdf(db, addon, tmp_path, monkeypatch)
        assert PYMUPDF_LOCK.acquire(timeout=1)
        release = threading.Timer(45.0, PYMUPDF_LOCK.release)
        release.start()
        started = time.monotonic()
        try:
            with caplog.at_level("DEBUG"):
                wire, body, pairs = await _after(
                    llm, call, calls(get_page_images(pages=[1, 2])),
                )
        finally:
            release.cancel()
            if PYMUPDF_LOCK.locked():
                PYMUPDF_LOCK.release()
        elapsed = time.monotonic() - started

        assert 29.0 <= elapsed < 44.0
        assert lines(results(body)[0]) == [
            page_images_head(0), "- page 1: not available", "- page 2: not available",
        ]
        assert images_of(wire, body) == []
        assert dict(pairs)["meta"]["image"] == "unavailable"
        assert caplog.text.count(f"page_image_failed: lock_timeout (file {PDF_ID})") == 2


class TestWhatThePathWrites:
    async def test_page_images_and_text_requests_write_nothing(
        self, db, addon, llm, call, tmp_path, monkeypatch, statements,
    ):
        """SPEC-ADDON-182 I9: nothing is written to any store."""
        _pdf(db, addon, tmp_path, monkeypatch)
        statements.reset()
        before = sorted(p.name for p in (tmp_path / "docs").rglob("*"))

        wire, body, _pairs = await _after(
            llm, call, calls(get_page_images(pages=[1]), get_text(page=2)),
        )

        assert images_of(wire, body) == [("[8] page 1", 1)]
        assert statements.writes() == []
        assert sorted(p.name for p in (tmp_path / "docs").rglob("*")) == before

