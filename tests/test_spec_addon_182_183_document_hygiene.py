"""SPEC-ADDON-182 I5 and SPEC-ADDON-183 I8: contents titles and the PDF's lifetime."""

from __future__ import annotations

import gc
import weakref

from tests.document_tool_doubles import (
    ANSWER,
    EPUB_BUDGET,
    EPUB_ID,
    PDF_BUDGET,
    PDF_ID,
    QUESTION,
    calls,
    get_page_images,
    plays,
    seed_epub,
    seed_pdf,
    user_prompt,
    wire_for,
)
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    Reply,
    addon,
    ask,
    call,
    db,
    llm,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture


async def test_a_section_title_carries_no_bracketed_number_or_line_break(
    db, addon, llm, call, tmp_path, monkeypatch,
):
    """SPEC-ADDON-182 I5: a contents line names a section with no bracketed number."""
    addon(budget=EPUB_BUDGET)
    seed_epub(db, tmp_path, monkeypatch, titles={3: "[1] Intro\nPart [22]"})
    wire = wire_for(llm, "openai_compatible", plays(then=Reply(text=ANSWER)))

    await ask(call, EPUB_ID, question=QUESTION)

    prompt = user_prompt(wire.requests[0])
    line = next(
        text for text in prompt.splitlines() if text.startswith("Section 3")
    )
    assert line.startswith("Section 3: Intro Part — ")


async def test_every_pymupdf_object_is_freed_under_the_lock_even_when_a_page_fails(
    db, addon, llm, call, tmp_path, monkeypatch,
):
    """SPEC-ADDON-183 I8: pages, pixmaps and the document are dropped while the lock is held."""
    import pymupdf

    from app.extractors.pdf import PYMUPDF_LOCK

    addon(budget=PDF_BUDGET)
    seed_pdf(db, tmp_path, monkeypatch)
    freed: list[tuple[str, bool]] = []
    render = pymupdf.Page.get_pixmap
    close = pymupdf.Document.close

    def _render(self, *args, **kwargs):
        weakref.finalize(
            self, lambda: freed.append(("page", PYMUPDF_LOCK.locked())),
        )
        if self.number == 1:
            raise RuntimeError("render failed")
        pixmap = render(self, *args, **kwargs)
        weakref.finalize(
            pixmap, lambda: freed.append(("pixmap", PYMUPDF_LOCK.locked())),
        )
        return pixmap

    def _close(self):
        freed.append(("document", PYMUPDF_LOCK.locked()))
        close(self)

    monkeypatch.setattr(pymupdf.Page, "get_pixmap", _render)
    monkeypatch.setattr(pymupdf.Document, "close", _close)
    wire = wire_for(
        llm, "openai_compatible",
        plays(calls(get_page_images(pages=[1, 2])), then=Reply(text=ANSWER)),
    )

    await ask(call, PDF_ID, question=QUESTION)
    gc.collect()

    assert len(wire.requests) == 2
    assert sorted(kind for kind, _ in freed) == [
        "document", "page", "page", "pixmap",
    ]
    assert all(locked for _, locked in freed), freed
