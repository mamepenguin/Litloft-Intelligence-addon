"""SPEC-ADDON-182 I5 and SPEC-ADDON-183 I8: contents titles and the PDF's lifetime."""

from __future__ import annotations

import threading
import time

from tests.document_tool_doubles import (
    ANSWER,
    EPUB_BUDGET,
    EPUB_ID,
    PDF_BUDGET,
    PDF_ID,
    QUESTION,
    plays,
    seed_epub,
    seed_pdf,
    user_prompt,
    wire_for,
)
from tests.file_question_doubles import DRIVE
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


def test_a_document_still_open_when_the_budget_runs_out_is_closed_under_the_lock(
    db, addon, tmp_path, monkeypatch,
):
    """SPEC-ADDON-183 I8: closing the PDF waits for the lock rather than leaving it to the collector."""
    import pymupdf

    from app.extractors.pdf import PYMUPDF_LOCK
    from app.rag import file_question

    addon(budget=PDF_BUDGET)
    seed_pdf(db, tmp_path, monkeypatch)
    material = file_question.load_material(
        PDF_ID, DRIVE, QUESTION, budget=PDF_BUDGET,
    )
    closes: list[bool] = []
    closed = threading.Event()
    real_close = pymupdf.Document.close

    def _close(self):
        closes.append(PYMUPDF_LOCK.locked())
        real_close(self)
        closed.set()

    monkeypatch.setattr(pymupdf.Document, "close", _close)

    pages = file_question.PageImages(material, lock_seconds=0.2)
    pages.open()
    assert PYMUPDF_LOCK.acquire(timeout=1)
    try:
        time.sleep(0.3)
        pages.close()
        assert closes == []
    finally:
        PYMUPDF_LOCK.release()

    assert closed.wait(timeout=5)
    assert closes == [True]
