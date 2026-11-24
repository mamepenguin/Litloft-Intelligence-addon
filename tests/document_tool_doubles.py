"""Seeding, scripts and readers for file questions on the document tool path.

The provider is the addon's real client over ``httpx.MockTransport``
(``tests.file_question_vision_doubles``). A PDF is a real file written with
PyMuPDF whose page sizes differ, so a page image names its page by its JPEG
dimensions: Pillow is a MagicMock in this suite and cannot decode it.
"""

from __future__ import annotations

import base64
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

import app.database as database
from app.models import PdfMarkdown
from tests.file_context_rules import S179, S181A, S181B, S181C
from tests.file_question_doubles import chunk, toward
from tests.file_question_vision_doubles import (
    Reply,
    ToolCall,
    Wire,
    ollama_config,
    openai_config,
    vision_client,
)
from tests.frame_label_layout import tool_messages
from tests.search_endpoint_db import patch_everywhere

PDF_ID = "pdf000000182"
EPUB_ID = "epub00000182"
WORD_ID = "doc000000182"
QUESTION = "QUESTION-MARKER what does it say"
ANSWER = "FINAL-ANSWER-MARKER"
TODAY_ANSWER = "TODAYS-ANSWER-MARKER"
PROVIDERS = ["openai_compatible", "ollama"]
FAILED = {"error": "Answer generation failed"}
PDF_MIME = "application/pdf"
EPUB_MIME = "application/epub+zip"
WORD_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

#: The seams the 120 s tool window, the 300 s deadline and the keep-alive
#: interval are read through, shared with the frames path.
WINDOW_SECONDS = "_FRAME_TOOL_WINDOW_SECONDS"
DEADLINE_SECONDS = "_FRAME_TOOL_DEADLINE_SECONDS"
KEEPALIVE_SECONDS = "_KEEPALIVE_SECONDS"

HIGH = 1.0
LOW = 0.3
ROW = 100

# ---------------------------------------------------------------------------
# The PDF fixture
# ---------------------------------------------------------------------------

#: ``(text, stored page, sent)``. Units 1 to 5; the fifth row has no unit.
#: With a budget of ``PDF_BUDGET`` the three ``sent`` rows are passages
#: ``[1]``–``[3]`` in file order.
PDF_ROWS = [
    (chunk("PDF-ROW-0", ROW), "1", False),
    (chunk("PDF-ROW-1", ROW), "1", False),
    (chunk("PDF-ROW-2", ROW), "2", False),
    (chunk("PDF-ROW-3", ROW), "2", True),
    (chunk("PDF-ROW-4", ROW), "", True),
    (chunk("PDF-ROW-5", ROW), "3", False),
    (chunk("PDF-ROW-6", ROW), "4", False),
    (chunk("PDF-ROW-7", ROW), "4", True),
    (chunk("PDF-ROW-8", ROW), "5", False),
]
PDF_TEXT = [text for text, _page, _sent in PDF_ROWS]
PDF_BUDGET = 3 * ROW + 50
#: Pages the PDF file itself has; one more than its rows' units.
PDF_FILE_PAGES = 6
#: ``pdf_markdown.page_count``; the prompt's page count is the larger of it and
#: the largest unit (5).
PDF_MARKDOWN_PAGES = 5
LANDSCAPE_PAGE = 3


def page_size(page: int) -> tuple[float, float]:
    """Width and height in points; each scales to 1280 px exactly, to a short edge of its own."""
    if page == 5:
        return 172.5, 320.0
    if page == 6:
        return 710.0, 1280.0
    short = 300 + 10 * page
    if page == LANDSCAPE_PAGE:
        return 640.0, float(short)
    return float(short), 640.0


def image_size(page: int) -> tuple[int, int]:
    width, height = page_size(page)
    scale = 1280 / max(width, height)
    return round(width * scale), round(height * scale)


def write_pdf(path: Path, pages: int) -> Path:
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for page in range(1, pages + 1):
        width, height = page_size(page)
        document.new_page(width=width, height=height)
    document.save(str(path))
    document.close()
    return path


def pdf_markdown(file_id: str, page_count: int) -> None:
    with database.get_search_db() as session:
        session.add(PdfMarkdown(
            file_id=file_id, markdown="# x", page_count=page_count,
            extractor="pymupdf4llm", generated_at=datetime.now(UTC),
        ))


def seed_rows(db, rows) -> None:
    for index, (text, page, sent) in enumerate(rows):
        db.document(text, chunk_index=index, page=page, cos=HIGH if sent else LOW)


def rank_by_seeding(monkeypatch, db) -> None:
    """The question's vector is the axis the ``HIGH`` rows point along."""
    patch_everywhere(monkeypatch, "embed_query", lambda _text: toward(db.db.dim, 1.0))


def seed_pdf(
    db, tmp_path, monkeypatch, rows=PDF_ROWS, *, file_pages: int | None = PDF_FILE_PAGES,
    markdown_pages: int | None = PDF_MARKDOWN_PAGES, file_id: str = PDF_ID,
    relative_to: Path | None = None,
) -> Path:
    """Seed a PDF; ``file_pages`` None writes bytes PyMuPDF cannot open."""
    path = tmp_path / "docs" / f"{file_id}.pdf"
    if file_pages is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"DOCUMENT-BYTES")
    else:
        write_pdf(path, file_pages)
    stored = str(path.relative_to(relative_to)) if relative_to is not None else str(path)
    db.file(file_id, mime=PDF_MIME, file_path=stored, file_size=path.stat().st_size)
    seed_rows(db, rows)
    if markdown_pages is not None:
        pdf_markdown(file_id, markdown_pages)
    rank_by_seeding(monkeypatch, db)
    return path


# ---------------------------------------------------------------------------
# The EPUB fixture
# ---------------------------------------------------------------------------

#: Sections 3 to 6; passages ``[1]`` (section 3) and ``[2]`` (section 6).
EPUB_ROWS = [
    (chunk("EPUB-ROW-0", ROW), "3", False),
    (chunk("EPUB-ROW-1", ROW), "3", True),
    (chunk("EPUB-ROW-2", ROW), "4", False),
    (chunk("EPUB-ROW-3", ROW), "5", False),
    (chunk("EPUB-ROW-4", ROW), "6", True),
    (chunk("EPUB-ROW-5", ROW), "6", False),
]
EPUB_TEXT = [text for text, _page, _sent in EPUB_ROWS]
EPUB_BUDGET = 2 * ROW + 50
EPUB_TITLES = {
    3: "Chapter Three",
    4: "Section four of the book has a title well beyond sixty letters in all",
    6: "Chapter Six",
}


def seed_epub(db, tmp_path, monkeypatch, rows=EPUB_ROWS, *, titles=None) -> None:
    path = tmp_path / "docs" / f"{EPUB_ID}.epub"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"EPUB-BYTES")
    db.file(EPUB_ID, mime=EPUB_MIME, file_path=str(path), file_size=10)
    seed_rows(db, rows)
    for section, title in (EPUB_TITLES if titles is None else titles).items():
        db.section(section, title)
    rank_by_seeding(monkeypatch, db)


# ---------------------------------------------------------------------------
# Clients and scripts
# ---------------------------------------------------------------------------


def wire_for(llm, provider: str, script, *, blind: bool = False, **config) -> Wire:
    """Install the client; ``script(wire, body)`` answers every request.

    ``blind`` gives the profile no vision model; ``config`` overrides its settings.
    """
    wire = Wire(provider)
    wire.script = lambda body: script(wire, body)
    if blind:
        config["vision_model"] = ""
    make = openai_config if provider == "openai_compatible" else ollama_config
    vision_client(llm, provider=provider, wire=wire, config=make(**config))
    return wire


def offers_tool(wire: Wire, body: dict) -> bool:
    if wire.provider == "ollama":
        return bool(body.get("tools"))
    return bool(body.get("tools")) and body.get("tool_choice") != "none"


def carries_exchange(body: dict) -> bool:
    return any(m.get("role") in ("tool", "assistant") for m in body.get("messages", []))


def is_final(wire: Wire, body: dict) -> bool:
    carries = any(m.get("role") == "tool" for m in body.get("messages", []))
    if wire.provider == "ollama":
        return carries and not body.get("tools")
    return carries and bool(body.get("tools")) and body.get("tool_choice") == "none"


def is_today(wire: Wire, body: dict) -> bool:
    return not body.get("tools") and not carries_exchange(body)


def tool_names(body: dict) -> list[str]:
    return [tool.get("function", {}).get("name") for tool in body.get("tools") or []]


def plays(*replies: Reply, then: Reply | None = None, today: Reply | None = None):
    """Tool requests get ``replies`` in turn; the final request ``then``; today's ``today``."""
    queue = list(replies)

    def script(wire, body):
        if offers_tool(wire, body) and queue:
            return queue.pop(0)
        if is_today(wire, body) and not offers_tool(wire, body):
            return today if today is not None else Reply(text=TODAY_ANSWER)
        return then if then is not None else Reply(text=ANSWER)

    return script


def get_text(**arguments) -> ToolCall:
    return ToolCall(arguments, name="get_text")


def get_page_images(**arguments) -> ToolCall:
    return ToolCall(arguments, name="get_page_images")


def calls(*tool_calls: ToolCall, text: str = "") -> Reply:
    return Reply(text=text, calls=tuple(tool_calls))


async def todays_body(
    llm, call, provider: str, file_id: str, *, question=QUESTION, **config,
) -> dict:
    """The request today's text path sends for the question: a profile that cannot see."""
    from tests.file_question_vision_doubles import ask

    wire = wire_for(
        llm, provider, lambda wire, body: Reply(text=TODAY_ANSWER), blind=True, **config,
    )
    response = await ask(call, file_id, question=question)
    assert response.status_code == 200, response.text
    assert len(wire.requests) == 1
    return wire.requests[0]


# ---------------------------------------------------------------------------
# Reading a request
# ---------------------------------------------------------------------------


def results(body: dict) -> list[str]:
    """The tool results of a request, in order."""
    return [message["content"] for message in tool_messages(body)]


def lines(text: str) -> list[str]:
    return text.split("\n")


def system_of(body: dict) -> str:
    for message in body.get("messages", []):
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                return content
            return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


def user_prompt(body: dict) -> str:
    """The user message that carries the question."""
    for message in body.get("messages", []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and QUESTION in content:
            return content
        if isinstance(content, list):
            joined = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
            if QUESTION in joined:
                return joined
    raise AssertionError("no user message carries the question")


def _jpeg_size(raw: bytes) -> tuple[int, int] | None:
    """Width and height from a JPEG's start-of-frame segment."""
    if not raw.startswith(b"\xff\xd8"):
        return None
    index = 2
    while index + 9 < len(raw):
        if raw[index] != 0xFF:
            return None
        marker = raw[index + 1]
        length = int.from_bytes(raw[index + 2:index + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            height = int.from_bytes(raw[index + 5:index + 7], "big")
            width = int.from_bytes(raw[index + 7:index + 9], "big")
            return width, height
        index += 2 + length
    return None


def _decoded(payload: str) -> bytes:
    if payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


def page_of_image(payload: str) -> int | str:
    """The page an image shows, or a description of what it is instead."""
    raw = _decoded(payload)
    size = _jpeg_size(raw)
    if size is None:
        return f"not a JPEG: {raw[:12]!r}"
    for page in range(1, 40):
        width, height = image_size(page)
        if abs(size[0] - width) <= 1 and abs(size[1] - height) <= 1:
            return page
    return f"no page is {size[0]}x{size[1]}"


def images_of(wire: Wire, body: dict) -> list[tuple[str, int | str]]:
    """Every image of the request in order, with the text right before it.

    OpenAI-compatible: label and image are alternating parts of one user
    message. Ollama: each label is the content of the user message carrying
    its one image. Any other shape is returned as a ``problem:`` label.
    """
    out: list[tuple[str, int | str]] = []
    for message in body.get("messages", []):
        if wire.provider == "ollama":
            images = message.get("images") or []
            if not images:
                continue
            if len(images) != 1 or message.get("role") != "user":
                out.append((f"problem: {len(images)} images in one message", ""))
                continue
            content = message.get("content")
            out.append((content if isinstance(content, str) else repr(content),
                        page_of_image(images[0])))
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if part.get("type") != "image_url":
                continue
            before = content[index - 1] if index else {}
            label = before.get("text", "") if before.get("type") == "text" else "problem"
            out.append((label, page_of_image(part["image_url"]["url"])))
    return out


def images_follow_results(wire: Wire, body: dict) -> bool:
    """Every image comes after the last tool message, in the provider's layout."""
    messages = body.get("messages", [])
    last = max((i for i, m in enumerate(messages) if m.get("role") == "tool"), default=-1)
    for index, message in enumerate(messages):
        has = bool(message.get("images")) or (
            isinstance(message.get("content"), list)
            and any(p.get("type") == "image_url" for p in message["content"])
        )
        if has and index < last:
            return False
    after = messages[last + 1:]
    if wire.provider == "ollama":
        return all(
            m.get("role") == "user" and len(m.get("images") or []) == 1 for m in after
        )
    if not after:
        return True
    if len(after) != 1 or after[0].get("role") != "user":
        return False
    kinds = [p.get("type") for p in after[0]["content"]]
    return len(kinds) % 2 == 0 and all(
        kinds[i] == "text" and kinds[i + 1] == "image_url" for i in range(0, len(kinds), 2)
    )


# ---------------------------------------------------------------------------
# Reading the stream
# ---------------------------------------------------------------------------


def kinds(pairs) -> list[str]:
    return [kind for kind, _ in pairs]


def answer_text(pairs) -> str:
    return "".join(data.get("delta", "") for kind, data in pairs if kind == "answer_chunk")


def cited(pairs) -> list[tuple[int, dict | None]]:
    return [(entry["n"], entry["position"]) for entry in dict(pairs).get("citations", [])]


def meta(pairs) -> dict:
    return dict(pairs)["meta"]


# ---------------------------------------------------------------------------
# What the spec words
# ---------------------------------------------------------------------------

GET_TEXT_BAD_PDF = (
    "No text follows: get_text takes a JSON object with either page, a page number "
    "from 1 to {n}, or after, the number of a passage you were given."
)
GET_TEXT_BAD_EPUB = (
    "No text follows: get_text takes a JSON object with either section, a section "
    "number from {a} to {b}, or after, the number of a passage you were given."
)
GET_PAGES_BAD = (
    "No page image follows: get_page_images takes a JSON object with pages, a list of "
    "page numbers, or passages, a list of numbers of passages you were given; at most "
    "4 pages in all."
)
OVER_THREE = "Nothing follows: this question has already been served its 3 tool calls."
FILE_ENDS = "The file ends here."
NOTE_999 = "The text continues: call get_text with after 999 for more."
LIMIT = 3000


def continues(m: int) -> str:
    return f"The text continues: call get_text with after {m} for more."


def cannot_serve(name: str, offered: list[str]) -> str:
    return (
        f"Nothing follows: {name or 'this call'} cannot be served. "
        f"Only {' and '.join(offered)} are available."
    )


def requested(n: int, text: str) -> list[str]:
    return [f"[{n}]", text, ""]


def page_images_head(k: int) -> str:
    if k == 0:
        return "No page image follows."
    if k == 1:
        return "1 page image follows."
    return f"{k} page images follow."


_PASSAGE_NUMBER = re.compile(r"\[\d{1,3}\]")


def excerpt(texts: list[str]) -> str:
    rows = [" ".join(_PASSAGE_NUMBER.sub("", text).split()) for text in texts]
    return " ".join(rows)[:120].rstrip()


def contents_groups(units: list[int], group_rows: dict[int, list[str]]) -> list[list[int]]:
    size = math.ceil(len(units) / 40)
    return [units[i:i + size] for i in range(0, len(units), size)]


def pdf_contents(page_count: int, rows) -> list[str]:
    by_unit: dict[int, list[str]] = {}
    for text, page, _sent in rows:
        if page.strip().isdigit():
            by_unit.setdefault(int(page), []).append(text)
    out = []
    for group in contents_groups(list(range(1, page_count + 1)), by_unit):
        label = f"Page {group[0]}" if len(group) == 1 else f"Pages {group[0]}–{group[-1]}"
        texts = [t for unit in group for t in by_unit.get(unit, [])]
        out.append(f"{label}: {excerpt(texts)}" if texts else label)
    return out


def epub_contents(rows, titles: dict[int, str]) -> list[str]:
    by_unit: dict[int, list[str]] = {}
    for text, page, _sent in rows:
        if page.strip().isdigit():
            by_unit.setdefault(int(page), []).append(text)
    out = []
    for group in contents_groups(sorted(by_unit), by_unit):
        label = (
            f"Section {group[0]}" if len(group) == 1
            else f"Sections {group[0]}–{group[-1]}"
        )
        title = titles.get(group[0])
        texts = [t for unit in group for t in by_unit.get(unit, [])]
        head = f"{label}: {title[:60]} — " if title else f"{label}: "
        out.append(head + excerpt(texts))
    return out


def expected_system_lines(
    *, cut: bool, text: bool, pages: bool, unit: str, tail: list[str],
    history: bool = False,
) -> list[str]:
    """The non-empty lines of the document tool path's system prompt.

    ``tail`` is what follows S181c on every path (the Answer, raw HTML and
    language lines), read from today's request for the same settings.
    """
    items = "the numbered passages below"
    if text:
        items += ", the numbered passages get_text returns"
    if pages:
        items += " and the numbered page images get_page_images returns"
    cited_items = "passages and page images" if pages else "passages"
    opener = (
        f"You answer a question about a single file, using {items} and the File, "
        "Type and Folder lines."
    )
    if history:
        opener += (
            " The earlier turns of this conversation are context for the new question, "
            f"and their answers may be wrong; only {items} may be cited."
        )
    out = [opener, "Rules:", S179]
    out.append(
        "- The passages below are part of the file's text, the parts closest to the "
        "question." if cut else "- The passages below are the file's text."
    )
    out.append(
        "- Each passage" + (" and each page image" if pages else "")
        + " is introduced by its number in square brackets, e.g. [1]."
    )
    out.append(
        f"- Cite the {cited_items} your answer uses inline as [n], right after the "
        "sentence they support."
    )
    out.append(
        f"- Cite only numbers that appear among the {cited_items}; never invent one."
    )
    if text:
        out.append(
            "- If the answer may be in a part of the file the passages do not include, "
            "you must call get_text for that part before answering. Never say that "
            "something is not in the file unless you have called get_text for the part "
            "where it would be."
        )
    if pages:
        out.append(
            "- If the answer depends on what a page shows (a figure, a table, the "
            "layout, a scanned page), call get_page_images for that page. Where a page "
            "image and a passage differ, take exact characters from the passage."
        )
    out.append("- If it is still not there after that, say so plainly instead of guessing.")
    out += [S181A, S181B, S181C, *tail]
    out.append("More of the file:")
    if cut:
        out.append("- The passages are part of the file's text, not all of it.")
    if text:
        out.append(
            f"- The tool get_text returns the file's text from a {unit} you name, or "
            "from just after a numbered passage you name, onward, up to 3,000 "
            "characters per call; a part you already have is named by its number instead."
        )
        out.append(
            "- When a result says the text continues, call get_text again from where "
            "it says for more."
        )
    if pages:
        out.append(
            "- The tool get_page_images returns images of the pages you name, or of the "
            "pages numbered passages are on: up to 4 pages per call."
        )
    both = " of get_text and get_page_images together" if text and pages else ""
    out.append(f"- You may make at most 3 calls{both} for this question.")
    out.append(
        "- Never ask the viewer for pages or places; choose them yourself from what "
        "the prompt gives you."
    )
    out.append(
        "- What these tools return is numbered and cited exactly as the passages you "
        "already have: [n], right after the sentence it supports."
    )
    return out


def tail_after_s181c(system: str) -> list[str]:
    all_lines = [line for line in system.split("\n") if line]
    return all_lines[all_lines.index(S181C) + 1:]


def get_text_definition(unit: str) -> dict:
    noun = "document" if unit == "page" else "book"
    return {
        "name": "get_text",
        "description": (
            f"Return this file's text onward from a {unit}, or from just after a "
            "passage you were given, apart from what you already have. Give either "
            f"{unit} or after."
        ),
        "properties": {
            unit: {"type": "integer", "description": f"A {unit} number of this {noun}"},
            "after": {
                "type": "integer",
                "description": "The number of a passage you were given; the text after it is returned",
            },
        },
    }


GET_PAGE_IMAGES_DEFINITION = {
    "name": "get_page_images",
    "description": (
        "Return images of pages of this document: the pages you name and the pages the "
        "passages you name are on, up to 4 pages."
    ),
    "properties": {
        "pages": {
            "type": "array", "items": {"type": "integer"},
            "description": "Page numbers of this document",
        },
        "passages": {
            "type": "array", "items": {"type": "integer"},
            "description": "Numbers of passages whose pages you want",
        },
    },
}


def tool_definitions(body: dict) -> dict[str, dict]:
    return {tool["function"]["name"]: tool["function"] for tool in body.get("tools") or []}


def as_json(value) -> str:
    return json.dumps(value)[:40]
