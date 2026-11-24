"""Readers and scenarios for the file block and rules of SPEC-ADDON-179/180/181.

A request is compared with today's by taking exactly the spec's additions back
out, so a test fails on anything else that moved.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from tests import document_tool_doubles as document_tools
from tests.file_context_rules import (  # noqa: F401 - re-exported
    ANSWER_LINE,
    IMAGE_RULE,
    LOFT_MIME,
    NEW_OPENER,
    OLD_OPENER,
    PASSAGES_ONLY_SENTENCE,
    S179,
    S180,
    S181A,
    S181B,
    S181C,
    TOP_OF_DRIVE,
    block,
)
from tests.file_question_doubles import chunk
from tests.file_question_vision_doubles import (
    Reply,
    Wire,
    ask,
    openai_config,
    refusing,
    seed_document,
    seed_picture,
    vision_client,
)
from tests.frame_request_doubles import (
    ANSWER,
    CUT_BUDGET,
    CUT_ROWS,
    EVEN_80,
    KINDS,
    PROVIDERS,
    QUESTION,
    VIDEO,
    asks,
    get_frames,
    get_transcript,
    offers_tool,
    plays,
    rank_in_file_order,
    seed_cut_video,
    seed_video,
    wire_for,
)
from tests.video_frames_doubles import FFmpeg, ms, scenes

PDF_ID = "pdf000000179"
PICTURE_ID = "pic000000179"
PDF_TEXT = chunk("REPORT-PASSAGE-ONE", 40)
PICTURE_TEXT = chunk("Pokemon: Since 1996", 40)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_prompts"


# ---------------------------------------------------------------------------
# Reading a request
# ---------------------------------------------------------------------------


def system_of(body: dict) -> str:
    for message in body.get("messages", []):
        if message.get("role") == "system":
            content = message.get("content")
            return content if isinstance(content, str) else "\n".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
    return ""


def _question_message(body: dict) -> dict | None:
    """The user message that carries ``QUESTION``: the file prompt."""
    for message in body.get("messages", []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and QUESTION in content:
            return message
        if isinstance(content, list) and any(
            QUESTION in part.get("text", "") for part in content if part.get("type") == "text"
        ):
            return message
    return None


def user_text(body: dict) -> str:
    """The first text of the file prompt: where the file block must start."""
    message = _question_message(body)
    assert message is not None, "no user message carries the question"
    content = message["content"]
    if isinstance(content, str):
        return content
    return next(part["text"] for part in content if part.get("type") == "text")


def question_bodies(wire: Wire) -> list[dict]:
    return [body for body in wire.requests if _question_message(body) is not None]


# ---------------------------------------------------------------------------
# Rule positions
# ---------------------------------------------------------------------------


def assert_s179_and_s181(system: str) -> None:
    """S179 right after ``Rules:``; S181a, S181b then S181c right above the Answer line."""
    lines = system.split("\n")
    assert lines.count("Rules:") == 1, system
    assert lines[lines.index("Rules:") + 1] == S179, system
    assert system.count(S179) == 1, system
    answer = [i for i, line in enumerate(lines) if line.startswith(ANSWER_LINE)]
    assert len(answer) == 1, system
    assert lines[answer[0] - 3: answer[0]] == [S181A, S181B, S181C], system
    for rule in (S181A, S181B, S181C):
        assert system.count(rule) == 1, system


# ---------------------------------------------------------------------------
# Taking the additions back out
# ---------------------------------------------------------------------------


def without_additions(body: dict, expected_block: str) -> dict:
    """``body`` with the file block, S179, S181a/b/c, the new opener and S180 undone.

    Asserts the user prompt starts with exactly ``expected_block``.
    """
    out = copy.deepcopy(body)
    message = _question_message(out)
    if message is None:
        return out
    for each in out["messages"]:
        if each.get("role") == "system" and isinstance(each.get("content"), str):
            each["content"] = _undo_rules(each["content"])
    content = message["content"]
    if isinstance(content, str):
        assert content.startswith(expected_block), content[:400]
        message["content"] = content[len(expected_block):]
    else:
        part = next(p for p in content if p.get("type") == "text")
        assert part["text"].startswith(expected_block), part["text"][:400]
        part["text"] = part["text"][len(expected_block):]
    return out


def _undo_rules(system: str) -> str:
    kept = [line for line in system.split("\n") if line not in (S179, S181A, S181B, S181C)]
    return (
        "\n".join(kept)
        .replace(NEW_OPENER, OLD_OPENER)
        .replace(S180, PASSAGES_ONLY_SENTENCE)
    )


def today(name: str) -> list[dict]:
    """Today's question requests for ``name``.

    Recorded from the addon at d9b7daf by running ``RUN[name]`` and keeping
    ``question_bodies(wire)``.
    """
    path = GOLDEN_DIR / f"file_question_today_{name}.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# The paths, each played to the end
# ---------------------------------------------------------------------------


async def _asked(call, file_id: str) -> None:
    response = await ask(call, file_id, question=QUESTION)
    assert response.status_code == 200, response.text


async def passages_only(db, addon, llm, call, tmp_path, monkeypatch, **settings) -> Wire:
    """Today's text path, pinned by a profile that cannot see."""
    if settings:
        addon(**settings)
    seed_document(db, PDF_ID, text=PDF_TEXT, path=tmp_path / "docs" / "report.pdf")
    _, wire = vision_client(llm, config=openai_config(vision_model=""))
    await _asked(call, PDF_ID)
    return wire


DOCUMENT_ID = "pdf000000182"


async def document_tool(
    db, addon, llm, call, tmp_path, monkeypatch, **settings,
) -> Wire:
    """A cut PDF with a seeing profile: one text request, then the answer."""
    addon(budget=document_tools.PDF_BUDGET, **settings)
    document_tools.seed_pdf(
        db, tmp_path, monkeypatch, file_id=DOCUMENT_ID,
    )
    wire = document_tools.wire_for(
        llm,
        "openai_compatible",
        document_tools.plays(
            document_tools.calls(document_tools.get_text(page=1)),
            then=Reply(text=ANSWER),
        ),
    )
    await _asked(call, DOCUMENT_ID)
    return wire


async def passages_and_image(db, addon, llm, call, tmp_path, monkeypatch, **settings) -> Wire:
    if settings:
        addon(**settings)
    seed_picture(db, PICTURE_ID, text=PICTURE_TEXT, path=tmp_path / "pics" / "logo.png")
    _, wire = vision_client(llm)
    await _asked(call, PICTURE_ID)
    return wire


async def image_only(db, addon, llm, call, tmp_path, monkeypatch, **settings) -> Wire:
    if settings:
        addon(**settings)
    seed_picture(db, PICTURE_ID, text=None, path=tmp_path / "pics" / "logo.png")
    _, wire = vision_client(llm)
    await _asked(call, PICTURE_ID)
    return wire


async def text_only_retry(db, addon, llm, call, tmp_path, monkeypatch, **settings) -> Wire:
    if settings:
        addon(**settings)
    seed_picture(db, PICTURE_ID, text=PICTURE_TEXT, path=tmp_path / "pics" / "logo.png")
    wire = Wire("openai_compatible")
    wire.script = refusing(wire)
    vision_client(llm, wire=wire)
    await _asked(call, PICTURE_ID)
    return wire


def _frames_with_tool(provider: str, kind: str):
    """Three served tool requests, then the final request."""
    async def run(db, addon, llm, call, tmp_path, monkeypatch, **settings) -> Wire:
        if settings:
            addon(**settings)
        seed_video(db, tmp_path, kind)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = wire_for(llm, provider, plays(asks(12.0), asks(22.0), asks(32.0)))
        await _asked(call, VIDEO)
        return wire

    return run


def _frames_fallback(provider: str, kind: str):
    """A refused tool request, then the request without the tool."""
    async def run(db, addon, llm, call, tmp_path, monkeypatch, **settings) -> Wire:
        if settings:
            addon(**settings)
        seed_video(db, tmp_path, kind)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = wire_for(llm, provider, lambda wire, body: (
            Reply(status=500) if offers_tool(wire, body) else Reply(text=ANSWER)
        ))
        await _asked(call, VIDEO)
        return wire

    return run


def _frames_refused(provider: str):
    """Tool and image requests refused: the text-only retry of a text video."""
    async def run(db, addon, llm, call, tmp_path, monkeypatch, **settings) -> Wire:
        if settings:
            addon(**settings)
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        wire = wire_for(llm, provider, lambda wire, body: (
            Reply(status=400) if offers_tool(wire, body) or wire.has_image(body)
            else Reply(text=ANSWER)
        ))
        await _asked(call, VIDEO)
        return wire

    return run


def _no_frame_prepared(provider: str):
    """A text video whose every frame fails: the text-only answer."""
    async def run(db, addon, llm, call, tmp_path, monkeypatch, **settings) -> Wire:
        if settings:
            addon(**settings)
        seed_video(db, tmp_path, "text-video")
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg(fail={ms(t) for t in EVEN_80}).install(monkeypatch)
        wire = wire_for(llm, provider, plays())
        await _asked(call, VIDEO)
        return wire

    return run


def _cut_material(provider: str):
    """Passages and frames with both tools: two served tool requests, then a third."""
    async def run(db, addon, llm, call, tmp_path, monkeypatch, **settings) -> Wire:
        addon(budget=CUT_BUDGET, **settings)
        seed_cut_video(db, tmp_path, CUT_ROWS)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        rank_in_file_order(monkeypatch, db)
        wire = wire_for(llm, provider, plays(
            Reply(calls=(get_transcript(20.0, 25.0),)),
            Reply(calls=(get_frames(12.0),)),
        ))
        await _asked(call, VIDEO)
        return wire

    return run


VIDEO_BLOCK = block(file=f"{VIDEO}.bin", type_="video/mp4")

RUN = {
    "passages-only": passages_only,
    "passages-and-image": passages_and_image,
    "image-only": image_only,
    "text-only-retry": text_only_retry,
    # This path did not exist before the document tools, so its golden is empty.
    "document-tool": document_tool,
}
#: The file block each path's file gets: ``db.file`` names it ``<id>.bin``,
#: and its absolute path under ``tmp_path`` has no mount, so no ``Folder:``.
BLOCKS = {
    "passages-only": block(file=f"{PDF_ID}.bin", type_="application/pdf"),
    "passages-and-image": block(file=f"{PICTURE_ID}.bin", type_="image/png"),
    "image-only": block(file=f"{PICTURE_ID}.bin", type_="image/png"),
    "text-only-retry": block(file=f"{PICTURE_ID}.bin", type_="image/png"),
    "document-tool": block(file=f"{DOCUMENT_ID}.bin", type_="application/pdf"),
}
for _provider in PROVIDERS:
    for _kind in KINDS:
        RUN[f"frames-tool-{_provider}-{_kind}"] = _frames_with_tool(_provider, _kind)
        RUN[f"frames-fallback-{_provider}-{_kind}"] = _frames_fallback(_provider, _kind)
    RUN[f"frames-refused-{_provider}"] = _frames_refused(_provider)
    RUN[f"no-frame-prepared-{_provider}"] = _no_frame_prepared(_provider)
    RUN[f"cut-material-{_provider}"] = _cut_material(_provider)
for _name in RUN:
    BLOCKS.setdefault(_name, VIDEO_BLOCK)

SCENARIOS = tuple(RUN)

#: The paths whose requests are today's apart from SPEC-ADDON-179/180/181's
#: lines; a document question sends the requests of SPEC-ADDON-182 instead.
TODAYS_PATHS = tuple(name for name in SCENARIOS if name != "document-tool")
