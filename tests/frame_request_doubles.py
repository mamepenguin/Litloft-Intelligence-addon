"""Seeding, scripts and readers for the questions where the model asks for frames.

The provider is the addon's real client over ``httpx.MockTransport``
(``tests.file_question_vision_doubles``); a script returns a ``Reply`` that may
hold ``get_frames`` calls. Frame bytes name their millisecond key, so the frames
a request carries can be read back as times.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.routers import rag as rag_router
from app.workers.video_visual_selection import SelectedFrame, Selection, SelectionInputs
from tests.file_context_rules import ANSWER_LINE, S179, S181A, S181B, S181C, block
from tests.file_question_doubles import chunk, toward
from tests.file_question_vision_doubles import Reply, ToolCall, Wire, vision_client
from tests.search_endpoint_db import patch_everywhere
from tests.video_frames_doubles import seed_text_less_video

VIDEO = "vid000000216"
QUESTION = "QUESTION-MARKER what does it show"
PASSAGE = chunk("SPOKEN-PASSAGE-ALPHA", 40)
EVEN_80 = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
PROVIDERS = ["openai_compatible", "ollama"]
KINDS = ["text-video", "text-less-video"]
ANSWER = "FINAL-ANSWER-MARKER"
FAILED = {"error": "Answer generation failed"}
#: The tools a cut-material tool request offers, in order.
BOTH_TOOLS = ["get_frames", "get_transcript"]

#: The seams the 120 s tool window and the 300 s total deadline are read
#: through. A test shrinks them instead of waiting minutes.
WINDOW_SECONDS = "_FRAME_TOOL_WINDOW_SECONDS"
DEADLINE_SECONDS = "_FRAME_TOOL_DEADLINE_SECONDS"

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_prompts"
_CONTENTS_LINE = re.compile(r"^(\d+:\d\d(?::\d\d)?)–(\d+:\d\d(?::\d\d)?)(?: (.*))?$")


def shrink(monkeypatch, name: str, seconds: float) -> None:
    monkeypatch.setattr(rag_router, name, seconds, raising=False)
    patch_everywhere(monkeypatch, name, seconds)


def todays_request(provider: str, kind: str) -> dict:
    """The request the frames path sent for ``seed_video`` before tools existed.

    Recorded from the addon at 895d334 with ``seed_video(kind)``, ``EVEN_80``
    frames and ``QUESTION``, then given the file block and rules of
    SPEC-ADDON-179/181. ``drive_mounts`` is empty in these tests, so the block
    has no ``Folder:`` line.
    """
    path = GOLDEN_DIR / f"frame_request_today_{provider}_{kind}.json"
    return with_file_context(
        json.loads(path.read_text()),
        block(file=f"{VIDEO}.bin", type_="video/mp4"),
    )


def with_file_context(request: dict, file_block: str) -> dict:
    """``request`` with ``file_block`` before the question and S179/S181a/S181b/S181c added."""
    system = request["messages"][0]
    assert system["role"] == "system"
    system["content"] = (
        system["content"]
        .replace("\nRules:\n", f"\nRules:\n{S179}\n", 1)
        .replace(f"\n{ANSWER_LINE}", f"\n{S181A}\n{S181B}\n{S181C}\n{ANSWER_LINE}", 1)
    )
    user = request["messages"][1]
    if isinstance(user["content"], str):
        user["content"] = file_block + user["content"]
    else:
        user["content"][0]["text"] = file_block + user["content"][0]["text"]
    return request


def seed_video(db, tmp_path, kind: str, *, duration: float | None = 80.0) -> int:
    """Seed ``VIDEO`` and return the number its first initial frame takes."""
    seed_text_less_video(db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=duration)
    if kind == "text-less-video":
        return 1
    db.on(VIDEO).transcript(PASSAGE, start=0.0)
    return 2


def frame_text(db, file_id: str, start: float, end: float | None, body: str) -> None:
    """A ``video_frame_texts`` row of ``file_id``; ``db`` is the ``Seeder``."""
    db.frame_texts = getattr(db, "frame_texts", 0) + 1
    db.db.index.sql(
        "INSERT INTO video_frame_texts(file_id, chunk_index, start_time, end_time, text) "
        "VALUES (:f, :c, :s, :e, :t)",
        f=file_id, c=db.frame_texts, s=start, e=end, t=body,
    )


def selection(
    monkeypatch, *, sections: list[tuple[float, float]], frames: list[float],
    duration: float | None, fail: bool = False,
) -> dict:
    """Make ``sections`` the navigation list and ``frames`` the Ask list's times.

    Returns a record of the calls: ``loaded`` files and ``selected`` runs.
    """
    def item(start: float, end: float, at: float, n: int) -> SelectedFrame:
        return SelectedFrame(
            section_start=float(start), section_end=float(end), frame_time=float(at),
            embedding_id=f"clip{n:03d}", embedding_type="clip", kind="visual",
            textual=False,
        )

    navigation = tuple(item(a, b, a, n) for n, (a, b) in enumerate(sections))
    ask_list = tuple(item(t, t, t, 100 + n) for n, t in enumerate(frames))
    record: dict = {"loaded": [], "selected": 0}

    def _load(file_id):
        record["loaded"].append(file_id)
        return SelectionInputs(duration=duration)

    def _select(*_args, **_kwargs):
        record["selected"] += 1
        if fail:
            raise RuntimeError("selection failed")
        return Selection(
            navigation=navigation, ask=ask_list,
            candidate_count=len(navigation) + len(ask_list),
        )

    patch_everywhere(monkeypatch, "load_selection_inputs", _load)
    patch_everywhere(monkeypatch, "select_scenes", _select)
    return record


def get_frames(*times) -> ToolCall:
    return ToolCall({"times": list(times)})


def asks(*times, text: str = "", thinking: str = "") -> Reply:
    return Reply(text=text, calls=(get_frames(*times),), thinking=thinking)


def get_transcript(start, end) -> ToolCall:
    return ToolCall({"start": start, "end": end}, name="get_transcript")


def transcript_asks(start, end, *, text: str = "", thinking: str = "") -> Reply:
    return Reply(
        text=text, calls=(get_transcript(start, end),), thinking=thinking,
    )


def rank_in_file_order(monkeypatch, db) -> None:
    """Rank every row equally, so fitting takes them in file order."""
    patch_everywhere(monkeypatch, "embed_query", lambda _text: toward(db.db.dim, 1.0))


def seed_cut_video(db, tmp_path, rows, *, duration: float = 80.0) -> None:
    """Seed ``VIDEO`` with transcript ``(text, start, end)`` rows and no passage."""
    seed_text_less_video(db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=duration)
    for text, start, end in rows:
        db.on(VIDEO).transcript(text, start=start, end=end)


#: Three equal-sized transcript rows; a 40-character budget sends only the first.
CUT_ROW_TEXTS = [chunk(f"CUT-ROW-{index}", 30) for index in range(3)]
CUT_ROWS = tuple(zip(CUT_ROW_TEXTS, (0.0, 20.0, 30.0), (5.0, 25.0, 35.0)))
CUT_BUDGET = 40


def wire_for(llm, provider: str, script) -> Wire:
    wire = Wire(provider)
    wire.script = lambda body: script(wire, body)
    vision_client(llm, provider=provider, wire=wire)
    return wire


def plays(*replies: Reply, then: Reply | None = None):
    """Answer tool requests with ``replies`` in turn, everything else with ``then``."""
    queue = list(replies)

    def script(wire, body):
        if offers_tool(wire, body) and queue:
            return queue.pop(0)
        return then if then is not None else Reply(text=ANSWER)

    return script


# ---------------------------------------------------------------------------
# Reading a request
# ---------------------------------------------------------------------------


def offers_tool(wire, body: dict) -> bool:
    if wire.provider == "ollama":
        return bool(body.get("tools"))
    return bool(body.get("tools")) and body.get("tool_choice") != "none"


def tool_names(body: dict) -> list[str]:
    return [tool.get("function", {}).get("name") for tool in body.get("tools") or []]


def is_final(wire, body: dict) -> bool:
    """A request that carries an exchange and offers no tool."""
    carries = any(m.get("role") == "tool" for m in body.get("messages", []))
    if wire.provider == "ollama":
        return carries and not body.get("tools")
    return carries and bool(body.get("tools")) and body.get("tool_choice") == "none"


def assistant_messages(body: dict) -> list[dict]:
    return [m for m in body.get("messages", []) if m.get("role") == "assistant"]


def prompt_text(wire, body: dict) -> str:
    """The text of the first user message after the system prompt and any history."""
    users = [m for m in body.get("messages", []) if m.get("role") == "user"]
    for message in reversed(users):
        content = message.get("content")
        if isinstance(content, str) and QUESTION in content:
            return content
        if isinstance(content, list):
            joined = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
            if QUESTION in joined:
                return joined
    return ""


def contents_lines(prompt: str) -> list[str]:
    """Lines shaped ``<start>–<end>[ <excerpt>]``, trailing space removed."""
    return [
        line.rstrip()
        for line in prompt.splitlines()
        if _CONTENTS_LINE.match(line.rstrip())
    ]


def split_contents_line(line: str) -> tuple[str, str, str]:
    match = _CONTENTS_LINE.match(line)
    assert match, line
    return match.group(1), match.group(2), match.group(3) or ""


def answer_text(pairs) -> str:
    return "".join(data.get("delta", "") for kind, data in pairs if kind == "answer_chunk")


def kinds(pairs) -> list[str]:
    return [kind for kind, _ in pairs]


def cited(pairs) -> list[tuple[int, dict | None]]:
    entries = dict(pairs).get("citations", [])
    return [(entry["n"], entry["position"]) for entry in entries]
