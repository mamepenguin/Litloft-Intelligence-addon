"""Reads the frames of a file-question request as the provider received them.

SPEC-ADDON-114 I11: every frame image is preceded, with nothing between, by a
text whose whole content is its label ``[n] <time>``. With an OpenAI-compatible
profile the labels and images are alternating parts of the user message after
the prompt text; with ollama each label and its one image are a user message of
their own after the prompt message.

``frame_layout`` never raises on another shape: it reports the shape it found as
``problem`` so that a test fails on the layout, not on the reader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tests.video_frames_doubles import MARKER, decoded

_NUMBER = re.compile(r"\[(\d+)\]")
_CLOCK = re.compile(r"\b\d+:\d\d\b")


@dataclass(frozen=True)
class Layout:
    prompt: str = ""
    frames: list[tuple[str, int]] = field(default_factory=list)
    problem: str | None = None


def stamp(seconds: float) -> str:
    whole = int(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _key(payload: str) -> int | None:
    try:
        raw = decoded(payload)
    except Exception:  # noqa: BLE001 - reported as a problem, not raised
        return None
    if not raw.startswith(MARKER):
        return None
    return int(raw[len(MARKER):])


def _openai(body: dict) -> Layout:
    messages = body.get("messages", [])
    users = [m for m in messages if m.get("role") == "user"]
    if not users:
        return Layout(problem="no user message")
    content = users[-1].get("content")
    if not isinstance(content, list) or not content:
        return Layout(problem=f"user content is not a list of parts: {content!r:.200}")
    head, rest = content[0], content[1:]
    if head.get("type") != "text":
        return Layout(problem=f"the first part is {head.get('type')}, not the prompt text")
    kinds = [part.get("type") for part in rest]
    if len(rest) % 2 or any(
        kinds[i] != "text" or kinds[i + 1] != "image_url" for i in range(0, len(rest), 2)
    ):
        return Layout(
            prompt=head.get("text", ""),
            problem=f"parts after the prompt text are not label/image pairs: {kinds}",
        )
    frames: list[tuple[str, int]] = []
    for i in range(0, len(rest), 2):
        key = _key(rest[i + 1]["image_url"]["url"])
        if key is None:
            return Layout(prompt=head.get("text", ""), problem="an image is not a frame")
        frames.append((rest[i].get("text", ""), key))
    elsewhere = sum(
        1
        for message in messages
        if message is not users[-1] and isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    )
    if elsewhere:
        return Layout(problem=f"{elsewhere} images outside the last user message")
    return Layout(prompt=head.get("text", ""), frames=frames)


def _ollama(body: dict) -> Layout:
    messages = body.get("messages", [])
    first = next((i for i, m in enumerate(messages) if m.get("images")), None)
    if first is None:
        return Layout(problem="no message carries an image")
    if first == 0:
        return Layout(problem="the first message carries an image")
    prompt_message = messages[first - 1]
    if prompt_message.get("role") != "user" or not isinstance(
        prompt_message.get("content"), str,
    ):
        return Layout(problem=f"the message before the frames is {prompt_message!r:.200}")
    frames: list[tuple[str, int]] = []
    for message in messages[first:]:
        images = message.get("images") or []
        if message.get("role") != "user" or len(images) != 1:
            return Layout(
                prompt=prompt_message["content"],
                problem=(
                    f"a message after the prompt is role {message.get('role')!r} "
                    f"with {len(images)} images"
                ),
            )
        key = _key(images[0])
        if key is None:
            return Layout(prompt=prompt_message["content"], problem="an image is not a frame")
        content = message.get("content")
        frames.append((content if isinstance(content, str) else repr(content), key))
    return Layout(prompt=prompt_message["content"], frames=frames)


def frame_layout(wire, body: dict) -> Layout:
    return _ollama(body) if wire.provider == "ollama" else _openai(body)


def pairing_lines(prompt: str, first: int) -> list[str]:
    """Lines of the prompt text that pair a frame number (``>= first``) with a time."""
    return [
        line
        for line in prompt.splitlines()
        if _CLOCK.search(line)
        and any(int(n) >= first for n in _NUMBER.findall(line))
    ]


def assert_labelled(wire, body: dict, *, first: int, times: list[float]) -> Layout:
    """Assert the I11 layout with frames ``[first]..`` at ``times``; return it."""
    layout = frame_layout(wire, body)
    assert layout.problem is None, layout.problem
    expected = [
        (f"[{n}] {stamp(t)}", round(t * 1000))
        for n, t in enumerate(times, start=first)
    ]
    assert layout.frames == expected
    assert pairing_lines(layout.prompt, first) == []
    return layout
