"""SPEC-ADDON-116 Terms (contents) §1.1 §1.2 I7: the table of contents and duration in the prompt.

The navigation list and the Ask list are pinned by a ``select_scenes`` double;
the excerpts come from ``transcript_chunks`` and ``video_frame_texts`` rows of
the search DB. Contents lines are read as ``<start>–<end>[ <excerpt>]``.
"""

from __future__ import annotations

import re

import pytest

from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    Reply,
    Wire,
    addon,
    ask,
    call,
    db,
    llm,
    vision_client,
)
from tests.frame_request_doubles import (
    EVEN_80,
    QUESTION,
    VIDEO,
    contents_lines,
    frame_text,
    offers_tool,
    prompt_text,
    selection,
    split_contents_line,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import FFmpeg, ms, seed_text_less_video, sent_keys

OTHER_VIDEO = "vid000000217"


def _video(db, tmp_path, *, duration):
    seed_text_less_video(db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=duration)


def _first_request(llm, provider: str = "openai_compatible") -> Wire:
    wire = Wire(provider)
    wire.script = lambda body: Reply(text="ANSWER")
    vision_client(llm, provider=provider, wire=wire)
    return wire


class TestTheContents:
    @pytest.mark.parametrize("provider", ["openai_compatible", "ollama"])
    async def test_each_section_lists_its_times_and_the_text_said_and_shown_in_it(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 Terms (contents) §1.1 §1.2 §4 I7: overlap, order, collapsed space, one selection run."""
        _video(db, tmp_path, duration=80.0)
        db.on(VIDEO).transcript("ALPHA  one", start=2.0, end=8.0)
        db.on(VIDEO).transcript("ECHO", start=25.0, end=30.0)
        db.on(VIDEO).transcript("BRAVO", start=28.0, end=33.0)
        db.on(VIDEO).transcript("DELTA", start=40.0, end=44.0)
        frame_text(db, VIDEO, 40.0, 45.0, "CHARLIE\n  sign")
        frame_text(db, VIDEO, 31.0, 32.0, "GOLF")
        seed_text_less_video(
            db, OTHER_VIDEO, path=tmp_path / f"{OTHER_VIDEO}.mp4", duration=80.0,
        )
        db.on(OTHER_VIDEO).transcript("HOTEL", start=5.0, end=10.0)
        frame_text(db, OTHER_VIDEO, 35.0, 40.0, "INDIA")
        record = selection(
            monkeypatch, sections=[(0.0, 30.0), (30.0, 60.0), (60.0, 80.0)],
            frames=[10.0, 40.0, 70.0], duration=80.0,
        )
        FFmpeg().install(monkeypatch)
        wire = _first_request(llm, provider)

        await ask(call, VIDEO, question=QUESTION)

        body = wire.requests[0]
        assert offers_tool(wire, body)
        prompt = prompt_text(wire, body)
        lines = contents_lines(prompt)
        assert lines == [
            "0:00–0:30 ALPHA one ECHO BRAVO",
            "0:30–1:00 BRAVO GOLF DELTA CHARLIE sign",
            "1:00–1:20",
        ]
        assert prompt.index(lines[0]) > prompt.index("DELTA")
        assert record["loaded"] == [VIDEO]
        assert record["selected"] == 1
        # The frame set's even fill (SPEC-ADDON-108) adds times around these.
        assert {ms(t) for t in [10.0, 40.0, 70.0]} <= set(sent_keys(wire))

    async def test_at_most_forty_sections_are_listed_in_time_order(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 Terms (contents) §10 I7: a navigation list of 50 gives at most 40 lines."""
        _video(db, tmp_path, duration=100.0)
        sections = [(2.0 * i, 2.0 * i + 2.0) for i in range(50)]
        selection(monkeypatch, sections=sections, frames=[10.0, 50.0, 90.0], duration=100.0)
        FFmpeg().install(monkeypatch)
        wire = _first_request(llm)

        await ask(call, VIDEO, question=QUESTION)

        lines = contents_lines(prompt_text(wire, wire.requests[0]))
        assert 1 <= len(lines) <= 40
        known = {
            (f"0:{int(a):02d}" if a < 60 else f"1:{int(a) - 60:02d}",
             f"0:{int(b):02d}" if b < 60 else f"1:{int(b) - 60:02d}")
            for a, b in sections
        }
        spans = [split_contents_line(line)[:2] for line in lines]
        assert all(span in known for span in spans)
        seconds = [_seconds(start) for start, _ in spans]
        assert seconds == sorted(set(seconds))

    @pytest.mark.parametrize("length", [119, 120, 300])
    async def test_an_excerpt_is_cut_at_120_characters(
        self, length, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 Terms (contents) §10 I7: an excerpt never passes 120 characters."""
        _video(db, tmp_path, duration=30.0)
        words = ("WORD " * 100)[:length].rstrip().ljust(length, "Z")
        db.on(VIDEO).transcript(words, start=1.0, end=5.0)
        selection(monkeypatch, sections=[(0.0, 30.0)], frames=[10.0], duration=30.0)
        FFmpeg().install(monkeypatch)
        wire = _first_request(llm)

        await ask(call, VIDEO, question=QUESTION)

        lines = contents_lines(prompt_text(wire, wire.requests[0]))
        assert len(lines) == 1
        excerpt = split_contents_line(lines[0])[2]
        if length <= 120:
            assert excerpt == words
        else:
            assert 100 <= len(excerpt) <= 120
            assert words.startswith(excerpt[:100])

    async def test_no_line_carries_a_bracketed_number(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 Terms (contents) I7: a ``[n]`` in the source text does not reach the contents."""
        _video(db, tmp_path, duration=30.0)
        db.on(VIDEO).transcript("see [3] and [12] here", start=1.0, end=5.0)
        frame_text(db, VIDEO, 6.0, 8.0, "panel [4]")
        selection(monkeypatch, sections=[(0.0, 30.0)], frames=[10.0], duration=30.0)
        FFmpeg().install(monkeypatch)
        wire = _first_request(llm)

        await ask(call, VIDEO, question=QUESTION)

        lines = contents_lines(prompt_text(wire, wire.requests[0]))
        assert len(lines) == 1
        assert "here" in lines[0]
        assert re.search(r"\[\d+\]", lines[0]) is None

    async def test_times_past_an_hour_are_written_h_mm_ss(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 Terms (contents): ``m:ss`` under an hour, ``h:mm:ss`` from it."""
        _video(db, tmp_path, duration=4000.0)
        selection(
            monkeypatch, sections=[(0.0, 3600.0), (3600.0, 3725.0), (3725.0, 4000.0)],
            frames=[100.0, 3650.0, 3800.0], duration=4000.0,
        )
        FFmpeg().install(monkeypatch)
        wire = _first_request(llm)

        await ask(call, VIDEO, question=QUESTION)

        assert contents_lines(prompt_text(wire, wire.requests[0])) == [
            "0:00–1:00:00", "1:00:00–1:02:05", "1:02:05–1:06:40",
        ]

    @pytest.mark.parametrize("case", ["empty", "fails"])
    async def test_an_empty_or_failed_selection_gives_no_contents_but_the_tool(
        self, case, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 Terms (contents) §2: no contents line; ``get_frames`` still offered."""
        _video(db, tmp_path, duration=80.0)
        db.on(VIDEO).transcript("ALPHA", start=2.0, end=8.0)
        selection(monkeypatch, sections=[], frames=[], duration=80.0, fail=case == "fails")
        FFmpeg().install(monkeypatch)
        wire = _first_request(llm)

        await ask(call, VIDEO, question=QUESTION)

        body = wire.requests[0]
        assert offers_tool(wire, body)
        assert sent_keys(wire)[:8] == [ms(t) for t in EVEN_80]
        assert contents_lines(prompt_text(wire, body)) == []


class TestTheDuration:
    async def test_a_known_duration_is_stated(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-116 §1.2: the duration is added to the prompt when known."""
        _video(db, tmp_path, duration=4321.0)
        selection(monkeypatch, sections=[], frames=[10.0, 20.0], duration=4321.0)
        FFmpeg().install(monkeypatch)
        wire = _first_request(llm)

        await ask(call, VIDEO, question=QUESTION)

        body = wire.requests[0]
        assert offers_tool(wire, body)
        prompt = prompt_text(wire, body)
        assert "1:12:01" in prompt or "4321" in prompt


def _seconds(stamp: str) -> int:
    total = 0
    for part in stamp.split(":"):
        total = total * 60 + int(part)
    return total
