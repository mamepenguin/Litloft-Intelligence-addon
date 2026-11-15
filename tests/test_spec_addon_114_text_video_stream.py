"""SPEC-ADDON-114 I6: the slot, the preparation bound and the keep-alive of a text video.

The provider is the addon's real client over ``httpx.MockTransport``; a stream
whose first token arrives late shows whether the line is kept alive while the
model has not answered yet.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx

import app.frame_cache as frame_cache
from app.routers import rag as rag_router
from tests.file_question_doubles import chunk
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    DRIVE,
    VISION_MODEL,
    Wire,
    addon,
    ask,
    ask_slots,
    by_kind,
    call,
    db,
    disconnect_during_preparation,
    llm,
    openai_config,
    resolved,
    vision_client,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import (
    FFmpeg,
    ms,
    scenes,
    seed_text_less_video,
    sent_keys,
    timed_post,
)

VIDEO = "vid000000124"
QUESTION = "QUESTION-MARKER what does it show"
PASSAGE = chunk("SPOKEN-PASSAGE-ALPHA", 40)
MAX = rag_router._MAX_CONCURRENT_ASK
KEEPALIVE = b": keepalive\n\n"
EVEN_80 = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]


def _text_video(db, tmp_path, *, duration=80.0):
    seed_text_less_video(db, VIDEO, path=tmp_path / f"{VIDEO}.mp4", duration=duration)
    db.on(VIDEO).transcript(PASSAGE, start=0.0)


def _delta(text: str, model: str) -> bytes:
    piece = {
        "id": "1", "object": "chat.completion.chunk", "created": 0,
        "model": model,
        "choices": [{
            "index": 0, "delta": {"role": "assistant", "content": text},
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(piece)}\n\n".encode()


class _FirstTokenLate(httpx.AsyncByteStream):
    def __init__(self, delay: float, model: str):
        self.delay = delay
        self.model = model

    async def __aiter__(self):
        await asyncio.sleep(self.delay)
        yield _delta("LATE-TOKEN", self.model)

    async def aclose(self) -> None:
        return None


class _Late:
    """A provider whose answers start late, optionally refusing image requests."""

    def __init__(self, llm, *, delay: float, refuse_images: bool = False, config=None):
        from openai import AsyncOpenAI

        from app.llm import LLMClient

        self.reader = Wire("openai_compatible")
        self.requests: list[tuple[float, dict]] = []
        settings = config or openai_config()

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.requests.append((time.monotonic(), body))
            if refuse_images and self.reader.has_image(body):
                return httpx.Response(400, json={"error": {"message": "no images"}})
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"},
                stream=_FirstTokenLate(delay, body.get("model") or VISION_MODEL),
            )

        client = LLMClient(settings)
        client._client = AsyncOpenAI(  # noqa: SLF001 - the test replaces the transport
            base_url=settings.base_url, api_key="test-key", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        llm.serve(resolved(client, settings))

    def times(self, *, images: bool) -> list[float]:
        return [at for at, body in self.requests if self.reader.has_image(body) == images]


def _keepalives(received) -> list[float]:
    return [at for at, data in received if data == KEEPALIVE]


def _first_chunk(received) -> float:
    return next(at for at, data in received if b"event: answer_chunk" in data)


class TestTheKeepAlive:
    async def test_the_line_is_kept_alive_while_the_frames_are_prepared(
        self, db, addon, llm, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I6 §1.4: a comment before the model request, while frames are read."""
        from app.main import app

        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg(hold=0.3).install(monkeypatch)
        monkeypatch.setattr(rag_router, "_KEEPALIVE_SECONDS", 0.05)
        late = _Late(llm, delay=0.0)

        _started, received, status = await timed_post(
            app, path=f"/files/{VIDEO}/ask", body={"question": QUESTION},
        )

        assert status == 200
        assert late.times(images=True), "no request with frames was sent"
        requested = late.times(images=True)[0]
        assert any(at <= requested for at in _keepalives(received))

    async def test_the_line_is_kept_alive_until_the_first_token_with_frames(
        self, db, addon, llm, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I6 §1.4: comments between the request with frames and its first token."""
        from app.main import app

        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        monkeypatch.setattr(rag_router, "_KEEPALIVE_SECONDS", 0.05)
        late = _Late(llm, delay=0.5)

        _started, received, status = await timed_post(
            app, path=f"/files/{VIDEO}/ask", body={"question": QUESTION},
        )

        assert status == 200
        assert late.times(images=True), "no request with frames was sent"
        requested = late.times(images=True)[0]
        first = _first_chunk(received)
        assert any(requested <= at <= first for at in _keepalives(received))

    async def test_the_line_is_kept_alive_during_the_text_only_retry(
        self, db, addon, llm, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I6 §1.4: after a refusal, comments until the retry's first token."""
        from app.main import app

        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        monkeypatch.setattr(rag_router, "_KEEPALIVE_SECONDS", 0.05)
        late = _Late(llm, delay=0.5, refuse_images=True)

        _started, received, status = await timed_post(
            app, path=f"/files/{VIDEO}/ask", body={"question": QUESTION},
        )

        assert status == 200
        assert late.times(images=True), "no request with frames was sent"
        assert late.times(images=False), "no text-only retry was sent"
        retried = late.times(images=False)[0]
        first = _first_chunk(received)
        assert any(retried <= at <= first for at in _keepalives(received))

    async def test_a_blind_profile_writes_no_keepalive_while_a_seeing_one_does(
        self, db, addon, llm, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I8 §1.7: today's text path is unchanged, keep-alive included."""
        from app.main import app

        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        FFmpeg().install(monkeypatch)
        monkeypatch.setattr(rag_router, "_KEEPALIVE_SECONDS", 0.05)

        _Late(llm, delay=0.5, config=openai_config(vision_model=""))
        _s, blind, _status = await timed_post(
            app, path=f"/files/{VIDEO}/ask", body={"question": QUESTION},
        )
        assert _keepalives(blind) == []

        _Late(llm, delay=0.5)
        _s, seeing, _status = await timed_post(
            app, path=f"/files/{VIDEO}/ask", body={"question": QUESTION},
        )
        assert _keepalives(seeing)


class TestTheSlot:
    async def test_one_slot_is_held_through_preparation_the_request_and_the_retry(
        self, db, addon, llm, call, ask_slots, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I6: one slot from after the material until the end, retry included."""
        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg()
        during_frames: list[int] = []

        real_run = frame_cache.subprocess.run

        def _run(command, **kwargs):
            # The module is the shared ``subprocess``: the HTTP client's own
            # platform probe goes through it too and is not frame work.
            if not command or command[0] != "ffmpeg":
                return real_run(command, **kwargs)
            during_frames.append(ask_slots._value)
            return ffmpeg.run(command, **kwargs)

        monkeypatch.setattr(frame_cache.subprocess, "run", _run)
        wire = Wire("openai_compatible")
        during_requests: list[int] = []

        def script(body):
            during_requests.append(ask_slots._value)
            if wire.has_image(body):
                return 400, ""
            return 200, wire.text_answer

        wire.script = script
        vision_client(llm, wire=wire)

        response = await ask(call, VIDEO)

        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert len(during_frames) == 8
        assert set(during_frames) == {MAX - 1}
        assert during_requests == [MAX - 1, MAX - 1]
        assert ask_slots._value == MAX

    async def test_a_full_semaphore_is_503_before_any_frame_work(
        self, db, addon, llm, call, ask_slots, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I6 §10: frames are prepared only under a slot."""
        _text_video(db, tmp_path)
        loaded = scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)
        for _ in range(MAX):
            await ask_slots.acquire()
        try:
            blocked = await ask(call, VIDEO)
        finally:
            for _ in range(MAX):
                ask_slots.release()

        assert blocked.status_code == 503
        assert ffmpeg.starts == []
        assert loaded == []
        assert wire.requests == []

        served = await ask(call, VIDEO)
        assert by_kind(served)["meta"]["image"] == "attached"
        assert sent_keys(wire) == [ms(t) for t in EVEN_80]
        assert ask_slots._value == MAX

    async def test_at_most_four_frames_are_prepared_at_once(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I6 §10: four preparations at most, every frame still sent."""
        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg(hold=0.2).install(monkeypatch)
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO, position=33.0)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert len(ffmpeg.starts) == 8
        assert 1 <= ffmpeg.peak <= 4
        assert len(wire.images) == 8

    async def test_a_disconnect_starts_no_more_frames_or_model_and_frees_the_slot(
        self, db, addon, llm, ask_slots, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 §2 I6: a closed tab stops the preparation and releases the slot."""
        from app.main import app

        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg(hang_all=True).install(monkeypatch)
        _, wire = vision_client(llm)

        try:
            await disconnect_during_preparation(
                app,
                path=f"/files/{VIDEO}/ask",
                body=json.dumps({"question": QUESTION}).encode(),
                headers=[
                    (b"content-type", b"application/json"),
                    (b"x-lit-drive", DRIVE.encode()),
                ],
                entered=ffmpeg.entered,
                release=ffmpeg.release_all,
                timeout=5.0,
            )
        except TimeoutError:
            pass
        assert ffmpeg.entered.is_set(), "no frame preparation started for the text video"
        started_at_disconnect = len(ffmpeg.starts)
        await asyncio.sleep(1.0)

        assert len(ffmpeg.starts) == started_at_disconnect
        assert wire.requests == []
        assert ask_slots._value == MAX


class TestTheWaitBound:
    async def test_frames_still_unready_at_45_seconds_give_way_to_the_text_answer(
        self, db, addon, llm, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-114 I4/I6: the request by 45 s, nothing started after, no 10 s of silence."""
        from app.main import app

        _text_video(db, tmp_path)
        scenes(monkeypatch, [], duration=80.0)
        ffmpeg = FFmpeg(hang_all=True).install(monkeypatch)
        wire = Wire("openai_compatible")
        requested_at: list[float] = []

        def script(body):
            requested_at.append(time.monotonic())
            return 200, wire.text_answer

        wire.script = script
        vision_client(llm, wire=wire)

        try:
            started, received, status = await timed_post(
                app, path=f"/files/{VIDEO}/ask", body={"question": QUESTION},
            )
            await asyncio.sleep(1.0)
        finally:
            ffmpeg.release_all.set()

        assert status == 200
        assert ffmpeg.starts, "no frame preparation started for the text video"
        assert requested_at, "no model request was sent"
        assert requested_at[0] - started <= 46.0
        assert [t for t in ffmpeg.start_times if t > requested_at[0]] == []
        assert wire.image_bodies == []
        assert len(wire.text_bodies) == 1
        body = b"".join(data for _, data in received).decode()
        assert '"unavailable"' in body
        assert "frames_unusable\"" not in body.split("event: done", 1)[-1]
        assert ": keepalive" in body

        first = _first_chunk(received)
        stamps = [started] + [t for t, data in received if data and t <= first]
        gaps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
        assert max(gaps) <= 10.5, gaps
