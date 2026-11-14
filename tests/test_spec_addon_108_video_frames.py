"""SPEC-ADDON-108: a video with no text chunk is asked about through its frames.

Every frame is read back from the request the provider received; its bytes
name the millisecond key it was cached under, so the sent images are the
chosen times, in the order they were sent.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

import app.config as config
import app.frame_cache as frame_cache
from app.routers import rag as rag_router
from tests.file_question_doubles import chunk
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    DRIVE,
    OTHER,
    VISION_MODEL,
    Wire,
    addon,
    ask,
    ask_slots,
    by_kind,
    call,
    db,
    disconnect_during_preparation,
    events,
    llm,
    openai_config,
    profiles,
    statements,
    vision_client,
    write_large_frame,
)
from tests.search_endpoint_db import patch_everywhere
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import (
    MARKER,
    FFmpeg,
    frame_bytes,
    listed_in_order,
    ms,
    scenes,
    seed_text_less_video,
    sent_keys,
    timed_post,
    user_prompt,
)
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    LOFT_MIME,
    YOUTUBE_URL,
    describe,
    needs_ffmpeg,
    youtube,
)

VIDEO = "vid000000108"
AUDIO = "aud000000108"
LOFT = "lft000000108"
OTHER_VIDEO = "vid000000118"
MAX = rag_router._MAX_CONCURRENT_ASK
QUESTION = "QUESTION-MARKER what happens here"


def _video(db, tmp_path, file_id=VIDEO, *, duration=80.0, **kwargs):
    seed_text_less_video(
        db, file_id, path=tmp_path / f"{file_id}.mp4", duration=duration, **kwargs,
    )


def _loft(db, tmp_path, file_id=LOFT, *, body, duration=60.0, raw=None):
    import app.database as database
    from app.models import IndexedFile

    path = tmp_path / f"{file_id}.loft"
    path.write_text(raw if raw is not None else json.dumps(body))
    db.file(file_id, mime=LOFT_MIME, file_path=str(path), duration=duration)
    with database.get_search_db() as s:
        s.query(IndexedFile).filter_by(file_id=file_id).update({"file_type": "video"})


async def _available(call, file_id) -> bool:
    response = await call("GET", f"/files/{file_id}/ask")
    assert response.status_code == 200, response.text
    return response.json()["available"]


def _answer(response) -> str:
    return "".join(
        data.get("delta", "") for kind, data in events(response) if kind == "answer_chunk"
    )


EVEN_80 = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestAvailability:
    async def test_a_text_less_video_is_available_only_when_some_profile_can_see(
        self, db, addon, call, tmp_path,
    ):
        """SPEC-ADDON-108 I1 (053 I6 revised): the same video flips with a vision profile."""
        _video(db, tmp_path)
        addon(profiles=profiles(vision_model=""))
        assert await _available(call, VIDEO) is False

        addon(profiles=profiles())
        assert await _available(call, VIDEO) is True

    async def test_audio_with_no_chunk_stays_unavailable_beside_a_text_less_video(
        self, db, addon, call, tmp_path,
    ):
        """SPEC-ADDON-108 I1: the widened rule is for video, not audio."""
        db.file(AUDIO, mime="audio/mpeg")
        _video(db, tmp_path)
        addon(profiles=profiles())

        assert await _available(call, AUDIO) is False
        assert await _available(call, VIDEO) is True

    @pytest.mark.parametrize("case", ["another drive", "inactive"])
    async def test_the_widened_rule_is_still_drive_scoped_and_active_only(
        self, case, db, addon, call, tmp_path,
    ):
        """SPEC-ADDON-108 Terms: a text-less video is an active row of the request drive."""
        if case == "another drive":
            _video(db, tmp_path, OTHER_VIDEO, drive=OTHER)
        else:
            _video(db, tmp_path, OTHER_VIDEO, active=False)
        _video(db, tmp_path)
        addon(profiles=profiles())

        assert await _available(call, OTHER_VIDEO) is False
        assert await _available(call, VIDEO) is True

    @pytest.mark.parametrize(
        ("body", "raw"),
        [
            ({"provider": "vimeo", "url": "https://vimeo.com/1"}, None),
            (None, "{not json"),
        ],
        ids=["other-provider", "unreadable"],
    )
    async def test_a_loft_is_text_less_only_when_it_names_youtube(
        self, body, raw, db, addon, call, tmp_path,
    ):
        """SPEC-ADDON-108 I1 / failure table step 1: never an error, ``false``."""
        _loft(db, tmp_path, "lft000000109", body=body, raw=raw)
        _loft(db, tmp_path, body={"provider": "youtube", "url": YOUTUBE_URL})
        addon(profiles=profiles())

        assert await _available(call, "lft000000109") is False
        assert await _available(call, LOFT) is True


# ---------------------------------------------------------------------------
# The route's checks, before a slot
# ---------------------------------------------------------------------------


class TestTheChecksBeforeASlot:
    async def test_a_blind_profile_is_409_video_needs_vision_without_a_slot(
        self, db, addon, llm, call, ask_slots, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 step 3: 409 ``video_needs_vision``; no slot, frame work or model."""
        _video(db, tmp_path)
        loaded = scenes(monkeypatch, [10.0], duration=80.0)
        ffmpeg = FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm, config=openai_config(vision_model=""))
        for _ in range(MAX):
            await ask_slots.acquire()
        try:
            response = await ask(call, VIDEO, question=QUESTION)
        finally:
            for _ in range(MAX):
                ask_slots.release()

        assert response.status_code == 409
        assert response.json()["detail"] == "video_needs_vision"
        assert wire.requests == []
        assert ffmpeg.starts == []
        assert loaded == []

    async def test_a_full_semaphore_is_503_before_any_frame_work(
        self, db, addon, llm, call, ask_slots, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 I7 / §10: frame work and candidate reads only under the slot."""
        _video(db, tmp_path)
        loaded = scenes(monkeypatch, [10.0], duration=80.0)
        ffmpeg = FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)
        for _ in range(MAX):
            await ask_slots.acquire()
        try:
            blocked = await ask(call, VIDEO, position=33.0)
        finally:
            for _ in range(MAX):
                ask_slots.release()

        assert blocked.status_code == 503
        assert ffmpeg.starts == []
        assert loaded == []

        served = await ask(call, VIDEO, position=33.0)
        assert by_kind(served)["meta"]["image"] == "attached"

    async def test_a_loft_of_another_provider_is_404_beside_an_answered_text_less_video(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 step 3: neither askable by text nor text-less is 404."""
        _loft(db, tmp_path, "lft000000109", body={"provider": "vimeo", "url": "x"})
        _video(db, tmp_path)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        refused = await ask(call, "lft000000109")
        assert refused.status_code == 404
        assert wire.requests == []

        answered = await ask(call, VIDEO)
        assert answered.status_code == 200
        assert len(wire.image_bodies) == 1


# ---------------------------------------------------------------------------
# Which frames are chosen
# ---------------------------------------------------------------------------


SELECTION = [
    pytest.param(
        [], 80.0, None, EVEN_80,
        id="no-candidates-evenly-spaced",
    ),
    pytest.param(
        [10.0 * (i + 1) for i in range(10)], 110.0, None,
        [10.0, 20.0, 40.0, 50.0, 60.0, 70.0, 90.0, 100.0],
        id="ten-thinned-to-eight",
    ),
    pytest.param(
        [10.0 * (i + 1) for i in range(10)], 110.0, 55.0,
        [10.0, 30.0, 40.0, 55.0, 60.0, 70.0, 90.0, 100.0],
        id="ten-thinned-to-seven-half-rounds-up",
    ),
    pytest.param(
        [5.5, 40.0], 80.0, None,
        [5.5, 15.0, 25.0, 35.0, 40.0, 45.0, 55.0, 65.0],
        id="fill-skips-near-and-stops-at-the-budget",
    ),
    pytest.param(
        [10.0, 20.0, 30.5, 31.0], 70.0, 30.0,
        [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 31.0, 35.0],
        id="within-a-second-of-the-position-removed-exactly-one-kept",
    ),
    pytest.param(
        [], 4.0, None,
        [0.25, 1.25, 2.25, 3.25],
        id="fill-runs-out-exactly-one-second-apart-kept",
    ),
    pytest.param(
        [], 70.0, 25.5,
        [5.0, 15.0, 25.5, 35.0, 45.0, 55.0, 65.0],
        id="fill-time-near-the-position-skipped",
    ),
    pytest.param(
        [12.0, 47.0], None, None,
        [12.0, 47.0],
        id="unknown-duration-no-fill",
    ),
    pytest.param(
        [], 80.0, 0.0, EVEN_80,
        id="position-zero-is-no-position",
    ),
]


class TestTheChosenFrames:
    @pytest.mark.parametrize(("selected", "duration", "position", "expected"), SELECTION)
    async def test_the_frames_sent_are_the_chosen_times_in_time_order(
        self, selected, duration, position, expected,
        db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 I3/I4: Terms steps 1-4, numbered in time order."""
        _video(db, tmp_path, duration=duration)
        scenes(monkeypatch, selected, duration=duration)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO, position=position)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert len(wire.requests) == 1
        assert sent_keys(wire) == [ms(t) for t in expected]

    async def test_position_zero_states_no_position(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 I3: ``position_seconds`` 0 sends the same prompt as none."""
        _video(db, tmp_path)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        await ask(call, VIDEO, question=QUESTION)
        await ask(call, VIDEO, question=QUESTION, position=0.0)

        assert len(wire.requests) == 2
        assert wire.prompts[0] == wire.prompts[1]
        assert wire.system_prompts[0] == wire.system_prompts[1]
        assert sent_keys(wire) == [ms(t) for t in EVEN_80] * 2

    @pytest.mark.parametrize("raises", ["_load_candidates", "select_candidates"])
    async def test_a_failed_candidate_read_falls_back_to_the_evenly_spaced_times(
        self, raises, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-108 failure table step 4: ``S`` starts empty and is logged."""
        _video(db, tmp_path)
        scenes(monkeypatch, [12.0, 47.0], duration=80.0)

        def _boom(*_args, **_kwargs):
            raise RuntimeError("vec_clip unreadable")

        patch_everywhere(monkeypatch, raises, _boom)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        with caplog.at_level("DEBUG"):
            response = await ask(call, VIDEO)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert sent_keys(wire) == [ms(t) for t in EVEN_80]
        assert "scene_candidates_failed" in caplog.text

    async def test_cached_frames_are_used_without_ffmpeg_and_misses_fill_the_large_cache(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 step 4: the ``large`` cache entry at each frame's millisecond key."""
        _video(db, tmp_path)
        scenes(monkeypatch, [12.3456], duration=80.0)
        for t in EVEN_80[:4]:
            write_large_frame(VIDEO, t, frame_bytes(ms(t)))
        ffmpeg = FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO)

        expected = sorted([12.3456, *EVEN_80[:7]])
        assert by_kind(response)["meta"]["image"] == "attached"
        assert sent_keys(wire) == [ms(t) for t in expected]
        assert sorted(ffmpeg.starts) == sorted(
            ms(t) for t in expected if t not in EVEN_80[:4]
        )
        cached = sorted(p.name for p in frame_cache.frame_cache_dir(VIDEO).iterdir())
        assert cached == sorted(f"{ms(t)}.large.webp" for t in expected)


# ---------------------------------------------------------------------------
# The request the vision model receives
# ---------------------------------------------------------------------------


class TestTheModelRequest:
    @pytest.mark.parametrize("provider", ["openai_compatible", "ollama"])
    async def test_all_frames_go_in_one_request_to_the_vision_model(
        self, provider, db, addon, llm, call, tmp_path, monkeypatch, statements,
    ):
        """SPEC-ADDON-108 §5 / I2: one request, the frames in order, the profile's vision model."""
        _video(db, tmp_path)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm, provider=provider)
        statements.reset()

        response = await ask(call, VIDEO, question=QUESTION)

        assert response.status_code == 200
        assert len(wire.requests) == 1
        body = wire.requests[0]
        assert body["model"] == VISION_MODEL
        assert sent_keys(wire) == [ms(t) for t in EVEN_80]
        assert statements.writes() == []
        meta = by_kind(response)["meta"]
        assert meta["image"] == "attached"
        assert meta["partial"] is False
        assert _answer(response) == Wire.vision_answer
        if provider == "openai_compatible":
            user = [m for m in body["messages"] if m.get("role") == "user"][-1]
            kinds = [part["type"] for part in user["content"]]
            assert kinds == ["text"] + ["image_url"] * 8
            assert all(
                part["image_url"]["url"].startswith("data:image/webp;base64,")
                for part in user["content"][1:]
            )

    async def test_the_prompt_numbers_each_frame_with_its_time_and_has_no_passages(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 step 5 / I5: ``[n]`` with ``m:ss`` truncated; no passages block."""
        _video(db, tmp_path, duration=None)
        scenes(monkeypatch, [5.9, 75.9, 3725.9], duration=None)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        await ask(call, VIDEO, question=QUESTION)

        prompt = user_prompt(wire)
        assert QUESTION in prompt
        assert "Passages from the file:" not in "\n".join(wire.prompts)
        assert listed_in_order(prompt, ["0:05", "1:15", "1:02:05"])
        assert "[4]" not in prompt

    async def test_a_failed_position_frame_is_stated_and_never_replaced(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 I3/I5: the position is stated; no frame carries its time."""
        _video(db, tmp_path, duration=70.0)
        FFmpeg(fail={ms(33.0)}).install(monkeypatch)
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO, position=33.0)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert sent_keys(wire) == [
            ms(t) for t in [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0]
        ]
        prompt = user_prompt(wire)
        assert "0:33" in prompt
        assert listed_in_order(
            prompt, ["0:05", "0:15", "0:25", "0:35", "0:45", "0:55", "1:05"],
        )
        assert "[8]" not in prompt

    async def test_a_kept_position_frame_is_listed_at_its_time(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 I3: the position frame takes its place in time order."""
        _video(db, tmp_path, duration=70.0)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        await ask(call, VIDEO, position=33.0)

        assert sent_keys(wire) == [
            ms(t) for t in [5.0, 15.0, 25.0, 33.0, 35.0, 45.0, 55.0, 65.0]
        ]
        assert listed_in_order(
            user_prompt(wire),
            ["0:05", "0:15", "0:25", "0:33", "0:35", "0:45", "0:55", "1:05"],
        )

    async def test_every_turn_prepares_and_sends_the_frames_again(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 step 5: the history is sent and the frames are chosen again."""
        _video(db, tmp_path)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        response = await call(
            "POST", f"/files/{VIDEO}/ask",
            json={
                "question": "and then what",
                "history": [{"question": "EARLIER-QUESTION", "answer": "EARLIER-ANSWER"}],
            },
        )

        assert by_kind(response)["meta"]["image"] == "attached"
        assert sent_keys(wire) == [ms(t) for t in EVEN_80]
        assert "EARLIER-QUESTION" in "\n".join(wire.prompts)


# ---------------------------------------------------------------------------
# Frames that cannot be read
# ---------------------------------------------------------------------------


class TestFramesThatFail:
    async def test_failed_frames_are_dropped_and_the_rest_are_numbered_from_one(
        self, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-108 failure table step 4: dropped, never replaced, ``partial`` false."""
        _video(db, tmp_path)
        FFmpeg(fail={ms(15.0)}, timeout={ms(55.0)}).install(monkeypatch)
        _, wire = vision_client(llm)

        with caplog.at_level("DEBUG"):
            response = await ask(call, VIDEO, question=QUESTION)

        kept = [5.0, 25.0, 35.0, 45.0, 65.0, 75.0]
        assert sent_keys(wire) == [ms(t) for t in kept]
        meta = by_kind(response)["meta"]
        assert meta["image"] == "attached"
        assert meta["partial"] is False
        assert listed_in_order(
            user_prompt(wire), ["0:05", "0:25", "0:35", "0:45", "1:05", "1:15"],
        )
        assert "[7]" not in user_prompt(wire)
        assert "frames_dropped" in caplog.text
        assert VIDEO in caplog.text

    async def test_a_scene_past_the_indexed_duration_is_dropped_without_extraction(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 failure table step 4: past the duration is a failed frame."""
        _video(db, tmp_path, duration=80.0)
        scenes(monkeypatch, [10.0, 90.0], duration=80.0)
        ffmpeg = FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        await ask(call, VIDEO)

        assert ms(90.0) not in ffmpeg.starts
        assert sent_keys(wire) == [
            ms(t) for t in [5.0, 10.0, 15.0, 25.0, 35.0, 45.0, 55.0]
        ]

    async def test_a_position_past_the_indexed_duration_is_left_out_but_stated(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 (071 I7 revised): no extraction, no substitute, budget stays 7."""
        _video(db, tmp_path, duration=70.0)
        ffmpeg = FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        response = await ask(call, VIDEO, position=90.0)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert ms(90.0) not in ffmpeg.starts
        assert sent_keys(wire) == [
            ms(t) for t in [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0]
        ]
        assert "1:30" in user_prompt(wire)

    @pytest.mark.parametrize(
        ("selected", "duration", "position", "fail_all"),
        [
            ([], None, None, False),
            ([], None, 0.0, False),
            ([], 80.0, None, True),
            ([], 70.0, 33.0, True),
        ],
        ids=["no-candidate-no-duration", "position-zero-no-duration",
             "every-frame-fails", "every-frame-fails-with-a-position"],
    )
    async def test_no_usable_frame_is_frames_unusable_with_no_model_call(
        self, selected, duration, position, fail_all,
        db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-108 failure table step 4: ``unavailable`` and ``frames_unusable``."""
        _video(db, tmp_path, duration=duration)
        scenes(monkeypatch, selected, duration=duration)
        ffmpeg = FFmpeg()
        if fail_all:
            ffmpeg.fail = {ms(t) for t in [*EVEN_80, 33.0, 5.0, 15.0, 25.0, 35.0,
                                           45.0, 55.0, 65.0]}
        ffmpeg.install(monkeypatch)
        _, wire = vision_client(llm)

        with caplog.at_level("DEBUG"):
            response = await ask(call, VIDEO, position=position)

        assert response.status_code == 200
        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert by_kind(response)["done"]["error"] == "frames_unusable"
        assert _answer(response) == ""
        assert wire.requests == []
        assert "frames_unusable" in caplog.text
        assert VIDEO in caplog.text


# ---------------------------------------------------------------------------
# The provider's answer
# ---------------------------------------------------------------------------


def _refusing(status):
    def script(body):
        return status, ""
    return script


class TestTheProvidersAnswer:
    @pytest.mark.parametrize("provider", ["openai_compatible", "ollama"])
    @pytest.mark.parametrize("status", [400, 404])
    async def test_a_refusal_is_frames_rejected_and_writes_no_verdict(
        self, status, provider, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 failure table step 5 / I6: no retry, no text answer, no verdict."""
        from app.llm import reset_vision_capability_cache
        from app.llm_routing import any_profile_can_see, profile_can_see

        reset_vision_capability_cache()
        _video(db, tmp_path)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm, provider=provider, script=_refusing(status))
        assert profile_can_see(llm.result) is True
        assert any_profile_can_see(config.settings) is True

        response = await ask(call, VIDEO)

        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert by_kind(response)["done"]["error"] == "frames_rejected"
        assert _answer(response) == ""
        assert len(wire.requests) == 1
        assert wire.text_bodies == []

        assert profile_can_see(llm.result) is True
        assert any_profile_can_see(config.settings) is True
        assert await _available(call, VIDEO) is True
        again = await ask(call, VIDEO)
        assert again.status_code == 200
        assert len(wire.image_bodies) == 2

    @pytest.mark.parametrize(
        "how", [413, 422, 500, "timeout", "empty"],
        ids=["413", "422", "500", "timeout", "empty-answer"],
    )
    async def test_any_other_failure_is_the_generic_generation_failure(
        self, how, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 failure table step 5: not ``frames_rejected``, not retried."""
        _video(db, tmp_path)
        FFmpeg().install(monkeypatch)

        def script(body):
            if how == "timeout":
                return httpx.ReadTimeout("timed out")
            if how == "empty":
                return 200, ""
            return how, ""

        _, wire = vision_client(llm, script=script)

        response = await ask(call, VIDEO)

        error = by_kind(response)["done"].get("error")
        assert error
        assert error not in ("frames_rejected", "frames_unusable")
        assert len(wire.requests) == 1
        assert wire.text_bodies == []


class _CutAfterFirstToken(httpx.AsyncByteStream):
    async def __aiter__(self):
        first = {
            "id": "1", "object": "chat.completion.chunk", "created": 0,
            "model": VISION_MODEL,
            "choices": [{
                "index": 0, "delta": {"role": "assistant", "content": "PARTIAL-TEXT"},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(first)}\n\n".encode()
        raise httpx.ReadError("connection lost")

    async def aclose(self) -> None:
        return None


class TestAFailureAfterTheFirstToken:
    async def test_the_text_received_is_kept_and_done_carries_an_error(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 failure table step 5: as today, the partial text stays."""
        from openai import AsyncOpenAI

        from app.llm import LLMClient
        from tests.file_question_vision_doubles import resolved

        _video(db, tmp_path)
        FFmpeg().install(monkeypatch)
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"},
                stream=_CutAfterFirstToken(),
            )

        settings = openai_config()
        client = LLMClient(settings)
        client._client = AsyncOpenAI(  # noqa: SLF001 - the test replaces the transport
            base_url=settings.base_url, api_key="test-key", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        llm.serve(resolved(client, settings))

        response = await ask(call, VIDEO)

        assert _answer(response) == "PARTIAL-TEXT"
        error = by_kind(response)["done"].get("error")
        assert error
        assert error not in ("frames_rejected", "frames_unusable")
        assert len(requests) == 1
        assert by_kind(response)["meta"]["image"] == "attached"


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------


class TestWhenTextArrives:
    async def test_the_next_question_after_a_transcript_lands_takes_the_text_path(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 §3 / I1: frames while text-less; passages and no frame after."""
        _video(db, tmp_path)
        scenes(monkeypatch, [12.0, 47.0], duration=80.0)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        before = await ask(call, VIDEO)
        assert by_kind(before)["meta"]["image"] == "attached"
        assert len(wire.image_bodies) == 1

        passage = chunk("TRANSCRIPT-PASSAGE", 40)
        db.on(VIDEO).transcript(passage, start=0.0)
        after = await ask(call, VIDEO)

        assert by_kind(after)["meta"]["image"] == "none"
        assert len(wire.image_bodies) == 1
        assert passage in wire.prompts[-1]

    async def test_a_video_with_an_empty_chunk_is_not_text_less(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 Terms: chunk rows of empty text keep today's 404, never frames."""
        _video(db, tmp_path)
        _video(db, tmp_path, OTHER_VIDEO)
        db.on(OTHER_VIDEO).transcript("", start=0.0, cos=None)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm)

        framed = await ask(call, VIDEO)
        empty = await ask(call, OTHER_VIDEO)

        assert by_kind(framed)["meta"]["image"] == "attached"
        assert empty.status_code == 404
        assert len(wire.requests) == 1


# ---------------------------------------------------------------------------
# Concurrency, the slot and a disconnect
# ---------------------------------------------------------------------------


async def _until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.02)


async def _settle(ffmpeg: FFmpeg, quiet: float = 0.4) -> None:
    seen = -1
    while seen != len(ffmpeg.starts):
        seen = len(ffmpeg.starts)
        await asyncio.sleep(quiet)


class TestPreparation:
    async def test_at_most_four_run_at_once_position_first_then_time_order(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 I7: at most 4 of the question's frames at once, in the stated order."""
        _video(db, tmp_path, duration=None)
        scenes(monkeypatch, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], duration=None)
        ffmpeg = FFmpeg(hang_all=True).install(monkeypatch)
        _, wire = vision_client(llm)
        expected = [ms(t) for t in [75.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]]

        task = asyncio.create_task(ask(call, VIDEO, position=75.0))
        try:
            await _until(lambda: ffmpeg.starts)
            released: set[int] = set()
            while len(released) < len(expected):
                await _settle(ffmpeg)
                started = list(ffmpeg.starts)
                assert ffmpeg.peak <= 4
                assert set(started) == set(expected[: len(started)])
                running = [k for k in started if k not in released]
                assert running, "nothing running and not every frame started"
                ffmpeg.release(running[0])
                released.add(running[0])
            response = await asyncio.wait_for(task, 20)
        finally:
            ffmpeg.release_all.set()
            if not task.done():
                task.cancel()

        assert ffmpeg.peak <= 4
        assert by_kind(response)["meta"]["image"] == "attached"
        assert sent_keys(wire) == sorted(expected)

    async def test_a_disconnect_starts_no_more_frames_or_model_and_frees_the_slot(
        self, db, addon, llm, call, tmp_path, monkeypatch, ask_slots,
    ):
        """SPEC-ADDON-108 I7 / failure table 4-5: stop waiting, start nothing, release."""
        from app.main import app

        _video(db, tmp_path)
        ffmpeg = FFmpeg(hang_all=True).install(monkeypatch)
        _, wire = vision_client(llm)

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
        )
        started_at_disconnect = len(ffmpeg.starts)
        await asyncio.sleep(1.0)

        assert len(ffmpeg.starts) == started_at_disconnect
        assert wire.requests == []
        assert ask_slots._value == MAX


class TestTheWaitBound:
    async def test_the_route_stops_waiting_by_45_seconds_and_keeps_the_line_alive(
        self, db, addon, llm, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-108 I8: ended by 45 s, nothing started after, no 10 s of silence."""
        from app.main import app

        _video(db, tmp_path)
        ffmpeg = FFmpeg(hang_all=True).install(monkeypatch)
        _, wire = vision_client(llm)

        try:
            started, received, status = await timed_post(
                app, path=f"/files/{VIDEO}/ask", body={"question": QUESTION},
            )
            ended = received[-1][0] if received else time.monotonic()
            await asyncio.sleep(1.0)
            late_starts = [t for t in ffmpeg.start_times if t > ended]
        finally:
            ffmpeg.release_all.set()

        assert status == 200
        body = b"".join(chunk_ for _, chunk_ in received).decode()
        assert ended - started <= 46.0
        assert late_starts == []
        assert wire.requests == []
        assert '"frames_unusable"' in body
        assert ": keepalive" in body

        stamps = [started] + [t for t, data in received if data]
        gaps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
        assert max(gaps) <= 10.5, gaps


# ---------------------------------------------------------------------------
# A YouTube reference with no text
# ---------------------------------------------------------------------------


@needs_ffmpeg
class TestAYouTubeLoftWithNoText:
    async def test_it_is_answered_from_frames_including_the_position_frame(
        self, db, addon, llm, call, tmp_path, youtube,
    ):
        """SPEC-ADDON-108 (077 I6 revised): a text-less YouTube ``.loft`` takes the frames path."""
        _loft(db, tmp_path, body={"provider": "youtube", "url": YOUTUBE_URL}, duration=60.0)
        youtube.set_ytdlp(describe=describe([youtube.format("f720", 720)]))
        addon(profiles=profiles())
        _, wire = vision_client(llm)

        assert await _available(call, LOFT) is True
        response = await ask(call, LOFT, position=12.0)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert len(wire.requests) == 1
        cached = frame_cache.frame_cache_path(LOFT, 12.0, "large")
        assert cached.exists()
        from tests.video_frames_doubles import decoded

        assert cached.read_bytes() in [decoded(p) for p in wire.images]


# ---------------------------------------------------------------------------
# What is logged
# ---------------------------------------------------------------------------


class TestWhatIsLogged:
    async def test_no_question_answer_or_frame_byte_is_logged(
        self, db, addon, llm, call, tmp_path, monkeypatch, caplog,
    ):
        """SPEC-ADDON-108 §8 (071 I8 revised): counts and ids only."""
        _video(db, tmp_path)
        FFmpeg(fail={ms(15.0)}).install(monkeypatch)
        _, wire = vision_client(llm)

        with caplog.at_level("DEBUG"):
            response = await ask(call, VIDEO, question=QUESTION)

        assert by_kind(response)["meta"]["image"] == "attached"
        logged = caplog.text
        assert "frames_dropped" in logged
        assert "QUESTION-MARKER" not in logged
        assert Wire.vision_answer not in logged
        assert MARKER.decode() not in logged
