"""SPEC-ADDON-071 I3/I4/I6/I8/I9: the stream around an image.

The failure table, the slot, the retry and what may be logged or written.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from pathlib import Path

import pytest

import app.config as config
import app.frame_cache as frame_cache
from app.routers import rag as rag_router
from tests.file_question_doubles import chunk
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    DRIVE,
    PICTURE_BYTES,
    TEXT_MODEL,
    Wire,
    addon,
    ask,
    ask_slots,
    by_kind,
    call,
    db,
    disconnect_during_preparation,
    events,
    failing,
    ffmpeg_fails,
    ffmpeg_writes,
    llm,
    mentioned,
    openai_config,
    record_vision_unsupported,
    refusing,
    seed_audio,
    seed_document,
    seed_picture,
    seed_video,
    statements,
    timing_out,
    toward,
    vision_client,
)
from tests.search_endpoint_db import patch_everywhere
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture

PICTURE = "pic000000051"
TEXT = chunk("the receipt says forty two", 40)
VIDEO = "vid000000051"
AUDIO = "aud000000051"
PDF = "pdf000000051"
MAX = rag_router._MAX_CONCURRENT_ASK

# tests/test_file_question.py::test_rag_off_answers_as_ask_does pins this status.
RAG_OFF_STATUS = 400


def _picture(db, file_id=PICTURE, *, text=TEXT, data=PICTURE_BYTES) -> Path:
    path = Path(config.settings.intelligence_data_dir).parent / "pictures" / f"{file_id}.png"
    seed_picture(db, file_id, text=text, path=path, data=data)
    return path


def _answer(response) -> str:
    return "".join(
        data["delta"] for kind, data in events(response) if kind == "answer_chunk"
    )


class TestThePolicyGate:
    @pytest.mark.parametrize(
        ("case", "expected"), [("skip", 400), ("defer", 503)],
        ids=["denied", "unreadable"],
    )
    async def test_a_refused_policy_reads_no_picture(
        self, case, expected, db, addon, llm, call, monkeypatch,
    ):
        """SPEC-ADDON-071 I2: no policy answer means neither image nor text."""
        _picture(db)
        getattr(llm, case)()

        def _trap(*_args, **_kwargs):
            raise AssertionError("no image may be read before the policy answers")

        patch_everywhere(monkeypatch, "_load_image_bytes", _trap)
        patch_everywhere(monkeypatch, "_preprocess_image", _trap)

        response = await ask(call, PICTURE)

        assert response.status_code == expected

        vision_client(llm)
        served = await ask(call, PICTURE)
        assert by_kind(served)["meta"]["image"] == "attached"


class TestTheRefusedImage:
    async def test_the_retry_is_text_only_and_reuses_the_same_passages(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 I4: one text retry, same passages, the text model."""
        _picture(db)
        wire = Wire("openai_compatible")
        wire.script = refusing(wire)
        vision_client(llm, wire=wire)

        response = await ask(call, PICTURE)

        assert response.status_code == 200
        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert _answer(response) == "TEXT ONLY ANSWER"
        assert len(wire.text_bodies) == 1
        retry = wire.text_bodies[0]
        assert retry["model"] == TEXT_MODEL
        assert not wire.has_image(retry)
        assert TEXT in wire.text_of(wire.image_bodies[0])
        assert TEXT in wire.text_of(retry)
        assert "image" not in wire.text_of(retry).lower()

    async def test_a_text_less_picture_is_never_answered_after_a_refusal(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 I6: no chunk and no image ends as ``image_unusable``."""
        _picture(db, text=None)
        wire = Wire("openai_compatible")
        wire.script = refusing(wire)
        vision_client(llm, wire=wire)

        response = await ask(call, PICTURE)

        assert response.status_code == 200
        assert by_kind(response)["meta"]["image"] == "unavailable"
        assert by_kind(response)["done"]["error"] == "image_unusable"
        assert _answer(response) == ""
        assert wire.text_bodies == []


class TestTheSamePassages:
    async def test_the_ranked_passages_are_the_same_with_and_without_an_image(
        self, db, addon, llm, call, monkeypatch, tmp_path,
    ):
        """SPEC-ADDON-071 I4: the image does not change the passage selection."""
        addon(budget=80)
        first = chunk("first passage", 50)
        second = chunk("second passage", 50)
        seed_picture(
            db, PICTURE, text=None,
            path=tmp_path / "pictures" / "selection.png",
        )
        db.document(first, chunk_index=0, cos=0.2)
        db.document(second, chunk_index=1, cos=0.9)
        patch_everywhere(
            monkeypatch, "embed_query", lambda _text: toward(db.db.dim, 1.0),
        )

        _, seeing = vision_client(llm)
        with_image = await ask(call, PICTURE)
        assert by_kind(with_image)["meta"]["image"] == "attached"
        assert mentioned(seeing.prompts[-1], [first, second]) == [second]

        _, blind = vision_client(llm, config=openai_config(vision_model=""))
        without_image = await ask(call, PICTURE)
        assert by_kind(without_image)["meta"]["image"] == "none"
        assert mentioned(blind.prompts[-1], [first, second]) == [second]


class TestTheOtherGenerationFailures:
    async def test_a_5xx_is_not_retried_and_keeps_the_text_answer_for_a_file_with_text(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 I4/I6: a 5xx is a generation failure, not a refusal."""
        _picture(db)
        wire = Wire("openai_compatible")
        wire.script = failing(wire)
        vision_client(llm, wire=wire)

        response = await ask(call, PICTURE)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert by_kind(response)["done"]["error"]
        assert _answer(response) == ""
        assert wire.text_bodies == []

    async def test_a_5xx_leaves_a_text_less_picture_with_the_generic_failure(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 I6: a 5xx with no text is not ``image_unusable``."""
        _picture(db, text=None)
        wire = Wire("openai_compatible")
        wire.script = failing(wire)
        vision_client(llm, wire=wire)

        response = await ask(call, PICTURE)

        assert by_kind(response)["done"]["error"]
        assert by_kind(response)["done"]["error"] != "image_unusable"
        assert _answer(response) == ""
        assert wire.text_bodies == []

    async def test_a_timeout_is_not_retried(self, db, addon, llm, call):
        """SPEC-ADDON-071 I4: a timeout is not the image refusal's retry."""
        _picture(db)
        wire = Wire("openai_compatible")
        wire.script = timing_out(wire)
        vision_client(llm, wire=wire)

        response = await ask(call, PICTURE)

        assert by_kind(response)["done"]["error"]
        assert wire.text_bodies == []


class TestWhatTheFailureTableReports:
    async def test_a_profile_that_cannot_see_is_silent(self, db, addon, llm, call):
        """SPEC-ADDON-071 I3: no vision model means ``none``, not ``unavailable``."""
        _picture(db)
        _, wire = vision_client(
            llm, config=openai_config(vision_model=""),
        )

        response = await ask(call, PICTURE)

        assert by_kind(response)["meta"]["image"] == "none"
        assert wire.image_bodies == []
        assert _answer(response) == "TEXT ONLY ANSWER"
        assert "image" not in wire.prompts[-1].lower()
        assert "exact" not in wire.prompts[-1].lower()

    async def test_a_recorded_vision_unsupported_profile_is_silent(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 I3: the description path's verdict suppresses the attempt."""
        _picture(db)
        wire = Wire("openai_compatible")
        wire.script = refusing(wire)
        client, wire = vision_client(llm, wire=wire)
        await record_vision_unsupported(client)
        wire.requests.clear()
        wire.paths.clear()

        response = await ask(call, PICTURE)

        assert by_kind(response)["meta"]["image"] == "none"
        assert wire.image_bodies == []
        assert _answer(response) == "TEXT ONLY ANSWER"
        assert "image" not in wire.prompts[-1].lower()
        assert "exact" not in wire.prompts[-1].lower()

    async def test_an_audio_file_gets_no_image(self, db, addon, llm, call, tmp_path):
        """SPEC-ADDON-071 I3: an image applies to nothing but a picture or a video."""
        seed_audio(
            db, AUDIO, text=chunk("spoken words", 30),
            path=tmp_path / "audio.mp3",
        )
        _, wire = vision_client(llm)

        response = await ask(call, AUDIO, position=65.0)

        assert by_kind(response)["meta"]["image"] == "none"
        assert wire.image_bodies == []
        assert "1:05" in wire.prompts[-1]

    async def test_a_pdf_gets_no_image(self, db, addon, llm, call, tmp_path):
        """SPEC-ADDON-071 I3: a document is not a picture for this route."""
        seed_document(
            db, PDF, text=chunk("page words", 30), path=tmp_path / "doc.pdf",
        )
        _, wire = vision_client(llm)

        response = await ask(call, PDF, position=65.0)

        assert by_kind(response)["meta"]["image"] == "none"
        assert wire.image_bodies == []


class TestThePictureNeedsVisionGate:
    async def test_a_text_less_picture_and_a_blind_profile_is_409_before_a_slot(
        self, db, addon, llm, call, ask_slots,
    ):
        """SPEC-ADDON-071 I6: 409 ``picture_needs_vision``, no slot, no model."""
        _picture(db, text=None)
        _, wire = vision_client(llm, config=openai_config(vision_model=""))
        for _ in range(MAX):
            await ask_slots.acquire()
        try:
            response = await ask(call, PICTURE)
            remaining = ask_slots._value
        finally:
            for _ in range(MAX):
                ask_slots.release()

        assert response.status_code == 409
        assert response.json()["detail"] == "picture_needs_vision"
        assert remaining == 0
        assert wire.requests == []

    async def test_a_recorded_blind_profile_is_409_for_a_text_less_picture(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 I6: the verdict makes the profile unable to see."""
        _picture(db, text=None)
        wire = Wire("openai_compatible")
        wire.script = refusing(wire)
        client, wire = vision_client(llm, wire=wire)
        await record_vision_unsupported(client)
        wire.requests.clear()

        response = await ask(call, PICTURE)

        assert response.status_code == 409
        assert response.json()["detail"] == "picture_needs_vision"
        assert wire.requests == []

    @pytest.mark.parametrize(
        ("case", "expected"),
        [
            ("rag off", RAG_OFF_STATUS),
            ("profile unavailable", 400),
            ("file missing", 404),
        ],
        ids=["rag-off", "profile-unavailable", "file-missing"],
    )
    async def test_the_checks_run_before_the_409(
        self, case, expected, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 step 3: rag, profile, file, then ``picture_needs_vision``."""
        if case == "rag off":
            addon(rag=False)
        if case == "profile unavailable":
            llm.skip()
            response = await ask(call, PICTURE, profile="chosen")
        else:
            vision_client(llm)
            response = await ask(call, PICTURE)

        assert response.status_code == expected
        if case == "profile unavailable":
            assert response.json()["detail"] == "profile_unavailable"
        if case == "rag off":
            addon(rag=True)

        _picture(db, text=None)
        vision_client(llm, config=openai_config(vision_model=""))
        blocked = await ask(call, PICTURE)
        assert blocked.status_code == 409
        assert blocked.json()["detail"] == "picture_needs_vision"


class TestTheSlot:
    async def test_the_picture_preparation_runs_under_the_slot(
        self, db, addon, llm, call, ask_slots, monkeypatch,
    ):
        """SPEC-ADDON-071 I9: one slot is held while the picture is re-encoded."""
        _picture(db)
        seen: list[int] = []

        def _preprocess(raw, mime):
            seen.append(ask_slots._value)
            return b"REENCODED", "image/jpeg"

        patch_everywhere(monkeypatch, "_preprocess_image", _preprocess)
        vision_client(llm)

        await ask(call, PICTURE)

        assert seen == [MAX - 1]

    async def test_the_frame_preparation_runs_under_the_slot(
        self, db, addon, llm, call, ask_slots, monkeypatch, tmp_path,
    ):
        """SPEC-ADDON-071 I9: one slot is held while the frame is extracted."""
        seed_video(
            db, VIDEO, text=chunk("video", 30),
            path=tmp_path / "video.mp4", duration=None,
        )
        seen: list[int] = []

        def _extract(*args, **kwargs):
            seen.append(ask_slots._value)
            cache_path = kwargs.get("cache_path")
            if cache_path is None:
                cache_path = args[1]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(b"FRAME")

        patch_everywhere(monkeypatch, "extract_frame_to_cache", _extract)
        vision_client(llm)

        await ask(call, VIDEO, position=5.0)

        assert seen == [MAX - 1]

    async def test_the_retry_runs_under_the_same_slot(
        self, db, addon, llm, call, ask_slots,
    ):
        """SPEC-ADDON-071 I9: the text retry does not take a second slot."""
        _picture(db)
        wire = Wire("openai_compatible")

        def script(body):
            if wire.has_image(body):
                return 400, ""
            wire.text_request_slots.append(ask_slots._value)
            return 200, "TEXT ONLY ANSWER"

        wire.script = script
        vision_client(llm, wire=wire)

        response = await ask(call, PICTURE)

        assert response.status_code == 200
        assert len(wire.image_bodies) == 1
        assert len(wire.text_bodies) == 1
        assert wire.text_request_slots == [MAX - 1]

    @pytest.mark.parametrize("kind", ["picture", "video"])
    async def test_a_full_semaphore_is_503_before_any_preparation(
        self, kind, db, addon, llm, call, ask_slots, monkeypatch, tmp_path,
    ):
        """SPEC-ADDON-071 I9: 503 before ffmpeg or a re-encode."""
        if kind == "picture":
            _picture(db)
            file_id = PICTURE
        else:
            seed_video(
                db, VIDEO, text=chunk("video", 30),
                path=tmp_path / "video.mp4", duration=None,
            )
            file_id = VIDEO

        def _trap(*_args, **_kwargs):
            raise AssertionError("no preparation may run before the slot")

        patch_everywhere(monkeypatch, "_preprocess_image", _trap)
        patch_everywhere(monkeypatch, "extract_frame_to_cache", _trap)
        _, wire = vision_client(llm)
        for _ in range(MAX):
            await ask_slots.acquire()
        try:
            response = await ask(call, file_id, position=5.0 if kind == "video" else None)
        finally:
            for _ in range(MAX):
                ask_slots.release()

        assert response.status_code == 503

        if kind == "picture":
            patch_everywhere(monkeypatch, "_preprocess_image", lambda raw, mime: (raw, mime))
        else:
            def _write(*args, **kwargs):
                cache_path = kwargs.get("cache_path") or args[1]
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(b"FRAME")

            patch_everywhere(monkeypatch, "extract_frame_to_cache", _write)
        served = await ask(call, file_id, position=5.0 if kind == "video" else None)
        assert by_kind(served)["meta"]["image"] == "attached"

    async def test_a_file_that_cannot_be_asked_about_is_404_before_a_slot(
        self, db, addon, llm, call, ask_slots,
    ):
        """SPEC-ADDON-071 I9: material is read before a slot is taken."""
        vision_client(llm)
        for _ in range(MAX):
            await ask_slots.acquire()
        try:
            response = await ask(call, "fil000000000")
        finally:
            for _ in range(MAX):
                ask_slots.release()

        assert response.status_code == 404

        _picture(db)
        served = await ask(call, PICTURE)
        assert by_kind(served)["meta"]["image"] == "attached"

    async def test_the_slot_returns_after_an_image_answer(
        self, db, addon, llm, call, ask_slots,
    ):
        """SPEC-ADDON-071 I9: the end of the stream frees the one slot."""
        _picture(db)
        _, wire = vision_client(llm)

        await ask(call, PICTURE)

        assert wire.image_bodies
        assert ask_slots._value == MAX

    async def test_the_slot_returns_after_an_image_failure(
        self, db, addon, llm, call, ask_slots,
    ):
        """SPEC-ADDON-071 I9: a failed image stream frees the slot too."""
        _picture(db)
        wire = Wire("openai_compatible")
        wire.script = failing(wire)
        vision_client(llm, wire=wire)

        await ask(call, PICTURE)

        assert wire.image_bodies
        assert ask_slots._value == MAX

    async def test_a_disconnect_during_preparation_releases_the_slot(
        self, db, addon, llm, call, ask_slots, monkeypatch,
    ):
        """SPEC-ADDON-071 I9: a closed tab cancels the preparation and frees the slot."""
        from app.main import app

        _picture(db)
        entered = threading.Event()
        release = threading.Event()

        def _blocking(raw, mime):
            entered.set()
            release.wait(5)
            return b"REENCODED", "image/jpeg"

        patch_everywhere(monkeypatch, "_preprocess_image", _blocking)
        vision_client(llm)

        try:
            await disconnect_during_preparation(
                app,
                path=f"/files/{PICTURE}/ask",
                body=json.dumps({"question": "what is this"}).encode(),
                headers=[
                    (b"x-lit-drive", DRIVE.encode()),
                    (b"content-type", b"application/json"),
                ],
                entered=entered,
                release=release,
                timeout=3.0,
            )
        except TimeoutError:
            pass

        assert entered.is_set(), "the picture preparation never started"
        assert ask_slots._value == MAX


class TestWhatIsNotStoredOrLogged:
    async def test_the_question_answer_and_image_are_not_logged(
        self, db, addon, llm, call, caplog,
    ):
        """SPEC-ADDON-071 I8: no question, answer or image byte in the log."""
        question = "what does the secret receipt say"
        data = b"PICTURE-IMAGE-MARKER"
        _picture(db, data=data)
        encoded = base64.b64encode(data).decode("ascii")
        _, wire = vision_client(llm, script=lambda body: (200, "SECRET-ANSWER"))
        caplog.set_level(logging.DEBUG)

        await ask(call, PICTURE, question=question)

        assert wire.image_bodies
        for record in caplog.records:
            message = record.getMessage()
            assert question not in message
            assert "SECRET-ANSWER" not in message
            assert "PICTURE-IMAGE-MARKER" not in message
            assert encoded not in message

    async def test_a_picture_question_transfers_nothing_to_a_store(
        self, db, addon, llm, call, statements,
    ):
        """SPEC-ADDON-071 I8: no row for a picture question."""
        settings = addon()
        _picture(db)
        _, wire = vision_client(llm)
        statements.reset()

        await ask(call, PICTURE)

        assert wire.image_bodies
        assert statements.writes() == []

    async def test_a_frame_question_writes_only_the_large_cache_entry(
        self, db, addon, llm, call, monkeypatch, tmp_path,
    ):
        """SPEC-ADDON-071 I8: the one file is the frame cache's own entry."""
        settings = addon()
        seed_video(
            db, VIDEO, text=chunk("video", 30),
            path=tmp_path / "video.mp4", duration=None,
        )
        monkeypatch.setattr(
            frame_cache.subprocess, "run", ffmpeg_writes(b"FRESH-FRAME"),
        )
        vision_client(llm)
        data_dir = Path(settings.intelligence_data_dir)
        before = {p for p in data_dir.rglob("*") if p.is_file()} if data_dir.exists() else set()

        await ask(call, VIDEO, position=5.0)

        after = {p for p in data_dir.rglob("*") if p.is_file()}
        new = after - before
        assert len(new) == 1
        entry = new.pop()
        assert entry.name == "5000.large.webp"
        assert entry.parent == data_dir / "frames" / VIDEO

    @pytest.mark.parametrize(
        ("failure", "slug"),
        [
            ("picture unreadable", "picture_unreadable"),
            ("frame failed", "frame_failed"),
            ("image rejected", "image_rejected"),
        ],
        ids=["picture-unreadable", "frame-failed", "image-rejected"],
    )
    async def test_the_failure_class_and_file_id_are_logged(
        self, failure, slug, db, addon, llm, call, caplog, monkeypatch, tmp_path,
    ):
        """SPEC-ADDON-071 I8: the class with the file id, and nothing else."""
        if failure == "picture unreadable":
            file_id = PICTURE
            missing = Path(settings_dir()) / "missing.png"
            db.file(PICTURE, mime="image/png", file_path=str(missing), file_size=10)
            db.document(TEXT, chunk_index=0)
            vision_client(llm)
        elif failure == "frame failed":
            file_id = VIDEO
            seed_video(
                db, VIDEO, text=chunk("video", 30),
                path=tmp_path / "video.mp4", duration=None,
            )
            monkeypatch.setattr(frame_cache.subprocess, "run", ffmpeg_fails())
            vision_client(llm)
        else:
            file_id = PICTURE
            _picture(db)
            wire = Wire("openai_compatible")
            wire.script = refusing(wire)
            vision_client(llm, wire=wire)
        caplog.set_level(logging.DEBUG)

        await ask(call, file_id, position=5.0 if failure == "frame failed" else None)

        messages = [record.getMessage() for record in caplog.records]
        assert any(slug in message and file_id in message for message in messages)


def settings_dir() -> str:
    return str(Path(config.settings.intelligence_data_dir).parent)
