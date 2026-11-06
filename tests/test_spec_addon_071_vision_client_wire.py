"""SPEC-ADDON-071 I2: both client classes carry the image in their own wire shape.

The addon's real client classes are used with their HTTP layer replaced, so
the request is the one the provider would have received.
"""

from __future__ import annotations

import base64
from pathlib import Path

import app.config as config
from tests.file_question_doubles import chunk
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    DRIVE,
    VISION_MODEL,
    addon,
    ask,
    call,
    db,
    llm,
    openai_config,
    seed_picture,
    vision_client,
)
from tests.file_question_vision_doubles import PICTURE_BYTES
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture

PICTURE = "pic000000061"
TEXT = chunk("receipt amount", 30)


def _picture(db):
    path = Path(config.settings.intelligence_data_dir).parent / "pictures" / "wire.png"
    seed_picture(db, PICTURE, text=TEXT, path=path)


def _payload(payload: str) -> bytes:
    if payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


class TestTheOpenAICompatibleWire:
    async def test_the_image_is_an_image_url_block_with_the_vision_model(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 I2: an OpenAI-compatible image_url data URL to vision_model."""
        _picture(db)
        _, wire = vision_client(llm)

        await ask(call, PICTURE)

        body = wire.image_bodies[0]
        assert wire.paths[0].endswith("/chat/completions")
        assert body["model"] == VISION_MODEL
        user = next(m for m in body["messages"] if m["role"] == "user")
        assert isinstance(user["content"], list)
        images = [
            part for part in user["content"] if part.get("type") == "image_url"
        ]
        assert len(images) == 1
        assert images[0]["image_url"]["url"].startswith("data:image/")
        assert _payload(images[0]["image_url"]["url"]) == PICTURE_BYTES
        assert any(
            part.get("type") == "text" and TEXT in part.get("text", "")
            for part in user["content"]
        )


class TestTheOllamaWire:
    async def test_the_image_is_in_the_native_chat_images_list(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 I2: ollama's /api/chat carries the image in ``images``."""
        _picture(db)
        _, wire = vision_client(llm, provider="ollama")

        await ask(call, PICTURE)

        body = wire.image_bodies[0]
        assert wire.paths[0].endswith("/api/chat")
        assert body["model"] == VISION_MODEL
        user = next(m for m in body["messages"] if m["role"] == "user")
        assert isinstance(user["content"], str)
        assert TEXT in user["content"]
        assert len(user["images"]) == 1
        assert user["images"][0] == base64.b64encode(PICTURE_BYTES).decode("ascii")
        assert "data:" not in user["images"][0]


class TestTheModelChoice:
    async def test_a_chosen_profile_is_never_replaced(self, db, addon, llm, call):
        """SPEC-ADDON-071 I2: the viewer's profile, and its vision model, are used."""
        _picture(db)
        _, wire = vision_client(
            llm, name="chosen",
            config=openai_config(vision_model="chosen-vision"),
        )

        await ask(call, PICTURE, profile="chosen")

        assert llm.asked == [(DRIVE, "rag", "chosen")]
        assert wire.image_bodies[0]["model"] == "chosen-vision"

    async def test_the_vision_model_answers_only_when_an_image_is_attached(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-071 I2: without an image the profile's text model answers."""
        _picture(db)
        blind = openai_config(vision_model="")
        _, blind_wire = vision_client(llm, config=blind)
        await ask(call, PICTURE)
        assert blind_wire.requests[-1]["model"] == blind.model
        assert blind_wire.images == []

        _, seeing_wire = vision_client(llm)
        await ask(call, PICTURE)
        assert seeing_wire.requests[-1]["model"] == VISION_MODEL
        assert len(seeing_wire.images) == 1
