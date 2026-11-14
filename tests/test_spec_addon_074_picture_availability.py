"""SPEC-ADDON-074: whether the Ask tab is offered for a picture.

Availability is the tab's only signal, so it must widen for a text-less
picture only when some profile can see. Each negative is paired with the
positive that the missing feature would supply.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.config as config
from app.llm import reset_vision_capability_cache
from tests.file_question_doubles import chunk
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    DRIVE,
    OTHER,
    Wire,
    addon,
    call,
    db,
    llm,
    openai_client,
    openai_config,
    profiles,
    record_vision_unsupported,
    refusing,
    seed_picture,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture

PICTURE = "pic000000041"
CONTROL = "pic000000045"
OTHER_DRIVE = "pic000000043"
INACTIVE = "pic000000044"


def _picture(db, file_id=PICTURE, *, text=None, drive=DRIVE, active=True):
    base = Path(config.settings.intelligence_data_dir).parent / "pictures"
    seed_picture(
        db, file_id, text=text, path=base / f"{file_id}.png",
        drive=drive, active=active,
    )


async def _available(call, file_id=PICTURE) -> bool:
    response = await call("GET", f"/files/{file_id}/ask")
    assert response.status_code == 200
    return response.json()["available"]


class TestAPictureWithoutText:
    async def test_widens_only_when_a_profile_names_a_vision_model(self, db, addon, call):
        """SPEC-ADDON-074 I6: the same picture flips with a usable vision profile."""
        _picture(db)
        addon(profiles=profiles(vision_model=""))
        assert await _available(call) is False

        addon(profiles=profiles())
        assert await _available(call) is True

    async def test_a_recorded_unsupported_verdict_removes_the_vision(self, db, addon, call):
        """SPEC-ADDON-074 I6: the description path's verdict closes the widened answer."""
        _picture(db)
        addon(profiles=profiles())
        wire = Wire("openai_compatible")
        wire.script = refusing(wire)
        client = openai_client(openai_config(), wire)
        await record_vision_unsupported(client)
        assert await _available(call) is False

        reset_vision_capability_cache()
        assert await _available(call) is True

    async def test_a_disabled_client_cannot_see(self, db, addon, call):
        """SPEC-ADDON-074 I6: a vision model on an unusable client does not count."""
        _picture(db)
        addon(profiles=profiles(base_url=""))
        assert await _available(call) is False

        addon(profiles=profiles())
        assert await _available(call) is True

    async def test_never_available_for_another_drive(self, db, addon, call):
        """SPEC-ADDON-074 I6: the widened rule is still drive-scoped."""
        _picture(db, OTHER_DRIVE, drive=OTHER)
        _picture(db, CONTROL)
        addon(profiles=profiles())

        assert await _available(call, OTHER_DRIVE) is False
        assert await _available(call, CONTROL) is True

    async def test_never_available_for_an_inactive_picture(self, db, addon, call):
        """SPEC-ADDON-074 I6: the widened rule still requires an active row."""
        _picture(db, INACTIVE, active=False)
        _picture(db, CONTROL)
        addon(profiles=profiles())

        assert await _available(call, INACTIVE) is False
        assert await _available(call, CONTROL) is True


class TestEveryOtherKind:
    @staticmethod
    def _seed(db, file_id, mime):
        db.file(file_id, mime=mime)

    @pytest.mark.parametrize(
        "mime",
        [
            "text/markdown",
            "application/pdf",
            "application/epub+zip",
            "audio/mpeg",
        ],
        ids=["text", "pdf", "epub", "audio"],
    )
    async def test_a_text_less_non_picture_still_needs_a_chunk(self, mime, db, addon, call):
        """SPEC-ADDON-074 I6: the widened rule is for pictures only."""
        file_id = "fil000000041"
        self._seed(db, file_id, mime)
        _picture(db, CONTROL)
        addon(profiles=profiles())

        assert await _available(call, file_id) is False
        assert await _available(call, CONTROL) is True

