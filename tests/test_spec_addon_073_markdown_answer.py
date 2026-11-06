"""SPEC-ADDON-073: the backend carries Markdown and renders nothing.

The answer is a byte stream; only the tab formats it. This file holds the
route side of 073 beside the availability and prompt tests it shares a setup
with.
"""

from __future__ import annotations

from pathlib import Path

import app.config as config
from tests.file_question_doubles import chunk
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    addon,
    ask,
    by_kind,
    call,
    db,
    events,
    llm,
    seed_picture,
    vision_client,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture

PICTURE = "pic000000031"
MARKDOWN = "## Amount\n\n- **42** dollars in [1]\n\n| a | b |\n|---|---|\n"


def _picture(db):
    path = Path(config.settings.intelligence_data_dir).parent / "pictures" / "ask.png"
    return seed_picture(db, PICTURE, text=chunk("amount", 20), path=path)


class TestTheMarkdownAnswer:
    def test_the_drive_wide_ask_prompt_does_not_gain_the_markdown_rule(self):
        """SPEC-ADDON-073: the file-question prompt changes, the drive-wide one does not."""
        from app.rag.prompt import (
            build_file_question_system_prompt,
            build_system_prompt,
        )

        assert "markdown" in build_file_question_system_prompt("auto").lower()
        assert "markdown" not in build_system_prompt("auto").lower()

    async def test_the_file_question_system_prompt_asks_for_markdown(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-073: the file-question prompt's output rules ask for Markdown."""
        _picture(db)
        _, wire = vision_client(llm)

        await ask(call, PICTURE)

        assert "markdown" in wire.system_prompts[0].lower()

    async def test_the_route_forwards_the_markdown_unchanged(
        self, db, addon, llm, call,
    ):
        """SPEC-ADDON-073: nothing renders in the backend; deltas are the model's bytes."""
        _picture(db)
        vision_client(
            llm, script=lambda body: (200, MARKDOWN),
        )

        response = await ask(call, PICTURE)

        chunks = [
            data["delta"] for kind, data in events(response)
            if kind == "answer_chunk"
        ]
        assert "".join(chunks) == MARKDOWN
        assert by_kind(response)["meta"]["image"] == "attached"
