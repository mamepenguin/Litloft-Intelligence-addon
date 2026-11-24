"""SPEC-ADDON-180 I5: with an image and passages, the "say so" rule names both."""

from __future__ import annotations

import pytest

from tests.file_context_doubles import (
    IMAGE_RULE,
    NEW_OPENER,
    PASSAGES_ONLY_SENTENCE,
    RUN,
    S179,
    S180,
    S181A,
    S181B,
    S181C,
    question_bodies,
    system_of,
    today,
)
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    addon,
    call,
    db,
    llm,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture

#: The first request of a refused picture is the passages-and-image request.
IMAGE_AND_PASSAGES = [("passages-and-image", 0), ("text-only-retry", 0)]
OTHER_BRANCHES = [
    "passages-only",
    "image-only",
    "no-frame-prepared-openai_compatible",
    "frames-tool-openai_compatible-text-video",
    "frames-fallback-ollama-text-video",
    "cut-material-openai_compatible",
]


def _without_new_lines(system: str) -> list[str]:
    return [line for line in system.split("\n") if line not in (S179, S181A, S181B, S181C)]


@pytest.mark.parametrize(("name", "index"), IMAGE_AND_PASSAGES)
async def test_spec_addon_180_i5_the_image_and_passages_rule_stands_where_the_passages_only_sentence_did(
    name, index, db, addon, llm, call, tmp_path, monkeypatch,
):
    """SPEC-ADDON-180 I5: S180 in the passages-only sentence's place; that sentence gone; the image rule kept."""
    wire = await RUN[name](db, addon, llm, call, tmp_path, monkeypatch)
    system = system_of(question_bodies(wire)[index])
    before = today(name)[index]

    old_lines = system_of(before).split("\n")
    new_lines = _without_new_lines(system)
    assert S180 in new_lines
    assert new_lines.index(S180) == old_lines.index(PASSAGES_ONLY_SENTENCE)
    assert system.count(S180) == 1
    assert PASSAGES_ONLY_SENTENCE not in system
    assert IMAGE_RULE in system


@pytest.mark.parametrize("name", OTHER_BRANCHES)
async def test_spec_addon_180_i5_no_other_branch_gains_the_image_and_passages_rule(
    name, db, addon, llm, call, tmp_path, monkeypatch,
):
    """SPEC-ADDON-180 item 5: only the passages-and-image branch changes its "say so" rule."""
    wire = await RUN[name](db, addon, llm, call, tmp_path, monkeypatch)

    for body, before in zip(question_bodies(wire), today(name), strict=True):
        system = system_of(body)
        assert S180 not in system
        assert S179 in system.split("\n")
        if PASSAGES_ONLY_SENTENCE in system_of(before):
            assert PASSAGES_ONLY_SENTENCE in system
            assert NEW_OPENER in system
