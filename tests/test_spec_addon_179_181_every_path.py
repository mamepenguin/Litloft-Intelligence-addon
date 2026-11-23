"""SPEC-ADDON-179 I1 I4 I7, SPEC-ADDON-181 I6: the block and rules on every path.

Each path of the file question is played to its end against the real client
over a mock transport, and every request it sends is read back.
"""

from __future__ import annotations

import pytest

from tests.file_context_doubles import (
    BLOCKS,
    NEW_OPENER,
    OLD_OPENER,
    RUN,
    SCENARIOS,
    assert_s179_and_s181,
    question_bodies,
    system_of,
    today,
    user_text,
    without_additions,
)
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    addon,
    call,
    db,
    llm,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture

#: Paths with a request on the passages-only branch, read from today's requests.
PASSAGES_ONLY_PATHS = [
    name for name in SCENARIOS
    if any(OLD_OPENER in system_of(body) for body in today(name))
]


async def _play(name, db, addon, llm, call, tmp_path, monkeypatch):
    wire = await RUN[name](db, addon, llm, call, tmp_path, monkeypatch)
    bodies = question_bodies(wire)
    assert len(bodies) == len(wire.requests)
    return bodies


@pytest.mark.parametrize("name", SCENARIOS)
async def test_spec_addon_179_i1_every_request_starts_with_the_file_block(
    name, db, addon, llm, call, tmp_path, monkeypatch,
):
    """SPEC-ADDON-179 I1: File then Type (no mount, so no Folder), one empty line, then the question."""
    bodies = await _play(name, db, addon, llm, call, tmp_path, monkeypatch)

    assert bodies
    for body in bodies:
        text = user_text(body)
        assert text.startswith(BLOCKS[name]), text[:300]
        assert text[len(BLOCKS[name]):].startswith("Question: "), text[:300]


@pytest.mark.parametrize("name", SCENARIOS)
async def test_spec_addon_179_i4_and_181_i6_every_system_prompt_carries_the_rules_in_place(
    name, db, addon, llm, call, tmp_path, monkeypatch,
):
    """SPEC-ADDON-179 I4, SPEC-ADDON-181 I6: S179 first after Rules:, S181a then S181b above the Answer line."""
    bodies = await _play(name, db, addon, llm, call, tmp_path, monkeypatch)

    assert bodies
    for body in bodies:
        assert_s179_and_s181(system_of(body))


@pytest.mark.parametrize("name", PASSAGES_ONLY_PATHS)
async def test_spec_addon_179_i4_the_passages_only_opener_names_the_file_lines(
    name, db, addon, llm, call, tmp_path, monkeypatch,
):
    """SPEC-ADDON-179 I4: the passages-only branch opens with the new opener."""
    bodies = await _play(name, db, addon, llm, call, tmp_path, monkeypatch)

    passages_only = [
        new for new, old in zip(bodies, today(name), strict=True)
        if OLD_OPENER in system_of(old)
    ]
    assert passages_only
    for body in passages_only:
        assert NEW_OPENER in system_of(body)
        assert OLD_OPENER not in system_of(body)


@pytest.mark.parametrize("name", SCENARIOS)
async def test_spec_addon_179_i7_every_request_is_todays_apart_from_the_additions(
    name, db, addon, llm, call, tmp_path, monkeypatch,
):
    """SPEC-ADDON-179 I7, SPEC-ADDON-181 I6: no request added; only the spec's lines differ.

    Taking out the file block, S179, S181a/b, the new opener and S180 leaves
    today's request byte for byte: every earlier rule, citation number and tool
    is unchanged.
    """
    bodies = await _play(name, db, addon, llm, call, tmp_path, monkeypatch)

    assert [without_additions(body, BLOCKS[name]) for body in bodies] == today(name)
