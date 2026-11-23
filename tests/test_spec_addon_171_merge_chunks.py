"""Acceptance tests for SPEC-ADDON-171 I9: ``_merge_chunks`` is unchanged."""

from __future__ import annotations

import pytest

from app.file_find import _merge_chunks

FIRST = "a" * 250
SECOND = FIRST[-200:] + "b" * 10
THIRD = SECOND[-200:] + "c"


@pytest.mark.parametrize(
    ("chunks", "limit", "expected"),
    [
        pytest.param([], 200, "", id="empty"),
        pytest.param(["abc"], 200, "abc", id="one-chunk"),
        pytest.param(["abc", "def"], 200, "abc\ndef", id="no-overlap"),
        pytest.param(
            ["a" * 30, "a" * 20 + "z"], 200, "a" * 30 + "z", id="overlap-20",
        ),
        pytest.param(
            ["a" * 60, "a" * 50 + "b" * 5], 200, "a" * 60 + "b" * 5,
            id="overlap-50",
        ),
        pytest.param(
            [FIRST, SECOND], 200, FIRST + "b" * 10, id="overlap-200",
        ),
        pytest.param(
            [FIRST, SECOND, THIRD], 200, FIRST + "b" * 10 + "c",
            id="three-chunks",
        ),
        pytest.param(
            ["a" * 30, "a" * 19 + "z"], 200, "a" * 30 + "\n" + "a" * 19 + "z",
            id="overlap-19-is-not-dropped",
        ),
        pytest.param(
            ["a" * 60, "a" * 50 + "b"], 20, "a" * 60 + "a" * 30 + "b",
            id="limit-twenty-drops-twenty",
        ),
    ],
)
def test_spec_addon_171_i9_merge_chunks_returns_the_same_text(chunks, limit, expected):
    """I9: the find route and the picture-text rebuild keep their merged text."""
    assert _merge_chunks(chunks, limit) == expected
