"""Citation location against a ground-truth segment hint (Ask evals)."""

from __future__ import annotations

import pytest

from app.evals.loader import SegmentHint
from app.evals.stages import _location_matches_hint


@pytest.mark.parametrize(
    ("loc", "expected"),
    [
        ("page 3", True),
        ("section 3", True),
        ("Section 3", True),
        ("section 30", False),
        ("page 4", False),
        ("chunk 3", False),
        ("section", False),
    ],
)
def test_section_location_matches_page_hint(loc: str, expected: bool) -> None:
    assert _location_matches_hint(loc, SegmentHint(page=3)) is expected


@pytest.mark.parametrize(
    ("loc", "expected"),
    [("0:10", True), ("1:10", False), ("section 3", False)],
)
def test_timestamp_location_matches_time_hint(loc: str, expected: bool) -> None:
    assert _location_matches_hint(loc, SegmentHint(time_range=(0.0, 20.0))) is expected
