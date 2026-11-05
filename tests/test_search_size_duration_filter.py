"""SPEC-ADDON-064: ``sizes`` and ``durations`` on intelligence's filter.

The addon validates them exactly as the core's listing and request model do
(SPEC-CORE-044 sending 5), rewrites them in the wire form with duplicates
removed, and passes both lists to ``/files/matching``.
"""

from __future__ import annotations

import json

import pytest

from app.file_matching import InvalidFilter, SearchFilter, fetch_matching_file_ids, parse_search_filter
from tests.matching_core_double import MatchingCore, install
from tests.search_endpoint_db import api, search_index  # noqa: F401

MATCHING_URL = "http://backend:8000/api/internal/files/matching"
MAX_SIZE = 2**53 - 1
MAX_DURATION = 10_000_000


@pytest.fixture()
def core(monkeypatch):
    return install(monkeypatch, MatchingCore())


def _parse(**conditions) -> SearchFilter:
    return parse_search_filter(json.dumps(conditions, ensure_ascii=False))


ACCEPTED = [
    pytest.param({}, id="empty-object"),
    pytest.param({"sizes": []}, id="no-sizes"),
    pytest.param({"durations": []}, id="no-durations"),
    pytest.param({"sizes": ["gte:1024"]}, id="one-size"),
    pytest.param({"sizes": ["gte:0"]}, id="zero-bytes"),
    pytest.param({"sizes": [f"lte:{MAX_SIZE}"]}, id="size-2-53-minus-1"),
    pytest.param({"sizes": ["gt:1", "lt:1073741824"]}, id="a-size-range"),
    pytest.param({"sizes": ["gte:1024", "gte:1024"]}, id="duplicate-size"),
    pytest.param(
        {"sizes": [f"gte:{i}" for i in range(4)]}, id="four-distinct-sizes"
    ),
    pytest.param({"durations": ["gte:0"]}, id="zero-seconds"),
    pytest.param({"durations": ["gt:90.4"]}, id="a-duration-fraction"),
    pytest.param({"durations": ["lt:0.001"]}, id="three-decimals"),
    pytest.param({"durations": [f"lte:{MAX_DURATION}"]}, id="duration-ten-million"),
    pytest.param({"durations": ["gte:89", "lte:90"]}, id="a-duration-range"),
    pytest.param({"durations": ["lt:90", "lt:90"]}, id="duplicate-duration"),
    pytest.param(
        {"durations": [f"gte:{i}" for i in range(4)]},
        id="four-distinct-durations",
    ),
    pytest.param(
        {
            "tags": ["budget"],
            "kinds": ["text"],
            "favorite": True,
            "liked": False,
            "folders": ["Photos"],
            "collections": ["Trip"],
            "sizes": ["gte:1024"],
            "durations": ["lte:600"],
        },
        id="every-condition-at-once",
    ),
]

REJECTED = [
    pytest.param({"sizes": ["gt:9007199254740992"]}, id="size-above-2-53-minus-1"),
    pytest.param({"sizes": ["gt:1024.5"]}, id="size-not-whole"),
    pytest.param({"sizes": ["gt:90.4"]}, id="size-fraction-not-whole"),
    pytest.param({"sizes": ["gt:-1"]}, id="size-negative"),
    pytest.param({"sizes": ["gt:1e3"]}, id="size-exponent"),
    pytest.param({"sizes": ["gt:1.0"]}, id="size-trailing-dot-zero"),
    pytest.param({"sizes": ["gt:"]}, id="size-empty-amount"),
    pytest.param({"sizes": ["1GB"]}, id="size-no-comparison"),
    pytest.param({"sizes": ["eq:1024"]}, id="size-equal-is-not-a-comparison"),
    pytest.param({"sizes": ["GT:1024"]}, id="size-operator-case"),
    pytest.param({"sizes": ["gt:1024KB"]}, id="size-unit-on-the-wire"),
    pytest.param({"sizes": [" gt:1024"]}, id="size-leading-space"),
    pytest.param({"sizes": [""]}, id="size-empty"),
    pytest.param({"sizes": [f"gte:{i}" for i in range(5)]}, id="five-distinct-sizes"),
    pytest.param({"durations": ["gt:10000001"]}, id="duration-above-ten-million"),
    pytest.param(
        {"durations": ["gt:10000000.001"]}, id="duration-above-ten-million-fraction"
    ),
    pytest.param({"durations": ["gt:90.0001"]}, id="duration-four-decimals"),
    pytest.param({"durations": ["gt:-1"]}, id="duration-negative"),
    pytest.param({"durations": ["gt:1e3"]}, id="duration-exponent"),
    pytest.param({"durations": ["gt:1.0"]}, id="duration-trailing-dot-zero"),
    pytest.param({"durations": ["gt:"]}, id="duration-empty-amount"),
    pytest.param({"durations": ["1m30s"]}, id="duration-unit-on-the-wire"),
    pytest.param({"durations": ["gt:1:30"]}, id="duration-clock-on-the-wire"),
    pytest.param({"durations": ["GTE:90"]}, id="duration-operator-case"),
    pytest.param(
        {"durations": [f"gte:{i}" for i in range(5)]},
        id="five-distinct-durations",
    ),
    pytest.param({"sizes": "gte:1024"}, id="sizes-not-a-list"),
    pytest.param({"durations": "gte:90"}, id="durations-not-a-list"),
    pytest.param({"sizes": [1]}, id="size-not-a-string"),
    pytest.param({"durations": [None]}, id="duration-not-a-string"),
]


@pytest.mark.parametrize("conditions", ACCEPTED)
def test_spec_addon_064_1_accepts_a_filter_the_core_accepts(conditions):
    parsed = _parse(**conditions)

    assert isinstance(parsed.sizes, list)
    assert isinstance(parsed.durations, list)
    assert len(parsed.sizes) <= 4
    assert len(parsed.durations) <= 4


@pytest.mark.parametrize("conditions", REJECTED)
def test_spec_addon_064_1_rejects_a_filter_the_core_rejects(conditions):
    field = next(iter(conditions))
    assert getattr(_parse(**{field: ["gte:0"]}), field) == ["gte:0"]

    with pytest.raises(InvalidFilter):
        _parse(**conditions)


def test_spec_addon_064_1_defaults_both_lists_to_empty():
    parsed = _parse()

    assert parsed.sizes == []
    assert parsed.durations == []


@pytest.mark.parametrize(
    ("conditions", "sizes", "durations"),
    [
        (
            {"sizes": ["gte:1024", "gte:1024", "gte:2048"]},
            ["gte:1024", "gte:2048"],
            [],
        ),
        (
            {"durations": ["lt:90", "lt:90", "gte:1800"]},
            [],
            ["lt:90", "gte:1800"],
        ),
        (
            {"sizes": ["gt:1", "lt:1073741824"], "durations": ["gte:89", "lte:90"]},
            ["gt:1", "lt:1073741824"],
            ["gte:89", "lte:90"],
        ),
        (
            {
                "sizes": [f"gte:{i}" for i in range(4)] + ["gte:0"],
            },
            [f"gte:{i}" for i in range(4)],
            [],
        ),
    ],
    ids=[
        "sizes-deduped",
        "durations-deduped",
        "both-lists-in-order",
        "the-cap-counts-after-dedupe",
    ],
)
def test_spec_addon_064_1_normalises_before_the_core_call(
    conditions, sizes, durations
):
    parsed = _parse(**conditions)

    assert parsed.sizes == sizes
    assert parsed.durations == durations


def test_spec_addon_064_1_an_unknown_key_is_invalid():
    raw = json.dumps({"sizes": [], "size": ["gte:1024"]})

    with pytest.raises(InvalidFilter):
        parse_search_filter(raw)


async def test_spec_addon_064_1_passes_both_lists_to_files_matching(monkeypatch):
    core = install(monkeypatch, MatchingCore(file_ids=["a"]))
    parsed = _parse(
        tags=["budget"],
        sizes=["gt:1073741824", "lt:524288000"],
        durations=["gte:1800"],
    )

    ids = await fetch_matching_file_ids("main", parsed)

    assert ids == ["a"]
    assert len(core.matching_requests) == 1
    request = core.matching_requests[0]
    assert request.method == "POST"
    assert str(request.url) == MATCHING_URL
    sent = json.loads(request.content)
    assert sent["drive"] == "main"
    assert sent["sizes"] == ["gt:1073741824", "lt:524288000"]
    assert sent["durations"] == ["gte:1800"]
    assert sent["tags"] == ["budget"]


async def test_spec_addon_064_1_sends_empty_lists_when_no_range_was_given(monkeypatch):
    core = install(monkeypatch, MatchingCore(file_ids=[]))

    await fetch_matching_file_ids("main", _parse())

    sent = json.loads(core.matching_requests[0].content)
    assert sent["sizes"] == []
    assert sent["durations"] == []


async def test_spec_addon_064_1_an_impossible_range_is_passed_through(monkeypatch):
    core = install(monkeypatch, MatchingCore(file_ids=[]))
    parsed = _parse(sizes=["gt:1073741824", "lt:524288000"])

    ids = await fetch_matching_file_ids("main", parsed)

    assert ids == []
    sent = json.loads(core.matching_requests[0].content)
    assert sent["sizes"] == ["gt:1073741824", "lt:524288000"]


async def test_spec_addon_064_1_a_rejected_range_makes_no_core_call(monkeypatch):
    core = install(monkeypatch, MatchingCore(file_ids=[]))
    assert _parse(durations=["gte:0"]).durations == ["gte:0"]

    with pytest.raises(InvalidFilter):
        _parse(durations=["gt:90.0001"])

    assert core.matching_requests == []


def test_spec_addon_064_1_a_malformed_range_answers_422(api, search_index, core):  # noqa: F811
    response = api(
        "/search",
        {
            "q": "budget",
            "filter": json.dumps({"durations": ["gt:90.0001"]}),
        },
    )

    assert response.status_code == 422, response.text
    assert core.matching_requests == []


def test_spec_addon_064_1_an_accepted_range_reaches_the_core(api, search_index, core):  # noqa: F811
    response = api(
        "/search",
        {
            "q": "budget",
            "filter": json.dumps({"sizes": ["gte:1024"], "durations": ["lt:90"]}),
        },
    )

    assert response.status_code == 200, response.text
    assert core.matching_requests
    sent = json.loads(core.matching_requests[-1].content)
    assert sent["sizes"] == ["gte:1024"]
    assert sent["durations"] == ["lt:90"]
