"""SPEC-ADDON-143: ``modified`` on intelligence's filter.

The addon validates it as the core's listing and ``/files/matching`` do
(SPEC-CORE-138 sending 3), rewrites it with duplicates removed, and passes it
to ``/files/matching``.
"""

from __future__ import annotations

import json

import pytest

from app.file_matching import InvalidFilter, SearchFilter, fetch_matching_file_ids, parse_search_filter
from tests.matching_core_double import MatchingCore, install
from tests.search_endpoint_db import api, search_index  # noqa: F401

MATCHING_URL = "http://backend:8000/api/internal/files/matching"
MAX_WIRE = 253402300799


@pytest.fixture()
def core(monkeypatch):
    return install(monkeypatch, MatchingCore())


def _parse(**conditions) -> SearchFilter:
    return parse_search_filter(json.dumps(conditions, ensure_ascii=False))


ACCEPTED = [
    pytest.param({"modified": []}, [], id="no-comparison"),
    pytest.param({"modified": ["gte:0"]}, ["gte:0"], id="zero"),
    pytest.param({"modified": [f"lte:{MAX_WIRE}"]}, [f"lte:{MAX_WIRE}"], id="the-largest"),
    pytest.param({"modified": ["lt:1704034800"]}, ["lt:1704034800"], id="a-boundary"),
    pytest.param(
        {"modified": ["gte:1717167600", "lt:1719759600"]},
        ["gte:1717167600", "lt:1719759600"],
        id="a-range-in-order",
    ),
    pytest.param(
        {"modified": ["lt:1704034800", "lt:1704034800", "gt:5"]},
        ["lt:1704034800", "gt:5"],
        id="duplicates-counted-once",
    ),
    pytest.param(
        {"modified": [f"gte:{i}" for i in range(4)]},
        [f"gte:{i}" for i in range(4)],
        id="four-distinct",
    ),
    pytest.param(
        {"modified": [f"gte:{i}" for i in range(4)] + ["gte:0"]},
        [f"gte:{i}" for i in range(4)],
        id="the-cap-counts-after-dedupe",
    ),
]

REJECTED = [
    pytest.param({"modified": ["lt:-1"]}, id="negative"),
    pytest.param({"modified": [f"lt:{MAX_WIRE + 1}"]}, id="above-9999-12-31T23-59-59Z"),
    pytest.param({"modified": ["lt:1.5"]}, id="a-fraction"),
    pytest.param({"modified": ["lt:1e3"]}, id="an-exponent"),
    pytest.param({"modified": ["lt:"]}, id="an-empty-amount"),
    pytest.param({"modified": ["lt"]}, id="no-amount"),
    pytest.param({"modified": ["1704034800"]}, id="no-comparison"),
    pytest.param({"modified": ["eq:1704034800"]}, id="equal-is-not-a-comparison"),
    pytest.param({"modified": ["LT:1704034800"]}, id="operator-case"),
    pytest.param({"modified": [" lt:1704034800"]}, id="a-leading-space"),
    pytest.param({"modified": ["lt:1704034800 "]}, id="a-trailing-space"),
    pytest.param({"modified": ["lt:2024-01-01"]}, id="a-date-on-the-wire"),
    pytest.param({"modified": ["lt:1y"]}, id="a-relative-amount-on-the-wire"),
    pytest.param({"modified": ["<2024"]}, id="the-typed-form"),
    pytest.param({"modified": [""]}, id="empty"),
    pytest.param({"modified": [f"gte:{i}" for i in range(5)]}, id="five-distinct"),
    pytest.param({"modified": "lt:1704034800"}, id="not-a-list"),
    pytest.param({"modified": [1704034800]}, id="not-a-string"),
]


@pytest.mark.parametrize(("conditions", "normalised"), ACCEPTED)
def test_spec_addon_143_1_accepts_and_normalises_what_the_core_accepts(conditions, normalised):
    assert _parse(**conditions).modified == normalised


@pytest.mark.parametrize("conditions", REJECTED)
def test_spec_addon_143_1_rejects_what_the_core_rejects(conditions):
    assert _parse(modified=["gte:0"]).modified == ["gte:0"]

    with pytest.raises(InvalidFilter):
        _parse(**conditions)


def test_spec_addon_143_1_defaults_to_empty():
    assert _parse().modified == []
    assert _parse(sizes=["gte:1"]).modified == []


async def test_spec_addon_143_1_passes_modified_to_files_matching(monkeypatch):
    core = install(monkeypatch, MatchingCore(file_ids=["a"]))
    parsed = _parse(tags=["budget"], modified=["lt:1704034800", "lt:1704034800", "gte:0"])

    ids = await fetch_matching_file_ids("main", parsed)

    assert ids == ["a"]
    request = core.matching_requests[0]
    assert request.method == "POST"
    assert str(request.url) == MATCHING_URL
    sent = json.loads(request.content)
    assert sent["modified"] == ["lt:1704034800", "gte:0"]
    assert sent["tags"] == ["budget"]


async def test_spec_addon_143_1_sends_an_empty_list_when_no_comparison_was_given(monkeypatch):
    core = install(monkeypatch, MatchingCore(file_ids=[]))

    await fetch_matching_file_ids("main", _parse())

    sent = json.loads(core.matching_requests[0].content)
    assert sent["modified"] == []


async def test_spec_addon_143_1_an_impossible_range_is_passed_through(monkeypatch):
    core = install(monkeypatch, MatchingCore(file_ids=[]))

    ids = await fetch_matching_file_ids("main", _parse(modified=["gt:1704034800", "lt:0"]))

    assert ids == []
    sent = json.loads(core.matching_requests[0].content)
    assert sent["modified"] == ["gt:1704034800", "lt:0"]


def test_spec_addon_143_1_a_malformed_modified_answers_422(api, search_index, core):  # noqa: F811
    accepted = api("/search", {"q": "budget", "filter": json.dumps({"modified": ["gte:0"]})})
    assert accepted.status_code == 200, accepted.text
    calls_before = len(core.matching_requests)

    response = api("/search", {"q": "budget", "filter": json.dumps({"modified": ["lt:-1"]})})

    assert response.status_code == 422, response.text
    assert len(core.matching_requests) == calls_before


def test_spec_addon_143_1_an_accepted_modified_reaches_the_core(api, search_index, core):  # noqa: F811
    response = api(
        "/search",
        {"q": "budget", "filter": json.dumps({"sizes": ["gte:1024"], "modified": ["lt:1704034800"]})},
    )

    assert response.status_code == 200, response.text
    assert core.matching_requests
    sent = json.loads(core.matching_requests[-1].content)
    assert sent["modified"] == ["lt:1704034800"]
    assert sent["sizes"] == ["gte:1024"]
