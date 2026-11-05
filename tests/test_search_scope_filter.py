"""SPEC-ADDON-063: ``folders`` and ``collections`` on intelligence's filter.

The addon validates them exactly as the core's request model does, rejects a
key it does not know, and passes both lists through to ``/files/matching``.
"""

from __future__ import annotations

import json

import pytest

from app.file_matching import InvalidFilter, SearchFilter, fetch_matching_file_ids, parse_search_filter
from tests.matching_core_double import MatchingCore, install
from tests.search_endpoint_db import api, search_index  # noqa: F401

MATCHING_URL = "http://backend:8000/api/internal/files/matching"


@pytest.fixture()
def core(monkeypatch):
    return install(monkeypatch, MatchingCore())


def _parse(**conditions) -> SearchFilter:
    return parse_search_filter(json.dumps(conditions, ensure_ascii=False))


ACCEPTED = [
    pytest.param({}, id="empty-object"),
    pytest.param({"folders": []}, id="no-folders"),
    pytest.param({"collections": []}, id="no-collections"),
    pytest.param({"folders": ["Photos/2024"]}, id="one-folder"),
    pytest.param({"folders": ["My Photos/2024"]}, id="a-folder-with-a-space"),
    pytest.param({"folders": ["/Photos/2024/"]}, id="folder-trimmed-of-slashes"),
    pytest.param({"folders": ["Cafe\u0301"]}, id="folder-nfc-normalised"),
    pytest.param({"folders": ["Photos", "/Photos/"]}, id="folders-deduped"),
    pytest.param({"folders": ["Photos", "photos"]}, id="folder-case-is-two-folders"),
    pytest.param(
        {"folders": [f"f{i}" for i in range(10)]}, id="ten-folders"
    ),
    pytest.param({"folders": ["x" * 1024]}, id="folder-1024-characters"),
    pytest.param({"collections": ["Trip"]}, id="one-collection"),
    pytest.param({"collections": ["旅行 メモ"]}, id="a-collection-with-a-space"),
    pytest.param({"collections": ["  Trip  "]}, id="collection-trimmed"),
    pytest.param({"collections": ["", "   ", "Trip"]}, id="blank-collections-dropped"),
    pytest.param({"collections": ["Trip", "trip"]}, id="collections-deduped-case-insensitively"),
    pytest.param({"collections": [f"c{i}" for i in range(10)]}, id="ten-collections"),
    pytest.param({"collections": ["x" * 100]}, id="collection-100-characters"),
    pytest.param(
        {
            "tags": ["budget"],
            "kinds": ["text"],
            "favorite": True,
            "liked": False,
            "folders": ["Photos/2024"],
            "collections": ["Trip"],
        },
        id="every-condition-at-once",
    ),
]

REJECTED = [
    pytest.param({"folders": [""]}, id="empty-folder"),
    pytest.param({"folders": ["/"]}, id="root-folder"),
    pytest.param({"folders": ["."]}, id="dot-folder"),
    pytest.param({"folders": [".."]}, id="dot-dot-folder"),
    pytest.param({"folders": ["a//b"]}, id="empty-segment"),
    pytest.param({"folders": ["a/./b"]}, id="dot-segment"),
    pytest.param({"folders": ["a/../b"]}, id="dot-dot-segment"),
    pytest.param({"folders": ["a\x00b"]}, id="nul-in-folder"),
    pytest.param({"folders": ["x" * 1025]}, id="folder-1025-characters"),
    pytest.param({"folders": [f"f{i}" for i in range(11)]}, id="eleven-folders"),
    pytest.param({"collections": ["x" * 101]}, id="collection-101-characters"),
    pytest.param({"collections": [f"c{i}" for i in range(11)]}, id="eleven-collections"),
    pytest.param({"folders": "Photos"}, id="folders-not-a-list"),
    pytest.param({"collections": "Trip"}, id="collections-not-a-list"),
    pytest.param({"folders": [1]}, id="folder-not-a-string"),
    pytest.param({"collections": [None]}, id="collection-not-a-string"),
]


@pytest.mark.parametrize("conditions", ACCEPTED)
def test_spec_addon_063_i7_accepts_a_scope_the_core_accepts(conditions):
    parsed = _parse(**conditions)

    assert isinstance(parsed.folders, list)
    assert isinstance(parsed.collections, list)
    assert len(parsed.folders) <= 10
    assert len(parsed.collections) <= 10


@pytest.mark.parametrize("conditions", REJECTED)
def test_spec_addon_063_i7_rejects_a_scope_the_core_rejects(conditions):
    with pytest.raises(InvalidFilter):
        _parse(**conditions)


def test_spec_addon_063_2_defaults_both_lists_to_empty():
    parsed = _parse()

    assert parsed.folders == []
    assert parsed.collections == []


@pytest.mark.parametrize(
    ("conditions", "folders", "collections"),
    [
        (
            {"folders": ["/Photos/2024/", "/Photos/"]},
            ["Photos/2024", "Photos"],
            [],
        ),
        (
            {"folders": ["Cafe\u0301"]},
            ["Café"],
            [],
        ),
        (
            {"collections": ["  Trip  ", "", "trip", "旅行 メモ"]},
            [],
            ["Trip", "旅行 メモ"],
        ),
        (
            {"folders": ["Photos", "photos"], "collections": ["Trip", "trip"]},
            ["Photos", "photos"],
            ["Trip"],
        ),
    ],
    ids=[
        "slashes-trimmed-and-folders-deduped",
        "folders-nfc-normalised",
        "collections-trimmed-blank-dropped-case-deduped",
        "folders-stay-case-sensitive",
    ],
)
def test_spec_addon_063_2_normalises_before_the_core_call(
    conditions, folders, collections
):
    parsed = _parse(**conditions)

    assert parsed.folders == folders
    assert parsed.collections == collections


def test_spec_addon_063_1_an_unknown_key_is_invalid():
    raw = json.dumps({"folders": [], "collections": [], "folder": ["Photos"]})

    with pytest.raises(InvalidFilter):
        parse_search_filter(raw)


def test_spec_addon_063_1_an_unknown_key_beside_a_known_one_is_invalid():
    raw = json.dumps({"tags": ["budget"], "unknown": True})

    with pytest.raises(InvalidFilter):
        parse_search_filter(raw)


async def test_spec_addon_063_3_passes_both_lists_to_files_matching(monkeypatch):
    core = install(monkeypatch, MatchingCore(file_ids=["a"]))
    parsed = _parse(
        tags=["budget"],
        folders=["My Photos", "Work/2024"],
        collections=["旅行"],
    )

    ids = await fetch_matching_file_ids("main", parsed)

    assert ids == ["a"]
    assert len(core.matching_requests) == 1
    request = core.matching_requests[0]
    assert request.method == "POST"
    assert str(request.url) == MATCHING_URL
    sent = json.loads(request.content)
    assert sent["drive"] == "main"
    assert sent["folders"] == ["My Photos", "Work/2024"]
    assert sent["collections"] == ["旅行"]
    assert sent["tags"] == ["budget"]


async def test_spec_addon_063_3_sends_empty_lists_when_no_scope_was_given(monkeypatch):
    core = install(monkeypatch, MatchingCore(file_ids=[]))

    await fetch_matching_file_ids("main", _parse())

    sent = json.loads(core.matching_requests[0].content)
    assert sent["folders"] == []
    assert sent["collections"] == []


async def test_spec_addon_063_2_a_rejected_scope_makes_no_core_call(monkeypatch):
    core = install(monkeypatch, MatchingCore(file_ids=[]))

    with pytest.raises(InvalidFilter):
        _parse(folders=["a//b"])

    assert core.matching_requests == []


def test_spec_addon_063_i7_an_unknown_filter_key_answers_422(api, search_index, core):  # noqa: F811
    response = api(
        "/search",
        {
            "q": "budget",
            "filter": json.dumps({"folders": [], "folder": ["Photos"]}),
        },
    )

    assert response.status_code == 422, response.text
    assert core.matching_requests == []
