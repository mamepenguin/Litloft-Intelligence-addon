"""SPEC-ADDON-177: the exclusion and ``has_`` keys on intelligence's filter.

The addon validates them as the core's ``/files/matching`` does (SPEC-CORE-172
sending 7), accepts ``epub`` as a kind, and passes each key to
``/files/matching`` only when it is set.
"""

from __future__ import annotations

import json

import pytest

from app.file_matching import InvalidFilter, SearchFilter, fetch_matching_file_ids, parse_search_filter
from tests.matching_core_double import MatchingCore, install
from tests.search_endpoint_db import api, search_index  # noqa: F401

MATCHING_URL = "http://backend:8000/api/internal/files/matching"
EIGHT_KINDS = ["video", "image", "audio", "document", "archive", "other", "text", "pdf"]
NEW_KEYS = (
    "exclude_tags",
    "exclude_kinds",
    "exclude_folders",
    "exclude_collections",
    "has_tag",
    "has_collection",
)


@pytest.fixture()
def core(monkeypatch):
    return install(monkeypatch, MatchingCore())


def _parse(**conditions) -> SearchFilter:
    return parse_search_filter(json.dumps(conditions, ensure_ascii=False))


def _strings(n: int, prefix: str) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


# (conditions, the field read back, its normalised value)
ACCEPTED = [
    pytest.param({"exclude_tags": ["trip"]}, "exclude_tags", ["trip"], id="one-excluded-tag"),
    pytest.param(
        {"exclude_tags": ["", "  ", "Trip", "trip", "  2023  "]},
        "exclude_tags",
        ["Trip", "2023"],
        id="excluded-tags-trimmed-blank-dropped-case-deduped",
    ),
    pytest.param(
        {"tags": _strings(20, "t"), "exclude_tags": _strings(20, "u")},
        "exclude_tags",
        _strings(20, "u"),
        id="twenty-tags-and-twenty-excluded-tags-counted-apart",
    ),
    pytest.param(
        {"exclude_tags": [*_strings(20, "t"), "T0"]},
        "exclude_tags",
        _strings(20, "t"),
        id="twenty-one-raw-twenty-distinct",
    ),
    pytest.param({"exclude_tags": ["x" * 200]}, "exclude_tags", ["x" * 200], id="excluded-tag-200"),
    pytest.param({"exclude_kinds": ["pdf"]}, "exclude_kinds", ["pdf"], id="one-excluded-kind"),
    pytest.param({"exclude_kinds": ["epub"]}, "exclude_kinds", ["epub"], id="excluded-epub"),
    pytest.param({"kinds": ["epub"]}, "kinds", ["epub"], id="epub-kind"),
    pytest.param(
        {"exclude_kinds": ["markdown", "text"]},
        "exclude_kinds",
        ["text"],
        id="excluded-markdown-is-text",
    ),
    pytest.param({"exclude_kinds": EIGHT_KINDS}, "exclude_kinds", EIGHT_KINDS, id="eight-excluded-kinds"),
    pytest.param(
        {"exclude_kinds": [*EIGHT_KINDS, "markdown"]},
        "exclude_kinds",
        EIGHT_KINDS,
        id="nine-raw-eight-distinct-kinds",
    ),
    pytest.param(
        {"exclude_folders": ["/Photos/2024/", "/Photos/", "Photos"]},
        "exclude_folders",
        ["Photos/2024", "Photos"],
        id="excluded-folders-trimmed-and-deduped",
    ),
    pytest.param({"exclude_folders": ["Café"]}, "exclude_folders", ["Café"], id="excluded-folder-nfc"),
    pytest.param(
        {"exclude_folders": ["Photos", "photos"]},
        "exclude_folders",
        ["Photos", "photos"],
        id="excluded-folders-case-sensitive",
    ),
    pytest.param(
        {"folders": _strings(10, "f"), "exclude_folders": _strings(10, "g")},
        "exclude_folders",
        _strings(10, "g"),
        id="ten-folders-and-ten-excluded-folders-counted-apart",
    ),
    pytest.param({"exclude_folders": ["x" * 1024]}, "exclude_folders", ["x" * 1024], id="excluded-folder-1024"),
    pytest.param(
        {"exclude_collections": ["  Trip  ", "", "trip", "旅行 メモ"]},
        "exclude_collections",
        ["Trip", "旅行 メモ"],
        id="excluded-collections-trimmed-blank-dropped-case-deduped",
    ),
    pytest.param(
        {"collections": _strings(10, "c"), "exclude_collections": _strings(10, "d")},
        "exclude_collections",
        _strings(10, "d"),
        id="ten-collections-and-ten-excluded-collections-counted-apart",
    ),
    pytest.param(
        {"exclude_collections": ["y" * 100]},
        "exclude_collections",
        ["y" * 100],
        id="excluded-collection-100",
    ),
    pytest.param({"has_tag": True}, "has_tag", True, id="has-tag-true"),
    pytest.param({"has_tag": False}, "has_tag", False, id="has-tag-false"),
    pytest.param({"has_collection": True}, "has_collection", True, id="has-collection-true"),
    pytest.param({"has_collection": False}, "has_collection", False, id="has-collection-false"),
    pytest.param({"has_tag": None}, "has_tag", None, id="has-tag-null"),
]

REJECTED = [
    pytest.param({"exclude_tags": _strings(21, "t")}, id="twenty-one-excluded-tags"),
    pytest.param({"exclude_tags": ["x" * 201]}, id="excluded-tag-201"),
    pytest.param({"exclude_tags": "trip"}, id="excluded-tags-not-a-list"),
    pytest.param({"exclude_tags": [1]}, id="excluded-tag-not-a-string"),
    pytest.param({"exclude_kinds": ["movie"]}, id="unknown-excluded-kind"),
    pytest.param({"exclude_kinds": [*EIGHT_KINDS, "epub"]}, id="nine-distinct-excluded-kinds"),
    pytest.param({"exclude_kinds": "pdf"}, id="excluded-kinds-not-a-list"),
    pytest.param({"exclude_folders": [""]}, id="empty-excluded-folder"),
    pytest.param({"exclude_folders": ["/"]}, id="root-excluded-folder"),
    pytest.param({"exclude_folders": [".."]}, id="dot-dot-excluded-folder"),
    pytest.param({"exclude_folders": ["a//b"]}, id="empty-segment"),
    pytest.param({"exclude_folders": ["a/./b"]}, id="dot-segment"),
    pytest.param({"exclude_folders": ["a/../b"]}, id="dot-dot-segment"),
    pytest.param({"exclude_folders": ["a\x00b"]}, id="nul"),
    pytest.param({"exclude_folders": ["x" * 1025]}, id="excluded-folder-1025"),
    pytest.param({"exclude_folders": _strings(11, "f")}, id="eleven-excluded-folders"),
    pytest.param({"exclude_collections": ["y" * 101]}, id="excluded-collection-101"),
    pytest.param({"exclude_collections": _strings(11, "c")}, id="eleven-excluded-collections"),
    pytest.param({"exclude_collections": [None]}, id="excluded-collection-not-a-string"),
    pytest.param({"has_tag": "true"}, id="has-tag-a-string"),
    pytest.param({"has_tag": 1}, id="has-tag-a-number"),
    pytest.param({"has_collection": "false"}, id="has-collection-a-string"),
    pytest.param({"has_collection": 0}, id="has-collection-a-number"),
]


@pytest.mark.parametrize(("conditions", "field", "normalised"), ACCEPTED)
def test_spec_addon_177_1_accepts_and_normalises_what_the_core_accepts(conditions, field, normalised):
    got = getattr(_parse(**conditions), field)

    if field == "exclude_tags":
        assert [t.lower() for t in got] == [t.lower() for t in normalised]
    else:
        assert got == normalised


@pytest.mark.parametrize("conditions", REJECTED)
def test_spec_addon_177_1_rejects_what_the_core_rejects(conditions):
    assert _parse(exclude_tags=["trip"]).exclude_tags == ["trip"]

    with pytest.raises(InvalidFilter):
        _parse(**conditions)


def test_spec_addon_177_1_defaults():
    parsed = _parse(tags=["budget"])

    assert parsed.exclude_tags == []
    assert parsed.exclude_kinds == []
    assert parsed.exclude_folders == []
    assert parsed.exclude_collections == []
    assert parsed.has_tag is None
    assert parsed.has_collection is None


async def test_spec_addon_177_2_passes_every_set_key_to_files_matching(monkeypatch):
    core = install(monkeypatch, MatchingCore(file_ids=["a"]))
    parsed = _parse(
        tags=["budget"],
        exclude_tags=["Trip", "trip", "2023"],
        exclude_kinds=["markdown"],
        exclude_folders=["/Work/old/"],
        exclude_collections=["  読了 "],
        has_tag=True,
        has_collection=False,
    )

    ids = await fetch_matching_file_ids("main", parsed)

    assert ids == ["a"]
    request = core.matching_requests[0]
    assert request.method == "POST"
    assert str(request.url) == MATCHING_URL
    sent = json.loads(request.content)
    assert sent["tags"] == ["budget"]
    assert [t.lower() for t in sent["exclude_tags"]] == ["trip", "2023"]
    assert sent["exclude_kinds"] == ["text"]
    assert sent["exclude_folders"] == ["Work/old"]
    assert sent["exclude_collections"] == ["読了"]
    assert sent["has_tag"] is True
    assert sent["has_collection"] is False


async def test_spec_addon_177_2_sends_false_but_not_an_unset_boolean(monkeypatch):
    core = install(monkeypatch, MatchingCore(file_ids=[]))

    await fetch_matching_file_ids("main", _parse(has_tag=False))

    sent = json.loads(core.matching_requests[0].content)
    assert sent["has_tag"] is False
    assert "has_collection" not in sent


async def test_spec_addon_177_2_a_filter_without_a_new_key_sends_none_of_them(monkeypatch):
    core = install(monkeypatch, MatchingCore(file_ids=[]))

    await fetch_matching_file_ids("main", _parse(exclude_tags=["x"]))
    await fetch_matching_file_ids(
        "main",
        _parse(tags=["budget"], kinds=["pdf"], favorite=False, folders=["A"], collections=["B"]),
    )

    with_one = json.loads(core.matching_requests[0].content)
    without = json.loads(core.matching_requests[1].content)
    assert with_one["exclude_tags"] == ["x"]
    for key in NEW_KEYS:
        assert key not in without
        if key != "exclude_tags":
            assert key not in with_one


def test_spec_addon_177_1_a_malformed_new_key_answers_422_without_a_core_call(
    api, search_index, core,  # noqa: F811
):
    accepted = api("/search", {"q": "budget", "filter": json.dumps({"has_tag": False})})
    assert accepted.status_code == 200, accepted.text
    calls_before = len(core.matching_requests)

    response = api("/search", {"q": "budget", "filter": json.dumps({"has_tag": "no"})})

    assert response.status_code == 422, response.text
    assert len(core.matching_requests) == calls_before


def test_spec_addon_177_1_type_epub_passes_the_meaning_stage(api, search_index, core):  # noqa: F811
    for conditions in ({"kinds": ["epub"]}, {"exclude_kinds": ["epub"]}):
        response = api("/search", {"q": "budget", "filter": json.dumps(conditions)})

        assert response.status_code == 200, response.text
    sent = [json.loads(r.content) for r in core.matching_requests]
    assert sent[0]["kinds"] == ["epub"]
    assert sent[1]["exclude_kinds"] == ["epub"]


def test_spec_addon_177_2_the_search_endpoint_forwards_the_new_keys(api, search_index, core):  # noqa: F811
    response = api(
        "/search",
        {
            "q": "budget",
            "filter": json.dumps(
                {"exclude_tags": ["2023"], "exclude_folders": ["Work"], "has_collection": True}
            ),
        },
    )

    assert response.status_code == 200, response.text
    sent = json.loads(core.matching_requests[-1].content)
    assert sent["exclude_tags"] == ["2023"]
    assert sent["exclude_folders"] == ["Work"]
    assert sent["has_collection"] is True
