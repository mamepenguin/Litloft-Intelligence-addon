"""SPEC-ADDON-054 I3: the Ask scope is the search filter's validator, closed.

``AskScope`` accepts exactly what the search ``filter`` accepts (same caps,
same normalisation, ``markdown`` accepted as ``text``), rejects a scope with
no condition, and refuses keys the search filter does not have. ``AskRequest``
forbids unknown top-level keys so a later field fails closed.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.routers.rag as rag_router
import app.schemas as schemas


def _ask_scope():
    scope = getattr(schemas, "AskScope", None)
    assert scope is not None, "AskScope is missing"
    return scope


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"folders": ["/Work/2024/"]}, {"folders": ["Work/2024"]}),
        (
            {"folders": ["Photos/2024", "Photos/2024", "Photos/2025"]},
            {"folders": ["Photos/2024", "Photos/2025"]},
        ),
        ({"tags": [" 会議 ", "会議", "会議 "]}, {"tags": ["会議"]}),
        (
            {"collections": ["読書メモ", " 読書メモ "]},
            {"collections": ["読書メモ"]},
        ),
        ({"kinds": ["markdown", "text"]}, {"kinds": ["text"]}),
        (
            {
                "folders": ["Work"],
                "collections": ["読書メモ"],
                "tags": ["会議"],
                "kinds": ["pdf"],
            },
            {
                "folders": ["Work"],
                "collections": ["読書メモ"],
                "tags": ["会議"],
                "kinds": ["pdf"],
            },
        ),
        (
            {
                "folders": [f"f{i}" for i in range(10)],
                "collections": [f"c{i}" for i in range(10)],
                "tags": [f"t{i}" for i in range(20)],
                "kinds": [
                    "video",
                    "image",
                    "audio",
                    "document",
                    "archive",
                    "other",
                    "text",
                    "pdf",
                ],
            },
            {"accepted": True},
        ),
    ],
    ids=[
        "folder-trimmed",
        "folders-deduped",
        "tags-deduped-case-insensitively",
        "collections-deduped",
        "markdown-alias-of-text",
        "all-four-kinds",
        "caps-at-the-boundary",
    ],
)
def test_ask_scope_accepts_and_normalises_like_the_search_filter(payload, expected):
    body = schemas.AskRequest(query="what did we decide", scope=payload)
    dumped = body.scope.model_dump()

    if expected == {"accepted": True}:
        assert len(dumped["folders"]) == 10
        assert len(dumped["collections"]) == 10
        assert len(dumped["tags"]) == 20
        assert len(dumped["kinds"]) == 8
        return

    for key, value in expected.items():
        assert dumped[key] == value


@pytest.mark.parametrize(
    ("payload", "no_condition"),
    [
        ({}, True),
        ({"tags": []}, True),
        ({"tags": ["   "]}, True),
        ({"folders": ["/"]}, False),
        ({"favorite": True}, False),
        ({"liked": True}, False),
        ({"tags": [f"t{i}" for i in range(21)]}, False),
        ({"folders": [f"f{i}" for i in range(11)]}, False),
        ({"collections": [f"c{i}" for i in range(11)]}, False),
        ({"kinds": ["hologram"]}, False),
        ({"folders": ["a/../b"]}, False),
        ({"folders": ["a//b"]}, False),
        ({"tags": ["x" * 201]}, False),
        ({"collections": ["x" * 101]}, False),
        ({"tags": "not-a-list"}, False),
    ],
    ids=[
        "empty-scope",
        "all-lists-empty",
        "whitespace-tag",
        "root-folder-only",
        "favorite-is-not-a-scope-key",
        "liked-is-not-a-scope-key",
        "over-tag-cap",
        "over-folder-cap",
        "over-collection-cap",
        "unknown-kind",
        "dot-dot-folder",
        "empty-folder-segment",
        "over-tag-length",
        "over-collection-length",
        "tag-list-is-a-string",
    ],
)
def test_ask_scope_rejects_what_the_search_filter_rejects(payload, no_condition):
    scope_cls = _ask_scope()

    with pytest.raises(ValidationError) as exc_info:
        scope_cls.model_validate(payload)

    if no_condition:
        assert any(
            error["type"] == "value_error"
            and error["msg"] == "Value error, scope has no condition"
            for error in exc_info.value.errors()
        )


def test_ask_request_scope_defaults_to_none():
    body = schemas.AskRequest(query="a valid question")
    assert body.scope is None


def test_ask_request_forbids_an_unknown_top_level_key():
    with pytest.raises(ValidationError):
        schemas.AskRequest(query="a valid question", not_a_real_key=1)


def test_ask_request_scope_must_be_an_object():
    with pytest.raises(ValidationError):
        schemas.AskRequest(query="a valid question", scope="path:Work")


def test_fastapi_answers_the_list_shaped_422_for_an_empty_scope():
    app = FastAPI()
    app.include_router(rag_router.router)

    response = TestClient(app).post(
        "/ask",
        headers={"X-Lit-Drive": "family"},
        json={"query": "a valid question", "scope": {}},
    )

    assert response.status_code == 422
    body = response.json()
    assert isinstance(body["detail"], list)
    assert body["detail"][0]["loc"][:2] == ["body", "scope"]
    assert body["detail"][0]["type"] == "value_error"
    assert body["detail"][0]["msg"] == "Value error, scope has no condition"
