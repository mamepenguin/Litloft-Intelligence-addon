"""SPEC-ADDON-115 I3: the two ``files/matching`` calls a narrowed POST makes
carry the filter's conditions.

A is the call ``_filter_scope`` makes for the filter *without* collections; it
must differ from the filter only in ``collections``, so S stays a subset of the
filter. B is the call with them. A double that answers on ``collections`` alone
cannot see a condition dropped from A, so this module asserts the payloads the
core receives.
"""

from __future__ import annotations

import json

import pytest

from app.file_matching import parse_search_filter
from tests.matching_core_double import MATCHING_PATH, MatchingCore, install
from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    patch_everywhere,
    search_index,
)
from tests.vec_partition_db import unit

ROUTES = ["/search", "/search/images"]

#: One value per filter key, so a dropped key is a visible difference.
FILTER = {
    "collections": ["Picks"],
    "tags": ["x", "y"],
    "kinds": ["video", "text"],
    "folders": ["Notes/2026"],
    "sizes": ["gt:1024"],
    "durations": ["lte:60"],
    "favorite": True,
    "liked": False,
    "exclude_tags": ["draft"],
    "exclude_kinds": ["pdf"],
    "exclude_folders": ["Notes/old"],
    "exclude_collections": ["Archive"],
    "has_tag": True,
    "has_collection": False,
}


@pytest.fixture()
def core(monkeypatch):
    return install(monkeypatch, MatchingCore(file_ids=["bonly", "alisted"]))


@pytest.fixture(autouse=True)
def _embedded(search_index, monkeypatch):  # noqa: F811
    patch_everywhere(
        monkeypatch, "embed_query",
        lambda *_a, **_k: unit(search_index.dim("vec_text"), 0),
    )
    patch_everywhere(
        monkeypatch, "embed_text_clip",
        lambda *_a, **_k: unit(search_index.dim("vec_clip"), 0),
    )


@pytest.fixture()
def post():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    def _post(path, params, body):
        return client.post(
            path, params=params, json=body, headers={"X-Lit-Drive": DRIVE},
        )

    return _post


def _sent_matching(core: MatchingCore) -> list[dict]:
    return [
        json.loads(request.content or b"{}")
        for request in core.matching_requests
        if request.url.path.endswith(MATCHING_PATH)
    ]


@pytest.mark.parametrize("path", ROUTES)
def test_spec_addon_115_i3_the_a_call_carries_every_other_condition(
    post, core, path,  # noqa: F811
):
    """A differs from the filter only in ``collections``; B is the filter."""
    params = {"q": "budget", "filter": json.dumps(FILTER)}
    if path == "/search":
        params["limit"] = 50

    response = post(path, params, {"file_ids": ["alisted"]})

    assert response.status_code == 200, response.text
    sent = _sent_matching(core)
    assert len(sent) == 2, sent

    expected = {"drive": DRIVE, **parse_search_filter(params["filter"]).model_dump()}
    a = [body for body in sent if body.get("collections") == []]
    b = [body for body in sent if body.get("collections") != []]

    assert len(a) == 1, sent
    assert len(b) == 1, sent
    assert a[0] == {**expected, "collections": []}
    assert b[0] == expected
