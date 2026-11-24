"""SPEC-ADDON-177 I1: the meaning filter's new keys on the POST path.

A narrowed ``/search`` makes two ``/files/matching`` calls: the filter without
collections and the filter as written. Both are copies of the parsed filter, so
both must carry a ``has_`` flag the filter sets and neither may gain a key it
leaves unset.
"""

from __future__ import annotations

import json

import pytest

from tests.matching_core_double import MATCHING_PATH, MatchingCore, install
from tests.search_endpoint_db import DRIVE, patch_everywhere, search_index  # noqa: F401
from tests.vec_partition_db import unit

UNSET_KEYS = (
    "exclude_tags",
    "exclude_kinds",
    "exclude_folders",
    "exclude_collections",
    "has_collection",
)


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


def test_spec_addon_177_i1_both_calls_carry_a_set_flag_and_no_unset_key(post, core):
    params = {
        "q": "budget",
        "filter": json.dumps(
            {"collections": ["Picks"], "tags": ["x"], "has_tag": False}
        ),
    }

    response = post("/search", params, {"file_ids": ["alisted"]})

    assert response.status_code == 200, response.text
    sent = _sent_matching(core)
    assert len(sent) == 2, sent
    assert all(body["has_tag"] is False for body in sent), sent
    for body in sent:
        for key in UNSET_KEYS:
            assert key not in body, (key, sent)
