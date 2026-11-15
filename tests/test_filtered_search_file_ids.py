"""SPEC-ADDON-115: ``POST /search`` and ``POST /search/images`` narrow the filter scope
by a body ``file_ids``: S = B ∪ (A ∩ file_ids) when the filter names collections and
A ∩ file_ids when it does not, A being the core's ``files/matching`` answer for the
filter without collections and B the same call with them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import numpy as np
import pytest

from tests.matching_core_double import MATCHING_PATH, MatchingCore, install
from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    api,
    patch_everywhere,
    search_index,
    statements,
)
from tests.vec_partition_db import at_distance, unit

MAX_IDS = 10_000
MANIFEST = Path(__file__).resolve().parents[1] / "manifest.json"
ROUTES = ["/search", "/search/images"]

B_ONLY = "bonly"
A_LISTED = "alisted"
A_UNLISTED = "aunlisted"
LISTED_NOT_A = "listednota"
NEITHER = "neither"


class SplitCore(MatchingCore):
    """Answers ``files/matching`` with ``with_collections`` when the request names
    collections and with ``without_collections`` otherwise."""

    def __init__(self, without_collections: list[str], with_collections: list[str]):
        super().__init__()
        self.without_collections = without_collections
        self.with_collections = with_collections

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(MATCHING_PATH) and self.raises is None:
            self.matching_requests.append(request)
            sent = json.loads(request.content or b"{}")
            ids = self.with_collections if sent.get("collections") else self.without_collections
            return httpx.Response(200, json={"file_ids": list(ids)})
        return super().handle(request)


@pytest.fixture()
def core(monkeypatch):
    return install(
        monkeypatch,
        SplitCore(
            without_collections=[B_ONLY, A_LISTED, A_UNLISTED],
            with_collections=[B_ONLY],
        ),
    )


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
def post(search_index):  # noqa: F811
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    def _post(path, params=None, body=None, *, raw_body: bool = False):
        headers = {"X-Lit-Drive": DRIVE}
        if raw_body:
            return client.post(path, params=params or {}, headers=headers)
        return client.post(path, params=params or {}, json=body, headers=headers)

    return _post


def _distance(score: float) -> float:
    return float(np.sqrt(2.0 * (1.0 - score)))


def _seed_every_region(index) -> None:
    """One strong hit per region, on the channel each route ranks."""
    regions = [B_ONLY, A_LISTED, A_UNLISTED, LISTED_NOT_A, NEITHER]
    text_dim = index.dim("vec_text")
    clip_dim = index.dim("vec_clip")
    rows = []
    for i, file_id in enumerate(regions):
        score = 0.95 - i * 0.001
        rows.append((
            file_id, DRIVE, True, f"t_{file_id}", "text_content", "vec_text",
            at_distance(text_dim, _distance(score), 100 + i),
        ))
        rows.append((
            file_id, DRIVE, True, f"c_{file_id}", "clip_thumbnail", "vec_clip",
            at_distance(clip_dim, _distance(score), 100 + i),
        ))
    index.add_many(rows)


def _filter(**conditions) -> str:
    return json.dumps(conditions, ensure_ascii=False)


def _ok(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


def _ids(body) -> set[str]:
    return {r["file_id"] for r in body["results"]}


def _params(path: str, **filter_conditions) -> dict:
    params = {"q": "budget", "filter": _filter(**filter_conditions)}
    if path == "/search":
        params["limit"] = 50
    return params


# ---------------------------------------------------------------------------
# SPEC-ADDON-115 step 2, I3: the scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ROUTES)
def test_spec_addon_115_i3_with_collections_the_scope_is_b_or_listed_a(
    post, search_index, core, path,  # noqa: F811
):
    _seed_every_region(search_index)

    body = _ok(post(
        path,
        _params(path, tags=["x"], collections=["Picks"]),
        {"file_ids": [A_LISTED, LISTED_NOT_A]},
    ))

    assert _ids(body) == {B_ONLY, A_LISTED}


@pytest.mark.parametrize("path", ROUTES)
def test_spec_addon_115_i3_without_collections_the_scope_is_listed_a(
    post, search_index, core, path,  # noqa: F811
):
    _seed_every_region(search_index)

    body = _ok(post(
        path,
        _params(path, tags=["x"]),
        {"file_ids": [A_LISTED, LISTED_NOT_A]},
    ))

    assert _ids(body) == {A_LISTED}


@pytest.mark.parametrize("path", ROUTES)
def test_spec_addon_115_i3_empty_file_ids_with_collections_leave_b(
    post, search_index, core, path,  # noqa: F811
):
    _seed_every_region(search_index)

    body = _ok(post(
        path, _params(path, collections=["Picks"]), {"file_ids": []},
    ))

    assert _ids(body) == {B_ONLY}


@pytest.mark.parametrize("path", ROUTES)
def test_spec_addon_115_i3_duplicates_change_nothing(
    post, search_index, core, path,  # noqa: F811
):
    _seed_every_region(search_index)

    body = _ok(post(
        path,
        _params(path, tags=["x"]),
        {"file_ids": [A_LISTED, A_LISTED, LISTED_NOT_A, LISTED_NOT_A]},
    ))

    assert _ids(body) == {A_LISTED}


@pytest.mark.parametrize("path", ROUTES)
def test_spec_addon_115_i3_ten_thousand_ids_are_accepted(
    post, search_index, core, path,  # noqa: F811
):
    _seed_every_region(search_index)
    file_ids = [A_LISTED] + [f"none{i:08d}" for i in range(MAX_IDS - 1)]

    body = _ok(post(path, _params(path, tags=["x"]), {"file_ids": file_ids}))

    assert _ids(body) == {A_LISTED}


# ---------------------------------------------------------------------------
# SPEC-ADDON-115 step 3, I3: an empty scope and a core failure
# ---------------------------------------------------------------------------


EMPTY_SCOPES = [
    pytest.param({"tags": ["x"]}, [], id="no-collections-empty-ids"),
    pytest.param({"tags": ["x"]}, [LISTED_NOT_A, NEITHER], id="no-collections-disjoint-ids"),
]


@pytest.mark.parametrize("path", ROUTES)
@pytest.mark.parametrize("conditions, file_ids", EMPTY_SCOPES)
def test_spec_addon_115_i3_an_empty_scope_answers_the_empty_response(
    post, search_index, core, monkeypatch, path, conditions, file_ids,  # noqa: F811
):
    _seed_every_region(search_index)

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("embedding ran for an empty scope")

    patch_everywhere(monkeypatch, "embed_query", _forbidden)
    patch_everywhere(monkeypatch, "embed_text_clip", _forbidden)

    body = _ok(post(path, _params(path, **conditions), {"file_ids": file_ids}))

    assert body["results"] == []
    assert body["total"] == 0


@pytest.mark.parametrize("path", ROUTES)
def test_spec_addon_115_i3_an_empty_scope_with_collections_answers_the_empty_response(
    post, search_index, monkeypatch, path,  # noqa: F811
):
    install(monkeypatch, SplitCore(without_collections=[A_UNLISTED], with_collections=[]))
    _seed_every_region(search_index)

    body = _ok(post(
        path, _params(path, collections=["Picks"]), {"file_ids": [LISTED_NOT_A]},
    ))

    assert body["results"] == []
    assert body["total"] == 0


@pytest.mark.parametrize("path", ROUTES)
@pytest.mark.parametrize("collections", [[], ["Picks"]], ids=["no-collections", "collections"])
def test_spec_addon_115_i3_a_failed_matching_call_answers_503(
    post, search_index, core, path, collections,  # noqa: F811
):
    _seed_every_region(search_index)
    core.raises = httpx.ConnectError("no core")

    response = post(
        path, _params(path, tags=["x"], collections=collections), {"file_ids": [A_LISTED]},
    )

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "Filter unavailable"


# ---------------------------------------------------------------------------
# SPEC-ADDON-115 step 2 and failure cases: 422
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ROUTES)
def test_spec_addon_115_i3_a_post_without_filter_is_422(
    post, search_index, core, path,  # noqa: F811
):
    _seed_every_region(search_index)

    with_filter = post(path, _params(path, tags=["x"]), {"file_ids": [A_LISTED]})
    without = post(path, {"q": "budget"}, {"file_ids": [A_LISTED]})

    assert with_filter.status_code == 200, with_filter.text
    assert without.status_code == 422, without.text


INCOMPATIBLE = [
    pytest.param("/search", {"include_scene_clip": "true"}, id="search-scene-clip"),
    pytest.param("/search", {"type": "pdf"}, id="search-type"),
    pytest.param("/search", {"mode": "recall"}, id="search-recall"),
    pytest.param("/search/images", {"type": "pdf"}, id="images-type"),
]


@pytest.mark.parametrize("path, extra", INCOMPATIBLE)
def test_spec_addon_115_i3_filter_with_an_incompatible_parameter_is_422(
    post, search_index, core, path, extra,  # noqa: F811
):
    _seed_every_region(search_index)

    good = post(path, _params(path, tags=["x"]), {"file_ids": [A_LISTED]})
    response = post(path, {**_params(path, tags=["x"]), **extra}, {"file_ids": [A_LISTED]})

    assert good.status_code == 200, good.text
    assert response.status_code == 422, response.text


BAD_BODIES = [
    pytest.param(None, id="missing-body"),
    pytest.param({}, id="missing-key"),
    pytest.param({"file_ids": "abc"}, id="not-a-list"),
    pytest.param({"file_ids": None}, id="null"),
    pytest.param({"file_ids": [A_LISTED, ""]}, id="an-empty-string"),
    pytest.param({"file_ids": [f"none{i:08d}" for i in range(MAX_IDS + 1)]}, id="ten-thousand-and-one"),
]


@pytest.mark.parametrize("path", ROUTES)
@pytest.mark.parametrize("body", BAD_BODIES)
def test_spec_addon_115_i3_a_bad_file_ids_body_is_422(
    post, search_index, core, path, body,  # noqa: F811
):
    _seed_every_region(search_index)

    good = post(path, _params(path, tags=["x"]), {"file_ids": [A_LISTED]})
    response = (
        post(path, _params(path, tags=["x"]), raw_body=True)
        if body is None
        else post(path, _params(path, tags=["x"]), body)
    )

    assert good.status_code == 200, good.text
    assert response.status_code == 422, response.text


# ---------------------------------------------------------------------------
# SPEC-ADDON-115 step 1, I3: the manifest gives the POST the GET's gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ROUTES)
def test_spec_addon_115_i3_the_manifest_gives_post_the_gets_gates(path):
    routes = json.loads(MANIFEST.read_text())["proxy"]["routes"]
    entries = [r for r in routes if r.get("path") == path]
    get = [r for r in entries if "GET" in r.get("methods", [])]
    posts = [r for r in entries if "POST" in r.get("methods", [])]

    assert len(get) == 1
    assert len(posts) == 1
    for key in ("pre_check", "response_filter", "timeout"):
        assert posts[0].get(key) == get[0].get(key), key
    assert posts[0]["pre_check"] == {"type": "addon_feature", "feature": "search"}


# ---------------------------------------------------------------------------
# SPEC-ADDON-115 I3 / error behavior: no log line carries the query or the scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ROUTES)
def test_spec_addon_115_i3_no_log_line_carries_the_query_or_an_id(
    post, api, search_index, core, caplog, path,  # noqa: F811
):
    _seed_every_region(search_index)
    caplog.set_level(logging.DEBUG)
    needles = ("uniquequeryword", "uniquetagword", "uniquecollectionword", A_LISTED, LISTED_NOT_A)
    params = {
        **_params(path, tags=["uniquetagword"], collections=["uniquecollectionword"]),
        "q": "uniquequeryword",
    }

    def _logged() -> set[str]:
        text = "\n".join(r.getMessage() for r in caplog.records)
        caplog.clear()
        return {n for n in needles if n in text}

    core.raises = None
    assert api(path, params).status_code == 200
    core.raises = httpx.ConnectError("no core")
    assert api(path, params).status_code == 503
    by_get = _logged()

    core.raises = None
    ok = post(path, params, {"file_ids": [A_LISTED, LISTED_NOT_A]})
    core.raises = httpx.ConnectError("no core")
    failed = post(path, params, {"file_ids": [A_LISTED, LISTED_NOT_A]})
    by_post = _logged()

    assert ok.status_code == 200, ok.text
    assert failed.status_code == 503, failed.text
    assert by_post <= by_get
    assert A_LISTED not in by_post and LISTED_NOT_A not in by_post
