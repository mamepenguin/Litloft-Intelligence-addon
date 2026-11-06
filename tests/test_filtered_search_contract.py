"""SPEC-ADDON-047: ``filter`` on ``/search`` and ``/search/images`` resolves the
scope through the core, refuses incompatible parameters, and never falls back to
an unfiltered search.

Layer 1 wire shape and the Layer 2 validator table run through the addon's HTTP
surface, so the core's transport is doubled at ``httpx``.
"""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from tests.matching_core_double import MatchingCore, install
from tests.search_endpoint_db import (  # noqa: F401
    DRIVE,
    Seed,
    api,
    patch_everywhere,
    search_index,
    statements,
)
from tests.vec_partition_db import at_distance, unit

MATCHING_URL = "http://backend:8000/api/internal/files/matching"
ALL_KINDS = [
    "video",
    "image",
    "audio",
    "document",
    "archive",
    "other",
    "text",
    "pdf",
]


@pytest.fixture()
def core(monkeypatch):
    return install(monkeypatch, MatchingCore())


@pytest.fixture(autouse=True)
def _embedded(search_index, monkeypatch):  # noqa: F811
    patch_everywhere(
        monkeypatch, "embed_query",
        lambda *_a, **_k: _unit(search_index.dim("vec_text")),
    )
    patch_everywhere(
        monkeypatch, "embed_text_clip",
        lambda *_a, **_k: _unit(search_index.dim("vec_clip")),
    )


def _filter(**conditions) -> str:
    return json.dumps(conditions, ensure_ascii=False)


def _ok(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


def _ids(body) -> list[str]:
    return [r["file_id"] for r in body["results"]]


def _distance(score: float) -> float:
    return float(np.sqrt(2.0 * (1.0 - score)))


def _unit(dim: int) -> np.ndarray:
    return unit(dim, 0)


# ---------------------------------------------------------------------------
# Layer 1: wire shape of the new client
# ---------------------------------------------------------------------------


def test_spec_addon_047_i2_resolves_the_filter_with_one_internal_matching_call(
    api, search_index, core,  # noqa: F811
):
    seed = Seed(search_index)
    seed.file("wanted", drive=DRIVE)
    seed.file_metadata("wanted", "budget", "")
    core.file_ids = ["wanted"]

    body = _ok(api("/search", {
        "q": "budget",
        "filter": _filter(tags=["budget"], kinds=["markdown"], favorite=True),
    }))

    assert _ids(body) == ["wanted"]
    assert len(core.matching_requests) == 1
    request = core.matching_requests[0]
    assert request.method == "POST"
    assert str(request.url) == MATCHING_URL
    assert request.headers["content-type"].startswith("application/json")

    sent = json.loads(request.content)
    assert set(sent) <= {
        "drive", "tags", "kinds", "favorite", "liked", "folders", "collections",
        "sizes", "durations",
    }
    assert sent["drive"] == DRIVE
    assert sent.get("tags") == ["budget"]
    assert sent.get("kinds") == ["text"]
    assert sent.get("favorite") is True
    assert sent.get("liked") is None
    assert sent.get("folders") == []
    assert sent.get("collections") == []

    names = {name.lower() for name in request.headers}
    assert "authorization" not in names
    assert "cookie" not in names
    assert not any("secret" in name for name in names)


def test_spec_addon_047_i2_asks_the_core_once_per_request_not_from_a_cache(
    api, search_index, core,  # noqa: F811
):
    seed = Seed(search_index)
    seed.file("wanted", drive=DRIVE)
    seed.file_metadata("wanted", "budget", "")
    core.file_ids = ["wanted"]
    params = {"q": "budget", "filter": _filter(tags=["budget"])}

    for _ in range(2):
        assert _ok(api("/search", params))["results"][0]["file_id"] == "wanted"

    assert len(core.matching_requests) == 2


def test_spec_addon_047_i2_parses_file_ids_into_the_scope(
    api, search_index, core,  # noqa: F811
):
    seed = Seed(search_index)
    for file_id in ("inside", "outside"):
        seed.file(file_id, drive=DRIVE)
        seed.file_metadata(file_id, "budget", "")
    core.file_ids = ["inside"]

    assert _ids(_ok(api("/search", {
        "q": "budget", "filter": _filter(tags=["budget"]),
    }))) == ["inside"]


# ---------------------------------------------------------------------------
# Layer 2: the addon side of the filter-validator parity table
# ---------------------------------------------------------------------------


VALID_FILTERS = [
    pytest.param({}, id="empty-object"),
    pytest.param({"tags": []}, id="no-tags"),
    pytest.param({"tags": ["budget"]}, id="one-tag"),
    pytest.param({"tags": ["", "   ", "budget"]}, id="blank-tags-dropped"),
    pytest.param({"tags": ["  padded  "]}, id="tag-trimmed"),
    pytest.param({"tags": ["Budget", "budget"]}, id="tags-deduped"),
    pytest.param({"tags": [f"t{i}" for i in range(20)]}, id="twenty-tags"),
    pytest.param(
        {"tags": [f"t{i}" for i in range(20)] + ["T0"]},
        id="twenty-one-raw-twenty-after-dedup",
    ),
    pytest.param({"tags": ["x" * 200]}, id="tag-200-characters"),
    pytest.param({"kinds": ["markdown"]}, id="markdown-is-text"),
    pytest.param({"kinds": ["text", "markdown"]}, id="markdown-deduped"),
    pytest.param({"kinds": ALL_KINDS}, id="all-eight-kinds"),
    pytest.param({"favorite": True}, id="favorite-true"),
    pytest.param({"liked": False}, id="liked-false"),
    pytest.param({"favorite": None}, id="favorite-unset"),
]

INVALID_FILTERS = [
    pytest.param({"tags": [f"t{i}" for i in range(21)]}, id="twenty-one-tags"),
    pytest.param({"tags": ["x" * 201]}, id="tag-201-characters"),
    pytest.param(
        {"tags": [f"t{i}" for i in range(19)] + ["straße", "STRASSE"]},
        id="casefold-only-pair-is-two-tags",
    ),
    pytest.param(
        {"tags": ["\u0130" * 101, "i\u0307" * 101]},
        id="long-duplicate-checked-before-dedup",
    ),
    pytest.param({"kinds": ["bogus"]}, id="unknown-kind"),
    pytest.param({"tags": "budget"}, id="tags-not-a-list"),
    pytest.param({"tags": [1]}, id="tag-not-a-string"),
    pytest.param({"kinds": "text"}, id="kinds-not-a-list"),
    pytest.param({"favorite": "true"}, id="favorite-not-a-bool"),
    pytest.param({"liked": 1}, id="liked-not-a-bool"),
]

NOT_AN_OBJECT = [
    pytest.param("not-json", id="not-json"),
    pytest.param("[]", id="array"),
    pytest.param('"text"', id="string"),
    pytest.param("null", id="null"),
    pytest.param("123", id="number"),
]


@pytest.mark.parametrize("conditions", VALID_FILTERS)
def test_spec_addon_047_i10_accepts_a_well_formed_filter(
    api, search_index, core, conditions,  # noqa: F811
):
    core.file_ids = []

    response = api("/search", {"q": "budget", "filter": json.dumps(conditions)})

    assert _ok(response)["results"] == []
    assert len(core.matching_requests) == 1


@pytest.mark.parametrize("conditions", INVALID_FILTERS)
def test_spec_addon_047_i10_rejects_a_filter_the_core_request_model_rejects(
    api, search_index, core, conditions,  # noqa: F811
):
    response = api("/search", {"q": "budget", "filter": json.dumps(conditions)})

    assert response.status_code == 422, response.text
    assert core.matching_requests == []


@pytest.mark.parametrize("raw", NOT_AN_OBJECT)
def test_spec_addon_047_i10_rejects_a_filter_that_is_not_a_json_object(
    api, search_index, core, raw,  # noqa: F811
):
    response = api("/search", {"q": "budget", "filter": raw})

    assert response.status_code == 422, response.text
    assert core.matching_requests == []


def test_spec_addon_047_i10_normalises_tags_and_kinds_before_the_core_call(
    api, search_index, core,  # noqa: F811
):
    core.file_ids = []

    _ok(api("/search", {
        "q": "budget",
        "filter": json.dumps({
            "tags": ["", "  ", "Budget", "budget", "  work  "],
            "kinds": ["markdown", "text"],
        }),
    }))

    sent = json.loads(core.matching_requests[0].content)
    assert [tag.lower() for tag in sent["tags"]] == ["budget", "work"]
    assert sent["kinds"] == ["text"]


# ---------------------------------------------------------------------------
# SPEC-ADDON-047 I2: incompatible parameters
# ---------------------------------------------------------------------------


INCOMPATIBLE = [
    pytest.param("/search", {"include_scene_clip": "true"}, id="search-scene-clip"),
    pytest.param("/search", {"type": "pdf"}, id="search-type"),
    pytest.param("/search", {"mode": "recall"}, id="search-recall"),
    pytest.param("/search/images", {"type": "pdf"}, id="images-type"),
]


@pytest.mark.parametrize("path, extra", INCOMPATIBLE)
def test_spec_addon_047_i2_rejects_filter_with_an_incompatible_parameter(
    api, search_index, core, path, extra,  # noqa: F811
):
    response = api(path, {"q": "budget", "filter": _filter(tags=["budget"]), **extra})

    assert response.status_code == 422, response.text
    assert core.matching_requests == []


# ---------------------------------------------------------------------------
# SPEC-ADDON-047 I2: a core failure answers 503 and runs no channel
# ---------------------------------------------------------------------------


FAILURES = [
    pytest.param(dict(raises=httpx.ConnectError("no core")), id="connection"),
    pytest.param(dict(status=404, payload={"detail": "unknown drive"}), id="404"),
    pytest.param(dict(status=422, payload={"detail": "bad filter"}), id="422"),
    pytest.param(dict(status=500, payload={"detail": "boom"}), id="500"),
    pytest.param(dict(raw=b"not json"), id="body-not-json"),
    pytest.param(dict(payload={"file_ids": "abc"}), id="file-ids-not-a-list"),
    pytest.param(dict(payload={"file_ids": [1, 2]}), id="file-ids-not-strings"),
    pytest.param(dict(payload={"files": []}), id="no-file-ids-key"),
]


@pytest.mark.parametrize("path", ["/search", "/search/images"])
@pytest.mark.parametrize("failure", FAILURES)
def test_spec_addon_047_i2_a_failed_filter_answers_503_without_falling_back(
    api, search_index, monkeypatch, core, path, failure,  # noqa: F811
):
    for name, value in failure.items():
        setattr(core, name, value)

    ran: list[str] = []

    def _forbidden(name):
        def _spy(*_args, **_kwargs):
            ran.append(name)
            raise AssertionError(f"{name} ran")

        return _spy

    from app.routers import search as search_router

    monkeypatch.setattr(search_router, "execute_search", _forbidden("search"))
    monkeypatch.setattr(
        search_router, "execute_search_images", _forbidden("search_images"),
    )

    response = api(path, {"q": "budget", "filter": _filter(tags=["budget"])})

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "Filter unavailable"
    assert ran == []


def test_spec_addon_047_i2_an_empty_scope_answers_no_results_without_running_a_channel(
    api, search_index, monkeypatch, core, statements,  # noqa: F811
):
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("embedding ran for an empty scope")

    patch_everywhere(monkeypatch, "embed_query", _forbidden)
    patch_everywhere(monkeypatch, "embed_text_clip", _forbidden)

    core.file_ids = []
    body = _ok(api("/search", {
        "q": "budget", "filter": _filter(tags=["nothing"]),
    }))

    assert body["results"] == []
    assert body["total"] == 0
    assert not [s for s in statements.statements if "MATCH" in s.upper()]


def test_spec_addon_047_i2_an_empty_scope_answers_images_without_running_a_channel(
    api, search_index, monkeypatch, core, statements,  # noqa: F811
):
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("embedding ran for an empty scope")

    patch_everywhere(monkeypatch, "embed_query", _forbidden)
    patch_everywhere(monkeypatch, "embed_text_clip", _forbidden)

    core.file_ids = []
    body = _ok(api("/search/images", {
        "q": "budget", "filter": _filter(tags=["nothing"]),
    }))

    assert body["results"] == []
    assert body["total"] == 0
    assert not [s for s in statements.statements if "MATCH" in s.upper()]


# ---------------------------------------------------------------------------
# SPEC-ADDON-047 I5: unfiltered requests keep today's path
# ---------------------------------------------------------------------------


def test_spec_addon_047_i5_unfiltered_search_never_resolves_a_filter(
    api, search_index, core,  # noqa: F811
):
    dim = search_index.dim("vec_text")
    for file_id, score, axis in (("near", 0.95, 10), ("far", 0.80, 20)):
        search_index.add_file(file_id, drive=DRIVE)
        search_index.add_embedding(f"e_{file_id}", file_id, "text_content", "vec_text")
        search_index.put_vector(
            "vec_text", f"e_{file_id}",
            at_distance(dim, _distance(score), axis), partition="text_content",
        )
    core.file_ids = ["far"]

    body = _ok(api("/search", {"q": "budget"}))

    assert _ids(body) == ["near"]
    assert core.matching_requests == []


def test_spec_addon_047_i5_unfiltered_image_search_never_resolves_a_filter(
    api, search_index, core,  # noqa: F811
):
    dim = search_index.dim("vec_clip")
    for file_id, score, axis in (("near", 0.95, 10), ("far", 0.0, 20)):
        search_index.add_file(file_id, drive=DRIVE, mime_type="image/jpeg")
        search_index.add_embedding(
            f"e_{file_id}", file_id, "clip_thumbnail", "vec_clip",
        )
        search_index.put_vector(
            "vec_clip", f"e_{file_id}",
            at_distance(dim, _distance(score), axis), partition="clip_thumbnail",
        )
    core.file_ids = ["far"]

    body = _ok(api("/search/images", {"q": "budget"}))

    assert _ids(body) == ["near"]
    assert core.matching_requests == []
