"""SPEC-ADDON-047: with a ``filter`` every retrieval channel is constrained to the
core's file ids, the relative cutoffs are measured within that scope, the flat-set
checks do not run on the scoped path, and the vector channels switch from exact
ranking to the widened KNN at 20,000 scope rows.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from app import search as search_mod
from app.config import SearchConfig
from tests.matching_core_double import MatchingCore, install
from tests.search_endpoint_db import (  # noqa: F401
    Seed,
    api,
    patch_everywhere,
    search_index,
)
from tests.vec_partition_db import at_distance, unit

HOME = "family"
CHANNELS = [
    "fts_files",
    "fts_transcripts",
    "fts_text_content",
    "fts_retrieval_keywords",
    "text_vector",
    "clip_thumbnail",
]
FTS_CHANNELS = CHANNELS[:4]


@pytest.fixture()
def core(monkeypatch):
    return install(monkeypatch, MatchingCore())


@pytest.fixture()
def embedded(search_index, monkeypatch):  # noqa: F811
    patch_everywhere(
        monkeypatch, "embed_query",
        lambda *_a, **_k: unit(search_index.dim("vec_text"), 0),
    )
    patch_everywhere(
        monkeypatch, "embed_text_clip",
        lambda *_a, **_k: unit(search_index.dim("vec_clip"), 0),
    )


def _use_config(monkeypatch, **kwargs) -> None:
    monkeypatch.setattr(
        search_mod, "settings",
        dataclasses.replace(
            search_mod.settings, search=SearchConfig(**kwargs),
        ),
    )


def _distance(score: float) -> float:
    return float(np.sqrt(2.0 * (1.0 - score)))


def _filter(**conditions) -> str:
    return json.dumps(conditions)


def _ok(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


def _ids(body) -> list[str]:
    return [r["file_id"] for r in body["results"]]


def _text_rows(index, prefix, scores, *, axis0=100, etype="text_content"):
    dim = index.dim("vec_text")
    return [
        (
            f"{prefix}{i}", HOME, True, f"e_{prefix}{i}", etype, "vec_text",
            at_distance(dim, _distance(score), axis0 + i),
        )
        for i, score in enumerate(scores)
    ]


def _clip_rows(index, prefix, scores, *, axis0=100):
    dim = index.dim("vec_clip")
    return [
        (
            f"{prefix}{i}", HOME, True, f"e_{prefix}{i}", "clip_thumbnail", "vec_clip",
            at_distance(dim, _distance(score), axis0 + i),
        )
        for i, score in enumerate(scores)
    ]


def _long(term: str) -> str:
    return " ".join([term] * 100)


def _seed_fts(seed, index, channel: str) -> None:
    seed.file("inscope", drive=HOME)
    if channel == "fts_files":
        seed.file_metadata("inscope", "budget", "")
    elif channel == "fts_transcripts":
        seed.transcript("inscope", 0, "budget", times=(0.0, 1.0))
    elif channel == "fts_text_content":
        seed.text_chunk("inscope", 0, "budget")
    else:
        index.sql(
            "INSERT INTO fts_retrieval_keywords(file_id, keywords) VALUES (:f, :k)",
            f="inscope", k="budget",
        )

    for i in range(60):
        file_id = f"out{i}"
        seed.file(file_id, drive=HOME)
        if channel == "fts_files":
            seed.file_metadata(file_id, _long("budget"), "")
        elif channel == "fts_transcripts":
            seed.transcript(file_id, 0, _long("budget"), times=(0.0, 1.0))
        elif channel == "fts_text_content":
            seed.text_chunk(file_id, 0, _long("budget"))
        else:
            index.sql(
                "INSERT INTO fts_retrieval_keywords(file_id, keywords) "
                "VALUES (:f, :k)",
                f=file_id, k=_long("budget"),
            )


def _seed_vector(index, channel: str) -> None:
    if channel == "text_vector":
        rows = _text_rows(index, "out", [0.999 - i * 1e-5 for i in range(60)])
        rows += _text_rows(index, "in", [0.90], axis0=500)
    else:
        rows = _clip_rows(index, "out", [0.999 - i * 1e-5 for i in range(60)])
        rows += _clip_rows(index, "in", [0.90], axis0=500)
    index.add_many(rows)


def _in_id(channel: str) -> str:
    return "inscope" if channel in FTS_CHANNELS else "in0"


# ---------------------------------------------------------------------------
# SPEC-ADDON-047 I2 / I3: every channel ranks only within the scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel", CHANNELS)
def test_spec_addon_047_i3_each_channel_ranks_only_within_the_scope(
    api, search_index, core, embedded, monkeypatch, channel,  # noqa: F811
):
    _use_config(monkeypatch, score_gap_threshold=0.0, score_cutoff_margin=10.0)
    if channel in FTS_CHANNELS:
        _seed_fts(Seed(search_index), search_index, channel)
    else:
        _seed_vector(search_index, channel)

    core.file_ids = [_in_id(channel)]

    body = _ok(api("/search", {
        "q": "budget", "filter": _filter(tags=["x"]),
    }))

    assert _ids(body) == [_in_id(channel)]


@pytest.mark.parametrize("path", ["/search", "/search/images"])
def test_spec_addon_047_i2_only_the_scope_is_returned_on_either_route(
    api, search_index, core, embedded, monkeypatch, path,  # noqa: F811
):
    _use_config(monkeypatch, score_gap_threshold=0.0, score_cutoff_margin=10.0)
    search_index.add_many(
        _text_rows(search_index, "textin", [0.95])
        + _text_rows(search_index, "textout", [0.99], axis0=300)
        + _clip_rows(search_index, "clipin", [0.95])
        + _clip_rows(search_index, "clipout", [0.99], axis0=300)
    )
    core.file_ids = (
        ["textin0", "clipin0"] if path == "/search" else ["clipin0"]
    )
    params = {"q": "budget", "filter": _filter(tags=["x"])}
    if path == "/search":
        params["limit"] = 50

    body = _ok(api(path, params))

    assert set(_ids(body)) <= set(core.file_ids)
    assert _ids(body)


# ---------------------------------------------------------------------------
# SPEC-ADDON-047 I3: relative cutoffs are measured inside the scope
# ---------------------------------------------------------------------------


def test_spec_addon_047_i3_text_margin_is_measured_from_the_best_in_scope_score(
    api, search_index, core, embedded, monkeypatch,  # noqa: F811
):
    _use_config(monkeypatch, score_gap_threshold=0.0)
    search_index.add_many(
        _text_rows(search_index, "out", [0.995 - i * 1e-5 for i in range(60)])
        + _text_rows(search_index, "top", [0.90], axis0=500)
        + _text_rows(search_index, "next", [0.87], axis0=510)
    )
    core.file_ids = ["top0", "next0"]

    body = _ok(api("/search", {
        "q": "budget", "filter": _filter(tags=["x"]),
    }))

    assert set(_ids(body)) == {"top0", "next0"}


def test_spec_addon_047_i3_clip_margin_is_measured_from_the_best_in_scope_score(
    api, search_index, core, embedded, monkeypatch,  # noqa: F811
):
    _use_config(monkeypatch, score_gap_threshold=0.0)
    search_index.add_many(
        _clip_rows(search_index, "out", [0.995 - i * 1e-5 for i in range(60)])
        + _clip_rows(search_index, "top", [0.90], axis0=500)
        + _clip_rows(search_index, "next", [0.87], axis0=510)
    )
    core.file_ids = ["top0", "next0"]

    body = _ok(api("/search", {
        "q": "budget", "filter": _filter(tags=["x"]),
    }))

    assert set(_ids(body)) == {"top0", "next0"}


def test_spec_addon_047_i3_the_combined_ratio_cutoff_counts_only_the_scope(
    api, search_index, core, embedded, monkeypatch,  # noqa: F811
):
    _use_config(
        monkeypatch, score_gap_threshold=0.0, score_cutoff_margin=10.0,
        min_score_text=0.0,
    )
    search_index.add_many(
        _text_rows(search_index, "out", [0.99])
        + _text_rows(search_index, "strong", [0.90], axis0=500)
        + _text_rows(search_index, "weak", [0.47], axis0=510)
    )
    core.file_ids = ["strong0", "weak0"]

    body = _ok(api("/search", {
        "q": "budget", "filter": _filter(tags=["x"]),
    }))

    assert set(_ids(body)) == {"strong0", "weak0"}


# ---------------------------------------------------------------------------
# SPEC-ADDON-047 I3: the text channel's flat-set checks are off when scoped
# ---------------------------------------------------------------------------


def test_spec_addon_047_i3_scoped_text_channel_does_not_run_the_gap_check(
    api, search_index, core, embedded,  # noqa: F811
):
    scores = [0.90 + i * 0.0001 for i in range(10)]
    search_index.add_many(_text_rows(search_index, "flat", scores))
    core.file_ids = [f"flat{i}" for i in range(10)]

    body = _ok(api("/search", {
        "q": "budget", "filter": _filter(tags=["x"]),
    }))

    assert set(_ids(body)) == {f"flat{i}" for i in range(10)}


def test_spec_addon_047_i3_scoped_text_channel_does_not_run_the_cv_check(
    api, search_index, core, embedded,  # noqa: F811
):
    scores = [0.93] + [0.9005 + i * 1e-6 for i in range(9)]
    search_index.add_many(_text_rows(search_index, "cv", scores))
    core.file_ids = [f"cv{i}" for i in range(10)]

    body = _ok(api("/search", {
        "q": "budget", "filter": _filter(tags=["x"]),
    }))

    assert set(_ids(body)) == {f"cv{i}" for i in range(10)}


# ---------------------------------------------------------------------------
# SPEC-ADDON-047 I4: exact ranking up to 20,000 scope rows, widened KNN above
# ---------------------------------------------------------------------------


def test_spec_addon_047_i4_scoped_text_vectors_are_exact_at_20000_rows_and_agree_above_it(
    api, search_index, core, embedded, monkeypatch,  # noqa: F811
):
    _use_config(monkeypatch, score_gap_threshold=0.0, score_cutoff_margin=10.0)
    n = 20_000
    # The 50 nearest scope rows are the expected ids; the rest sit further out.
    scores = [0.99 - i * 1e-3 for i in range(50)] + [0.90] * (n + 1 - 50)
    search_index.add_many(_text_rows(search_index, "s", scores, axis0=0))
    expected = [f"s{i}" for i in range(50)]

    core.file_ids = [f"s{i}" for i in range(n)]
    assert _ids(_ok(api("/search", {
        "q": "budget", "filter": _filter(tags=["x"]), "limit": 50,
    }))) == expected

    core.file_ids = [f"s{i}" for i in range(n + 1)]
    assert _ids(_ok(api("/search", {
        "q": "budget", "filter": _filter(tags=["x"]), "limit": 50,
    }))) == expected

    # 4,200 out-of-scope embeddings, all nearer than the scope's best row. A
    # KNN that is not constrained to the ids cannot reach the scope; the exact
    # path reads the scope's own rows.
    search_index.add_many(_text_rows(
        search_index, "out", [0.9999 - i * 1e-7 for i in range(4200)], axis0=20_100,
    ))
    core.file_ids = [f"s{i}" for i in range(n)]
    assert _ids(_ok(api("/search", {
        "q": "budget", "filter": _filter(tags=["x"]), "limit": 50,
    }))) == expected
