"""SPEC-ADDON-054 I1: a scoped retrieval ranks within the effective set.

``retrieve_with_keywords`` gains a keyword-only ``file_id_constraint`` and
forwards it to ``app.search.search()``; an unscoped call sends exactly the
keyword set it sent before. A 60-file drive whose out-of-scope files all
outscore the one in-scope file still retrieves the in-scope file.
"""

from __future__ import annotations

import dataclasses
import sys
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app import search as search_mod  # noqa: E402
from app.config import SearchConfig  # noqa: E402
from app.rag import retriever  # noqa: E402
from app.rag.retriever import retrieve_with_keywords  # noqa: E402
from app.search import SearchResponse  # noqa: E402
from tests.search_endpoint_db import patch_everywhere, search_index  # noqa: F401,E402
from tests.vec_partition_db import at_distance, unit  # noqa: E402

HOME = "family"


def _empty_response() -> SearchResponse:
    return SearchResponse(
        results=(), total=0, indexed_files=0, service_version="test"
    )


async def _run(monkeypatch, **kwargs):
    search_spy = MagicMock(return_value=_empty_response())
    monkeypatch.setattr("app.rag.retriever.search", search_spy)
    monkeypatch.setattr(
        "app.rag.retriever._filter_file_ids_via_internal_api",
        AsyncMock(return_value=set()),
    )
    monkeypatch.setattr(
        "app.rag.retriever._get_indexed_files_meta", lambda fids: {}
    )
    await retrieve_with_keywords(
        keywords="kw", top_k=5, credential=None, **kwargs
    )
    return search_spy


@pytest.mark.asyncio
async def test_constraint_reaches_search_only_when_it_is_given(monkeypatch):
    unscoped = await _run(monkeypatch)
    assert "file_id_constraint" not in unscoped.call_args.kwargs

    scoped = await _run(monkeypatch, file_id_constraint=["in0"])
    assert set(scoped.call_args.kwargs["file_id_constraint"]) == {"in0"}
    assert scoped.call_args.kwargs["file_id_scope"] is None


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


def _distance(score: float) -> float:
    return float(np.sqrt(2.0 * (1.0 - score)))


def _rows(index, prefix, scores, *, axis0=100):
    dim = index.dim("vec_text")
    return [
        (
            f"{prefix}{i}", HOME, True, f"e_{prefix}{i}", "text_content", "vec_text",
            at_distance(dim, _distance(score), axis0 + i),
        )
        for i, score in enumerate(scores)
    ]


@pytest.mark.asyncio
async def test_sixty_out_of_scope_hits_do_not_hide_the_in_scope_file(
    search_index, monkeypatch, embedded  # noqa: F811
):
    monkeypatch.setattr(
        search_mod, "settings",
        dataclasses.replace(
            search_mod.settings,
            search=SearchConfig(
                score_gap_threshold=0.0,
                score_cutoff_margin=10.0,
                min_score_text=0.0,
            ),
        ),
    )
    # ``recall`` mode reads its thresholds from the frozen module-level
    # parameters, not from ``settings.search``; zero the similarity floor so
    # the one in-scope file at 0.50 is a candidate at all.
    monkeypatch.setattr(
        search_mod,
        "_RECALL_PARAMS",
        dataclasses.replace(
            search_mod._RECALL_PARAMS,
            min_score_text=0.0,
            score_cutoff_margin=10.0,
        ),
    )
    search_index.add_many(
        _rows(search_index, "out", [0.99 - i * 1e-5 for i in range(60)])
        + _rows(search_index, "in", [0.50], axis0=500)
    )
    monkeypatch.setattr(
        retriever, "_filter_file_ids_via_internal_api",
        AsyncMock(return_value={"in0"}),
    )
    monkeypatch.setattr(retriever, "_get_indexed_files_meta", lambda fids: {})

    results = await retrieve_with_keywords(
        keywords="budget",
        top_k=5,
        credential=None,
        drive=HOME,
        file_id_constraint=["in0"],
    )

    assert [result.file_id for result in results] == ["in0"]
