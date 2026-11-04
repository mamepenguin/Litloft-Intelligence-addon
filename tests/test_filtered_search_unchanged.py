"""SPEC-ADDON-047 I11: Ask and Find keep the post-filter ``file_id_scope`` path.

The new scope constraint is an argument only the ``filter`` path sets; an
existing ``file_id_scope`` caller must still rank unscoped and narrow afterwards.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from app import search as search_mod
from app.config import SearchConfig
from app.rag import retriever
from app.search import SearchResponse
from tests.search_endpoint_db import patch_everywhere, search_index  # noqa: F401
from tests.vec_partition_db import at_distance, unit

HOME = "family"


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


def test_spec_addon_047_i11_file_id_scope_still_ranks_unscoped_and_filters_after(
    search_index, monkeypatch, embedded,  # noqa: F811
):
    monkeypatch.setattr(
        search_mod, "settings",
        dataclasses.replace(
            search_mod.settings,
            search=SearchConfig(
                score_gap_threshold=0.0, score_cutoff_margin=10.0, min_score_text=0.0,
            ),
        ),
    )
    search_index.add_many(
        _rows(search_index, "out", [0.99 - i * 1e-5 for i in range(60)])
        + _rows(search_index, "in", [0.50], axis0=500)
    )

    response = search_mod.search("budget", drive=HOME, file_id_scope=["in0"])

    assert response.results == ()


@pytest.mark.asyncio
async def test_spec_addon_047_i11_ask_retrieval_passes_the_scope_unchanged(monkeypatch):
    from unittest.mock import MagicMock

    spy = MagicMock(return_value=SearchResponse(
        results=(), total=0, indexed_files=0, service_version="test",
    ))
    monkeypatch.setattr(retriever, "search", spy)

    await retriever.retrieve_with_keywords(
        keywords="kw", top_k=5, credential=None, file_id_scope=["a", "b"],
    )

    kwargs = spy.call_args.kwargs
    assert kwargs["file_id_scope"] == ["a", "b"]
    assert set(kwargs) == {
        "limit",
        "file_type",
        "drive",
        "mode",
        "semantic_query",
        "file_id_scope",
        "required",
        "include_scene_clip",
    }
