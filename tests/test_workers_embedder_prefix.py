"""Unit tests for ``app.workers.embedder`` model-registry helpers.

Covers the pure-Python lookup tables and ``_detect_prefix_family``
that decide how a configured ``models.text_embedding`` string is
mapped to a (query, passage) prefix pair. These functions are import-
time cheap (no sentence-transformers load) so the conftest ML stubs
are sufficient.
"""

from __future__ import annotations

from app.workers.embedder import (
    _MODEL_DIMS,
    _MODEL_PREFIXES,
    _detect_prefix_family,
)


# ---------------------------------------------------------------------------
# _MODEL_DIMS allowlist shape
# ---------------------------------------------------------------------------


def test_model_dims_includes_granite_r2_multilingual_sizes() -> None:
    """Granite Embedding R2 multilingual 97m + 311m must be in the
    allowlist with their published dimensions (384 / 768)."""
    assert (
        _MODEL_DIMS["ibm-granite/granite-embedding-97m-multilingual-r2"] == 384
    )
    assert (
        _MODEL_DIMS["ibm-granite/granite-embedding-311m-multilingual-r2"] == 768
    )


def test_model_dims_does_not_expose_e5_family() -> None:
    """The multilingual-e5 series was replaced by Granite R2 (spec
    2026-05-20-granite-embedding-r2-adoption). The allowlist must
    not surface them — they would silently round-trip an override
    written by an older client."""
    e5_ids = {
        "intfloat/multilingual-e5-small",
        "intfloat/multilingual-e5-base",
        "intfloat/multilingual-e5-large",
    }
    assert e5_ids.isdisjoint(set(_MODEL_DIMS))


def test_model_dims_retains_ruri_japanese_family() -> None:
    """Japanese-specialised ruri-v3 line is unchanged by the Granite
    swap; users on Japanese-only data keep that path."""
    assert _MODEL_DIMS["cl-nagoya/ruri-v3-30m"] == 256
    assert _MODEL_DIMS["cl-nagoya/ruri-v3-130m"] == 768
    assert _MODEL_DIMS["cl-nagoya/ruri-v3-310m"] == 1024


# ---------------------------------------------------------------------------
# _detect_prefix_family
# ---------------------------------------------------------------------------


def test_detect_prefix_family_granite_r2() -> None:
    assert (
        _detect_prefix_family(
            "ibm-granite/granite-embedding-97m-multilingual-r2"
        )
        == "granite"
    )
    assert (
        _detect_prefix_family(
            "ibm-granite/granite-embedding-311m-multilingual-r2"
        )
        == "granite"
    )


def test_detect_prefix_family_ruri_wins_over_granite() -> None:
    """If a hypothetical model name contained both substrings,
    ``ruri`` must win — its Japanese prefixes are mandatory for the
    Ruri model card to work correctly, whereas granite is prefix-free
    and a wrong empty prefix on a ruri model would silently degrade
    semantic match."""
    assert _detect_prefix_family("cl-nagoya/ruri-v3-30m") == "ruri"
    # Pathological synthetic case: both substrings present.
    assert _detect_prefix_family("ruri-granite-test") == "ruri"


def test_detect_prefix_family_unknown_falls_back_to_e5() -> None:
    """An out-of-allowlist model id (the allowlist check in
    embedding_overrides drops these on read, but the embedder itself
    must still produce *some* family). The conservative choice is
    e5: a wrong empty prefix would silently degrade semantic match
    for the worst-case stray override."""
    assert _detect_prefix_family("sentence-transformers/all-MiniLM-L6-v2") == "e5"


# ---------------------------------------------------------------------------
# _MODEL_PREFIXES content
# ---------------------------------------------------------------------------


def test_granite_prefix_is_empty_pair() -> None:
    """IBM Granite Embedding R2 is documented as prefix-free
    (https://huggingface.co/blog/ibm-granite/granite-embedding-multilingual-r2).
    Either prefix being non-empty would corrupt every query."""
    assert _MODEL_PREFIXES["granite"] == ("", "")


def test_ruri_and_e5_prefixes_are_preserved() -> None:
    """Existing families must not regress when granite was added."""
    assert _MODEL_PREFIXES["ruri"] == ("検索クエリ: ", "検索文書: ")
    assert _MODEL_PREFIXES["e5"] == ("query: ", "passage: ")
