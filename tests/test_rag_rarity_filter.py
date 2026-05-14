"""Tests for app.rag.rarity_filter (SIRA-inspired clue DF filter).

The filter is a soft post-processor for Stage 2 clue generation: it
takes an LLM-generated keyword string, looks up each token's document
frequency in the word-FTS vocab tables, and drops tokens whose DF
ratio exceeds the threshold.

These tests focus on the filter's branching — DB interactions are
stubbed so the suite stays fast and order-independent. The ``fts5vocab``
contract is upstream SQLite behaviour; we trust it and only assert
that we read the right columns from it.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

# Match the stubbing dance other rag tests use so the import graph
# doesn't drag in real torch / sqlite-vec.
for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "numpy",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.rag import rarity_filter  # noqa: E402
from app.rag.rarity_filter import (  # noqa: E402
    DEFAULT_THRESHOLD_RATIO,
    filter_clue_by_rarity,
    reset_cache,
)


# ---------------------------------------------------------------------------
# Stub DB session helpers
# ---------------------------------------------------------------------------


def _stub_db(
    monkeypatch,
    *,
    corpus_counts: dict[str, int] | None = None,
    df_lookups: dict[tuple[str, str], int] | None = None,
    raise_on_count: bool = False,
    raise_on_df: bool = False,
) -> None:
    """Install a fake ``get_search_db_read`` that answers the two query shapes.

    Args:
        corpus_counts: ``{table_name: row_count}`` for the
            ``SELECT count(*) FROM <table>`` corpus-size query.
        df_lookups: ``{(vocab_name, normalised_term): df}`` for the
            ``SELECT doc FROM <vocab> WHERE term = :t`` per-token
            lookup. Misses return 0 (matching production semantics).
        raise_on_count: When True, the corpus-size query raises so the
            fail-open path can be asserted.
        raise_on_df: When True, every DF lookup raises so the
            keep-on-error path can be asserted.
    """
    counts = corpus_counts or {}
    dfs = df_lookups or {}

    class _Row:
        def __init__(self, value):
            self._value = value

        def __getitem__(self, idx):
            if idx != 0:
                raise IndexError(idx)
            return self._value

        def __bool__(self):
            return self._value is not None

    class _Result:
        def __init__(self, value):
            self._value = value

        def fetchone(self):
            return _Row(self._value) if self._value is not None else None

    class _Session:
        def execute(self, statement, params=None):
            sql = str(statement).strip().lower()
            if sql.startswith("select count(*) from "):
                if raise_on_count:
                    raise RuntimeError("simulated count failure")
                table = sql.rsplit(" ", 1)[-1]
                return _Result(counts.get(table, 0))
            if sql.startswith("select doc from "):
                if raise_on_df:
                    raise RuntimeError("simulated df failure")
                # "select doc from <vocab> where term = :t"
                vocab = sql.split()[3]
                term = (params or {}).get("t", "")
                return _Result(dfs.get((vocab, term), 0))
            raise AssertionError(f"Unexpected SQL: {sql!r}")

    @contextmanager
    def _get_search_db_read():
        yield _Session()

    monkeypatch.setattr(
        rarity_filter, "get_search_db_read", _get_search_db_read
    )
    reset_cache()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFilterEmptyAndEdge:
    def setup_method(self):
        reset_cache()

    def test_empty_string_returns_empty(self):
        # No DB needed — short-circuits before the corpus lookup.
        assert filter_clue_by_rarity("") == ""

    def test_whitespace_only_returns_empty(self):
        assert filter_clue_by_rarity("   \t  ") == ""

    def test_corpus_size_zero_returns_unchanged(self, monkeypatch):
        # Fail-open when the corpus is uninitialised / empty.
        _stub_db(monkeypatch, corpus_counts={})
        assert filter_clue_by_rarity("foo bar baz") == "foo bar baz"

    def test_corpus_count_raises_returns_unchanged(self, monkeypatch):
        # Any DB-level failure on corpus lookup → fail-open.
        _stub_db(monkeypatch, raise_on_count=True)
        assert filter_clue_by_rarity("foo bar") == "foo bar"

    def test_threshold_one_disables_filtering(self, monkeypatch):
        # Explicit escape hatch: threshold=1.0 returns the input unchanged
        # even if every token is corpus-common.
        _stub_db(
            monkeypatch,
            corpus_counts={"fts_transcripts_word": 100},
            df_lookups={
                ("fts_transcripts_word_vocab", "foo"): 100,
                ("fts_transcripts_word_vocab", "bar"): 100,
            },
        )
        assert (
            filter_clue_by_rarity("foo bar", threshold_ratio=1.0)
            == "foo bar"
        )

    def test_negative_threshold_clamped(self, monkeypatch):
        # threshold < 0 clamps to 0 — the cap becomes 0 so every token
        # with df > 0 is dropped. Tokens with df=0 (unknown) survive.
        _stub_db(
            monkeypatch,
            corpus_counts={"fts_transcripts_word": 100},
            df_lookups={
                ("fts_transcripts_word_vocab", "known"): 5,
                # "unknown" not in lookup map → df=0 → kept
            },
        )
        result = filter_clue_by_rarity("known unknown", threshold_ratio=-1.0)
        assert result == "unknown"


class TestFilterDropsCommonTokens:
    def setup_method(self):
        reset_cache()

    def test_drops_token_above_threshold(self, monkeypatch):
        # corpus = 100, threshold = 0.5 → cap = 50. "common" appears in
        # 80 chunks → dropped; "rare" appears in 3 → kept.
        _stub_db(
            monkeypatch,
            corpus_counts={"fts_transcripts_word": 100},
            df_lookups={
                ("fts_transcripts_word_vocab", "common"): 80,
                ("fts_transcripts_word_vocab", "rare"): 3,
            },
        )
        assert filter_clue_by_rarity("common rare") == "rare"

    def test_sums_df_across_both_vocab_tables(self, monkeypatch):
        # transcripts vocab has df=30, text_content has df=40 → total
        # df=70, corpus=(50+50)=100 → ratio 0.70 > 0.50 → drop.
        _stub_db(
            monkeypatch,
            corpus_counts={
                "fts_transcripts_word": 50,
                "fts_text_content_word": 50,
            },
            df_lookups={
                ("fts_transcripts_word_vocab", "mixed"): 30,
                ("fts_text_content_word_vocab", "mixed"): 40,
            },
        )
        assert filter_clue_by_rarity("mixed") == ""

    def test_unknown_token_kept(self, monkeypatch):
        # A token absent from both vocab tables has df=0 → treated as
        # rare and preserved. This is the fail-open promise.
        _stub_db(
            monkeypatch,
            corpus_counts={"fts_transcripts_word": 100},
            df_lookups={},
        )
        assert filter_clue_by_rarity("novelterm") == "novelterm"

    def test_short_token_skips_lookup(self, monkeypatch):
        # Single-char tokens skip the DF lookup entirely and are kept.
        # They wouldn't have meaningful FTS hits anyway.
        _stub_db(
            monkeypatch,
            corpus_counts={"fts_transcripts_word": 100},
            df_lookups={
                # If a lookup were attempted, this would force a drop —
                # the test asserts no lookup happens.
                ("fts_transcripts_word_vocab", "a"): 999,
            },
        )
        assert filter_clue_by_rarity("a") == "a"

    def test_df_lookup_raises_keeps_token(self, monkeypatch):
        # Per-token DB failure → token kept (fail-open at lookup layer).
        _stub_db(
            monkeypatch,
            corpus_counts={"fts_transcripts_word": 100},
            raise_on_df=True,
        )
        assert filter_clue_by_rarity("anything else") == "anything else"

    def test_all_tokens_dropped_returns_empty(self, monkeypatch):
        # When every token exceeds the threshold the rejoined string is
        # empty — caller (clue_generator) treats this as "drop this
        # clue" via existing fallback.
        _stub_db(
            monkeypatch,
            corpus_counts={"fts_transcripts_word": 100},
            df_lookups={
                ("fts_transcripts_word_vocab", "ab"): 90,
                ("fts_transcripts_word_vocab", "cd"): 95,
            },
        )
        assert filter_clue_by_rarity("ab cd") == ""

    def test_preserves_order(self, monkeypatch):
        _stub_db(
            monkeypatch,
            corpus_counts={"fts_transcripts_word": 100},
            df_lookups={
                ("fts_transcripts_word_vocab", "common"): 80,
            },
        )
        # "common" is dropped from the middle — surviving tokens
        # preserve their original positions.
        result = filter_clue_by_rarity("first common last")
        assert result == "first last"


class TestNormalisation:
    def setup_method(self):
        reset_cache()

    def test_uppercase_token_normalised_for_lookup(self, monkeypatch):
        # The vocab stores lowercase forms (unicode61 lowercases by
        # default). Our normaliser must lowercase before lookup so the
        # cap correctly applies to mixed-case clues.
        _stub_db(
            monkeypatch,
            corpus_counts={"fts_transcripts_word": 100},
            df_lookups={
                ("fts_transcripts_word_vocab", "common"): 80,
            },
        )
        # Input is uppercase; the lookup must succeed via the lowercase
        # form. Result: dropped.
        assert filter_clue_by_rarity("COMMON") == ""

    def test_diacritic_token_normalised_for_lookup(self, monkeypatch):
        # unicode61 with ``remove_diacritics 2`` stores diacritic-stripped
        # forms. ``Café`` should normalise to ``cafe`` and hit the cap.
        _stub_db(
            monkeypatch,
            corpus_counts={"fts_transcripts_word": 100},
            df_lookups={
                ("fts_transcripts_word_vocab", "cafe"): 80,
            },
        )
        assert filter_clue_by_rarity("Café") == ""


class TestCacheReset:
    def setup_method(self):
        reset_cache()

    def test_reset_clears_corpus_cache(self, monkeypatch):
        _stub_db(
            monkeypatch,
            corpus_counts={"fts_transcripts_word": 100},
            df_lookups={("fts_transcripts_word_vocab", "common"): 80},
        )
        # First call caches corpus_size=100 and DF lookups.
        assert filter_clue_by_rarity("common") == ""

        # Swap to an empty corpus and reset; without reset_cache the
        # stale 100 would still gate the lookup, but reset clears it.
        _stub_db(monkeypatch, corpus_counts={})
        assert filter_clue_by_rarity("common") == "common"

    def test_zero_corpus_self_heals_without_explicit_reset(self, monkeypatch):
        # Regression: a 0 corpus must NOT lock the filter to no-op
        # forever. filter_clue_by_rarity itself clears the cache slot
        # when corpus is 0, so the next call after the indexer warms
        # up picks up the real count — without anyone having to call
        # reset_cache externally.
        _stub_db(monkeypatch, corpus_counts={})
        assert filter_clue_by_rarity("foo bar") == "foo bar"

        # Swap the underlying get_search_db_read WITHOUT calling
        # reset_cache (which _stub_db would normally do). This isolates
        # the self-heal behaviour: if filter_clue_by_rarity didn't
        # clear the cache on the previous 0 result, the cached 0 would
        # win and "foo" would NOT be dropped.
        class _Session:
            def execute(self, statement, params=None):
                sql = str(statement).strip().lower()
                if sql.startswith("select count(*) from "):
                    return type("R", (), {"fetchone": lambda self_: (100,)})()
                if sql.startswith("select doc from "):
                    term = (params or {}).get("t", "")
                    if term == "foo":
                        return type("R", (), {"fetchone": lambda self_: (90,)})()
                    return type("R", (), {"fetchone": lambda self_: None})()
                raise AssertionError(sql)

        @contextmanager
        def _get_search_db_read():
            yield _Session()

        monkeypatch.setattr(
            rarity_filter, "get_search_db_read", _get_search_db_read
        )

        # Self-heal: the previous call cleared _corpus_size cache, so
        # this call re-reads corpus=100 and drops "foo".
        assert filter_clue_by_rarity("foo bar") == "bar"


# Default threshold sanity — guards against an accidental constant move.
def test_default_threshold_is_half():
    assert DEFAULT_THRESHOLD_RATIO == pytest.approx(0.5)
