"""Tests for single-file TF-IDF keyword extraction (auto-tag candidates).

Uses fixture data rather than the real Janome tokenizer so tests run
without the 100MB model. The pure scoring paths (_tfidf_vector,
_top_keywords_with_scores, IDF filters) are what these tests exercise.
"""

from unittest.mock import MagicMock

import pytest

from app import tfidf


# ---------------------------------------------------------------------------
# _tfidf_vector + _top_keywords_with_scores (internal building blocks)
# ---------------------------------------------------------------------------


class TestTopKeywordsFromVector:
    """Sanity checks on the existing building blocks used by the new API."""

    def test_top_keywords_with_scores_orders_descending(self):
        vec = {"a": 0.1, "b": 0.9, "c": 0.5}
        result = tfidf._top_keywords_with_scores(vec, k=3)
        assert [r["word"] for r in result] == ["b", "c", "a"]

    def test_top_keywords_with_scores_respects_k(self):
        vec = {f"w{i}": float(i) for i in range(20)}
        result = tfidf._top_keywords_with_scores(vec, k=5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# get_tfidf_keywords_for_file: integration-level, mocks the DB + tokenizer
# ---------------------------------------------------------------------------


class TestGetTfidfKeywordsForFile:
    """Tests for the public single-file keyword extraction API."""

    def _setup(
        self,
        monkeypatch,
        *,
        file_id: str = "f1",
        filename: str = "料理動画.mp4",
        tokens: list[str] | None = None,
        idf: dict[str, float] | None = None,
        file_exists: bool = True,
    ):
        """Stub DB lookup + tokenizer so tests run deterministically."""
        tokens = tokens or []
        idf = idf if idf is not None else {}

        # Fake IndexedFile row
        file_row = MagicMock()
        file_row.file_id = file_id
        file_row.filename = filename

        # Mock DB session for the lookup in get_tfidf_keywords_for_file
        session = MagicMock()
        query = MagicMock()
        filter_a = MagicMock()
        filter_a.first.return_value = file_row if file_exists else None
        query.filter.return_value = filter_a
        session.query.return_value = query

        from contextlib import contextmanager

        @contextmanager
        def fake_db():
            yield session

        monkeypatch.setattr(tfidf, "get_search_db", fake_db)

        # Stub out the tokenization helper
        monkeypatch.setattr(
            tfidf, "_tokenize_file", lambda fid, fn, tok: tokens
        )

        # Stub Janome import inside the function
        import sys
        sys.modules["janome.tokenizer"] = MagicMock(Tokenizer=MagicMock())

        # Stub corpus IDF (new signature returns (idf, n_docs))
        n_docs = 100  # above the small-corpus short-circuit
        monkeypatch.setattr(
            tfidf, "_get_corpus_idf", lambda *a, **kw: (idf, n_docs)
        )

    def test_returns_empty_when_file_missing(self, monkeypatch):
        self._setup(monkeypatch, file_exists=False)
        result = tfidf.get_tfidf_keywords_for_file("nope")
        assert result == []

    def test_returns_empty_when_no_idf_cache(self, monkeypatch):
        self._setup(monkeypatch, tokens=["a", "b"], idf={})
        result = tfidf.get_tfidf_keywords_for_file("f1")
        assert result == []

    def test_min_doc_freq_drops_likely_whisper_noise(self, monkeypatch):
        """Words that appear in exactly 1 doc (likely Whisper noise)
        are dropped when min_doc_freq=2."""
        import math
        # Build a realistic IDF: "料理" in many docs (low IDF),
        # "フジヤシフ" in 1 doc out of 100 (high IDF).
        n_docs = 100
        idf = {
            "料理": math.log(n_docs / 30) + 1.0,   # df=30, common
            "フジヤシフ": math.log(n_docs / 1) + 1.0,  # df=1, rare
        }
        tokens = ["料理", "料理", "フジヤシフ"]
        self._setup(monkeypatch, tokens=tokens, idf=idf)
        # n_docs=100 so the heuristic enables the filter
        monkeypatch.setattr(
            tfidf, "_get_corpus_idf", lambda *a, **kw: (idf, n_docs)
        )
        result = tfidf.get_tfidf_keywords_for_file("f1", min_doc_freq=2)
        words = [r["word"] for r in result]
        assert "料理" in words
        assert "フジヤシフ" not in words

    def test_min_doc_freq_disabled_for_tiny_corpus(self, monkeypatch):
        """Don't apply the filter on corpora too small to learn from."""
        import math
        n_docs = 5  # below the size cutoff
        idf = {"フジヤシフ": math.log(n_docs / 1) + 1.0}
        tokens = ["フジヤシフ"]
        self._setup(monkeypatch, tokens=tokens, idf=idf)
        monkeypatch.setattr(
            tfidf, "_get_corpus_idf", lambda *a, **kw: (idf, n_docs)
        )
        result = tfidf.get_tfidf_keywords_for_file("f1", min_doc_freq=2)
        # With a 5-doc corpus, the filter is off — every word survives.
        assert any(r["word"] == "フジヤシフ" for r in result)

    def test_explicit_idf_max_overrides_min_doc_freq(self, monkeypatch):
        """idf_max=inf means accept everything regardless of min_doc_freq."""
        import math
        n_docs = 100
        idf = {"フジヤシフ": math.log(n_docs / 1) + 1.0}
        tokens = ["フジヤシフ"]
        self._setup(monkeypatch, tokens=tokens, idf=idf)
        monkeypatch.setattr(
            tfidf, "_get_corpus_idf", lambda *a, **kw: (idf, n_docs)
        )
        # Large idf_max passes the word through even though
        # min_doc_freq=2 would otherwise reject it.
        result = tfidf.get_tfidf_keywords_for_file(
            "f1", min_doc_freq=2, idf_max=float("inf")
        )
        assert any(r["word"] == "フジヤシフ" for r in result)

    def test_returns_empty_when_no_tokens(self, monkeypatch):
        self._setup(monkeypatch, tokens=[], idf={"x": 1.0})
        result = tfidf.get_tfidf_keywords_for_file("f1")
        assert result == []

    def test_basic_top_keywords(self, monkeypatch):
        tokens = ["料理", "料理", "料理", "パスタ", "パスタ", "トマト"]
        # Equal IDF across words → ranking follows TF
        idf = {"料理": 1.0, "パスタ": 1.0, "トマト": 1.0}
        self._setup(monkeypatch, tokens=tokens, idf=idf)
        result = tfidf.get_tfidf_keywords_for_file("f1", k=3)
        assert [r["word"] for r in result] == ["料理", "パスタ", "トマト"]

    def test_min_word_length_drops_short_tokens(self, monkeypatch):
        tokens = ["a", "bb", "ccc"]
        idf = {"a": 1.0, "bb": 1.0, "ccc": 1.0}
        self._setup(monkeypatch, tokens=tokens, idf=idf)
        result = tfidf.get_tfidf_keywords_for_file("f1", min_word_length=2)
        words = [r["word"] for r in result]
        assert "a" not in words
        assert set(words) == {"bb", "ccc"}

    def test_idf_max_filter_drops_rare_likely_noise(self, monkeypatch):
        # "フジヤシフ" shows up once in corpus (likely Whisper noise, high IDF)
        # "料理" appears in many files (low IDF)
        tokens = ["料理", "料理", "フジヤシフ"]
        idf = {"料理": 1.5, "フジヤシフ": 7.0}
        self._setup(monkeypatch, tokens=tokens, idf=idf)
        result = tfidf.get_tfidf_keywords_for_file("f1", idf_max=5.0)
        words = [r["word"] for r in result]
        assert "料理" in words
        assert "フジヤシフ" not in words

    def test_idf_min_filter_drops_overly_common(self, monkeypatch):
        # "こと" would slip past stopwords in theory; idf_min=1.5 drops it.
        tokens = ["料理", "料理", "こと", "こと", "こと"]
        idf = {"料理": 2.0, "こと": 1.0}
        self._setup(monkeypatch, tokens=tokens, idf=idf)
        result = tfidf.get_tfidf_keywords_for_file("f1", idf_min=1.5)
        words = [r["word"] for r in result]
        assert "料理" in words
        assert "こと" not in words

    def test_k_limits_output(self, monkeypatch):
        tokens = [f"w{i}" for i in range(20) for _ in range(i + 1)]
        idf = {f"w{i}": 1.0 for i in range(20)}
        self._setup(monkeypatch, tokens=tokens, idf=idf)
        result = tfidf.get_tfidf_keywords_for_file("f1", k=5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# _get_corpus_idf caching behavior
# ---------------------------------------------------------------------------


class TestCorpusIdfCache:
    """The corpus IDF must be cached so auto-tagging doesn't rebuild
    it for every file processed."""

    def test_reuses_cached_idf_within_ttl(self, monkeypatch):
        tfidf.reset_corpus_idf_cache()
        build_calls = []

        def fake_build():
            build_calls.append(1)
            return {"料理": 1.5}, 42

        monkeypatch.setattr(tfidf, "_build_corpus_idf", fake_build)

        tfidf._get_corpus_idf()
        tfidf._get_corpus_idf()
        tfidf._get_corpus_idf()
        assert len(build_calls) == 1
        tfidf.reset_corpus_idf_cache()

    def test_force_reload_rebuilds(self, monkeypatch):
        tfidf.reset_corpus_idf_cache()
        build_calls = []

        def fake_build():
            build_calls.append(1)
            return {"料理": 1.5}, 42

        monkeypatch.setattr(tfidf, "_build_corpus_idf", fake_build)

        tfidf._get_corpus_idf()
        tfidf._get_corpus_idf(force_reload=True)
        assert len(build_calls) == 2
        tfidf.reset_corpus_idf_cache()
