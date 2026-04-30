"""Tests for the required-keyword hard filter in app.search.

Phase 2 of the required-keyword hard filter spec
(``2026-04-30-required-semantic-hybrid-retrieval.md``) introduces:

* ``_build_required_or_clause`` — pure FTS5 query string builder for
  a single ``RequiredTerm`` (OR across all aliases, deduplicated and
  quoted).
* ``_required_keyword_filter`` — runs each required group as an FTS5
  lookup against the existing trigram tables, intersects results
  across groups (AND between groups, OR within a group's aliases),
  and returns the surviving ``set[str]`` of file_ids.
* ``search(..., required=...)`` — top-level wiring. The required
  filter is computed first; the resulting file_id set narrows the
  effective ``file_id_scope`` so every retrieval channel ranks only
  among files that pass the hard filter.

These tests stub the FTS lookups so the logic is verifiable without
a live SQLite FTS5 index. Integration with the real index is covered
by the eval harness baseline -> Phase 2 comparison.
"""

import sys
from unittest.mock import MagicMock

for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.rag.query_transform import RequiredTerm  # noqa: E402
from app.search import (  # noqa: E402
    _build_required_or_clause,
    _required_keyword_filter,
)


class TestBuildRequiredOrClause:
    """Pure builder for a single required-term FTS5 query string."""

    def test_single_alias_returns_quoted_phrase(self):
        term = RequiredTerm(canonical="芥川", script="han", aliases=("芥川",))
        assert _build_required_or_clause(term) == '"芥川"'

    def test_multiple_aliases_joined_by_or(self):
        term = RequiredTerm(
            canonical="ちいかわ",
            script="hira",
            aliases=("ちいかわ", "チイカワ"),
        )
        clause = _build_required_or_clause(term)
        # Each alias gets phrase-quoted, joined by " OR ", wrapped in parens
        # so the boolean is unambiguous when AND-combined with siblings.
        assert clause == '("ちいかわ" OR "チイカワ")'

    def test_deduplicates_aliases(self):
        term = RequiredTerm(
            canonical="Python",
            script="latin",
            aliases=("Python", "Python", "python"),
        )
        clause = _build_required_or_clause(term)
        # "Python" appears once.
        assert clause.count('"Python"') == 1
        assert '"python"' in clause

    def test_strips_double_quote_from_alias(self):
        # FTS5 phrase tokens cannot contain a literal double-quote;
        # strip it rather than failing the query.
        term = RequiredTerm(
            canonical='ab"cd', script="latin", aliases=('ab"cd',)
        )
        clause = _build_required_or_clause(term)
        assert '"' not in clause.replace('"', "", 2)  # only the wrapping quotes

    def test_skips_empty_alias(self):
        term = RequiredTerm(
            canonical="x", script="latin", aliases=("x", "", "  ")
        )
        clause = _build_required_or_clause(term)
        # Whitespace and empty aliases dropped; only "x" survives.
        assert clause == '"x"'

    def test_empty_aliases_returns_empty_string(self):
        # Defensive: never construct an FTS5 query from an empty term.
        term = RequiredTerm(canonical="", script="latin", aliases=())
        assert _build_required_or_clause(term) == ""


class TestRequiredKeywordFilter:
    """End-to-end logic of the required-keyword hard filter."""

    def _patch_fts(self, monkeypatch, lookup):
        """Replace the FTS lookup with a callable returning file_id sets.

        The callable receives the OR clause string and returns a set of
        file_ids that match in any of the filter's underlying tables
        (fts_files / fts_transcripts / fts_text_content unioned).
        """
        monkeypatch.setattr(
            "app.search._fts_lookup_required", lookup
        )

    def test_returns_none_when_no_required_terms(self, monkeypatch):
        # ``required=()`` means "no hard filter" — the function returns
        # None so the caller leaves ``file_id_scope`` unchanged.
        called = False

        def _lookup(clause):
            nonlocal called
            called = True
            return set()

        self._patch_fts(monkeypatch, _lookup)

        result = _required_keyword_filter(())

        assert result is None
        assert called is False  # no FTS lookup run

    def test_single_term_returns_matching_set(self, monkeypatch):
        term = RequiredTerm(
            canonical="東福寺", script="han", aliases=("東福寺",)
        )

        def _lookup(clause):
            assert "東福寺" in clause
            return {"file_a", "file_b"}

        self._patch_fts(monkeypatch, _lookup)

        result = _required_keyword_filter((term,))

        assert result == {"file_a", "file_b"}

    def test_multiple_terms_intersected_with_and(self, monkeypatch):
        # Two required terms: only files that match BOTH survive.
        term1 = RequiredTerm(canonical="A", script="latin", aliases=("A",))
        term2 = RequiredTerm(canonical="B", script="latin", aliases=("B",))

        responses = {
            '"A"': {"file_a", "file_b", "file_c"},
            '"B"': {"file_b", "file_c", "file_d"},
        }

        def _lookup(clause):
            return responses[clause]

        self._patch_fts(monkeypatch, _lookup)

        result = _required_keyword_filter((term1, term2))

        # Intersection: only file_b and file_c match both.
        assert result == {"file_b", "file_c"}

    def test_alias_or_within_group(self, monkeypatch):
        # Within a single required group, aliases are OR-combined: a
        # file matching ANY alias passes the group.
        term = RequiredTerm(
            canonical="ちいかわ",
            script="hira",
            aliases=("ちいかわ", "チイカワ"),
        )

        def _lookup(clause):
            assert "ちいかわ" in clause and "チイカワ" in clause
            return {"file_x"}  # transcripts spell the name in katakana

        self._patch_fts(monkeypatch, _lookup)

        result = _required_keyword_filter((term,))

        assert result == {"file_x"}

    def test_zero_match_term_short_circuits_to_empty(self, monkeypatch):
        # When any required term matches zero files, the AND-intersection
        # is empty regardless of the others; the function short-circuits
        # to ``set()`` before issuing further FTS lookups.
        term1 = RequiredTerm(canonical="A", script="latin", aliases=("A",))
        term2 = RequiredTerm(canonical="B", script="latin", aliases=("B",))
        term3 = RequiredTerm(canonical="C", script="latin", aliases=("C",))

        calls: list[str] = []
        responses = {
            '"A"': {"file_a"},
            '"B"': set(),  # zero hits — short-circuit point
            '"C"': {"file_x"},  # should not be evaluated
        }

        def _lookup(clause):
            calls.append(clause)
            return responses[clause]

        self._patch_fts(monkeypatch, _lookup)

        result = _required_keyword_filter((term1, term2, term3))

        assert result == set()
        # Only the first two clauses are issued; the third is skipped.
        assert calls == ['"A"', '"B"']

    def test_skips_term_with_empty_or_clause(self, monkeypatch):
        # A malformed RequiredTerm whose alias list normalises to empty
        # is silently dropped — the filter applies only the other terms
        # rather than failing the whole pipeline. This protects against
        # an LLM emitting a canonical with no usable alias content.
        good = RequiredTerm(canonical="A", script="latin", aliases=("A",))
        bad = RequiredTerm(canonical="", script="latin", aliases=("",))

        def _lookup(clause):
            return {"file_a"}

        self._patch_fts(monkeypatch, _lookup)

        result = _required_keyword_filter((bad, good))

        assert result == {"file_a"}
