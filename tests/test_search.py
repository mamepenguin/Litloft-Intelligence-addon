"""Tests for search scoring and utility functions in app.search.

Covers pure functions that require no database or ML model access:
- L2-to-cosine conversion
- Kana conversion utilities
- FTS5 query building
- Segment grouping
- Result merging
- Embedding type selection
- File ranking helpers
"""

import math
import sys
from unittest.mock import MagicMock

import pytest

# Stub out heavy ML/image dependencies that app.search transitively imports
# so tests can run without installing PIL, open_clip, torch, etc.
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

from app.search import (
    SEGMENT_GROUP_WINDOW,
    MatchInfo,
    SegmentGroup,
    _build_fts_query,
    _combine_scores_cosine,
    _combine_scores_rrf,
    _file_ranking_from_keywords,
    _file_ranking_from_vector,
    _FileScore,
    _group_matches_into_segments,
    _KeywordMatch,
    _l2_to_cosine_similarity,
    _merge_similar_results,
    _select_embedding_types,
    _TextContentKeywordMatch,
    _to_hiragana,
    _to_katakana,
    _TranscriptKeywordMatch,
    _VectorMatch,
)


# ---------------------------------------------------------------------------
# 1. _l2_to_cosine_similarity
# ---------------------------------------------------------------------------


class TestL2ToCosineSimilarity:
    """Tests for L2 distance to cosine similarity conversion."""

    def test_identical_vectors(self) -> None:
        """Distance 0 means identical vectors -> similarity 1.0."""
        assert _l2_to_cosine_similarity(0.0) == 1.0

    def test_orthogonal_vectors(self) -> None:
        """Distance sqrt(2) means orthogonal vectors -> similarity 0.0."""
        assert _l2_to_cosine_similarity(math.sqrt(2)) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """Distance 2.0 means opposite vectors -> similarity -1.0."""
        assert _l2_to_cosine_similarity(2.0) == pytest.approx(-1.0)

    def test_distance_one(self) -> None:
        """Distance 1.0 -> similarity 0.5."""
        assert _l2_to_cosine_similarity(1.0) == pytest.approx(0.5)

    def test_monotonically_decreasing(self) -> None:
        """Larger distances must produce lower similarities."""
        distances = [0.0, 0.3, 0.7, 1.0, math.sqrt(2), 1.8, 2.0]
        similarities = [_l2_to_cosine_similarity(d) for d in distances]

        for i in range(len(similarities) - 1):
            assert similarities[i] > similarities[i + 1], (
                f"Not monotonically decreasing at index {i}: "
                f"sim({distances[i]})={similarities[i]} vs "
                f"sim({distances[i + 1]})={similarities[i + 1]}"
            )

    def test_small_distance(self) -> None:
        """Very small distance should be close to 1.0."""
        result = _l2_to_cosine_similarity(0.01)
        assert result == pytest.approx(1.0 - 0.01 * 0.01 / 2.0)


# ---------------------------------------------------------------------------
# 2. _to_katakana / _to_hiragana
# ---------------------------------------------------------------------------


class TestKanaConversion:
    """Tests for hiragana <-> katakana conversion."""

    def test_hiragana_to_katakana(self) -> None:
        assert _to_katakana("みかん") == "ミカン"

    def test_katakana_to_hiragana(self) -> None:
        assert _to_hiragana("ミカン") == "みかん"

    def test_katakana_preserves_non_kana(self) -> None:
        """Non-hiragana characters pass through unchanged."""
        assert _to_katakana("Hello みかん World") == "Hello ミカン World"

    def test_hiragana_preserves_non_kana(self) -> None:
        """Non-katakana characters pass through unchanged."""
        assert _to_hiragana("Hello ミカン World") == "Hello みかん World"

    def test_roundtrip_via_katakana(self) -> None:
        """to_katakana(to_hiragana(x)) == to_katakana(x) for any string."""
        texts = ["ミカン", "みかん", "テスト", "てすと", "Hello", "漢字"]
        for text in texts:
            assert _to_katakana(_to_hiragana(text)) == _to_katakana(text), (
                f"Round-trip failed for {text!r}"
            )

    def test_roundtrip_via_hiragana(self) -> None:
        """to_hiragana(to_katakana(x)) == to_hiragana(x)."""
        texts = ["ミカン", "みかん", "テスト", "てすと"]
        for text in texts:
            assert _to_hiragana(_to_katakana(text)) == _to_hiragana(text)

    def test_empty_string(self) -> None:
        assert _to_katakana("") == ""
        assert _to_hiragana("") == ""

    def test_ascii_only(self) -> None:
        """Pure ASCII is returned unchanged by both functions."""
        assert _to_katakana("hello123") == "hello123"
        assert _to_hiragana("hello123") == "hello123"

    def test_mixed_kana(self) -> None:
        """String with both hiragana and katakana."""
        assert _to_katakana("あいウエ") == "アイウエ"
        assert _to_hiragana("あいウエ") == "あいうえ"


# ---------------------------------------------------------------------------
# 3. _build_fts_query
# ---------------------------------------------------------------------------


class TestBuildFtsQuery:
    """Tests for FTS5 trigram query builder."""

    def test_empty_string(self) -> None:
        assert _build_fts_query("") == ""

    def test_only_whitespace(self) -> None:
        assert _build_fts_query("   ") == ""

    def test_single_ascii_term(self) -> None:
        """ASCII term has one variant -> simple quoted string."""
        result = _build_fts_query("hello")
        assert result == '"hello"'

    def test_hiragana_term(self) -> None:
        """Hiragana term produces OR with katakana variant."""
        result = _build_fts_query("みかん")
        assert result == '("みかん" OR "ミカン")'

    def test_katakana_term(self) -> None:
        """Katakana term produces OR with hiragana variant."""
        result = _build_fts_query("ミカン")
        assert result == '("ミカン" OR "みかん")'

    def test_multiple_terms(self) -> None:
        """Multiple terms joined with AND."""
        result = _build_fts_query("hello みかん")
        assert result == '"hello" AND ("みかん" OR "ミカン")'

    def test_double_quotes_stripped(self) -> None:
        """Double quote characters are removed from terms."""
        result = _build_fts_query('"test"')
        assert result == '"test"'

    def test_term_that_is_only_double_quotes(self) -> None:
        """A term that becomes empty after quote removal is skipped."""
        result = _build_fts_query('""')
        assert result == ""

    def test_mixed_terms_with_quotes(self) -> None:
        """Valid terms survive even if some terms are stripped."""
        result = _build_fts_query('"" hello')
        assert result == '"hello"'

    def test_multiple_ascii_terms(self) -> None:
        result = _build_fts_query("foo bar")
        assert result == '"foo" AND "bar"'

    def test_kanji_term(self) -> None:
        """Kanji-only term has one variant (not affected by kana conversion)."""
        result = _build_fts_query("漢字")
        assert result == '"漢字"'

    def test_mixed_kanji_hiragana(self) -> None:
        """Term with both kanji and hiragana produces variants."""
        result = _build_fts_query("みかん先輩")
        # Original has hiragana, so katakana variant differs
        assert "みかん先輩" in result
        assert "ミカン先輩" in result
        assert " OR " in result


# ---------------------------------------------------------------------------
# 4. _group_matches_into_segments
# ---------------------------------------------------------------------------


class TestGroupMatchesIntoSegments:
    """Tests for time-based segment grouping."""

    def test_empty_list(self) -> None:
        assert _group_matches_into_segments([]) == []

    def test_only_general_matches(self) -> None:
        """Matches without timestamps form a single general segment."""
        matches = [
            MatchInfo(match_type="keyword", text="foo", score=0.9),
            MatchInfo(match_type="keyword", text="bar", score=0.8),
        ]
        segments = _group_matches_into_segments(matches)

        assert len(segments) == 1
        assert segments[0].time_range is None
        assert len(segments[0].matches) == 2

    def test_single_timed_match(self) -> None:
        """One timed match creates one timed segment."""
        matches = [
            MatchInfo(
                match_type="transcript",
                text="hello",
                score=0.9,
                timestamp_start=10.0,
                timestamp_end=15.0,
            ),
        ]
        segments = _group_matches_into_segments(matches)

        assert len(segments) == 1
        assert segments[0].time_range == (10.0, 15.0)
        assert len(segments[0].matches) == 1

    def test_timed_matches_within_window_merged(self) -> None:
        """Two timed matches within SEGMENT_GROUP_WINDOW are merged."""
        matches = [
            MatchInfo(
                match_type="transcript",
                text="first",
                score=0.9,
                timestamp_start=10.0,
                timestamp_end=15.0,
            ),
            MatchInfo(
                match_type="transcript",
                text="second",
                score=0.8,
                timestamp_start=20.0,
                timestamp_end=25.0,
            ),
        ]
        segments = _group_matches_into_segments(matches)

        # Both should be merged: 20.0 <= 15.0 + 30 = 45.0
        timed_segments = [s for s in segments if s.time_range is not None]
        assert len(timed_segments) == 1
        assert len(timed_segments[0].matches) == 2

    def test_timed_matches_far_apart_separate(self) -> None:
        """Two timed matches more than 60s apart create separate segments."""
        matches = [
            MatchInfo(
                match_type="transcript",
                text="early",
                score=0.9,
                timestamp_start=10.0,
                timestamp_end=15.0,
            ),
            MatchInfo(
                match_type="transcript",
                text="late",
                score=0.8,
                timestamp_start=120.0,
                timestamp_end=125.0,
            ),
        ]
        segments = _group_matches_into_segments(matches)

        timed_segments = [s for s in segments if s.time_range is not None]
        assert len(timed_segments) == 2
        assert timed_segments[0].time_range[0] == 10.0
        assert timed_segments[1].time_range[0] == 120.0

    def test_mix_of_general_and_timed(self) -> None:
        """General and timed matches produce separate segment types."""
        matches = [
            MatchInfo(match_type="keyword", text="general", score=0.7),
            MatchInfo(
                match_type="transcript",
                text="timed",
                score=0.9,
                timestamp_start=50.0,
                timestamp_end=55.0,
            ),
        ]
        segments = _group_matches_into_segments(matches)

        general = [s for s in segments if s.time_range is None]
        timed = [s for s in segments if s.time_range is not None]
        assert len(general) == 1
        assert len(timed) == 1

    def test_general_segment_comes_first(self) -> None:
        """General segment appears before timed segments."""
        matches = [
            MatchInfo(
                match_type="transcript",
                text="timed",
                score=0.9,
                timestamp_start=50.0,
                timestamp_end=55.0,
            ),
            MatchInfo(match_type="keyword", text="general", score=0.7),
        ]
        segments = _group_matches_into_segments(matches)

        assert segments[0].time_range is None
        assert segments[1].time_range is not None

    def test_three_timed_groups(self) -> None:
        """Three time clusters far apart create three segments."""
        matches = [
            MatchInfo(
                match_type="transcript", text="a", score=0.9,
                timestamp_start=0.0, timestamp_end=5.0,
            ),
            MatchInfo(
                match_type="transcript", text="b", score=0.8,
                timestamp_start=100.0, timestamp_end=105.0,
            ),
            MatchInfo(
                match_type="transcript", text="c", score=0.7,
                timestamp_start=200.0, timestamp_end=205.0,
            ),
        ]
        segments = _group_matches_into_segments(matches)

        timed = [s for s in segments if s.time_range is not None]
        assert len(timed) == 3

    def test_unsorted_timed_matches_sorted_by_start(self) -> None:
        """Timed matches are sorted by timestamp_start before grouping."""
        matches = [
            MatchInfo(
                match_type="transcript", text="later", score=0.8,
                timestamp_start=100.0, timestamp_end=105.0,
            ),
            MatchInfo(
                match_type="transcript", text="earlier", score=0.9,
                timestamp_start=10.0, timestamp_end=15.0,
            ),
        ]
        segments = _group_matches_into_segments(matches)

        timed = [s for s in segments if s.time_range is not None]
        assert len(timed) == 2
        assert timed[0].time_range[0] == 10.0
        assert timed[1].time_range[0] == 100.0

    def test_segment_group_is_frozen(self) -> None:
        """SegmentGroup should be immutable."""
        sg = SegmentGroup(time_range=None, matches=())
        with pytest.raises(AttributeError):
            sg.time_range = (0.0, 1.0)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 5. _merge_similar_results
# ---------------------------------------------------------------------------


class TestMergeSimilarResults:
    """Tests for primary/secondary result merging."""

    @staticmethod
    def _make_result(file_id: str, score: float, drive: str = "d1") -> dict:
        return {
            "file_id": file_id,
            "score": score,
            "drive": drive,
            "filename": f"{file_id}.mp4",
            "file_type": "video",
            "mime_type": "video/mp4",
        }

    def test_both_lists_have_same_file(self) -> None:
        """File in both lists gets combined score (0.5 * norm_p + 0.5 * norm_s)."""
        primary = [self._make_result("a", 0.9)]
        secondary = [self._make_result("a", 0.8)]

        merged = _merge_similar_results(primary, secondary, 10)

        assert len(merged) == 1
        assert merged[0]["file_id"] == "a"
        # Single item in each list -> norm is 1.0 each
        assert merged[0]["score"] == pytest.approx(0.5 * 1.0 + 0.5 * 1.0)

    def test_file_only_in_primary(self) -> None:
        """File only in primary gets penalized (norm * 0.7)."""
        primary = [self._make_result("a", 0.9)]
        secondary = []

        merged = _merge_similar_results(primary, secondary, 10)

        assert len(merged) == 1
        # Single item -> norm = 1.0, multiplied by 0.7
        assert merged[0]["score"] == pytest.approx(1.0 * 0.7)

    def test_file_only_in_secondary(self) -> None:
        """File only in secondary gets penalized (norm * 0.7)."""
        primary = []
        secondary = [self._make_result("a", 0.8)]

        merged = _merge_similar_results(primary, secondary, 10)

        assert len(merged) == 1
        assert merged[0]["score"] == pytest.approx(1.0 * 0.7)

    def test_sorted_by_score_descending(self) -> None:
        """Results are sorted by combined score, highest first."""
        primary = [
            self._make_result("a", 0.9),
            self._make_result("b", 0.5),
        ]
        secondary = [
            self._make_result("b", 1.0),
            self._make_result("a", 0.1),
        ]
        merged = _merge_similar_results(primary, secondary, 10)

        scores = [r["score"] for r in merged]
        assert scores == sorted(scores, reverse=True)

    def test_respects_limit(self) -> None:
        """Output is truncated to the requested limit."""
        primary = [self._make_result(f"f{i}", 1.0 - i * 0.1) for i in range(5)]
        secondary = []

        merged = _merge_similar_results(primary, secondary, 3)
        assert len(merged) == 3

    def test_empty_both(self) -> None:
        """Both empty -> empty result."""
        assert _merge_similar_results([], [], 10) == []

    def test_normalization_with_score_range(self) -> None:
        """When primary has a range, scores are normalized to [0, 1]."""
        primary = [
            self._make_result("a", 1.0),  # norm -> 1.0
            self._make_result("b", 0.5),  # norm -> 0.0
        ]
        secondary = []

        merged = _merge_similar_results(primary, secondary, 10)

        # Sort merged by file_id for deterministic assertions
        by_id = {r["file_id"]: r["score"] for r in merged}
        # a: norm=1.0, single_signal -> 1.0 * 0.7 = 0.7
        assert by_id["a"] == pytest.approx(0.7)
        # b: norm=0.0, single_signal -> 0.0 * 0.7 = 0.0
        assert by_id["b"] == pytest.approx(0.0)

    def test_single_item_normalization(self) -> None:
        """Single item in a list has zero range -> norm defaults to 1.0."""
        primary = [self._make_result("a", 0.42)]
        merged = _merge_similar_results(primary, [], 10)

        # Single item -> range=0 -> norm=1.0 -> 1.0 * 0.7 = 0.7
        assert merged[0]["score"] == pytest.approx(0.7)

    def test_metadata_preserved(self) -> None:
        """Merged result preserves file metadata from source dict."""
        primary = [self._make_result("a", 0.9)]
        merged = _merge_similar_results(primary, [], 10)

        assert merged[0]["file_id"] == "a"
        assert merged[0]["drive"] == "d1"
        assert merged[0]["filename"] == "a.mp4"
        assert merged[0]["file_type"] == "video"


# ---------------------------------------------------------------------------
# 6. _select_embedding_types
# ---------------------------------------------------------------------------


class TestSelectEmbeddingTypes:
    """Tests for file-type to embedding-type mapping."""

    def test_image(self) -> None:
        assert _select_embedding_types("image") == ("clip", None)

    def test_video(self) -> None:
        assert _select_embedding_types("video") == ("clip", "tfidf")

    def test_audio(self) -> None:
        assert _select_embedding_types("audio") == ("whisper", None)

    def test_document(self) -> None:
        assert _select_embedding_types("document") == ("text_content", None)

    def test_unknown_type(self) -> None:
        assert _select_embedding_types("spreadsheet") == ("metadata", None)

    def test_empty_string(self) -> None:
        assert _select_embedding_types("") == ("metadata", None)


# ---------------------------------------------------------------------------
# 7. _file_ranking_from_vector / _file_ranking_from_keywords
# ---------------------------------------------------------------------------


class TestFileRankingFromVector:
    """Tests for vector-match file-level ranking."""

    @staticmethod
    def _make_vmatch(
        file_id: str, score: float, embedding_id: str = "e1",
    ) -> _VectorMatch:
        return _VectorMatch(
            embedding_id=embedding_id,
            file_id=file_id,
            score=score,
            embedding_type="metadata",
            content_preview="preview",
            timestamp_start=None,
            timestamp_end=None,
        )

    def test_single_file(self) -> None:
        matches = [self._make_vmatch("f1", 0.9)]
        ranking, best = _file_ranking_from_vector(matches)

        assert ranking == {"f1": 0}
        assert best == {"f1": 0.9}

    def test_multiple_files_ranked_by_score(self) -> None:
        matches = [
            self._make_vmatch("f1", 0.7, "e1"),
            self._make_vmatch("f2", 0.9, "e2"),
            self._make_vmatch("f3", 0.5, "e3"),
        ]
        ranking, best = _file_ranking_from_vector(matches)

        assert ranking["f2"] == 0  # highest score
        assert ranking["f1"] == 1
        assert ranking["f3"] == 2

    def test_same_file_multiple_matches_uses_best_score(self) -> None:
        matches = [
            self._make_vmatch("f1", 0.5, "e1"),
            self._make_vmatch("f1", 0.9, "e2"),
            self._make_vmatch("f1", 0.3, "e3"),
        ]
        ranking, best = _file_ranking_from_vector(matches)

        assert ranking == {"f1": 0}
        assert best["f1"] == pytest.approx(0.9)

    def test_empty_matches(self) -> None:
        ranking, best = _file_ranking_from_vector([])
        assert ranking == {}
        assert best == {}

    def test_deduplication_across_files(self) -> None:
        """Multiple matches per file: only best score determines rank."""
        matches = [
            self._make_vmatch("f1", 0.3, "e1"),
            self._make_vmatch("f1", 0.8, "e2"),
            self._make_vmatch("f2", 0.7, "e3"),
        ]
        ranking, best = _file_ranking_from_vector(matches)

        # f1 best=0.8, f2 best=0.7 -> f1 rank 0, f2 rank 1
        assert ranking["f1"] == 0
        assert ranking["f2"] == 1


class TestFileRankingFromKeywords:
    """Tests for keyword-match file-level ranking."""

    @staticmethod
    def _make_kmatch(file_id: str, score: float) -> _KeywordMatch:
        return _KeywordMatch(file_id=file_id, score=score, matched_field="fts")

    def test_single_file(self) -> None:
        matches = [self._make_kmatch("f1", 0.9)]
        ranking = _file_ranking_from_keywords(matches)
        assert ranking == {"f1": 0}

    def test_multiple_files_ranked(self) -> None:
        matches = [
            self._make_kmatch("f1", 0.5),
            self._make_kmatch("f2", 0.9),
            self._make_kmatch("f3", 0.7),
        ]
        ranking = _file_ranking_from_keywords(matches)

        assert ranking["f2"] == 0
        assert ranking["f3"] == 1
        assert ranking["f1"] == 2

    def test_same_file_deduplicated(self) -> None:
        """Multiple keyword matches for same file use best score."""
        matches = [
            self._make_kmatch("f1", 0.3),
            self._make_kmatch("f1", 0.8),
        ]
        ranking = _file_ranking_from_keywords(matches)
        assert ranking == {"f1": 0}

    def test_empty_matches(self) -> None:
        ranking = _file_ranking_from_keywords([])
        assert ranking == {}


# ---------------------------------------------------------------------------
# 8. _combine_scores_cosine
# ---------------------------------------------------------------------------


class TestCombineScoresCosine:
    """Tests for weighted cosine similarity score combination."""

    @staticmethod
    def _make_vmatch(
        file_id: str,
        score: float,
        embedding_type: str = "metadata",
    ) -> _VectorMatch:
        return _VectorMatch(
            embedding_id=f"emb_{file_id}",
            file_id=file_id,
            score=score,
            embedding_type=embedding_type,
            content_preview="preview",
            timestamp_start=None,
            timestamp_end=None,
        )

    @staticmethod
    def _make_kmatch(file_id: str, score: float) -> _KeywordMatch:
        return _KeywordMatch(file_id=file_id, score=score, matched_field="fts")

    @staticmethod
    def _make_tkmatch(
        file_id: str, score: float,
    ) -> _TranscriptKeywordMatch:
        return _TranscriptKeywordMatch(
            file_id=file_id,
            score=score,
            text="transcript text",
            timestamp_start=10.0,
            timestamp_end=20.0,
        )

    @staticmethod
    def _make_tcmatch(
        file_id: str, score: float,
    ) -> _TextContentKeywordMatch:
        return _TextContentKeywordMatch(
            file_id=file_id,
            score=score,
            text="document text",
            page=1,
        )

    @staticmethod
    def _patch_search_settings(
        monkeypatch: pytest.MonkeyPatch, **search_kwargs: object,
    ) -> None:
        """Patch settings in app.search module namespace."""
        import app.search as search_mod
        from app import config
        monkeypatch.setattr(search_mod, "settings", config.Settings(
            intelligence_data_dir=config.settings.intelligence_data_dir,
            homevault_db_path=config.settings.homevault_db_path,
            model_cache_dir=config.settings.model_cache_dir,
            search_db_path=config.settings.search_db_path,
            search=config.SearchConfig(**search_kwargs),
        ))

    def test_text_vector_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """File matched only by text vector gets alpha-weighted score."""
        self._patch_search_settings(monkeypatch, alpha=0.7, type_weight_metadata=1.3)
        result = _combine_scores_cosine(
            text_matches=[self._make_vmatch("f1", 0.9, "metadata")],
            clip_matches=[],
            keyword_matches=[],
            transcript_keyword_matches=[],
        )
        assert "f1" in result
        expected = 0.7 * (0.9 * 1.3)
        assert result["f1"].combined_score == pytest.approx(expected)

    def test_keyword_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """File matched only by keyword gets (1-alpha)-weighted score."""
        self._patch_search_settings(monkeypatch, alpha=0.7)
        result = _combine_scores_cosine(
            text_matches=[],
            clip_matches=[],
            keyword_matches=[self._make_kmatch("f1", 0.8)],
            transcript_keyword_matches=[],
        )
        assert "f1" in result
        expected = 0.3 * 0.8
        assert result["f1"].combined_score == pytest.approx(expected)

    def test_both_vector_and_keyword(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """File matched by both gets combined alpha-weighted score."""
        self._patch_search_settings(monkeypatch, alpha=0.7, type_weight_metadata=1.0)
        result = _combine_scores_cosine(
            text_matches=[self._make_vmatch("f1", 0.9, "metadata")],
            clip_matches=[],
            keyword_matches=[self._make_kmatch("f1", 0.8)],
            transcript_keyword_matches=[],
        )
        expected = 0.7 * 0.9 + 0.3 * 0.8
        assert result["f1"].combined_score == pytest.approx(expected)

    def test_clip_weighted_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLIP matches use type_weight_clip for weighting."""
        self._patch_search_settings(monkeypatch, alpha=0.7, type_weight_clip=0.5)
        result = _combine_scores_cosine(
            text_matches=[],
            clip_matches=[self._make_vmatch("f1", 0.8, "clip")],
            keyword_matches=[],
            transcript_keyword_matches=[],
        )
        expected = 0.7 * (0.8 * 0.5)
        assert result["f1"].combined_score == pytest.approx(expected)

    def test_best_keyword_across_sources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keyword score is max across keyword, transcript_keyword, text_content_keyword."""
        self._patch_search_settings(monkeypatch, alpha=0.5)
        result = _combine_scores_cosine(
            text_matches=[],
            clip_matches=[],
            keyword_matches=[self._make_kmatch("f1", 0.3)],
            transcript_keyword_matches=[self._make_tkmatch("f1", 0.9)],
            text_content_keyword_matches=[self._make_tcmatch("f1", 0.5)],
        )
        expected = 0.5 * 0.0 + 0.5 * 0.9
        assert result["f1"].combined_score == pytest.approx(expected)

    def test_multiple_files(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple files each get their own combined score."""
        self._patch_search_settings(monkeypatch, alpha=0.7, type_weight_metadata=1.0)
        result = _combine_scores_cosine(
            text_matches=[
                self._make_vmatch("f1", 0.9, "metadata"),
                self._make_vmatch("f2", 0.5, "metadata"),
            ],
            clip_matches=[],
            keyword_matches=[self._make_kmatch("f2", 0.8)],
            transcript_keyword_matches=[],
        )
        assert len(result) == 2
        assert result["f1"].combined_score == pytest.approx(0.7 * 0.9)
        assert result["f2"].combined_score == pytest.approx(0.7 * 0.5 + 0.3 * 0.8)

    def test_match_types_tracked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Match types are correctly tracked per file."""
        self._patch_search_settings(monkeypatch)
        result = _combine_scores_cosine(
            text_matches=[self._make_vmatch("f1", 0.9, "metadata")],
            clip_matches=[self._make_vmatch("f1", 0.7, "clip")],
            keyword_matches=[self._make_kmatch("f1", 0.5)],
            transcript_keyword_matches=[],
        )
        assert "metadata" in result["f1"].match_types
        assert "clip" in result["f1"].match_types
        assert "keyword" in result["f1"].match_types

    def test_empty_all_sources(self) -> None:
        """No matches returns empty dict."""
        result = _combine_scores_cosine(
            text_matches=[],
            clip_matches=[],
            keyword_matches=[],
            transcript_keyword_matches=[],
        )
        assert result == {}


# ---------------------------------------------------------------------------
# 9. _combine_scores_rrf
# ---------------------------------------------------------------------------


class TestCombineScoresRrf:
    """Tests for Reciprocal Rank Fusion score combination."""

    @staticmethod
    def _make_vmatch(
        file_id: str, score: float, embedding_type: str = "metadata",
    ) -> _VectorMatch:
        return _VectorMatch(
            embedding_id=f"emb_{file_id}",
            file_id=file_id,
            score=score,
            embedding_type=embedding_type,
            content_preview="preview",
            timestamp_start=None,
            timestamp_end=None,
        )

    @staticmethod
    def _make_kmatch(file_id: str, score: float) -> _KeywordMatch:
        return _KeywordMatch(file_id=file_id, score=score, matched_field="fts")

    @staticmethod
    def _patch_search_settings(
        monkeypatch: pytest.MonkeyPatch, **search_kwargs: object,
    ) -> None:
        import app.search as search_mod
        from app import config
        monkeypatch.setattr(search_mod, "settings", config.Settings(
            intelligence_data_dir=config.settings.intelligence_data_dir,
            homevault_db_path=config.settings.homevault_db_path,
            model_cache_dir=config.settings.model_cache_dir,
            search_db_path=config.settings.search_db_path,
            search=config.SearchConfig(**search_kwargs),
        ))

    def test_single_system_single_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One file from one system: score = 1/(k + 0 + 1)."""
        self._patch_search_settings(monkeypatch, rrf_weight_clip=0.5)
        result = _combine_scores_rrf(
            text_matches=[self._make_vmatch("f1", 0.9)],
            clip_matches=[],
            keyword_matches=[],
            transcript_keyword_matches=[],
            k=60,
        )
        assert result["f1"].combined_score == pytest.approx(1.0 / 61)

    def test_multi_system_boost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """File appearing in multiple systems gets boosted RRF score."""
        self._patch_search_settings(monkeypatch, rrf_weight_clip=0.5)
        result = _combine_scores_rrf(
            text_matches=[self._make_vmatch("f1", 0.9)],
            clip_matches=[],
            keyword_matches=[self._make_kmatch("f1", 0.8)],
            transcript_keyword_matches=[],
            k=60,
        )
        expected = 1.0 / 61 + 1.0 / 61
        assert result["f1"].combined_score == pytest.approx(expected)

    def test_clip_weight_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLIP system uses reduced weight from config."""
        self._patch_search_settings(monkeypatch, rrf_weight_clip=0.3)
        result = _combine_scores_rrf(
            text_matches=[],
            clip_matches=[self._make_vmatch("f1", 0.9, "clip")],
            keyword_matches=[],
            transcript_keyword_matches=[],
            k=60,
        )
        expected = 0.3 / 61
        assert result["f1"].combined_score == pytest.approx(expected)

    def test_empty_all(self) -> None:
        result = _combine_scores_rrf(
            text_matches=[],
            clip_matches=[],
            keyword_matches=[],
            transcript_keyword_matches=[],
            k=60,
        )
        assert result == {}


# ---------------------------------------------------------------------------
# Recall mode: _combine_scores_rrf with _RecallParams overrides
# ---------------------------------------------------------------------------


class TestRecallModeRRF:
    """Recall mode rebalances the RRF channel weights for RAG use.

    Precision mode (existing callers): CLIP uses the configured
    ``rrf_weight_clip`` (default 0.5), everything else 1.0.
    Recall mode: transcript/text_content upweighted to 1.5, CLIP
    pushed down to 0.2. BLIP-disabled environments drop CLIP to 0.
    """

    def _transcript_match(self, file_id: str, score: float) -> _TranscriptKeywordMatch:
        return _TranscriptKeywordMatch(
            file_id=file_id,
            score=score,
            text="segment text",
            timestamp_start=0.0,
            timestamp_end=10.0,
        )

    def _clip_match(self, file_id: str, score: float) -> _VectorMatch:
        return _VectorMatch(
            embedding_id=f"{file_id}-emb",
            file_id=file_id,
            score=score,
            embedding_type="clip",
            content_preview="image clip",
            timestamp_start=None,
            timestamp_end=None,
        )

    def test_recall_mode_uses_override_weights(self):
        """A file hit only by transcript_keyword should outrank a CLIP-only
        file of the same rank in recall mode — the opposite of precision."""
        from app.search import _RECALL_PARAMS

        transcript_only = [self._transcript_match("f_text", 0.9)]
        clip_only = [self._clip_match("f_clip", 0.9)]

        recall_scores = _combine_scores_rrf(
            text_matches=[],
            clip_matches=clip_only,
            keyword_matches=[],
            transcript_keyword_matches=transcript_only,
            k=60,
            recall_params=_RECALL_PARAMS,
            include_clip=True,
        )

        # Both files appear, but transcript-only file has the higher
        # combined score because of the 1.5 weight vs. CLIP's 0.2.
        assert recall_scores["f_text"].combined_score > recall_scores[
            "f_clip"
        ].combined_score

    def test_precision_mode_keeps_clip_weight(self):
        """Without recall_params, the legacy precision-mode weights apply."""
        transcript_only = [self._transcript_match("f_text", 0.9)]
        clip_only = [self._clip_match("f_clip", 0.9)]

        precision_scores = _combine_scores_rrf(
            text_matches=[],
            clip_matches=clip_only,
            keyword_matches=[],
            transcript_keyword_matches=transcript_only,
            k=60,
        )

        # Under precision mode the two channels are closer together
        # (clip is 0.5, transcript is 1.0). They won't be equal because
        # of the different weights but transcript still wins — what
        # matters is that the ratio is narrower than in recall mode.
        text_score = precision_scores["f_text"].combined_score
        clip_score = precision_scores["f_clip"].combined_score
        assert text_score > clip_score
        assert text_score / clip_score < 3.0  # much narrower than recall's 1.5/0.2 = 7.5x

    def test_include_clip_false_zeroes_clip_channel(self):
        """include_clip=False forces the CLIP channel weight to 0.

        This is how the runtime handles BLIP-disabled environments:
        without BLIP captions the LLM cannot read an image match, so
        the candidate slot is wasted.
        """
        from app.search import _RECALL_PARAMS

        clip_only = [self._clip_match("f_clip", 0.9)]

        scores = _combine_scores_rrf(
            text_matches=[],
            clip_matches=clip_only,
            keyword_matches=[],
            transcript_keyword_matches=[],
            k=60,
            recall_params=_RECALL_PARAMS,
            include_clip=False,
        )

        # The CLIP-only file is still present in the score map (the
        # combiner iterates every unique file_id), but its combined
        # score is 0 because every channel that could have matched it
        # contributed weight 0.
        assert scores["f_clip"].combined_score == 0.0
