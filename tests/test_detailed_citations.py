"""Unit tests for the detailed_summary parser + citation calculator.

Parser covers:

* ``##`` section detection with per-section counters
* Paragraph accumulation across multi-line runs
* Bullet extraction including nested bullets
* Table rows (GFM header + separator + body, malformed tables tolerated)
* Section replacement for the Phase 2 edit flow

Citation calculator covers:

* Full compute path with embeddings stubbed to deterministic vectors
* Persistence round-trip (``write_citations`` → ``get_citations``)
* Threshold-driven ``has_citation`` flag
* Segments with empty or embedding-failing text degrade gracefully
* ``calculate_and_store`` returns (cited_count, no_citation_count)
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

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
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import numpy as np  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseSegments:
    """`parse_segments` must split the 4-section detailed_summary format."""

    def test_empty_input_returns_empty_list(self):
        from app.summary_parser import parse_segments
        assert parse_segments("") == []
        assert parse_segments("   \n\n") == []

    def test_paragraph_segment_with_section_path(self):
        from app.summary_parser import parse_segments

        markdown = (
            "## 全体像\n"
            "本動画は3つのレシピを紹介する。\n"
            "各レシピには分量と手順が含まれる。\n"
        )
        segments = parse_segments(markdown)
        assert len(segments) == 1
        assert segments[0].section_path == "全体像/0"
        assert segments[0].segment_type == "paragraph"
        assert "本動画は3つのレシピ" in segments[0].segment_text
        assert "分量と手順" in segments[0].segment_text

    def test_multiple_paragraphs_have_incrementing_index(self):
        from app.summary_parser import parse_segments

        markdown = (
            "## 全体像\n"
            "第一段落。\n"
            "\n"
            "第二段落。\n"
        )
        segments = parse_segments(markdown)
        assert [s.section_path for s in segments] == [
            "全体像/0",
            "全体像/1",
        ]

    def test_bullets_become_individual_segments(self):
        from app.summary_parser import parse_segments

        markdown = (
            "## 主要な章/場面\n"
            "- 第1章: 序章\n"
            "- 第2章: 本編\n"
            "- 第3章: 結論\n"
        )
        segments = parse_segments(markdown)
        # Section title contains a "/" — which the path scheme reuses. The
        # parser should still use the full heading text verbatim.
        assert [s.section_path for s in segments] == [
            "主要な章/場面/0",
            "主要な章/場面/1",
            "主要な章/場面/2",
        ]
        assert all(s.segment_type == "bullet" for s in segments)
        assert segments[0].segment_text == "第1章: 序章"

    def test_nested_bullets_flatten_with_own_paths(self):
        from app.summary_parser import parse_segments

        markdown = (
            "## 主要な章\n"
            "- 親項目\n"
            "  - ネスト1\n"
            "  - ネスト2\n"
            "- 次の親項目\n"
        )
        segments = parse_segments(markdown)
        # Expect 4 bullets (parent1, nested1, nested2, parent2) and
        # monotonically increasing counter.
        assert len(segments) == 4
        assert [s.segment_text for s in segments] == [
            "親項目",
            "ネスト1",
            "ネスト2",
            "次の親項目",
        ]

    def test_table_rows_become_bullet_segments(self):
        from app.summary_parser import parse_segments

        markdown = (
            "## 重要ポイントまとめ\n"
            "| 項目 | 内容 |\n"
            "|---|---|\n"
            "| 時間 | 30分 |\n"
            "| 温度 | 180度 |\n"
        )
        segments = parse_segments(markdown)
        # Header + separator are skipped, 2 body rows remain.
        assert len(segments) == 2
        assert all(s.segment_type == "bullet" for s in segments)
        assert [s.section_path for s in segments] == [
            "重要ポイントまとめ/row/0",
            "重要ポイントまとめ/row/1",
        ]
        assert segments[0].segment_text == "時間 | 30分"

    def test_headings_without_body_produce_no_segments(self):
        from app.summary_parser import parse_segments

        markdown = "## 全体像\n\n## 詳細\n- only this\n"
        segments = parse_segments(markdown)
        assert len(segments) == 1
        assert segments[0].section_path == "詳細/0"

    def test_full_4_section_document(self):
        from app.summary_parser import parse_segments

        markdown = (
            "## 全体像\n"
            "概要段落。\n"
            "\n"
            "## 重要ポイントまとめ\n"
            "| k | v |\n"
            "|---|---|\n"
            "| a | 1 |\n"
            "| b | 2 |\n"
            "\n"
            "## 主要な章/場面\n"
            "- 章1\n"
            "- 章2\n"
            "\n"
            "## 要確認の固有名詞\n"
            "- 「堀井」: 表記揺れの可能性\n"
        )
        segments = parse_segments(markdown)
        paths = [s.section_path for s in segments]
        assert paths == [
            "全体像/0",
            "重要ポイントまとめ/row/0",
            "重要ポイントまとめ/row/1",
            "主要な章/場面/0",
            "主要な章/場面/1",
            "要確認の固有名詞/0",
        ]


class TestReplaceSectionBody:
    """Phase 2 edit helper must preserve surrounding sections."""

    def test_replaces_target_section_only(self):
        from app.summary_parser import replace_section_body

        source = (
            "## A\n"
            "original A\n"
            "\n"
            "## B\n"
            "original B\n"
        )
        result = replace_section_body(source, "A", "edited A body")
        assert "edited A body" in result
        # B section untouched.
        assert "original B" in result
        # Original A body removed.
        assert "original A" not in result
        # Headings preserved exactly.
        assert "## A" in result
        assert "## B" in result

    def test_replacement_of_last_section(self):
        from app.summary_parser import replace_section_body

        source = (
            "## A\n"
            "body A\n"
            "\n"
            "## B\n"
            "body B\n"
        )
        result = replace_section_body(source, "B", "new B body")
        assert "new B body" in result
        assert "body A" in result
        assert "body B" not in result

    def test_missing_section_raises(self):
        from app.summary_parser import replace_section_body

        with pytest.raises(ValueError):
            replace_section_body("## A\nbody\n", "missing", "whatever")

    def test_empty_markdown_raises(self):
        from app.summary_parser import replace_section_body

        with pytest.raises(ValueError):
            replace_section_body("", "anything", "body")

    def test_multiline_replacement_preserved(self):
        from app.summary_parser import replace_section_body

        source = "## A\nold\n\n## B\nb\n"
        new_content = "line 1\n\nline 2\n- bullet"
        result = replace_section_body(source, "A", new_content)
        assert "line 1" in result
        assert "line 2" in result
        assert "- bullet" in result
        # B stays.
        assert "## B" in result


# ---------------------------------------------------------------------------
# Citation calculator tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def citations_db(tmp_path, monkeypatch):
    """Real SQLite DB wired into ``app.citations.get_search_db``.

    The citations module writes to ``detailed_summary_citations``, so we
    set up the table via the canonical creator — this also ensures the
    DDL stays aligned between tests and production.
    """
    db_path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    from app.database import _create_detailed_summary_citations_table

    with engine.begin() as conn:
        _create_detailed_summary_citations_table(conn)

    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _get_search_db():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("app.citations.get_search_db", _get_search_db)
    return engine


class TestWriteAndGetCitations:
    """Citations round-trip through the DB without embedding the text."""

    def test_write_and_fetch(self, citations_db):
        from app.citations import write_citations, get_citations

        citations = [
            {
                "section_path": "全体像/0",
                "segment_type": "paragraph",
                "segment_text": "本動画は…",
                "citation_chunk_ids": ["transcript:5", "transcript:7"],
                "top_score": 0.82,
                "has_citation": True,
            },
            {
                "section_path": "主要な章/場面/0",
                "segment_type": "bullet",
                "segment_text": "第1章",
                "citation_chunk_ids": [],
                "top_score": 0.31,
                "has_citation": False,
            },
        ]
        cited, no_cite = write_citations("file-1", citations)
        assert cited == 1
        assert no_cite == 1

        rows = get_citations("file-1")
        assert len(rows) == 2
        assert rows[0]["section_path"] == "全体像/0"
        assert rows[0]["chunk_ids"] == ["transcript:5", "transcript:7"]
        assert rows[0]["has_citation"] is True
        assert rows[1]["has_citation"] is False

    def test_write_replaces_existing_rows(self, citations_db):
        from app.citations import write_citations, get_citations

        first = [
            {
                "section_path": "全体像/0",
                "segment_type": "paragraph",
                "segment_text": "old",
                "citation_chunk_ids": ["transcript:1"],
                "top_score": 0.9,
                "has_citation": True,
            },
        ]
        write_citations("file-1", first)

        second = [
            {
                "section_path": "全体像/0",
                "segment_type": "paragraph",
                "segment_text": "new",
                "citation_chunk_ids": [],
                "top_score": 0.1,
                "has_citation": False,
            },
        ]
        write_citations("file-1", second)

        rows = get_citations("file-1")
        assert len(rows) == 1
        assert rows[0]["segment_text"] == "new"
        assert rows[0]["has_citation"] is False


class TestComputeCitations:
    """End-to-end compute path with embedder + vec_text stubbed out."""

    def test_segments_below_threshold_get_has_citation_false(
        self, citations_db, monkeypatch,
    ):
        from app import citations

        # Embedder returns the same unit vector for every segment so
        # the KNN stub's score dictates the outcome.
        monkeypatch.setattr(
            citations, "_embed_segment", lambda seg: np.ones(4, dtype=np.float32)
        )

        # Simulate "file has only loose matches" — score below default 0.55.
        monkeypatch.setattr(
            citations,
            "_query_top_chunks",
            lambda fid, vec, k: [("transcript:0", 0.3)],
        )

        markdown = "## 全体像\n段落本文。\n- 箇条書き\n"
        computed = citations.compute_citations("file-1", markdown)
        assert len(computed) == 2
        assert all(c["has_citation"] is False for c in computed)
        assert all(c["citation_chunk_ids"] == [] for c in computed)

    def test_segments_above_threshold_keep_passing_chunks_only(
        self, citations_db, monkeypatch,
    ):
        from app import citations

        monkeypatch.setattr(
            citations, "_embed_segment", lambda seg: np.ones(4, dtype=np.float32)
        )

        # Three candidates; only the first two clear the default 0.55.
        monkeypatch.setattr(
            citations,
            "_query_top_chunks",
            lambda fid, vec, k: [
                ("transcript:0", 0.82),
                ("transcript:1", 0.64),
                ("transcript:2", 0.40),
            ],
        )

        computed = citations.compute_citations(
            "file-1", "## A\nonly paragraph.\n"
        )
        assert len(computed) == 1
        assert computed[0]["has_citation"] is True
        assert computed[0]["top_score"] == pytest.approx(0.82)
        # The weak third candidate should not persist.
        assert computed[0]["citation_chunk_ids"] == [
            "transcript:0",
            "transcript:1",
        ]

    def test_empty_segment_text_degrades_without_crash(
        self, citations_db, monkeypatch,
    ):
        from app import citations

        # Embedder returns None to simulate a vectoriser failure; the
        # row must still be produced so UI can render the ⚠ badge.
        monkeypatch.setattr(citations, "_embed_segment", lambda seg: None)

        computed = citations.compute_citations(
            "file-1", "## A\n- item\n"
        )
        assert len(computed) == 1
        assert computed[0]["has_citation"] is False
        assert computed[0]["citation_chunk_ids"] == []

    def test_calculate_and_store_returns_counts(
        self, citations_db, monkeypatch,
    ):
        from app import citations

        monkeypatch.setattr(
            citations, "_embed_segment", lambda seg: np.ones(4, dtype=np.float32)
        )
        # Alternate strong / weak matches so we can verify the counts.
        scores = iter([
            [("transcript:0", 0.80)],
            [("transcript:1", 0.20)],
        ])
        monkeypatch.setattr(
            citations, "_query_top_chunks",
            lambda fid, vec, k: next(scores),
        )

        markdown = "## A\n- one\n- two\n"
        cited, no_cite = citations.calculate_and_store("file-1", markdown)
        assert cited == 1
        assert no_cite == 1

        rows = citations.get_citations("file-1")
        assert len(rows) == 2


class TestQueryTopChunksDistanceConversion:
    """sqlite-vec returns L2 distance; citations must convert to cosine.

    Regression test for the bug where ``_query_top_chunks`` used the
    naive ``score = 1 - distance`` formula. sqlite-vec's ``vec0`` virtual
    table returns Euclidean distance, and for L2-normalised vectors the
    correct inversion is ``cos = 1 − d²/2``. Using the wrong formula
    under-estimates cosine similarity for every non-identical match
    (e.g. true cos 0.55 ⇢ L2 ≈ 0.949 ⇢ naive ≈ 0.05), which silently
    drops almost every real summary segment below the default 0.55
    threshold — leaving the UI stuck on ⚠ for everything.
    """

    def _stub_execute(self, rows):
        """Build a ``session.execute`` stub returning the given rows."""
        class _FakeResult:
            def __init__(self, data):
                self._data = data

            def fetchall(self):
                return list(self._data)

        class _FakeSession:
            def execute(self, _stmt, _params):
                return _FakeResult(rows)

        @contextmanager
        def _get_search_db():
            yield _FakeSession()

        return _get_search_db

    def test_l2_zero_maps_to_cosine_one(self, monkeypatch):
        """Identical vectors (L2 = 0) must score 1.0, not 1.0 by luck."""
        from app import citations

        monkeypatch.setattr(
            citations, "get_search_db",
            self._stub_execute([("wh_file-1_0_abc", 0.0)]),
        )
        results = citations._query_top_chunks(
            "file-1", np.ones(4, dtype=np.float32), top_k=3
        )
        assert results == [("transcript:0", pytest.approx(1.0))]

    def test_real_distance_converts_via_cos_formula(self, monkeypatch):
        """L2 ≈ 0.949 corresponds to cosine ≈ 0.55, not 0.05.

        Before the fix this row would score 1 − 0.949 ≈ 0.051 and fail
        the default ``citation_threshold`` 0.55. After the fix it lands
        exactly on the threshold, matching the documented contract that
        the config value is a cosine-similarity floor.
        """
        from app import citations

        # L2 = sqrt(2 - 2·0.55) → cosine 0.55 exactly.
        l2 = (2.0 * (1.0 - 0.55)) ** 0.5
        monkeypatch.setattr(
            citations, "get_search_db",
            self._stub_execute([("wh_file-1_3_xyz", l2)]),
        )
        results = citations._query_top_chunks(
            "file-1", np.ones(4, dtype=np.float32), top_k=3
        )
        assert len(results) == 1
        chunk_id, score = results[0]
        assert chunk_id == "transcript:3"
        assert score == pytest.approx(0.55, abs=1e-6)

    def test_scores_stay_sorted_descending(self, monkeypatch):
        """Two rows with different L2 distances round-trip in score order."""
        from app import citations

        rows = [
            ("wh_file-1_0_a", 0.316),  # ≈ cos 0.95
            ("wh_file-1_1_b", 0.949),  # ≈ cos 0.55
        ]
        monkeypatch.setattr(
            citations, "get_search_db", self._stub_execute(rows),
        )
        results = citations._query_top_chunks(
            "file-1", np.ones(4, dtype=np.float32), top_k=3
        )
        assert [cid for cid, _ in results] == [
            "transcript:0",
            "transcript:1",
        ]
        assert results[0][1] > results[1][1]
        assert results[0][1] == pytest.approx(0.95, abs=1e-3)
        assert results[1][1] == pytest.approx(0.55, abs=1e-3)


class TestDeleteCitations:
    def test_delete_returns_rowcount(self, citations_db):
        from app.citations import delete_citations, write_citations

        write_citations("file-1", [
            {
                "section_path": "A/0",
                "segment_type": "paragraph",
                "segment_text": "a",
                "citation_chunk_ids": ["transcript:1"],
                "top_score": 0.9,
                "has_citation": True,
            },
            {
                "section_path": "B/0",
                "segment_type": "paragraph",
                "segment_text": "b",
                "citation_chunk_ids": [],
                "top_score": 0.1,
                "has_citation": False,
            },
        ])
        assert delete_citations("file-1") == 2
        # Second call is a no-op.
        assert delete_citations("file-1") == 0
