"""Unit tests for the citation eval harness.

Covers the pure / DB-free surface: YAML loader, aggregation logic, and
report rendering. The DB-hitting parts (``resolve_file_id``,
``load_detailed_summary``, ``_hint_matches_chunk``) are covered by the
``run_case`` integration path when a snapshot is available; that isn't
exercised here because the test image ships without one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Same heavyweight-dep stubs used by test_detailed_citations — keeps
# module imports cheap without pulling torch / sentence_transformers.
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

import pytest  # noqa: E402


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestCitationCaseLoader:
    def test_minimal_case_loads(self, tmp_path: Path):
        from app.evals_citations.loader import load_case

        body = (
            "id: test_01\n"
            "file_path: audio/example.mp3\n"
            "expectations:\n"
            '  - section_path: "全体像/0"\n'
            "    must_have_citation: true\n"
        )
        p = tmp_path / "c.yml"
        p.write_text(body, encoding="utf-8")
        case = load_case(p)
        assert case.id == "test_01"
        assert case.file_path == "audio/example.mp3"
        assert len(case.expectations) == 1
        assert case.expectations[0].section_path == "全体像/0"
        assert case.expectations[0].must_have_citation is True

    def test_expectation_with_time_range(self, tmp_path: Path):
        from app.evals_citations.loader import load_case

        body = (
            "id: t2\n"
            "file_path: v/clip.mp4\n"
            "expectations:\n"
            '  - section_path: "T/row/0"\n'
            "    segment_hint:\n"
            "      time_range: [0.0, 30.0]\n"
        )
        p = tmp_path / "c.yml"
        p.write_text(body, encoding="utf-8")
        case = load_case(p)
        hint = case.expectations[0].hint
        assert hint is not None
        assert hint.time_range == (0.0, 30.0)
        assert hint.page is None

    def test_expectation_with_page(self, tmp_path: Path):
        from app.evals_citations.loader import load_case

        body = (
            "id: doc1\n"
            "file_path: docs/paper.pdf\n"
            "expectations:\n"
            '  - section_path: "A/0"\n'
            "    segment_hint:\n"
            "      page: 3\n"
        )
        p = tmp_path / "c.yml"
        p.write_text(body, encoding="utf-8")
        case = load_case(p)
        hint = case.expectations[0].hint
        assert hint is not None
        assert hint.page == 3

    def test_expectation_with_chunk_ids(self, tmp_path: Path):
        from app.evals_citations.loader import load_case

        body = (
            "id: exact\n"
            "file_path: x\n"
            "expectations:\n"
            '  - section_path: "A/0"\n'
            "    chunk_ids: [transcript:0, transcript:2]\n"
        )
        p = tmp_path / "c.yml"
        p.write_text(body, encoding="utf-8")
        case = load_case(p)
        assert case.expectations[0].chunk_ids == (
            "transcript:0",
            "transcript:2",
        )

    def test_missing_id_rejected(self, tmp_path: Path):
        from app.evals_citations.loader import load_case

        p = tmp_path / "c.yml"
        p.write_text("file_path: a\nexpectations: []\n", encoding="utf-8")
        with pytest.raises(ValueError, match="id"):
            load_case(p)

    def test_missing_file_path_rejected(self, tmp_path: Path):
        from app.evals_citations.loader import load_case

        p = tmp_path / "c.yml"
        p.write_text("id: x\nexpectations: []\n", encoding="utf-8")
        with pytest.raises(ValueError, match="file_path"):
            load_case(p)

    def test_expectation_without_any_signal_rejected(self, tmp_path: Path):
        """An expectation with only section_path is ambiguous — fail loudly."""
        from app.evals_citations.loader import load_case

        body = (
            "id: x\n"
            "file_path: a\n"
            "expectations:\n"
            '  - section_path: "A/0"\n'
        )
        p = tmp_path / "c.yml"
        p.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match="chunk_ids"):
            load_case(p)

    def test_load_cases_directory(self, tmp_path: Path):
        """Multiple yml files in a directory are all loaded."""
        from app.evals_citations.loader import load_cases

        for name in ("a.yml", "b.yml"):
            body = (
                f"id: {name.split('.')[0]}\n"
                "file_path: f\n"
                "expectations:\n"
                '  - section_path: "A/0"\n'
                "    must_have_citation: true\n"
            )
            (tmp_path / name).write_text(body, encoding="utf-8")
        cases = load_cases(tmp_path)
        assert [c.id for c in cases] == ["a", "b"]

    def test_load_cases_with_filter(self, tmp_path: Path):
        from app.evals_citations.loader import load_cases

        for name, cid in (("a.yml", "alpha"), ("b.yml", "beta")):
            body = (
                f"id: {cid}\n"
                "file_path: f\n"
                "expectations:\n"
                '  - section_path: "A/0"\n'
                "    must_have_citation: true\n"
            )
            (tmp_path / name).write_text(body, encoding="utf-8")
        cases = load_cases(tmp_path, filter_substr="alpha")
        assert [c.id for c in cases] == ["alpha"]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def _make_segment(
        self,
        stype: str = "paragraph",
        top1: bool = False,
        recall3: bool = False,
        has_cit: bool = False,
        must: bool | None = None,
    ):
        from app.evals_citations.runner import SegmentScore

        return SegmentScore(
            section_path=f"{stype}/0",
            segment_type=stype,
            top1_hit=top1,
            recall_at_3=recall3,
            has_citation=has_cit,
            must_have_citation=must,
            top_score=0.8 if top1 else 0.3,
            top_chunk_ids=("transcript:0",) if has_cit else (),
        )

    def _make_report(self, segments):
        from app.evals_citations.runner import CaseReport

        return CaseReport(
            case_id="t", file_path="f", file_id="fid", segments=tuple(segments)
        )

    def test_empty_reports_return_zeros(self):
        from app.evals_citations.runner import aggregate

        agg = aggregate([])
        assert agg.total_segments == 0
        assert agg.top1_accuracy == 0.0
        assert agg.recall_at_3 == 0.0
        assert agg.has_citation_precision == 0.0
        assert agg.missing_required_citations == 0
        assert agg.by_segment_type == {}

    def test_mixed_outcomes_aggregate_correctly(self):
        from app.evals_citations.runner import aggregate

        reports = [
            self._make_report([
                self._make_segment(
                    "paragraph", top1=True, recall3=True, has_cit=True
                ),
                self._make_segment(
                    "paragraph", top1=False, recall3=True, has_cit=True
                ),
                self._make_segment("bullet", top1=True, recall3=True),
                self._make_segment("bullet", top1=False, recall3=False),
            ])
        ]
        agg = aggregate(reports)
        assert agg.total_segments == 4
        # 2 / 4 top-1 hits
        assert agg.top1_accuracy == 0.5
        # 3 / 4 recall@3 hits
        assert agg.recall_at_3 == 0.75
        # has_cit True: 2 segments. top1 correct in them: 1. Precision = 0.5.
        assert agg.has_citation_precision == 0.5
        # By segment type: paragraph 1/2 = 0.5, bullet 1/2 = 0.5.
        assert agg.by_segment_type["paragraph"][0] == 0.5
        assert agg.by_segment_type["bullet"][0] == 0.5
        # Counts match.
        assert agg.by_segment_type["paragraph"][2] == 2
        assert agg.by_segment_type["bullet"][2] == 2

    def test_missing_required_citation_is_counted(self):
        from app.evals_citations.runner import aggregate

        report = self._make_report([
            self._make_segment(has_cit=False, must=True),
            self._make_segment(has_cit=True, must=True),
            self._make_segment(has_cit=False, must=None),
        ])
        agg = aggregate([report])
        assert agg.missing_required_citations == 1


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


class TestReportRendering:
    def _setup(self):
        from app.evals_citations.runner import (
            CaseReport,
            SegmentScore,
            aggregate,
        )

        segments = (
            SegmentScore(
                section_path="A/0",
                segment_type="paragraph",
                top1_hit=True,
                recall_at_3=True,
                has_citation=True,
                must_have_citation=True,
                top_score=0.81,
                top_chunk_ids=("transcript:0", "transcript:3"),
            ),
            SegmentScore(
                section_path="T/row/0",
                segment_type="bullet",
                top1_hit=False,
                recall_at_3=False,
                has_citation=False,
                must_have_citation=True,
                top_score=0.42,
                top_chunk_ids=(),
            ),
        )
        reports = [
            CaseReport(
                case_id="c1",
                file_path="a/x.mp3",
                file_id="fid",
                segments=segments,
            )
        ]
        agg = aggregate(reports)
        return reports, agg

    def test_markdown_has_aggregate_and_case_sections(self):
        from app.evals_citations.report import render_markdown

        reports, agg = self._setup()
        md = render_markdown(reports, agg, label="phase1")
        assert "# detailed_summary Citation Eval — phase1" in md
        assert "## Aggregate" in md
        assert "## Cases" in md
        # Missing required citation is surfaced distinctly.
        assert "❌(required)" in md
        # By-segment-type table renders.
        assert "| paragraph |" in md
        assert "| bullet |" in md

    def test_sidecar_roundtrips_as_json(self):
        from app.evals_citations.report import build_sidecar

        reports, agg = self._setup()
        side = build_sidecar(reports, agg, label="phase1")
        # JSON-serialisable (dataclass asdict is plain dicts).
        encoded = json.dumps(side, ensure_ascii=False)
        decoded = json.loads(encoded)
        assert decoded["aggregate"]["top1_accuracy"] == 0.5
        assert decoded["aggregate"]["missing_required_citations"] == 1
        assert len(decoded["cases"]) == 1

    def test_compare_aggregates_reports_deltas(self):
        from app.evals_citations.report import compare_aggregates

        baseline = {
            "aggregate": {
                "top1_accuracy": 0.4,
                "recall_at_3": 0.6,
                "has_citation_precision": 0.7,
            }
        }
        current = {
            "aggregate": {
                "top1_accuracy": 0.55,
                "recall_at_3": 0.6,
                "has_citation_precision": 0.5,
            }
        }
        md = compare_aggregates(baseline, current, epsilon=0.02)
        # Improvement on top1_accuracy (delta 0.15 > epsilon).
        assert "top1_accuracy" in md
        assert "✅" in md  # at least one improvement marker
        # Tied on recall_at_3.
        assert "(tied)" in md
        # Regression on has_citation_precision.
        assert "❌" in md
