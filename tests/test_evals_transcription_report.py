"""Tests for markdown + JSON sidecar emission."""

from __future__ import annotations

import json

from app.evals_transcription.loader import Case, SplitTest
from app.evals_transcription.metrics import SpeakerSegment
from app.evals_transcription.report import write_report
from app.evals_transcription.runner import CaseResult


def _case(name: str = "c1", tier: str = "short", language: str = "en"):
    return Case(
        name=name,
        case_path=f"/tmp/{name}.yml",
        audio_path=f"/tmp/audio/{name}.wav",
        language=language,
        duration_s=5.0,
        tier=tier,
        reference_transcript="hello world",
    )


def _ok_result(
    case_name: str,
    provider_name: str,
    *,
    mode: str = "default",
    wer: float | None = 0.05,
    cer: float | None = 0.07,
    sa_wer: float | None = None,
    latency: float | None = 1.5,
    detected: str | None = "en",
):
    return CaseResult(
        case_name=case_name,
        provider_name=provider_name,
        mode=mode,
        wer=wer,
        cer=cer,
        sa_wer=sa_wer,
        latency_s=latency,
        detected_language=detected,
        hypothesis="hello earth",
    )


def test_write_report_creates_md_and_json_sidecar(tmp_path) -> None:
    output = tmp_path / "report.md"
    write_report(
        results=[_ok_result("c1", "whisper_local")],
        cases=[_case("c1")],
        output_path=output,
        baseline_path=None,
    )
    assert output.exists()
    sidecar = output.with_suffix(".json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["schema_version"] == 1
    assert payload["n_results"] == 1
    assert payload["results"][0]["case_name"] == "c1"


def test_aggregate_table_lists_provider_row(tmp_path) -> None:
    output = tmp_path / "report.md"
    write_report(
        results=[
            _ok_result("c1", "whisper_local"),
            _ok_result("c1", "deepgram"),
        ],
        cases=[_case("c1")],
        output_path=output,
        baseline_path=None,
    )
    md = output.read_text()
    assert "| provider | N | WER" in md
    assert "whisper_local" in md
    assert "deepgram" in md


def test_whisper_family_rows_get_dagger_marker(tmp_path) -> None:
    """Whisper-family providers must be flagged so readers don't take
    self-curated GT comparisons at face value."""
    output = tmp_path / "report.md"
    write_report(
        results=[
            _ok_result("c1", "whisper_local"),
            _ok_result("c1", "deepgram"),
        ],
        cases=[_case("c1")],
        output_path=output,
        baseline_path=None,
    )
    md = output.read_text()
    assert "whisper_local†" in md
    # Deepgram should NOT be marked
    assert "deepgram†" not in md
    # Footnote text present
    assert "Whisper-family" in md


def test_skipped_section_appears_when_provider_skipped(tmp_path) -> None:
    output = tmp_path / "report.md"
    write_report(
        results=[
            _ok_result("c1", "whisper_local"),
            CaseResult(
                case_name="c1",
                provider_name="deepgram",
                mode="default",
                wer=None,
                cer=None,
                sa_wer=None,
                latency_s=None,
                detected_language=None,
                skipped=True,
                skipped_reason="DEEPGRAM_API_KEY not configured",
            ),
        ],
        cases=[_case("c1")],
        output_path=output,
        baseline_path=None,
    )
    md = output.read_text()
    assert "## Skipped" in md
    assert "DEEPGRAM_API_KEY" in md


def test_split_overhead_section_present_when_split_modes_emitted(tmp_path) -> None:
    output = tmp_path / "report.md"
    write_report(
        results=[
            _ok_result("c1", "openai_compatible"),
            _ok_result("c1", "openai_compatible", mode="no_split", wer=0.04),
            _ok_result("c1", "openai_compatible", mode="split", wer=0.05),
        ],
        cases=[_case("c1", tier="long")],
        output_path=output,
        baseline_path=None,
    )
    md = output.read_text()
    assert "## Phase 2B split overhead" in md


def test_per_tier_breakdown(tmp_path) -> None:
    output = tmp_path / "report.md"
    write_report(
        results=[
            _ok_result("c1", "whisper_local"),
            _ok_result("c2", "whisper_local"),
        ],
        cases=[
            _case("c1", tier="short"),
            _case("c2", tier="medium"),
        ],
        output_path=output,
        baseline_path=None,
    )
    md = output.read_text()
    assert "Tier: short" in md
    assert "Tier: medium" in md


def test_japanese_only_case_shows_dash_for_wer(tmp_path) -> None:
    """ja-only WER comes back as None from score_text → table cell --."""
    output = tmp_path / "report.md"
    write_report(
        results=[
            _ok_result(
                "ja_case", "whisper_local", wer=None, cer=0.08
            ),
        ],
        cases=[_case("ja_case", language="ja")],
        output_path=output,
        baseline_path=None,
    )
    md = output.read_text()
    # The aggregate row must display -- in the WER column, not 0
    assert "| -- |" in md
