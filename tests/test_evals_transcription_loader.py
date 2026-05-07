"""Tests for the case YAML loader."""

from __future__ import annotations

import textwrap

import pytest

from app.evals_transcription.loader import (
    duration_within_tolerance,
    load_cases,
)


def _write_case(
    cases_dir, name: str, body: str, audio_bytes: bytes = b"\x00" * 64
) -> None:
    """Write a YAML case + dummy audio fixture."""
    audio_dir = cases_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.joinpath(f"{name}.wav").write_bytes(audio_bytes)
    cases_dir.joinpath(f"{name}.yml").write_text(body, encoding="utf-8")


def test_loader_reads_minimal_case(tmp_path) -> None:
    body = textwrap.dedent(
        """
        name: minimal
        audio_path: audio/minimal.wav
        language: ja
        duration_s: 5.0
        tier: short
        reference_transcript: |
          こんにちは
          世界
        """
    ).strip()
    _write_case(tmp_path, "minimal", body)
    cases = load_cases(tmp_path)
    assert len(cases) == 1
    case = cases[0]
    assert case.name == "minimal"
    assert case.language == "ja"
    assert case.duration_s == 5.0
    assert case.tier == "short"
    # Literal block preserves the newline between the two lines
    assert "\n" in case.reference_transcript
    assert case.speakers == ()
    assert case.split_test is None


def test_loader_skips_yml_example_files(tmp_path) -> None:
    body = textwrap.dedent(
        """
        name: real
        audio_path: audio/real.wav
        language: ja
        duration_s: 5.0
        tier: short
        reference_transcript: |
          x
        """
    ).strip()
    _write_case(tmp_path, "real", body)
    # An example file must be ignored even when its YAML is otherwise
    # valid; the loader only looks at *.yml.
    (tmp_path / "sample.yml.example").write_text(body, encoding="utf-8")
    cases = load_cases(tmp_path)
    assert {c.name for c in cases} == {"real"}


def test_loader_returns_empty_for_empty_directory(tmp_path) -> None:
    assert load_cases(tmp_path) == []


def test_loader_raises_for_missing_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_cases(tmp_path / "nope")


def test_loader_rejects_unknown_tier(tmp_path) -> None:
    body = textwrap.dedent(
        """
        name: bad_tier
        audio_path: audio/bad_tier.wav
        language: ja
        duration_s: 5.0
        tier: ginormous
        reference_transcript: |
          x
        """
    ).strip()
    _write_case(tmp_path, "bad_tier", body)
    with pytest.raises(ValueError, match="tier"):
        load_cases(tmp_path)


def test_loader_rejects_missing_audio_file(tmp_path) -> None:
    body = textwrap.dedent(
        """
        name: missing_audio
        audio_path: audio/does_not_exist.wav
        language: ja
        duration_s: 5.0
        tier: short
        reference_transcript: |
          x
        """
    ).strip()
    (tmp_path / "missing_audio.yml").write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match="audio_path"):
        load_cases(tmp_path)


def test_loader_rejects_speaker_segment_out_of_range(tmp_path) -> None:
    body = textwrap.dedent(
        """
        name: bad_speakers
        audio_path: audio/bad_speakers.wav
        language: ja
        duration_s: 5.0
        tier: short
        reference_transcript: |
          x
        speakers:
          - id: A
            segments:
              - [0.0, 100.0]
        """
    ).strip()
    _write_case(tmp_path, "bad_speakers", body)
    with pytest.raises(ValueError, match="out of"):
        load_cases(tmp_path)


def test_loader_parses_split_test(tmp_path) -> None:
    body = textwrap.dedent(
        """
        name: with_split
        audio_path: audio/with_split.wav
        language: ja
        duration_s: 5.0
        tier: long
        reference_transcript: |
          x
        split_test:
          forced_cap_bytes: 100000
          providers:
            - openai_compatible
            - gemini
        """
    ).strip()
    _write_case(tmp_path, "with_split", body)
    cases = load_cases(tmp_path)
    case = cases[0]
    assert case.split_test is not None
    assert case.split_test.forced_cap_bytes == 100000
    assert case.split_test.providers == ("openai_compatible", "gemini")


def test_loader_rejects_invalid_yaml(tmp_path) -> None:
    """Truly malformed YAML (unclosed bracket) → invalid YAML error."""
    (tmp_path / "broken.yml").write_text(
        "name: [unclosed\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid YAML"):
        load_cases(tmp_path)


def test_loader_returns_alphabetical_order(tmp_path) -> None:
    body = textwrap.dedent(
        """
        name: NAME_PLACEHOLDER
        audio_path: audio/NAME_PLACEHOLDER.wav
        language: ja
        duration_s: 5.0
        tier: short
        reference_transcript: |
          x
        """
    ).strip()
    for name in ["zulu", "alpha", "mike"]:
        _write_case(tmp_path, name, body.replace("NAME_PLACEHOLDER", name))
    cases = load_cases(tmp_path)
    assert [c.name for c in cases] == ["alpha", "mike", "zulu"]


def test_duration_within_tolerance_uses_absolute_floor() -> None:
    # 2 s case → ±0.5 s floor (5% would only be ±0.1 s).
    assert duration_within_tolerance(2.0, 2.4) is True
    assert duration_within_tolerance(2.0, 1.6) is True
    assert duration_within_tolerance(2.0, 2.6) is False


def test_duration_within_tolerance_uses_relative_for_long() -> None:
    # 100 s case → ±5 s (5% > 0.5 s).
    assert duration_within_tolerance(100.0, 104.0) is True
    assert duration_within_tolerance(100.0, 96.0) is True
    assert duration_within_tolerance(100.0, 110.0) is False


def test_loader_preserves_literal_block_newlines(tmp_path) -> None:
    """`|` literal block must keep newlines (R0 review M4 contract)."""
    body = textwrap.dedent(
        """
        name: literal
        audio_path: audio/literal.wav
        language: ja
        duration_s: 5.0
        tier: short
        reference_transcript: |
          line one
          line two
        """
    ).strip()
    _write_case(tmp_path, "literal", body)
    cases = load_cases(tmp_path)
    assert "line one\nline two" in cases[0].reference_transcript
