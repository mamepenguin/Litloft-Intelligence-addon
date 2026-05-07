"""Unit tests for the transcription overrides reader/writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.transcription_overrides import (
    SCHEMA_VERSION,
    TranscriptionOverrides,
    delete_overrides,
    merge_into_dict,
    overrides_path,
    read_overrides,
    write_overrides,
)


def test_read_returns_none_when_file_absent(tmp_path: Path) -> None:
    assert read_overrides(tmp_path) is None


def test_write_then_read_roundtrips_all_fields(tmp_path: Path) -> None:
    write_overrides(
        TranscriptionOverrides(
            provider="deepgram",
            language_hint="ja",
            hotwords=("Litloft", "Cloudflare"),
        ),
        data_dir=tmp_path,
        updated_at="2026-05-08T00:00:00Z",
    )
    out = read_overrides(tmp_path)
    assert out is not None
    assert out.provider == "deepgram"
    assert out.language_hint == "ja"
    assert out.hotwords == ("Litloft", "Cloudflare")


def test_write_emits_atomic_rename_target(tmp_path: Path) -> None:
    """No leftover ``*.tmp`` and the target file is the canonical name."""
    write_overrides(
        TranscriptionOverrides(provider="deepgram"),
        data_dir=tmp_path,
    )
    target = overrides_path(tmp_path)
    assert target.is_file()
    # Atomic rename should not leave a temp sibling.
    assert not (tmp_path / "transcription-overrides.json.tmp").exists()


def test_write_omits_absent_fields_so_baseline_is_preserved(tmp_path: Path) -> None:
    write_overrides(
        TranscriptionOverrides(provider="deepgram"),
        data_dir=tmp_path,
    )
    raw = json.loads(overrides_path(tmp_path).read_text(encoding="utf-8"))
    assert raw["provider"] == "deepgram"
    assert "language_hint" not in raw
    assert "hotwords" not in raw


def test_write_keeps_empty_string_language_hint(tmp_path: Path) -> None:
    """Empty string is meaningful (= no hint, distinct from absent)."""
    write_overrides(
        TranscriptionOverrides(language_hint=""),
        data_dir=tmp_path,
    )
    raw = json.loads(overrides_path(tmp_path).read_text(encoding="utf-8"))
    assert raw["language_hint"] == ""


def test_read_ignores_unknown_schema_version(tmp_path: Path) -> None:
    overrides_path(tmp_path).write_text(
        json.dumps({"schema_version": 99, "provider": "deepgram"}),
        encoding="utf-8",
    )
    assert read_overrides(tmp_path) is None


def test_read_returns_none_for_malformed_json(tmp_path: Path) -> None:
    overrides_path(tmp_path).write_text("not json at all", encoding="utf-8")
    assert read_overrides(tmp_path) is None


def test_read_returns_none_for_non_object_top_level(tmp_path: Path) -> None:
    overrides_path(tmp_path).write_text("[1, 2, 3]", encoding="utf-8")
    assert read_overrides(tmp_path) is None


def test_read_drops_fields_with_wrong_types(tmp_path: Path) -> None:
    """Defence in depth: a corrupt writer cannot smuggle a list as
    provider — the typed dataclass falls back to ``None``."""
    overrides_path(tmp_path).write_text(
        json.dumps({
            "schema_version": SCHEMA_VERSION,
            "provider": ["not", "a", "string"],
            "language_hint": 42,
            "hotwords": [1, 2, 3],
        }),
        encoding="utf-8",
    )
    out = read_overrides(tmp_path)
    assert out is not None
    assert out.provider is None
    assert out.language_hint is None
    assert out.hotwords is None


def test_delete_returns_false_when_absent(tmp_path: Path) -> None:
    assert delete_overrides(tmp_path) is False


def test_delete_returns_true_when_removed(tmp_path: Path) -> None:
    write_overrides(TranscriptionOverrides(provider="deepgram"), data_dir=tmp_path)
    assert delete_overrides(tmp_path) is True
    assert not overrides_path(tmp_path).exists()


def test_merge_into_dict_preserves_baseline_when_overrides_absent() -> None:
    base = {
        "provider": "whisper_local",
        "language_hint": "",
        "hotwords": ["x"],
    }
    out = merge_into_dict(base, None)
    assert out == base
    # Defensive: not the same dict object.
    assert out is not base


def test_merge_into_dict_overrides_only_set_fields() -> None:
    base = {
        "provider": "whisper_local",
        "language_hint": "",
        "hotwords": ["x"],
    }
    out = merge_into_dict(
        base,
        TranscriptionOverrides(provider="deepgram"),
    )
    assert out["provider"] == "deepgram"
    assert out["language_hint"] == ""
    assert out["hotwords"] == ["x"]


def test_merge_into_dict_distinguishes_empty_string_from_none() -> None:
    """An override with ``language_hint=""`` overrides the baseline;
    an override with ``language_hint=None`` (absent key) leaves it."""
    base = {"language_hint": "ja", "provider": "x", "hotwords": []}
    out_empty = merge_into_dict(
        base,
        TranscriptionOverrides(language_hint=""),
    )
    assert out_empty["language_hint"] == ""

    out_none = merge_into_dict(
        base,
        TranscriptionOverrides(language_hint=None),
    )
    assert out_none["language_hint"] == "ja"
