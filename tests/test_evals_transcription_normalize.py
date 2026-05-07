"""Tests for the language-aware text normalizer used by the eval harness."""

from __future__ import annotations

import pytest

from app.evals_transcription.normalize import normalize


def test_nfkc_full_width_to_half_width() -> None:
    assert normalize("ＡＢＣ１２３", "en") == "abc123"


def test_english_lowercases() -> None:
    assert normalize("Hello World", "en") == "hello world"


def test_japanese_katakana_to_hiragana() -> None:
    """Providers vary on script preference; we fold to hiragana so the
    same word matches regardless of which side the provider chose."""
    assert normalize("カタカナ", "ja") == "かたかな"


def test_japanese_strips_punctuation_and_collapses_whitespace() -> None:
    out = normalize("こんにちは、世界。元気？", "ja")
    assert "、" not in out and "。" not in out and "？" not in out
    assert "  " not in out


def test_english_strips_punctuation() -> None:
    assert normalize("Hello, world! It's me.", "en") == "hello world it s me"


def test_unsupported_language_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        normalize("hola", "es")


def test_language_subtag_accepted() -> None:
    """``en-US`` / ``ja-JP`` should be treated as their primary tag."""
    assert normalize("Hello", "en-US") == "hello"
    assert normalize("カタカナ", "ja-JP") == "かたかな"


def test_idempotent() -> None:
    once = normalize("こんにちは、世界", "ja")
    twice = normalize(once, "ja")
    assert once == twice


def test_non_string_raises_type_error() -> None:
    with pytest.raises(TypeError):
        normalize(42, "en")  # type: ignore[arg-type]


def test_empty_string_returns_empty_string() -> None:
    assert normalize("", "en") == ""
    assert normalize("   ", "en") == ""
