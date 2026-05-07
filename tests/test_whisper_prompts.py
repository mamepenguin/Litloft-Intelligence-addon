"""Tests for the language-keyed initial_prompt resolver.

Whisper's ``initial_prompt`` biases the decoder toward a punctuation
style. The values are language-specific, so we ship sensible defaults
in code (keyed by detected language) and only ask users to set the
config field when they need a domain-specific override.
"""

from app.workers.whisper_prompts import (
    DEFAULT_INITIAL_PROMPTS,
    resolve_initial_prompt,
)


def test_user_override_wins_over_language_default():
    prompt = resolve_initial_prompt(
        detected_language="ja", override="my domain glossary"
    )
    assert prompt == "my domain glossary"


def test_user_override_wins_even_when_language_is_unknown():
    prompt = resolve_initial_prompt(
        detected_language=None, override="custom prompt"
    )
    assert prompt == "custom prompt"


def test_blank_override_falls_back_to_language_default():
    prompt = resolve_initial_prompt(detected_language="ja", override="")
    assert prompt == DEFAULT_INITIAL_PROMPTS["ja"]


def test_whitespace_override_treated_as_unset():
    prompt = resolve_initial_prompt(detected_language="ja", override="   \n")
    assert prompt == DEFAULT_INITIAL_PROMPTS["ja"]


def test_unknown_language_returns_none():
    prompt = resolve_initial_prompt(detected_language="xx", override="")
    assert prompt is None


def test_missing_language_returns_none():
    prompt = resolve_initial_prompt(detected_language=None, override="")
    assert prompt is None


def test_japanese_default_present():
    assert "ja" in DEFAULT_INITIAL_PROMPTS
    assert DEFAULT_INITIAL_PROMPTS["ja"].strip()


def test_english_default_present():
    assert "en" in DEFAULT_INITIAL_PROMPTS
    assert DEFAULT_INITIAL_PROMPTS["en"].strip()


def test_chinese_default_present():
    # Chinese also benefits from punctuation bias (no spaces, sparse
    # native punctuation in transcripts).
    assert "zh" in DEFAULT_INITIAL_PROMPTS
    assert DEFAULT_INITIAL_PROMPTS["zh"].strip()
