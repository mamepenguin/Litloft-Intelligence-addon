"""Tests for ``_resolve_model_size`` — the config-name → faster-whisper
model identifier mapping.

The resolver accepts three input shapes:

* Known ``openai/whisper-*`` aliases → short size name
* faster-whisper size shortcuts and HF CT2 repo IDs → passed through verbatim
* Unknown strings → passed through so faster-whisper raises a clear error
  instead of silently loading a different model
"""

from app.workers.whisper import _resolve_model_size


def test_openai_aliases_map_to_short_names():
    assert _resolve_model_size("openai/whisper-tiny") == "tiny"
    assert _resolve_model_size("openai/whisper-base") == "base"
    assert _resolve_model_size("openai/whisper-small") == "small"
    assert _resolve_model_size("openai/whisper-medium") == "medium"
    assert _resolve_model_size("openai/whisper-large") == "large-v3"
    assert _resolve_model_size("openai/whisper-large-v3") == "large-v3"


def test_openai_turbo_alias_maps_to_turbo():
    assert (
        _resolve_model_size("openai/whisper-large-v3-turbo") == "large-v3-turbo"
    )


def test_faster_whisper_shortcut_passes_through():
    assert _resolve_model_size("large-v3-turbo") == "large-v3-turbo"
    assert _resolve_model_size("large-v3") == "large-v3"
    assert _resolve_model_size("distil-large-v3") == "distil-large-v3"


def test_huggingface_repo_id_passes_through():
    repo_id = "deepdml/faster-whisper-large-v3-turbo-ct2"
    assert _resolve_model_size(repo_id) == repo_id


def test_unknown_string_passes_through_for_clear_error():
    # Previously the resolver silently fell back to "small"; now it
    # returns the input so faster-whisper surfaces the real mistake.
    assert _resolve_model_size("not-a-real-model") == "not-a-real-model"
