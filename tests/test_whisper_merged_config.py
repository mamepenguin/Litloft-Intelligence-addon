"""SPEC-ADDON-011: the whisper worker reads the merged ``transcription.whisper_local`` settings.

Each case runs the worker three times: the value under the documented key only,
the same value under the legacy key only, and neither. The documented key must
act exactly as the legacy key does, and the legacy run must differ from the
default run, or the case would not show that the value is read at all.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from app.workers import whisper as whisper_mod
from tests.test_loft_vtt_sidecar import (  # noqa: F401  (Session is a fixture)
    Session,
    _chunks,
    _index,
    _make_loft,
    _seed_loft,
)
from tests.whisper_fakes import (
    FakeWhisperModel,
    load_settings_from,
    segment,
    use_settings,
    write_tone,
)

THREE_RUNS = ("documented", "legacy", "default")


def _tree(run: str, values: dict) -> dict:
    if run == "documented":
        return {"transcription": {"whisper_local": dict(values)}}
    if run == "legacy":
        return {"indexing": {"whisper": dict(values)}}
    return {}


QUALITY_SEGMENTS = (
    segment("plain speech here", 0.0, 2.0),
    segment("looping looping looping", 2.0, 4.0, compression_ratio=1.8),
    segment("maybe silence", 4.0, 6.0, no_speech_prob=0.35, avg_logprob=-0.8),
    segment("low confidence words", 6.0, 8.0, avg_logprob=-0.8),
)


def _observe_transcribe(tmp_path, monkeypatch, data: dict):
    use_settings(monkeypatch, load_settings_from(tmp_path, monkeypatch, data))
    model = FakeWhisperModel(language="en", probability=0.2, segments=QUALITY_SEGMENTS)
    monkeypatch.setattr(whisper_mod, "_ensure_loaded", lambda: (model, None))
    path = write_tone(tmp_path / "clip.wav", 5.0)

    result = whisper_mod._transcribe_file(str(path))

    return model.transcribe_calls, result


@pytest.mark.parametrize(
    "values",
    [
        {"initial_prompt": "domain glossary."},
        {"beam_size": 3},
        {"condition_on_previous_text": False},
        {"compression_ratio_threshold": 1.7},
        {"no_speech_threshold": 0.3},
        {"log_prob_threshold": -0.7},
    ],
    ids=lambda v: next(iter(v)),
)
def test_spec_addon_011_documented_transcription_key_acts_as_the_legacy_key(
    tmp_path, monkeypatch, values
):
    seen = {run: _observe_transcribe(tmp_path, monkeypatch, _tree(run, values)) for run in THREE_RUNS}

    assert seen["legacy"] != seen["default"]
    assert seen["documented"] == seen["legacy"]


def test_spec_addon_011_documented_initial_prompt_reaches_whisper(tmp_path, monkeypatch):
    calls, _ = _observe_transcribe(
        tmp_path, monkeypatch, _tree("documented", {"initial_prompt": "domain glossary."})
    )

    assert calls[0].get("initial_prompt") == "domain glossary."


def test_spec_addon_011_documented_key_wins_when_both_trees_set_it(tmp_path, monkeypatch):
    calls, _ = _observe_transcribe(
        tmp_path,
        monkeypatch,
        {
            "indexing": {"whisper": {"beam_size": 5, "initial_prompt": "old"}},
            "transcription": {"whisper_local": {"beam_size": 2, "initial_prompt": "new"}},
        },
    )

    assert calls[0].get("beam_size") == 2
    assert calls[0].get("initial_prompt") == "new"


def _fake_faster_whisper(log: list) -> types.ModuleType:
    mod = types.ModuleType("faster_whisper")

    class WhisperModel:
        def __init__(self, *args, **kwargs):
            log.append(("WhisperModel", args, kwargs))
            self.feature_extractor = types.SimpleNamespace(sampling_rate=16000)

    def _plain(value):
        return "<model>" if isinstance(value, WhisperModel) else value

    class BatchedInferencePipeline:
        def __init__(self, *args, **kwargs):
            log.append((
                "BatchedInferencePipeline",
                tuple(_plain(a) for a in args),
                {k: _plain(v) for k, v in kwargs.items()},
            ))

    mod.WhisperModel = WhisperModel
    mod.BatchedInferencePipeline = BatchedInferencePipeline
    return mod


def _observe_load(tmp_path, monkeypatch, data: dict):
    use_settings(monkeypatch, load_settings_from(tmp_path, monkeypatch, data))
    log: list = []
    monkeypatch.setitem(sys.modules, "faster_whisper", _fake_faster_whisper(log))
    monkeypatch.setattr(whisper_mod, "_model", None)
    monkeypatch.setattr(whisper_mod, "_batched_pipeline", None)
    monkeypatch.setattr(whisper_mod, "_loaded", False)

    _model, batched = whisper_mod._ensure_loaded()

    return batched is None, log


def test_spec_addon_011_documented_batch_size_acts_as_the_legacy_key(tmp_path, monkeypatch):
    values = {"batch_size": 4}
    seen = {run: _observe_load(tmp_path, monkeypatch, _tree(run, values)) for run in THREE_RUNS}

    assert seen["legacy"] != seen["default"]
    assert seen["documented"] == seen["legacy"]


def _cue_vtt(path, count: int) -> None:
    lines = ["WEBVTT", ""]
    for i in range(count):
        start, end = i * 2, i * 2 + 2
        lines += [f"00:00:{start:02d}.000 --> 00:00:{end:02d}.000", f"cue number {i}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.parametrize(
    "values",
    [
        {"min_segment_duration": 2, "max_segment_duration": 4},
        {"max_segment_duration": 6},
        {"min_segment_duration": 26},
    ],
    ids=["both", "max-only", "min-only"],
)
async def test_spec_addon_011_loft_caption_chunks_use_documented_segment_durations(
    Session, tmp_path, monkeypatch, values  # noqa: F811
):
    monkeypatch.setattr(
        whisper_mod,
        "embed_passages",
        lambda texts, *a, **k: np.zeros((len(texts), 8), dtype=np.float32),
    )
    seen = {}
    for i, run in enumerate(THREE_RUNS):
        use_settings(monkeypatch, load_settings_from(tmp_path, monkeypatch, _tree(run, values)))
        file_id = f"loft{i:08d}"
        loft = _make_loft(tmp_path / run, "Clip")
        _cue_vtt(tmp_path / run / "Clip.vtt", 25)
        _seed_loft(Session, file_id, loft, whisper_indexed=False)

        await _index(file_id)

        seen[run] = _chunks(Session, file_id)

    assert seen["legacy"] != seen["default"]
    assert seen["documented"] == seen["legacy"]
