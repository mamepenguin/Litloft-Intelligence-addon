"""SPEC-ADDON-009, SPEC-ADDON-012: whisper_local detects the language from the speech in decoded audio and applies the built-in prompt."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest

from app.workers import whisper as whisper_mod
from app.workers.whisper_prompts import DEFAULT_INITIAL_PROMPTS
from tests.whisper_fakes import (
    FakeBatchedPipeline,
    FakeWhisperModel,
    ffmpeg_spy,
    load_settings_from,
    use_settings,
    write_empty_wav,
    write_garbage,
    write_tone,
)

REAL_FFMPEG = shutil.which("ffmpeg")
JA_PROMPT = DEFAULT_INITIAL_PROMPTS["ja"]


def _records(caplog, level: int) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.name.startswith("app.workers.whisper") and r.levelno == level
    ]


class SpeechSpy:
    def __init__(self) -> None:
        self.seconds: float = 30.0
        self.raises: Exception | None = None
        self.calls: list[tuple[object, object]] = []

    def __call__(self, audio, sampling_rate):
        self.calls.append((audio, sampling_rate))
        if self.raises is not None:
            raise self.raises
        return self.seconds


# The real helper runs Silero VAD through faster-whisper, which conftest stubs out.
@pytest.fixture(autouse=True)
def speech(monkeypatch) -> SpeechSpy:
    spy = SpeechSpy()
    monkeypatch.setattr(whisper_mod, "_speech_seconds", spy, raising=False)
    return spy


@pytest.fixture()
def tone_45s(tmp_path):
    return write_tone(tmp_path / "long.wav", 45.0)


@pytest.fixture(scope="module")
def tone_130s(tmp_path_factory):
    return write_tone(tmp_path_factory.mktemp("tone") / "opening.wav", 130.0, rate=8000)


class TestDetectLanguage:
    @pytest.mark.parametrize("sampling_rate", [16000, 8000])
    def test_spec_addon_009_012_passes_at_most_120s_of_samples_at_the_models_rate_with_vad(
        self, tone_130s, sampling_rate
    ):
        model = FakeWhisperModel(language="ja", probability=0.9, sampling_rate=sampling_rate)

        assert whisper_mod._detect_language(model, str(tone_130s)) == "ja"

        assert len(model.detect_calls) == 1
        audio = model.detect_calls[0]
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert audio.dtype == np.float32
        assert 0.95 * 120 * sampling_rate <= audio.shape[0] <= 120 * sampling_rate
        assert model.detect_kwargs[0]["vad_filter"] is True

    @pytest.mark.parametrize(
        "probability, expected",
        [(0.9, "ja"), (0.5, "ja"), (0.49, None), (0.1, None)],
    )
    def test_spec_addon_009_returns_language_only_at_probability_half_or_more(
        self, tone_45s, caplog, probability, expected
    ):
        model = FakeWhisperModel(language="ja", probability=probability)

        with caplog.at_level(logging.INFO):
            got = whisper_mod._detect_language(model, str(tone_45s))

        assert got == expected
        assert len(model.detect_calls) == 1
        assert isinstance(model.detect_calls[0], np.ndarray)
        assert _records(caplog, logging.WARNING) == []
        if expected is None:
            assert _records(caplog, logging.INFO)

    def test_spec_addon_009_unreadable_file_returns_none_with_one_warning(
        self, tmp_path, caplog
    ):
        model = FakeWhisperModel()
        bad = write_garbage(tmp_path / "broken.wav")

        with caplog.at_level(logging.INFO):
            got = whisper_mod._detect_language(model, str(bad))

        assert got is None
        assert all(isinstance(a, np.ndarray) for a in model.detect_calls)
        assert len(_records(caplog, logging.WARNING)) == 1

    def test_spec_addon_009_ffmpeg_that_cannot_start_returns_none_with_one_warning(
        self, tone_45s, tmp_path, monkeypatch, caplog
    ):
        empty_bin = tmp_path / "nobin"
        empty_bin.mkdir()
        monkeypatch.setenv("PATH", str(empty_bin))
        model = FakeWhisperModel()

        with caplog.at_level(logging.INFO):
            got = whisper_mod._detect_language(model, str(tone_45s))

        assert got is None
        assert all(isinstance(a, np.ndarray) for a in model.detect_calls)
        assert len(_records(caplog, logging.WARNING)) == 1

    def test_spec_addon_009_decode_timeout_is_60s_and_returns_none_with_one_warning(
        self, tone_45s, monkeypatch, caplog
    ):
        seen: list[object] = []

        def _timeout(cmd, *args, **kwargs):
            seen.append(kwargs.get("timeout"))
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or 0)

        monkeypatch.setattr(subprocess, "run", _timeout)
        model = FakeWhisperModel()

        with caplog.at_level(logging.INFO):
            got = whisper_mod._detect_language(model, str(tone_45s))

        assert got is None
        assert seen == [60]
        assert model.detect_calls == []
        assert len(_records(caplog, logging.WARNING)) == 1

    def test_spec_addon_009_detect_language_raising_returns_none_with_one_warning(
        self, tone_45s, caplog
    ):
        model = FakeWhisperModel(detect_raises=RuntimeError("model exploded"))

        with caplog.at_level(logging.INFO):
            got = whisper_mod._detect_language(model, str(tone_45s))

        assert got is None
        assert len(model.detect_calls) == 1
        assert isinstance(model.detect_calls[0], np.ndarray)
        assert len(_records(caplog, logging.WARNING)) == 1

    def test_spec_addon_009_no_samples_returns_none_without_calling_the_model(
        self, tmp_path, caplog
    ):
        model = FakeWhisperModel()
        silent = write_empty_wav(tmp_path / "empty.wav")

        with caplog.at_level(logging.INFO):
            got = whisper_mod._detect_language(model, str(silent))

        assert got is None
        assert model.detect_calls == []
        assert _records(caplog, logging.WARNING) == []
        assert _records(caplog, logging.INFO)


def _configure(tmp_path, monkeypatch, whisper_local: dict | None = None, legacy: dict | None = None):
    data: dict = {}
    if legacy is not None:
        data["indexing"] = {"whisper": legacy}
    if whisper_local is not None:
        data["transcription"] = {"whisper_local": whisper_local}
    use_settings(monkeypatch, load_settings_from(tmp_path, monkeypatch, data))


def _install(monkeypatch, model, pipeline=None):
    monkeypatch.setattr(whisper_mod, "_ensure_loaded", lambda: (model, pipeline))


class TestTranscribeFilePrompt:
    @pytest.mark.parametrize("batched", [False, True], ids=["sequential", "batched"])
    @pytest.mark.parametrize(
        "file_kind, probability, expected",
        [
            ("tone", 0.9, JA_PROMPT),
            ("tone", 0.5, JA_PROMPT),
            ("tone", 0.3, None),
            ("garbage", 0.9, None),
        ],
        ids=["ja-0.9", "ja-0.5", "ja-0.3", "unreadable"],
    )
    def test_spec_addon_009_transcribes_with_the_prompt_of_the_detected_language(
        self, tmp_path, monkeypatch, batched, file_kind, probability, expected
    ):
        both = {"batch_size": 4} if batched else {"batch_size": 0}
        _configure(tmp_path, monkeypatch, whisper_local=dict(both), legacy=dict(both))
        model = FakeWhisperModel(language="ja", probability=probability)
        pipeline = FakeBatchedPipeline(model) if batched else None
        _install(monkeypatch, model, pipeline)
        path = (
            write_tone(tmp_path / "clip.wav", 45.0)
            if file_kind == "tone"
            else write_garbage(tmp_path / "clip.wav")
        )

        whisper_mod._transcribe_file(str(path))

        calls = pipeline.transcribe_calls if batched else model.transcribe_calls
        assert len(calls) == 1
        assert calls[0].get("initial_prompt") == expected
        assert all(isinstance(a, np.ndarray) for a in model.detect_calls)

    def test_spec_addon_009_language_outside_the_built_in_ten_gets_no_prompt(
        self, tmp_path, monkeypatch
    ):
        assert "xx" not in DEFAULT_INITIAL_PROMPTS
        _configure(tmp_path, monkeypatch)
        model = FakeWhisperModel(language="xx", probability=0.99)
        _install(monkeypatch, model)

        whisper_mod._transcribe_file(str(write_tone(tmp_path / "clip.wav", 45.0)))

        assert len(model.detect_calls) == 1
        assert isinstance(model.detect_calls[0], np.ndarray)
        assert model.transcribe_calls[0].get("initial_prompt") is None

    @pytest.mark.parametrize(
        "config_prompt, caller_prompt, expected, detects",
        [
            ("domain glossary.", None, "domain glossary.", False),
            ("", "previous piece text.", "previous piece text.", False),
            ("   \n", None, JA_PROMPT, True),
            ("", "", JA_PROMPT, True),
            ("", "  \t", JA_PROMPT, True),
        ],
        ids=[
            "config-override",
            "caller-prior-text",
            "whitespace-config-is-absent",
            "empty-caller-text-is-absent",
            "whitespace-caller-text-is-absent",
        ],
    )
    def test_spec_addon_009_override_or_prior_text_skips_detection_entirely(
        self, tmp_path, monkeypatch, config_prompt, caller_prompt, expected, detects
    ):
        _configure(tmp_path, monkeypatch, whisper_local={"initial_prompt": config_prompt})
        model = FakeWhisperModel(language="ja", probability=0.95)
        _install(monkeypatch, model)
        log = ffmpeg_spy(tmp_path, monkeypatch, REAL_FFMPEG)
        path = write_tone(tmp_path / "clip.wav", 45.0)

        if caller_prompt is None:
            whisper_mod._transcribe_file(str(path))
        else:
            whisper_mod._transcribe_file(str(path), initial_prompt_override=caller_prompt)

        assert model.transcribe_calls[0].get("initial_prompt") == expected
        ran_ffmpeg = log.exists() and log.read_text().strip() != ""
        if detects:
            assert len(model.detect_calls) == 1
            assert isinstance(model.detect_calls[0], np.ndarray)
            assert ran_ffmpeg
        else:
            assert model.detect_calls == []
            assert not ran_ffmpeg


class TestSpeechGate:
    def test_spec_addon_012_vad_sees_the_decoded_samples_at_the_models_rate(
        self, tone_130s, speech
    ):
        model = FakeWhisperModel(language="ja", probability=0.9, sampling_rate=8000)

        assert whisper_mod._detect_language(model, str(tone_130s)) == "ja"

        assert len(speech.calls) == 1
        audio, rate = speech.calls[0]
        assert rate == 8000
        assert isinstance(audio, np.ndarray)
        assert 0.95 * 120 * 8000 <= audio.shape[0] <= 120 * 8000
        assert np.array_equal(audio, model.detect_calls[0])

    @pytest.mark.parametrize("seconds", [0.0, 0.5, 0.99])
    def test_spec_addon_012_less_than_one_second_of_speech_skips_detection_with_one_info(
        self, tone_45s, speech, caplog, seconds
    ):
        speech.seconds = seconds
        model = FakeWhisperModel(language="en", probability=0.615)

        with caplog.at_level(logging.INFO):
            got = whisper_mod._detect_language(model, str(tone_45s))

        assert got is None
        assert len(speech.calls) == 1
        assert model.detect_calls == []
        assert len(_records(caplog, logging.INFO)) == 1
        assert _records(caplog, logging.WARNING) == []

    @pytest.mark.parametrize("seconds", [1.0, 1.5, 120.0])
    def test_spec_addon_012_one_second_of_speech_or_more_detects_with_vad(
        self, tone_45s, speech, caplog, seconds
    ):
        speech.seconds = seconds
        model = FakeWhisperModel(language="ja", probability=0.9)

        with caplog.at_level(logging.INFO):
            got = whisper_mod._detect_language(model, str(tone_45s))

        assert got == "ja"
        assert len(speech.calls) == 1
        assert len(model.detect_calls) == 1
        assert model.detect_kwargs[0]["vad_filter"] is True
        assert _records(caplog, logging.WARNING) == []

    def test_spec_addon_012_vad_raising_returns_none_with_one_warning(
        self, tone_45s, speech, caplog
    ):
        speech.raises = RuntimeError("silero exploded")
        model = FakeWhisperModel(language="ja", probability=0.9)

        with caplog.at_level(logging.INFO):
            got = whisper_mod._detect_language(model, str(tone_45s))

        assert got is None
        assert len(speech.calls) == 1
        assert model.detect_calls == []
        assert len(_records(caplog, logging.WARNING)) == 1

    def test_spec_addon_012_no_samples_runs_neither_vad_nor_detection(
        self, tmp_path, speech, caplog
    ):
        model = FakeWhisperModel()
        silent = write_empty_wav(tmp_path / "empty.wav")

        with caplog.at_level(logging.INFO):
            got = whisper_mod._detect_language(model, str(silent))

        assert got is None
        assert speech.calls == []
        assert model.detect_calls == []
        assert len(_records(caplog, logging.INFO)) == 1
        assert _records(caplog, logging.WARNING) == []


class TestSpeechGateTranscribe:
    @pytest.mark.parametrize("batched", [False, True], ids=["sequential", "batched"])
    @pytest.mark.parametrize(
        "seconds, raises, expected, detects",
        [
            (0.5, None, None, False),
            (0.0, None, None, False),
            (1.0, None, JA_PROMPT, True),
            (30.0, RuntimeError("silero exploded"), None, False),
        ],
        ids=["half-second", "no-speech", "one-second", "vad-raises"],
    )
    def test_spec_addon_012_prompt_follows_the_speech_gate(
        self, tmp_path, monkeypatch, speech, batched, seconds, raises, expected, detects
    ):
        both = {"batch_size": 4} if batched else {"batch_size": 0}
        _configure(tmp_path, monkeypatch, whisper_local=dict(both), legacy=dict(both))
        speech.seconds = seconds
        speech.raises = raises
        model = FakeWhisperModel(language="ja", probability=0.9)
        pipeline = FakeBatchedPipeline(model) if batched else None
        _install(monkeypatch, model, pipeline)

        whisper_mod._transcribe_file(str(write_tone(tmp_path / "clip.wav", 45.0)))

        calls = pipeline.transcribe_calls if batched else model.transcribe_calls
        assert len(calls) == 1
        assert calls[0].get("initial_prompt") == expected
        assert len(speech.calls) == 1
        assert len(model.detect_calls) == (1 if detects else 0)


class TestSpeechSeconds:
    @staticmethod
    def _install_vad(monkeypatch, chunks):
        seen: list[dict] = []

        def get_speech_timestamps(audio, vad_options=None, sampling_rate=16000, **kwargs):
            seen.append(
                {"audio": audio, "vad_options": vad_options, "kwargs": kwargs}
            )
            return [dict(c) for c in chunks]

        vad = ModuleType("faster_whisper.vad")
        vad.get_speech_timestamps = get_speech_timestamps
        monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
        monkeypatch.setattr(sys.modules["faster_whisper"], "vad", vad, raising=False)
        return seen

    @pytest.mark.parametrize(
        "chunks, expected",
        [
            ([], 0.0),
            ([{"start": 0, "end": 8000}], 0.5),
            ([{"start": 16000, "end": 32000}, {"start": 48000, "end": 56000}], 1.5),
        ],
        ids=["none", "half-second", "two-chunks"],
    )
    def test_spec_addon_012_returns_total_speech_seconds(
        self, monkeypatch, chunks, expected
    ):
        # Undoes the autouse stand-in so the real helper runs against the fake VAD.
        monkeypatch.undo()
        seen = self._install_vad(monkeypatch, chunks)
        audio = np.zeros(16000 * 4, dtype=np.float32)

        got = whisper_mod._speech_seconds(audio, 16000)

        assert got == pytest.approx(expected)
        assert len(seen) == 1
        assert seen[0]["audio"] is audio
        assert set(seen[0]["kwargs"]) <= {"sampling_rate"}
