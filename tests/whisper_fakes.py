"""Stand-ins for faster-whisper 1.1.0, plus audio files and settings for the whisper worker."""

from __future__ import annotations

import math
import os
import stat
import struct
import uuid
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


def write_tone(path: Path, seconds: float, rate: int = 22050) -> Path:
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
                for i in range(frames)
            )
        )
    return path


def write_empty_wav(path: Path, rate: int = 22050) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"")
    return path


def write_garbage(path: Path) -> Path:
    path.write_bytes(b"this is not an audio file at all" * 64)
    return path


def segment(text: str, start: float, end: float, **quality) -> SimpleNamespace:
    tokens = text.split()
    step = (end - start) / max(len(tokens), 1)
    words = [
        SimpleNamespace(
            start=start + i * step,
            end=start + (i + 1) * step,
            word=" " + tok,
            probability=0.9,
        )
        for i, tok in enumerate(tokens)
    ]
    return SimpleNamespace(
        id=0,
        seek=0,
        start=start,
        end=end,
        text=" " + text,
        tokens=[1, 2, 3],
        avg_logprob=quality.get("avg_logprob", -0.1),
        compression_ratio=quality.get("compression_ratio", 1.0),
        no_speech_prob=quality.get("no_speech_prob", 0.0),
        words=words,
        temperature=0.0,
    )


DEFAULT_SEGMENTS = (segment("hello there", 0.0, 2.0),)


def _info(language: str) -> SimpleNamespace:
    return SimpleNamespace(
        language=language,
        language_probability=0.99,
        duration=45.0,
        duration_after_vad=45.0,
        all_language_probs=None,
        transcription_options=None,
        vad_options=None,
    )


class FakeWhisperModel:
    """Enforces faster-whisper 1.1.0's ``detect_language(audio=ndarray, ...)``."""

    def __init__(
        self,
        *,
        language: str = "ja",
        probability: float = 0.9,
        sampling_rate: int = 16000,
        detect_raises: Exception | None = None,
        segments=DEFAULT_SEGMENTS,
    ) -> None:
        self.feature_extractor = SimpleNamespace(sampling_rate=sampling_rate)
        self.language = language
        self.probability = probability
        self.detect_raises = detect_raises
        self.segments = tuple(segments)
        self.detect_calls: list[object] = []
        self.detect_kwargs: list[dict] = []
        self.transcribe_calls: list[dict] = []

    def detect_language(
        self,
        audio=None,
        features=None,
        vad_filter=False,
        vad_parameters=None,
        language_detection_segments=1,
        language_detection_threshold=0.5,
    ):
        self.detect_calls.append(audio)
        self.detect_kwargs.append(
            {
                "features": features,
                "vad_filter": vad_filter,
                "vad_parameters": vad_parameters,
            }
        )
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"audio must be a numpy.ndarray, got {type(audio).__name__}")
        if self.detect_raises is not None:
            raise self.detect_raises
        return (
            self.language,
            self.probability,
            [(self.language, self.probability)],
        )

    def transcribe(self, audio, **kwargs):
        self.transcribe_calls.append(dict(kwargs))
        return iter(self.segments), _info(self.language)


class FakeBatchedPipeline:
    def __init__(self, model: FakeWhisperModel) -> None:
        self.model = model
        self.transcribe_calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.transcribe_calls.append(dict(kwargs))
        return iter(self.model.segments), _info(self.model.language)


def ffmpeg_spy(tmp_path: Path, monkeypatch, real_ffmpeg: str) -> Path:
    """Put an ``ffmpeg`` first on PATH that logs each invocation, then runs the real one."""
    bin_dir = tmp_path / f"spybin-{uuid.uuid4().hex[:8]}"
    bin_dir.mkdir()
    log = bin_dir / "calls.log"
    script = bin_dir / "ffmpeg"
    script.write_text(
        f'#!/bin/sh\necho "$@" >> "{log}"\nexec "{real_ffmpeg}" "$@"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return log


def load_settings_from(tmp_path: Path, monkeypatch, data: dict):
    import app.config as cfg

    yml = tmp_path / f"search-config-{uuid.uuid4().hex[:8]}.yml"
    yml.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("SEARCH_CONFIG_PATH", str(yml))
    monkeypatch.setenv("INTELLIGENCE_DATA_DIR", str(tmp_path / "intelligence-data"))
    return cfg.load_settings()


def use_settings(monkeypatch, settings) -> None:
    import app.config as cfg
    from app.workers import whisper as whisper_mod

    monkeypatch.setattr(whisper_mod, "settings", settings)
    monkeypatch.setattr(cfg, "settings", settings)
