"""Stand-ins for the external pieces of the ``haas_local`` provider.

Replaced: ``sherpa_onnx`` (recognizer), ``huggingface_hub`` (model fetch),
``faster_whisper.vad`` (speech spans) and the ``ffmpeg`` executable (a shell
script first on PATH). Everything else in the provider runs for real.
"""

from __future__ import annotations

import os
import stat
import sys
import threading
import time
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = "oboroge0/Haas-ja-v1.0-onnx-quantized"
COMMIT = "7f5ab0a4b1bf39db78afd6691eeecbc4f9d42e39"
MODEL_FILES = (
    "encoder.quantized.onnx",
    "encoder.data",
    "decoder.onnx",
    "joiner.onnx",
    "tokens.txt",
    "LICENSE",
    "NOTICE",
)
SAMPLE_RATE = 16000

_FFMPEG_SCRIPT = """#!/bin/sh
printf '%s\\n' "$@" >> "$HAAS_FFMPEG_LOG"
echo '--END--' >> "$HAAS_FFMPEG_LOG"
case "$HAAS_FFMPEG_MODE" in
  exit1) exit 1 ;;
  killed) kill -9 $$ ;;
esac
sleep "${HAAS_FFMPEG_DELAY:-0}"
head -c "$HAAS_FFMPEG_BYTES" /dev/zero
"""


class HfHubHTTPError(Exception):
    pass


class RepositoryNotFoundError(HfHubHTTPError):
    pass


class RevisionNotFoundError(HfHubHTTPError):
    pass


class EntryNotFoundError(HfHubHTTPError):
    pass


class LocalEntryNotFoundError(EntryNotFoundError, FileNotFoundError, ValueError):
    pass


def result(tokens, timestamps, durations=()):
    """One recognizer result as sherpa-onnx's ``stream.result`` exposes it."""
    return SimpleNamespace(
        text="".join(tokens),
        tokens=list(tokens),
        timestamps=list(timestamps),
        durations=list(durations),
    )


class Haas:
    def __init__(self, monkeypatch, tmp_path, make_settings):
        self.mp = monkeypatch
        self.tmp = tmp_path
        self.make_settings = make_settings
        self.model_cache_dir = tmp_path / "model-cache"
        self.model_dir = (
            self.model_cache_dir
            / "models--oboroge0--Haas-ja-v1.0-onnx-quantized"
            / "snapshots"
            / COMMIT
        )
        self.audio_file = tmp_path / "input.mp3"
        self.audio_file.write_bytes(b"x")

        self.audio_seconds = 10
        self.spans: list[tuple[float, float]] = [(1.0, 4.0)]
        self.vad_exc: BaseException | None = None
        self.vad_calls: list[dict] = []

        self.results: list = []
        self.decode_calls: list[dict] = []
        self.decode_exc: BaseException | None = None
        self.decode_exc_on_call: int | None = None
        self.decode_hook = None

        self.hf_calls: list[dict] = []
        self.net_error: BaseException | None = None
        self.net_hook = None

        self.builds: list[dict] = []
        self.build_exc: BaseException | None = None

        self.threads: dict[str, set[int]] = {}
        self._active = 0
        self.max_active = 0
        self._state_lock = threading.Lock()

        self.ffmpeg_log = tmp_path / "ffmpeg.log"
        self._install_ffmpeg()
        self._install_modules()
        self.configure()

    # -- environment -------------------------------------------------------

    def _install_ffmpeg(self) -> None:
        bindir = self.tmp / "fakebin"
        bindir.mkdir()
        script = bindir / "ffmpeg"
        script.write_text(_FFMPEG_SCRIPT)
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        self.bindir = bindir
        self.mp.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
        self.mp.setenv("HAAS_FFMPEG_LOG", str(self.ffmpeg_log))
        self.mp.setenv("HAAS_FFMPEG_MODE", "ok")
        self.mp.setenv("HAAS_FFMPEG_DELAY", "0")
        self.set_audio_seconds(self.audio_seconds)

    def set_audio_seconds(self, seconds: float) -> None:
        self.audio_seconds = seconds
        self.mp.setenv(
            "HAAS_FFMPEG_BYTES", str(int(SAMPLE_RATE * seconds) * 4)
        )

    def ffmpeg_mode(self, mode: str) -> None:
        self.mp.setenv("HAAS_FFMPEG_MODE", mode)

    def ffmpeg_delay(self, seconds: float) -> None:
        self.mp.setenv("HAAS_FFMPEG_DELAY", str(seconds))

    def no_ffmpeg(self) -> None:
        empty = self.tmp / "emptybin"
        empty.mkdir(exist_ok=True)
        self.mp.setenv("PATH", str(empty))

    def ffmpeg_args(self) -> list[list[str]]:
        if not self.ffmpeg_log.exists():
            return []
        calls: list[list[str]] = [[]]
        for line in self.ffmpeg_log.read_text().splitlines():
            if line == "--END--":
                calls.append([])
            else:
                calls[-1].append(line)
        return [c for c in calls if c]

    def _record_thread(self, op: str) -> None:
        self.threads.setdefault(op, set()).add(threading.get_ident())

    def _enter(self) -> None:
        with self._state_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)

    def _exit(self) -> None:
        with self._state_lock:
            self._active -= 1

    def _install_modules(self) -> None:
        h = self

        class _Stream:
            def __init__(self):
                self.result = None
                self.samples = None
                self.rate = None

            def accept_waveform(self, sample_rate, samples):
                self.rate = sample_rate
                self.samples = samples

        class _Recognizer:
            def create_stream(self):
                return _Stream()

            def decode_stream(self, stream):
                h._record_thread("decode")
                h._enter()
                try:
                    call = {
                        "rate": stream.rate,
                        "n": len(stream.samples),
                        "samples": stream.samples,
                    }
                    h.decode_calls.append(call)
                    if h.decode_hook is not None:
                        h.decode_hook(len(h.decode_calls))
                    if h.decode_exc is not None and (
                        h.decode_exc_on_call is None
                        or h.decode_exc_on_call == len(h.decode_calls)
                    ):
                        raise h.decode_exc
                    if h.results:
                        stream.result = h.results.pop(0)
                    else:
                        stream.result = result(["あ"], [0.0], [0.1])
                finally:
                    h._exit()

        class _OfflineRecognizer:
            @classmethod
            def from_transducer(
                cls, encoder, decoder, joiner, tokens, **kwargs
            ):
                h._record_thread("build")
                h._enter()
                try:
                    h.builds.append(
                        {
                            "encoder": str(encoder),
                            "decoder": str(decoder),
                            "joiner": str(joiner),
                            "tokens": str(tokens),
                            **kwargs,
                        }
                    )
                    if h.build_exc is not None:
                        raise h.build_exc
                    return _Recognizer()
                finally:
                    h._exit()

        sherpa = types.ModuleType("sherpa_onnx")
        sherpa.OfflineRecognizer = _OfflineRecognizer
        self.sherpa = sherpa

        def snapshot_download(
            repo_id,
            *,
            revision=None,
            cache_dir=None,
            local_files_only=False,
            allow_patterns=None,
            **kw,
        ):
            h._record_thread("fetch")
            h._enter()
            try:
                h.hf_calls.append(
                    {
                        "repo_id": repo_id,
                        "revision": revision,
                        "cache_dir": cache_dir,
                        "local_files_only": local_files_only,
                        "allow_patterns": allow_patterns,
                        "extra": kw,
                    }
                )
                if local_files_only:
                    if h.model_dir.exists():
                        return str(h.model_dir)
                    raise LocalEntryNotFoundError("not cached")
                if h.net_hook is not None:
                    h.net_hook()
                if h.net_error is not None:
                    raise h.net_error
                h.model_dir.mkdir(parents=True, exist_ok=True)
                for name in MODEL_FILES:
                    (h.model_dir / name).write_bytes(b"0")
                return str(h.model_dir)
            finally:
                h._exit()

        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = snapshot_download
        errors = types.ModuleType("huggingface_hub.errors")
        utils = types.ModuleType("huggingface_hub.utils")
        for mod in (hub, errors, utils):
            for cls in (
                HfHubHTTPError,
                RepositoryNotFoundError,
                RevisionNotFoundError,
                EntryNotFoundError,
                LocalEntryNotFoundError,
            ):
                setattr(mod, cls.__name__, cls)
        hub.errors = errors
        hub.utils = utils
        self.hub = hub

        class _VadOptions:
            def __init__(self, **kw):
                self.kw = kw

        def get_speech_timestamps(audio, vad_options=None, **kwargs):
            h._record_thread("vad")
            opts = dict(vad_options.kw) if vad_options is not None else {}
            opts.update(kwargs)
            h.vad_calls.append(
                {"n": len(audio), "dtype": str(audio.dtype), "options": opts}
            )
            if h.vad_exc is not None:
                raise h.vad_exc
            return [
                {"start": int(s * SAMPLE_RATE), "end": int(e * SAMPLE_RATE)}
                for s, e in h.spans
            ]

        vad = types.ModuleType("faster_whisper.vad")
        vad.VadOptions = _VadOptions
        vad.get_speech_timestamps = get_speech_timestamps
        self.vad = vad

        self.mp.setitem(sys.modules, "sherpa_onnx", sherpa)
        self.mp.setitem(sys.modules, "huggingface_hub", hub)
        self.mp.setitem(sys.modules, "huggingface_hub.errors", errors)
        self.mp.setitem(sys.modules, "huggingface_hub.utils", utils)
        self.mp.setitem(sys.modules, "faster_whisper.vad", vad)
        self.mp.setattr(sys.modules["faster_whisper"], "vad", vad, raising=False)

    # -- configuration -----------------------------------------------------

    def configure(
        self,
        *,
        provider="haas_local",
        idle_unload=300,
        haas=None,
        **haas_kwargs,
    ):
        import app.config as cfg
        from app.config import MemoryConfig, TranscriptionConfig

        transcription = TranscriptionConfig(provider=provider)
        if haas is not None or haas_kwargs:
            transcription = replace(
                transcription,
                haas_local=haas or cfg.HaasLocalProviderConfig(**haas_kwargs),
            )
        settings = replace(
            self.make_settings(model_cache_dir=self.model_cache_dir),
            transcription=transcription,
            memory=MemoryConfig(whisper_idle_unload=idle_unload),
        )
        self.settings = settings
        self.mp.setattr(cfg, "settings", settings)
        from app.workers import whisper as whisper_mod

        self.mp.setattr(whisper_mod, "settings", settings)

    def precache(self, files=MODEL_FILES) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            (self.model_dir / name).write_bytes(b"0")

    # -- the provider ------------------------------------------------------

    def fresh_module(self) -> None:
        """Re-import the provider module so no recognizer or lock carries over."""
        self.mp.delitem(sys.modules, "app.workers.transcription.haas_local", raising=False)

    def inner(self):
        from app.workers.transcription import build_inner_provider

        self.fresh_module()
        return build_inner_provider("haas_local")

    def wrapped(self):
        from app.workers.transcription import get_provider

        self.fresh_module()
        return get_provider("haas_local")

    def indexer(self):
        """An Indexer imported after the provider module, so both see the same one."""
        self.mp.delitem(sys.modules, "app.indexer", raising=False)
        import app.indexer as indexer_mod

        self.mp.setattr(indexer_mod, "settings", self.settings)
        return indexer_mod.IndexManager()

    async def idle_pass(self, indexer) -> float:
        """Run one iteration of the indexer's idle-unload worker; return seconds taken."""
        import asyncio

        real_sleep = asyncio.sleep
        calls = {"n": 0}

        async def fake_sleep(_delay, *a, **k):
            calls["n"] += 1
            if calls["n"] > 1:
                raise asyncio.CancelledError
            await real_sleep(0)

        started = time.perf_counter()
        with self.mp.context() as m:
            m.setattr(asyncio, "sleep", fake_sleep)
            await indexer._idle_unload_worker()
        return time.perf_counter() - started


@pytest.fixture()
def haas(monkeypatch, tmp_path, make_settings):
    return Haas(monkeypatch, tmp_path, make_settings)
