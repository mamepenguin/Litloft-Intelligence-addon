"""SPEC-ADDON-039: ``transcription.haas_local`` settings, the provider default, and what loads when."""

from __future__ import annotations

import importlib.abc
import logging
import sys
from unittest.mock import MagicMock

import pytest

for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.config import TranscriptionConfig  # noqa: E402
from tests.haas_fakes import haas  # noqa: E402,F401  (fixture)
from tests.whisper_fakes import load_settings_from  # noqa: E402


def _haas(tmp_path, monkeypatch, section):
    data = {"transcription": {"haas_local": section}} if section is not None else {}
    return load_settings_from(tmp_path, monkeypatch, data).transcription.haas_local


def test_spec_addon_039_defaults(tmp_path, monkeypatch):
    absent = _haas(tmp_path, monkeypatch, None)
    empty = _haas(tmp_path, monkeypatch, {})

    for cfg in (absent, empty, TranscriptionConfig().haas_local):
        assert cfg.num_threads == 4
        assert cfg.max_segment_seconds == 45


def test_spec_addon_039_i5_provider_defaults_to_whisper_local():
    assert TranscriptionConfig().provider == "whisper_local"
    assert TranscriptionConfig().haas_local.num_threads == 4


@pytest.mark.parametrize("threads", [1, 2, 4, 16])
def test_spec_addon_039_i3_num_threads_accepted(tmp_path, monkeypatch, caplog, threads):
    with caplog.at_level(logging.WARNING):
        cfg = _haas(tmp_path, monkeypatch, {"num_threads": threads})
    assert cfg.num_threads == threads
    assert not [r for r in caplog.records if "num_threads" in r.getMessage()]


@pytest.mark.parametrize("seconds", [15, 16, 45, 119, 120])
def test_spec_addon_039_i3_max_segment_seconds_accepted(
    tmp_path, monkeypatch, caplog, seconds
):
    with caplog.at_level(logging.WARNING):
        cfg = _haas(tmp_path, monkeypatch, {"max_segment_seconds": seconds})
    assert cfg.max_segment_seconds == seconds
    assert not [r for r in caplog.records if "max_segment_seconds" in r.getMessage()]


@pytest.mark.parametrize("threads", [0, -1, 2.5, "4", True, 4.0])
def test_spec_addon_039_i3_num_threads_out_of_range_falls_back_with_one_warning(
    tmp_path, monkeypatch, caplog, threads
):
    with caplog.at_level(logging.WARNING):
        cfg = _haas(tmp_path, monkeypatch, {"num_threads": threads})

    assert cfg.num_threads == 4
    assert len([r for r in caplog.records if "num_threads" in r.getMessage()]) == 1


@pytest.mark.parametrize(
    "seconds", [0, 14, 121, 1000, -45, 45.5, "45", True, 45.0]
)
def test_spec_addon_039_i3_max_segment_seconds_out_of_range_falls_back_with_one_warning(
    tmp_path, monkeypatch, caplog, seconds
):
    with caplog.at_level(logging.WARNING):
        cfg = _haas(tmp_path, monkeypatch, {"max_segment_seconds": seconds})

    assert cfg.max_segment_seconds == 45
    assert (
        len([r for r in caplog.records if "max_segment_seconds" in r.getMessage()])
        == 1
    )


@pytest.mark.parametrize("key", ["num_threads", "max_segment_seconds"])
def test_spec_addon_039_i3_null_falls_back_silently(
    tmp_path, monkeypatch, caplog, key
):
    with caplog.at_level(logging.WARNING):
        cfg = _haas(tmp_path, monkeypatch, {key: None})

    assert (cfg.num_threads, cfg.max_segment_seconds) == (4, 45)
    assert not [r for r in caplog.records if key in r.getMessage()]


def test_spec_addon_039_i3_one_bad_value_leaves_the_other(tmp_path, monkeypatch):
    cfg = _haas(tmp_path, monkeypatch, {"num_threads": 0, "max_segment_seconds": 30})
    assert (cfg.num_threads, cfg.max_segment_seconds) == (4, 30)


class _Guard(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.hits: list[str] = []

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in {"sherpa_onnx", "huggingface_hub"}:
            self.hits.append(name)
            raise ImportError(name)
        return None


@pytest.mark.asyncio
async def test_spec_addon_039_i5_other_provider_never_touches_the_engine(
    haas, monkeypatch
):
    for name in ("sherpa_onnx", "huggingface_hub", "huggingface_hub.errors",
                 "huggingface_hub.utils"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    guard = _Guard()
    monkeypatch.setattr(sys, "meta_path", [guard, *sys.meta_path])
    haas.configure(provider="whisper_local", idle_unload=1)

    from app.workers.transcription import build_inner_provider

    haas.fresh_module()
    build_inner_provider("haas_local")
    indexer = haas.indexer()
    await haas.idle_pass(indexer)

    assert guard.hits == []
