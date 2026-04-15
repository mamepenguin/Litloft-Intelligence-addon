"""Tests for the :mod:`app.workers.aligner` wrapper state management.

These cover the acquire/release reference counter and the waveform
cache — the WhisperX-backed align_segment itself is exercised by the
integration path (Phase 7 manual check) since mocking wav2vec2 output
doesn't add useful coverage over the refine-side tests that already
mock ``aligner.align_segment``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.workers import aligner


def test_release_job_frees_only_after_last_job():
    aligner.release_all()
    # Fake two loaded models.
    aligner._align_models["ja"] = (MagicMock(), MagicMock())
    aligner._align_models["en"] = (MagicMock(), MagicMock())
    aligner._waveform_cache["/a"] = object()

    aligner.acquire_job()
    aligner.acquire_job()
    aligner.acquire_job()

    # First two releases: state must remain loaded.
    aligner.release_job()
    assert len(aligner._align_models) == 2
    aligner.release_job()
    assert len(aligner._align_models) == 2

    # Last release: state cleared.
    aligner.release_job()
    assert aligner._align_models == {}
    assert aligner._waveform_cache == {}


def test_release_job_floors_counter_at_zero():
    aligner.release_all()
    # Unpaired release shouldn't go negative or crash.
    aligner.release_job()
    aligner.release_job()
    assert aligner._active_jobs == 0


def test_load_waveform_caches_result(monkeypatch):
    aligner.release_all()
    call_count = {"n": 0}

    def _fake_whisperx_load_audio(path):
        call_count["n"] += 1
        return f"waveform-for-{path}"

    fake_module = MagicMock()
    fake_module.load_audio = _fake_whisperx_load_audio
    monkeypatch.setitem(__import__("sys").modules, "whisperx", fake_module)

    w1 = aligner.load_waveform("/path/a.mp4")
    w2 = aligner.load_waveform("/path/a.mp4")
    w3 = aligner.load_waveform("/path/b.mp4")

    assert w1 == "waveform-for-/path/a.mp4"
    assert w2 is w1
    assert w3 == "waveform-for-/path/b.mp4"
    assert call_count["n"] == 2


def test_load_waveform_returns_none_on_failure(monkeypatch):
    aligner.release_all()
    fake_module = MagicMock()
    fake_module.load_audio = MagicMock(side_effect=RuntimeError("decode failed"))
    monkeypatch.setitem(__import__("sys").modules, "whisperx", fake_module)

    assert aligner.load_waveform("/missing.mp4") is None


def test_align_segment_returns_none_without_waveform():
    aligner.release_all()
    assert aligner.align_segment(
        waveform=None,
        chunk_start=0.0,
        chunk_end=1.0,
        text="hello",
        language="en",
    ) is None


def test_align_segment_returns_none_on_empty_text():
    aligner.release_all()
    assert aligner.align_segment(
        waveform=object(),
        chunk_start=0.0,
        chunk_end=1.0,
        text="   ",
        language="en",
    ) is None


def test_align_segment_returns_none_on_bad_range():
    aligner.release_all()
    assert aligner.align_segment(
        waveform=object(),
        chunk_start=5.0,
        chunk_end=5.0,
        text="hello",
        language="en",
    ) is None
