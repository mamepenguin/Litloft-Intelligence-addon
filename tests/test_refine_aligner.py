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


def _install_fake_whisperx(monkeypatch, items, *, want_chars=False):
    """Patch ``whisperx`` so ``align_segment`` sees a controlled segment.

    ``items`` shape matches the wav2vec2 align output: each item is a
    dict with ``char`` / ``word`` and ``start`` / ``end`` keys. The
    caller picks ``want_chars`` to match the language path.
    """
    fake_whisperx = MagicMock()
    key = "chars" if want_chars else "words"
    fake_whisperx.align = MagicMock(
        return_value={"segments": [{key: items}]}
    )
    monkeypatch.setitem(__import__("sys").modules, "whisperx", fake_whisperx)
    # Seed the model cache so load_align_model returns without calling
    # into whisperx.load_align_model (which would also need mocking).
    lang = "en" if not want_chars else "ja"
    aligner._align_models[lang] = (MagicMock(), MagicMock())
    return fake_whisperx


def test_align_segment_clamps_units_outside_window(monkeypatch):
    """WhisperX CTC can emit edge tokens slightly outside the segment
    window (rounding, padding frames). align_segment drops those so
    callers receive a window-bounded result by construction — no need
    for a second clamp in the refine worker.
    """
    aligner.release_all()
    _install_fake_whisperx(monkeypatch, [
        # Two units inside [1.0, 5.0].
        {"word": "hello", "start": 1.1, "end": 2.0},
        {"word": "world", "start": 2.0, "end": 3.0},
        # Leaks below the window.
        {"word": "before", "start": 0.5, "end": 1.1},
        # Leaks above the window.
        {"word": "after", "start": 4.9, "end": 5.2},
    ], want_chars=False)

    units = aligner.align_segment(
        waveform=object(),
        chunk_start=1.0,
        chunk_end=5.0,
        text="hello world",
        language="en",
    )
    assert units is not None
    texts = [u["text"] for u in units]
    assert texts == ["hello", "world"]
    for u in units:
        assert u["timestamp_start"] >= 1.0
        assert u["timestamp_end"] <= 5.0


def test_job_scope_releases_on_exception():
    """Context manager must release models even when the body raises.

    Regression guard: an asyncio CancelledError or a DB commit error
    inside the refine loop used to risk pinning ~1 GB of model memory
    if the ``release_job()`` call in ``finally`` was missed.
    """
    aligner.release_all()
    aligner._align_models["ja"] = (MagicMock(), MagicMock())

    class _Boom(RuntimeError):
        pass

    try:
        with aligner.job_scope():
            assert aligner._active_jobs == 1
            raise _Boom("simulated failure inside scope")
    except _Boom:
        pass

    assert aligner._active_jobs == 0
    assert aligner._align_models == {}


def test_model_cache_evicts_lru_when_over_cap(monkeypatch):
    """Seeding the model cache beyond ``_MODEL_CACHE_MAX`` evicts the
    least-recently-used language. Prevents pathological language
    diversity from pinning unbounded memory (~1 GB per model).
    """
    aligner.release_all()
    cap = aligner._MODEL_CACHE_MAX  # usually 2

    def _fake_load_align_model(language_code, device, model_dir):
        return (MagicMock(name=f"model-{language_code}"), MagicMock())

    fake_whisperx = MagicMock()
    fake_whisperx.load_align_model = _fake_load_align_model
    monkeypatch.setitem(__import__("sys").modules, "whisperx", fake_whisperx)

    # Load cap + 1 distinct languages.
    langs = [f"l{i}" for i in range(cap + 1)]
    for lang in langs:
        aligner._load_align_model(lang)

    assert len(aligner._align_models) == cap
    # The first-loaded lang must have been evicted.
    assert langs[0] not in aligner._align_models
    # The most recent ones must remain.
    for lang in langs[1:]:
        assert lang in aligner._align_models

    # Touching an older entry should mark it recent, so the next load
    # evicts the one that wasn't touched.
    aligner._load_align_model(langs[1])  # cache hit, moves to end
    aligner._load_align_model("fresh")

    assert "fresh" in aligner._align_models
    assert langs[1] in aligner._align_models
    # langs[-1] was the least-recently-used after the touch above.
    assert langs[-1] not in aligner._align_models
