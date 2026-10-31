"""SPEC-ADDON-039: the ``haas_local`` transcription provider.

The recognizer, model fetch, VAD and ``ffmpeg`` are stand-ins (see
``haas_fakes``); the provider's own logic runs for real.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import pytest

from app.workers.transcription import (
    FatalError,
    ProviderCapabilities,
    TranscriptionSegment,
    TransientError,
)
from tests.haas_fakes import (
    COMMIT,
    MODEL_FILES,
    REPO,
    SAMPLE_RATE,
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    haas,  # noqa: F401  (fixture)
    result,
)

pytestmark = pytest.mark.asyncio


async def _run(provider, haas, **kw):
    return await provider.transcribe(str(haas.audio_file), **kw)


# --- normal flow (1) -------------------------------------------------------


async def test_spec_addon_039_capabilities(haas):
    assert haas.inner().capabilities == ProviderCapabilities(
        sends_audio_offhost=False,
        supports_diarization=False,
        supports_hotwords=False,
        supports_word_timestamps=True,
        max_input_bytes=50 * 1024 * 1024,
        accepts_initial_prompt=False,
        handles_own_retry=False,
    )
    assert haas.inner().name == "haas_local"


async def test_spec_addon_039_factory_wraps_in_splitting_transcriber(haas):
    from app.workers.transcription.splitting_transcriber import (
        SplittingTranscriber,
    )

    provider = haas.wrapped()
    assert isinstance(provider, SplittingTranscriber)
    assert provider._inner.name == "haas_local"
    assert provider._cap_bytes == 50 * 1024 * 1024


async def test_spec_addon_039_segment_shape_and_word_text(haas):
    haas.spans = [(2.0, 5.0)]
    haas.results = [
        result(
            [" こ", "ん", "に", "ち", "は"],
            [0.0, 0.2, 0.4, 0.6, 0.8],
            [0.2] * 5,
        )
    ]
    segs = await _run(haas.inner(), haas)

    assert len(segs) == 1
    seg = segs[0]
    assert isinstance(seg, TranscriptionSegment)
    assert seg.language == "ja"
    assert seg.text == "こんにちは"
    assert [w.text for w in seg.words] == ["こ", "ん", "に", "ち", "は"]
    assert all(w.speaker_id is None for w in seg.words)
    assert seg.start == pytest.approx(2.0)
    assert seg.end == pytest.approx(3.0)
    assert seg.words[0].start == pytest.approx(2.0)
    assert seg.words[-1].end == pytest.approx(3.0)


async def test_spec_addon_039_word_times_are_file_offsets(haas):
    haas.spans = [(2.0, 5.0), (7.0, 9.0)]
    haas.results = [
        result(["あ", "い"], [0.0, 0.5], [0.5, 0.5]),
        result(["う"], [0.25], [0.5]),
    ]
    segs = await _run(haas.inner(), haas)

    assert [s.words[0].start for s in segs] == [
        pytest.approx(2.0),
        pytest.approx(7.25),
    ]
    assert segs[1].end == pytest.approx(7.75)


async def test_spec_addon_039_each_span_decoded_separately_in_order(haas):
    haas.spans = [(1.0, 3.0), (4.0, 4.5), (6.0, 9.0)]
    haas.results = [
        result(["あ"], [0.0], [0.1]),
        result(["い"], [0.0], [0.1]),
        result(["う"], [0.0], [0.1]),
    ]
    segs = await _run(haas.inner(), haas)

    assert [c["n"] for c in haas.decode_calls] == [
        2 * SAMPLE_RATE,
        SAMPLE_RATE // 2,
        3 * SAMPLE_RATE,
    ]
    assert all(c["rate"] == SAMPLE_RATE for c in haas.decode_calls)
    assert [s.text for s in segs] == ["あ", "い", "う"]


async def test_spec_addon_039_ffmpeg_decodes_to_16k_mono_f32(haas):
    haas.set_audio_seconds(10)
    await _run(haas.inner(), haas)

    (args,) = haas.ffmpeg_args()
    assert "-vn" in args
    assert args[args.index("-ac") + 1] == "1"
    assert args[args.index("-ar") + 1] == "16000"
    assert args[args.index("-f") + 1] == "f32le"
    assert str(haas.audio_file) in args
    (vad,) = haas.vad_calls
    assert vad["dtype"] == "float32"
    assert vad["n"] == 10 * SAMPLE_RATE


async def test_spec_addon_039_vad_options(haas):
    haas.configure()
    await _run(haas.inner(), haas)

    assert haas.vad_calls[0]["options"] == {
        "max_speech_duration_s": 45,
        "min_silence_duration_ms": 300,
        "speech_pad_ms": 200,
    }


async def test_spec_addon_039_i3_configured_values_reach_vad_and_recognizer(haas):
    haas.configure(num_threads=2, max_segment_seconds=30)
    await _run(haas.inner(), haas)

    assert haas.vad_calls[0]["options"]["max_speech_duration_s"] == 30
    assert haas.builds[0]["num_threads"] == 2


async def test_spec_addon_039_i3_defaults_reach_recognizer(haas):
    await _run(haas.inner(), haas)

    (build,) = haas.builds
    assert build["num_threads"] == 4
    assert build["model_type"] == "nemo_transducer"
    assert build["feature_dim"] == 128


async def test_spec_addon_039_recognizer_built_from_model_files_only(haas):
    await _run(haas.inner(), haas)

    (build,) = haas.builds
    assert set(build) == {
        "encoder",
        "decoder",
        "joiner",
        "tokens",
        "num_threads",
        "model_type",
        "feature_dim",
    }
    d = str(haas.model_dir)
    assert build["encoder"] == f"{d}/encoder.quantized.onnx"
    assert build["decoder"] == f"{d}/decoder.onnx"
    assert build["joiner"] == f"{d}/joiner.onnx"
    assert build["tokens"] == f"{d}/tokens.txt"


async def test_spec_addon_039_hints_and_prompt_are_ignored(haas):
    segs = await _run(
        haas.inner(),
        haas,
        language_hint="en",
        hotwords=["x"],
        initial_prompt="prev",
    )
    assert [s.language for s in segs] == ["ja"]


# --- word reconstruction (1.7, I1, I2, I4) ---------------------------------


@pytest.mark.parametrize(
    "tokens,timestamps,durations,words",
    [
        pytest.param(
            [" ", "  あ", "い "],
            [0.0, 0.1, 0.2],
            [0.1, 0.1, 0.1],
            ["あ", "い"],
            id="whitespace-stripped-empty-token-dropped",
        ),
        pytest.param(
            ["h", "e", "l", "l", "o"],
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.1] * 5,
            ["h", "e", "l", "l", "o"],
            id="latin-one-token-per-letter",
        ),
        pytest.param(
            ["<0xE7>", "<0x84>", "<0x9A>", "あ"],
            [0.0, 0.1, 0.2, 0.3],
            [0.1] * 4,
            ["焚", "あ"],
            id="byte-run-becomes-one-character",
        ),
        pytest.param(
            ["<0xE7>", "<0x84>", "<0x9A>", "い", "<0xE6>", "<0xA2>", "<0xB1>"],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.1] * 7,
            ["焚", "い", "梱"],
            id="two-byte-runs-split-by-a-normal-token",
        ),
        pytest.param(
            ["<0xFF>", "あ"],
            [0.0, 0.1],
            [0.1, 0.1],
            ["あ"],
            id="invalid-bytes-dropped",
        ),
        pytest.param(
            ["う", "<0xE7>", "<0x84>"],
            [0.0, 0.1, 0.2],
            [0.1] * 3,
            ["う"],
            id="truncated-utf8-run-leaves-nothing",
        ),
    ],
)
async def test_spec_addon_039_i1_word_texts(
    haas, tokens, timestamps, durations, words
):
    haas.spans = [(0.0, 3.0)]
    haas.results = [result(tokens, timestamps, durations)]
    (seg,) = await _run(haas.inner(), haas)

    assert [w.text for w in seg.words] == words
    assert seg.text == "".join(words)
    assert "<0x" not in seg.text and "�" not in seg.text
    assert all("<0x" not in w.text and "�" not in w.text for w in seg.words)


async def test_spec_addon_039_byte_run_spans_first_start_to_last_end(haas):
    haas.spans = [(1.0, 4.0)]
    haas.results = [
        result(
            ["<0xE7>", "<0x84>", "<0x9A>"],
            [0.0, 0.1, 0.2],
            [0.1, 0.1, 0.3],
        )
    ]
    (seg,) = await _run(haas.inner(), haas)

    (w,) = seg.words
    assert w.text == "焚"
    assert w.start == pytest.approx(1.0)
    assert w.end == pytest.approx(1.5)


async def test_spec_addon_039_missing_durations_use_next_start_and_span_end(haas):
    haas.spans = [(1.0, 4.0)]
    haas.results = [result(["あ", "い", "う"], [0.0, 0.5, 1.0], [])]
    (seg,) = await _run(haas.inner(), haas)

    assert [w.start for w in seg.words] == [
        pytest.approx(1.0),
        pytest.approx(1.5),
        pytest.approx(2.0),
    ]
    assert [w.end for w in seg.words] == [
        pytest.approx(1.5),
        pytest.approx(2.0),
        pytest.approx(4.0),
    ]


async def test_spec_addon_039_overlapping_durations_clipped_to_next_start(haas):
    haas.spans = [(0.0, 2.0)]
    haas.results = [result(["あ", "い"], [0.0, 0.5], [1.0, 1.0])]
    (seg,) = await _run(haas.inner(), haas)

    assert seg.words[0].end == pytest.approx(0.5)
    assert seg.words[1].start == pytest.approx(0.5)
    assert seg.words[1].end == pytest.approx(1.5)


async def test_spec_addon_039_duration_past_span_end_is_clipped(haas):
    haas.spans = [(1.0, 2.0)]
    haas.results = [result(["あ"], [0.5], [5.0])]
    (seg,) = await _run(haas.inner(), haas)

    assert seg.words[0].end == pytest.approx(2.0)
    assert seg.end == pytest.approx(2.0)


@pytest.mark.parametrize(
    "tokens,timestamps,durations",
    [
        (["あ", "い", "う"], [0.0, 0.2, 0.1], [0.3, 0.3, 0.3]),
        (["あ", "い", "う"], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
        (["あ", "い", "う"], [0.4, 0.2, 0.0], []),
        (["あ", "い"], [0.0, 0.5], [0.0, 0.0]),
    ],
    ids=["unordered", "all-equal", "descending-no-durations", "zero-durations"],
)
async def test_spec_addon_039_i2_words_stay_ordered_inside_span(
    haas, tokens, timestamps, durations
):
    span_start, span_end = 3.0, 6.0
    haas.spans = [(span_start, span_end)]
    haas.results = [result(tokens, timestamps, durations)]
    segs = await _run(haas.inner(), haas)

    words = [w for s in segs for w in s.words]
    assert words
    for w in words:
        assert span_start <= w.start <= w.end <= span_end
    for a, b in zip(words, words[1:]):
        assert a.end <= b.start


@pytest.mark.parametrize(
    "tokens",
    [[], [" "], ["", "  "], ["<0xFF>"], ["<0xFF>", " "]],
    ids=["none", "blank", "empty", "invalid-bytes", "invalid-and-blank"],
)
async def test_spec_addon_039_i4_span_without_words_yields_no_segment(haas, tokens):
    haas.spans = [(1.0, 2.0), (3.0, 4.0)]
    haas.results = [
        result(tokens, [0.0] * len(tokens), [0.1] * len(tokens)),
        result(["あ"], [0.0], [0.1]),
    ]
    segs = await _run(haas.inner(), haas)

    assert [s.text for s in segs] == ["あ"]
    assert segs[0].start == pytest.approx(3.0)
    assert all(s.words for s in segs)


async def test_spec_addon_039_no_span_yields_words_returns_empty_list(haas):
    haas.results = [result([], [])]
    assert await _run(haas.inner(), haas) == []


# --- no speech / bad input: empty list, model untouched (I4) ---------------


def _model_untouched(haas):
    assert haas.hf_calls == []
    assert haas.builds == []
    assert haas.decode_calls == []


async def test_spec_addon_039_i4_no_spans_never_fetches_or_loads(haas):
    haas.spans = []
    assert await _run(haas.inner(), haas) == []
    _model_untouched(haas)


async def test_spec_addon_039_i4_vad_index_error_is_empty(haas):
    haas.vad_exc = IndexError("list index out of range")
    assert await _run(haas.inner(), haas) == []
    _model_untouched(haas)


async def test_spec_addon_039_i4_ffmpeg_positive_exit_is_empty_with_warning(
    haas, caplog
):
    haas.ffmpeg_mode("exit1")
    with caplog.at_level(logging.WARNING):
        assert await _run(haas.inner(), haas) == []

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert haas.vad_calls == []
    _model_untouched(haas)


async def test_spec_addon_039_ffmpeg_signal_death_is_transient(haas):
    haas.ffmpeg_mode("killed")
    with pytest.raises(TransientError):
        await _run(haas.inner(), haas)
    _model_untouched(haas)


async def test_spec_addon_039_missing_ffmpeg_is_fatal(haas):
    haas.no_ffmpeg()
    with pytest.raises(FatalError):
        await _run(haas.inner(), haas)
    _model_untouched(haas)


async def test_spec_addon_039_i7_other_vad_exception_is_fatal(haas):
    haas.vad_exc = ValueError("boom")
    with pytest.raises(FatalError):
        await _run(haas.inner(), haas)
    _model_untouched(haas)


# --- model fetch (6, I7) ---------------------------------------------------


async def test_spec_addon_039_i7_fetch_names_pinned_commit_and_cache_dir(haas):
    await _run(haas.inner(), haas)

    first, second = haas.hf_calls
    assert first["local_files_only"] is True
    assert second["local_files_only"] is False
    for call in haas.hf_calls:
        assert call["repo_id"] == REPO
        assert call["revision"] == COMMIT
        assert str(call["cache_dir"]) == str(haas.model_cache_dir)
        assert set(call["allow_patterns"]) == set(MODEL_FILES)


async def test_spec_addon_039_i7_complete_cache_makes_no_network_call(haas):
    haas.precache()
    await _run(haas.inner(), haas)

    assert [c["local_files_only"] for c in haas.hf_calls] == [True]
    assert len(haas.builds) == 1


async def test_spec_addon_039_i7_cache_missing_a_file_goes_online(haas):
    haas.precache([f for f in MODEL_FILES if f != "NOTICE"])
    await _run(haas.inner(), haas)

    assert [c["local_files_only"] for c in haas.hf_calls] == [True, False]
    assert (haas.model_dir / "NOTICE").exists()


async def test_spec_addon_039_i7_second_job_after_fetch_needs_no_network(haas):
    provider = haas.inner()
    await _run(provider, haas)
    online_before = [c for c in haas.hf_calls if not c["local_files_only"]]
    await _run(provider, haas)

    assert len(online_before) == 1
    assert len([c for c in haas.hf_calls if not c["local_files_only"]]) == 1


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("down"),
        OSError(28, "No space left on device"),
        HfHubHTTPError("503"),
        TimeoutError("read timeout"),
    ],
    ids=["connection", "disk-full", "http-5xx", "timeout"],
)
async def test_spec_addon_039_i7_fetch_failure_is_transient(haas, error):
    haas.net_error = error
    with pytest.raises(TransientError) as exc:
        await _run(haas.inner(), haas)

    assert REPO in str(exc.value)
    assert haas.builds == []


async def test_spec_addon_039_i7_partial_snapshot_without_network_is_transient(haas):
    haas.precache([f for f in MODEL_FILES if f != "tokens.txt"])
    haas.net_error = ConnectionError("down")
    with pytest.raises(TransientError):
        await _run(haas.inner(), haas)


@pytest.mark.parametrize(
    "error",
    [
        RepositoryNotFoundError("gone"),
        RevisionNotFoundError("gone"),
        EntryNotFoundError("gone"),
    ],
    ids=["repository", "commit", "file"],
)
async def test_spec_addon_039_i7_vanished_remote_is_fatal(haas, error):
    haas.net_error = error
    with pytest.raises(FatalError) as exc:
        await _run(haas.inner(), haas)

    assert REPO in str(exc.value)


async def test_spec_addon_039_fetch_failure_then_next_attempt_resumes(haas):
    provider = haas.inner()
    haas.net_error = ConnectionError("down")
    with pytest.raises(TransientError):
        await _run(provider, haas)
    haas.net_error = None
    segs = await _run(provider, haas)

    assert len(segs) == 1


async def test_spec_addon_039_transient_fetch_failure_makes_three_attempts(haas):
    haas.net_error = ConnectionError("down")

    async def instant(_delay, *a, **k):
        return None

    with pytest.MonkeyPatch.context() as m:
        m.setattr(asyncio, "sleep", instant)
        with pytest.raises(TransientError):
            await haas.wrapped().transcribe(str(haas.audio_file))

    assert len([c for c in haas.hf_calls if not c["local_files_only"]]) == 3


async def test_spec_addon_039_fatal_fetch_failure_is_not_retried(haas):
    haas.net_error = RepositoryNotFoundError("gone")
    with pytest.raises(FatalError):
        await haas.wrapped().transcribe(str(haas.audio_file))

    assert len([c for c in haas.hf_calls if not c["local_files_only"]]) == 1


# --- load failure (2, 3) ---------------------------------------------------


async def test_spec_addon_039_i7_sherpa_import_failure_is_fatal(haas):
    import sys

    haas.mp.setitem(sys.modules, "sherpa_onnx", None)
    with pytest.raises(FatalError):
        await _run(haas.inner(), haas)


async def test_spec_addon_039_i7_load_failure_is_fatal_names_repo_and_dir(haas):
    haas.build_exc = RuntimeError("bad onnx")
    with pytest.raises(FatalError) as exc:
        await _run(haas.inner(), haas)

    assert REPO in str(exc.value)
    assert str(haas.model_dir) in str(exc.value)


async def test_spec_addon_039_failed_load_is_retried_by_next_job(haas):
    provider = haas.inner()
    haas.build_exc = RuntimeError("bad onnx")
    with pytest.raises(FatalError):
        await _run(provider, haas)
    haas.build_exc = None
    segs = await _run(provider, haas)

    assert len(haas.builds) == 2
    assert len(segs) == 1


async def test_spec_addon_039_load_failure_is_not_retried_in_wrapper(haas):
    haas.build_exc = RuntimeError("bad onnx")
    with pytest.raises(FatalError):
        await haas.wrapped().transcribe(str(haas.audio_file))

    assert len(haas.builds) == 1


# --- recognizer lifecycle (3, I8) ------------------------------------------


async def test_spec_addon_039_i8_recognizer_is_built_once_across_jobs(haas):
    provider = haas.inner()
    await _run(provider, haas)
    await _run(provider, haas)

    assert len(haas.builds) == 1


async def test_spec_addon_039_i8_recognizer_shared_by_separate_provider_objects(haas):
    from app.workers.transcription import build_inner_provider

    first = haas.inner()
    second = build_inner_provider("haas_local")
    await _run(first, haas)
    await _run(second, haas)

    assert len(haas.builds) == 1


async def test_spec_addon_039_i8_concurrent_jobs_never_overlap_native_work(haas):
    haas.spans = [(0.0, 1.0), (2.0, 3.0)]

    def slow(_n):
        time.sleep(0.05)

    haas.decode_hook = slow
    provider = haas.inner()
    await asyncio.gather(*(_run(provider, haas) for _ in range(3)))

    assert len(haas.builds) == 1
    assert haas.max_active == 1


async def test_spec_addon_039_i8_blocking_work_runs_off_the_event_loop(haas):
    loop_thread = threading.get_ident()
    await _run(haas.inner(), haas)

    for op in ("vad", "fetch", "build", "decode"):
        assert haas.threads[op], op
        assert loop_thread not in haas.threads[op], op


async def test_spec_addon_039_i8_event_loop_stays_responsive(haas):
    haas.ffmpeg_delay(0.3)

    def slow(_n):
        time.sleep(0.3)

    haas.decode_hook = slow
    stop = asyncio.Event()
    stamps: list[float] = []

    async def ticker():
        while not stop.is_set():
            stamps.append(time.perf_counter())
            await asyncio.sleep(0.01)

    task = asyncio.create_task(ticker())
    await _run(haas.inner(), haas)
    stop.set()
    await task

    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert len(stamps) > 20
    assert max(gaps) < 0.15


async def test_spec_addon_039_i8_loop_responsive_while_a_job_waits_on_the_lock(haas):
    gate = threading.Event()
    entered = threading.Event()

    def hold(_n):
        entered.set()
        gate.wait(10)

    haas.decode_hook = hold
    provider = haas.inner()
    first = asyncio.create_task(_run(provider, haas))
    while not entered.is_set():
        await asyncio.sleep(0.01)
    second = asyncio.create_task(_run(provider, haas))
    started = time.perf_counter()
    for _ in range(10):
        await asyncio.sleep(0.02)
    elapsed = time.perf_counter() - started
    gate.set()
    haas.decode_hook = None
    await asyncio.gather(first, second)

    assert elapsed < 0.6
    assert len(haas.builds) == 1


# --- failures during decode (2, I7, I9) ------------------------------------


async def test_spec_addon_039_i7_recognizer_exception_is_fatal_and_keeps_nothing(haas):
    haas.spans = [(0.0, 1.0), (2.0, 3.0)]
    haas.decode_exc = RuntimeError("native failure")
    haas.decode_exc_on_call = 2
    with pytest.raises(FatalError):
        await _run(haas.inner(), haas)


async def test_spec_addon_039_i9_recognizer_exception_releases_recognizer(haas):
    provider = haas.inner()
    haas.decode_exc = RuntimeError("native failure")
    with pytest.raises(FatalError):
        await _run(provider, haas)
    haas.decode_exc = None
    segs = await _run(provider, haas)

    assert len(haas.builds) == 2
    assert len(segs) == 1


# --- idle unload (3, I9) ---------------------------------------------------


async def test_spec_addon_039_i9_idle_recognizer_is_released_and_reloaded(haas):
    haas.configure(idle_unload=1)
    provider = haas.inner()
    await _run(provider, haas)
    indexer = haas.indexer()

    await asyncio.sleep(1.3)
    await haas.idle_pass(indexer)
    segs = await _run(provider, haas)

    assert len(haas.builds) == 2
    assert len(segs) == 1


async def test_spec_addon_039_i9_recognizer_kept_within_timeout(haas):
    haas.configure(idle_unload=60)
    provider = haas.inner()
    await _run(provider, haas)
    indexer = haas.indexer()

    await asyncio.sleep(0.2)
    await haas.idle_pass(indexer)
    await _run(provider, haas)

    assert len(haas.builds) == 1


@pytest.mark.parametrize("timeout", [0, -1])
async def test_spec_addon_039_i9_non_positive_timeout_never_releases(haas, timeout):
    haas.configure(idle_unload=timeout)
    provider = haas.inner()
    await _run(provider, haas)
    indexer = haas.indexer()

    await asyncio.sleep(0.3)
    await haas.idle_pass(indexer)
    await _run(provider, haas)

    assert len(haas.builds) == 1


async def test_spec_addon_039_i9_clock_restarts_at_end_of_span_decode(haas):
    haas.configure(idle_unload=1)
    haas.decode_hook = lambda _n: time.sleep(1.3)
    provider = haas.inner()
    await _run(provider, haas)
    haas.decode_hook = None
    indexer = haas.indexer()

    await haas.idle_pass(indexer)
    await _run(provider, haas)

    assert len(haas.builds) == 1


async def test_spec_addon_039_i9_not_released_during_a_call_and_pass_does_not_block(
    haas,
):
    haas.configure(idle_unload=1)
    gate = threading.Event()
    entered = threading.Event()

    def hold(n):
        if n == 1:
            entered.set()
            gate.wait(10)

    haas.decode_hook = hold
    provider = haas.inner()
    indexer = haas.indexer()
    running = asyncio.create_task(_run(provider, haas))
    waiting = asyncio.create_task(_run(provider, haas))
    while not entered.is_set():
        await asyncio.sleep(0.01)

    await asyncio.sleep(1.3)
    pass_seconds = await haas.idle_pass(indexer)
    gate.set()
    await asyncio.gather(running, waiting)

    assert pass_seconds < 0.5
    assert len(haas.builds) == 1
