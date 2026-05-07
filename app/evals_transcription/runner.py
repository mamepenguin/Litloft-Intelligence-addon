"""Iterate cases × providers and collect comparison results.

For each case + each available provider, the runner runs the default
transcribe path (with whatever wrapping ``get_provider`` applies) and
optionally a split / no-split pair when the case carries a
``split_test`` block. Per-case failures are caught so one bad YAML or
one provider error does not abort the entire bulk run.

Provider availability is checked lazily by *trying* to construct the
provider — every Phase 1 / 2A provider raises ``FatalError`` from
``__init__`` when its API key env is missing, so a single try/except
serves as the skip gate.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Iterable

from app.evals_transcription.loader import (
    Case,
    SplitTest,
)
from app.evals_transcription.metrics import (
    score_speaker_attributed_wer,
    score_text,
)
from app.workers.transcription import (
    FatalError,
    ProviderCapabilities,
    TranscriptionError,
    TranscriptionProvider,
    TranscriptionSegment,
    build_inner_provider,
    get_provider,
)
from app.workers.transcription.splitting_transcriber import (
    SplittingTranscriber,
)

logger = logging.getLogger(__name__)

ALL_PROVIDERS: tuple[str, ...] = (
    "whisper_local",
    "openai_compatible",
    "deepgram",
    "elevenlabs_scribe",
    "assemblyai",
    "gemini",
)


@dataclass(frozen=True)
class CaseResult:
    """One row in the report's per-case section."""

    case_name: str
    provider_name: str
    mode: str  # "default" | "split" | "no_split"
    wer: float | None
    cer: float | None
    sa_wer: float | None
    latency_s: float | None
    detected_language: str | None
    error: str | None = None
    skipped: bool = False
    skipped_reason: str | None = None
    # Sidecar-only fields (CaseResult is the table-row projection;
    # report.py reads these to populate the JSON sidecar).
    ref_normalized: str = ""
    hyp_normalized: str = ""
    hypothesis: str = ""


@dataclass(frozen=True)
class _ProviderInstance:
    """Result of resolving a provider name into a usable instance.

    ``provider`` is None when the provider was unavailable; the
    runner emits a single skipped CaseResult per case in that case.
    """

    name: str
    provider: TranscriptionProvider | None
    skipped_reason: str | None = None


class _CapsOverride:
    """Wrap an inner provider so its ``capabilities`` reflect a
    runtime override.

    ``capabilities`` on each provider is a class attribute (frozen
    dataclass). To force a different ``max_input_bytes`` for one
    eval invocation we wrap the inner provider in this proxy whose
    instance attribute shadows the class attribute. The wrapper
    forwards ``transcribe`` directly so SplittingTranscriber sees
    the overridden caps when it inspects them.
    """

    def __init__(
        self,
        inner: TranscriptionProvider,
        capabilities: ProviderCapabilities,
    ) -> None:
        self._inner = inner
        self.capabilities = capabilities

    @property
    def name(self) -> str:
        return self._inner.name

    async def transcribe(self, *args, **kwargs):
        return await self._inner.transcribe(*args, **kwargs)


def resolve_providers(requested: list[str] | None) -> list[_ProviderInstance]:
    """Build a provider instance per requested name.

    ``requested=None`` resolves all known providers. Unknown names
    in ``requested`` raise ``ValueError`` so the CLI can fail loud
    with the available list (the dispatch loop must not silently
    skip a typo).
    """
    if requested is None:
        names = list(ALL_PROVIDERS)
    else:
        unknown = [n for n in requested if n not in ALL_PROVIDERS]
        if unknown:
            raise ValueError(
                f"Unknown providers {unknown!r}. "
                f"Available: {list(ALL_PROVIDERS)!r}"
            )
        names = list(requested)

    out: list[_ProviderInstance] = []
    for name in names:
        try:
            provider = get_provider(name)
        except (ValueError, FatalError) as exc:
            out.append(
                _ProviderInstance(
                    name=name,
                    provider=None,
                    skipped_reason=str(exc),
                )
            )
            continue
        out.append(
            _ProviderInstance(
                name=name, provider=provider, skipped_reason=None
            )
        )
    return out


async def run_eval(
    cases: Iterable[Case],
    providers: list[_ProviderInstance],
) -> list[CaseResult]:
    """Run every case × provider × mode combination and collect results.

    Per-case failures are recorded as a CaseResult with ``error`` set;
    the loop never aborts. Split-test runs run after the default mode
    so the report can show ``no_split`` / ``split`` deltas alongside
    the default row.
    """
    cases = list(cases)
    forced_run_count = sum(
        1 for c in cases if c.split_test is not None
        for _ in (c.split_test.providers if c.split_test else ())
    )
    if forced_run_count:
        logger.warning(
            "Phase 2C eval: %s split-test runs scheduled "
            "(each runs a case 2x against the same provider — "
            "double the API spend). Set split_test on at most one "
            "long case per run if cost is a concern.",
            forced_run_count,
        )

    results: list[CaseResult] = []
    for case in cases:
        for provider_instance in providers:
            results.extend(
                await _run_case_provider(case, provider_instance)
            )
    return results


async def _run_case_provider(
    case: Case,
    pi: _ProviderInstance,
) -> list[CaseResult]:
    if pi.provider is None:
        return [
            CaseResult(
                case_name=case.name,
                provider_name=pi.name,
                mode="default",
                wer=None,
                cer=None,
                sa_wer=None,
                latency_s=None,
                detected_language=None,
                skipped=True,
                skipped_reason=pi.skipped_reason,
            )
        ]

    out: list[CaseResult] = []
    out.append(
        await _safe_run(
            case=case,
            provider=pi.provider,
            mode="default",
        )
    )

    split = case.split_test
    if split is not None and pi.name in split.providers:
        out.extend(
            await _split_test_runs(
                case=case,
                provider_name=pi.name,
                split=split,
            )
        )
    return out


async def _split_test_runs(
    case: Case,
    provider_name: str,
    split: SplitTest,
) -> list[CaseResult]:
    """Build no_split and split variants of the same provider.

    Constructed inner providers go through ``build_inner_provider`` so
    the factory's auto-wrap is bypassed; we rewrap manually with
    ``SplittingTranscriber`` for the split run.
    """
    try:
        inner = build_inner_provider(provider_name)
    except (ValueError, FatalError) as exc:
        # Should not happen because the default-mode run succeeded,
        # but be defensive.
        return [
            CaseResult(
                case_name=case.name,
                provider_name=provider_name,
                mode="split_test_failed",
                wer=None,
                cer=None,
                sa_wer=None,
                latency_s=None,
                detected_language=None,
                error=type(exc).__name__,
            )
        ]
    inner_cap = inner.capabilities.max_input_bytes
    if inner_cap is not None and split.forced_cap_bytes >= inner_cap:
        logger.warning(
            "split_test for %s on case %s: forced_cap_bytes %s >= "
            "inner cap %s; split would not fire, skipping split run",
            provider_name, case.name, split.forced_cap_bytes, inner_cap,
        )
        return []

    no_split_caps = replace(inner.capabilities, max_input_bytes=None)
    no_split_provider = _CapsOverride(inner, no_split_caps)

    split_caps = replace(
        inner.capabilities, max_input_bytes=split.forced_cap_bytes
    )
    split_provider = SplittingTranscriber(
        _CapsOverride(inner, split_caps)
    )

    return [
        await _safe_run(case=case, provider=no_split_provider, mode="no_split"),
        await _safe_run(case=case, provider=split_provider, mode="split"),
    ]


async def _safe_run(
    case: Case,
    provider,
    mode: str,
) -> CaseResult:
    """Run one transcribe + score, catching every failure mode."""
    try:
        start = time.monotonic()
        segments: list[TranscriptionSegment] = await provider.transcribe(
            case.audio_path,
            language_hint=case.language,
            hotwords=None,
        )
        latency = time.monotonic() - start
    except TranscriptionError as exc:
        return CaseResult(
            case_name=case.name,
            provider_name=provider.name,
            mode=mode,
            wer=None,
            cer=None,
            sa_wer=None,
            latency_s=None,
            detected_language=None,
            error=type(exc).__name__,
        )
    except (ValueError, OSError) as exc:
        # Ground-truth / language / file errors. Surface so the user
        # can fix the case YAML; do not crash the bulk run.
        return CaseResult(
            case_name=case.name,
            provider_name=provider.name,
            mode=mode,
            wer=None,
            cer=None,
            sa_wer=None,
            latency_s=None,
            detected_language=None,
            error=type(exc).__name__,
        )

    hypothesis = " ".join(s.text for s in segments).strip()
    detected = segments[0].language if segments else None

    try:
        wer, cer = score_text(
            case.reference_transcript, hypothesis, case.language
        )
    except (ValueError, TypeError) as exc:
        return CaseResult(
            case_name=case.name,
            provider_name=provider.name,
            mode=mode,
            wer=None,
            cer=None,
            sa_wer=None,
            latency_s=latency,
            detected_language=detected,
            error=f"ScoringError: {exc}",
            hypothesis=hypothesis,
        )

    flat_words = [w for s in segments for w in s.words]
    sa_wer = (
        score_speaker_attributed_wer(list(case.speakers), flat_words)
        if case.speakers
        else None
    )

    return CaseResult(
        case_name=case.name,
        provider_name=provider.name,
        mode=mode,
        wer=wer,
        cer=cer,
        sa_wer=sa_wer,
        latency_s=latency,
        detected_language=detected,
        hypothesis=hypothesis,
    )
