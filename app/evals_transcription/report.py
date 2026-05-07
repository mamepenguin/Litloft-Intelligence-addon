"""Render the final markdown + JSON sidecar.

Phase 2C: this layer is intentionally simple — the runner produces
``CaseResult`` rows; the report formats them into:

* an aggregate table per provider (mean WER / CER / sa-WER, latency
  p50 / mean / max, language-detection accuracy),
* per-tier tables,
* a split-test comparison block when any ``no_split`` / ``split``
  rows are present,
* per-case detail tables.

Whisper-family providers get a ``†`` annotation so readers know the
GT they used may have been Whisper-curated and bias the row.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from app.evals_transcription.loader import Case
from app.evals_transcription.runner import CaseResult


_WHISPER_FAMILY = {"whisper_local", "openai_compatible"}
_WHISPER_FOOTNOTE = (
    "† Whisper-family provider. If the ground truth was curated from a "
    "Whisper output, this row's WER/CER will be optimistically biased."
)


def write_report(
    results: list[CaseResult],
    cases: list[Case],
    output_path: Path,
    baseline_path: Path | None,
) -> None:
    """Write the markdown report and a paired JSON sidecar.

    Sidecar lives alongside the markdown with ``.json`` suffix
    (``foo.md`` → ``foo.json``).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_path.with_suffix(".json")

    baseline_lookup = _load_baseline(baseline_path)

    md = _render_markdown(results, cases, baseline_lookup)
    output_path.write_text(md, encoding="utf-8")

    sidecar = _render_sidecar(results, cases)
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _render_markdown(
    results: list[CaseResult],
    cases: list[Case],
    baseline: dict[tuple[str, str, str], CaseResult] | None,
) -> str:
    lines: list[str] = []
    lines.append("# Transcription Provider Comparison")
    lines.append("")

    by_tier = {
        c.name: c.tier for c in cases
    }
    n_cases = len(cases)
    tier_counts: dict[str, int] = {}
    for c in cases:
        tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1
    tier_summary = " / ".join(
        f"{tier}:{tier_counts.get(tier, 0)}"
        for tier in ("short", "medium", "long")
    )
    lines.append(
        f"Cases: {n_cases} ({tier_summary}). "
        f"Total runs: {len(results)}."
    )

    skipped = [r for r in results if r.skipped]
    if skipped:
        lines.append("")
        lines.append("## Skipped")
        seen: set[tuple[str, str | None]] = set()
        for r in skipped:
            key = (r.provider_name, r.skipped_reason)
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- **{r.provider_name}**: {r.skipped_reason}"
            )

    default = [r for r in results if r.mode == "default" and not r.skipped]
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(_aggregate_table(default, cases, baseline))

    lines.append("")
    lines.append(f"_{_WHISPER_FOOTNOTE}_")

    for tier in ("short", "medium", "long"):
        rows = [
            r for r in default if by_tier.get(r.case_name) == tier
        ]
        if not rows:
            continue
        lines.append("")
        lines.append(f"### Tier: {tier}")
        lines.append("")
        lines.append(_aggregate_table(rows, cases, baseline))

    split_rows = [
        r for r in results
        if r.mode in ("no_split", "split") and not r.skipped
    ]
    if split_rows:
        lines.append("")
        lines.append("## Phase 2B split overhead")
        lines.append("")
        lines.append(_split_overhead_table(split_rows))

    lines.append("")
    lines.append("## Per-case detail")
    case_index = {c.name: c for c in cases}
    for case in cases:
        rows = [
            r for r in results
            if r.case_name == case.name and r.mode == "default"
            and not r.skipped
        ]
        if not rows:
            continue
        lines.append("")
        speaker_count = len({s.speaker_id for s in case.speakers}) or 1
        lines.append(
            f"### {case.name} — {case.duration_s:.1f}s, "
            f"{case.language}, {speaker_count} speaker(s)"
        )
        lines.append("")
        lines.append(_per_case_table(rows))

    return "\n".join(lines) + "\n"


def _aggregate_table(
    rows: list[CaseResult],
    cases: list[Case],
    baseline: dict[tuple[str, str, str], CaseResult] | None,
) -> str:
    by_provider: dict[str, list[CaseResult]] = {}
    for r in rows:
        by_provider.setdefault(r.provider_name, []).append(r)

    case_lang = {c.name: c.language for c in cases}

    header = (
        "| provider | N | WER | CER | latency p50 | latency mean | "
        "latency max | sa-WER | lang_acc |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    table = [header, sep]
    for name in sorted(by_provider):
        provider_rows = by_provider[name]
        wer_values = [r.wer for r in provider_rows if r.wer is not None]
        cer_values = [r.cer for r in provider_rows if r.cer is not None]
        latencies = [
            r.latency_s for r in provider_rows if r.latency_s is not None
        ]
        sa_values = [
            r.sa_wer for r in provider_rows if r.sa_wer is not None
        ]
        lang_match = sum(
            1
            for r in provider_rows
            if r.detected_language
            and r.detected_language.lower().startswith(
                case_lang.get(r.case_name, "").lower().split("-")[0]
            )
        )
        lang_total = sum(
            1 for r in provider_rows if r.detected_language
        )

        suffix = "†" if name in _WHISPER_FAMILY else ""
        table.append(
            "| {p}{s} | {n} | {wer} | {cer} | {p50} | {mean} | {mx} | "
            "{sa} | {lang} |".format(
                p=name,
                s=suffix,
                n=len(provider_rows),
                wer=_fmt(_mean(wer_values), suffix=""),
                cer=_fmt(_mean(cer_values), suffix=""),
                p50=_fmt_secs(_quantile(latencies, 0.5)),
                mean=_fmt_secs(_mean(latencies)),
                mx=_fmt_secs(max(latencies) if latencies else None),
                sa=_fmt(_mean(sa_values), suffix=""),
                lang=(
                    f"{lang_match}/{lang_total}"
                    if lang_total
                    else "--"
                ),
            )
        )
    return "\n".join(table)


def _split_overhead_table(rows: list[CaseResult]) -> str:
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r.case_name, r.provider_name), {})[r.mode] = r
    header = "| case | provider | mode | WER | CER | latency |"
    sep = "|---|---|---|---|---|---|"
    table = [header, sep]
    for (case_name, provider_name), modes in sorted(by_pair.items()):
        for mode in ("no_split", "split"):
            r = modes.get(mode)
            if r is None:
                continue
            table.append(
                "| {c} | {p} | {m} | {wer} | {cer} | {lat} |".format(
                    c=case_name,
                    p=provider_name,
                    m=mode,
                    wer=_fmt(r.wer, suffix=""),
                    cer=_fmt(r.cer, suffix=""),
                    lat=_fmt_secs(r.latency_s),
                )
            )
    return "\n".join(table)


def _per_case_table(rows: list[CaseResult]) -> str:
    header = "| provider | WER | CER | sa-WER | latency | detected | error |"
    sep = "|---|---|---|---|---|---|---|"
    table = [header, sep]
    for r in sorted(rows, key=lambda x: x.provider_name):
        table.append(
            "| {p} | {wer} | {cer} | {sa} | {lat} | {lang} | {err} |".format(
                p=r.provider_name,
                wer=_fmt(r.wer, suffix=""),
                cer=_fmt(r.cer, suffix=""),
                sa=_fmt(r.sa_wer, suffix=""),
                lat=_fmt_secs(r.latency_s),
                lang=r.detected_language or "--",
                err=r.error or "ok",
            )
        )
    return "\n".join(table)


# ---------------------------------------------------------------------------
# JSON sidecar
# ---------------------------------------------------------------------------


def _render_sidecar(
    results: list[CaseResult],
    cases: list[Case],
) -> dict:
    case_index = {c.name: c for c in cases}
    rows = []
    for r in results:
        case = case_index.get(r.case_name)
        ref = case.reference_transcript if case else ""
        rows.append({
            **asdict(r),
            "ref_hash": _sha256(ref),
            "hyp_hash": _sha256(r.hypothesis or ""),
        })
    return {
        "schema_version": 1,
        "n_cases": len(cases),
        "n_results": len(results),
        "cases": [asdict_case(c) for c in cases],
        "results": rows,
    }


def asdict_case(case: Case) -> dict:
    return {
        "name": case.name,
        "case_path": case.case_path,
        "audio_path": case.audio_path,
        "language": case.language,
        "duration_s": case.duration_s,
        "tier": case.tier,
        "speakers": [
            {"speaker_id": s.speaker_id, "start": s.start, "end": s.end}
            for s in case.speakers
        ],
        "split_test": (
            {
                "forced_cap_bytes": case.split_test.forced_cap_bytes,
                "providers": list(case.split_test.providers),
            }
            if case.split_test
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_baseline(
    path: Path | None,
) -> dict[tuple[str, str, str], CaseResult] | None:
    """Reserved for future use: parse a previous sidecar and return a
    lookup keyed by (case_name, provider_name, mode). Returns None
    when ``path`` is None or unreadable. Currently the markdown does
    not yet emit a delta column; the lookup is built so callers can
    wire it in without touching the sidecar schema."""
    if path is None:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return None
    out: dict[tuple[str, str, str], CaseResult] = {}
    for raw in data.get("results", []):
        try:
            out[
                (raw["case_name"], raw["provider_name"], raw["mode"])
            ] = CaseResult(
                case_name=raw["case_name"],
                provider_name=raw["provider_name"],
                mode=raw["mode"],
                wer=raw.get("wer"),
                cer=raw.get("cer"),
                sa_wer=raw.get("sa_wer"),
                latency_s=raw.get("latency_s"),
                detected_language=raw.get("detected_language"),
                error=raw.get("error"),
                skipped=raw.get("skipped", False),
                skipped_reason=raw.get("skipped_reason"),
            )
        except (KeyError, TypeError):
            continue
    return out


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_v = sorted(values)
    pos = q * (len(sorted_v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = pos - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


def _fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "--"
    return f"{value:.3f}{suffix}"


def _fmt_secs(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}s"
