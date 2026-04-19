"""Markdown + JSON sidecar output for citation eval runs.

The markdown is human-scanned ("did table rows improve?"); the JSON
sidecar is the machine-diffable form used by ``--baseline`` compare.
Both share the same aggregate fields so downstream tooling can read
either.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.evals_citations.runner import AggregateMetrics, CaseReport


def render_markdown(
    reports: list[CaseReport],
    aggregate: AggregateMetrics,
    label: str = "",
    generated_at: str | None = None,
) -> str:
    """Render a human-readable markdown report.

    The per-case tables intentionally show section_path + top_score
    + matched chunk ids so authors can eyeball whether the linker's
    choice was the expected one. Long cases are truncated to the
    segments with expectations — the goal is to see the *expected*
    behaviour, not every linker output.
    """
    ts = generated_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    lines: list[str] = []
    header = "# detailed_summary Citation Eval"
    if label:
        header += f" — {label}"
    lines.append(header)
    lines.append("")
    lines.append(f"_Generated: {ts}_")
    lines.append("")

    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- total segments scored: **{aggregate.total_segments}**")
    lines.append(
        f"- has_citation precision: "
        f"**{aggregate.has_citation_precision:.1%}**  _(when a citation was returned, it pointed at an exact-hit chunk)_"
    )
    lines.append(
        f"- missing required citations: "
        f"**{aggregate.missing_required_citations}**  _(must_have_citation=true segments flipped to ⚠)_"
    )
    lines.append("")

    lines.append("### Location offset (primary metric)")
    lines.append("")
    lines.append(
        "``offset_at_top1`` = chunk-index distance between the system's "
        "top-1 chunk and the nearest ground-truth chunk. 0 = exact hit, "
        "1–2 = adjacent, 5+ = different part of the file. Computed only "
        f"for segments with known GT ({aggregate.n_with_offset} of "
        f"{aggregate.total_segments})."
    )
    lines.append("")
    lines.append(
        f"- mean: **{aggregate.mean_offset:.2f}**  "
        f"median (p50): **{aggregate.p50_offset:.1f}**  "
        f"p95: **{aggregate.p95_offset:.1f}**  "
        f"max: **{aggregate.max_offset}**"
    )
    lines.append("")
    lines.append("| threshold | hit rate (offset ≤ N) |")
    lines.append("|---|---:|")
    for n, ratio in sorted(aggregate.hit_at_offset.items()):
        tag = (
            " _(== strict top-1 accuracy)_" if n == 0 else ""
        )
        lines.append(f"| offset ≤ {n} | {ratio:.1%}{tag} |")
    lines.append("")

    if any(
        v["n"] for v in aggregate.calibration_by_score_band.values()
    ):
        lines.append("### Calibration by top_score band")
        lines.append("")
        lines.append(
            "Sanity-checks whether the system's own confidence signal "
            "predicts location correctness. If mean offset does NOT "
            "decrease as score increases, the 2-state ⚠/citation UI is "
            "discarding information."
        )
        lines.append("")
        lines.append(
            "| top_score | n | mean offset | median offset | hit@0 |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        band_order = [label for _, _, label in
                      ((0.00, 0.70, "<0.70"),
                       (0.70, 0.80, "[0.70-0.80)"),
                       (0.80, 0.85, "[0.80-0.85)"),
                       (0.85, 0.90, "[0.85-0.90)"),
                       (0.90, 1.01, "≥0.90"))]
        for label in band_order:
            row = aggregate.calibration_by_score_band.get(label, {})
            n = int(row.get("n", 0))
            if n == 0:
                lines.append(f"| {label} | 0 | — | — | — |")
                continue
            lines.append(
                f"| {label} | {n} | "
                f"{row['mean_offset']:.2f} | "
                f"{row['median_offset']:.1f} | "
                f"{row['hit_at_0']:.1%} |"
            )
        lines.append("")

    if aggregate.by_segment_type:
        lines.append("### By segment type (legacy binary)")
        lines.append("")
        lines.append("| type | top-1 (offset==0) | recall@3 | n |")
        lines.append("|---|---:|---:|---:|")
        for stype, (t1, r3, n) in sorted(aggregate.by_segment_type.items()):
            lines.append(
                f"| {stype} | {t1:.1%} | {r3:.1%} | {n} |"
            )
        lines.append("")

    lines.append("## Cases")
    lines.append("")
    for r in reports:
        lines.append(f"### `{r.case_id}`")
        lines.append("")
        lines.append(f"- file: `{r.file_path}`")
        lines.append(f"- file_id: `{r.file_id or '—'}`")
        if r.error:
            lines.append(f"- **error**: {r.error}")
            lines.append("")
            continue
        lines.append(
            "| section_path | type | offset | r@3 | has_cit | score | chunks |"
        )
        lines.append("|---|---|:-:|:-:|:-:|---:|---|")
        for s in r.segments:
            if s.offset_at_top1 is None:
                off = "—"
            elif s.offset_at_top1 == 0:
                off = "0 ✅"
            else:
                off = f"{s.offset_at_top1}"
            recall = "✅" if s.recall_at_3 else "❌"
            has_cit = "✅" if s.has_citation else "⚠"
            if s.must_have_citation and not s.has_citation:
                has_cit = "❌(required)"
            chunks = ", ".join(f"`{c}`" for c in s.top_chunk_ids[:3]) or "—"
            lines.append(
                f"| `{s.section_path}` | {s.segment_type} | "
                f"{off} | {recall} | {has_cit} | "
                f"{s.top_score:.2f} | {chunks} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_sidecar(
    reports: list[CaseReport],
    aggregate: AggregateMetrics,
    label: str = "",
    generated_at: str | None = None,
) -> dict:
    """Build a JSON-serialisable snapshot of the run.

    The schema mirrors the markdown: aggregate fields at the top,
    per-case / per-segment data under ``cases``. All dataclasses are
    flattened via ``asdict`` for stable round-tripping.
    """
    return {
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(),
        "label": label,
        "aggregate": {
            "total_segments": aggregate.total_segments,
            "top1_accuracy": aggregate.top1_accuracy,
            "recall_at_3": aggregate.recall_at_3,
            "has_citation_precision": aggregate.has_citation_precision,
            "missing_required_citations": (
                aggregate.missing_required_citations
            ),
            "by_segment_type": {
                k: {"top1_accuracy": t1, "recall_at_3": r3, "n": n}
                for k, (t1, r3, n) in aggregate.by_segment_type.items()
            },
            "n_with_offset": aggregate.n_with_offset,
            "mean_offset": aggregate.mean_offset,
            "p50_offset": aggregate.p50_offset,
            "p95_offset": aggregate.p95_offset,
            "max_offset": aggregate.max_offset,
            "hit_at_offset": {str(k): v for k, v in aggregate.hit_at_offset.items()},
            "calibration_by_score_band": aggregate.calibration_by_score_band,
        },
        "cases": [asdict(r) for r in reports],
    }


def write_report(path: Path, content: str) -> None:
    """Write the markdown body, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_sidecar(markdown_path: Path, sidecar: dict) -> None:
    """Write ``<report>.json`` next to the markdown output."""
    sidecar_path = markdown_path.with_suffix(".json")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compare_aggregates(
    baseline: dict, current: dict, epsilon: float = 0.01
) -> str:
    """Render a compact ``baseline → current`` delta as markdown.

    Useful for ``--baseline``: shows whether each metric improved,
    regressed, or stayed within ``epsilon`` ("tied"). The format is
    intentionally boring so diffs between runs stay scannable.
    """
    lines: list[str] = ["## Baseline comparison", ""]
    lines.append("| metric | baseline | current | delta |")
    lines.append("|---|---:|---:|---:|")

    def _row(name: str, b: float, c: float) -> str:
        d = c - b
        if abs(d) <= epsilon:
            marker = " (tied)"
        elif d > 0:
            marker = " ✅"
        else:
            marker = " ❌"
        return (
            f"| {name} | {b:.1%} | {c:.1%} | "
            f"{d:+.1%}{marker} |"
        )

    bagg = baseline.get("aggregate", {})
    cagg = current.get("aggregate", {})
    for key in (
        "top1_accuracy",
        "recall_at_3",
        "has_citation_precision",
    ):
        lines.append(
            _row(key, float(bagg.get(key, 0.0)), float(cagg.get(key, 0.0)))
        )
    return "\n".join(lines) + "\n"
