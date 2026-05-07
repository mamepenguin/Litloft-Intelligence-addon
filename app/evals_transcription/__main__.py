"""CLI entry point: ``python -m app.evals_transcription``."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import sys
from pathlib import Path

# jiwer / jaconv are dev-only deps. Import inside main and convert
# ImportError into an actionable hint so the operator knows to install
# requirements-dev.txt.
try:
    import jiwer  # noqa: F401
    import jaconv  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised manually
    print(
        f"ERROR: missing eval dependency: {exc}.\n"
        "Run inside the intelligence container:\n"
        "  pip install -r requirements-dev.txt",
        file=sys.stderr,
    )
    sys.exit(2)


from app.evals_transcription.loader import load_cases
from app.evals_transcription.report import write_report
from app.evals_transcription.runner import (
    ALL_PROVIDERS,
    resolve_providers,
    run_eval,
)


_DEFAULT_CASES_DIR = "app/evals_transcription/cases"
_DEFAULT_REPORTS_DIR = "app/evals_transcription/reports"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.evals_transcription",
        description=(
            "Run every available transcription provider over the "
            "curated case set and emit a comparison report."
        ),
    )
    parser.add_argument(
        "--cases-dir",
        default=_DEFAULT_CASES_DIR,
        help=(
            "Directory containing case YAMLs and audio fixtures "
            f"(default: {_DEFAULT_CASES_DIR})"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Markdown report path (default: "
            f"{_DEFAULT_REPORTS_DIR}/<ISO datetime>.md)"
        ),
    )
    parser.add_argument(
        "--providers",
        default=None,
        help=(
            "Comma-separated subset of providers to run. Default: "
            f"all available ({','.join(ALL_PROVIDERS)})."
        ),
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional path to a previous JSON sidecar for deltas.",
    )
    args = parser.parse_args(argv)

    cases_dir = Path(args.cases_dir).resolve()
    cases = load_cases(cases_dir)
    if not cases:
        print(
            f"ERROR: no cases found under {cases_dir}. "
            f"Add *.yml files (see {cases_dir}/README.md).",
            file=sys.stderr,
        )
        return 1

    requested = (
        [n.strip() for n in args.providers.split(",") if n.strip()]
        if args.providers
        else None
    )
    try:
        providers = resolve_providers(requested)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    results = asyncio.run(run_eval(cases, providers))

    if args.output:
        output = Path(args.output)
    else:
        stamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output = Path(_DEFAULT_REPORTS_DIR) / f"{stamp}.md"
    baseline = Path(args.baseline) if args.baseline else None

    write_report(results, cases, output, baseline)
    print(f"Wrote {output} and {output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
