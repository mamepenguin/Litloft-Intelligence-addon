#!/usr/bin/env python3
"""Ask the collector what it measured, and compare it with a declared population.

`--cov-fail-under` is a lower bound, and a lower bound is not a detector on its
own: shrink the denominator and the percentage goes up. This script is what
earns the floor the right to be a lower bound. It reads the JSON report
coverage.py wrote — the population the collector actually produced — and
compares it, as a set, against an independent walk of this checkout.

Three things about how it does that are deliberate.

**Two sides, two implementations.** One side is the report; the other is the
walk below, with its predicates written out. Reading the report twice — counting
its entries, or filtering it for test paths — is not a second implementation: a
module missing from the report is missing from the filtered copy too, so both
sides lose it in the same step. That is detector rule 5.

**Sets, not counts.** A count cannot catch a deletion, and a hand-maintained
count teaches everyone to update it rather than to look at it.

**It fails in both directions.** A check built from the report alone holds only
the direction where something *enters* the population. Measured in the sibling `media_import` repository, where both vectors exist:

    a production module leaves, dropped by a coverage `omit` entry
        -> 88.83% against a floor of 80.50, every step green, and the module
           count printed in the log fell from 14 to 13, asserted by nothing
    a production module never arrives, because its directory has no
    `__init__.py`
        -> 80.5045%, identical to four decimals, because coverage.py does not
           walk such a directory at all

Both are failures here. The second is why the walk does not require an
`__init__.py`: replicating coverage.py's own rule would reproduce its blind spot
on the side that is supposed to catch it.

This file is duplicated in the sibling addon repositories, like `frontmatter.py`
and `credentials.py`, because they are separate repositories that cannot share
code. Keeping them in step is a PR-review duty.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (directory in this checkout, prefix its files carry in the coverage report).
# Declared, not discovered. `pytest.ini`'s `--cov=app` is the other half of this
# claim; the two are meant to be read against each other.
SOURCE_ROOTS = [("app", "app/")]

# Directory names the walk never descends. `tests` is here because the sibling
# `media_import` package has already shipped with its own tests in the
# denominator — 62% of it, on files that cannot fail to be covered. This package keeps its tests outside `app/`, and this line is what says so
# rather than leaving it to be assumed.
SKIP_DIRS = {"__pycache__", "tests"}

# This package has live instances of the vector the walk exists to catch:
# directories under `app/` with no `__init__.py`, which coverage.py does not
# descend. They are `app/blocklists/`, `app/data/`, `app/evals_transcription/
# cases/`, `app/prompts/`, and `app/prompts/{auto_tags,chapter_suggestions,rag,
# refine,retrieval_keywords,summaries,video_visual_scene,vision}/`. None holds a
# `.py` today, which is why the walk and the report agree at 142. One dropped
# into any of them would never reach coverage.py, and nothing said so before.

# Trees under `app/` that are NOT the code this addon ships, and so are not in
# the denominator: the evaluation harnesses, each with its own `__main__.py`,
# reached from nothing the service imports. `.coveragerc` keeps them out of the
# report and this list keeps them out of the walk, so the two agree.
#
# Written out one tree at a time to mirror `.coveragerc` exactly. Neither side
# uses a wildcard over `app/evals*`, because a wildcard would take in a fourth
# tree without anyone deciding to.
#
# What these lines assume — that nothing shipped imports them — is held by
# `tests/test_evals_not_in_runtime.py`. This file cannot hold it: excluding a
# tree from both sides keeps them consistent whatever the import graph does.
EXCLUDED_TREES = (
    "app/evals/",
    "app/evals_citations/",
    "app/evals_transcription/",
)

# Files on disk that the report is right not to contain, declared by name with a
# reason. Empty here, measured: outside the excluded trees, the walk and the
# report agree at 117 files. Adding an entry is a claim that coverage.py is
# correct to omit that file — it is not a way to silence a disagreement.
NOT_MEASURED: list[str] = []


def production_sources(root: Path, prefix: str) -> set[str]:
    """Every `.py` file under `root`, as the report would name it."""
    found: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
        ]
        for name in filenames:
            if name.startswith(".") or not name.endswith(".py"):
                continue
            rel = (Path(dirpath) / name).relative_to(root)
            found.add(f"{prefix}{rel.as_posix()}")
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <coverage.json> <floor>", file=sys.stderr)
        return 2
    report_path, floor = Path(argv[1]), float(argv[2])

    if not report_path.is_file():
        print(f"::error::{report_path} was not written — PYTEST_ADDOPTS did not reach pytest,")
        print("::error::which means --cov-fail-under did not either and nothing was gated.")
        return 1

    data = json.loads(report_path.read_text())
    measured = set(data["files"])
    total = data["totals"]["percent_covered"]

    declared: set[str] = set()
    for rel_root, prefix in SOURCE_ROOTS:
        root = REPO_ROOT / rel_root
        if not root.is_dir():
            print(f"::error::{root} is not a directory — this check is walking the wrong tree,")
            print("::error::so its agreement with the report would mean nothing.")
            return 1
        declared |= production_sources(root, prefix)
    declared = {
        p for p in declared if not p.startswith(EXCLUDED_TREES)
    }
    declared -= set(NOT_MEASURED)

    missing = sorted(declared - measured)
    unexpected = sorted(measured - declared)

    if missing:
        print(f"::error::{len(missing)} source file(s) are in this tree but were not measured:")
        for path in missing:
            print(f"::error::  - {path}")
        print("::error::coverage.py does not walk a directory that has no __init__.py, and an")
        print("::error::`omit` entry drops files without a word. Either one shrinks the")
        print("::error::denominator, which raises the percentage: the floor cannot see it.")
    if unexpected:
        print(f"::error::{len(unexpected)} measured file(s) are not production sources of this package:")
        for path in unexpected:
            print(f"::error::  - {path}")
        print("::error::Tests inside the measured package inflate the total with files that")
        print("::error::cannot fail to be covered. A file under one of EXCLUDED_TREES means")
        print("::error::.coveragerc's omit is not matching. Anything else means NOT_MEASURED")
        print("::error::is stale, or the walk no longer matches what --cov selects.")
    if missing or unexpected:
        return 1

    print(
        f"coverage {total:.4f}% against a floor of {floor}%, "
        f"over {len(measured)} production modules matching an independent walk"
    )
    if total < floor:
        print(f"::error::{total:.4f}% is below {floor}% and the run did not fail — the floor is not being enforced")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
