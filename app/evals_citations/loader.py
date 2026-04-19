"""Case YAML loader for the citation eval harness.

A case is one file + a list of per-segment expectations. We load the
actual ``detailed_summary`` text from the snapshot DB (not from the
YAML) so authors can't accidentally drift from what the product shows.
Expectations are a small dataclass: either ``chunk_ids`` (exact chunk
match, best when the author inspected DB rows) or a hint range
(``time_range`` for audio/video, ``page`` for documents) that a chunk
must overlap to count as a match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SegmentHint:
    """Either ``time_range`` (seconds) or ``page`` (document). Both optional."""

    time_range: tuple[float, float] | None = None
    page: int | None = None


@dataclass(frozen=True)
class SegmentExpectation:
    """Ground-truth for one segment of a detailed_summary.

    One of ``chunk_ids`` or ``hint`` is required. When both are given,
    ``chunk_ids`` takes precedence for "match the retrieved chunk"
    scoring and ``hint`` is used to derive a readable label only.
    ``must_have_citation`` lets a case assert that the segment should
    never land on the "⚠ no strong source" path (e.g. the table row
    that carries the core numeric claim).
    """

    section_path: str
    chunk_ids: tuple[str, ...] = ()
    hint: SegmentHint | None = None
    must_have_citation: bool | None = None


@dataclass(frozen=True)
class CitationCase:
    """One file's worth of citation ground truth."""

    id: str
    file_path: str
    expectations: tuple[SegmentExpectation, ...]
    notes: str = ""
    source_path: Path | None = field(default=None, compare=False)


def _parse_hint(raw: object) -> SegmentHint | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("segment_hint must be a mapping")
    time_range: tuple[float, float] | None = None
    page: int | None = None
    if "time_range" in raw:
        tr = raw["time_range"]
        if (
            not isinstance(tr, list)
            or len(tr) != 2
            or not all(isinstance(x, (int, float)) for x in tr)
        ):
            raise ValueError("segment_hint.time_range must be [start, end]")
        time_range = (float(tr[0]), float(tr[1]))
    if "page" in raw:
        if not isinstance(raw["page"], int):
            raise ValueError("segment_hint.page must be an integer")
        page = raw["page"]
    if time_range is None and page is None:
        return None
    return SegmentHint(time_range=time_range, page=page)


def _parse_expectations(raw: object) -> tuple[SegmentExpectation, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("expectations must be a list")
    out: list[SegmentExpectation] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("expectations entries must be mappings")
        sp = entry.get("section_path")
        if not isinstance(sp, str) or not sp.strip():
            raise ValueError("expectations[].section_path must be a string")
        chunk_ids_raw = entry.get("chunk_ids") or ()
        if not isinstance(chunk_ids_raw, (list, tuple)):
            raise ValueError("expectations[].chunk_ids must be a list")
        chunk_ids = tuple(str(c) for c in chunk_ids_raw)
        hint = _parse_hint(entry.get("segment_hint"))
        must = entry.get("must_have_citation")
        if must is not None and not isinstance(must, bool):
            raise ValueError(
                "expectations[].must_have_citation must be bool"
            )
        if not chunk_ids and hint is None and must is None:
            raise ValueError(
                f"expectations[{sp!r}] must set chunk_ids, "
                "segment_hint, or must_have_citation"
            )
        out.append(
            SegmentExpectation(
                section_path=sp,
                chunk_ids=chunk_ids,
                hint=hint,
                must_have_citation=must,
            )
        )
    return tuple(out)


def load_case(path: Path) -> CitationCase:
    """Parse a single citation case yml."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping")

    case_id = data.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"{path}: 'id' is required and must be a string")

    file_path = data.get("file_path")
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError(f"{path}: 'file_path' is required")

    expectations = _parse_expectations(data.get("expectations"))
    notes = data.get("notes") or ""
    if not isinstance(notes, str):
        notes = str(notes)

    return CitationCase(
        id=case_id,
        file_path=file_path,
        expectations=expectations,
        notes=notes,
        source_path=path,
    )


def load_cases(
    cases_path: Path, filter_substr: str | None = None
) -> list[CitationCase]:
    """Load every .yml (or a single file) from a directory."""
    if cases_path.is_file():
        candidates = [cases_path]
    elif cases_path.is_dir():
        candidates = sorted(cases_path.glob("*.yml")) + sorted(
            cases_path.glob("*.yaml")
        )
    else:
        raise FileNotFoundError(f"Cases path does not exist: {cases_path}")

    cases: list[CitationCase] = []
    for p in candidates:
        try:
            case = load_case(p)
        except Exception as e:
            raise ValueError(f"Failed to load citation case {p}: {e}") from e
        if filter_substr and filter_substr not in case.id:
            continue
        cases.append(case)
    return cases
