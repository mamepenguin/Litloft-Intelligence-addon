"""Case YAML → typed dataclass loader for the eval harness.

Intentionally lightweight: pydantic is overkill for a dev-time tool
with a fixed, small schema. We validate just enough to fail loudly on
authoring mistakes (missing id/query, malformed segment_hint) without
imposing a hard dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentHint:
    """Optional segment-level ground-truth hint.

    ``time_range`` (audio/video, seconds) and ``page`` (documents) are
    mutually informative — a case may set either or neither.
    """

    time_range: tuple[float, float] | None = None
    page: int | None = None


@dataclass(frozen=True)
class GroundTruthFile:
    """A single ground-truth entry resolved from a case yml."""

    path: str
    segment_hint: SegmentHint | None = None


@dataclass(frozen=True)
class ExpectedKeywords:
    must_include: tuple[str, ...] = ()
    must_exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class Case:
    """A single eval case loaded from yml."""

    id: str
    query: str
    expected_keywords: ExpectedKeywords
    ground_truth_files: tuple[GroundTruthFile, ...]
    must_mention: tuple[str, ...] = ()
    notes: str = ""
    # Optional ordered list of tool names the agentic loop is expected
    # to call (Phase 1.A). Empty means "no expectation" — the
    # route_correctness metric returns ``None`` for such cases and they
    # are excluded from the aggregate, not scored as zero. Prefix
    # match: the actual call sequence must start with these tools (in
    # order); extra trailing calls are tolerated.
    expected_tool_sequence: tuple[str, ...] = ()
    source_path: Path | None = field(default=None, compare=False)


def _coerce_str_list(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list, got {type(value).__name__}")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} entries must be strings")
        out.append(item)
    return tuple(out)


def _parse_segment_hint(raw: object) -> SegmentHint | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("segment_hint must be a mapping")
    time_range = None
    page = None
    if "time_range" in raw:
        tr = raw["time_range"]
        if (
            not isinstance(tr, list)
            or len(tr) != 2
            or not all(isinstance(x, (int, float)) for x in tr)
        ):
            raise ValueError("segment_hint.time_range must be [start, end] floats")
        time_range = (float(tr[0]), float(tr[1]))
    if "page" in raw:
        if not isinstance(raw["page"], int):
            raise ValueError("segment_hint.page must be an integer")
        page = raw["page"]
    if time_range is None and page is None:
        return None
    return SegmentHint(time_range=time_range, page=page)


def _parse_ground_truth(raw: object) -> tuple[GroundTruthFile, ...]:
    # Empty list is permitted for "no-answer" cases (Phase G, 006).
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("ground_truth_files must be a list")
    if not raw:
        return ()
    out: list[GroundTruthFile] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("ground_truth_files entries must be mappings")
        path = entry.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("ground_truth_files[].path must be a non-empty string")
        hint = _parse_segment_hint(entry.get("segment_hint"))
        out.append(GroundTruthFile(path=path, segment_hint=hint))
    return tuple(out)


def load_case(path: Path) -> Case:
    """Parse a single yml file into a Case dataclass."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping")

    case_id = data.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"{path}: 'id' is required and must be a string")

    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{path}: 'query' is required and must be a string")

    expected = data.get("expected_keywords") or {}
    if not isinstance(expected, dict):
        raise ValueError(f"{path}: expected_keywords must be a mapping")

    expected_kw = ExpectedKeywords(
        must_include=_coerce_str_list(
            expected.get("must_include"), "expected_keywords.must_include"
        ),
        must_exclude=_coerce_str_list(
            expected.get("must_exclude"), "expected_keywords.must_exclude"
        ),
    )

    gt = _parse_ground_truth(data.get("ground_truth_files"))
    must_mention = _coerce_str_list(data.get("must_mention"), "must_mention")
    expected_tool_sequence = _coerce_str_list(
        data.get("expected_tool_sequence"), "expected_tool_sequence"
    )
    notes = data.get("notes") or ""
    if not isinstance(notes, str):
        notes = str(notes)

    return Case(
        id=case_id,
        query=query,
        expected_keywords=expected_kw,
        ground_truth_files=gt,
        must_mention=must_mention,
        expected_tool_sequence=expected_tool_sequence,
        notes=notes,
        source_path=path,
    )


def load_cases(cases_path: Path, filter_substr: str | None = None) -> list[Case]:
    """Load every yml in a directory (or a single yml file)."""
    if cases_path.is_file():
        candidates = [cases_path]
    elif cases_path.is_dir():
        candidates = sorted(cases_path.glob("*.yml")) + sorted(
            cases_path.glob("*.yaml")
        )
    else:
        raise FileNotFoundError(f"Cases path does not exist: {cases_path}")

    cases: list[Case] = []
    for p in candidates:
        try:
            case = load_case(p)
        except Exception as e:
            raise ValueError(f"Failed to load case {p}: {e}") from e
        if filter_substr and filter_substr not in case.id:
            continue
        cases.append(case)

    if not cases:
        raise ValueError(
            f"No cases loaded from {cases_path}"
            + (f" matching filter {filter_substr!r}" if filter_substr else "")
        )
    return cases
