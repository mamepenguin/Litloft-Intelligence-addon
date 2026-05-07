"""Load and validate transcription eval cases from YAML.

A case file describes one audio fixture + its ground-truth transcript
+ optional speaker layout for diarization scoring + optional split-
test parameters. Loader returns a fully-validated ``Case`` dataclass;
failures raise ``ValueError`` with a path-prefixed message so the
caller can record / continue or abort as appropriate.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.evals_transcription.metrics import SpeakerSegment


_SUPPORTED_TIERS = ("short", "medium", "long")
_DURATION_TOLERANCE_FRACTION = 0.05  # ±5% relative
_DURATION_TOLERANCE_ABSOLUTE = 0.5   # 0.5 s absolute floor


@dataclass(frozen=True)
class SplitTest:
    forced_cap_bytes: int
    providers: tuple[str, ...]


@dataclass(frozen=True)
class Case:
    """One audio + GT pair to score every provider against."""

    name: str
    case_path: str        # absolute path to the YAML
    audio_path: str       # absolute path to the audio fixture
    language: str
    duration_s: float
    tier: str
    reference_transcript: str
    speakers: tuple[SpeakerSegment, ...] = ()
    split_test: SplitTest | None = None
    raw: dict = field(default_factory=dict)


def load_cases(cases_dir: str | os.PathLike) -> list[Case]:
    """Read every ``*.yml`` file in ``cases_dir`` (NOT ``*.yml.example``).

    Cases are returned in filename-sorted order so the report rows
    are stable across runs. An empty directory returns ``[]``; the
    CLI is responsible for surfacing "no cases found" to the user.
    """
    cases_dir = Path(cases_dir)
    if not cases_dir.is_dir():
        raise ValueError(
            f"Cases directory does not exist: {cases_dir}"
        )

    paths = sorted(
        p
        for p in glob.glob(str(cases_dir / "*.yml"))
        if not p.endswith(".yml.example")
    )
    return [_load_case(Path(p), cases_dir) for p in paths]


def _load_case(case_path: Path, cases_dir: Path) -> Case:
    try:
        raw = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{case_path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(
            f"{case_path}: top-level must be a mapping, got "
            f"{type(raw).__name__}"
        )

    name = _required_str(raw, "name", case_path)
    audio_rel = _required_str(raw, "audio_path", case_path)
    language = _required_str(raw, "language", case_path)
    duration_s = _required_float(raw, "duration_s", case_path)
    tier = _required_str(raw, "tier", case_path)
    reference = _required_str(raw, "reference_transcript", case_path)

    if tier not in _SUPPORTED_TIERS:
        raise ValueError(
            f"{case_path}: tier {tier!r} not in {_SUPPORTED_TIERS!r}"
        )
    audio_path = (cases_dir / audio_rel).resolve()
    if not audio_path.is_file():
        raise ValueError(
            f"{case_path}: audio_path resolves to missing file: {audio_path}"
        )

    speakers = _parse_speakers(raw.get("speakers"), duration_s, case_path)
    split_test = _parse_split_test(raw.get("split_test"), case_path)

    return Case(
        name=name,
        case_path=str(case_path),
        audio_path=str(audio_path),
        language=language,
        duration_s=float(duration_s),
        tier=tier,
        reference_transcript=reference,
        speakers=speakers,
        split_test=split_test,
        raw=raw,
    )


def _required_str(raw: dict, key: str, case_path: Path) -> str:
    if key not in raw:
        raise ValueError(f"{case_path}: missing required field {key!r}")
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{case_path}: field {key!r} must be a non-empty string"
        )
    return value


def _required_float(raw: dict, key: str, case_path: Path) -> float:
    if key not in raw:
        raise ValueError(f"{case_path}: missing required field {key!r}")
    value = raw[key]
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(
            f"{case_path}: field {key!r} must be a positive number"
        )
    return float(value)


def _parse_speakers(
    raw_speakers,
    duration_s: float,
    case_path: Path,
) -> tuple[SpeakerSegment, ...]:
    if raw_speakers is None:
        return ()
    if not isinstance(raw_speakers, list):
        raise ValueError(
            f"{case_path}: speakers must be a list, got "
            f"{type(raw_speakers).__name__}"
        )
    out: list[SpeakerSegment] = []
    for entry in raw_speakers:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{case_path}: each speaker entry must be a mapping"
            )
        speaker_id = entry.get("id")
        if not isinstance(speaker_id, str) or not speaker_id:
            raise ValueError(
                f"{case_path}: speaker entry missing string 'id'"
            )
        segments = entry.get("segments") or []
        if not isinstance(segments, list):
            raise ValueError(
                f"{case_path}: speaker {speaker_id!r} segments must be a list"
            )
        for seg in segments:
            if not isinstance(seg, list) or len(seg) != 2:
                raise ValueError(
                    f"{case_path}: speaker {speaker_id!r} segment must be"
                    f" [start, end]"
                )
            start, end = float(seg[0]), float(seg[1])
            if not 0 <= start < end <= duration_s + _DURATION_TOLERANCE_ABSOLUTE:
                raise ValueError(
                    f"{case_path}: speaker {speaker_id!r} segment "
                    f"[{start}, {end}] out of [0, {duration_s}]"
                )
            out.append(SpeakerSegment(speaker_id, start, end))
    return tuple(out)


def _parse_split_test(raw_split, case_path: Path) -> SplitTest | None:
    if raw_split is None:
        return None
    if not isinstance(raw_split, dict):
        raise ValueError(
            f"{case_path}: split_test must be a mapping"
        )
    cap = raw_split.get("forced_cap_bytes")
    if not isinstance(cap, int) or cap <= 0:
        raise ValueError(
            f"{case_path}: split_test.forced_cap_bytes must be a "
            f"positive integer"
        )
    providers = raw_split.get("providers") or []
    if not isinstance(providers, list) or not all(
        isinstance(p, str) and p for p in providers
    ):
        raise ValueError(
            f"{case_path}: split_test.providers must be a list of strings"
        )
    return SplitTest(forced_cap_bytes=cap, providers=tuple(providers))


def duration_within_tolerance(
    expected: float,
    actual: float,
) -> bool:
    """Return True when ``actual`` is within ±5% (or ±0.5 s, whichever
    is larger) of ``expected``. Used by the runner to flag YAML /
    audio mismatches without aborting the whole run."""
    tolerance = max(
        _DURATION_TOLERANCE_FRACTION * expected,
        _DURATION_TOLERANCE_ABSOLUTE,
    )
    return abs(actual - expected) <= tolerance
