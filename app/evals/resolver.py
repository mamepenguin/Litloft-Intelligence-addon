"""Resolve case ground-truth paths to snapshot DB ``file_id``s.

The runner needs to map case-relative paths (``videos/kyoto.mp4``) to
the opaque file_ids stored in the snapshot's ``indexed_files`` table.
We do this once at startup so per-case stage logic stays a pure
in-memory lookup.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.evals.loader import Case, GroundTruthFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedGroundTruth:
    """A ground-truth entry with file_id resolved (or marked missing)."""

    case_id: str
    path: str
    file_id: str | None
    segment_hint: object | None  # SegmentHint, kept loose to avoid cycle


def build_path_to_file_id(snapshot_db: Path, drive: str) -> dict[str, str]:
    """Read the snapshot's indexed_files table → ``rel_path -> file_id``.

    The DB stores absolute paths (``/drives/eval-drive/foo/bar.mp4``).
    We strip the conventional ``/drives/{drive}/`` prefix so callers can
    look up by case-relative paths.
    """
    if not snapshot_db.exists():
        raise FileNotFoundError(f"Snapshot DB not found: {snapshot_db}")

    conn = sqlite3.connect(str(snapshot_db))
    try:
        cur = conn.execute(
            "SELECT file_id, file_path FROM indexed_files WHERE drive = ?",
            (drive,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    prefix = f"/drives/{drive}/"
    out: dict[str, str] = {}
    for file_id, file_path in rows:
        if isinstance(file_path, str) and file_path.startswith(prefix):
            rel = file_path[len(prefix):]
            out[rel] = file_id
        else:
            # Tolerate snapshot/DB schema drift: keep absolute path too.
            out[str(file_path)] = file_id
    return out


def resolve_case(
    case: Case, path_to_id: dict[str, str]
) -> list[ResolvedGroundTruth]:
    """Resolve every ground_truth entry, warning on misses."""
    resolved: list[ResolvedGroundTruth] = []
    for gt in case.ground_truth_files:
        file_id = path_to_id.get(gt.path)
        if file_id is None:
            logger.warning(
                "Case %s: ground_truth path %r not found in snapshot",
                case.id,
                gt.path,
            )
        resolved.append(
            ResolvedGroundTruth(
                case_id=case.id,
                path=gt.path,
                file_id=file_id,
                segment_hint=gt.segment_hint,
            )
        )
    return resolved


def hash_file_sha256(path: Path) -> str:
    """Compute sha256 of a file (used for snapshot integrity in the report)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_git_commit() -> str:
    """Best-effort current git short hash; returns 'unknown' on failure."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 — eval-time best-effort metadata
        return "unknown"
