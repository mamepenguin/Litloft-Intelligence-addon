"""Turn a stored (NFC) drive-relative path into the path that exists on disk.

Linux compares names byte for byte, so a file copied onto a drive with a
decomposed (NFD) name is not found by its stored NFC path. Nothing here writes,
renames or creates anything, which keeps it usable on the read-only mounts this
addon gets.

This is a copy of the core resolver (``backend/app/services/diskpath.py``) kept
in step by review, because the addon runs in its own container. Every call
reads the disk: there is no cache, so a stale answer cannot outlive a rename.
"""
from __future__ import annotations

import logging
import os
import threading
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

_WARNED_MAX = 1024
_warned: set[tuple[str, str]] = set()
_warned_lock = threading.Lock()


class AmbiguousPath(Exception):
    """More than one directory entry matches a component and none equals it byte for byte."""


def _warn_unreadable(directory: str, error: OSError) -> None:
    key = (directory, type(error).__name__)
    with _warned_lock:
        if key in _warned or len(_warned) >= _WARNED_MAX:
            return
        _warned.add(key)
    logger.warning("Cannot list %s while resolving a stored path: %s", directory, error)


def _matching_entries(directory: str, component: str) -> list[str]:
    """Entries of ``directory`` whose NFC form equals the NFC form of ``component``."""
    wanted = unicodedata.normalize("NFC", component)
    try:
        with os.scandir(directory) as it:
            found = [e.name for e in it if unicodedata.normalize("NFC", e.name) == wanted]
    except (FileNotFoundError, NotADirectoryError):
        return []
    except OSError as exc:
        _warn_unreadable(directory, exc)
        return []
    return sorted(found, key=lambda n: n.encode("utf-8", "surrogateescape"))


def _walk(root: Path, rel: str, strict: bool) -> Path:
    """Resolve ``rel`` component by component; the unmatched tail is kept as given."""
    parts = rel.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return root / rel
    current = os.fspath(root)
    for index, part in enumerate(parts):
        exact = os.path.join(current, part)
        if os.path.lexists(exact):
            current = exact
            continue
        matches = _matching_entries(current, part)
        if not matches:
            return Path(current) / "/".join(parts[index:])
        if strict and len(matches) > 1:
            raise AmbiguousPath(rel)
        current = os.path.join(current, matches[0])
    return Path(current)


def resolve_on_disk(root: str | Path, rel: str, *, strict: bool = False) -> Path:
    """The path under ``root`` that exists for ``rel``, or ``root/rel`` when none does.

    ``strict`` raises AmbiguousPath instead of picking the first of several
    equally good matches.
    """
    root = Path(root)
    direct = root / rel
    if os.path.lexists(direct):
        return direct
    return _walk(root, rel, strict)


def resolve_parent(root: str | Path, rel: str, *, strict: bool = False) -> Path:
    """``rel`` with its parent folder resolved and the leaf left as given."""
    parent, _, leaf = rel.rpartition("/")
    if not parent:
        return Path(root) / rel
    return _walk(Path(root), parent, strict) / leaf
