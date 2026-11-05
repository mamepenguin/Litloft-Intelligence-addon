"""Helpers that build NFC/NFD-named trees for the SPEC-ADDON-062 tests."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest


def form_name(component: str) -> str:
    """Resolve a parity-table component: ``nfc:X`` / ``nfd:X`` / literal."""
    if component.startswith("nfc:"):
        return unicodedata.normalize("NFC", component[4:])
    if component.startswith("nfd:"):
        return unicodedata.normalize("NFD", component[4:])
    return component


def form_path(spec: str) -> str:
    return "/".join(form_name(part) for part in spec.split("/"))


def filesystem_folds_normalization(directory: Path) -> bool:
    """True when NFC and NFD spellings of one name are the same entry here."""
    nfc = unicodedata.normalize("NFC", "ぱ")
    nfd = unicodedata.normalize("NFD", "ぱ")
    (directory / nfd).write_bytes(b"")
    try:
        return (directory / nfc).exists()
    finally:
        (directory / nfd).unlink()


def require_distinguishing_filesystem(directory: Path) -> None:
    """Skip, never pass, on a filesystem that folds NFC and NFD."""
    if filesystem_folds_normalization(directory):
        pytest.skip("filesystem folds NFC/NFD; run the tests in Docker (Linux)")


def make_file(root: Path, spec: str, data: bytes = b"x") -> Path:
    path = root / form_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
