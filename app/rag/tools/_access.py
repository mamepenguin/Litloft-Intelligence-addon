"""Shared access-control + input-validation helpers for the agentic tools.

The agentic loop receives untrusted ``file_id`` values from the LLM —
any of which could be a guess, a leaked ID from a previous unlocked
session, or a value injected via a poisoned tool result. Every tool
that takes ``file_id`` as input MUST gate it through ``ensure_access``
before issuing an Internal-API call; otherwise the wrapper turns the
host's drive-isolation guarantee (``design-decisions.md``: "When a
protected drive is locked, exclude it from API responses entirely")
into Swiss cheese.

This module also enforces a strict shape on ``file_id`` and ``kind``
so a malicious LLM cannot bend the URL with ``..``, ``/``, ``?``, etc.

Why a separate module: the helpers are shared across four tools and
imported defensively from a few tests. Keeping them in one place
means a single regex / cap update propagates everywhere.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

from app.credentials import CallerCredential
from app.rag.retriever import _filter_file_ids_via_internal_api

logger = logging.getLogger(__name__)


# Litloft file_ids are short opaque strings (typically 8-12 chars) but
# we accept a generous shape to avoid coupling to one specific shape.
# The deny-list is what matters: no path separators, no query, no
# percent escapes, no whitespace.
_FILE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Mirrors the host's ``FileRelation.kind`` constraint (internal.py:583).
_KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,31}$")

# Cap on the citation allow-list to keep budget bounded.
ALLOW_LIST_CAP = 200


def is_valid_file_id(file_id: object) -> bool:
    """Strict shape check before any URL interpolation."""
    return isinstance(file_id, str) and bool(_FILE_ID_RE.fullmatch(file_id))


def is_valid_kind(kind: object) -> bool:
    """Strict shape check for ``file_relations.kind`` filter values."""
    return isinstance(kind, str) and bool(_KIND_RE.fullmatch(kind))


async def ensure_access(
    file_ids: Iterable[str], credential: CallerCredential | None
) -> set[str]:
    """Return the subset of ``file_ids`` the caller is allowed to see.

    Thin re-export of the retriever's internal helper. Centralised so
    every tool has one obvious choice — and so a future refactor that
    renames or tightens the helper updates the tools together.
    """
    ids = [fid for fid in file_ids if isinstance(fid, str) and fid]
    if not ids:
        return set()
    return await _filter_file_ids_via_internal_api(
        file_ids=ids, credential=credential
    )


__all__ = [
    "ALLOW_LIST_CAP",
    "ensure_access",
    "is_valid_file_id",
    "is_valid_kind",
]
