"""Rare-keyword lookup that widens Ask's hierarchical shortlist.

A file that mentions the asked-about thing only in passing has a summary
that never names it, so the summary-embedding shortlist misses it. This
finds the files whose transcript or text contains a keyword token as an
FTS phrase, when few enough files do that the token is a specific thing
rather than a topic.
"""

import asyncio
import logging
import re

from sqlalchemy import text as sql_text

from app.database import get_search_engine

logger = logging.getLogger(__name__)

_MIN_TOKEN_LENGTH = 2

# Only the ``text`` column is searched: ``file_id``, ``chunk_index`` and
# ``page`` are indexed too, so an unrestricted MATCH would let ``LED`` hit a
# file id or ``150`` hit a chunk number.
_TABLES = (
    "fts_transcripts",
    "fts_text_content",
    "fts_transcripts_word",
    "fts_text_content_word",
)


def clean_tokens(keywords: str) -> list[str]:
    """Split ``keywords`` on Unicode whitespace into distinct usable tokens."""
    seen: set[str] = set()
    tokens: list[str] = []
    for raw in re.split(r"\s+", keywords):
        token = raw.replace('"', "").strip()
        if len(token) < _MIN_TOKEN_LENGTH or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _files_with_token(conn, table: str, token: str, drive: str) -> set[str]:
    rows = conn.execute(
        sql_text(
            f"SELECT DISTINCT {table}.file_id FROM {table} "
            f"JOIN indexed_files i ON i.file_id = {table}.file_id "
            f"WHERE {table} MATCH :q AND i.drive = :drive AND i.active = 1"
        ),
        {"q": f'text : "{token}"', "drive": drive},
    ).fetchall()
    return {row[0] for row in rows}


def find_exact_match_files(
    keywords: str, drive: str, *, max_files: int
) -> tuple[str, ...]:
    """Files of ``drive`` holding a keyword token that at most ``max_files`` files hold.

    Ordered by token, then ``file_id`` ascending within a token; each file
    once. A token whose lookup fails on any table contributes nothing,
    because a count from fewer tables could fall under the cap by accident.
    Synchronous; call through ``asyncio.to_thread``.
    """
    tokens = clean_tokens(keywords)
    if not tokens:
        return ()

    added: list[str] = []
    counts: dict[str, int] = {}
    over_cap: list[str] = []
    with get_search_engine().connect() as conn:
        for token in tokens:
            files: set[str] = set()
            failed = False
            for table in _TABLES:
                try:
                    files |= _files_with_token(conn, table, token, drive)
                except Exception as exc:
                    logger.warning(
                        "exact-match lookup failed: table=%s token=%r error=%s",
                        table, token, exc,
                    )
                    failed = True
                    break
            if failed:
                continue
            counts[token] = len(files)
            if not 1 <= len(files) <= max_files:
                if len(files) > max_files:
                    over_cap.append(token)
                continue
            added.extend(f for f in sorted(files) if f not in added)

    logger.debug(
        "exact-match lookup: drive=%s counts=%s over_cap=%s added=%d",
        drive, counts, over_cap, len(added),
    )
    return tuple(added)


async def lookup_exact_match_files(
    keywords: str, drive: str, *, max_files: int
) -> tuple[str, ...]:
    """Run the lookup off the event loop; any failure adds nothing."""
    try:
        return await asyncio.to_thread(
            find_exact_match_files, keywords, drive, max_files=max_files
        )
    except Exception as exc:
        logger.warning("exact-match lookup failed entirely: %s", exc)
        return ()
