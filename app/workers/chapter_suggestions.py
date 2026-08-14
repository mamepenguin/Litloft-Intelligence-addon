"""LLM-derived media chapter candidates and core promotion client."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx
from sqlalchemy import text as sql_text

from app.config import settings
from app.llm import LLMClient
from app.output_language import configured_language_requirement
from app.prompt_loader import render
from app.workers.transcription.errors import TransientError

logger = logging.getLogger(__name__)

# An internal implementation budget, not an operator tuning surface. Every
# transcript row is included in exactly one time-ordered window; long input is
# never reduced to a head excerpt.
_WINDOW_CHAR_BUDGET = 12_000
_CORE_BASE_DEFAULT = "http://backend:8000"
_READY_EVENT = "intelligence.chapter_suggestions.ready"
_FAILED_EVENT = "intelligence.chapter_suggestions.failed"


def _build_system_prompt(output_language: str | None) -> str:
    return render(
        "chapter_suggestions/system.jinja2",
        language_requirement=configured_language_requirement(
            output_language,
            auto_requirement="Use the transcript language.",
        ),
    )


async def is_chapter_suggestions_enabled(drive: str) -> bool:
    """Fail closed when the per-drive privacy policy is unavailable."""
    from app.policy_client import is_feature_enabled

    try:
        return await is_feature_enabled(
            drive,
            "chapter_suggestions",
            default_on_failure=False,
        )
    except TransientError:
        logger.warning(
            "Chapter suggestions policy unavailable for drive=%s; skipping",
            drive,
        )
        return False


def normalise_chapter_candidates(raw: Any) -> list[dict[str, Any]]:
    """Validate untrusted LLM/core-bound chapter values.

    Semantics intentionally mirror core's external-input normaliser: preserve
    producer order, drop entries with blank titles or invalid/non-finite starts,
    and turn invalid/non-finite ends into null. Core assigns dense ordering.
    """
    items = raw.get("chapters") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title_raw = item.get("title")
        if not isinstance(title_raw, str):
            continue
        title = title_raw.strip()
        if not title:
            continue
        try:
            start = float(item.get("start_time"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start):
            continue
        end: float | None
        try:
            raw_end = item.get("end_time")
            end = None if raw_end is None else float(raw_end)
        except (TypeError, ValueError):
            end = None
        if end is not None and not math.isfinite(end):
            end = None
        result.append({
            "start_time": start,
            "end_time": end,
            "title": title,
        })
    return result


def _build_windows(chunks: list[Any]) -> list[str]:
    """Cover all chunk text in windows whose serialized size is bounded."""
    windows: list[str] = []
    current: list[str] = []
    current_chars = 0
    for chunk in chunks:
        prefix = (
            f"[{float(chunk.timestamp_start):.3f}-"
            f"{float(chunk.timestamp_end):.3f}] "
        )
        text_value = chunk.text.strip()
        # A provider can emit one pathological giant chunk. Split its body
        # while repeating the timestamp range so no window exceeds budget and
        # every source character still reaches the LLM.
        body_budget = max(1, _WINDOW_CHAR_BUDGET - len(prefix))
        parts = [
            text_value[offset:offset + body_budget]
            for offset in range(0, len(text_value), body_budget)
        ] or [""]
        for part in parts:
            line = prefix + part
            added = len(line) + (1 if current else 0)
            if current and current_chars + added > _WINDOW_CHAR_BUDGET:
                windows.append("\n".join(current))
                current = []
                current_chars = 0
            current.append(line)
            current_chars += len(line) + (1 if len(current) > 1 else 0)
    if current:
        windows.append("\n".join(current))
    return windows


def _group_candidate_sets(
    candidate_sets: list[list[dict[str, Any]]],
) -> list[list[list[dict[str, Any]]]]:
    """Group adjacent candidate sets under the consolidation budget."""
    groups: list[list[list[dict[str, Any]]]] = []
    current: list[list[dict[str, Any]]] = []
    chars = 2
    for candidate_set in candidate_sets:
        encoded_len = len(json.dumps(candidate_set, ensure_ascii=False)) + 1
        # Always combine at least two adjacent sets. That guarantees the
        # hierarchy converges without dropping a tail candidate merely to
        # satisfy an implementation budget.
        if (
            len(current) >= 2
            and chars + encoded_len > _WINDOW_CHAR_BUDGET
        ):
            groups.append(current)
            current = []
            chars = 2
        current.append(candidate_set)
        chars += encoded_len
    if current:
        groups.append(current)
    return groups


async def _generate_usable_candidates(
    llm_client: LLMClient,
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, Any]]:
    """Request complete JSON once, then retry with a coarser repair prompt."""
    for attempt in range(2):
        prompt = user_prompt if attempt == 0 else render(
            "chapter_suggestions/retry_user.jinja2",
            original_prompt=user_prompt,
        )
        raw = await llm_client.generate_json(
            system_prompt,
            prompt,
            temperature=0.1,
        )
        candidates = normalise_chapter_candidates(raw)
        if candidates:
            return candidates
    return []


async def promote_chapters_to_core(
    file_id: str,
    chapters: list[dict[str, Any]],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """Replace core's active chapter set with an approved candidate set."""
    secret = os.environ.get("CORE_INTERNAL_SECRET", "")
    if not secret.strip():
        raise RuntimeError("CORE_INTERNAL_SECRET is required for chapter approval")
    base = os.environ.get("HOMEVAULT_INTERNAL_URL", _CORE_BASE_DEFAULT).rstrip("/")
    url = f"{base}/api/internal/files/{quote(file_id, safe='')}/chapters"
    async with httpx.AsyncClient(
        timeout=10.0,
        transport=transport,
        trust_env=False,
    ) as client:
        response = await client.put(
            url,
            headers={"X-Internal-Secret": secret},
            json={"chapters": chapters},
        )
    response.raise_for_status()


async def emit_chapter_suggestions_event(event: str, data: dict[str, Any]) -> None:
    """Best-effort, drive-scoped completion event through the core WS bridge."""
    base = os.environ.get(
        "HOMEVAULT_INTERNAL_API_URL", "http://backend:8000/api/internal"
    ).rstrip("/")
    secret = os.environ.get("CORE_INTERNAL_SECRET", "")
    headers = {"X-Internal-Secret": secret} if secret else {}
    payload: dict[str, Any] = {"event": event, "data": data}
    if drive := data.get("drive"):
        payload["drive"] = drive
    try:
        async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
            response = await client.post(
                f"{base}/addon-events",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPError:
        logger.warning("Could not emit chapter suggestions event %s", event)


class ChapterSuggestionsWorker:
    """Serial generation queue for staged chapter candidates."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._queue: asyncio.Queue[tuple[str, bool]] = asyncio.Queue()
        self._queued: set[str] = set()
        self._processing: list[str] = []

    def get_status(self) -> dict[str, object]:
        return {
            "waiting": self._queue.qsize(),
            "processing": list(self._processing),
        }

    async def enqueue(self, file_id: str, *, force: bool = False) -> bool:
        from app.database import get_search_db_read
        from app.models import IndexedFile

        with get_search_db_read() as session:
            row = session.query(IndexedFile.drive).filter(
                IndexedFile.file_id == file_id,
                IndexedFile.active.is_(True),
            ).first()
        if row is None or not await is_chapter_suggestions_enabled(row.drive):
            return False
        if file_id in self._queued or file_id in self._processing:
            return False
        self._queued.add(file_id)
        await self._queue.put((file_id, force))
        return True

    async def enqueue_unprocessed(self) -> int:
        from app.database import get_search_db_read

        with get_search_db_read() as session:
            rows = session.execute(sql_text(
                "SELECT f.file_id, f.drive FROM indexed_files f "
                "WHERE f.active = 1 AND f.file_type IN ('video', 'audio') "
                "AND EXISTS (SELECT 1 FROM transcript_chunks t "
                "            WHERE t.file_id = f.file_id) "
                "AND NOT EXISTS (SELECT 1 FROM suggested_chapters s "
                "                WHERE s.file_id = f.file_id)"
            )).fetchall()
        allowed_by_drive: dict[str, bool] = {}
        count = 0
        for file_id, drive in rows:
            if drive not in allowed_by_drive:
                allowed_by_drive[drive] = await is_chapter_suggestions_enabled(drive)
            if allowed_by_drive[drive] and await self.enqueue(file_id):
                count += 1
        return count

    async def run(self) -> None:
        while True:
            try:
                file_id, force = await self._queue.get()
                self._queued.discard(file_id)
                self._processing.append(file_id)
                try:
                    await self._process_file(file_id, force=force)
                finally:
                    self._processing.remove(file_id)
                    self._queue.task_done()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Chapter suggestions worker failed")

    async def _process_file(self, file_id: str, *, force: bool = False) -> None:
        if settings.features.chapter_suggestions == "false":
            return
        if not self._llm_client.enabled:
            return

        from app.database import get_search_db, get_search_db_read
        from app.models import IndexedFile, TranscriptChunk

        with get_search_db_read() as session:
            indexed = session.query(IndexedFile).filter(
                IndexedFile.file_id == file_id,
                IndexedFile.active.is_(True),
                IndexedFile.file_type.in_(("video", "audio")),
            ).first()
            if indexed is None:
                return
            if not force:
                exists = session.execute(sql_text(
                    "SELECT 1 FROM suggested_chapters WHERE file_id=:fid"
                ), {"fid": file_id}).first()
                if exists is not None:
                    return
            chunks = session.query(TranscriptChunk).filter(
                TranscriptChunk.file_id == file_id
            ).order_by(
                TranscriptChunk.timestamp_start,
                TranscriptChunk.chunk_index,
            ).all()
            filename = indexed.filename
            drive = indexed.drive
        if not chunks:
            return
        # Policy may change while a job waits in the serial queue. Re-check at
        # the last safe point before any transcript reaches the LLM.
        if not await is_chapter_suggestions_enabled(drive):
            return

        system = _build_system_prompt(settings.llm.output_language)
        candidate_sets: list[list[dict[str, Any]]] = []
        windows = _build_windows(chunks)
        generation_started = time.monotonic()
        logger.info(
            "Chapter suggestions: generating for %s (%d windows)",
            file_id,
            len(windows),
        )
        try:
            for window_index, window in enumerate(windows):
                window_candidates = await _generate_usable_candidates(
                    self._llm_client,
                    system,
                    render(
                        "chapter_suggestions/window_user.jinja2",
                        filename=filename,
                        transcript=window,
                    ),
                )
                if not window_candidates:
                    logger.warning(
                        "Chapter suggestions: no usable model output for %s "
                        "window %d/%d after %.2fs",
                        file_id,
                        window_index + 1,
                        len(windows),
                        time.monotonic() - generation_started,
                    )
                    await emit_chapter_suggestions_event(
                        _FAILED_EVENT,
                        {
                            "file_id": file_id,
                            "drive": drive,
                            "reason": "invalid_model_output",
                        },
                    )
                    return
                candidate_sets.append(window_candidates)

            # Every result receives an editorial pass. Long inputs consolidate
            # hierarchically; a single window is still edited so transcript
            # timestamps cannot become one micro-chapter each.
            while candidate_sets:
                groups = (
                    _group_candidate_sets(candidate_sets)
                    if len(candidate_sets) > 1
                    else [candidate_sets]
                )
                merged_sets: list[list[dict[str, Any]]] = []
                for group in groups:
                    flattened = [chapter for item in group for chapter in item]
                    merged = await _generate_usable_candidates(
                        self._llm_client,
                        system,
                        render(
                            "chapter_suggestions/consolidate_user.jinja2",
                            candidates_json=json.dumps(
                                flattened,
                                ensure_ascii=False,
                            ),
                        ),
                    )
                    if not merged:
                        logger.warning(
                            "Chapter suggestions: editorial output invalid "
                            "for %s after %.2fs",
                            file_id,
                            time.monotonic() - generation_started,
                        )
                        await emit_chapter_suggestions_event(
                            _FAILED_EVENT,
                            {
                                "file_id": file_id,
                                "drive": drive,
                                "reason": "invalid_model_output",
                            },
                        )
                        return
                    merged_sets.append(merged)
                candidate_sets = merged_sets
                if len(candidate_sets) == 1:
                    break
            chapters = candidate_sets[0]
        except Exception:
            logger.exception(
                "Chapter suggestion generation crashed for %s", file_id
            )
            await emit_chapter_suggestions_event(
                _FAILED_EVENT,
                {
                    "file_id": file_id,
                    "drive": drive,
                    "reason": "generation_error",
                },
            )
            return

        created_at = datetime.now(UTC).isoformat()
        with get_search_db() as session:
            session.execute(sql_text(
                "INSERT INTO suggested_chapters "
                "(file_id, chapters_json, model, created_at, status) "
                "VALUES (:fid, :chapters, :model, :created_at, 'pending') "
                "ON CONFLICT(file_id) DO UPDATE SET "
                "chapters_json=excluded.chapters_json, model=excluded.model, "
                "created_at=excluded.created_at, status='pending'"
            ), {
                "fid": file_id,
                "chapters": json.dumps(chapters, ensure_ascii=False),
                "model": settings.llm.model or "unknown",
                "created_at": created_at,
            })
        await emit_chapter_suggestions_event(
            _READY_EVENT,
            {
                "file_id": file_id,
                "drive": drive,
                "created_at": created_at,
            },
        )
        logger.info(
            "Chapter suggestions: saved %d chapters for %s "
            "(%d windows, %.2fs)",
            len(chapters),
            file_id,
            len(windows),
            time.monotonic() - generation_started,
        )
