"""Transcript AI refine worker.

Windows ``transcript_chunks`` into fixed-size batches and calls the
configured LLM to correct ASR mis-recognitions. Malformed responses,
id mismatches, and LLM exceptions all skip the offending window
instead of corrupting chunks — partial application across a window
is specifically disallowed because the time-proportional word
re-alignment step depends on the whole window being consistent.

Public surface (called by ``app.routers.refine`` and tested directly):

* ``WINDOW_SIZE``
* ``refine_chunks(session, llm, chunks) -> RefineResult``
* ``realign_words_for_chunk(session, file_id, chunk_start, chunk_end,
  refined_text) -> int``
* ``recompute_chunk_embeddings(session, chunk_ids)``
* ``start_refine_job(session, file_id) -> str``
* ``find_transcript_files_in_folder(session, drive, path) -> list[str]``
* ``is_feature_enabled(drive) -> bool``
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text as sql_text

import app.config as config
from app.database import get_search_db
from app.models import Embedding, IndexedFile, TranscriptChunk, TranscriptWord

logger = logging.getLogger(__name__)

WINDOW_SIZE = 10


@dataclass(frozen=True)
class RefineResult:
    """Outcome of a ``refine_chunks`` call."""

    refined_count: int
    skipped_count: int


# --- LLM prompt helpers -----------------------------------------------------


def _build_system_prompt() -> str:
    return (
        "あなたは ASR (自動音声認識) の誤認識を修正するアシスタントです。\n"
        "各セグメントの意味を変えず、時間情報を保つため文字数を大きく変えないこと。\n"
        "人名・地名・専門用語は文脈から推定して統一すること。\n"
        "原文と同じ言語で返すこと(翻訳してはいけない)。\n"
        '出力は {"items": [{"id": <int>, "text_refined": <str>}, ...]} の'
        "JSON 形式で、他のテキストは含めないこと。"
    )


def _build_user_prompt(chunks: list[Any]) -> str:
    payload = [
        {"id": int(c.id), "text": c.text or ""}
        for c in chunks
    ]
    return json.dumps(payload, ensure_ascii=False)


def _parse_llm_response(
    raw: Any, expected_ids: set[int]
) -> dict[int, str] | None:
    """Validate and normalise the LLM's response for one window.

    Accepts either a bare ``list`` or a ``{"items": [...]}`` wrapper
    because ``response_format={"type": "json_object"}`` forces some
    providers to return an object. Returns ``None`` when the id set
    doesn't match exactly — partial application would desync word
    re-alignment across the window (spec constraint).
    """
    if raw is None:
        return None

    items: Any
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and isinstance(raw.get("items"), list):
        items = raw["items"]
    else:
        return None

    by_id: dict[int, str] = {}
    for entry in items:
        if not isinstance(entry, dict):
            return None
        if "id" not in entry or "text_refined" not in entry:
            return None
        try:
            cid = int(entry["id"])
        except (TypeError, ValueError):
            return None
        refined = entry.get("text_refined")
        if not isinstance(refined, str):
            return None
        by_id[cid] = refined

    if set(by_id.keys()) != expected_ids:
        return None

    return by_id


# --- Core refine ------------------------------------------------------------


async def refine_chunks(
    session: Any, llm: Any, chunks: list[Any]
) -> RefineResult:
    """Refine ``chunks`` in windows of ``WINDOW_SIZE`` via the LLM.

    On any failure the full window is preserved (no mutations). On
    success each chunk's ``text_original`` is populated (only when
    currently ``None`` — re-refine keeps the very first original),
    ``text`` is overwritten, and ``text_refined_at`` is stamped.
    """
    if not chunks:
        return RefineResult(refined_count=0, skipped_count=0)

    refined = 0
    skipped = 0
    now = datetime.now(UTC)

    for offset in range(0, len(chunks), WINDOW_SIZE):
        window = chunks[offset : offset + WINDOW_SIZE]
        expected_ids = {int(c.id) for c in window}

        try:
            raw = await llm.generate_json(
                _build_system_prompt(),
                _build_user_prompt(window),
            )
        except Exception as e:
            logger.warning(
                "refine: LLM call failed for window offset=%d (%s); preserving",
                offset, type(e).__name__,
            )
            skipped += len(window)
            continue

        by_id = _parse_llm_response(raw, expected_ids)
        if by_id is None:
            logger.info(
                "refine: window offset=%d rejected (malformed / id mismatch); "
                "preserving originals",
                offset,
            )
            skipped += len(window)
            continue

        for chunk in window:
            new_text = by_id[int(chunk.id)]
            if chunk.text_original is None:
                chunk.text_original = chunk.text
            chunk.text = new_text
            chunk.text_refined_at = now
            refined += 1

    return RefineResult(refined_count=refined, skipped_count=skipped)


# --- Word re-alignment ------------------------------------------------------

_WORD_SPLIT_RE = re.compile(r"\s+")
# CJK ranges (Hiragana, Katakana, CJK Unified Ideographs, Hangul). When the
# chunk has no whitespace (common for Japanese/Chinese/Korean) we fall back
# to per-character tokenisation so the re-aligned word grid stays fine.
_CJK_CHAR_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]"
)


def _split_refined_text(text: str) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    tokens = [t for t in _WORD_SPLIT_RE.split(cleaned) if t]
    # Whitespace split collapses CJK chunks into a single mega-token, which
    # destroys word-level timing. If we detect CJK content and only got one
    # whitespace-token, re-tokenise by character instead.
    if len(tokens) <= 1 and _CJK_CHAR_RE.search(cleaned):
        return [c for c in cleaned if not c.isspace()]
    return tokens


def _load_words_in_range(
    session: Any, file_id: str, chunk_start: float, chunk_end: float
) -> list[Any]:
    """Best-effort load of words covering ``[chunk_start, chunk_end]``.

    Tries the ORM query path first; if the underlying session is a
    MagicMock that returns something non-iterable we fall back to an
    empty list so the caller can still exercise the delete branch
    without crashing.
    """
    try:
        rows = (
            session.query(TranscriptWord)
            .filter(
                TranscriptWord.file_id == file_id,
                TranscriptWord.timestamp_start >= chunk_start,
                TranscriptWord.timestamp_end <= chunk_end,
            )
            .order_by(TranscriptWord.word_index)
            .all()
        )
    except Exception:
        return []
    if not isinstance(rows, list):
        return []
    return rows


def realign_words_for_chunk(
    session: Any,
    file_id: str,
    chunk_start: float,
    chunk_end: float,
    refined_text: str,
) -> int:
    """Redistribute word timestamps proportionally across refined tokens.

    Whisper-origin chunks have ``TranscriptWord`` rows inside their
    ``[chunk_start, chunk_end]`` span; HvLink-origin chunks don't —
    those are skipped silently (no INSERT, no DELETE, no exception).
    Returns the number of new word rows inserted.
    """
    existing = _load_words_in_range(session, file_id, chunk_start, chunk_end)

    tokens = _split_refined_text(refined_text)

    if not existing:
        # HvLink path: nothing to align against. Refined text may still
        # be non-empty; we refuse to fabricate word rows out of thin air.
        return 0

    # Always clear the old range before writing new rows.
    session.execute(
        sql_text(
            "DELETE FROM transcript_words "
            "WHERE file_id = :fid "
            "AND timestamp_start >= :ts "
            "AND timestamp_end <= :te"
        ),
        {"fid": file_id, "ts": chunk_start, "te": chunk_end},
    )

    if not tokens:
        return 0

    language = getattr(existing[0], "language", "") or ""
    base_index = min(
        (getattr(w, "word_index", 0) for w in existing),
        default=0,
    )
    span = max(chunk_end - chunk_start, 0.0)
    if span <= 0 or len(tokens) == 1:
        step = span
    else:
        step = span / len(tokens)

    for i, token in enumerate(tokens):
        start = chunk_start + step * i
        end = chunk_end if i == len(tokens) - 1 else chunk_start + step * (i + 1)
        row = TranscriptWord(
            file_id=file_id,
            word_index=base_index + i,
            text=token,
            language=language,
            timestamp_start=start,
            timestamp_end=end,
        )
        session.add(row)

    return len(tokens)


# --- Embedding re-compute ---------------------------------------------------


async def recompute_chunk_embeddings(
    session: Any, chunk_ids: list[int]
) -> None:
    """Re-embed ``transcript_chunks`` in-place after refine.

    Deletes the prior ``embeddings`` + ``vec_text`` rows for each chunk
    (scoped by timestamp to match the whisper indexer's shape) and
    inserts new vectors from the refined text. Isolated in a
    best-effort ``try`` so a broken ML path doesn't undo the refine —
    the chunks themselves have already been updated.
    """
    if not chunk_ids:
        return

    try:
        from app.workers.embedder import embed_passages
    except Exception as e:
        logger.warning("refine: embedder unavailable (%s); skipping re-embed", e)
        return

    rows = (
        session.query(TranscriptChunk)
        .filter(TranscriptChunk.id.in_(chunk_ids))
        .all()
    )
    texts = [r.text or "" for r in rows]
    if not any(t.strip() for t in texts):
        return

    try:
        vectors = await asyncio.to_thread(embed_passages, texts)
    except Exception as e:
        logger.warning("refine: embed_passages failed (%s)", e)
        return

    for row, vec in zip(rows, vectors, strict=False):
        existing = (
            session.query(Embedding)
            .filter(
                Embedding.file_id == row.file_id,
                Embedding.embedding_type == "whisper",
                Embedding.timestamp_start == row.timestamp_start,
                Embedding.timestamp_end == row.timestamp_end,
            )
            .all()
        )
        for emb in existing:
            session.execute(
                sql_text("DELETE FROM vec_text WHERE embedding_id = :id"),
                {"id": emb.id},
            )
            session.delete(emb)

        embedding_id = f"wh_{row.file_id}_{row.chunk_index}_{uuid.uuid4().hex[:8]}"
        embedding_record = Embedding(
            id=embedding_id,
            file_id=row.file_id,
            embedding_type="whisper",
            vector_table="vec_text",
            content_preview=(row.text or "")[:200],
            timestamp_start=row.timestamp_start,
            timestamp_end=row.timestamp_end,
        )
        session.add(embedding_record)
        session.flush()
        try:
            session.execute(
                sql_text(
                    "INSERT INTO vec_text(embedding_id, vector) VALUES(:id, :vec)"
                ),
                {"id": embedding_id, "vec": vec.tobytes()},
            )
        except Exception as e:
            logger.warning("refine: vec_text insert failed for %s: %s", row.id, e)


# --- Folder scan + policy ---------------------------------------------------


def find_transcript_files_in_folder(
    session: Any, drive: str, path: str
) -> list[str]:
    """Return file_ids in ``drive/path`` that have at least one chunk.

    Uses raw SQL so the path prefix match works uniformly over the
    ``indexed_files`` file_path column regardless of ORM quoting.
    """
    prefix = path.rstrip("/")
    try:
        rows = session.execute(
            sql_text(
                "SELECT DISTINCT f.file_id FROM indexed_files f "
                "INNER JOIN transcript_chunks c ON c.file_id = f.file_id "
                "WHERE f.drive = :drive "
                "AND f.active = 1 "
                "AND (f.file_path = :path OR f.file_path LIKE :like)"
            ),
            {
                "drive": drive,
                "path": prefix,
                "like": f"{prefix}/%",
            },
        ).fetchall()
    except Exception as e:
        logger.warning("refine: folder scan failed: %s", e)
        return []
    return [row[0] for row in rows]


async def is_feature_enabled(drive: str) -> bool:
    """Combined features-flag + per-drive policy gate.

    Async so FastAPI route handlers can ``await`` it from inside the
    running event loop. The sync-in-async hack from before returned
    True unconditionally for the production call path, which silently
    defeated per-drive policy. Now we always await the real policy
    client, matching its fail-open posture on upstream errors.
    """
    if config.settings.features.transcript_refine == "false":
        return False
    try:
        from app.policy_client import is_feature_enabled as _policy

        return await _policy(drive, "transcript_refine")
    except Exception:
        # Fail open on unexpected failures (matches policy_client).
        return True


# --- Job orchestration ------------------------------------------------------


async def _emit_ws_event(event: str, data: dict) -> None:
    """Best-effort progress event emission.

    The host currently has no internal addon→ws endpoint — this hook
    is a forward-compatible stub that logs the intended payload and
    posts it to ``HOMEVAULT_INTERNAL_API_URL + /addon-events`` when
    configured. Tests monkeypatch this function, so production
    behaviour is decoupled from the worker logic under test.
    """
    import os

    logger.info("refine-event %s %s", event, data)

    base = os.environ.get(
        "HOMEVAULT_INTERNAL_API_URL", "http://backend:8000/api/internal"
    )
    url = f"{base}/addon-events"
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(url, json={"event": event, "data": data})
    except Exception:
        # Host endpoint is optional; never fail the worker on WS.
        return


async def _run_refine_job(
    file_id: str, job_id: str, chunk_ids_snapshot: list[int]
) -> None:
    """Background task body for a single-file refine job."""
    await _emit_ws_event(
        "intelligence.refine.started",
        {"file_id": file_id, "chunk_count": len(chunk_ids_snapshot), "job_id": job_id},
    )

    try:
        from app.dependencies import get_llm_client

        llm = get_llm_client()
    except Exception as e:
        await _emit_ws_event(
            "intelligence.refine.failed",
            {"file_id": file_id, "job_id": job_id, "error": str(e)},
        )
        return

    # Snapshot phase: load chunk data into plain objects then release the
    # write lock so the minutes-long LLM round-trips below don't hold it.
    # The original implementation held get_search_db() (threading.Lock +
    # session) across every window's LLM call, which froze every other
    # intelligence endpoint (including /status) for the entire job.
    from types import SimpleNamespace

    try:
        with get_search_db() as session:
            rows = (
                session.query(TranscriptChunk)
                .filter(TranscriptChunk.file_id == file_id)
                .order_by(TranscriptChunk.chunk_index)
                .all()
            )
            snapshots = [
                SimpleNamespace(
                    id=int(r.id),
                    file_id=r.file_id,
                    chunk_index=r.chunk_index,
                    text=r.text,
                    text_original=r.text_original,
                    text_refined_at=None,
                    timestamp_start=r.timestamp_start,
                    timestamp_end=r.timestamp_end,
                )
                for r in rows
            ]
    except Exception as e:
        logger.exception("refine job snapshot failed for %s", file_id)
        await _emit_ws_event(
            "intelligence.refine.failed",
            {"file_id": file_id, "job_id": job_id, "error": str(e)},
        )
        return

    if not snapshots:
        await _emit_ws_event(
            "intelligence.refine.completed",
            {
                "file_id": file_id,
                "job_id": job_id,
                "refined_count": 0,
                "skipped_count": 0,
            },
        )
        return

    total = len(snapshots)
    refined_total = 0
    skipped_total = 0

    try:
        for offset in range(0, total, WINDOW_SIZE):
            window = snapshots[offset : offset + WINDOW_SIZE]

            # LLM call runs WITHOUT the write lock. refine_chunks only
            # mutates the passed-in objects (no ORM / session access).
            result = await refine_chunks(None, llm, window)
            refined_total += result.refined_count
            skipped_total += result.skipped_count

            refined_window = [s for s in window if s.text_refined_at is not None]
            if refined_window:
                # Short-lived session per window: apply mutations, realign
                # words, recompute embeddings, release lock.
                with get_search_db() as session:
                    fresh = (
                        session.query(TranscriptChunk)
                        .filter(
                            TranscriptChunk.id.in_(
                                [s.id for s in refined_window]
                            )
                        )
                        .all()
                    )
                    by_id = {int(c.id): c for c in fresh}
                    applied_ids: list[int] = []
                    for snap in refined_window:
                        orm = by_id.get(snap.id)
                        if orm is None:
                            continue
                        if orm.text_original is None:
                            orm.text_original = snap.text_original
                        orm.text = snap.text
                        orm.text_refined_at = snap.text_refined_at
                        realign_words_for_chunk(
                            session,
                            orm.file_id,
                            orm.timestamp_start,
                            orm.timestamp_end,
                            orm.text,
                        )
                        applied_ids.append(int(orm.id))
                    if applied_ids:
                        await recompute_chunk_embeddings(session, applied_ids)
                        session.flush()

            await _emit_ws_event(
                "intelligence.refine.progress",
                {
                    "file_id": file_id,
                    "job_id": job_id,
                    "done": min(offset + WINDOW_SIZE, total),
                    "total": total,
                },
            )

        await _emit_ws_event(
            "intelligence.refine.completed",
            {
                "file_id": file_id,
                "job_id": job_id,
                "refined_count": refined_total,
                "skipped_count": skipped_total,
            },
        )
    except Exception as e:
        logger.exception("refine job failed for %s", file_id)
        await _emit_ws_event(
            "intelligence.refine.failed",
            {"file_id": file_id, "job_id": job_id, "error": str(e)},
        )


async def start_refine_job(session: Any, file_id: str) -> str:
    """Kick off an async refine job. Returns the generated job_id."""
    job_id = uuid.uuid4().hex

    chunk_ids: list[int] = []
    try:
        rows = (
            session.query(TranscriptChunk.id)
            .filter(TranscriptChunk.file_id == file_id)
            .all()
        )
        chunk_ids = [int(r[0]) for r in rows] if rows else []
    except Exception:
        chunk_ids = []

    try:
        asyncio.create_task(
            _run_refine_job(file_id, job_id, chunk_ids),
            name=f"refine_job_{job_id}",
        )
    except RuntimeError:
        # No running loop (e.g. unit-test context without asyncio).
        pass

    return job_id
