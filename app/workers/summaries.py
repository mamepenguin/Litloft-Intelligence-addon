"""AI summaries worker using LLM.

Generates two-layer summaries (1-sentence short + paragraph long) for
videos, audio files, and documents. Runs as a dedicated async queue
processing one file at a time.

Unlike auto_tags, summaries do not have an approve/dismiss workflow:
the generated summary is stored in the intelligence DB and displayed
directly. The host HomeVault DB is never touched. Users can "hide" a
summary (status='hidden') or "regenerate" it (delete + re-enqueue).

Long content (transcripts, document text) that exceeds the configured
threshold is sampled from beginning/middle/end windows rather than
truncated from the front — this preserves coverage of the full file
without requiring expensive map-reduce over many LLM calls.
"""

import asyncio
import logging
from datetime import UTC, datetime

from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_search_db
from app.llm import LLMClient
from app.models import IndexedFile, TranscriptChunk
from app.text_utils import trim_to_sentence_boundary

logger = logging.getLogger(__name__)

# Re-export the shared helper under its legacy private name so existing
# callers (and the summaries tests) that imported `_trim_to_sentence_boundary`
# from this module keep working after the helper moved to app.text_utils.
_trim_to_sentence_boundary = trim_to_sentence_boundary

_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ja": "- 要約は日本語で生成すること\n",
    "en": "- Summaries must be in English\n",
}

# Context types this worker produces summaries for.
# Images are intentionally excluded — BLIP captions already fill
# that role and running a second LLM pass adds no value.
_SUPPORTED_CONTEXT_TYPES: frozenset[str] = frozenset({"video", "audio", "document"})

# Separator inserted between sampled windows in truncated contexts.
_WINDOW_SEPARATOR = "\n\n[...中略...]\n\n"


def _build_system_prompt() -> str:
    """Build the system prompt with language instruction from config."""
    lang = settings.llm.output_language
    lang_line = _LANGUAGE_INSTRUCTIONS.get(lang, "")

    return (
        "あなたはファイル管理システムの要約アシスタントです。\n"
        "以下のコンテンツを読んで、短いサマリーと段落要約を生成してください。\n"
        "\n"
        "規則:\n"
        '- JSON形式で返すこと: {"short": "1文サマリー", "long": "段落要約"}\n'
        "- short: 1文（30-80文字程度）でファイル全体の要点を表すこと\n"
        "- long: 3-5文（200-400文字程度）で主要な内容を説明すること\n"
        f"{lang_line}"
        "- JSONのみ返し、他のテキストは含めないこと"
    )


def _sample_windows(text: str, window_chars: int, window_count: int) -> str:
    """Sample head/middle/tail windows from a long text and join them.

    Uses endpoint distribution: the first window starts at the beginning,
    the last window ends at the end, and middle windows are centered at
    evenly-spaced fractions between. For window_count=3 this places
    windows at 0%, 50%, and 100% of the text — guaranteeing head and
    tail coverage, which an "even centers" approach would miss.

    Adjacent windows are merged if they overlap so duplicated content
    doesn't waste LLM context. Each window is then trimmed to sentence
    boundaries before being joined with a separator marker.

    Args:
        text: Source text to sample from.
        window_chars: Width of each window in characters.
        window_count: Number of windows to take (use odd numbers for
            symmetric head/middle/tail coverage).

    Returns:
        Joined windows with a separator marker between non-adjacent spans.
    """
    total = len(text)
    if window_count <= 0 or window_chars <= 0 or total == 0:
        return text

    # Compute (start, end) spans. For n=1 we take a single head window;
    # for n>1 we distribute centers at i/(n-1) so windows 0 and n-1
    # are pinned to the head and tail respectively.
    spans: list[tuple[int, int]] = []
    for i in range(window_count):
        if window_count == 1:
            center = 0
        else:
            center = int(total * i / (window_count - 1))
        half = window_chars // 2
        start = max(0, center - half)
        end = min(total, start + window_chars)
        # If we hit the tail, shift start back so the window stays full-width.
        # The shifted start can overlap the previous window — that's fine
        # because the merge pass below dedupes overlapping spans.
        if end == total:
            start = max(0, end - window_chars)
        spans.append((start, end))

    # Merge overlapping spans so we don't duplicate text between windows.
    merged: list[tuple[int, int]] = []
    for span in spans:
        if merged and span[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
        else:
            merged = [*merged, span]

    snippets = [trim_to_sentence_boundary(text[s:e]) for s, e in merged]
    return _WINDOW_SEPARATOR.join(snippets)


def _prepare_context(text: str) -> tuple[str, bool]:
    """Prepare the context text for the LLM, sampling windows if needed.

    Args:
        text: Raw context text (full transcript, full document, etc.).

    Returns:
        Tuple of (prepared_text, was_truncated). was_truncated is True
        when the input exceeded the threshold and windows were sampled.
    """
    cfg = settings.summaries
    if len(text) <= cfg.max_context_chars:
        return (text, False)

    sampled = _sample_windows(text, cfg.window_chars, cfg.window_count)
    return (sampled, True)


def _classify_file_type(file_type: str) -> str | None:
    """Map a raw file_type to a summaries context type, or None if unsupported."""
    if file_type == "video":
        return "video"
    if file_type == "audio":
        return "audio"
    if file_type in ("document", "text"):
        return "document"
    return None


def classify_missing_reason(file_id: str) -> str:
    """Explain why a file does not currently have a stored summary.

    Called by the router when the file_summaries table has no row for
    a given file, so the frontend can render the right UI state rather
    than always offering a "Generate" button that would silently skip.

    Return values:
        "file_not_found"        — no indexed_files row at all
        "unsupported_type"      — image/archive/etc.
        "insufficient_content"  — supported type but below threshold
        "not_generated"         — ready to generate, waiting for user
    """
    indexed_file = _get_indexed_file(file_id)
    if indexed_file is None:
        return "file_not_found"

    context_type = _classify_file_type(indexed_file["file_type"])
    if context_type is None:
        return "unsupported_type"

    # Reuse _build_context so the threshold check is applied exactly
    # the same way the worker would apply it — no second source of truth.
    if _build_context(indexed_file, context_type) is None:
        return "insufficient_content"

    return "not_generated"


def _get_indexed_file(file_id: str) -> dict | None:
    """Fetch basic indexed-file info for a file, or None if not indexed."""
    with get_search_db() as session:
        f = (
            session.query(IndexedFile)
            .filter(
                IndexedFile.file_id == file_id,
                IndexedFile.active.is_(True),
            )
            .first()
        )
        if f is None:
            return None
        return {
            "file_id": f.file_id,
            "filename": f.filename,
            "file_type": f.file_type,
            "title": f.title,
            "description": f.description,
        }


def _get_full_transcript(file_id: str) -> str:
    """Load the entire Whisper transcript for a file as one string."""
    with get_search_db() as session:
        chunks = (
            session.query(TranscriptChunk)
            .filter(TranscriptChunk.file_id == file_id)
            .order_by(TranscriptChunk.chunk_index)
            .all()
        )
        if not chunks:
            return ""
        return " ".join(c.text for c in chunks if c.text)


def _get_full_document_text(file_id: str) -> str:
    """Load the entire extracted document text for a file as one string.

    Reads from the fts_text_content FTS5 table rather than the embeddings
    table — the latter only stores 200-char content_previews (see
    app/workers/metadata.py), while fts_text_content holds the full
    chunk text written during indexing. Chunks are concatenated in
    numeric chunk_index order.
    """
    with get_search_db() as session:
        # chunk_index is stored as a string in the FTS5 table, so we cast
        # to INTEGER for correct ordering (otherwise "10" sorts before "2").
        rows = session.execute(
            sql_text(
                "SELECT text FROM fts_text_content "
                "WHERE file_id = :fid "
                "ORDER BY CAST(chunk_index AS INTEGER)"
            ),
            {"fid": file_id},
        ).fetchall()
        if not rows:
            return ""
        return "\n\n".join(row[0] for row in rows if row[0])


def _build_context(indexed_file: dict, context_type: str) -> str | None:
    """Build raw context text for a file, or None if no content is available.

    Enforces settings.summaries.min_context_chars: any file whose usable
    content (after stripping whitespace) is shorter than that threshold
    returns None and is skipped by the worker. This guards against the
    LLM hallucinating a summary from only the filename when Whisper
    produces a trivial transcript (e.g., "you" on a silent piano video)
    or a document extractor yields a near-empty text layer.

    Args:
        indexed_file: File info dict from _get_indexed_file.
        context_type: "video" | "audio" | "document".

    Returns:
        The raw (untruncated) context text, or None if the file has no
        usable transcript / document text to summarize.
    """
    file_id = indexed_file["file_id"]
    raw: str = ""

    if context_type in ("video", "audio"):
        raw = _get_full_transcript(file_id)
    elif context_type == "document":
        raw = _get_full_document_text(file_id)

    if not raw:
        return None

    min_chars = settings.summaries.min_context_chars
    if len(raw.strip()) < min_chars:
        return None

    return raw


def _build_user_prompt(
    indexed_file: dict,
    context_type: str,
    context: str,
    was_truncated: bool,
) -> str:
    """Build the user prompt for LLM summary generation."""
    parts: list[str] = [
        f"ファイル名: {indexed_file['filename']}",
        f"タイプ: {context_type}",
    ]

    title = indexed_file.get("title") or ""
    if title and title != indexed_file["filename"]:
        parts = [*parts, f"タイトル: {title}"]

    description = indexed_file.get("description") or ""
    if description:
        parts = [*parts, f"説明: {description}"]

    if was_truncated:
        parts = [
            *parts,
            "\n注: 以下は長いコンテンツの抜粋です。冒頭・中盤・終盤から取得しています。",
        ]

    parts = [*parts, "\n--- コンテンツ ---", context]

    return "\n".join(parts)


def _has_summary(file_id: str) -> bool:
    """True if a summary (in any status) already exists for this file."""
    with get_search_db() as session:
        row = session.execute(
            sql_text("SELECT 1 FROM file_summaries WHERE file_id = :fid"),
            {"fid": file_id},
        ).fetchone()
        return row is not None


def _save_summary(
    *,
    file_id: str,
    short_summary: str,
    long_summary: str,
    model: str,
    context_type: str,
    context_chars: int,
    was_truncated: bool,
) -> None:
    """Insert or replace the summary record for a file."""
    now = datetime.now(UTC).isoformat()
    with get_search_db() as session:
        session.execute(
            sql_text(
                "INSERT OR REPLACE INTO file_summaries "
                "(file_id, short_summary, long_summary, model, context_type, "
                "context_chars, was_truncated, status, created_at) "
                "VALUES (:file_id, :short_summary, :long_summary, :model, "
                ":context_type, :context_chars, :was_truncated, 'generated', "
                ":created_at)"
            ),
            {
                "file_id": file_id,
                "short_summary": short_summary,
                "long_summary": long_summary,
                "model": model,
                "context_type": context_type,
                "context_chars": context_chars,
                "was_truncated": 1 if was_truncated else 0,
                "created_at": now,
            },
        )


class SummariesWorker:
    """Async worker that processes summary generation requests via a queue."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def enqueue(self, file_id: str) -> None:
        """Add a file to the summaries queue."""
        await self._queue.put(file_id)

    async def enqueue_unprocessed(self) -> int:
        """Find indexed files without summaries and enqueue them.

        Only enqueues files whose file_type is supported by summaries
        (video/audio/document) and whose metadata has been indexed.

        Returns:
            Number of files queued.
        """
        with get_search_db() as session:
            rows = session.execute(
                sql_text(
                    "SELECT f.file_id FROM indexed_files f "
                    "WHERE f.active = 1 AND f.metadata_indexed = 1 "
                    "AND f.file_type IN ('video', 'audio', 'document', 'text') "
                    "AND f.file_id NOT IN (SELECT file_id FROM file_summaries)"
                )
            ).fetchall()

        count = 0
        for (file_id,) in rows:
            await self._queue.put(file_id)
            count += 1
        return count

    async def run(self) -> None:
        """Main worker loop. Processes one file at a time."""
        while True:
            try:
                file_id = await self._queue.get()
                await self._process_file(file_id)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Summaries worker error: %s", e)

    async def _process_file(self, file_id: str) -> None:
        """Generate a summary for a single file.

        Skips the file if summaries are disabled, the LLM client is
        unavailable, a summary already exists, the file is unsupported,
        or no context can be extracted.
        """
        if settings.features.summaries == "false":
            return

        if not self._llm_client.enabled:
            return

        if _has_summary(file_id):
            return

        indexed_file = _get_indexed_file(file_id)
        if indexed_file is None:
            return

        context_type = _classify_file_type(indexed_file["file_type"])
        if context_type not in _SUPPORTED_CONTEXT_TYPES:
            return

        raw_context = _build_context(indexed_file, context_type)
        if not raw_context:
            logger.debug(
                "Summaries: no context available for %s (%s)",
                file_id, context_type,
            )
            return

        prepared, was_truncated = _prepare_context(raw_context)
        user_prompt = _build_user_prompt(
            indexed_file, context_type, prepared, was_truncated
        )

        parsed = await self._llm_client.generate_json(
            _build_system_prompt(), user_prompt
        )

        if not isinstance(parsed, dict):
            logger.warning(
                "Summaries LLM returned non-dict for %s, skipping", file_id
            )
            return

        short_raw = parsed.get("short")
        long_raw = parsed.get("long")
        if not isinstance(short_raw, str) or not isinstance(long_raw, str):
            logger.warning(
                "Summaries LLM response missing short/long fields for %s", file_id
            )
            return

        short_summary = short_raw.strip()
        long_summary = long_raw.strip()
        if not short_summary or not long_summary:
            logger.warning(
                "Summaries LLM produced empty short/long for %s", file_id
            )
            return

        _save_summary(
            file_id=file_id,
            short_summary=short_summary,
            long_summary=long_summary,
            model=settings.llm.model,
            context_type=context_type,
            context_chars=len(prepared),
            was_truncated=was_truncated,
        )
        logger.info(
            "Summaries: saved summary for %s (%s, %d chars, truncated=%s)",
            file_id, context_type, len(prepared), was_truncated,
        )
