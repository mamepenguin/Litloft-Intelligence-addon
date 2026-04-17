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
from app.workers.whisper import HVLINK_MIME
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

# Language instructions for the detailed (Markdown long-form) summary.
# Unlike the short/long prompt above these spell out the required writing
# style because the detailed output is user-facing prose, not a caption.
_DETAILED_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ja": (
        "- 日本語で、自然で読みやすい文体。各セクションの見出しを使用\n"
    ),
    "en": (
        "- Write in English with a natural, readable style. "
        "Use headings for each section\n"
    ),
}

# Context types this worker produces summaries for.
# Images are intentionally excluded — BLIP captions already fill
# that role and running a second LLM pass adds no value.
_SUPPORTED_CONTEXT_TYPES: frozenset[str] = frozenset({"video", "audio", "document"})

# Separator inserted between sampled windows in truncated contexts.
_WINDOW_SEPARATOR = "\n\n[...中略...]\n\n"

# Per-request token budget for detailed summaries. Generous because the
# output is a full Markdown document (intro + bullets + table + conclusion)
# and can legitimately run past the default 2048 cap. 4096 leaves head
# room for Japanese output where each character costs one token.
_DETAILED_MAX_TOKENS = 4096

# detailed_status transitions: None → "generating" → "generated" | "failed".
# "generating" is the only non-terminal state; transitions through it are
# taken while a background task is working on the file.
DETAILED_STATUS_GENERATING = "generating"
DETAILED_STATUS_GENERATED = "generated"
DETAILED_STATUS_FAILED = "failed"


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
        "- short: 1文（30-80文字）で要点を表す。断定的な誇張を避けること\n"
        "- long: 3-5文（200-400文字）で主要内容を説明する。原文にないニュアンスを付け加えないこと\n"
        f"{lang_line}"
        "- JSONのみ返し、他のテキストは含めないこと"
    )


def _build_detailed_system_prompt() -> str:
    """Build the system prompt for detailed (Markdown long-form) summaries.

    The user-visible output is a structured Markdown document with four
    sections (intro / detailed bullets / key-points table / conclusion).
    Language style is selected from ``settings.llm.output_language``;
    ``"auto"`` omits the style line so the model mirrors the source.
    """
    lang = settings.llm.output_language
    lang_line = _DETAILED_LANGUAGE_INSTRUCTIONS.get(lang, "")

    return (
        "あなたはファイル管理システムの要約アシスタントです。\n"
        "以下のコンテンツを読んで、長文の構造化要約を Markdown 形式で生成してください。\n"
        "\n"
        "規則:\n"
        "- 出力は Markdown のみ。JSON や他のラッパーで包まないこと\n"
        "- 次のセクションで構成すること:\n"
        "  1. **導入**（1-2文）: 全体像と魅力を簡潔に\n"
        "  2. **詳細内容**: 時系列や論理順で箇条書き"
        "（各項目に簡潔な説明文を添える。「・」で開始）\n"
        "  3. **重要ポイントまとめ**: Markdown 表で数値・比較・注意点を整理\n"
        "  4. **結論**（1-2文）: 価値や次のアクションを提案\n"
        "- 要約者の視点ではなく、語り手の視点をそのまま維持する\n"
        "- 原文にない事実やニュアンスを付け加えないこと\n"
        f"{lang_line}"
    )


def _build_detailed_user_prompt(
    indexed_file: dict,
    context_type: str,
    context: str,
    was_truncated: bool,
) -> str:
    """Build the user prompt for detailed-summary generation.

    Shares the same header layout (filename / type / title / description
    / truncation notice / content marker) as ``_build_user_prompt`` so
    the LLM sees a consistent structure across both summary variants.
    Labels stay Japanese regardless of output language — they are model
    instructions, not output, and the model does not mirror them.
    """
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


def _prepare_context(
    text: str,
    *,
    max_chars: int | None = None,
    window_chars: int | None = None,
    window_count: int | None = None,
) -> tuple[str, bool]:
    """Prepare the context text for the LLM, sampling windows if needed.

    Args:
        text: Raw context text (full transcript, full document, etc.).
        max_chars: Full-text threshold override. Falls back to
            ``settings.summaries.max_context_chars`` when None.
        window_chars: Per-window width override for the sampling
            fallback. Falls back to ``settings.summaries.window_chars``.
        window_count: Window-count override for the sampling fallback.
            Falls back to ``settings.summaries.window_count``.

    Returns:
        Tuple of (prepared_text, was_truncated). was_truncated is True
        when the input exceeded the threshold and windows were sampled.
    """
    cfg = settings.summaries
    effective_max = cfg.max_context_chars if max_chars is None else max_chars
    effective_window_chars = (
        cfg.window_chars if window_chars is None else window_chars
    )
    effective_window_count = (
        cfg.window_count if window_count is None else window_count
    )
    if len(text) <= effective_max:
        return (text, False)

    sampled = _sample_windows(
        text, effective_window_chars, effective_window_count
    )
    return (sampled, True)


def _classify_file_type(file_type: str, mime_type: str | None = None) -> str | None:
    """Map a raw file_type to a summaries context type, or None if unsupported.

    HVLink files (external video references like YouTube) are classified as
    "video" so their VTT-derived transcripts feed the same summary path as
    local videos — their host-side file_type is "other" by MIME heuristics.
    """
    if mime_type == HVLINK_MIME:
        return "video"
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

    context_type = _classify_file_type(
        indexed_file["file_type"], indexed_file.get("mime_type")
    )
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
            "mime_type": f.mime_type,
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


def _has_detailed_summary(file_id: str) -> bool:
    """True if a detailed summary exists for this file (any status).

    ``generating`` rows count as "has" — the router uses this to 409 a
    second generation request while one is already in flight.
    """
    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT 1 FROM file_summaries "
                "WHERE file_id = :fid AND detailed_status IS NOT NULL"
            ),
            {"fid": file_id},
        ).fetchone()
        return row is not None


def _set_detailed_status(
    file_id: str,
    status: str,
    *,
    error: str | None = None,
    model: str | None = None,
) -> None:
    """Upsert the row and set detailed_status.

    On first write we may land on a file that has no file_summaries row
    at all (short/long never generated). INSERT OR IGNORE establishes
    the row with empty short/long placeholders, then UPDATE writes the
    detailed-related columns. Using placeholders keeps the NOT NULL
    constraints on short_summary/long_summary happy without lying about
    their presence — the status column is the source of truth for
    "does detailed exist".
    """
    now = datetime.now(UTC).isoformat()
    with get_search_db() as session:
        session.execute(
            sql_text(
                "INSERT OR IGNORE INTO file_summaries "
                "(file_id, short_summary, long_summary, model, context_type, "
                "context_chars, was_truncated, status, created_at) "
                "VALUES (:fid, '', '', '', '', 0, 0, 'hidden', :now)"
            ),
            {"fid": file_id, "now": now},
        )
        session.execute(
            sql_text(
                "UPDATE file_summaries SET "
                "detailed_status = :status, "
                "detailed_error = :error, "
                "detailed_model = COALESCE(:model, detailed_model) "
                "WHERE file_id = :fid"
            ),
            {
                "fid": file_id,
                "status": status,
                "error": error,
                "model": model,
            },
        )


def _save_detailed_summary(
    *,
    file_id: str,
    detailed_summary: str,
    model: str,
    context_chars: int,
    was_truncated: bool,
) -> None:
    """Write the generated Markdown summary and transition to 'generated'."""
    now = datetime.now(UTC).isoformat()
    with get_search_db() as session:
        session.execute(
            sql_text(
                "UPDATE file_summaries SET "
                "detailed_summary = :detailed_summary, "
                "detailed_status = :status, "
                "detailed_model = :model, "
                "detailed_generated_at = :generated_at, "
                "detailed_context_chars = :context_chars, "
                "detailed_was_truncated = :was_truncated, "
                "detailed_error = NULL "
                "WHERE file_id = :fid"
            ),
            {
                "fid": file_id,
                "detailed_summary": detailed_summary,
                "status": DETAILED_STATUS_GENERATED,
                "model": model,
                "generated_at": now,
                "context_chars": context_chars,
                "was_truncated": 1 if was_truncated else 0,
            },
        )


def _get_detailed_summary(file_id: str) -> dict | None:
    """Fetch the detailed-summary record for a file.

    Returns a dict with the user-facing fields, or None if no detailed
    work has been started yet (status column NULL).
    """
    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT detailed_summary, detailed_status, detailed_model, "
                "detailed_generated_at, detailed_context_chars, "
                "detailed_was_truncated, detailed_error "
                "FROM file_summaries WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()
        if row is None or row[1] is None:
            return None
        return {
            "detailed_summary": row[0],
            "detailed_status": row[1],
            "detailed_model": row[2],
            "detailed_generated_at": row[3],
            "detailed_context_chars": row[4],
            "detailed_was_truncated": (
                bool(row[5]) if row[5] is not None else None
            ),
            "detailed_error": row[6],
        }


def _delete_detailed_summary(file_id: str) -> bool:
    """Clear all detailed-* columns for a file.

    Returns True when at least one row matched, False when the file had
    no summary row at all. The row itself is preserved when short/long
    are still present; if it has no other content, the row is deleted
    entirely so repeat generation starts from a clean slate.
    """
    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT short_summary, long_summary FROM file_summaries "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchone()
        if row is None:
            return False

        short_val = row[0] or ""
        long_val = row[1] or ""
        if short_val or long_val:
            session.execute(
                sql_text(
                    "UPDATE file_summaries SET "
                    "detailed_summary = NULL, "
                    "detailed_status = NULL, "
                    "detailed_model = NULL, "
                    "detailed_generated_at = NULL, "
                    "detailed_context_chars = NULL, "
                    "detailed_was_truncated = NULL, "
                    "detailed_error = NULL "
                    "WHERE file_id = :fid"
                ),
                {"fid": file_id},
            )
        else:
            # Placeholder row created by _set_detailed_status on a file
            # that never had short/long — drop it to return to the
            # pristine "no summary" state.
            session.execute(
                sql_text(
                    "DELETE FROM file_summaries WHERE file_id = :fid"
                ),
                {"fid": file_id},
            )
        return True


def classify_detailed_missing_reason(file_id: str) -> str:
    """Explain why no detailed summary exists for a file.

    Mirrors :func:`classify_missing_reason` for the detailed column so
    the router can tell the frontend why a "Generate" button would be
    a no-op. Return values match the short/long variant:

    * ``"file_not_found"``       — no indexed_files row
    * ``"unsupported_type"``     — image / archive / other
    * ``"insufficient_content"`` — supported type but below threshold
    * ``"not_generated"``        — ready to generate
    """
    indexed_file = _get_indexed_file(file_id)
    if indexed_file is None:
        return "file_not_found"

    context_type = _classify_file_type(
        indexed_file["file_type"], indexed_file.get("mime_type")
    )
    if context_type is None:
        return "unsupported_type"

    if _build_context(indexed_file, context_type) is None:
        return "insufficient_content"

    return "not_generated"


async def generate_detailed_summary(
    file_id: str,
    llm_client: LLMClient,
) -> None:
    """Generate a detailed Markdown summary for a single file.

    Intended to be scheduled via FastAPI BackgroundTasks from the router
    — runs synchronously against the configured LLM and takes tens of
    seconds to a few minutes for long transcripts on local ollama.

    Lifecycle:
    1. Policy gate: skip if ``features.detailed_summaries`` is off.
    2. LLM gate: skip if the client is disabled.
    3. Pre-checks via :func:`classify_detailed_missing_reason` — any
       non ``"not_generated"`` result is a silent skip (router has
       already rejected the request in that case, this is a defence).
    4. Status transitions: ``generating`` → ``generated`` or ``failed``
       (with the error message stored in ``detailed_error``).

    Raises nothing — all failure modes land in the ``failed`` row so
    the frontend polling surface is uniform. The background-task
    harness only sees a clean return.
    """
    if settings.features.detailed_summaries == "false":
        return

    if not llm_client.enabled:
        _set_detailed_status(
            file_id,
            DETAILED_STATUS_FAILED,
            error="LLM provider is disabled",
        )
        return

    indexed_file = _get_indexed_file(file_id)
    if indexed_file is None:
        # Router would have returned 404 already; defence in depth.
        return

    context_type = _classify_file_type(
        indexed_file["file_type"], indexed_file.get("mime_type")
    )
    if context_type not in _SUPPORTED_CONTEXT_TYPES:
        _set_detailed_status(
            file_id,
            DETAILED_STATUS_FAILED,
            error=f"Unsupported file type: {indexed_file['file_type']}",
        )
        return

    raw_context = _build_context(indexed_file, context_type)
    if not raw_context:
        _set_detailed_status(
            file_id,
            DETAILED_STATUS_FAILED,
            error="No usable content to summarize",
        )
        return

    prepared, was_truncated = _prepare_context(
        raw_context,
        max_chars=settings.summaries.detailed_max_context_chars,
        window_count=settings.summaries.detailed_window_count,
    )
    system_prompt = _build_detailed_system_prompt()
    user_prompt = _build_detailed_user_prompt(
        indexed_file, context_type, prepared, was_truncated
    )

    _set_detailed_status(
        file_id,
        DETAILED_STATUS_GENERATING,
        model=settings.llm.model,
    )

    try:
        raw = await llm_client.generate(
            system_prompt,
            user_prompt,
            max_tokens_override=_DETAILED_MAX_TOKENS,
        )
    except Exception as e:  # noqa: BLE001 - surface any LLM error uniformly
        logger.exception(
            "Detailed summary generation crashed for %s", file_id
        )
        _set_detailed_status(
            file_id,
            DETAILED_STATUS_FAILED,
            error=f"LLM error: {e}",
        )
        return

    if not raw or not raw.strip():
        _set_detailed_status(
            file_id,
            DETAILED_STATUS_FAILED,
            error="LLM returned empty output",
        )
        return

    _save_detailed_summary(
        file_id=file_id,
        detailed_summary=raw.strip(),
        model=settings.llm.model,
        context_chars=len(prepared),
        was_truncated=was_truncated,
    )
    logger.info(
        "Detailed summary: saved for %s (%s, %d chars, truncated=%s)",
        file_id, context_type, len(prepared), was_truncated,
    )


class SummariesWorker:
    """Async worker that processes summary generation requests via a queue."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def enqueue(self, file_id: str) -> None:
        """Add a file to the summaries queue.

        The queue is shared between the short/long path and the
        detailed-summary chain, so a file is accepted when either
        per-drive policy allows its respective feature. Drives that
        disable both features drop out here. Fails open on transient
        internal-API failure (`is_file_feature_enabled` returns True).
        """
        from app.policy_client import is_file_feature_enabled
        if await is_file_feature_enabled(file_id, "summaries"):
            await self._queue.put(file_id)
            return
        # Short/long blocked — still enqueue if detailed is enabled for
        # this drive so the detailed chain in _process_file can run.
        if (
            settings.features.detailed_summaries == "on_index"
            and await is_file_feature_enabled(file_id, "detailed_summaries")
        ):
            await self._queue.put(file_id)

    async def enqueue_unprocessed(self) -> int:
        """Find indexed files needing summary work and enqueue them.

        Walks two gaps in the search DB:
        - short/long: files with no ``file_summaries`` row at all
        - detailed:   files that have short/long but no detailed yet
          (``detailed_status IS NULL``)

        The short/long gap is only walked when
        ``features.summaries == "on_index"``; the detailed gap only
        when ``features.detailed_summaries == "on_index"``. Per-drive
        policy for each feature is consulted once per drive so
        opted-out drives don't generate work.

        Returns:
            Number of files queued (after policy filtering).
        """
        from app.policy_client import is_feature_enabled

        want_short = settings.features.summaries == "on_index"
        want_detailed = settings.features.detailed_summaries == "on_index"
        if not want_short and not want_detailed:
            return 0

        pending: list[tuple[str, str, str]] = []  # (file_id, drive, feature)
        with get_search_db() as session:
            if want_short:
                rows = session.execute(
                    sql_text(
                        "SELECT f.file_id, f.drive FROM indexed_files f "
                        "WHERE f.active = 1 AND f.metadata_indexed = 1 "
                        "AND f.file_type IN ('video', 'audio', 'document', 'text') "
                        "AND f.file_id NOT IN (SELECT file_id FROM file_summaries)"
                    )
                ).fetchall()
                pending.extend(
                    (file_id, drive, "summaries") for file_id, drive in rows
                )
            if want_detailed:
                rows = session.execute(
                    sql_text(
                        "SELECT f.file_id, f.drive FROM indexed_files f "
                        "WHERE f.active = 1 AND f.metadata_indexed = 1 "
                        "AND f.file_type IN ('video', 'audio', 'document', 'text') "
                        "AND f.file_id NOT IN ("
                        "  SELECT file_id FROM file_summaries "
                        "  WHERE detailed_status IS NOT NULL"
                        ")"
                    )
                ).fetchall()
                pending.extend(
                    (file_id, drive, "detailed_summaries")
                    for file_id, drive in rows
                )

        # De-dupe file_ids: a file may appear in both gaps. Keep the
        # first occurrence (short/long if both modes are on, detailed
        # otherwise) — _process_file handles each layer independently.
        seen: set[str] = set()
        allowed_cache: dict[tuple[str, str], bool] = {}
        count = 0
        for file_id, drive, feature in pending:
            if file_id in seen:
                continue
            key = (drive, feature)
            allowed = allowed_cache.get(key)
            if allowed is None:
                allowed = await is_feature_enabled(drive, feature)
                allowed_cache[key] = allowed
            if not allowed:
                continue
            await self._queue.put(file_id)
            seen.add(file_id)
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
        """Generate summaries for a single file.

        Handles both the short/long summary and (when configured) the
        detailed summary. Each layer has its own feature gate and
        existence check, so callers can enable them independently:

        - ``features.summaries``: ``"manual"`` or ``"on_index"`` runs the
          short/long generation when no file_summaries row exists yet
        - ``features.detailed_summaries``: only ``"on_index"`` triggers
          automatic detailed generation from this worker; ``"manual"``
          is handled by the router's BackgroundTasks route instead
        """
        if not self._llm_client.enabled:
            return

        want_short = (
            settings.features.summaries != "false"
            and not _has_summary(file_id)
        )
        want_detailed = (
            settings.features.detailed_summaries == "on_index"
            and not _has_detailed_summary(file_id)
        )
        if not want_short and not want_detailed:
            return

        indexed_file = _get_indexed_file(file_id)
        if indexed_file is None:
            return

        context_type = _classify_file_type(
            indexed_file["file_type"], indexed_file.get("mime_type")
        )
        if context_type not in _SUPPORTED_CONTEXT_TYPES:
            return

        raw_context = _build_context(indexed_file, context_type)
        if not raw_context:
            logger.debug(
                "Summaries: no context available for %s (%s)",
                file_id, context_type,
            )
            return

        if want_short:
            await self._generate_short_long(
                file_id, indexed_file, context_type, raw_context
            )

        if want_detailed:
            # Per-drive policy for "detailed_summaries" gates independently
            # of "summaries" so operators can opt individual drives in/out
            # without disabling the short/long path.
            from app.policy_client import is_file_feature_enabled
            if await is_file_feature_enabled(file_id, "detailed_summaries"):
                await generate_detailed_summary(file_id, self._llm_client)

    async def _generate_short_long(
        self,
        file_id: str,
        indexed_file: dict,
        context_type: str,
        raw_context: str,
    ) -> None:
        """Run the short/long generation path and persist the result."""
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
