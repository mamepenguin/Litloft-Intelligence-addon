"""Auto-tagging worker using LLM.

Processes files through an LLM to suggest tags based on file
metadata, transcripts, captions, and text content. Runs as a
dedicated async queue processing one file at a time.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime

from sqlalchemy import text as sql_text

from app.config import settings
from app.database import get_homevault_db, get_search_db
from app.llm import LLMClient
from app.models import Embedding, IndexedFile, TranscriptChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "あなたはファイル管理システムのタグ付けアシスタントです。\n"
    "ファイルの内容に基づいて、検索に役立つタグを5-10個提案してください。\n"
    "\n"
    "規則:\n"
    "- JSON配列で返すこと: [\"tag1\", \"tag2\", ...]\n"
    "- 既存タグと重複しないこと\n"
    "- 具体的で検索に有用なタグにすること\n"
    "- ファイルの内容を要約するタグを含めること\n"
    "- タグは短く（1-3語）\n"
    "- JSON配列のみ返し、他のテキストは含めないこと"
)

_MAX_CONTEXT_CHARS = 2000


class AutoTagsWorker:
    """Async worker that processes auto-tagging requests via a queue."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def enqueue(self, file_id: str) -> None:
        """Add a file to the auto-tagging queue.

        Args:
            file_id: The file ID to process.
        """
        await self._queue.put(file_id)

    async def enqueue_unprocessed(self) -> int:
        """Find indexed files without suggested tags and enqueue them.

        Returns:
            Number of files queued.
        """
        with get_search_db() as session:
            rows = session.execute(
                sql_text(
                    "SELECT f.file_id FROM indexed_files f "
                    "WHERE f.active = 1 AND f.metadata_indexed = 1 "
                    "AND f.file_id NOT IN (SELECT file_id FROM suggested_tags)"
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
                logger.error("Auto-tags worker error: %s", e)

    async def _process_file(self, file_id: str) -> None:
        """Process a single file for auto-tagging.

        Checks preconditions, builds context, queries LLM,
        and saves the result.

        Args:
            file_id: The file ID to process.
        """
        if not settings.features.auto_tags:
            return

        if not self._llm_client.enabled:
            return

        # Check if already processed
        if _has_suggested_tags(file_id):
            return

        # Get indexed file info
        indexed_file = _get_indexed_file(file_id)
        if indexed_file is None:
            return

        # Get existing tags from HomeVault DB
        existing_tags = _get_existing_tags(file_id)

        # Build context based on file type
        context_type = _classify_file_type(indexed_file["file_type"])
        context = _build_context(indexed_file, context_type, existing_tags)

        # Build user prompt
        user_prompt = _build_user_prompt(indexed_file, context_type, context, existing_tags)

        # Query LLM
        tags = await self._llm_client.generate_json(_SYSTEM_PROMPT, user_prompt)

        if not isinstance(tags, list):
            logger.warning(
                "Auto-tags LLM returned non-list for %s, skipping", file_id
            )
            return

        # Filter to strings only and deduplicate against existing
        existing_lower = {t.lower() for t in existing_tags}
        filtered_tags = [
            t for t in tags
            if isinstance(t, str) and t.strip() and t.lower() not in existing_lower
        ]

        if not filtered_tags:
            logger.info("Auto-tags: no new tags for %s after filtering", file_id)
            return

        # Save to database
        _save_suggested_tags(
            file_id=file_id,
            tags=filtered_tags,
            model=settings.llm.model,
            context_type=context_type,
        )
        logger.info(
            "Auto-tags: saved %d suggested tags for %s",
            len(filtered_tags), file_id,
        )


def _has_suggested_tags(file_id: str) -> bool:
    """Check if suggested tags already exist for a file."""
    with get_search_db() as session:
        row = session.execute(
            sql_text("SELECT 1 FROM suggested_tags WHERE file_id = :fid"),
            {"fid": file_id},
        ).fetchone()
        return row is not None


def _get_indexed_file(file_id: str) -> dict | None:
    """Get indexed file info from search DB."""
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
            "tags_text": f.tags_text,
        }


def _get_existing_tags(file_id: str) -> list[str]:
    """Get existing tags from HomeVault DB for a file."""
    try:
        with get_homevault_db() as session:
            rows = session.execute(
                sql_text(
                    "SELECT t.name FROM tags t "
                    "JOIN file_tags ft ON t.id = ft.tag_id "
                    "WHERE ft.file_id = :file_id"
                ),
                {"file_id": file_id},
            ).fetchall()
            return [row[0] for row in rows]
    except Exception:
        return []


def _classify_file_type(file_type: str) -> str:
    """Classify file type into context categories."""
    if file_type == "video":
        return "video"
    if file_type == "audio":
        return "audio"
    if file_type == "image":
        return "image"
    if file_type in ("document", "text"):
        return "document"
    return "other"


def _build_context(
    indexed_file: dict, context_type: str, existing_tags: list[str]
) -> str:
    """Build additional context based on file type.

    Args:
        indexed_file: File info dict.
        context_type: Classified file type.
        existing_tags: List of existing tag names.

    Returns:
        Additional context string.
    """
    file_id = indexed_file["file_id"]
    parts: list[str] = []

    if context_type in ("video", "audio"):
        # Get transcript chunks
        transcript = _get_transcript_text(file_id)
        if transcript:
            parts = [*parts, f"Transcript:\n{transcript}"]

    elif context_type == "image":
        # Get BLIP captions if available
        captions = _get_blip_captions(file_id)
        if captions:
            parts = [*parts, f"Image captions:\n{captions}"]

    elif context_type == "document":
        # Get text content chunks
        text_content = _get_text_content(file_id)
        if text_content:
            parts = [*parts, f"Content:\n{text_content}"]

    return "\n".join(parts)


def _get_transcript_text(file_id: str) -> str:
    """Get transcript text for a file, truncated to max context chars."""
    with get_search_db() as session:
        chunks = (
            session.query(TranscriptChunk)
            .filter(TranscriptChunk.file_id == file_id)
            .order_by(TranscriptChunk.chunk_index)
            .all()
        )
        if not chunks:
            return ""
        text = " ".join(c.text for c in chunks)
        return text[:_MAX_CONTEXT_CHARS]


def _get_blip_captions(file_id: str) -> str:
    """Get BLIP captions from embedding content_preview."""
    with get_search_db() as session:
        embeddings = (
            session.query(Embedding)
            .filter(
                Embedding.file_id == file_id,
                Embedding.embedding_type == "blip_caption",
            )
            .all()
        )
        if not embeddings:
            return ""
        return " ".join(e.content_preview for e in embeddings if e.content_preview)


def _get_text_content(file_id: str) -> str:
    """Get text content chunks for a file, truncated to max context chars."""
    with get_search_db() as session:
        embeddings = (
            session.query(Embedding)
            .filter(
                Embedding.file_id == file_id,
                Embedding.embedding_type == "text_content",
            )
            .order_by(Embedding.timestamp_start.asc().nullsfirst())
            .all()
        )
        if not embeddings:
            return ""
        text = " ".join(e.content_preview for e in embeddings if e.content_preview)
        return text[:_MAX_CONTEXT_CHARS]


def _build_user_prompt(
    indexed_file: dict,
    context_type: str,
    context: str,
    existing_tags: list[str],
) -> str:
    """Build the user prompt for LLM tag generation.

    Args:
        indexed_file: File info dict.
        context_type: Classified file type.
        context: Additional context string.
        existing_tags: List of existing tag names.

    Returns:
        Formatted user prompt.
    """
    parts: list[str] = [
        f"ファイル名: {indexed_file['filename']}",
        f"タイプ: {context_type}",
    ]

    # Add metadata if available
    if indexed_file["title"] and indexed_file["title"] != indexed_file["filename"]:
        parts = [*parts, f"タイトル: {indexed_file['title']}"]
    if indexed_file["description"]:
        parts = [*parts, f"説明: {indexed_file['description']}"]
    if indexed_file["tags_text"]:
        parts = [*parts, f"メタデータタグ: {indexed_file['tags_text']}"]

    if context:
        parts = [*parts, context]

    tags_display = ", ".join(existing_tags) if existing_tags else "なし"
    parts = [*parts, f"既存タグ: {tags_display}"]

    return "\n".join(parts)


def _save_suggested_tags(
    file_id: str,
    tags: list[str],
    model: str,
    context_type: str,
) -> None:
    """Save suggested tags to the search database.

    Args:
        file_id: The file ID.
        tags: List of suggested tag strings.
        model: LLM model name used.
        context_type: File type classification.
    """
    now = datetime.now(UTC).isoformat()
    tags_json = json.dumps(tags, ensure_ascii=False)

    with get_search_db() as session:
        session.execute(
            sql_text(
                "INSERT OR REPLACE INTO suggested_tags "
                "(file_id, tags, model, context_type, created_at, status) "
                "VALUES (:file_id, :tags, :model, :context_type, :created_at, 'pending')"
            ),
            {
                "file_id": file_id,
                "tags": tags_json,
                "model": model,
                "context_type": context_type,
                "created_at": now,
            },
        )
