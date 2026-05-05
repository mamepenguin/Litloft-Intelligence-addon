"""SQLAlchemy models for the semantic search index database.

These models track indexed files, their embeddings, and transcript chunks.
The search DB is separate from Litloft's main SQLite database.
"""

import secrets
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def generate_insight_id() -> str:
    """12-char URL-safe ID for FileInsight rows (matches core File.id width)."""
    return secrets.token_urlsafe(9)[:12]


class IndexedFile(Base):
    """Tracks files that have been indexed for semantic search.

    Maps 1:1 with Litloft's File table via file_id.
    The active flag supports soft-delete synchronization.
    """

    __tablename__ = "indexed_files"

    file_id: Mapped[str] = mapped_column(String(12), primary_key=True)
    drive: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    file_type: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str] = mapped_column(String, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Projection of core File.thumbnail_path. Read by the CLIP worker
    # to embed the representative thumbnail (`embedding_type="clip_thumbnail"`)
    # without a per-file Internal API roundtrip. May be NULL when core
    # has not generated/downloaded a thumbnail yet (e.g. legacy `.loft`
    # rows from before media_import Phase 2).
    thumbnail_path: Mapped[str | None] = mapped_column(String, nullable=True)

    # Index state flags
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="1"
    )
    metadata_indexed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    clip_indexed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    # Tracks completion of `embedding_type="clip_thumbnail"` indexing
    # (the "1 representative frame" route, distinct from `clip_indexed`
    # which covers scene-detected video frames). Spec
    # 2026-05-02-thumbnail-clip-default-shallow-search.md.
    clip_thumbnail_indexed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    whisper_indexed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    text_indexed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )

    # Metadata snapshot (for keyword matching)
    title: Mapped[str] = mapped_column(String, nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tags_text: Mapped[str] = mapped_column(Text, nullable=False, default="")

    indexed_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index("idx_indexed_files_drive", "drive"),
        Index("idx_indexed_files_active", "active"),
        Index("idx_indexed_files_file_type", "file_type"),
        Index("idx_indexed_files_metadata_indexed", "metadata_indexed"),
        Index("idx_indexed_files_clip_indexed", "clip_indexed"),
        Index(
            "idx_indexed_files_clip_thumbnail_indexed",
            "clip_thumbnail_indexed",
        ),
        Index("idx_indexed_files_whisper_indexed", "whisper_indexed"),
        Index("idx_indexed_files_text_indexed", "text_indexed"),
    )


class Embedding(Base):
    """Stores embedding references for vector search.

    Each embedding links to a vector in the sqlite-vec virtual tables.
    The embedding_type determines which vec table to query.
    """

    __tablename__ = "embeddings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    file_id: Mapped[str] = mapped_column(String(12), nullable=False, index=True)

    # "metadata", "clip", "whisper", "text_content"
    embedding_type: Mapped[str] = mapped_column(String(20), nullable=False)

    # For time-based segments (clip frames, whisper chunks)
    timestamp_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp_end: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Human-readable content snippet for search result display
    content_preview: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # "text" or "clip" — determines which vec table to query
    vector_table: Mapped[str] = mapped_column(String(10), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("idx_embeddings_file_type", "file_id", "embedding_type"),
        Index("idx_embeddings_vector_table", "vector_table"),
    )


class TranscriptChunk(Base):
    """Stores Whisper transcription results as timestamped text chunks.

    These are separate from embeddings to allow full-text display
    and re-embedding if the text model changes.
    """

    __tablename__ = "transcript_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_id: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="")

    timestamp_start: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp_end: Mapped[float] = mapped_column(Float, nullable=False)

    # Non-null marks the chunk as AI-refined.
    text_refined_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("idx_transcript_file_chunk", "file_id", "chunk_index"),
    )


class TranscriptWord(Base):
    """Stores Whisper word-level timestamps for subtitle rendering and precise seek.

    Complements TranscriptChunk (which is optimised for embedding/search at
    10–30 s granularity). Word rows are the fine-grained source of truth
    used to build subtitles (1–5 s cues) and to seek inside a segment hit.

    Populated only when Whisper runs against the file. LoftRef files (which
    derive chunks from adjacent .vtt) do not produce word rows because
    WebVTT cues rarely carry word-level timing.
    """

    __tablename__ = "transcript_words"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_id: Mapped[str] = mapped_column(String(12), nullable=False, index=True)

    text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="")

    timestamp_start: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp_end: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("idx_transcript_words_file_ts", "file_id", "timestamp_start"),
    )


class FileInsight(Base):
    """Append-only history of AI-generated artefacts for a file.

    Each row is one Transformation output event (e.g. detailed_summary
    generation, user edit, revert, regenerate). Exactly one row per
    (file_id, kind) is ``status='active'`` at a time; older rows move
    to ``'superseded'``. See hako vdPrMz0_adP-3C6Ogkjds for the role
    definition — this table is the AI-output log layer, not a generic
    metadata change history.

    Lifecycle hooks:
    - ``files.purged`` webhook → rows for that file_id are deleted via
      ``_purge_file`` (same pattern as ``file_summaries`` /
      ``transcript_chunks``; no cross-DB FK).
    - Regenerate (``_delete_detailed_summary``) → rows for (file_id,
      kind) are deleted outright; history is discarded along with the
      body because the user explicitly asked for a clean slate.
    - Drive purge (policy flip) → inherits the per-file purge loop in
      ``purge_drive``.
    """

    __tablename__ = "file_insights"

    id: Mapped[str] = mapped_column(
        String(12), primary_key=True, default=generate_insight_id
    )
    file_id: Mapped[str] = mapped_column(String(12), nullable=False)

    # "detailed_summary" in Step 1. Reserved for future kinds:
    # "auto_tags", "key_questions", etc. Free-form string so adding a
    # new kind does not require a migration.
    kind: Mapped[str] = mapped_column(String(32), nullable=False)

    # Raw LLM output. Markdown for summaries, JSON string for tag
    # lists (when auto_tags migrates). Frontend / consumer decides how
    # to render based on ``kind``.
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # JSON-encoded generation metadata: model, context_chars,
    # was_truncated, edited_at (for manual edits), reverted_from_manual
    # (for reverts), prompt_version (future). Never structured into
    # columns — the shape is kind-dependent and we do not query on it.
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # "active" | "superseded" | "invalidated"
    # Step 1 uses only active/superseded. ``invalidated_at`` + status
    # "invalidated" is reserved for chain-invalidation (transcript
    # refine → dependent insights) in a later step.
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="active"
    )

    # "intelligence" — generated by summaries worker or similar LLM run
    # "manual"       — user-edited content saved through edit endpoint
    # "knowledge"    — reserved for Knowledge-originated insights (future)
    created_by: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    invalidated_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )

    __table_args__ = (
        Index(
            "idx_file_insights_file_kind_status",
            "file_id", "kind", "status",
        ),
        Index("idx_file_insights_kind_status", "kind", "status"),
    )


class PickupCache(Base):
    """Precomputed recommendation cache for the pickup dashboard widget.

    One row per (drive_id, viewer_id) pair. The background worker refreshes
    the row whenever WatchHistory changes (detected via checkpoint hash).
    File IDs are stored as a JSON array; the pickup endpoint returns them
    directly without additional DB lookups.
    """

    __tablename__ = "pickup_cache"

    drive_id: Mapped[str] = mapped_column(String, primary_key=True)
    viewer_id: Mapped[str] = mapped_column(String(16), primary_key=True)

    # JSON array of file_id strings (up to 12 items, ordered by relevance)
    file_ids: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    computed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # MD5 of the last-N watched file_ids — used to skip recomputation when
    # the viewer's history has not changed since the previous run.
    watch_history_checkpoint: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )


class PdfMarkdown(Base):
    """Markdown rendering of a PDF file produced by PyMuPDF4LLM.

    One row per indexed PDF. Generated during the text-content indexing
    step alongside chunk embeddings; exposed through a read-only API so
    the frontend can show a structured Markdown view of the document.

    ``extractor`` records which path produced the body:
    - ``"pymupdf4llm"``  Markdown-rich primary path.
    - ``"fitz_fallback"``  fitz raw-text fallback after PyMuPDF4LLM
      raised. The fallback is plain text rather than Markdown, so the
      indexing pipeline writes ``markdown=None`` (no row inserted) in
      that case. The literal is reserved here for forward compatibility
      and external readers that may persist plain-text bodies.

    The row is removed automatically when the corresponding
    ``indexed_files`` row goes away (FK ON DELETE CASCADE).
    """

    __tablename__ = "pdf_markdown"

    file_id: Mapped[str] = mapped_column(
        String(12),
        ForeignKey("indexed_files.file_id", ondelete="CASCADE"),
        primary_key=True,
    )
    markdown: Mapped[str] = mapped_column(Text, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    extractor: Mapped[str] = mapped_column(String(32), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
