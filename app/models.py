"""SQLAlchemy models for the semantic search index database.

These models track indexed files, their embeddings, and transcript chunks.
The search DB is separate from HomeVault's main SQLite database.
"""

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class IndexedFile(Base):
    """Tracks files that have been indexed for semantic search.

    Maps 1:1 with HomeVault's File table via file_id.
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

    Populated only when Whisper runs against the file. HvLink files (which
    derive chunks from adjacent .vtt) do not produce word rows because
    WebVTT cues rarely carry word-level timing.
    """

    __tablename__ = "transcript_words"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_id: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    word_index: Mapped[int] = mapped_column(Integer, nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="")

    timestamp_start: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp_end: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("idx_transcript_words_file_idx", "file_id", "word_index"),
        Index("idx_transcript_words_file_time", "file_id", "timestamp_start"),
    )


class SimilarCache(Base):
    """Cache for similar files search results.

    Stores serialized find_similar() results to avoid expensive
    recomputation (vector search + TF-IDF) on repeated requests.
    Invalidated in bulk on index update events (webhooks).
    """

    __tablename__ = "similar_cache"

    cache_key: Mapped[str] = mapped_column(String, primary_key=True)
    # cache_key format: "{file_id}:{limit}:{drive or '_'}"

    file_id: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
