"""Metadata embedding worker.

Generates text embeddings from file metadata (name, tags, description)
and text content extracted from documents. This is the fastest indexing
step and runs in batch mode.
"""

import logging
import uuid
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_search_db, get_search_engine, validate_vector_table
from app.extractors.base import TextChunk
from app.extractors.pdf import PdfExtractor
from app.extractors.text import TextExtractor
from app.models import Embedding, IndexedFile
from app.workers.embedder import embed_passages

logger = logging.getLogger(__name__)

# Content extractors (instantiated once)
_extractors = [TextExtractor(), PdfExtractor()]


def _build_metadata_text(file: IndexedFile) -> str:
    """Build a single text string from file metadata for embedding.

    Args:
        file: The indexed file record.

    Returns:
        Combined metadata text.
    """
    parts: list[str] = []

    if file.filename:
        parts = [*parts, file.filename]
    if file.title and file.title != file.filename:
        parts = [*parts, file.title]
    if file.description:
        parts = [*parts, file.description]
    if file.tags_text:
        parts = [*parts, file.tags_text]

    return " ".join(parts)


def _extract_file_content(file: IndexedFile) -> list[TextChunk]:
    """Extract text content from a file using available extractors.

    Args:
        file: The indexed file record.

    Returns:
        List of extracted text chunks.
    """
    for extractor in _extractors:
        if extractor.can_handle(file.file_path):
            try:
                return extractor.extract(file.file_path)
            except Exception as e:
                logger.warning(
                    "Content extraction failed for %s: %s",
                    file.file_id, e,
                )
    return []


def _store_embedding(
    session: Session,
    embedding_id: str,
    file_id: str,
    embedding_type: str,
    vector_table: str,
    vector: Any,
    content_preview: str = "",
    timestamp_start: float | None = None,
    timestamp_end: float | None = None,
) -> None:
    """Store an embedding record and its vector in the vec table.

    Args:
        session: Database session.
        embedding_id: Unique ID for this embedding.
        file_id: The file this embedding belongs to.
        embedding_type: Type of embedding (metadata, text_content, etc).
        vector_table: Which vec table to store in (vec_text, vec_clip).
        vector: The embedding vector as numpy array.
        content_preview: Human-readable snippet for display.
        timestamp_start: Optional start timestamp.
        timestamp_end: Optional end timestamp.
    """
    embedding_record = Embedding(
        id=embedding_id,
        file_id=file_id,
        embedding_type=embedding_type,
        vector_table=vector_table,
        content_preview=content_preview[:500],
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
    )
    session.add(embedding_record)
    session.flush()

    # Insert vector into sqlite-vec virtual table
    engine = get_search_engine()
    vec_bytes = vector.tobytes()
    with engine.connect() as conn:
        table = validate_vector_table(vector_table)
        conn.execute(
            sql_text(f"INSERT INTO {table}(embedding_id, vector) VALUES(:id, :vec)"),
            {"id": embedding_id, "vec": vec_bytes},
        )
        conn.commit()


def index_metadata_batch(file_ids: list[str]) -> int:
    """Index metadata embeddings for a batch of files.

    Args:
        file_ids: List of file IDs to index.

    Returns:
        Number of files successfully indexed.
    """
    if not file_ids:
        return 0

    indexed_count = 0

    with get_search_db() as session:
        files = (
            session.query(IndexedFile)
            .filter(
                IndexedFile.file_id.in_(file_ids),
                IndexedFile.active.is_(True),
            )
            .all()
        )

        if not files:
            return 0

        # Build metadata texts for batch embedding
        metadata_texts: list[str] = []
        valid_files: list[IndexedFile] = []

        for file in files:
            meta_text = _build_metadata_text(file)
            if meta_text.strip():
                metadata_texts = [*metadata_texts, meta_text]
                valid_files = [*valid_files, file]

        if not metadata_texts:
            return 0

        try:
            vectors = embed_passages(metadata_texts)
        except Exception as e:
            logger.error("Metadata embedding batch failed: %s", e)
            return 0

        for file, vector in zip(valid_files, vectors):
            try:
                embedding_id = f"meta_{file.file_id}_{uuid.uuid4().hex[:8]}"

                # Remove old metadata embeddings for this file
                _remove_embeddings(session, file.file_id, "metadata")

                _store_embedding(
                    session=session,
                    embedding_id=embedding_id,
                    file_id=file.file_id,
                    embedding_type="metadata",
                    vector_table="vec_text",
                    vector=vector,
                    content_preview=metadata_texts[valid_files.index(file)][:200],
                )

                file.metadata_indexed = True
                indexed_count += 1

            except Exception as e:
                logger.error(
                    "Failed to store metadata embedding for %s: %s",
                    file.file_id, e,
                )

    return indexed_count


def index_text_content(file_id: str) -> bool:
    """Index text content from a document file.

    Extracts text using the appropriate extractor and creates
    embeddings for each chunk.

    Args:
        file_id: The file ID to index.

    Returns:
        True if indexing succeeded.
    """
    with get_search_db() as session:
        file = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()

        if file is None:
            return False

        chunks = _extract_file_content(file)
        if not chunks:
            file.text_indexed = True
            return True

        chunk_texts = [c.text for c in chunks]

        try:
            vectors = embed_passages(chunk_texts)
        except Exception as e:
            logger.error("Text content embedding failed for %s: %s", file_id, e)
            return False

        # Remove old text content embeddings
        _remove_embeddings(session, file_id, "text_content")

        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            try:
                embedding_id = f"txt_{file_id}_{idx}_{uuid.uuid4().hex[:8]}"
                page_info = f" (page {chunk.page})" if chunk.page is not None else ""

                _store_embedding(
                    session=session,
                    embedding_id=embedding_id,
                    file_id=file_id,
                    embedding_type="text_content",
                    vector_table="vec_text",
                    vector=vector,
                    content_preview=f"{chunk.text[:200]}{page_info}",
                )
            except Exception as e:
                logger.error(
                    "Failed to store text embedding %d for %s: %s",
                    idx, file_id, e,
                )

        file.text_indexed = True
        return True


def _remove_embeddings(
    session: Session, file_id: str, embedding_type: str
) -> None:
    """Remove existing embeddings for a file and type.

    Args:
        session: Database session.
        file_id: The file ID.
        embedding_type: The embedding type to remove.
    """
    existing = (
        session.query(Embedding)
        .filter_by(file_id=file_id, embedding_type=embedding_type)
        .all()
    )

    if not existing:
        return

    engine = get_search_engine()
    with engine.connect() as conn:
        for emb in existing:
            table = validate_vector_table(emb.vector_table)
            conn.execute(
                sql_text(f"DELETE FROM {table} WHERE embedding_id = :id"),
                {"id": emb.id},
            )
        conn.commit()

    for emb in existing:
        session.delete(emb)
    session.flush()
