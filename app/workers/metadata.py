"""Metadata embedding worker.

Generates text embeddings from file metadata (name, tags, description)
and text content extracted from documents. This is the fastest indexing
step and runs in batch mode.
"""

import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_search_db, upsert_fts_text_content, validate_vector_table
from app.extractors.base import ExtractionResult
from app.extractors.html import HtmlExtractor
from app.extractors.office import OfficeExtractor
from app.extractors.pdf import PdfExtractor
from app.extractors.text import TextExtractor
from app.models import Embedding, IndexedFile
from app.workers.embedder import embed_passages

logger = logging.getLogger(__name__)

# Content extractors (instantiated once)
_extractors = [TextExtractor(), PdfExtractor(), OfficeExtractor(), HtmlExtractor()]

# Cap for the persisted PDF Markdown body. Anything larger is dropped
# from the ``pdf_markdown`` table (chunks/embeddings are still stored).
# See spec ``2026-04-27-intelligence-pdf-markdown-indexing.md``.
MAX_PDF_MARKDOWN_BYTES = 5 * 1024 * 1024


def _clean_filename(filename: str) -> str:
    """Convert a filename into natural text for embedding.

    Strips extension, replaces separators with spaces, and collapses
    whitespace so the embedding model processes it as natural language.

    Args:
        filename: Original filename (e.g. "Piano_Recital_2024.mp4").

    Returns:
        Cleaned text (e.g. "Piano Recital 2024").
    """
    stem = PurePosixPath(filename).stem
    cleaned = re.sub(r"[_\-\.]+", " ", stem)
    return re.sub(r"\s+", " ", cleaned).strip()


def _build_metadata_text(
    file: IndexedFile,
    long_summary: str | None = None,
    visual_description: str | None = None,
) -> str:
    """Build a single text string from file metadata for embedding.

    The embedding text is the file's "topic surface" — what the
    hierarchical RAG Stage 1 coarse retrieval matches against. We
    deliberately include AI-generated descriptive text (long summary
    for video / audio / document, visual description for image) here
    too so the summary-derived domain vocabulary can steer chunk
    retrieval, while the final-generation LLM context still pulls only
    original chunks (citation source invariant unchanged). See spec
    ``2026-04-26-intelligence-ask-hierarchical-retrieval.md`` (Approach B).

    ``long_summary`` and ``visual_description`` are mutually exclusive
    in practice — the summaries worker only fires for context_types
    video/audio/document, the vision worker only for images — but we
    accept both as independent ``Optional`` params so the caller does
    not have to know which is which.

    Args:
        file: The indexed file record.
        long_summary: Optional 3-5 sentence AI summary text for
            transcribable / textual files. Pass ``None`` when no
            summary exists OR the row has ``status='hidden'`` — both
            must omit the summary so the user's opt-out is respected.
        visual_description: Optional structured AI description of an
            image's visual content. Pass ``None`` for non-image files
            or when the description has not been generated / is in
            non-success status.

    Returns:
        Combined metadata text.
    """
    parts: list[str] = []

    if file.filename:
        parts = [*parts, _clean_filename(file.filename)]
    if file.title and file.title != file.filename:
        parts = [*parts, file.title]
    if file.description:
        parts = [*parts, file.description]
    if file.tags_text:
        parts = [*parts, file.tags_text]
    if long_summary:
        parts = [*parts, long_summary]
    if visual_description:
        parts = [*parts, visual_description]

    return " ".join(parts)


def _extract_file_content(file: IndexedFile) -> ExtractionResult:
    """Extract text content from a file using available extractors.

    Returns the full ``ExtractionResult`` so the indexing pipeline can
    persist the optional Markdown rendering (PDF) alongside the chunk
    embeddings. Files with no matching extractor — or that raise during
    extraction — yield an empty result.

    Args:
        file: The indexed file record.

    Returns:
        ExtractionResult with chunks plus any Markdown rendering.
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
                return ExtractionResult()
    return ExtractionResult()


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
    page: int | None = None,
    chunk_index: int | None = None,
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
        page: Optional one-based document page.
        chunk_index: Optional index of the chunk within the file, the
            join key into ``fts_text_content``.
    """
    embedding_record = Embedding(
        id=embedding_id,
        file_id=file_id,
        embedding_type=embedding_type,
        vector_table=vector_table,
        content_preview=content_preview[:500],
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        page=page,
        chunk_index=chunk_index,
    )
    session.add(embedding_record)
    session.flush()

    # Use session's connection (not a separate engine.connect()) to avoid
    # dual-connection conflicts with SQLite's single-writer constraint.
    table = validate_vector_table(vector_table)
    vec_bytes = vector.tobytes()
    session.execute(
        sql_text(f"INSERT INTO {table}(embedding_id, vector) VALUES(:id, :vec)"),
        {"id": embedding_id, "vec": vec_bytes},
    )


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

    # --- Phase 1: Read file info (short DB access) ---
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

        # Bulk-fetch generated summaries + visual descriptions so the
        # per-file builder doesn't incur an extra round trip. The two
        # fields live on the same row in ``file_summaries`` but are
        # gated by independent statuses:
        #   - ``status='hidden'`` (user opted out) → omit long_summary
        #   - ``visual_description_status != 'success'`` → omit visual
        # Both filters live in the SQL so the per-file builder never
        # sees opted-out / pending / failed text.
        active_ids = [f.file_id for f in files]
        summary_map: dict[str, str] = {}
        vision_map: dict[str, str] = {}
        if active_ids:
            placeholders = ",".join(f":id{i}" for i in range(len(active_ids)))
            params = {f"id{i}": fid for i, fid in enumerate(active_ids)}
            rows = session.execute(
                sql_text(
                    "SELECT file_id, "
                    "  CASE WHEN status = 'generated' THEN long_summary END, "
                    "  CASE WHEN visual_description_status = 'success' "
                    "    THEN visual_description END "
                    "FROM file_summaries "
                    f"WHERE file_id IN ({placeholders})"
                ),
                params,
            ).fetchall()
            for fid, long_s, vis_s in rows:
                if long_s:
                    summary_map[fid] = long_s
                if vis_s:
                    vision_map[fid] = vis_s

        # Build metadata texts for batch embedding
        metadata_texts: list[str] = []
        valid_file_ids: list[str] = []

        for file in files:
            meta_text = _build_metadata_text(
                file,
                long_summary=summary_map.get(file.file_id),
                visual_description=vision_map.get(file.file_id),
            )
            if meta_text.strip():
                metadata_texts = [*metadata_texts, meta_text]
                valid_file_ids = [*valid_file_ids, file.file_id]

    if not metadata_texts:
        return 0

    # --- Phase 2: Compute embeddings (no DB lock, may be slow) ---
    try:
        vectors = embed_passages(metadata_texts)
    except Exception as e:
        logger.error("Metadata embedding batch failed: %s", e)
        return 0

    # --- Phase 3: Write results to DB (short transaction) ---
    with get_search_db() as session:
        for file_id_val, vector, meta_text in zip(
            valid_file_ids, vectors, metadata_texts
        ):
            try:
                embedding_id = f"meta_{file_id_val}_{uuid.uuid4().hex[:8]}"

                # Remove old metadata embeddings for this file
                _remove_embeddings(session, file_id_val, "metadata")

                _store_embedding(
                    session=session,
                    embedding_id=embedding_id,
                    file_id=file_id_val,
                    embedding_type="metadata",
                    vector_table="vec_text",
                    vector=vector,
                    content_preview=meta_text[:200],
                )

                file = session.query(IndexedFile).filter_by(
                    file_id=file_id_val
                ).first()
                if file is not None:
                    file.metadata_indexed = True
                indexed_count += 1

            except Exception as e:
                logger.error(
                    "Failed to store metadata embedding for %s: %s",
                    file_id_val, e,
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
    # --- Phase 1: Read file info (short DB access) ---
    with get_search_db() as session:
        file = session.query(IndexedFile).filter_by(
            file_id=file_id, active=True
        ).first()

        if file is None:
            return False

        result = _extract_file_content(file)
        chunks = list(result.chunks)
        if not chunks:
            # No chunks: still call _upsert_pdf_markdown so an empty
            # PyMuPDF4LLM extraction (markdown="") clears any stale
            # row from a prior re-index. markdown=None (fitz fallback /
            # non-PDF) is a no-op inside the helper.
            _upsert_pdf_markdown(session, file_id, result)
            file.text_indexed = True
            return True

    # --- Phase 2: Compute embeddings (no DB lock, may be slow) ---
    chunk_texts = [c.text for c in chunks]

    try:
        vectors = embed_passages(chunk_texts)
    except Exception as e:
        logger.error("Text content embedding failed for %s: %s", file_id, e)
        return False

    # --- Phase 3: Write results to DB (short transaction) ---
    with get_search_db() as session:
        # Remove old text content embeddings
        _remove_embeddings(session, file_id, "text_content")

        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            try:
                embedding_id = f"txt_{file_id}_{idx}_{uuid.uuid4().hex[:8]}"
                _store_embedding(
                    session=session,
                    embedding_id=embedding_id,
                    file_id=file_id,
                    embedding_type="text_content",
                    vector_table="vec_text",
                    vector=vector,
                    content_preview=chunk.text[:200],
                    page=chunk.page,
                    chunk_index=idx,
                )
            except Exception as e:
                logger.error(
                    "Failed to store text embedding %d for %s: %s",
                    idx, file_id, e,
                )

        # Write full text chunks to FTS5 for keyword search
        fts_chunks = [
            {
                "chunk_index": idx,
                "page": chunk.page,
                "text": chunk.text,
            }
            for idx, chunk in enumerate(chunks)
        ]
        upsert_fts_text_content(session, file_id, fts_chunks)

        _upsert_pdf_markdown(session, file_id, result)

        file_record = session.query(IndexedFile).filter_by(
            file_id=file_id
        ).first()
        if file_record is not None:
            file_record.text_indexed = True
        return True


def _upsert_pdf_markdown(
    session: Session, file_id: str, result: ExtractionResult
) -> None:
    """Persist (or skip / clear) the PDF Markdown body for ``file_id``.

    - ``markdown is None``: extractor produced no Markdown (non-PDF
      files, or PDFs that fell back to fitz raw text). Skip silently.
    - ``markdown == ""``: PyMuPDF4LLM ran successfully but the document
      has no extractable pages (e.g. scan-only PDF). Drop any prior
      row so the DB reflects the latest extraction state instead of
      keeping stale Markdown.
    - body exceeds ``MAX_PDF_MARKDOWN_BYTES``: log a warning and skip
      the UPSERT so chunks/embeddings still land but the table stays
      bounded.

    Uses SQLite UPSERT so re-indexing preserves the original
    ``generated_at`` while bumping ``updated_at``.
    """
    markdown = result.markdown
    if markdown is None:
        return

    if not markdown:
        # Empty extraction: clear stale Markdown if any.
        session.execute(
            sql_text("DELETE FROM pdf_markdown WHERE file_id = :fid"),
            {"fid": file_id},
        )
        return

    if len(markdown.encode("utf-8")) > MAX_PDF_MARKDOWN_BYTES:
        logger.warning(
            "PDF Markdown for %s exceeds %d bytes; skipping pdf_markdown UPSERT",
            file_id, MAX_PDF_MARKDOWN_BYTES,
        )
        return

    now = datetime.now(UTC).isoformat()
    session.execute(
        sql_text(
            "INSERT INTO pdf_markdown "
            "(file_id, markdown, page_count, extractor, generated_at, updated_at) "
            "VALUES (:fid, :md, :pc, :ex, :now, :now) "
            "ON CONFLICT(file_id) DO UPDATE SET "
            "  markdown = excluded.markdown, "
            "  page_count = excluded.page_count, "
            "  extractor = excluded.extractor, "
            "  updated_at = excluded.updated_at"
        ),
        {
            "fid": file_id,
            "md": markdown,
            "pc": result.page_count if result.page_count is not None else 0,
            "ex": result.extractor,
            "now": now,
        },
    )


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

    for emb in existing:
        table = validate_vector_table(emb.vector_table)
        session.execute(
            sql_text(f"DELETE FROM {table} WHERE embedding_id = :id"),
            {"id": emb.id},
        )

    for emb in existing:
        session.delete(emb)
    session.flush()
