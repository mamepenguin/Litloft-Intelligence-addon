"""Seeding and counting for the acceptance tests of the frame-text repeat rule.

The Ask tests seed a frame-text row in both ``fts_text_content`` and
``video_frame_texts``; the search tests seed the same pair through ``Seed``.
"""

from __future__ import annotations

from sqlalchemy import text

FRAME_ROW = (
    "INSERT INTO video_frame_texts(file_id, chunk_index, start_time, end_time, text) "
    "VALUES (:f, :c, :s, :e, :t)"
)

COUNTED_TABLES = (
    "indexed_files",
    "fts_text_content",
    "video_frame_texts",
    "transcript_chunks",
    "embeddings",
    "document_sections",
    "video_visual_runs",
    "video_visual_scenes",
)


def ask_frame_chunk(db, file_id, chunk_index, start, end, body, *, cos=None):
    """One frame-text chunk of a video, as ``load_material`` reads it."""
    db.on(file_id)
    db.document(body, chunk_index=chunk_index, cos=cos)
    db.db.index.sql(FRAME_ROW, f=file_id, c=chunk_index, s=start, e=end, t=body)


def search_frame_chunk(seed, file_id, chunk_index, start, end, body):
    """One frame-text chunk of a video, as find and exact search read it."""
    seed.text_chunk(file_id, chunk_index, body)
    seed.index.sql(FRAME_ROW, f=file_id, c=chunk_index, s=start, e=end, t=body)


def bulk_frame_segments(index, file_id, count, body, *, start=0.0, step=1.0):
    """``count`` one-chunk segments, ``step`` seconds apart, all with ``body``."""
    fts = [{"f": file_id, "c": str(i), "t": body} for i in range(count)]
    frames = [
        {"f": file_id, "c": i, "s": start + i * step, "e": start + i * step + 0.5,
         "t": body}
        for i in range(count)
    ]
    with index.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO fts_text_content(file_id, chunk_index, page, text) "
                "VALUES (:f, :c, '', :t)"
            ),
            fts,
        )
        conn.execute(text(FRAME_ROW), frames)


def table_counts(index):
    """The row count of every store a reader could write to."""
    with index.engine.connect() as conn:
        return {
            name: conn.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar()
            for name in COUNTED_TABLES
        }
