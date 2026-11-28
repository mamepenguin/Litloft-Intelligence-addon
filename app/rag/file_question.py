"""The material a question about one file is answered from (SPEC-ADDON-053).

A file's text is its indexed chunks, read from ``search.db``; nothing is
extracted at question time, which is why a file becomes askable only once
indexing has run. Every chunk is sent when they fit
``rag.file_question_max_chars``. Otherwise the chunks are ranked by the
cosine similarity of their stored embedding to the ranking text, taken in
rank order while they fit, and sent in file order with ``partial`` set.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from sqlalchemy import text as sql_text

import app.config as config
from app import llm_routing
from app.config import settings
from app.database import get_search_db_read, get_search_engine
from app.document_sections import EPUB_MIME
from app.extractors.pdf import PYMUPDF_LOCK
from app.frame_cache import extract_frame_to_cache, frame_cache_path, round_ms
from app.frame_text_lines import frame_segments
from app.rag.prompt import format_position
from app.vec_partition import read_vectors
from app.workers.embedder import embed_query
from app.workers.video_visual_selection import load_selection_inputs, select_scenes
from app.workers.vision import _load_image_bytes, _preprocess_image
from app.workers.whisper import LOFT_MIME
from app.youtube_source import (
    YOUTUBE_PROVIDER,
    LoftSource,
    LoftUnreadable,
    ensure_youtube_frame,
    is_frame_source,
    read_loft,
)

logger = logging.getLogger(__name__)

PDF_MIME = "application/pdf"

#: The Folder of a file that sits at the top of its drive.
TOP_OF_DRIVE = "(top of the drive)"

#: How many frames a text-less video's question may carry.
FRAME_BUDGET = 8
#: The budget once the position frame has taken one of the places.
FRAME_BUDGET_WITH_POSITION = 7

#: A scene this close to the position or to a kept time is not added.
FRAME_MIN_GAP_SECONDS = 1.0

#: How many of one question's frames may be prepared at once.
FRAME_PREPARATIONS = 4

#: How long a question waits for its frames once the slot is taken.
FRAME_WAIT_SECONDS = 45.0

#: How many section lines the video's contents list.
CONTENTS_SECTIONS = 40

#: How long one section's excerpt may be.
CONTENTS_EXCERPT_CHARS = 120

#: How many lines a document's contents list.
DOCUMENT_CONTENTS_LINES = 40

#: How long one section's title may be in a document's contents.
DOCUMENT_TITLE_CHARS = 60

#: The long edge of a page image, in pixels.
PAGE_IMAGE_LONG_EDGE = 1280

#: How long one page request may wait for the PyMuPDF lock, counted from its
#: first PyMuPDF call.
PAGE_IMAGE_LOCK_SECONDS = 30.0

#: The failure a page request's lock wait is logged by.
PAGE_IMAGE_LOCK_TIMEOUT = "lock_timeout"

#: Why an image could not be attached; the route logs the class, never
#: the question or the bytes.
IMAGE_FAILURE_PICTURE = "picture_unreadable"
IMAGE_FAILURE_FRAME = "frame_failed"
IMAGE_FAILURE_REJECTED = "image_rejected"

# ``[n]`` markers in the answer, up to a three-digit chunk number.
_CITATION_MARKER = re.compile(r"\[(\d{1,3})\]")

# Times a model may ask a frame at: a plain number, ``m:ss`` or ``h:mm:ss``,
# the clock forms with optional fractional seconds and every field below 60.
_PLAIN_SECONDS = re.compile(r"^\d+(?:\.\d+)?$")
_CLOCK_SECONDS = re.compile(r"^(\d+):(\d{1,2})(?:\.(\d+))?$")
_CLOCK_HOURS = re.compile(r"^(\d+):(\d{1,2}):(\d{1,2})(?:\.(\d+))?$")

# An integer a model may give as a string of ASCII digits.
_DIGITS = re.compile(r"^[0-9]+$")


@dataclass(frozen=True)
class SelectedChunk:
    """One chunk as sent: its text, and where inside the file it sits."""

    text: str
    position: dict[str, Any] | None


@dataclass(frozen=True)
class MaterialRow:
    """One spanned row of a file's material, as a range of the file reads it."""

    start: float
    end: float | None
    #: ``"transcript"`` or ``"frame_text"``.
    kind: str
    #: The text a passage of this row carries; ``None`` when the repeat filter
    #: removed a frame-text row whole.
    text: str | None
    #: A frame-text row's own text, before the repeat filter.
    stored_text: str | None
    #: The ``\n``-parts the repeat filter removed, in their order.
    removed: tuple[str, ...]
    #: Position in the file: the last tie-break of time order.
    order: int
    #: The passage number the row was sent with, or ``None`` when it was not.
    number: int | None = None


@dataclass(frozen=True)
class Material:
    """The chunks a question is answered from, in file order."""

    chunks: tuple[SelectedChunk, ...]
    #: True when only some of the file's chunks fit the budget.
    partial: bool
    file_id: str
    file_type: str
    mime_type: str
    file_path: str | None
    duration: float | None
    filename: str
    #: The drive-relative folder; ``TOP_OF_DRIVE`` at the top of the drive, and
    #: ``None`` when a stored absolute path cannot be placed in the drive.
    folder: str | None
    type_label: str
    drive: str = ""
    #: Every row with a span, kept when ``partial`` so a range can read the text
    #: the passages left out; empty otherwise.
    rows: tuple[MaterialRow, ...] = ()
    #: A document's rows in file order, each with the unit it sits in and the
    #: number it was sent with; empty for every other kind.
    document_rows: tuple["DocumentRow", ...] = ()
    #: A PDF's page count: the larger of ``pdf_markdown.page_count`` and the
    #: largest page its rows name; ``None`` when neither names one.
    page_count: int | None = None
    #: An EPUB's first and last section among its rows.
    sections: tuple[int, int] | None = None


@dataclass(frozen=True)
class DocumentRow:
    """One row of a document's text, in file order."""

    text: str
    #: The page of a PDF or section of an EPUB; ``None`` when the stored page is
    #: not a number.
    unit: int | None
    position: dict[str, Any] | None
    #: Position in the file, which is also the order a walk reads the rows in.
    order: int
    #: The passage number the row was sent with, or ``None`` when it was not.
    number: int | None = None


@dataclass(frozen=True)
class HistoryTurn:
    """One earlier turn as the model receives it, markers removed."""

    question: str
    answer: str


@dataclass(frozen=True)
class ImageMaterial:
    """One image as the provider receives it."""

    data: bytes
    mime_type: str


@dataclass(frozen=True)
class Frame:
    """One prepared frame, at the millisecond time its cache entry is keyed on."""

    time: float
    data: bytes
    mime_type: str = "image/webp"


class ImageUnavailable(Exception):
    """An image that applies to the question could not be prepared."""

    def __init__(self, failure: str) -> None:
        super().__init__(failure)
        self.failure = failure


class PageImageUnavailable(Exception):
    """A page image could not be made; ``failure`` is what the log line names."""

    def __init__(self, failure: str) -> None:
        super().__init__(failure)
        self.failure = failure


@dataclass(frozen=True)
class _Row:
    text: str
    position: dict[str, Any] | None
    embedding_id: str | None
    #: Position in the file, which is also the order the chunks are sent in.
    order: int
    #: The row's span, when it has one.
    start: float | None = None
    end: float | None = None
    #: ``"transcript"`` or ``"frame_text"`` for a row with a span.
    kind: str = ""
    #: A frame-text row's own text, before the repeat filter.
    stored_text: str | None = None
    #: The ``\n``-parts the repeat filter removed, in their order.
    removed: tuple[str, ...] = ()
    #: True for a frame-text row the repeat filter removed whole.
    removed_whole: bool = False


def _is_picture(file_type: str, mime_type: str) -> bool:
    return file_type == "image" or mime_type.startswith("image/")


def is_picture(material: Material) -> bool:
    return _is_picture(material.file_type, material.mime_type)


def _loft_source(file_path: str | None, drive: str) -> LoftSource | None:
    """The reference a ``.loft`` stores; ``None`` when it holds no usable one.

    The request drive resolves a drive-relative stored path. A stored absolute
    path has no relative spelling to join, and ``read_loft`` would lose it when
    the drive has no mount recorded, so that shape is read as stored.
    """
    try:
        return read_loft(file_path, drive)
    except LoftUnreadable:
        try:
            return read_loft(file_path)
        except LoftUnreadable:
            return None


def _folder_in_drive(file_path: str | None, drive: str) -> str | None:
    """The folder the stored ``file_path`` sits in, or ``None`` when unknown.

    A relative path is read as stored. An absolute one is placed against the
    drive's mount by whole path segments, so a sibling mount that shares a name
    prefix is not mistaken for the drive.
    """
    if not file_path:
        return None
    if not os.path.isabs(file_path):
        directory = os.path.dirname(file_path)
    else:
        mount = settings.drive_mounts.get(drive)
        if not mount:
            return None
        mount = os.path.normpath(mount)
        path = os.path.normpath(file_path)
        try:
            inside = os.path.commonpath([mount, path]) == mount
        except ValueError:
            return None
        if not inside:
            return None
        directory = os.path.dirname(os.path.relpath(path, mount))
    return directory or TOP_OF_DRIVE


def _type_label(mime_type: str, file_path: str | None, drive: str) -> str:
    """The file's format for the prompt; a ``.loft`` names what it links to."""
    if mime_type != LOFT_MIME:
        return mime_type
    source = _loft_source(file_path, drive)
    if source is None:
        return "Litloft link file (.loft)"
    if source.provider == YOUTUBE_PROVIDER:
        return "YouTube video, kept as a Litloft link file (.loft)"
    return f"web link ({source.provider}), kept as a Litloft link file (.loft)"


def _loft_frame_url(
    file_id: str, file_path: str | None, drive: str,
) -> str | None:
    """The stream URL of a `.loft` that is a frame source; ``None`` otherwise."""
    with get_search_db_read() as session:
        row = session.execute(
            sql_text(
                "SELECT loft_source, loft_fetch_failures FROM indexed_files "
                "WHERE file_id = :fid AND drive = :drive AND active = 1"
            ),
            {"fid": file_id, "drive": drive},
        ).first()
    if row is None:
        return None
    source = _loft_source(file_path, drive)
    if source is None or not is_frame_source(source.provider, row[0], row[1]):
        return None
    return source.url


def _has_frame_source(file_id: str, drive: str) -> bool:
    """Whether a frame of this file can be read at all.

    A ``.loft`` needs a reference a live read can name: YouTube, or a web one
    its fetch classified and has not given up. A native video is its own
    source.
    """
    with get_search_db_read() as session:
        row = session.execute(
            sql_text(
                "SELECT file_type, mime_type, file_path FROM indexed_files "
                "WHERE file_id = :fid AND drive = :drive AND active = 1"
            ),
            {"fid": file_id, "drive": drive},
        ).first()
    if row is None or row[0] != "video":
        return False
    if (row[1] or "") != LOFT_MIME:
        return True
    return _loft_frame_url(file_id, row[2], drive) is not None


def is_text_less_video(material: Material) -> bool:
    """Whether a question about this material goes through its frames alone."""
    return (
        material.file_type == "video"
        and not material.chunks
        and not is_picture(material)
    )


def is_text_video(material: Material) -> bool:
    """Whether a video with text also has frames a question could carry.

    A picture is not a video, and a ``.loft`` that is not a frame source has no
    frame to send, so its text path keeps the one image attempt it has.
    """
    return (
        material.file_type == "video"
        and bool(material.chunks)
        and not is_picture(material)
        and _has_frame_source(material.file_id, material.drive)
    )


def uses_document_tools(material: Material, *, can_see: bool) -> bool:
    """Whether a question about this material takes the document tool path.

    A profile that cannot see keeps today's text path, and an EPUB keeps it
    until its material is cut.
    """
    return can_see and (
        material.mime_type == PDF_MIME
        or (material.mime_type == EPUB_MIME and material.partial)
    )


def available(file_id: str, drive: str) -> bool:
    """Whether this drive may ask about this file (SPEC-ADDON-053 I6).

    An active ``indexed_files`` row of the request drive with at least one
    text chunk. A picture or a video with no text is askable when some profile
    can see (SPEC-ADDON-074, -108); every other kind still needs a chunk.
    Anything else is ``False`` rather than an error, so the tab simply has no
    button.
    """
    with get_search_db_read() as session:
        indexed = session.execute(
            sql_text(
                "SELECT file_type, mime_type, file_path FROM indexed_files "
                "WHERE file_id = :fid AND drive = :drive AND active = 1"
            ),
            {"fid": file_id, "drive": drive},
        ).first()
        if indexed is None:
            return False
        chunks = session.execute(
            sql_text(
                "SELECT EXISTS(SELECT 1 FROM transcript_chunks "
                "WHERE file_id = :fid) "
                "OR EXISTS(SELECT 1 FROM fts_text_content "
                "WHERE file_id = :fid)"
            ),
            {"fid": file_id},
        ).scalar()
    if chunks:
        return True
    file_type = indexed[0] or ""
    mime_type = indexed[1] or ""
    if _is_picture(file_type, mime_type):
        return llm_routing.any_profile_can_see(settings)
    if _has_frame_source(file_id, drive):
        return llm_routing.any_profile_can_see(settings)
    return False


def trim_history(
    turns: Sequence[Any], max_chars: int,
) -> list[HistoryTurn]:
    """The earlier turns the model gets, oldest first, within half the limit.

    ``[n]`` markers numbered an earlier request's passages, so they are
    removed from every answer before it is counted or sent. Turns are taken
    newest first; the first one that does not fit ends the block, so a kept
    turn is never older than a dropped one. The newest turn is kept cut to
    the budget when it does not fit on its own, question first.
    """
    budget = max_chars // 2
    kept: list[HistoryTurn] = []
    used = 0
    for turn in reversed(turns):
        question = turn.question
        answer = _without_markers(turn.answer)
        size = len(question) + len(answer)
        if not kept:
            if size > budget:
                question = question[:budget]
                answer = answer[:budget - len(question)]
                kept = [HistoryTurn(question, answer)]
                break
            kept = [HistoryTurn(question, answer)]
            used = size
            continue
        if used + size > budget:
            break
        kept = [*kept, HistoryTurn(question, answer)]
        used += size
    kept.reverse()
    return kept


def _without_markers(text: str) -> str:
    while True:
        stripped = _CITATION_MARKER.sub("", text)
        if stripped == text:
            return text
        text = stripped


def load_material(
    file_id: str, drive: str, question: str, *,
    budget: int | None = None,
    ranking_text: str | None = None,
) -> Material | None:
    """The chunks to send, or ``None`` when this file cannot be asked about.

    A picture, or a video with no text chunk, is material even with no chunk;
    every other kind needs one. ``budget`` is what a history block leaves of
    ``rag.file_question_max_chars``; ``ranking_text`` is what chunks are
    ranked against when not all of them fit.
    """
    if budget is None:
        budget = settings.rag.file_question_max_chars
    if ranking_text is None:
        ranking_text = question
    with get_search_db_read() as session:
        indexed = session.execute(
            sql_text(
                "SELECT mime_type, file_type, file_path, duration, filename "
                "FROM indexed_files "
                "WHERE file_id = :fid AND drive = :drive AND active = 1"
            ),
            {"fid": file_id, "drive": drive},
        ).first()
        if indexed is None:
            return None
        mime = indexed[0] or ""
        file_type = indexed[1] or ""
        file_path = indexed[2]
        duration = indexed[3]
        filename = indexed[4] or ""
        folder = _folder_in_drive(file_path, drive)
        type_label = _type_label(mime, file_path, drive)
        transcripts = session.execute(
            sql_text(
                "SELECT chunk_index, text, timestamp_start, timestamp_end "
                "FROM transcript_chunks WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchall()
        documents = session.execute(
            sql_text(
                "SELECT chunk_index, page, text FROM fts_text_content "
                "WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchall()
        if not transcripts and not documents:
            if not _is_picture(file_type, mime) and not _has_frame_source(
                file_id, drive
            ):
                return None
            return Material(
                chunks=(),
                partial=False,
                file_id=file_id,
                file_type=file_type,
                mime_type=mime,
                file_path=file_path,
                duration=duration,
                filename=filename,
                folder=folder,
                type_label=type_label,
                drive=drive,
            )
        section_titles = {
            int(page): title
            for page, title in session.execute(
                sql_text(
                    "SELECT page, title FROM document_sections "
                    "WHERE file_id = :fid"
                ),
                {"fid": file_id},
            ).fetchall()
        }
        markdown_pages = (
            session.execute(
                sql_text(
                    "SELECT page_count FROM pdf_markdown WHERE file_id = :fid"
                ),
                {"fid": file_id},
            ).scalar()
            if mime == PDF_MIME
            else None
        )
        whisper_embeddings = session.execute(
            sql_text(
                "SELECT id, timestamp_start, timestamp_end FROM embeddings "
                "WHERE file_id = :fid AND embedding_type = 'whisper' "
                "AND vector_table = 'vec_text'"
            ),
            {"fid": file_id},
        ).fetchall()
        text_embeddings = session.execute(
            sql_text(
                "SELECT id, chunk_index FROM embeddings "
                "WHERE file_id = :fid AND embedding_type = 'text_content' "
                "AND vector_table = 'vec_text'"
            ),
            {"fid": file_id},
        ).fetchall()
        frame_rows = (
            session.execute(
                sql_text(
                    "SELECT chunk_index, start_time, end_time, text "
                    "FROM video_frame_texts WHERE file_id = :fid"
                ),
                {"fid": file_id},
            ).fetchall()
            if file_type == "video"
            else []
        )
        frame_starts = {int(row[0]): row[1] for row in frame_rows}
        frame_ends = {
            int(row[0]): None if row[2] is None else float(row[2])
            for row in frame_rows
        }
        repeated_by_chunk = {
            index: segment.repeated
            for segment in frame_segments(frame_rows)
            for index, _text in segment.chunks
        }

    by_time = {
        (round(row[1], 3), round(row[2], 3)): row[0]
        for row in whisper_embeddings
        if row[1] is not None and row[2] is not None
    }
    by_chunk = {
        int(row[1]): row[0] for row in text_embeddings if row[1] is not None
    }

    rows: list[_Row] = []
    # Every row with a span, in file order; the rows below are those of them a
    # passage may carry.
    span_rows: list[_Row] = []
    order = 0
    # Transcript chunks first, by timestamp; a file with both kinds uses both.
    for _chunk_index, text, start, end in sorted(
        transcripts, key=lambda row: (row[2], row[0]),
    ):
        if not text:
            continue
        row = _Row(
            text=text,
            position={"seconds": start},
            embedding_id=by_time.get((round(start, 3), round(end, 3))),
            order=order,
            start=None if start is None else float(start),
            end=None if end is None else float(end),
            kind="transcript",
        )
        order += 1
        rows = [*rows, row]
        if row.start is not None:
            span_rows = [*span_rows, row]
    # The FTS column is untyped, so "10" would sort before "2" as text.
    for chunk_index, page, text in sorted(
        documents, key=lambda row: _chunk_order(row[0]),
    ):
        if not text:
            continue
        index = _as_int(chunk_index)
        if file_type == "video" and index is not None and index in frame_starts:
            repeated = repeated_by_chunk.get(index, frozenset())
            stored = text
            filtered = _filter_frame_text(stored, repeated)
            start = frame_starts[index]
            row = _Row(
                text="" if filtered is None else filtered,
                position={"seconds": start},
                embedding_id=by_chunk.get(index),
                order=order,
                start=None if start is None else float(start),
                end=frame_ends.get(index),
                kind="frame_text",
                stored_text=stored,
                removed=tuple(
                    part for part in stored.split("\n")
                    if part.strip() in repeated
                ),
                removed_whole=filtered is None,
            )
            order += 1
            if row.start is not None:
                span_rows = [*span_rows, row]
            if filtered is not None:
                rows = [*rows, row]
            continue
        position = _document_position(mime, page, section_titles)
        rows = [
            *rows,
            _Row(
                text=text,
                position=position,
                embedding_id=by_chunk.get(index),
                order=order,
            ),
        ]
        order += 1

    if not rows:
        return None

    material = dict(
        file_id=file_id,
        file_type=file_type,
        mime_type=mime,
        file_path=file_path,
        duration=duration,
        filename=filename,
        folder=folder,
        type_label=type_label,
        drive=drive,
    )
    if sum(len(row.text) for row in rows) <= budget:
        selected = rows
        partial = False
    else:
        selected = _fit(_rank(rows, ranking_text), budget)
        partial = True
    document_rows, page_count, sections = (
        _document_material(mime, rows, selected, markdown_pages)
        if mime in (PDF_MIME, EPUB_MIME)
        else ((), None, None)
    )
    return Material(
        chunks=_selected(selected),
        partial=partial,
        rows=_material_rows(span_rows, selected) if partial else (),
        document_rows=document_rows,
        page_count=page_count,
        sections=sections,
        **material,
    )


def _document_material(
    mime: str,
    rows: Sequence[_Row],
    sent: Sequence[_Row],
    markdown_pages: int | None,
) -> tuple[tuple[DocumentRow, ...], int | None, tuple[int, int] | None]:
    """A document's rows with their units and numbers, its page count and sections."""
    numbers = {row.order: index + 1 for index, row in enumerate(sent)}
    document_rows = tuple(
        DocumentRow(
            text=row.text,
            unit=_unit(row.position),
            position=row.position,
            order=row.order,
            number=numbers.get(row.order),
        )
        for row in rows
    )
    units = [row.unit for row in document_rows if row.unit is not None]
    if mime == PDF_MIME:
        largest = max(units) if units else 0
        return document_rows, max(markdown_pages or 0, largest) or None, None
    return document_rows, None, ((min(units), max(units)) if units else None)


def _unit(position: Mapping[str, Any] | None) -> int | None:
    """The page or section a row's position names, if it names one."""
    if not position:
        return None
    for key in ("page", "section"):
        if key in position:
            return position[key]
    return None


def _material_rows(
    span_rows: Sequence[_Row], sent: Sequence[_Row],
) -> tuple[MaterialRow, ...]:
    """The spanned rows of cut material, with the number each sent row took."""
    numbers = {row.order: index + 1 for index, row in enumerate(sent)}
    return tuple(
        MaterialRow(
            start=row.start,
            end=row.end,
            kind=row.kind,
            text=None if row.removed_whole else row.text,
            stored_text=row.stored_text,
            removed=row.removed,
            order=row.order,
            number=numbers.get(row.order),
        )
        for row in span_rows
        if row.start is not None
    )


def load_image(
    material: Material, position_seconds: float | None,
) -> ImageMaterial:
    """Prepare the one image the question may carry.

    A picture is re-encoded through the description worker's own helpers;
    a video is one ``large`` frame at the millisecond key its cache uses.
    Raises :class:`ImageUnavailable` with the class the route logs.
    """
    if is_picture(material):
        loaded = _load_image_bytes(material.file_id)
        if loaded is None:
            raise ImageUnavailable(IMAGE_FAILURE_PICTURE)
        raw, mime_type = loaded
        processed = _preprocess_image(raw, mime_type)
        if processed is None:
            raise ImageUnavailable(IMAGE_FAILURE_PICTURE)
        data, out_mime = processed
        return ImageMaterial(data, out_mime)
    if material.file_type == "video" and position_seconds is not None:
        return _frame_image(material, position_seconds)
    raise ImageUnavailable(IMAGE_FAILURE_FRAME)


def _frame_image(
    material: Material, position_seconds: float,
    cancelled: threading.Event | None = None,
) -> ImageMaterial:
    """One ``large`` frame at ``position_seconds``, from the cache or ffmpeg.

    ``cancelled`` is the question's own stop signal: a preparation that has
    not reached the extraction yet returns no frame instead of starting one,
    so a tab that closed starts no further work.
    """
    if material.duration is not None and position_seconds > material.duration:
        raise ImageUnavailable(IMAGE_FAILURE_FRAME)
    if cancelled is not None and cancelled.is_set():
        raise ImageUnavailable(IMAGE_FAILURE_FRAME)

    ms = int(round(position_seconds * 1000))
    timestamp = ms / 1000.0
    cache_path = frame_cache_path(material.file_id, timestamp, "large")
    if not (cache_path.exists() and cache_path.stat().st_size > 0):
        if cancelled is not None and cancelled.is_set():
            raise ImageUnavailable(IMAGE_FAILURE_FRAME)
        if material.mime_type == LOFT_MIME:
            _youtube_frame_to_cache(material, cache_path, timestamp, cancelled)
        else:
            _native_frame_to_cache(material, cache_path, timestamp, cancelled)
    if not (cache_path.exists() and cache_path.stat().st_size > 0):
        raise ImageUnavailable(IMAGE_FAILURE_FRAME)
    try:
        return ImageMaterial(cache_path.read_bytes(), "image/webp")
    except OSError as exc:
        raise ImageUnavailable(IMAGE_FAILURE_FRAME) from exc


def _youtube_frame_to_cache(
    material: Material, cache_path: Any, timestamp: float,
    cancelled: threading.Event | None = None,
) -> None:
    try:
        source = _loft_frame_url(
            material.file_id, material.file_path, material.drive,
        )
        if source is None:
            raise ImageUnavailable(IMAGE_FAILURE_FRAME)
        if cancelled is not None and cancelled.is_set():
            raise ImageUnavailable(IMAGE_FAILURE_FRAME)
        ensure_youtube_frame(
            material.file_id, source, cache_path, timestamp, "large",
        )
    except ImageUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - the frame route's result decides
        logger.warning(
            "file question frame extraction failed: %s", type(exc).__name__,
        )
        raise ImageUnavailable(IMAGE_FAILURE_FRAME) from exc


def _native_frame_to_cache(
    material: Material, cache_path: Any, timestamp: float,
    cancelled: threading.Event | None = None,
) -> None:
    if not material.file_path or not config.validate_file_path(
        material.file_path
    ):
        raise ImageUnavailable(IMAGE_FAILURE_FRAME)
    try:
        extract_frame_to_cache(
            material.file_path, cache_path, timestamp, size="large",
            cancelled=cancelled.is_set if cancelled is not None else None,
        )
    except Exception as exc:  # noqa: BLE001 - ffmpeg's result decides
        logger.warning(
            "file question frame extraction failed: %s",
            type(exc).__name__,
        )
        raise ImageUnavailable(IMAGE_FAILURE_FRAME) from exc


def frame_time(timestamp: float) -> float:
    """A frame's time: the millisecond its cache entry is keyed on."""
    return round_ms(timestamp) / 1000.0


def read_time(value: Any) -> float | None:
    """A time a model asked for, in seconds, or ``None`` when it is unreadable.

    A readable time is a JSON number (not a boolean), or a string that is a
    number, ``m:ss`` or ``h:mm:ss`` once surrounding whitespace is removed, with
    optional fractional seconds and every clock field below 60. A negative or
    non-finite time is unreadable.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if _PLAIN_SECONDS.match(text):
        return float(text)
    fields = _CLOCK_SECONDS.match(text) or _CLOCK_HOURS.match(text)
    if fields is None:
        return None
    parts = [float(part) for part in fields.groups()[:-1]]
    fraction = fields.groups()[-1]
    if any(part >= 60 for part in parts[1:]):
        return None
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds + (float(f"0.{fraction}") if fraction else 0.0)


def read_integer(value: Any) -> int | None:
    """An integer a model asked for, or ``None`` when it is unreadable.

    A readable one is a JSON integer, a JSON number with no fractional part, or
    a string of ASCII digits once surrounding whitespace is removed; never a
    boolean.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    return int(text) if _DIGITS.match(text) else None


def video_contents(
    file_id: str, sections: Sequence[tuple[float, float]],
) -> tuple[str, ...]:
    """The video's table of contents as the prompt lists it, one line a section.

    Each line is the section's time range and, when the file's transcript or
    frame text falls inside it, what is said and shown there; a section with
    none of that text is its time range alone.
    """
    kept = list(sections)[:CONTENTS_SECTIONS]
    if not kept:
        return ()
    rows = _excerpt_rows(file_id)
    lines: list[str] = []
    for start, end in kept:
        excerpt = _excerpt(rows, start, end)
        span = f"{format_position(start)}–{format_position(end)}"
        lines = [*lines, f"{span} {excerpt}" if excerpt else span]
    return tuple(lines)


def document_contents(material: Material) -> tuple[str, ...]:
    """The document's table of contents as the prompt lists it, one line a group.

    A PDF lists every page it has; an EPUB the sections its rows name. The
    pages or sections are split in order into at most
    :data:`DOCUMENT_CONTENTS_LINES` groups, each line pairing the group with the
    start of what it says; a group with no row is its range alone.
    """
    units = _contents_units(material)
    if not units:
        return ()
    size = math.ceil(len(units) / DOCUMENT_CONTENTS_LINES)
    pdf = material.mime_type == PDF_MIME
    titles = {
        row.unit: (row.position or {}).get("title")
        for row in material.document_rows
    }
    lines: list[str] = []
    for start in range(0, len(units), size):
        group = units[start:start + size]
        label = _contents_label(group, pdf)
        excerpt = _document_excerpt(material.document_rows, set(group))
        title = None if pdf else _contents_title(titles.get(group[0]))
        if not excerpt:
            lines = [*lines, label]
        elif title:
            lines = [
                *lines,
                f"{label}: {title} — {excerpt}",
            ]
        else:
            lines = [*lines, f"{label}: {excerpt}"]
    return tuple(lines)


def _contents_title(title: str | None) -> str | None:
    """A section title as one line with no citation-like number, cut short."""
    if not title:
        return None
    return " ".join(_without_markers(title).split())[:DOCUMENT_TITLE_CHARS] or None


def _contents_units(material: Material) -> list[int]:
    """The pages or sections a document's contents list, in order."""
    if material.mime_type == PDF_MIME:
        return list(range(1, (material.page_count or 0) + 1))
    return sorted(
        {row.unit for row in material.document_rows if row.unit is not None}
    )


def _contents_label(group: Sequence[int], pdf: bool) -> str:
    """One contents line's range: ``Page 3``, ``Pages 3–5`` or the section forms."""
    noun = "Page" if pdf else "Section"
    if len(group) == 1:
        return f"{noun} {group[0]}"
    return f"{noun}s {group[0]}–{group[-1]}"


def _document_excerpt(
    rows: Sequence[DocumentRow], units: set[int],
) -> str:
    """What the rows of one contents group say, cut short."""
    parts = [
        " ".join(_without_markers(row.text).split())
        for row in rows
        if row.unit in units
    ]
    return " ".join(part for part in parts if part)[
        :CONTENTS_EXCERPT_CHARS
    ].rstrip()


def page_image_path(material: Material) -> str | None:
    """The PDF on disk: a stored absolute path, or a relative one under its mount."""
    if not material.file_path:
        return None
    if os.path.isabs(material.file_path):
        return material.file_path
    return config.resolve_stored_path(
        config.resolve_file_path(material.drive, material.file_path)
    )


def _render_page(document: Any, number: int) -> bytes:
    """Page ``number`` from 1 as a JPEG whose long edge is 1280 px.

    Its page and pixmap are locals of this frame alone, so they are freed when
    it returns, or with the traceback of what it raised.
    """
    import pymupdf

    page = document[number - 1]
    long_edge = max(float(page.rect.width), float(page.rect.height))
    zoom = PAGE_IMAGE_LONG_EDGE / long_edge
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    return pixmap.tobytes("jpeg")


class PageImages:
    """One page request's PDF, read inside one hold of the shared lock.

    PyMuPDF objects call into MuPDF when they are freed, and MuPDF must not
    run beside an extraction. So the lock is taken on the request's first need,
    waiting at most :data:`PAGE_IMAGE_LOCK_SECONDS`, and held until
    :meth:`close`, which closes and drops the document before releasing it;
    a page's failure is handled by the caller while the lock is still held.
    """

    def __init__(
        self, material: Material, *,
        lock_seconds: float = PAGE_IMAGE_LOCK_SECONDS,
    ) -> None:
        self._path = page_image_path(material)
        self._lock_seconds = lock_seconds
        self._held = False
        self._document: Any = None

    def open(self) -> None:
        if self._path is None:
            raise FileNotFoundError("the file has no path on disk")
        if not PYMUPDF_LOCK.acquire(timeout=self._lock_seconds):
            raise PageImageUnavailable(PAGE_IMAGE_LOCK_TIMEOUT)
        self._held = True
        import pymupdf

        self._document = pymupdf.open(self._path)

    def page_count(self) -> int:
        return int(self._document.page_count)

    def image(self, number: int) -> bytes:
        """Page ``number`` from 1 as a JPEG whose long edge is 1280 px."""
        return _render_page(self._document, number)

    def close(self) -> None:
        if not self._held:
            return
        try:
            if self._document is not None:
                self._document.close()
                self._document = None
        finally:
            self._held = False
            PYMUPDF_LOCK.release()


@dataclass(frozen=True)
class _ExcerptRow:
    text: str
    start: float
    end: float | None
    #: 0 for a transcript row, 1 for a frame-text row: the order at one start.
    kind: int


def _excerpt_rows(file_id: str) -> tuple[_ExcerptRow, ...]:
    """The file's spoken and shown text rows, in the order a section reads them."""
    with get_search_db_read() as session:
        transcripts = session.execute(
            sql_text(
                "SELECT text, timestamp_start, timestamp_end "
                "FROM transcript_chunks WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchall()
        frames = session.execute(
            sql_text(
                "SELECT text, start_time, end_time "
                "FROM video_frame_texts WHERE file_id = :fid"
            ),
            {"fid": file_id},
        ).fetchall()
    rows: list[_ExcerptRow] = []
    for text, start, end in transcripts:
        if text and start is not None:
            rows = [
                *rows,
                _ExcerptRow(
                    text=text,
                    start=float(start),
                    end=None if end is None else float(end),
                    kind=0,
                ),
            ]
    for text, start, end in frames:
        if text and start is not None:
            rows = [
                *rows,
                _ExcerptRow(
                    text=text,
                    start=float(start),
                    end=None if end is None else float(end),
                    kind=1,
                ),
            ]
    rows.sort(key=lambda row: (row.start, row.kind))
    return tuple(rows)


def _overlaps(row: _ExcerptRow, start: float, end: float) -> bool:
    """Whether the row's span, or its start alone, falls inside ``[start, end)``."""
    row_end = row.start if row.end is None else row.end
    return row.start < end and row_end > start


def _excerpt(rows: Sequence[_ExcerptRow], start: float, end: float) -> str:
    """What the rows overlapping one section say, in time order and cut short."""
    parts = [
        " ".join(_without_markers(row.text).split())
        for row in rows
        if _overlaps(row, start, end)
    ]
    joined = " ".join(part for part in parts if part)
    return joined[:CONTENTS_EXCERPT_CHARS].rstrip()


def choose_frame_set(
    material: Material, position_seconds: float | None,
) -> tuple[float | None, list[float], tuple[tuple[float, float], ...]]:
    """The position frame's time, the scene frame times, and the sections listed.

    The scene frames are the frame times of the Ask list of SPEC-ADDON-112 over
    the whole file, then thinned to the budget or filled with evenly spaced
    times; the sections are that same run's navigation list. A failed read
    leaves both empty and the fill still runs.
    """
    budget = (
        FRAME_BUDGET_WITH_POSITION
        if position_seconds is not None
        else FRAME_BUDGET
    )
    sections: tuple[tuple[float, float], ...] = ()
    try:
        inputs = load_selection_inputs(material.file_id)
        selection = select_scenes(inputs, 0.0, material.duration)
        times = [item.frame_time for item in selection.ask]
        sections = tuple(
            (item.section_start, item.section_end)
            for item in selection.navigation
        )
    except Exception:  # noqa: BLE001 - an empty selection is still answerable
        logger.warning("scene_candidates_failed (file %s)", material.file_id)
        times = []
    if position_seconds is not None:
        times = [
            time for time in times
            if abs(time - position_seconds) >= FRAME_MIN_GAP_SECONDS
        ]
    times.sort()
    count = len(times)
    if count > budget:
        times = [
            times[math.floor(index * (count - 1) / (budget - 1) + 0.5)]
            for index in range(budget)
        ]
    duration = material.duration
    if (
        len(times) < budget
        and duration is not None
        and math.isfinite(duration)
        and duration > 0
    ):
        for index in range(budget):
            if len(times) >= budget:
                break
            candidate = (index + 0.5) * duration / budget
            if any(
                abs(candidate - time) < FRAME_MIN_GAP_SECONDS for time in times
            ):
                continue
            if (
                position_seconds is not None
                and abs(candidate - position_seconds) < FRAME_MIN_GAP_SECONDS
            ):
                continue
            times = [*times, candidate]
    position = (
        frame_time(position_seconds) if position_seconds is not None else None
    )
    return position, sorted(frame_time(time) for time in times), sections


async def prepare_frames(
    material: Material, position_seconds: float | None, *, deadline: float,
    cancelled: threading.Event | None = None,
) -> tuple[list[Frame], int, tuple[str, ...]]:
    """Prepare one question's frames, the ones dropped, and its contents.

    The position frame is started first, then the scene frames in time
    order, and at most :data:`FRAME_PREPARATIONS` of them run at once. No
    preparation is started at or after ``deadline``; one that failed or is
    still running then is dropped, and none is replaced by another time.
    ``cancelled`` is the tab's own stop signal, so a preparation that is
    still waiting for a worker does not begin its extraction. The contents are
    read from the selection's sections while the deadline leaves time for them.
    """
    loop = asyncio.get_running_loop()
    remaining = deadline - loop.time()
    if remaining <= 0 or (cancelled is not None and cancelled.is_set()):
        return [], 0, ()
    try:
        position, scene_times, sections = await asyncio.wait_for(
            asyncio.to_thread(choose_frame_set, material, position_seconds),
            timeout=remaining,
        )
    except asyncio.TimeoutError:
        logger.warning("scene_candidates_failed (file %s)", material.file_id)
        position, scene_times, sections = None, [], ()
    except asyncio.CancelledError:
        raise
    timestamps = ([position] if position is not None else []) + scene_times
    if not timestamps:
        return [], 0, await _contents(material, sections, deadline, cancelled)

    slots = asyncio.Semaphore(FRAME_PREPARATIONS)

    async def prepare(timestamp: float) -> Frame | None:
        async with slots:
            if loop.time() >= deadline:
                return None
            try:
                image = await asyncio.to_thread(
                    _frame_image, material, timestamp, cancelled,
                )
            except Exception:  # noqa: BLE001 - a frame's own failure is a drop
                return None
            return Frame(time=timestamp, data=image.data)

    tasks = [
        asyncio.create_task(prepare(timestamp)) for timestamp in timestamps
    ]
    try:
        done, pending = await asyncio.wait(
            tasks, timeout=max(0.0, deadline - loop.time()),
        )
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    frames = [task.result() for task in done if task.result() is not None]
    frames.sort(key=lambda frame: frame.time)
    dropped = len(timestamps) - len(frames)
    logger.info(
        "frames_dropped: chosen=%d dropped=%d (file %s)",
        len(frames), dropped, material.file_id,
    )
    return frames, dropped, await _contents(material, sections, deadline, cancelled)


async def _contents(
    material: Material,
    sections: tuple[tuple[float, float], ...],
    deadline: float,
    cancelled: threading.Event | None,
) -> tuple[str, ...]:
    """The contents lines, or none when they do not arrive by the deadline."""
    if not sections or (cancelled is not None and cancelled.is_set()):
        return ()
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        return ()
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(video_contents, material.file_id, sections),
            timeout=remaining,
        )
    except Exception:  # noqa: BLE001 - the question is answerable without them
        return ()


async def cut_frames(
    material: Material, times: Sequence[float], *, deadline: float,
    cancelled: threading.Event | None = None,
) -> list[Frame]:
    """Cut the frames at ``times`` in time order, and report the ones that came.

    At most :data:`FRAME_PREPARATIONS` run at once and none is started at or
    after ``deadline`` or once the question is cancelled. A frame that fails is
    dropped, not replaced; anything else that goes wrong is the caller's.
    """
    if not times:
        return []
    loop = asyncio.get_running_loop()
    slots = asyncio.Semaphore(FRAME_PREPARATIONS)

    async def cut(timestamp: float) -> Frame | None:
        async with slots:
            if loop.time() >= deadline or (
                cancelled is not None and cancelled.is_set()
            ):
                return None
            image = await asyncio.to_thread(
                _frame_image, material, timestamp, cancelled,
            )
            return Frame(time=timestamp, data=image.data)

    tasks = [asyncio.create_task(cut(timestamp)) for timestamp in times]
    try:
        done, pending = await asyncio.wait(
            tasks, timeout=max(0.0, deadline - loop.time()),
        )
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    frames: list[Frame] = []
    error: BaseException | None = None
    for task in done:
        try:
            frame = task.result()
        except ImageUnavailable:
            continue
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            error = error or exc
            continue
        if frame is not None:
            frames = [*frames, frame]
    if error is not None:
        raise error
    frames.sort(key=lambda frame: frame.time)
    return frames


def citations(
    answer: str, chunks: tuple[SelectedChunk, ...],
) -> list[dict[str, Any]]:
    """Every ``[n]`` the answer names, with that chunk's position.

    A number that was not sent is dropped (SPEC-ADDON-053 I3). Each number
    appears once, in the order the answer first names it.
    """
    return citations_from(
        answer, {index: chunk.position for index, chunk in enumerate(chunks, 1)},
    )


def citations_from(
    answer: str, positions: Mapping[int, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Every ``[n]`` the answer names that a number was sent for.

    ``positions`` maps each number that was given to what came with it. Each
    number appears once, in the order the answer first names it.
    """
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for match in _CITATION_MARKER.finditer(answer):
        number = int(match.group(1))
        if number in seen or number not in positions:
            continue
        seen = {*seen, number}
        out = [*out, {"n": number, "position": positions[number]}]
    return out


def _selected(rows: list[_Row]) -> tuple[SelectedChunk, ...]:
    return tuple(
        SelectedChunk(text=row.text, position=row.position) for row in rows
    )


def _rank(rows: list[_Row], question: str) -> list[_Row]:
    """The rows by descending similarity to the question, then file order.

    A chunk whose embedding row is missing ranks after every embedded one.
    ``embed_query`` raising means the question cannot be embedded, and file
    order is used instead.
    """
    try:
        query_vector = embed_query(question)
    except Exception as exc:  # noqa: BLE001 - file order is a valid answer
        logger.warning(
            "file question embedding failed: %s", type(exc).__name__,
        )
        return rows

    try:
        vectors = _vectors(
            [row.embedding_id for row in rows if row.embedding_id]
        )
    except Exception as exc:  # noqa: BLE001 - file order is a valid answer
        logger.warning(
            "file question vector scan failed: %s", type(exc).__name__,
        )
        return rows

    scored: list[tuple[float, _Row]] = []
    unembedded: list[_Row] = []
    for row in rows:
        vector = vectors.get(row.embedding_id) if row.embedding_id else None
        if vector is None:
            unembedded = [*unembedded, row]
        else:
            scored = [*scored, (_cosine(query_vector, vector), row)]
    scored.sort(key=lambda item: (-item[0], item[1].order))
    return [*[row for _score, row in scored], *unembedded]


def _fit(rows: list[_Row], budget: int) -> list[_Row]:
    """Greedily take rows while they fit, then restore file order.

    An overflow is skipped rather than ending the fill. When not even the
    best-ranked row fits, it is cut to the budget and sent alone.
    """
    taken: list[_Row] = []
    used = 0
    for row in rows:
        if used + len(row.text) > budget:
            continue
        taken = [*taken, row]
        used += len(row.text)
    if not taken and rows:
        taken = [replace(rows[0], text=rows[0].text[:budget])]
    taken.sort(key=lambda row: row.order)
    return taken


def _vectors(embedding_ids: list[str]) -> dict[str, np.ndarray]:
    """The stored ``vec_text`` vectors of ``embedding_ids``."""
    wanted = sorted(set(embedding_ids))
    with get_search_engine().connect() as conn:
        stored = read_vectors(conn, "vec_text", wanted)
    return {
        embedding_id: np.frombuffer(blob, dtype=np.float32)
        for embedding_id, blob in stored.items()
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right)) / denominator


def _filter_frame_text(text: str, repeated: frozenset[str]) -> str | None:
    """``text`` without its parts that repeat, or ``None`` when none is left."""
    parts = text.split("\n")
    kept = [part for part in parts if part.strip() not in repeated]
    if len(kept) == len(parts):
        return text
    if not any(part.strip() for part in kept):
        return None
    return "\n".join(kept)


def _document_position(
    mime: str, page_raw: Any, section_titles: dict[int, str],
) -> dict[str, Any] | None:
    """Where a document chunk sits, in the vocabulary its viewer moves by.

    A PDF's page is the page number; an EPUB's is the spine section number
    ``?section=`` takes, with the stored section title. Every other kind
    has no position a viewer can move to.
    """
    page = _as_int(page_raw)
    if page is None:
        return None
    if mime == PDF_MIME:
        return {"page": page}
    if mime == EPUB_MIME:
        return {"section": page, "title": section_titles.get(page)}
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _chunk_order(value: Any) -> int:
    """A sortable chunk index; an unreadable one sorts first, as text would."""
    number = _as_int(value)
    return -1 if number is None else number
