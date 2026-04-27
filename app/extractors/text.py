"""Text file content extractor.

Handles plain text files: .txt, .md, .csv, .json, .srt, .vtt
"""

import json
import re
from pathlib import Path

from app.config import settings
from app.extractors.base import ContentExtractor, ExtractionResult, TextChunk

# Maximum file size to process (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

EXTRACTOR_NAME = "text"


class TextExtractor(ContentExtractor):
    """Extracts text content from plain text files."""

    supported_extensions: list[str] = [
        ".txt", ".md", ".csv", ".json", ".srt", ".vtt",
    ]

    def extract(self, file_path: str) -> ExtractionResult:
        """Extract text chunks from a text-based file.

        Plain text formats are not converted to Markdown; the resulting
        ``ExtractionResult.markdown`` is always ``None``.

        Args:
            file_path: Absolute path to the file.

        Returns:
            ExtractionResult with chunks suitable for embedding.
        """
        from app.config import validate_file_path

        empty = ExtractionResult(chunks=[], markdown=None, extractor=EXTRACTOR_NAME)

        if not validate_file_path(file_path):
            return empty
        path = Path(file_path)
        if not path.exists():
            return empty

        if path.stat().st_size > MAX_FILE_SIZE:
            return empty

        try:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return empty

        if not raw_text.strip():
            return empty

        suffix = path.suffix.lower()
        cleaned = _clean_by_type(raw_text, suffix)

        if not cleaned.strip():
            return empty

        chunk_config = settings.indexing.text_chunking
        text_chunks = self.chunk_text(
            cleaned,
            max_size=chunk_config.max_chunk_size,
            overlap=chunk_config.overlap,
        )

        chunks = [
            TextChunk(text=chunk, page=None, metadata=suffix)
            for chunk in text_chunks
        ]
        return ExtractionResult(
            chunks=chunks, markdown=None, extractor=EXTRACTOR_NAME
        )


def _clean_by_type(text: str, suffix: str) -> str:
    """Clean text based on file type."""
    cleaners = {
        ".srt": _clean_srt,
        ".vtt": _clean_vtt,
        ".json": _clean_json,
        ".csv": _clean_csv,
    }
    cleaner = cleaners.get(suffix)
    if cleaner is not None:
        return cleaner(text)
    return text


def _clean_srt(text: str) -> str:
    """Remove SRT timing and index lines, keep only dialogue text."""
    lines = text.strip().split("\n")
    result_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Skip empty lines, index numbers, and timing lines
        if not stripped:
            continue
        if stripped.isdigit():
            continue
        if re.match(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->", stripped):
            continue
        # Remove HTML-like tags from subtitles
        cleaned = re.sub(r"<[^>]+>", "", stripped)
        if cleaned:
            result_lines = [*result_lines, cleaned]

    return " ".join(result_lines)


def _clean_vtt(text: str) -> str:
    """Remove VTT header, timing lines, and cue settings."""
    lines = text.strip().split("\n")
    result_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip WebVTT header and metadata
        if stripped.startswith("WEBVTT") or stripped.startswith("NOTE"):
            continue
        # Skip timing lines
        if re.match(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->", stripped):
            continue
        # Skip cue identifiers (standalone numbers or names)
        if stripped.isdigit():
            continue
        cleaned = re.sub(r"<[^>]+>", "", stripped)
        if cleaned:
            result_lines = [*result_lines, cleaned]

    return " ".join(result_lines)


def _clean_json(text: str) -> str:
    """Extract string values from JSON for meaningful text content."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text

    strings: list[str] = []
    _collect_strings(data, strings)
    return " ".join(strings)


def _collect_strings(obj: object, strings: list[str]) -> None:
    """Recursively collect string values from a JSON structure."""
    if isinstance(obj, str):
        if len(obj) > 2:
            strings.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_strings(value, strings)
    elif isinstance(obj, list):
        for item in obj:
            _collect_strings(item, strings)


def _clean_csv(text: str) -> str:
    """Clean CSV by joining cells with spaces, removing empty cells."""
    lines = text.strip().split("\n")
    result_parts: list[str] = []

    for line in lines:
        cells = line.split(",")
        cleaned_cells = [c.strip().strip('"') for c in cells if c.strip()]
        if cleaned_cells:
            result_parts = [*result_parts, " ".join(cleaned_cells)]

    return "\n".join(result_parts)
