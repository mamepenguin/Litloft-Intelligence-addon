"""Base class and data structures for content extractors.

Extractors convert file contents into text chunks suitable for embedding.
Each extractor declares which file extensions it supports.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TextChunk:
    """A chunk of text extracted from a file.

    Attributes:
        text: The extracted text content.
        page: Optional page/section number (for PDFs, etc.).
        metadata: Optional metadata about the chunk source.
    """

    text: str
    page: int | None = None
    metadata: str = ""


@dataclass(frozen=True)
class ExtractionResult:
    """The full output of a content extraction call.

    Carries embedding-ready chunks plus an optional reading-friendly
    Markdown rendering (currently produced only by the PDF extractor
    via PyMuPDF4LLM). Phase 3 of the PDF Markdown work persists the
    Markdown into the ``pdf_markdown`` table.

    Attributes:
        chunks: Embedding-ready text chunks.
        markdown: Full reading-friendly Markdown if available, else None.
        extractor: Identifier of the concrete pipeline that produced
            this result. Examples: ``"text"``, ``"pymupdf4llm"``,
            ``"fitz_fallback"``.
        page_count: Source page count for paginated formats (PDF). None
            for formats with no native pagination (plain text, Markdown,
            etc.). Used by the indexing pipeline when persisting the
            ``pdf_markdown`` row.
    """

    chunks: list[TextChunk] = field(default_factory=list)
    markdown: str | None = None
    extractor: str = ""
    page_count: int | None = None


class ContentExtractor:
    """Base class for file content extractors.

    Subclasses must define supported_extensions and implement extract().
    """

    supported_extensions: list[str] = []

    def can_handle(self, file_path: str) -> bool:
        """Check if this extractor can handle the given file."""
        suffix = Path(file_path).suffix.lower()
        return suffix in self.supported_extensions

    def extract(self, file_path: str) -> ExtractionResult:
        """Extract content from a file.

        Args:
            file_path: Absolute path to the file.

        Returns:
            ExtractionResult containing embedding-ready chunks plus an
            optional Markdown rendering and an extractor identifier.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError

    @staticmethod
    def chunk_text(
        text: str,
        max_size: int = 1000,
        overlap: int = 200,
    ) -> list[str]:
        """Split text into overlapping chunks.

        Attempts to split on paragraph boundaries first, then sentence
        boundaries, falling back to character-level splitting.

        Args:
            text: The text to split.
            max_size: Maximum characters per chunk.
            overlap: Number of overlapping characters between chunks.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        if len(text) <= max_size:
            return [text]

        chunks: list[str] = []
        start = 0
        # Track the previous chunk boundary so ``rfind`` cannot
        # rediscover it. Without this, when ``split_pos - overlap``
        # falls before ``start + 1``, the loop crept forward 1 char at
        # a time and produced a sequence of strict-suffix tail chunks
        # (e.g. ``"short", "hort", "ort", "rt", "t"``) until ``start``
        # finally crossed the separator.
        last_split = -1

        while start < len(text):
            end = start + max_size

            if end >= len(text):
                chunks = [*chunks, text[start:].strip()]
                break

            # Constrain search to positions strictly after the
            # previous boundary so we never re-pick the same one.
            search_start = max(start, last_split + 1)

            split_pos = text.rfind("\n\n", search_start, end)
            if split_pos == -1:
                split_pos = text.rfind(". ", search_start, end)
            if split_pos == -1:
                split_pos = text.rfind("\n", search_start, end)
            if split_pos == -1:
                split_pos = text.rfind(" ", search_start, end)
            if split_pos == -1:
                split_pos = end  # Hard split

            chunk = text[start:split_pos].strip()
            if chunk:
                chunks = [*chunks, chunk]

            # Advance: prefer ``overlap`` chars of replay for boundary
            # recall, but never regress below the previous chunk start.
            # ``search_start`` (above) handles the orthogonal concern of
            # not rediscovering ``last_split`` in the next iteration.
            last_split = split_pos
            start = max(split_pos - overlap, start + 1)

        return chunks
