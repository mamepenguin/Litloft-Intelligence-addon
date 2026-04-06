"""Base class and data structures for content extractors.

Extractors convert file contents into text chunks suitable for embedding.
Each extractor declares which file extensions it supports.
"""

from dataclasses import dataclass
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


class ContentExtractor:
    """Base class for file content extractors.

    Subclasses must define supported_extensions and implement extract().
    """

    supported_extensions: list[str] = []

    def can_handle(self, file_path: str) -> bool:
        """Check if this extractor can handle the given file."""
        suffix = Path(file_path).suffix.lower()
        return suffix in self.supported_extensions

    def extract(self, file_path: str) -> list[TextChunk]:
        """Extract text chunks from a file.

        Args:
            file_path: Absolute path to the file.

        Returns:
            List of TextChunk objects extracted from the file.

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

        while start < len(text):
            end = start + max_size

            if end >= len(text):
                chunks = [*chunks, text[start:].strip()]
                break

            # Try to find a paragraph break
            split_pos = text.rfind("\n\n", start, end)
            if split_pos == -1 or split_pos <= start:
                # Try sentence boundary
                split_pos = text.rfind(". ", start, end)
            if split_pos == -1 or split_pos <= start:
                # Try any newline
                split_pos = text.rfind("\n", start, end)
            if split_pos == -1 or split_pos <= start:
                # Fall back to space
                split_pos = text.rfind(" ", start, end)
            if split_pos == -1 or split_pos <= start:
                # Hard split
                split_pos = end

            chunk = text[start:split_pos].strip()
            if chunk:
                chunks = [*chunks, chunk]

            start = max(split_pos - overlap, start + 1)

        return chunks
