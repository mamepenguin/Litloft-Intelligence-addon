"""PDF content extractor using PyMuPDF (fitz).

Extracts text from PDF files page by page, then chunks for embedding.
"""

from pathlib import Path

from app.config import settings
from app.extractors.base import ContentExtractor, TextChunk

# Maximum file size to process (50MB)
MAX_PDF_SIZE = 50 * 1024 * 1024

# Maximum pages to process
MAX_PAGES = 500


class PdfExtractor(ContentExtractor):
    """Extracts text content from PDF files using PyMuPDF."""

    supported_extensions: list[str] = [".pdf"]

    def extract(self, file_path: str) -> list[TextChunk]:
        """Extract text chunks from a PDF file.

        Each page is extracted separately and then chunked if needed.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            List of TextChunk objects, one or more per page.
        """
        from app.config import validate_file_path

        if not validate_file_path(file_path):
            return []
        path = Path(file_path)
        if not path.exists():
            return []

        if path.stat().st_size > MAX_PDF_SIZE:
            return []

        try:
            import fitz  # PyMuPDF
        except ImportError:
            return []

        chunks: list[TextChunk] = []
        chunk_config = settings.indexing.text_chunking

        try:
            doc = fitz.open(str(path))
        except Exception:
            return []

        try:
            page_count = min(doc.page_count, MAX_PAGES)

            for page_num in range(page_count):
                try:
                    page = doc[page_num]
                    page_text = page.get_text("text")
                except Exception:
                    continue

                if not page_text or not page_text.strip():
                    continue

                text_chunks = self.chunk_text(
                    page_text.strip(),
                    max_size=chunk_config.max_chunk_size,
                    overlap=chunk_config.overlap,
                )

                page_chunks = [
                    TextChunk(
                        text=chunk,
                        page=page_num + 1,
                        metadata=f"page {page_num + 1}",
                    )
                    for chunk in text_chunks
                ]
                chunks = [*chunks, *page_chunks]
        finally:
            doc.close()

        return chunks
