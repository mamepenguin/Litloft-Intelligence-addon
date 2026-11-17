"""SPEC-ADDON-122 I2: mixed-page vertical boxes already present are not duplicated."""
import fitz
import pymupdf4llm
import app.config as config
from app.extractors.pdf import PdfExtractor
from tests.document_text_api import api, reader, search_index, seed  # noqa: F401


def test_mixed_page_deduplicates_embedded_vertical_label(tmp_path, monkeypatch, reader):
    monkeypatch.setattr(config, 'validate_file_path', lambda _path: True)
    path = tmp_path / 'mixed.pdf'
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 100), 'Horizontal paragraph with plenty of searchable words.')
        page.insert_text((350, 100), 'MIXEDLABEL', rotate=270)
        doc.save(path)
    reader.register('mixed', path, 'application/pdf')
    original = pymupdf4llm.to_markdown
    def markdown_with_label(*args, **kwargs):
        pages = original(*args, **kwargs)
        for page in pages:
            page['text'] += '\nMIXEDLABEL\n'
        return pages
    monkeypatch.setattr(pymupdf4llm, 'to_markdown', markdown_with_label)
    indexed = PdfExtractor().extract(str(path)).markdown
    read = reader.ok('mixed', start=1, count=1)['units'][0]['text']
    assert indexed.count('MIXEDLABEL') == 1
    assert read.count('MIXEDLABEL') == 1
