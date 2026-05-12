"""Unit tests for ``HtmlExtractor`` (HTML indexing Phase 1).

The HTML extractor parses HTML / XHTML with BeautifulSoup4, strips
``<script>`` / ``<style>`` / ``<noscript>`` subtrees so AI-artifact
code does not pollute embeddings, converts the remainder to Markdown
via ``html2text``, and yields ``TextChunk`` entries tagged with the
nearest ``section: <heading>`` metadata.

Spec: ``docs/superpowers/specs/2026-05-12-html-indexing.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.config as config  # noqa: E402
from app.extractors.base import ExtractionResult  # noqa: E402
from app.extractors.html import (  # noqa: E402
    EXTRACTOR_NAME,
    MAX_HTML_BYTES,
    HtmlExtractor,
)


@pytest.fixture(autouse=True)
def allow_tmp_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bypass the production ``allowed_base_dirs`` check for fixture files."""
    monkeypatch.setattr(config, "validate_file_path", lambda _path: True)


def _write(tmp_path: Path, name: str, html: str) -> str:
    path = tmp_path / name
    path.write_text(html, encoding="utf-8")
    return str(path)


def test_can_handle_html_htm_xhtml() -> None:
    extractor = HtmlExtractor()
    assert extractor.can_handle("page.html") is True
    assert extractor.can_handle("page.htm") is True
    assert extractor.can_handle("page.xhtml") is True
    assert extractor.can_handle("page.txt") is False
    assert extractor.can_handle("page.pdf") is False


def test_extracts_body_text_with_heading_metadata(tmp_path: Path) -> None:
    """T1: A standard article with a single ``<h1>`` and paragraphs.

    The first chunk's metadata should record the heading as ``section:``.
    """
    html = """
    <html><body>
      <h1>Introduction</h1>
      <p>First paragraph about cats.</p>
      <p>Second paragraph about dogs.</p>
    </body></html>
    """
    file_path = _write(tmp_path, "article.html", html)

    result = HtmlExtractor().extract(file_path)

    assert isinstance(result, ExtractionResult)
    assert result.extractor == EXTRACTOR_NAME
    assert result.markdown is None  # Phase 1: no html_markdown table
    assert len(result.chunks) >= 1
    assert any("cats" in c.text for c in result.chunks)
    assert any("Introduction" in (c.metadata or "") for c in result.chunks)


def test_section_metadata_tracks_latest_heading(tmp_path: Path) -> None:
    """T2: Multiple headings → later chunks pick up the closer one."""
    body_a = "<p>Alpha paragraph.</p>" * 30
    body_b = "<p>Beta paragraph.</p>" * 30
    html = f"""
    <html><body>
      <h1>Section A</h1>
      {body_a}
      <h2>Section B</h2>
      {body_b}
    </body></html>
    """
    file_path = _write(tmp_path, "multi.html", html)

    result = HtmlExtractor().extract(file_path)

    sections = {c.metadata for c in result.chunks if c.metadata}
    assert any("Section A" in s for s in sections)
    assert any("Section B" in s for s in sections)


def test_script_and_style_content_excluded(tmp_path: Path) -> None:
    """T3+T4: ``<script>``/``<style>``/``<noscript>`` bodies must not leak.

    An AI artifact with a huge inline ``<script>`` and an empty
    ``<div id="root">`` should produce no embeddings derived from code.
    """
    html = """
    <html><head>
      <title>AI artifact</title>
      <style>.x { color: red; } /* secret_marker_css */</style>
      <script>
        const SECRET_MARKER_JS = "should_not_be_indexed";
        function render() { return SECRET_MARKER_JS; }
      </script>
      <noscript>secret_marker_noscript fallback</noscript>
    </head><body>
      <div id="root"></div>
    </body></html>
    """
    file_path = _write(tmp_path, "artifact.html", html)

    result = HtmlExtractor().extract(file_path)

    combined = "\n".join(c.text for c in result.chunks)
    assert "SECRET_MARKER_JS" not in combined
    assert "secret_marker_css" not in combined
    assert "secret_marker_noscript" not in combined


def test_size_limit_returns_empty(tmp_path: Path) -> None:
    """T5: Files larger than MAX_HTML_BYTES yield an empty result."""
    path = tmp_path / "huge.html"
    # Write MAX_HTML_BYTES + 1 byte
    path.write_bytes(b"<html><body>" + b"a" * (MAX_HTML_BYTES + 1) + b"</body></html>")

    result = HtmlExtractor().extract(str(path))

    assert result.chunks == []
    assert result.markdown is None


def test_empty_html_returns_empty(tmp_path: Path) -> None:
    """T6: A document with no body text yields no chunks."""
    file_path = _write(tmp_path, "empty.html", "<html><body></body></html>")

    result = HtmlExtractor().extract(file_path)

    assert result.chunks == []


def test_xhtml_extension_handled(tmp_path: Path) -> None:
    """T7: ``.xhtml`` extension routes through the same pipeline."""
    html = """<?xml version="1.0" encoding="UTF-8"?>
    <html xmlns="http://www.w3.org/1999/xhtml"><body>
      <h1>XHTML Doc</h1>
      <p>Body content in XHTML.</p>
    </body></html>
    """
    file_path = _write(tmp_path, "doc.xhtml", html)

    result = HtmlExtractor().extract(file_path)

    assert len(result.chunks) >= 1
    assert any("Body content" in c.text for c in result.chunks)


def test_malformed_html_still_extracts(tmp_path: Path) -> None:
    """T8: BeautifulSoup is forgiving; broken markup must not raise."""
    file_path = _write(
        tmp_path, "bad.html",
        "<html><body><p>unclosed paragraph <h1>still readable</h1></body>",
    )

    result = HtmlExtractor().extract(file_path)

    combined = " ".join(c.text for c in result.chunks)
    assert "unclosed paragraph" in combined or "still readable" in combined


def test_code_block_hash_not_treated_as_heading(tmp_path: Path) -> None:
    """M1 fix: `#` lines inside fenced code blocks must not become sections.

    html2text turns ``<pre><code>`` into triple-backtick fences in the
    Markdown output. A Python comment like ``# this`` inside the fence
    would falsely register as a section heading and over-segment the
    surrounding prose.
    """
    html = """
    <html><body>
      <h1>Real Heading</h1>
      <p>Intro paragraph before code.</p>
      <pre><code>
# this is a code comment, not a heading
def foo():
    # nested comment
    return 1
      </code></pre>
      <p>Body paragraph after code.</p>
    </body></html>
    """
    file_path = _write(tmp_path, "code.html", html)

    result = HtmlExtractor().extract(file_path)

    # All section metadata for produced chunks must reference "Real Heading"
    # (or empty for pre-heading content). Code comments must never appear
    # as section names.
    sections = {c.metadata for c in result.chunks if c.metadata}
    assert all(
        ("Real Heading" in s) for s in sections
    ), f"unexpected section metadata: {sections}"
    assert not any("code comment" in s for s in sections)
    assert not any("nested comment" in s for s in sections)


def test_japanese_utf8_preserved(tmp_path: Path) -> None:
    """T9: UTF-8 Japanese content survives the parse → markdown roundtrip."""
    html = """
    <html><body>
      <h1>日本語の見出し</h1>
      <p>本文に日本語のテキストを書きます。</p>
    </body></html>
    """
    file_path = _write(tmp_path, "ja.html", html)

    result = HtmlExtractor().extract(file_path)

    combined = " ".join(c.text for c in result.chunks)
    assert "日本語" in combined
    assert "テキスト" in combined
