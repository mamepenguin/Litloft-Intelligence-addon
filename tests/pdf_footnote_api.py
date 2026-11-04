"""Where the SPEC-ADDON-060 tests reach the names the spec leaves implicit.

The spec fixes ``PAGE_MARGINS``, ``PDF_TEXT_RULES_VERSION`` and the
``index_meta`` key but not the function that runs the version check, so that name
is bound here and nowhere else. No assertion lives here.
"""

from __future__ import annotations

from app import database
from app.extractors import pdf


def check_text_rules() -> None:
    """The startup text-rule check the lifespan runs."""
    database.check_pdf_text_rules()


def rules_version() -> str:
    return pdf.PDF_TEXT_RULES_VERSION


def page_margins() -> tuple[int, int, int, int]:
    return pdf.PAGE_MARGINS
