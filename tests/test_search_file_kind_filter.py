"""``?type=markdown`` / ``?type=pdf`` reach semantic search.

Core's toolbar names eight kinds — the six flat ``file_type`` values
plus ``markdown`` and ``pdf`` nested under ``document``. ``IndexedFile``
is a snapshot carrying the flat column, so the old
``file_type == file_type`` predicate matched **nothing** for those two.
That failure was silent: the semantic half of the result list emptied
out and the page degraded to filename matches with nothing saying so,
which is worst exactly where semantic search earns its keep (a drive of
notes).

These run through the real ``_build_results`` against a real SQLite
``indexed_files`` table, so what is asserted is the SQL the search
actually issues — not the classifier's table read back to itself.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import Base  # noqa: E402
from app.models import IndexedFile  # noqa: E402

import app.search as search  # noqa: E402


# (file_id, filename, file_type, mime_type)
ROWS = [
    ("note0000mark", "design.md", "document", "text/markdown"),
    # Mime never recorded — the case the two old filters disagreed
    # about. The extension is the only thing that names it.
    ("note0000mime", "plan.MD", "document", ""),
    ("note0000long", "readme.markdown", "document", ""),
    ("pdf00000mime", "invoice.pdf", "document", "application/pdf"),
    ("pdf00000ext0", "scan.PDF", "document", ""),
    ("doc00000othr", "notes.docx", "document",
     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("video0000001", "clip.mp4", "video", "video/mp4"),
    ("audio0000001", "talk.mp3", "audio", "audio/mpeg"),
    ("image0000001", "shot.jpg", "image", "image/jpeg"),
    # Filed as `other` by core but named like a note, and with no mime
    # to say otherwise. It is here to pin the nested branch's one real
    # decision: it does *not* also require `file_type == "document"`,
    # because a row whose mime was never recorded may well carry the
    # wrong file_type from the same writer. Tightening the predicate
    # "for correctness" drops exactly the rows the fallback exists for,
    # and this row is what turns that into a red test.
    ("note0000othr", "stray.md", "other", ""),
]


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        for file_id, filename, file_type, mime_type in ROWS:
            session.add(
                IndexedFile(
                    file_id=file_id,
                    drive="notes",
                    filename=filename,
                    file_path=f"/drives/notes/{filename}",
                    file_type=file_type,
                    mime_type=mime_type,
                    file_size=10,
                    active=True,
                )
            )
        session.commit()

    @contextmanager
    def _read():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    monkeypatch.setattr(search, "get_search_db_read", _read)
    yield
    engine.dispose()


def _search(kind: str | None) -> set[str]:
    """Every seeded file scores equally; the filter is the only sieve."""
    scores = {
        file_id: search._FileScore(
            file_id=file_id,
            combined_score=1.0,
            matches=[search.MatchInfo(match_type="metadata", text="x", score=1.0)],
            match_types={"metadata"},
        )
        for file_id, *_ in ROWS
    }
    results = search._build_results(scores, kind, "notes", limit=50, skip_cutoff=True)
    return {r.file_id for r in results}


def test_no_filter_returns_everything(search_db):
    assert _search(None) == {file_id for file_id, *_ in ROWS}


def test_markdown_finds_the_mime_and_the_extension(search_db):
    # ``note0000othr`` is in here on its name alone, with a file_type
    # of ``other``: the nested branch does not require the row to also
    # be a document. See the comment on that row.
    assert _search("markdown") == {
        "note0000mark",
        "note0000mime",
        "note0000long",
        "note0000othr",
    }


def test_pdf_finds_the_mime_and_the_extension(search_db):
    assert _search("pdf") == {"pdf00000mime", "pdf00000ext0"}


def test_extension_match_is_case_insensitive(search_db):
    # ``plan.MD`` and ``scan.PDF`` are in the two sets above. Named
    # separately because a ``LIKE`` without ``lower()`` passes every
    # other assertion here on SQLite and fails on a case-sensitive
    # collation.
    assert "note0000mime" in _search("markdown")
    assert "pdf00000ext0" in _search("pdf")


def test_document_is_the_flat_kind_and_nothing_more(search_db):
    # ``document`` is a flat kind: it matches the column, not the mime
    # or the name. That markdown and PDFs come back with it is core's
    # classifier's doing — it files them under document — and is not
    # something this filter can be asked to demonstrate. So the claim
    # here is the narrow, checkable one: the rows whose column says
    # document, and only those. ``note0000othr`` is filed ``other``
    # despite its ``.md`` name, and stays out.
    assert _search("document") == {
        "note0000mark",
        "note0000mime",
        "note0000long",
        "pdf00000mime",
        "pdf00000ext0",
        "doc00000othr",
    }


def test_flat_kinds_are_unchanged(search_db):
    assert _search("video") == {"video0000001"}


def test_every_find_hint_names_a_kind_that_can_match(search_db):
    # Find's query decomposer has its own label set, and "text" is not
    # a ``File.file_type`` — core files text documents as ``document``.
    # Passing it through narrowed every Find hinted that way to nothing,
    # silently. Seeded rows cover one file of each kind a hint can name,
    # so a label that cannot match anything fails here.
    from app.rag.query_decomposer import _FILE_TYPE_LABELS
    from app.rag.service import _find_kind_for_hint

    unmatched = []
    for label in sorted(_FILE_TYPE_LABELS):
        kind = _find_kind_for_hint(label)
        if kind is None:
            continue  # "none" means no filter, which is not a failure
        if not _search(kind):
            unmatched.append(f"{label} -> {kind}")
    assert unmatched == []


def test_unknown_kind_returns_nothing_rather_than_everything(search_db):
    # An unrecognised value narrows to ``file_type == kind``, which
    # matches nothing — the same answer core gives. Falling through to
    # "no filter" would quietly widen a search the user narrowed.
    assert _search("nonsense") == set()
