"""The core's file-kind vocabulary, applied to ``IndexedFile``.

Core's ``?type=`` filter names eight kinds — the six flat ``file_type``
values plus ``markdown`` and ``pdf`` nested under ``document`` — and the
search toolbar is the same control as the folder toolbar. ``IndexedFile``
is a snapshot of core's ``File`` and carries the flat ``file_type``
column, so ``file_type == "markdown"`` matches nothing at all: not an
error, just an empty result, which reads as "semantic search found
nothing about this" rather than "this filter cannot be honoured here".

This is the second implementation of the classifier that
``backend/app/routers/drives.py`` owns (``_KIND_MIMES`` /
``_KIND_SUFFIXES`` / ``_apply_kind_filter``). The addon runs in its own
container and cannot import core, the same situation as the two
``frontmatter.py`` parsers. Unlike those, the drift here is caught
mechanically: ``frontend/src/__tests__/file-kind-parity.test.ts`` in the
core repository reads ``_KIND_MIMES`` / ``_KIND_SUFFIXES`` out of
``drives.py`` and ``KIND_MIMES`` / ``KIND_SUFFIXES`` out of this file
and compares them. Renaming either pair breaks that test rather than
slipping past it.

The suffix fallback is not decoration: rows whose mime was never
recorded are exactly the ones the two old filters used to disagree
about.
"""

from sqlalchemy import func, or_

from app.models import IndexedFile

# Parity-checked against core's ``_KIND_MIMES`` / ``_KIND_SUFFIXES``.
KIND_MIMES: dict[str, tuple[str, ...]] = {
    "markdown": ("text/markdown",),
    "pdf": ("application/pdf",),
}
KIND_SUFFIXES: dict[str, tuple[str, ...]] = {
    "markdown": (".md", ".markdown"),
    "pdf": (".pdf",),
}


def apply_kind_filter(query, kind: str | None):
    """Narrow a query over ``IndexedFile`` to one kind of the vocabulary.

    ``kind`` is whatever the toolbar sent. An unknown value narrows to
    ``file_type == kind`` and returns nothing, which is what core does
    with it too.
    """
    if not kind:
        return query

    mimes = KIND_MIMES.get(kind)
    if mimes is None:
        return query.filter(IndexedFile.file_type == kind)

    # A nested kind. ``file_type == "document"`` is deliberately not
    # also required: the nesting holds because core's classifier files
    # both under document, and demanding it here would drop precisely
    # the rows the suffix fallback exists for — one whose mime was
    # never recorded may well carry the wrong ``file_type`` too.
    suffixes = KIND_SUFFIXES[kind]
    return query.filter(
        or_(
            IndexedFile.mime_type.in_(mimes),
            *(
                func.lower(IndexedFile.filename).like(f"%{suffix}")
                for suffix in suffixes
            ),
        )
    )
