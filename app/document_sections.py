"""Section titles of sectioned documents (EPUB), keyed by section number."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from app.database import get_search_db

EPUB_MIME = "application/epub+zip"


def replace_document_sections(
    session: Session, file_id: str, titles: Iterable[tuple[int, str]],
) -> None:
    session.execute(
        sql_text("DELETE FROM document_sections WHERE file_id = :fid"),
        {"fid": file_id},
    )
    rows = [
        {"fid": file_id, "page": page, "title": title}
        for page, title in titles
    ]
    if rows:
        session.execute(
            sql_text(
                "INSERT INTO document_sections (file_id, page, title) "
                "VALUES (:fid, :page, :title)"
            ),
            rows,
        )


def load_section_titles(
    session: Session, pairs: Iterable[tuple[str, int]],
) -> dict[tuple[str, int], str]:
    wanted = frozenset(pairs)
    file_ids = sorted({file_id for file_id, _ in wanted})
    if not file_ids:
        return {}
    placeholders = ", ".join(f":f{i}" for i in range(len(file_ids)))
    rows = session.execute(
        sql_text(
            "SELECT file_id, page, title FROM document_sections "
            f"WHERE file_id IN ({placeholders})"
        ),
        {f"f{i}": fid for i, fid in enumerate(file_ids)},
    ).fetchall()
    return {
        (row[0], int(row[1])): row[2]
        for row in rows
        if (row[0], int(row[1])) in wanted
    }


def section_title(file_id: str, page: int | None) -> str | None:
    if page is None:
        return None
    with get_search_db() as session:
        return load_section_titles(session, [(file_id, page)]).get((file_id, page))
