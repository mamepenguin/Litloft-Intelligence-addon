"""Picture-text guards that the mutation run found untested.

SPEC-ID: SPEC-ADDON-067
"""

from __future__ import annotations

import pytest

from app.extractors.base import ContentExtractor
from tests.extraction_fixtures import (
    api,
    search_db,
    HEADERS,
    build_api,
    seed_file,
    seed_picture,
)
from tests.ocr_fixtures import install, picture, pillow_real  # noqa: F401

FILE = "picture0101"
DOC = "picture0102"


def _get(client, file_id: str = FILE):
    return client.get(f"/files/{file_id}/picture-text", headers=HEADERS)


def _put(client, text: str, expected: str, file_id: str = FILE):
    return client.put(
        f"/files/{file_id}/picture-text",
        json={"text": text, "expected_text": expected},
        headers=HEADERS,
    )


class TestOnlyPictures:
    def test_a_document_with_text_rows_is_not_a_picture(self, search_db, api):
        """SPEC-ADDON-067 B1/B4: the section belongs to pictures only."""
        client, _ = api
        seed_file(
            search_db, DOC, file_type="document", mime_type="application/pdf"
        )
        for table in ("fts_text_content", "fts_text_content_word"):
            search_db.execute(
                f"INSERT INTO {table}(file_id, chunk_index, page, text) "
                "VALUES(:f, '0', '1', 'body text')",
                f=DOC,
            )

        assert _get(client, DOC).status_code == 404
        assert _put(client, "changed", "body text", DOC).status_code == 404
        rows = search_db.fts("fts_text_content", DOC)
        assert [row[1] for row in rows] == ["body text"]


class TestLegacyMultiChunkText:
    def test_a_long_legacy_picture_rebuilds_its_whole_text(self, search_db, api):
        """SPEC-ADDON-067 B8, I11."""
        client, _ = api
        long_text = "".join(
            f"line {index} words here " for index in range(120)
        ).strip()
        assert len(long_text) > 1000
        seed_picture(search_db, FILE, ContentExtractor.chunk_text(long_text))

        response = _get(client)

        assert response.status_code == 200, response.text
        assert response.json()["text"] == long_text


class TestOcrStoresTheJoinedText:
    @pytest.mark.usefixtures("pillow_real")
    def test_a_long_read_stores_the_exact_join_it_chunked(
        self, search_db, tmp_path, monkeypatch, api
    ):
        """SPEC-ADDON-067 B8, I11."""
        client, _ = api
        path = picture(tmp_path / "long.png")
        seed_file(
            search_db, FILE, file_type="image", mime_type="image/png",
            file_path=str(path), ocr_indexed=False,
        )
        lines = [f"recognised line {index} " * 4 for index in range(30)]
        assert len("\n".join(lines)) > 1000
        harness = install(monkeypatch, lines=lines)

        assert harness.module.index_ocr(FILE) is True

        payload = _get(client).json()
        assert payload["text"] == "\n".join(lines)
