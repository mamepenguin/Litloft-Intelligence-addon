"""Correct one picture's text (SPEC-ADDON-067).

SPEC-ID: SPEC-ADDON-067

Uses ``GET`` / ``PUT /files/{id}/picture-text`` and the OCR worker, so the
stored string, its chunking and the legacy rebuild are asserted where the
spec puts them.
"""

from __future__ import annotations

import pytest

from app.extractors.base import ContentExtractor
from tests.ocr_fixtures import pillow_real  # noqa: F401 - fixture used by name
from tests.extraction_fixtures import (
    api,
    search_db,
    DRIVE,
    HEADERS,
    OTHER_DRIVE,
    SearchDB,
    build_api,
    coded_detail,
    seed_file,
    seed_picture,
)

FILE = "picture00001"
OTHER = "picture00002"


def _get(client, file_id: str = FILE, drive: str = DRIVE):
    return client.get(f"/files/{file_id}/picture-text", headers={"X-Lit-Drive": drive})


def _put(client, text: str, expected_text: str, file_id: str = FILE, drive: str = DRIVE):
    return client.put(
        f"/files/{file_id}/picture-text",
        json={"text": text, "expected_text": expected_text},
        headers={"X-Lit-Drive": drive},
    )


def _picture_chunks(db: SearchDB, file_id: str = FILE) -> list[tuple]:
    return db.rows(
        "SELECT CAST(chunk_index AS INTEGER), page, text FROM fts_text_content "
        "WHERE file_id = :f ORDER BY CAST(chunk_index AS INTEGER)",
        f=file_id,
    )


class TestPictureTextAnswer:
    def test_a_legacy_picture_rebuilds_its_text_on_the_first_read(self, search_db, api):
        """SPEC-ADDON-067 B8, D."""
        client, _ = api
        seed_picture(search_db, FILE, ["会議は10時から"])

        response = _get(client)

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["file_id"] == FILE
        assert payload["text"] == "会議は10時から"
        assert payload["corrected"] is False
        assert payload["has_original"] is False
        assert payload["busy"] is False

    @pytest.mark.parametrize(
        "case",
        ["missing", "inactive", "other-drive", "not-a-picture", "no-text"],
    )
    def test_a_file_without_a_picture_text_answers_404(self, search_db, api, case):
        client, _ = api
        if case == "missing":
            pass
        elif case == "inactive":
            seed_picture(search_db, FILE, ["old"], active=False)
        elif case == "other-drive":
            seed_picture(search_db, FILE, ["old"], drive=OTHER_DRIVE)
        elif case == "not-a-picture":
            seed_file(search_db, FILE, file_type="video", mime_type="video/mp4")
        else:
            seed_picture(search_db, FILE, [])

        response = _get(client)

        assert response.status_code == 404

    def test_the_answer_does_not_depend_on_the_ocr_feature(
        self, search_db, api, monkeypatch
    ):
        """SPEC-ADDON-067 B1: the section works under every OCR mode."""
        import dataclasses

        import app.config as config
        import app.routers.files as files

        client, _ = api
        seed_picture(search_db, FILE, ["text"])
        features = dataclasses.replace(config.settings.features, image_ocr="false")
        monkeypatch.setattr(
            files, "settings", dataclasses.replace(config.settings, features=features)
        )

        assert _get(client).status_code == 200


class TestPictureSaveChecks:
    def test_a_busy_picture_answers_busy_before_checking_the_text(self, search_db, api):
        """SPEC-ADDON-067 B4."""
        client, _ = api
        seed_picture(search_db, FILE, ["old text"])
        assert _get(client).status_code == 200
        assert _put(client, "fixed", "old text").status_code == 200

        confirmed = client.post(
            f"/files/{FILE}/reindex",
            json={"tasks": ["ocr"], "overwrite_corrections": True},
            headers=HEADERS,
        )
        assert confirmed.status_code in (200, 202)

        response = _put(client, "different", "not the stored text")

        assert response.status_code == 409
        assert coded_detail(response) == {"code": "busy"}
        assert _get(client).json()["busy"] is True

    def test_changed_text_writes_nothing(self, search_db, api):
        client, _ = api
        seed_picture(search_db, FILE, ["old text"])
        assert _get(client).status_code == 200
        before = _picture_chunks(search_db)

        response = _put(client, "new text", "not the stored text")

        assert response.status_code == 409
        assert coded_detail(response) == {"code": "changed"}
        assert _picture_chunks(search_db) == before
        assert _get(client).json()["has_original"] is False

    def test_a_text_over_100000_code_points_is_too_long(self, search_db, api):
        client, _ = api
        seed_picture(search_db, FILE, ["old text"])
        assert _get(client).status_code == 200

        response = _put(client, "x" * 100001, "old text")

        assert response.status_code == 422
        assert coded_detail(response) == {"code": "too_long"}
        assert _get(client).json()["text"] == "old text"

    def test_changed_runs_before_too_long(self, search_db, api):
        client, _ = api
        seed_picture(search_db, FILE, ["old text"])
        assert _get(client).status_code == 200

        response = _put(client, "x" * 100001, "wrong")

        assert response.status_code == 409
        assert coded_detail(response) == {"code": "changed"}

    def test_the_request_body_is_normalised_before_the_comparison(self, search_db, api):
        """SPEC-ADDON-067 B3: line breaks stay, trailing space and outer blanks do not."""
        client, _ = api
        seed_picture(search_db, FILE, ["old text"])
        assert _get(client).status_code == 200

        response = _put(client, "\n\nfirst line   \nsecond line  \n\n", "old text")

        assert response.status_code == 200, response.text
        assert response.json()["text"] == "first line\nsecond line"

    def test_saving_the_same_text_writes_nothing(self, search_db, api):
        """SPEC-ADDON-067 B5."""
        client, _ = api
        seed_picture(search_db, FILE, ["old text"])
        assert _get(client).status_code == 200
        before = _picture_chunks(search_db)
        before_embeddings = search_db.embeddings(FILE, "text_content")

        response = _put(client, "old text  ", "old text")

        assert response.status_code == 200
        assert response.json()["text"] == "old text"
        assert _picture_chunks(search_db) == before
        assert search_db.embeddings(FILE, "text_content") == before_embeddings
        assert _get(client).json()["has_original"] is False


class TestPictureSaveEffects:
    def test_the_new_text_is_chunked_and_replaces_the_search_rows(self, search_db, api):
        """SPEC-ADDON-067 B6."""
        client, _ = api
        seed_picture(search_db, FILE, ["old text"], ocr_indexed=True)
        assert _get(client).status_code == 200

        response = _put(client, "new line one\nnew line two", "old text")

        assert response.status_code == 200, response.text
        expected = ContentExtractor.chunk_text("new line one\nnew line two")
        assert _picture_chunks(search_db) == [
            (index, "", chunk) for index, chunk in enumerate(expected)
        ]
        assert search_db.embeddings(FILE, "text_content") == []
        assert bool(search_db.flag(FILE, "ocr_indexed")) is True
        assert search_db.flag(FILE, "ocr_indexed") == 1
        payload = _get(client).json()
        assert payload["text"] == "new line one\nnew line two"
        assert payload["corrected"] is True
        assert payload["has_original"] is True

    def test_an_empty_text_is_kept_as_a_corrected_empty_picture(self, search_db, api):
        """SPEC-ADDON-067 B6/B7."""
        client, _ = api
        seed_picture(search_db, FILE, ["old text"])
        assert _get(client).status_code == 200

        response = _put(client, "", "old text")

        assert response.status_code == 200, response.text
        payload = _get(client)
        assert payload.status_code == 200
        assert payload.json()["text"] == ""
        assert payload.json()["corrected"] is True
        assert payload.json()["has_original"] is True
        assert _picture_chunks(search_db) == []
        assert search_db.embeddings(FILE, "text_content") == []


class TestOcrStoresThePictureText:
    @pytest.mark.usefixtures("pillow_real")
    def test_the_ocr_worker_stores_the_joined_text_it_chunked(
        self, search_db, tmp_path, monkeypatch, api
    ):
        """SPEC-ADDON-067 B8: the exact joined string is the picture text."""
        from tests.ocr_fixtures import install, picture

        client, _ = api
        path = picture(tmp_path / "sign.png")
        seed_file(
            search_db, FILE, file_type="image", mime_type="image/png",
            file_path=str(path), ocr_indexed=False,
        )
        lines = ["会議は10時から", "first line", "second line"]
        harness = install(monkeypatch, lines=lines)

        assert harness.module.index_ocr(FILE) is True

        payload = _get(client).json()
        assert payload["text"] == "\n".join(lines)
        assert payload["corrected"] is False

    @pytest.mark.usefixtures("pillow_real")
    def test_a_picture_with_a_corrected_text_is_not_read_again(
        self, search_db, tmp_path, monkeypatch, api
    ):
        """SPEC-ADDON-067 B6/C6: the worker's guard leaves the correction alone."""
        from tests.ocr_fixtures import install

        client, _ = api
        seed_picture(search_db, FILE, ["old text"])
        assert _get(client).status_code == 200
        assert _put(client, "fixed", "old text").status_code == 200
        before = _picture_chunks(search_db)
        harness = install(monkeypatch, lines=["engine text"])

        harness.module.index_ocr(FILE)

        assert harness.factory.calls == []
        assert _picture_chunks(search_db) == before
        assert _get(client).json()["text"] == "fixed"


class TestLegacyRebuildFailure:
    def test_a_failed_rebuild_stores_nothing_and_the_next_read_tries_again(
        self, search_db, monkeypatch
    ):
        """SPEC-ADDON-067 failure table B8."""
        client, _ = build_api(monkeypatch, raise_server_exceptions=False)
        seed_picture(search_db, FILE, ["old text"])
        search_db.execute("ALTER TABLE fts_text_content RENAME TO fts_off")

        failed = _get(client)
        assert failed.status_code == 500

        search_db.execute("ALTER TABLE fts_off RENAME TO fts_text_content")
        retried = _get(client)
        assert retried.status_code == 200
        assert retried.json()["text"] == "old text"
