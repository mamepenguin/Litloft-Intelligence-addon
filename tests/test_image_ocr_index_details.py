"""The OCR mode rides on the index-details payload (SPEC-ADDON-050 I10)."""

from __future__ import annotations

import pytest

from tests.ocr_fixtures import build_search_db, patch_features, seed_indexed_file


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    return build_search_db(tmp_path, monkeypatch)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["on_index", "manual", "false"])
async def test_index_details_carries_the_ocr_mode(
    search_db, tmp_path, monkeypatch, mode
):
    """SPEC-ADDON-050 I10: the dialog reads the mode from index-details."""
    picture = tmp_path / "slide.png"
    picture.write_bytes(b"\x89PNG\r\n\x1a\n")
    seed_indexed_file(search_db, "pic1", "image/png", picture)
    patch_features(monkeypatch, image_ocr=mode)

    from app.routers.files import get_index_details

    response = await get_index_details(file_id="pic1", drive="drive1")

    assert response.image_ocr == mode
