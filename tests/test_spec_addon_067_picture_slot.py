"""Registration the picture-text section and the new routes need.

SPEC-ID: SPEC-ADDON-067

The addon proxy only forwards a route the manifest declares, and the file
page only draws a section the manifest lists. The slot table's own entry is
checked by ``PictureTextSection.test.tsx``, where the component can render.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _manifest() -> dict:
    return json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))


def _routes() -> list[dict]:
    return _manifest()["proxy"]["routes"]


def test_picture_text_is_a_file_detail_section_after_visual_description():
    """SPEC-ADDON-067 B1."""
    sections = _manifest()["slots"]["file-detail-sections"]
    by_id = {section["id"]: section for section in sections}

    assert by_id["picture-text"]["priority"] == 10
    assert by_id["visual-description"]["priority"] == 9
    assert by_id["picture-text"]["priority"] > by_id["visual-description"]["priority"]


def test_the_new_routes_are_driven_and_file_checked():
    """SPEC-ADDON-067/068 §6."""
    routes: dict[str, dict] = {}
    for route in _routes():
        routes.setdefault(route["path"], {"methods": set(), "route": route})
        routes[route["path"]]["methods"].update(route["methods"])

    expected = [
        ("/files/{file_id}/transcript/utterances/{index}", {"PUT"}),
        ("/files/{file_id}/transcript/restore", {"POST"}),
        ("/files/{file_id}/picture-text", {"GET", "PUT"}),
        ("/files/{file_id}/picture-text/restore", {"POST"}),
    ]

    for path, methods in expected:
        entry = routes[path]
        assert methods <= entry["methods"], path
        assert entry["route"]["require_drive"] is True, path
        assert entry["route"]["pre_check"] == {
            "type": "file_access",
            "param": "file_id",
        }, path
