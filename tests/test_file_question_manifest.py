"""SPEC-ADDON-053 and SPEC-ADDON-056: what the manifest declares to the core.

The core proxy gates and streams from these fields, and the inspector draws the
tab from the slot entry, so a missing or changed field is behaviour, not
documentation.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASK_PATH = "/files/{file_id}/ask"


def _manifest() -> dict:
    return json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))


def _routes() -> list[dict]:
    return _manifest()["proxy"]["routes"]


def _ask_routes() -> list[dict]:
    return [route for route in _routes() if route["path"] == ASK_PATH]


def test_the_file_question_route_is_a_get_and_a_post() -> None:
    """SPEC-ADDON-053 6: one path serves availability and the question."""
    assert sorted(
        [method for route in _ask_routes() for method in route["methods"]]
    ) == ["GET", "POST"]


def test_both_routes_check_file_access_and_the_rag_feature() -> None:
    """SPEC-ADDON-053 6 and I1: the file must be reachable and the drive's rag on."""
    routes = _ask_routes()
    assert len(routes) == 2
    for route in routes:
        assert route["pre_check"] == {"type": "file_access", "param": "file_id"}
        assert route["addon_feature"] == "rag"


def test_both_routes_filter_nothing_out_of_the_response() -> None:
    """SPEC-ADDON-053 6: citations carry positions, not drives, so nothing is filtered."""
    routes = _ask_routes()
    assert len(routes) == 2
    for route in routes:
        assert route["response_filter"] is None


def test_the_question_route_streams_and_takes_no_timeout() -> None:
    """SPEC-ADDON-053 6: the proxy passes a stream through and applies no timeout to it."""
    post = next(route for route in _ask_routes() if route["methods"] == ["POST"])
    get = next(route for route in _ask_routes() if route["methods"] == ["GET"])

    assert post["stream"] is True
    assert "timeout" not in post
    assert not get.get("stream")


def test_the_ask_tab_is_an_inspector_tabs_entry_with_a_translation() -> None:
    """SPEC-ADDON-056 I7: the tab is an inspector-tabs entry with an i18n_key."""
    entries = _manifest()["slots"]["inspector-tabs"]

    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry["id"], str) and entry["id"]
    assert isinstance(entry["label"], str) and entry["label"]
    assert isinstance(entry["i18n_key"], str) and entry["i18n_key"]
