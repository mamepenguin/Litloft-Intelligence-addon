"""The navigation entry this addon declares to the core sidebar."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_declares_the_ask_entry() -> None:
    assert _load("manifest.json")["navigation"] == {
        "label": "Ask",
        "i18n_key": "intelligence.nav.label",
        "icon": "message-circle-question",
        "placement": "primary",
        "priority": 10,
    }


def test_keeps_the_generated_route() -> None:
    assert _load("manifest.json")["href"] == "/drive/{drive}/addons/intelligence"

