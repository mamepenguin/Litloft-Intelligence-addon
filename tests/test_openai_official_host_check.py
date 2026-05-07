"""Strict host check for the official OpenAI Whisper endpoint.

The 25 MB pre-check fires only when the configured ``base_url``
points at the official ``api.openai.com`` endpoint — Groq /
Fireworks / self-hosted compatibles must NOT be blocked because
they have no equivalent cap.

Previously the implementation used a substring match
(``"api.openai.com" in cfg.base_url``), which mis-classified

    https://api.openai.com.attacker.com/v1
    https://proxy.example.com/api.openai.com

as "official". The fix uses ``urlparse(...).hostname`` exact-match
(see hako ``tJV51mfYZWLqMBIHm9Qvi``).

These tests parametrize the boundary cases.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.workers.transcription.openai_compatible import OpenAICompatibleProvider


@pytest.fixture()
def with_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    yield


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        # Legitimate official endpoints (path / port / scheme variants).
        ("https://api.openai.com/v1", True),
        ("https://api.openai.com", True),
        ("https://api.openai.com:443/v1", True),
        ("http://api.openai.com/v1", True),  # http for completeness
        # Hostile / boundary cases — must NOT match.
        ("https://api.openai.com.attacker.com/v1", False),
        ("https://attacker.com/api.openai.com/v1", False),
        ("https://proxy.example.com/api.openai.com", False),
        ("https://my-api.openai.com.evil.test/v1", False),
        ("https://API-OPENAI.COM/v1", False),  # different host entirely
        # Legitimate self-hosted / managed compatibles — must NOT match.
        ("https://api.groq.com/openai/v1", False),
        ("https://api.fireworks.ai/inference/v1", False),
        ("http://localhost:8080/v1", False),
        ("https://whisper.internal/v1", False),
        # Edge cases.
        ("", False),
        ("not-a-url", False),
    ],
    ids=lambda v: str(v),
)
def test_is_openai_official_strict_host_match(
    with_api_key, base_url, expected
) -> None:
    """``_is_openai_official`` must use exact hostname equality.

    Substring matching would falsely mark
    ``https://api.openai.com.attacker.com/v1`` as official and
    pre-block 26 MB uploads on what is actually a third-party
    endpoint with no 25 MB cap.
    """
    with patch(
        "app.workers.transcription.openai_compatible.config"
    ) as fake_config:
        fake_config.settings.transcription.openai_compatible.base_url = base_url
        fake_config.settings.transcription.openai_compatible.model = "whisper-1"
        fake_config.settings.transcription.openai_compatible.timeout_s = 600
        provider = OpenAICompatibleProvider()
        assert provider._is_openai_official is expected, (
            f"base_url={base_url!r} expected official={expected}, "
            f"got {provider._is_openai_official}"
        )
