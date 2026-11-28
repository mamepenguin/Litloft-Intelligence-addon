"""SPEC-ADDON-193 I14: the images install yt-dlp with curl_cffi, and can impersonate.

The test image is the one these tests run in: ``curl_cffi`` must import and
yt-dlp must list at least one impersonation target. The image definitions are
read from the source tree where it is mounted beside ``tests/``; the test image
itself copies neither file, so that half skips there.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1]
MINIMUM_VERSION = (2026, 10, 26)


@pytest.mark.parametrize("name", ["Dockerfile", "Dockerfile.test"])
def test_spec_addon_193_i14_both_images_install_yt_dlp_with_curl_cffi(name):
    path = SOURCE_ROOT / name
    if not path.exists():
        pytest.skip(f"{name} is not copied into the test image")

    text = path.read_text()
    installed = re.findall(r'"(yt-dlp\[[^\]]*\][^"]*)"', text)

    assert installed, f"{name} installs no yt-dlp extra"
    assert any(
        "yt-dlp[default,curl-cffi]" in line and "2026.10.26" in line
        for line in installed
    ), installed


def test_spec_addon_193_i14_the_test_image_can_impersonate():
    import curl_cffi  # noqa: F401 - the extra that gives yt-dlp impersonation
    import yt_dlp
    from yt_dlp import YoutubeDL

    version = tuple(int(part) for part in re.findall(r"\d+", yt_dlp.version.__version__))
    assert version[:3] >= MINIMUM_VERSION

    targets = YoutubeDL({"quiet": True})._get_available_impersonate_targets()
    assert targets, "yt-dlp lists no impersonation target"
    assert all(handler == "curl_cffi" for _, handler in targets)
