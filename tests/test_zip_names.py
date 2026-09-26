from __future__ import annotations

import zipfile

import pytest

from app.utils.zip_names import decode_zip_filename


def _info(filename: str, flag_bits: int = 0) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename)
    info.flag_bits = flag_bits
    return info


_CP932_AS_READ = "本文/第一章.xhtml".encode("cp932").decode("cp437")
_INVALID_CP932_AS_READ = b"bad\x81 name.xhtml".decode("cp437")


@pytest.mark.parametrize(
    ("name", "flag_bits", "expected"),
    [
        pytest.param("本文/第一章.xhtml", 0x800, "本文/第一章.xhtml", id="utf8-flag"),
        pytest.param("Übersicht.xhtml", 0x800, "Übersicht.xhtml", id="utf8-flag-cp437-range"),
        pytest.param(_CP932_AS_READ, 0, "本文/第一章.xhtml", id="cp932-redecoded"),
        pytest.param(_INVALID_CP932_AS_READ, 0, _INVALID_CP932_AS_READ, id="invalid-cp932-kept"),
    ],
)
def test_decode_zip_filename(name: str, flag_bits: int, expected: str) -> None:
    assert decode_zip_filename(_info(name, flag_bits=flag_bits)) == expected
