from __future__ import annotations

import zipfile

import pytest

from app.utils.zip_names import decode_zip_filename


def _info(filename: str, flag_bits: int = 0) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename)
    info.flag_bits = flag_bits
    return info


@pytest.mark.parametrize("name", ["本文/第一章.xhtml", "Übersicht.xhtml"])
def test_utf8_flag_kept(name: str) -> None:
    assert decode_zip_filename(_info(name, flag_bits=0x800)) == name


def test_cp932_name_redecoded() -> None:
    as_read_by_zipfile = "本文/第一章.xhtml".encode("cp932").decode("cp437")

    assert decode_zip_filename(_info(as_read_by_zipfile)) == "本文/第一章.xhtml"


def test_ascii_passthrough() -> None:
    assert decode_zip_filename(_info("OEBPS/text/ch01.xhtml")) == "OEBPS/text/ch01.xhtml"


def test_invalid_cp932_falls_back() -> None:
    as_read_by_zipfile = b"bad\x81 name.xhtml".decode("cp437")

    assert decode_zip_filename(_info(as_read_by_zipfile)) == as_read_by_zipfile
