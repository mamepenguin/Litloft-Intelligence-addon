"""Builders for EPUB test books."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

OPF_NS = "http://www.idpf.org/2007/opf"

XHTML_DOCTYPE = (
    '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" '
    '"http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">'
)


@dataclass(frozen=True)
class Item:
    id: str
    href: str
    content: str | bytes | None
    media_type: str = "application/xhtml+xml"
    properties: str = ""


def xhtml(body: str, *, doctype: str = "") -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"{doctype}"
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops">'
        f"<head><title>t</title></head><body>{body}</body></html>"
    )


def nav_doc(entries: list[tuple[str, str]], *, extra_navs: str = "") -> str:
    links = "".join(f'<li><a href="{href}">{label}</a></li>' for href, label in entries)
    return xhtml(
        f"{extra_navs}"
        f'<nav epub:type="toc"><h1>Contents</h1><ol>{links}</ol></nav>'
    )


def ncx_doc(entries: list[tuple[str, str]]) -> str:
    points = "".join(
        f'<navPoint id="p{i}" playOrder="{i}">'
        f"<navLabel><text>{escape(label)}</text></navLabel>"
        f'<content src="{href}"/></navPoint>'
        for i, (href, label) in enumerate(entries, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE ncx PUBLIC "-//NISO//DTD ncx 2005-1//EN" '
        '"http://www.daisy.org/z3986/2005/ncx-2005-1.dtd">'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        f"<navMap>{points}</navMap></ncx>"
    )


def container_xml(opf_path: str) -> str:
    return (
        '<?xml version="1.0"?>'
        '<container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        f'<rootfiles><rootfile full-path="{opf_path}" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )


def opf_xml(
    items: list[Item],
    spine: list[str],
    *,
    spine_toc: str | None = None,
    spine_extra: str = "",
    package_extra: str = "",
) -> str:
    manifest = "".join(
        f'<item id="{i.id}" href="{i.href}" media-type="{i.media_type}"'
        + (f' properties="{i.properties}"' if i.properties else "")
        + "/>"
        for i in items
    )
    toc_attr = f' toc="{spine_toc}"' if spine_toc else ""
    itemrefs = "".join(f'<itemref idref="{idref}"/>' for idref in spine)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<package xmlns="{OPF_NS}" version="3.0" unique-identifier="id">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        '<dc:identifier id="id">x</dc:identifier></metadata>'
        f"<manifest>{manifest}</manifest>"
        f"<spine{toc_attr}>{itemrefs}{spine_extra}</spine>"
        f"{package_extra}</package>"
    )


class _Cp932ZipInfo(zipfile.ZipInfo):
    def _encodeFilenameFlags(self):  # noqa: N802 - zipfile's own name
        return self.filename.encode("cp932"), self.flag_bits & ~0x800


def make_epub(
    path: Path,
    items: list[Item],
    spine: list[str] | None = None,
    *,
    opf_dir: str = "OEBPS",
    spine_toc: str | None = None,
    opf: str | None = None,
    container: str | None = None,
    extra: dict[str, str | bytes] | None = None,
    cp932_names: bool = False,
    encrypted_members: tuple[str, ...] = (),
) -> Path:
    """Write an EPUB. ``spine`` defaults to every XHTML item in order."""
    opf_path = f"{opf_dir}/content.opf" if opf_dir else "content.opf"
    prefix = f"{opf_dir}/" if opf_dir else ""
    spine_ids = spine if spine is not None else [
        i.id for i in items if i.media_type == "application/xhtml+xml"
    ]
    members: list[tuple[str, str | bytes]] = [
        ("mimetype", "application/epub+zip"),
        (
            "META-INF/container.xml",
            container if container is not None else container_xml(opf_path),
        ),
        (opf_path, opf if opf is not None else opf_xml(items, spine_ids, spine_toc=spine_toc)),
        *(
            (prefix + i.href, i.content)
            for i in items
            if i.content is not None
        ),
        *((extra or {}).items()),
    ]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            info = (
                _Cp932ZipInfo(name) if cp932_names and not name.isascii()
                else zipfile.ZipInfo(name)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    for name in encrypted_members:
        _set_encrypted_flag(path, name)
    return path


def _set_encrypted_flag(path: Path, name: str) -> None:
    # zipfile cannot write an encrypted member, so the flag is set in both
    # headers afterwards; reading the member then needs a password.
    data = bytearray(path.read_bytes())
    wanted = name.encode()
    for signature, name_len_at, name_at, flag_at in (
        (b"PK\x03\x04", 26, 30, 6),
        (b"PK\x01\x02", 28, 46, 8),
    ):
        offset = data.find(signature)
        while offset != -1:
            length = int.from_bytes(data[offset + name_len_at:offset + name_len_at + 2], "little")
            if bytes(data[offset + name_at:offset + name_at + length]) == wanted:
                data[offset + flag_at] |= 0x1
            offset = data.find(signature, offset + 4)
    path.write_bytes(bytes(data))
