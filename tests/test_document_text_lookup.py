"""SPEC-ADDON-036: lookup, parameters, failures and side effects of
``GET /files/{file_id}/document-text``."""

from __future__ import annotations

import builtins
import io
import json
import logging
import os
from pathlib import Path

import pytest

from tests.document_fixtures import (
    DOCX,
    EPUB,
    PDF,
    PPTX,
    XLSX,
    make_docx,
    make_pdf,
    make_pptx,
    make_xlsx,
    sparse_file,
)
from tests.document_text_api import (  # noqa: F401
    DRIVE,
    OTHER,
    api,
    reader,
    search_index,
    seed,
    statements,
)
from tests.epub_fixtures import Item, make_epub, xhtml

MANIFEST = Path(__file__).resolve().parent.parent / "manifest.json"
MB = 1024 * 1024


def _epub(tmp_path: Path, name: str = "b.epub", sections: int = 2) -> Path:
    return make_epub(
        tmp_path / name,
        [Item(f"c{i}", f"c{i}.xhtml", xhtml(f"<p>MARK-{i}</p>")) for i in range(1, sections + 1)],
    )


class TestLookup:
    @pytest.mark.parametrize(
        "case", ["unknown", "inactive", "other-drive"],
    )
    def test_spec_addon_036_unknown_inactive_or_other_drive_is_404_and_opens_nothing(
        self, reader, tmp_path, monkeypatch, case,
    ):
        sentinel = _epub(tmp_path, "sentinel.epub")
        if case == "inactive":
            reader.register("f1", sentinel, EPUB, active=False)
        elif case == "other-drive":
            reader.register("f1", sentinel, EPUB, drive=OTHER)
        touched: list[str] = []
        real_open, real_io_open, real_stat = builtins.open, io.open, os.stat

        def spy_open(file, *args, **kwargs):
            touched.append(str(file))
            return real_open(file, *args, **kwargs)

        def spy_io_open(file, *args, **kwargs):
            touched.append(str(file))
            return real_io_open(file, *args, **kwargs)

        def spy_stat(path, *args, **kwargs):
            touched.append(str(path))
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", spy_open)
        monkeypatch.setattr(io, "open", spy_io_open)
        monkeypatch.setattr(os, "stat", spy_stat)

        response = reader.get("f1")

        assert response.status_code == 404, response.text
        assert response.json()["detail"] == "File not found"
        assert str(sentinel) not in touched

    def test_spec_addon_036_the_same_file_reads_in_its_own_drive(self, reader, tmp_path):
        reader.register("f1", _epub(tmp_path), EPUB, drive=OTHER)

        response = reader.get("f1", drive=OTHER)

        assert response.status_code == 200, response.text
        assert response.json()["file_id"] == "f1"

    @pytest.mark.parametrize("mime", ["text/plain", "image/png", "application/msword"])
    def test_spec_addon_036_an_unsupported_mime_is_415_naming_the_supported_types(
        self, reader, tmp_path, mime,
    ):
        path = tmp_path / "x.bin"
        path.write_bytes(b"x")
        reader.register("f1", path, mime)

        response = reader.get("f1")

        assert response.status_code == 415, response.text
        detail = json.dumps(response.json()["detail"]).lower()
        assert "pdf" in detail
        assert "epub" in detail


def _kinds(tmp_path: Path) -> dict[str, tuple[Path, str]]:
    return {
        "pdf": (make_pdf(tmp_path / "a.pdf", ["MARK-p1"]), PDF),
        "pptx": (make_pptx(tmp_path / "a.pptx", [(None, ["MARK-s1"])]), PPTX),
        "xlsx": (make_xlsx(tmp_path / "a.xlsx", [("Data", [["MARK-x1"]])]), XLSX),
        "epub": (_epub(tmp_path), EPUB),
        "docx": (make_docx(tmp_path / "a.docx", paragraphs=("MARK-d1",)), DOCX),
    }


class TestParameters:
    @pytest.mark.parametrize(
        "params",
        [
            pytest.param({"start": 0}, id="start-0"),
            pytest.param({"start": -1}, id="start-negative"),
            pytest.param({"start": "one"}, id="start-not-int"),
            pytest.param({"count": 0}, id="count-0"),
            pytest.param({"count": "x"}, id="count-not-int"),
            pytest.param({"offset": -1}, id="offset-negative"),
            pytest.param({"max_chars": 0}, id="max-chars-0"),
            pytest.param({"max_chars": 200_001}, id="max-chars-200001"),
            pytest.param({"max_chars": 1.5}, id="max-chars-fraction"),
        ],
    )
    def test_spec_addon_036_out_of_bounds_parameters_are_422(self, reader, tmp_path, params):
        reader.register("f1", _epub(tmp_path), EPUB)

        assert reader.get("f1", **params).status_code == 422

    @pytest.mark.parametrize(
        "kind,maximum",
        [("pdf", 10), ("pptx", 200), ("xlsx", 50), ("epub", 100), ("docx", 1)],
    )
    def test_spec_addon_036_count_above_the_kind_maximum_is_422_and_the_maximum_is_accepted(
        self, reader, tmp_path, kind, maximum,
    ):
        path, mime = _kinds(tmp_path)[kind]
        reader.register("f1", path, mime)

        assert reader.get("f1", count=maximum + 1).status_code == 422
        assert reader.get("f1", count=maximum).status_code == 200

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param({"start": 1}, id="start-1"),
            pytest.param({"count": 1}, id="count-1"),
            pytest.param({"offset": 0}, id="offset-0"),
            pytest.param({"max_chars": 1}, id="max-chars-1"),
            pytest.param({"max_chars": 200_000}, id="max-chars-200000"),
        ],
    )
    def test_spec_addon_036_boundary_parameters_are_accepted(self, reader, tmp_path, params):
        reader.register("f1", _epub(tmp_path), EPUB)

        assert reader.get("f1", **params).status_code == 200


def _corrupt(tmp_path: Path, suffix: str) -> Path:
    path = tmp_path / f"corrupt{suffix}"
    path.write_bytes(b"PK\x03\x04 this is not a document at all" * 40)
    return path


def _unreadable_cases():
    return [
        pytest.param(lambda t: t / "gone.pdf", PDF, id="pdf-missing-on-disk"),
        pytest.param(lambda t: t / "gone.docx", DOCX, id="docx-missing-on-disk"),
        pytest.param(lambda t: t / "gone.epub", EPUB, id="epub-missing-on-disk"),
        pytest.param(lambda t: _corrupt(t, ".pdf"), PDF, id="pdf-corrupt"),
        pytest.param(lambda t: _corrupt(t, ".docx"), DOCX, id="docx-corrupt"),
        pytest.param(lambda t: _corrupt(t, ".xlsx"), XLSX, id="xlsx-corrupt"),
        pytest.param(lambda t: _corrupt(t, ".pptx"), PPTX, id="pptx-corrupt"),
        pytest.param(lambda t: _corrupt(t, ".epub"), EPUB, id="epub-corrupt"),
        pytest.param(
            lambda t: make_pdf(t / "locked.pdf", ["MARK-secret"], user_password="pw"),
            PDF, id="pdf-password-protected",
        ),
        pytest.param(lambda t: sparse_file(t / "big.pdf", 200 * MB + 1), PDF, id="pdf-over-200mb"),
        pytest.param(lambda t: sparse_file(t / "big.docx", 50 * MB + 1), DOCX, id="docx-over-50mb"),
        pytest.param(lambda t: sparse_file(t / "big.xlsx", 50 * MB + 1), XLSX, id="xlsx-over-50mb"),
        pytest.param(lambda t: sparse_file(t / "big.pptx", 100 * MB + 1), PPTX, id="pptx-over-100mb"),
        pytest.param(_no_container, EPUB, id="epub-without-container"),
        pytest.param(_no_package, EPUB, id="epub-without-package-document"),
    ]


def _no_container(tmp_path: Path) -> Path:
    import zipfile

    path = tmp_path / "nocontainer.epub"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("OEBPS/content.opf", "<package/>")
        zf.writestr("OEBPS/c1.xhtml", xhtml("<p>MARK-1</p>"))
    return path


def _no_package(tmp_path: Path) -> Path:
    from tests.epub_fixtures import container_xml

    return make_epub(
        tmp_path / "nopackage.epub",
        [Item("c1", "c1.xhtml", xhtml("<p>MARK-1</p>"))],
        container=container_xml("OEBPS/elsewhere.opf"),
    )


class TestUnreadable:
    @pytest.mark.parametrize("build,mime", _unreadable_cases())
    def test_spec_addon_036_an_unreadable_file_is_422_and_logged_without_its_text(
        self, reader, tmp_path, caplog, build, mime,
    ):
        reader.register("unreadable-id", build(tmp_path), mime)
        caplog.set_level(logging.DEBUG)

        response = reader.get("unreadable-id")

        assert response.status_code == 422, response.text
        assert response.json()["detail"] == "Document could not be read"
        warnings = [
            r for r in caplog.records
            if r.name.startswith("app") and r.levelno >= logging.WARNING
        ]
        assert any("unreadable-id" in r.getMessage() for r in warnings)
        assert not any("MARK-" in r.getMessage() for r in caplog.records)


def _tree(tmp_path: Path) -> list[Path]:
    import app.config as config

    roots = [tmp_path, Path(config.settings.intelligence_data_dir)]
    return sorted(
        p for root in roots if root.exists() for p in root.rglob("*")
        if not p.name.startswith("search.db")
    )


class TestSideEffects:
    def test_spec_addon_036_reads_write_nothing_and_call_no_llm(
        self, reader, tmp_path, statements,
    ):
        kinds = _kinds(tmp_path)
        for kind, (path, mime) in kinds.items():
            reader.register(kind, path, mime)
        before = _tree(tmp_path)
        statements.reset()

        for kind in kinds:
            body = reader.ok(kind)
            assert body["units"], kind

        after = _tree(tmp_path)
        assert statements.statements, "the endpoint must read the search DB"
        assert statements.writes() == []
        assert after == before

    def test_spec_addon_036_extraction_never_reads_the_file_on_the_event_loop(
        self, reader, tmp_path, monkeypatch,
    ):
        import asyncio
        import zipfile

        reader.register("f1", _epub(tmp_path), EPUB)
        on_loop: list[str] = []
        opened: list[str] = []
        real_open = zipfile.ZipFile.open

        def spy(self, name, *args, **kwargs):
            opened.append(str(name))
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                on_loop.append(str(name))
            return real_open(self, name, *args, **kwargs)

        monkeypatch.setattr(zipfile.ZipFile, "open", spy)

        reader.ok("f1")

        assert opened
        assert on_loop == []


class TestRouteDeclaration:
    def test_spec_addon_036_manifest_route_has_the_file_access_precheck_and_a_25s_timeout(self):
        manifest = json.loads(MANIFEST.read_text())
        entries = [
            r for r in manifest["proxy"]["routes"]
            if r.get("path") == "/files/{file_id}/document-text"
        ]

        assert len(entries) == 1
        entry = entries[0]
        assert entry["methods"] == ["GET"]
        assert entry["pre_check"] == {"type": "file_access", "param": "file_id"}
        assert entry["timeout"] == 25
