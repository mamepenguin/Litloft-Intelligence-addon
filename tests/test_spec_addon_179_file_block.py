"""SPEC-ADDON-179 I1 I2 I3: what the File, Type and Folder lines carry.

Folder and Type are read from the prompt a PDF question sends (the
passages-only path) and from the ``Material`` ``load_material`` returns.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

import pytest

from app.rag.file_question import load_material
from tests.file_context_doubles import (
    BLOCKS,
    LOFT_MIME,
    PDF_ID,
    PDF_TEXT,
    RUN,
    SCENARIOS,
    TOP_OF_DRIVE,
    assert_s179_and_s181,
    block,
    question_bodies,
    system_of,
    today,
    user_text,
    without_additions,
)
from tests.file_question_doubles import DRIVE, chunk
from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    addon,
    call,
    db,
    llm,
    openai_config,
    vision_client,
)
from tests.frame_request_doubles import QUESTION
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture

FILENAME = f"{PDF_ID}.bin"
PDF = "application/pdf"
NFD_FOLDER = unicodedata.normalize("NFD", "かぞく/がっこう")
LOFT_ID = "loft00000179"
LOFT_TEXT = chunk("LOFT-PASSAGE", 30)


def _seed(db, tmp_path, *, file_path: str, mime: str = PDF, file_id: str = PDF_ID):
    """A file with one passage; an absolute ``file_path`` is written to disk."""
    if file_path.startswith("/"):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).write_bytes(b"DOCUMENT-BYTES")
    db.file(file_id, mime=mime, file_path=file_path, file_size=14)
    db.document(PDF_TEXT, chunk_index=0)


async def _prompt(call, file_id: str = PDF_ID, **body):
    response = await call(
        "POST", f"/files/{file_id}/ask", json={"question": QUESTION, **body},
    )
    assert response.status_code == 200, response.text
    return response


def _only_user_text(wire) -> str:
    bodies = question_bodies(wire)
    assert len(bodies) == 1
    return user_text(bodies[0])


# ---------------------------------------------------------------------------
# Folder
# ---------------------------------------------------------------------------

#: (stored path under the test root or relative, mount under the test root, Folder)
#: ``{root}`` is the test's ``tmp_path``.
FOLDER_ROWS = [
    pytest.param("trips/2024/report.pdf", None, "trips/2024", id="relative-no-mount"),
    pytest.param("trips/2024/report.pdf", "{root}/drive", "trips/2024", id="relative-with-mount"),
    pytest.param("report.pdf", None, TOP_OF_DRIVE, id="relative-top"),
    pytest.param("report.pdf", "{root}/drive", TOP_OF_DRIVE, id="relative-top-with-mount"),
    pytest.param(
        f"{NFD_FOLDER}/report.pdf", None, NFD_FOLDER, id="relative-nfd-kept",
    ),
    pytest.param(
        "{root}/drive/trips/2024/report.pdf", "{root}/drive", "trips/2024",
        id="absolute-inside",
    ),
    pytest.param(
        "{root}/drive/trips/2024/report.pdf", "{root}/drive/", "trips/2024",
        id="absolute-inside-trailing-slash",
    ),
    pytest.param("{root}/drive/report.pdf", "{root}/drive", TOP_OF_DRIVE, id="absolute-top"),
    pytest.param(
        "{root}/drive/report.pdf", "{root}/drive/", TOP_OF_DRIVE,
        id="absolute-top-trailing-slash",
    ),
    pytest.param(
        f"{{root}}/drive/{NFD_FOLDER}/report.pdf", "{root}/drive", NFD_FOLDER,
        id="absolute-nfd-kept",
    ),
    pytest.param("{root}/drive/trips/report.pdf", None, None, id="absolute-no-mount"),
    pytest.param(
        "{root}/videos/trips/report.pdf", "{root}/video", None,
        id="absolute-sibling-prefix",
    ),
    pytest.param(
        "{root}/elsewhere/trips/report.pdf", "{root}/drive", None,
        id="absolute-outside",
    ),
]


@pytest.mark.parametrize(("stored", "mount", "folder"), FOLDER_ROWS)
async def test_spec_addon_179_i2_the_folder_line_carries_the_folder_inside_the_drive(
    stored, mount, folder, db, addon, llm, call, tmp_path,
):
    """SPEC-ADDON-179 I2 I3: Folder of Terms, or no line; never the mount path."""
    root = str(tmp_path)
    if mount is not None:
        addon(drive_mounts={DRIVE: mount.format(root=root)})
    _seed(db, tmp_path, file_path=stored.format(root=root))
    _, wire = vision_client(llm)

    await _prompt(call)

    text = _only_user_text(wire)
    expected = block(file=FILENAME, type_=PDF, folder=folder)
    assert text.startswith(expected + "Question: "), text[:300]
    if folder == NFD_FOLDER:
        assert unicodedata.normalize("NFC", NFD_FOLDER) not in text
    assert root not in json.dumps(question_bodies(wire), ensure_ascii=False)


@pytest.mark.parametrize(
    ("stored", "mount", "folder"),
    [row for row in FOLDER_ROWS if row.values[2] is not None],
)
def test_spec_addon_179_material_carries_the_filename_and_the_folder(
    stored, mount, folder, db, addon, tmp_path,
):
    """SPEC-ADDON-179 item 1.2: ``Material.filename`` and ``Material.folder`` as Terms states."""
    root = str(tmp_path)
    if mount is not None:
        addon(drive_mounts={DRIVE: mount.format(root=root)})
    _seed(db, tmp_path, file_path=stored.format(root=root))

    material = load_material(PDF_ID, DRIVE, "question", budget=1000)

    assert material is not None
    assert material.filename == FILENAME
    assert material.folder == folder
    assert material.type_label == PDF


@pytest.mark.parametrize("name", SCENARIOS)
async def test_spec_addon_179_i3_no_path_sends_the_mount(
    name, db, addon, llm, call, tmp_path, monkeypatch,
):
    """SPEC-ADDON-179 I2 I3 on every path: the mount resolves the Folder and is never sent."""
    mount = str(tmp_path)
    wire = await RUN[name](
        db, addon, llm, call, tmp_path, monkeypatch, drive_mounts={DRIVE: mount},
    )
    folder = {
        "passages-only": "docs", "passages-and-image": "pics", "image-only": "pics",
        "text-only-retry": "pics", "document-tool": "docs",
    }.get(name, TOP_OF_DRIVE)
    file_line, type_line = BLOCKS[name].split("\n")[:2]
    expected = f"{file_line}\n{type_line}\nFolder: {folder}\n\n"

    bodies = question_bodies(wire)
    assert bodies
    for body in bodies:
        assert user_text(body).startswith(expected), user_text(body)[:300]
        assert mount not in json.dumps(body, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Empty values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "mime", "expected"),
    [
        pytest.param("", PDF, block(file=None, type_=PDF), id="no-filename"),
        pytest.param(FILENAME, "", block(file=FILENAME, type_=None), id="no-mime"),
    ],
)
async def test_spec_addon_179_i1_a_line_whose_value_is_empty_is_left_out(
    filename, mime, expected, db, addon, llm, call, tmp_path,
):
    """SPEC-ADDON-179 I1, failure cases: an empty ``filename`` or ``mime_type`` has no line."""
    _seed(db, tmp_path, file_path=str(tmp_path / "docs" / "report.pdf"))
    db.db.index.sql(
        "UPDATE indexed_files SET filename = :n, mime_type = :m WHERE file_id = :f",
        n=filename, m=mime, f=PDF_ID,
    )
    _, wire = vision_client(llm)

    await _prompt(call)

    assert _only_user_text(wire).startswith(expected + "Question: ")


async def test_spec_addon_179_i1_with_every_line_empty_the_prompt_starts_as_today(
    db, addon, llm, call, tmp_path,
):
    """SPEC-ADDON-179 I1: no line, and no empty line either; the rules are still added."""
    _seed(db, tmp_path, file_path=str(tmp_path / "docs" / "report.pdf"))
    db.db.index.sql(
        "UPDATE indexed_files SET filename = '', mime_type = '' WHERE file_id = :f",
        f=PDF_ID,
    )
    _, wire = vision_client(llm)

    await _prompt(call)

    bodies = question_bodies(wire)
    assert len(bodies) == 1
    assert_s179_and_s181(system_of(bodies[0]))
    assert without_additions(bodies[0], "") == today("passages-only")[0]


# ---------------------------------------------------------------------------
# Type of a .loft
# ---------------------------------------------------------------------------

LOFT_ROWS = [
    pytest.param(
        {"provider": "youtube", "url": "https://www.youtube.com/watch?v=abcdefghijk"},
        "YouTube video, kept as a Litloft link file (.loft)", id="youtube",
    ),
    pytest.param(
        {"provider": "vimeo", "url": "https://vimeo.com/1"},
        "web link (vimeo), kept as a Litloft link file (.loft)", id="vimeo",
    ),
    pytest.param(
        {"provider": "generic", "url": "https://example.com/v"},
        "web link (generic), kept as a Litloft link file (.loft)", id="generic",
    ),
    pytest.param(
        {"provider": "youtube"}, "Litloft link file (.loft)", id="provider-without-url",
    ),
    pytest.param(
        {"url": "https://example.com/v"}, "Litloft link file (.loft)", id="url-without-provider",
    ),
    pytest.param("{not json", "Litloft link file (.loft)", id="unreadable-body"),
    pytest.param(None, "Litloft link file (.loft)", id="missing-file"),
]


def _seed_loft(db, tmp_path, body) -> str:
    path = tmp_path / "links" / f"{LOFT_ID}.loft"
    if body is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body if isinstance(body, str) else json.dumps(body))
    db.file(LOFT_ID, mime=LOFT_MIME, file_path=str(path), file_size=1)
    db.document(LOFT_TEXT, chunk_index=0)
    return str(path)


@pytest.mark.parametrize(("body", "type_label"), LOFT_ROWS)
def test_spec_addon_179_a_lofts_type_names_what_it_links_to(
    body, type_label, db, addon, tmp_path, caplog,
):
    """SPEC-ADDON-179 Terms, Type; failure cases: an unreadable ``.loft`` logs nothing."""
    path = _seed_loft(db, tmp_path, body)

    with caplog.at_level(logging.DEBUG):
        material = load_material(LOFT_ID, DRIVE, "question", budget=1000)

    assert material is not None
    assert material.type_label == type_label
    assert material.filename == f"{LOFT_ID}.bin"
    assert [r.getMessage() for r in caplog.records if LOFT_ID in r.getMessage()
            or path in r.getMessage()] == []


@pytest.mark.parametrize(("body", "type_label"), LOFT_ROWS)
async def test_spec_addon_179_i1_a_lofts_type_line_reaches_the_prompt(
    body, type_label, db, addon, llm, call, tmp_path,
):
    """SPEC-ADDON-179 I1: the Type line of a ``.loft`` question (a profile that cannot see)."""
    _seed_loft(db, tmp_path, body)
    _, wire = vision_client(llm, config=openai_config(vision_model=""))

    await _prompt(call, LOFT_ID)

    expected = block(file=f"{LOFT_ID}.bin", type_=type_label)
    for body_sent in question_bodies(wire):
        assert user_text(body_sent).startswith(expected + "Question: ")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


async def test_spec_addon_179_i1_the_block_comes_before_the_history(
    db, addon, llm, call, tmp_path,
):
    """SPEC-ADDON-179 I1: the block starts the prompt, ahead of the earlier turns and the question."""
    _seed(db, tmp_path, file_path=str(tmp_path / "docs" / "report.pdf"))
    _, wire = vision_client(llm)

    await _prompt(call, history=[{
        "question": "EARLIER-QUESTION-MARKER", "answer": "EARLIER-ANSWER-MARKER",
    }])

    text = _only_user_text(wire)
    expected = block(file=FILENAME, type_=PDF)
    assert text.startswith(expected), text[:300]
    rest = text[len(expected):]
    assert "Earlier in this conversation" in rest
    assert rest.index("EARLIER-QUESTION-MARKER") < rest.index("Question: " + QUESTION)
