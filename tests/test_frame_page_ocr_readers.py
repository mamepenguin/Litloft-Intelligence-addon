"""End-to-end tests for the frame and page readers (SPEC-ADDON-066/069).

Real ffmpeg-generated clips and PDFs; the OCR engine is a fake that answers
from the frame's colour, so the tests hold selection, geometry, segments,
storage, failure handling and pause without depending on model output.
"""

from __future__ import annotations

import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.models import VideoFrameText  # noqa: F401 - registers the table
from tests.ocr_fixtures import (
    build_search_db,
    pillow_real,  # noqa: F401  - the usefixtures marker resolves it
    real_pillow,
    seed_indexed_file,
)

pytestmark = pytest.mark.usefixtures("pillow_real")


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------


def _encode(path: Path, width: int, height: int, *, seconds: int = 13) -> Path:
    real_pillow()
    from PIL import Image, ImageDraw

    fps = 5
    process = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-g", "10", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for index in range(seconds * fps):
        t = index / fps
        color = (220, 20, 20) if t < 6 else (20, 20, 220)
        frame = Image.new("RGB", (width, height), color)
        draw = ImageDraw.Draw(frame)
        left = int(width * (0.1 if t < 6 else 0.5))
        draw.rectangle(
            (left, int(height * 0.2), left + int(width * 0.4), int(height * 0.8)),
            fill=(255, 255, 255),
        )
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    assert process.wait() == 0
    return path


def _rotate(source: Path, target: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-display_rotation", "90",
            "-i", str(source), "-c", "copy", str(target),
        ],
        check=True,
    )
    return target


def _copy_trim(source: Path, target: Path, start: float) -> Path:
    """A stream-copied cut at a non-key position (an edit list and a leading KD)."""
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-ss", str(start),
            "-i", str(source), "-c", "copy", str(target),
        ],
        check=True,
    )
    return target


def _scanned_pdf(path: Path, pages: int = 2) -> Path:
    import fitz

    document = fitz.open()
    for _ in range(pages):
        page = document.new_page(width=300, height=400)
        page.draw_rect(fitz.Rect(30, 30, 270, 370), width=4)
    document.save(str(path))
    document.close()
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    import sys

    # ``_encode`` needs real Pillow; other modules expect conftest's stub.
    saved = {
        name: module for name, module in sys.modules.items()
        if name == "PIL" or name.startswith("PIL.")
    }
    root = tmp_path_factory.mktemp("frame-ocr-media")
    try:
        landscape = _encode(root / "landscape.mp4", 320, 240)
        portrait_landscape = _encode(root / "portrait-base.mp4", 320, 240)
        portrait = _rotate(portrait_landscape, root / "portrait.mp4")
        large_base = _encode(root / "large-base.mp4", 1600, 1200)
        large = _rotate(large_base, root / "large.mp4")
        large_trim = _copy_trim(
            large_base, root / "large-trim.mp4", start=0.5,
        )
        corrupt = root / "corrupt.mp4"
        corrupt.write_bytes(b"this is not a video")
        scanned = _scanned_pdf(root / "scanned.pdf")
        from tests.document_fixtures import make_pdf

        text_pdf = make_pdf(root / "text.pdf", ["TEXT ONE", "TEXT TWO"])
    finally:
        for name in [n for n in sys.modules if n == "PIL" or n.startswith("PIL.")]:
            del sys.modules[name]
        sys.modules.update(saved)
    return {
        "landscape": landscape,
        "portrait": portrait,
        "large": large,
        "large_trim": large_trim,
        "corrupt": corrupt,
        "scanned": scanned,
        "text_pdf": text_pdf,
    }


class ColourEngine:
    """Answers ALPHA/BETA from the frame's dominant colour."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []
        self.error: BaseException | None = None
        self.delay = 0.0

    def __call__(self, image):
        self.calls.append((image.width, image.height))
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        red, _green, blue = image.getpixel((10, 10))
        if red > 150 and blue < 100:
            return SimpleNamespace(txts=("ALPHA",))
        if blue > 150 and red < 100:
            return SimpleNamespace(txts=("BETA",))
        return SimpleNamespace(txts=())


class CountingEngine:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls = 0

    def __call__(self, _image):
        answer = self.answers[self.calls] if self.calls < len(self.answers) else ""
        self.calls += 1
        return SimpleNamespace(txts=(answer,) if answer else ())


@pytest.fixture()
def search_db(tmp_path, monkeypatch):
    return build_search_db(tmp_path, monkeypatch)


@contextmanager
def _write(db):
    session = db.Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def _read(db):
    session = db.Session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def frame_reader(search_db, monkeypatch):
    import app.workers.frame_ocr as module
    import app.workers.ocr as ocr

    monkeypatch.setattr(module, "get_search_db", lambda: _write(search_db))
    monkeypatch.setattr(module, "get_search_db_read", lambda: _read(search_db))
    monkeypatch.setattr(module, "embed_passages", lambda texts: [
        __import__("numpy").zeros(4, dtype="float32") for _ in texts
    ])
    engine = ColourEngine()
    monkeypatch.setattr(module, "_get_engine", lambda: engine)
    return module, engine


@pytest.fixture()
def page_reader(search_db, monkeypatch):
    import app.workers.page_ocr as module

    monkeypatch.setattr(module, "get_search_db", lambda: _write(search_db))
    monkeypatch.setattr(module, "get_search_db_read", lambda: _read(search_db))
    engine = CountingEngine(["PAGE ONE", "PAGE TWO"])
    monkeypatch.setattr(module, "_get_engine", lambda: engine)
    return module, engine


def _video(db, tmp_path, path: Path, file_id: str = "vid1", **flags) -> None:
    seed_indexed_file(
        db, file_id, "video/mp4", path, file_type="video", **flags,
    )


def _frame_rows(db, file_id: str = "vid1"):
    return db.rows(
        "SELECT chunk_index, start_time, end_time, text FROM video_frame_texts "
        "WHERE file_id = :f ORDER BY chunk_index",
        f=file_id,
    )


# ---------------------------------------------------------------------------
# I1/I2 - selection, upright geometry and storage
# ---------------------------------------------------------------------------


def test_spec_addon_066_i1_i2_landscape_reads_the_first_and_the_ten_second_change(
    frame_reader, search_db, tmp_path, media,
):
    module, engine = frame_reader
    _video(search_db, tmp_path, media["landscape"])

    assert module.index_frame_ocr("vid1") is True

    assert engine.calls == [(320, 240), (320, 240)]
    assert _frame_rows(search_db) == [
        (0, 0.0, 10.0, "ALPHA"),
        (1, 10.0, 13.0, "BETA"),
    ]
    embeddings = search_db.rows(
        "SELECT chunk_index, page, timestamp_start, timestamp_end, id "
        "FROM embeddings WHERE file_id = 'vid1' AND embedding_type = 'text_content' "
        "ORDER BY chunk_index"
    )
    assert [(r[0], r[1], r[2], r[3]) for r in embeddings] == [
        (0, None, 0.0, 10.0), (1, None, 10.0, 13.0),
    ]
    assert all(r[4].startswith("txt_vid1_") for r in embeddings)
    assert search_db.flag("vid1", "frame_ocr_read") == 1
    assert search_db.flag("vid1", "frame_ocr_indexed") == 1


def test_spec_addon_066_i1_a_portrait_video_is_read_upright_at_source_size(
    frame_reader, search_db, tmp_path, media,
):
    module, engine = frame_reader
    assert module._probe(str(media["portrait"])) == (240, 320)
    _video(search_db, tmp_path, media["portrait"])

    assert module.index_frame_ocr("vid1") is True

    assert engine.calls == [(240, 320), (240, 320)]
    assert _frame_rows(search_db) == [
        (0, 0.0, 10.0, "ALPHA"),
        (1, 10.0, 13.0, "BETA"),
    ]


def test_spec_addon_066_i1_above_1080_uses_the_short_side_and_the_key_frame_path(
    frame_reader, search_db, tmp_path, media,
):
    module, engine = frame_reader
    assert module._probe(str(media["large"])) == (1200, 1600)
    _video(search_db, tmp_path, media["large"])

    assert module.index_frame_ocr("vid1") is True

    assert engine.calls == [(1080, 1440), (1080, 1440)]
    assert [row[:3] for row in _frame_rows(search_db)] == [
        (0, 0.0, 10.0), (1, 10.0, 13.0),
    ]


def test_spec_addon_066_a1_a_copy_trimmed_video_uses_the_decoded_key_frames(
    frame_reader, search_db, tmp_path, media,
):
    """A stream-copied cut keeps a leading KD packet at a negative time; the
    decoder drops it, so the reader must list the decoder's key frames."""
    module, _engine = frame_reader
    assert module._probe(str(media["large_trim"])) == (1600, 1200)
    _video(search_db, tmp_path, media["large_trim"])

    assert module.index_frame_ocr("vid1") is True

    rows = _frame_rows(search_db)
    starts = [row[1] for row in rows]
    assert starts == [1.4, 11.4]
    assert all(start >= 0 for start in starts)
    assert rows == [
        (0, 1.4, 11.4, "ALPHA"),
        (1, 11.4, 12.4, "BETA"),
    ]


# ---------------------------------------------------------------------------
# Failure cases and I4
# ---------------------------------------------------------------------------


def test_spec_addon_066_failure_a_corrupt_video_is_closed_as_read_with_no_rows(
    frame_reader, search_db, tmp_path, media,
):
    module, engine = frame_reader
    _video(search_db, tmp_path, media["corrupt"])

    assert module.index_frame_ocr("vid1") is True

    assert engine.calls == []
    assert _frame_rows(search_db) == []
    assert search_db.flag("vid1", "frame_ocr_read") == 1
    assert search_db.flag("vid1", "frame_ocr_indexed") == 1


def test_spec_addon_066_i4_an_unreachable_video_writes_nothing(
    frame_reader, search_db, tmp_path, media,
):
    module, engine = frame_reader
    _video(search_db, tmp_path, tmp_path / "gone.mp4")

    assert module.index_frame_ocr("vid1") is False

    assert engine.calls == []
    assert _frame_rows(search_db) == []
    assert search_db.flag("vid1", "frame_ocr_read") == 0
    assert search_db.flag("vid1", "frame_ocr_indexed") == 0


def test_spec_addon_066_i4_an_inactive_row_writes_nothing(
    frame_reader, search_db, tmp_path, media,
):
    module, _engine = frame_reader
    _video(search_db, tmp_path, media["landscape"], active=False)

    assert module.index_frame_ocr("vid1") is False

    assert _frame_rows(search_db) == []


def test_spec_addon_066_a4_an_unparsable_key_frame_time_writes_nothing(
    frame_reader, search_db, tmp_path, media, monkeypatch,
):
    module, _engine = frame_reader
    _video(search_db, tmp_path, media["large_trim"])
    real_run = module.subprocess.run

    def fake_run(command, *args, **kwargs):
        if any("pts_time" in str(part) for part in command):
            return SimpleNamespace(stdout="N/A\n", returncode=0)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.index_frame_ocr("vid1") is False
    assert _frame_rows(search_db) == []
    assert search_db.flag("vid1", "frame_ocr_read") == 0
    assert search_db.flag("vid1", "frame_ocr_indexed") == 0


def test_spec_addon_066_a4_a_key_frame_count_mismatch_writes_nothing(
    frame_reader, search_db, tmp_path, media, monkeypatch,
):
    module, _engine = frame_reader
    _video(search_db, tmp_path, media["large_trim"])
    real_run = module.subprocess.run

    def fake_run(command, *args, **kwargs):
        text = " ".join(str(part) for part in command)
        if "packet=pts_time" in text:
            return SimpleNamespace(
                stdout=(
                    "1.4,K__\n3.4,K__\n5.4,K__\n7.4,K__\n"
                    "9.4,K__\n11.4,K__\n13.4,K__\n"
                ),
                returncode=0,
            )
        if "frame=pts_time" in text:
            return SimpleNamespace(
                stdout=(
                    "1.4\n3.4\n5.4\n7.4\n9.4\n11.4\n13.4\n"
                ),
                returncode=0,
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.index_frame_ocr("vid1") is False
    assert _frame_rows(search_db) == []
    assert search_db.flag("vid1", "frame_ocr_read") == 0
    assert search_db.flag("vid1", "frame_ocr_indexed") == 0


# ---------------------------------------------------------------------------
# I3 - embedding failure and rebuild
# ---------------------------------------------------------------------------


def test_spec_addon_066_i3_an_embedding_failure_keeps_the_read_and_rebuilds_without_reading(
    frame_reader, search_db, tmp_path, media, monkeypatch,
):
    module, engine = frame_reader
    _video(
        search_db, tmp_path, media["landscape"],
        frame_ocr_read=True, frame_ocr_indexed=False,
    )
    search_db.execute(
        "INSERT INTO video_frame_texts"
        "(file_id, chunk_index, start_time, end_time, text) "
        "VALUES('vid1', 0, 0.0, 10.0, 'ALPHA'),"
        "('vid1', 1, 10.0, 13.0, 'BETA')"
    )

    def fail(_texts):
        raise RuntimeError("embed down")

    monkeypatch.setattr(module, "embed_passages", fail)
    assert module.index_frame_ocr("vid1") is False
    assert engine.calls == []
    assert search_db.flag("vid1", "frame_ocr_read") == 1
    assert search_db.flag("vid1", "frame_ocr_indexed") == 0
    assert len(_frame_rows(search_db)) == 2

    monkeypatch.setattr(module, "embed_passages", lambda texts: [
        __import__("numpy").zeros(4, dtype="float32") for _ in texts
    ])
    assert module.index_frame_ocr("vid1") is True
    assert engine.calls == []
    assert search_db.flag("vid1", "frame_ocr_indexed") == 1
    assert len(search_db.rows(
        "SELECT id FROM embeddings WHERE file_id = 'vid1' "
        "AND embedding_type = 'text_content'"
    )) == 2


# ---------------------------------------------------------------------------
# I8 - pause
# ---------------------------------------------------------------------------


def test_spec_addon_066_i8_the_frame_reader_makes_no_engine_call_while_paused(
    frame_reader, media,
):
    module, engine = frame_reader
    state = {"paused": True}
    result = {}

    def work():
        result["value"] = module._read_video(
            str(media["landscape"]), paused=lambda: state["paused"],
        )

    thread = threading.Thread(target=work)
    thread.start()
    time.sleep(0.6)
    assert engine.calls == []
    state["paused"] = False
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert engine.calls


# ---------------------------------------------------------------------------
# I6 - page OCR storage and the B5 flag rule
# ---------------------------------------------------------------------------


def _pdf(db, path: Path, file_id: str = "pdf1", **flags) -> None:
    seed_indexed_file(
        db, file_id, "application/pdf", path, file_type="document", **flags,
    )


def _page_rows(db, file_id: str = "pdf1"):
    return db.rows(
        "SELECT page, text FROM pdf_page_ocr_texts "
        "WHERE file_id = :f ORDER BY page",
        f=file_id,
    )


def test_spec_addon_069_i6_scanned_pages_are_stored_and_reset_text_indexed(
    page_reader, search_db, tmp_path, media,
):
    module, engine = page_reader
    _pdf(search_db, media["scanned"], text_indexed=True)

    assert module.index_page_ocr("pdf1") is True

    assert engine.calls == 2
    assert _page_rows(search_db) == [(1, "PAGE ONE"), (2, "PAGE TWO")]
    assert search_db.flag("pdf1", "page_ocr_indexed") == 1
    assert search_db.flag("pdf1", "text_indexed") == 0


def test_spec_addon_069_i6_a_text_layer_pdf_stores_no_rows_and_leaves_text_indexed(
    page_reader, search_db, tmp_path, media,
):
    module, engine = page_reader
    _pdf(search_db, media["text_pdf"], text_indexed=True)

    assert module.index_page_ocr("pdf1") is False

    assert engine.calls == 0
    assert _page_rows(search_db) == []
    assert search_db.flag("pdf1", "page_ocr_indexed") == 1
    assert search_db.flag("pdf1", "text_indexed") == 1


def test_spec_addon_069_i4_an_unreachable_page_render_writes_nothing(
    page_reader, search_db, tmp_path, media, monkeypatch,
):
    module, _engine = page_reader
    _pdf(search_db, media["scanned"])
    state = {"reachable": True}

    def render(_page):
        state["reachable"] = False
        raise RuntimeError("drive gone")

    monkeypatch.setattr(module, "_render", render)
    monkeypatch.setattr(module, "_reachable", lambda _path: state["reachable"])

    assert module.index_page_ocr("pdf1") is False
    assert _page_rows(search_db) == []
    assert search_db.flag("pdf1", "page_ocr_indexed") == 0


def test_spec_addon_069_i4_a_missing_engine_writes_nothing(
    page_reader, search_db, tmp_path, media, monkeypatch,
):
    module, _engine = page_reader
    _pdf(search_db, media["scanned"])
    monkeypatch.setattr(module, "_get_engine", lambda: None)

    assert module.index_page_ocr("pdf1") is False
    assert _page_rows(search_db) == []
    assert search_db.flag("pdf1", "page_ocr_indexed") == 0
