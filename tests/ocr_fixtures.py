"""Fixture pictures, a fake OCR engine and a search.db double for SPEC-ADDON-050.

The engine is not installed in the test image, so tests replace
``app.workers.ocr._create_engine`` with :class:`EngineFactory`. The fake stands in
for RapidOCR 3.x's call shape: the engine takes a PIL image and answers an object
whose ``txts`` are the recognised lines in reading order.

``conftest`` stubs ``PIL`` for the modules that import it eagerly. The builders
here need the real package, and ``app.workers.ocr`` is reloaded after
``real_pillow()`` so the worker binds it too.
"""

from __future__ import annotations

import importlib
import os
import sqlite3
import struct
import sys
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import IndexedFile


# ---------------------------------------------------------------------------
# Real Pillow
# ---------------------------------------------------------------------------


def real_pillow() -> None:
    for name in [n for n in sys.modules if n == "PIL" or n.startswith("PIL.")]:
        del sys.modules[name]
    import PIL.Image  # noqa: F401
    import PIL.ImageOps  # noqa: F401


@pytest.fixture()
def pillow_real():
    """Real Pillow for one test, with the conftest stub put back afterwards.

    Modules other than the OCR worker import PIL lazily and are written
    against the stub, so a real package left in ``sys.modules`` changes
    their behaviour for every test after this one.
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "PIL" or name.startswith("PIL.")
    }
    real_pillow()
    yield
    for name in [n for n in sys.modules if n == "PIL" or n.startswith("PIL.")]:
        del sys.modules[name]
    sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Fixture pictures
# ---------------------------------------------------------------------------


def picture(path: Path, size: tuple[int, int] = (40, 20)) -> Path:
    """A plain RGB PNG with two darker blocks, so a decode is worth something."""
    real_pillow()
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle([2, 2, size[0] // 2 - 2, size[1] - 3], fill=(20, 20, 20))
    draw.rectangle([size[0] // 2 + 1, 4, size[0] - 3, size[1] // 2], fill=(90, 90, 90))
    image.save(path)
    return path


def rgba_picture(path: Path) -> Path:
    """An RGBA PNG: left half fully transparent, right half opaque red."""
    real_pillow()
    from PIL import Image

    image = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
    for x in range(10, 20):
        for y in range(10):
            image.putpixel((x, y), (200, 30, 30, 255))
    image.save(path)
    return path


def animated_gif(path: Path) -> Path:
    """A two-frame GIF whose first frame is red and second is blue."""
    real_pillow()
    from PIL import Image

    first = Image.new("RGB", (30, 10), (220, 0, 0))
    second = Image.new("RGB", (30, 10), (0, 0, 220))
    first.save(path, save_all=True, append_images=[second], duration=100, loop=0)
    return path


def jpeg_with_exif_orientation_6(path: Path) -> Path:
    """A 40x20 JPEG whose EXIF orientation rotates it a quarter turn."""
    real_pillow()
    from PIL import Image

    image = Image.new("RGB", (40, 20), (250, 250, 250))
    for x in range(0, 20):
        for y in range(20):
            image.putpixel((x, y), (10, 10, 10))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, "JPEG", exif=exif)
    return path


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def huge_pixel_png(path: Path, width: int = 10000, height: int = 6000) -> Path:
    """A PNG header declaring 60 million pixels, with no pixel data behind it."""
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IEND", b"")
    )
    return path


def oversize_png(path: Path, size_bytes: int = 60 * 1024 * 1024) -> Path:
    """A valid small PNG extended past the byte cap without writing the zeros."""
    picture(path, size=(20, 10))
    with path.open("r+b") as handle:
        handle.truncate(size_bytes)
    return path


def heic_picture(path: Path) -> Path:
    """A HEIC picture, skipped when the test image cannot encode one."""
    real_pillow()
    try:
        import pillow_heif
    except ImportError:  # pragma: no cover - depends on the image
        pytest.skip("pillow-heif is not installed in the test image")
    pillow_heif.register_heif_opener()
    from PIL import Image

    image = Image.new("RGB", (32, 16), (245, 245, 245))
    try:
        image.save(path, format="HEIF")
    except Exception as exc:  # pragma: no cover - depends on the encoder
        pytest.skip(f"pillow-heif cannot encode HEIC here: {type(exc).__name__}")
    return path


# ---------------------------------------------------------------------------
# Fake engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PictureSeen:
    size: tuple[int, int]
    mode: str
    pixel: tuple[int, ...] | None


class FakeOcrResult:
    """RapidOCR 3.x's answer shape: the kept lines, in the engine's order."""

    def __init__(self, lines: tuple[str, ...]) -> None:
        self.txts = lines
        self.scores = tuple(0.99 for _ in lines)
        self.boxes = ()


class FakeEngine:
    def __init__(self, lines: tuple[str, ...] = ()) -> None:
        self.lines = lines
        self.error: BaseException | None = None
        self.on_call: Any = None
        self.seen: list[PictureSeen] = []
        self.concurrent = 0
        self.max_concurrent = 0

    def __call__(self, image: Any, *args: Any, **kwargs: Any) -> FakeOcrResult:
        self.seen.append(
            PictureSeen(
                size=tuple(image.size),
                mode=image.mode,
                pixel=(image.getpixel((0, 0)) if image.mode in ("RGB", "RGBA") else None),
            )
        )
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.on_call is not None:
                self.on_call(image)
            if self.error is not None:
                raise self.error
        finally:
            self.concurrent -= 1
        return FakeOcrResult(self.lines)


class EngineFactory:
    """The one place the worker creates its engine, recorded call by call."""

    def __init__(
        self,
        engine: FakeEngine | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.engine = engine
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> FakeEngine:
        self.calls.append({"args": args, "kwargs": kwargs})
        if self.error is not None:
            raise self.error
        assert self.engine is not None
        return self.engine

    @property
    def threads(self) -> Any:
        call = self.calls[0]
        if "threads" in call["kwargs"]:
            return call["kwargs"]["threads"]
        return call["args"][0] if call["args"] else None


@dataclass
class OcrHarness:
    module: Any
    factory: EngineFactory
    engine: FakeEngine


def _install_embedding(monkeypatch, *, error: BaseException | None, custom: Any) -> None:
    def _embed(passages: list[str]) -> list[np.ndarray]:
        if error is not None:
            raise error
        if custom is not None:
            return custom(passages)
        return [np.zeros(4, dtype=np.float32) for _ in passages]

    for name in ("app.workers.embedder", "app.workers.metadata", "app.workers.ocr"):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "embed_passages"):
            monkeypatch.setattr(module, "embed_passages", _embed)


def install(
    monkeypatch,
    *,
    lines: tuple[str, ...] | list[str] = (),
    engine: FakeEngine | None = None,
    engine_error: BaseException | None = None,
    embed_error: BaseException | None = None,
    embed: Any = None,
) -> OcrHarness:
    """Reload the worker against real Pillow and stand in for its engine."""
    real_pillow()
    module = importlib.import_module("app.workers.ocr")
    module = importlib.reload(module)
    fake_engine = engine if engine is not None else FakeEngine(tuple(lines))
    factory = EngineFactory(fake_engine, error=engine_error)
    monkeypatch.setattr(module, "_create_engine", factory)
    _install_embedding(monkeypatch, error=embed_error, custom=embed)
    return OcrHarness(module=module, factory=factory, engine=fake_engine)


# ---------------------------------------------------------------------------
# search.db double
# ---------------------------------------------------------------------------

_SEARCH_DB_MODULES = (
    "app.database",
    "app.indexer",
    "app.workers.ocr",
    "app.workers.metadata",
    "app.routers.files",
    "app.main",
)


@dataclass
class SearchDB:
    engine: Any
    path: Path
    Session: Any = field(repr=False)

    def add(self, row: Any) -> None:
        with self.Session() as session:
            session.add(row)
            session.commit()

    def execute(self, sql: str, **params: Any) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(sql), params)

    def rows(self, sql: str, **params: Any) -> list[tuple]:
        with self.engine.connect() as conn:
            return list(conn.execute(text(sql), params).fetchall())

    def scalar(self, sql: str, **params: Any) -> Any:
        rows = self.rows(sql, **params)
        return rows[0][0] if rows else None

    def column_names(self, table: str) -> set[str]:
        with self.engine.connect() as conn:
            return {
                row[1]
                for row in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            }

    def flag(self, file_id: str, name: str) -> Any:
        return self.scalar(
            f"SELECT {name} FROM indexed_files WHERE file_id = :f", f=file_id
        )

    def set_flag(self, file_id: str, name: str, value: Any) -> None:
        self.execute(
            f"UPDATE indexed_files SET {name} = :v WHERE file_id = :f",
            v=1 if value else 0,
            f=file_id,
        )

    def ocr_embeddings(self, file_id: str) -> list[tuple[int, Any, str]]:
        return self.rows(
            "SELECT chunk_index, page, content_preview FROM embeddings "
            "WHERE file_id = :f AND embedding_type = 'text_content' "
            "ORDER BY chunk_index",
            f=file_id,
        )

    def ocr_fts(self, file_id: str) -> list[tuple[int, Any, str]]:
        return self.rows(
            "SELECT CAST(chunk_index AS INTEGER), page, text FROM fts_text_content "
            "WHERE file_id = :f ORDER BY CAST(chunk_index AS INTEGER)",
            f=file_id,
        )


def build_search_db(tmp_path: Path, monkeypatch) -> SearchDB:
    """A real search.db skeleton wired into every module the OCR path touches."""
    path = tmp_path / "search.db"
    engine = create_engine(
        f"sqlite:///{path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS vec_text ("
            "  embedding_id TEXT PRIMARY KEY, embedding_type TEXT, "
            "  vector BLOB NOT NULL)"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content "
            "USING fts5(file_id, chunk_index, page, text, tokenize='trigram')"
        ))
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_content_word "
            "USING fts5(file_id, chunk_index, page, text, "
            "tokenize=\"unicode61 remove_diacritics 2\")"
        ))
        # Written by a second connection from inside the engine call: a worker
        # holding the write lock there makes this INSERT fail.
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS ocr_lock_probe (id INTEGER PRIMARY KEY)"
        ))

    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _write():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def _read():
        session = Session()
        try:
            yield session
        finally:
            session.expunge_all()
            session.rollback()
            session.close()

    for name in _SEARCH_DB_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        for attr, factory in (("get_search_db", _write), ("get_search_db_read", _read)):
            if hasattr(module, attr):
                monkeypatch.setattr(module, attr, factory)

    monkeypatch.setattr("app.workers.metadata.get_search_db_read", _read, raising=False)
    return SearchDB(engine=engine, path=path, Session=Session)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def patch_features(monkeypatch, **overrides: Any):
    """Rebind every module's ``settings`` to one carrying these feature modes."""
    import dataclasses

    import app.config as config

    features = dataclasses.replace(config.settings.features, **overrides)
    new_settings = dataclasses.replace(config.settings, features=features)
    for name in (
        "app.config",
        "app.indexer",
        "app.routers.files",
        "app.routers.admin",
        "app.main",
        "app.workers.ocr",
    ):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "settings"):
            monkeypatch.setattr(module, "settings", new_settings)
    return new_settings


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def seed_indexed_file(
    db: SearchDB,
    file_id: str,
    mime_type: str,
    file_path: Path | str,
    *,
    file_type: str | None = None,
    active: bool = True,
    **flags: bool,
) -> None:
    db.add(IndexedFile(
        file_id=file_id,
        drive="drive1",
        filename=Path(str(file_path)).name,
        file_path=str(file_path),
        file_type=file_type or ("image" if mime_type.startswith("image/") else "document"),
        mime_type=mime_type,
        file_size=(os.path.getsize(file_path) if Path(str(file_path)).is_file() else 1),
        active=active,
        **flags,
    ))


def seed_ocr_rows(
    db: SearchDB, file_id: str, texts: tuple[str, ...] = ("earlier text",)
) -> None:
    """Earlier OCR rows for a file, the ones a fresh run must replace."""
    with db.engine.begin() as conn:
        for index, chunk in enumerate(texts):
            conn.execute(
                text(
                    "INSERT INTO embeddings(id, file_id, embedding_type, "
                    "vector_table, content_preview, page, chunk_index, created_at) "
                    "VALUES(:id, :f, 'text_content', 'vec_text', :preview, 3, :idx, "
                    "CURRENT_TIMESTAMP)"
                ),
                {
                    "id": f"chars-{file_id}-{index}",
                    "f": file_id,
                    "preview": chunk,
                    "idx": index,
                },
            )
            conn.execute(
                text(
                    "INSERT INTO fts_text_content(file_id, chunk_index, page, text) "
                    "VALUES(:f, :idx, '3', :text)"
                ),
                {"f": file_id, "idx": str(index), "text": chunk},
            )
            conn.execute(
                text(
                    "INSERT INTO fts_text_content_word"
                    "(file_id, chunk_index, page, text) "
                    "VALUES(:f, :idx, '3', :text)"
                ),
                {"f": file_id, "idx": str(index), "text": chunk},
            )


def probe_write_outside_the_lock(db: SearchDB) -> None:
    """A write from another connection, used from inside the engine call."""
    conn = sqlite3.connect(str(db.path), timeout=0.05)
    try:
        conn.execute("PRAGMA busy_timeout=50")
        conn.execute("INSERT INTO ocr_lock_probe(id) VALUES (1)")
        conn.commit()
    finally:
        conn.close()
