"""A YouTube ``.loft`` fetch over a real search DB, for SPEC-ADDON-117/118.

``yt-dlp`` and ``ffmpeg`` are the doubles of ``tests.youtube_frame_doubles``; the
rendition is a real local video served over HTTP, so the download moves those
bytes and the scan and OCR read real frames. The CLIP model and the OCR engine
are stand-ins.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sqlalchemy import text

import app.database as database
from app.config import FeaturesConfig, FrameExtractionConfig, IndexingConfig
from app.models import IndexedFile
from tests.ocr_fixtures import WholeImageBoxShim
from tests.search_endpoint_db import patch_everywhere
from tests.youtube_frame_doubles import (
    DRIVE,
    LOFT_MIME,
    SIGNED_TOKEN,
    YOUTUBE_URL,
    describe,
    video_format,
)

GIB = 1024 ** 3
FLAGS = ("frame_scan_indexed", "frame_ocr_read", "frame_ocr_indexed")


async def allow(*_args, **_kwargs) -> bool:
    return True


async def index_off(*args, **kwargs) -> bool:
    return "index" not in list(args) + list(kwargs.values())


class GreyEngine:
    """Answers DARK or LIGHT from the brightness of the frame's left edge."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        pixel = image.convert("L").getpixel((4, image.height // 2))
        return SimpleNamespace(txts=("DARK",) if pixel < 128 else ("LIGHT",))


def fetch(file_id: str, **kwargs):
    """One fetch of ``file_id``: the ``loft_media`` task's work for one ``.loft``.

    The spec names the task and its module, not the function; this is the one
    place a test reaches it.
    """
    from app.workers import loft_media

    result = loft_media.index_loft_media(file_id, **kwargs)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    return result


class LoftWorld:
    def __init__(self, index, youtube, monkeypatch, tmp_path, make_settings):
        self.index = index
        self.youtube = youtube
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        self.make_settings = make_settings
        self.drive_dir = tmp_path / "drives" / DRIVE
        self.drive_dir.mkdir(parents=True)
        self.embedded = 0
        self.engine = GreyEngine()
        self.configure()

    # -- configuration ---------------------------------------------------

    def configure(self, *, frame_ocr: str = "false", policy=allow) -> None:
        try:
            import app.workers.loft_media  # noqa: F401 - so the patches below reach it
        except ImportError:
            pass
        import app.workers.embedder  # noqa: F401
        import app.workers.frame_ocr  # noqa: F401
        import app.workers.frame_scan  # noqa: F401

        self.settings = self.make_settings(
            allowed_base_dirs=(str(self.tmp_path / "drives") + os.sep,),
            features=FeaturesConfig(video_frame_ocr=frame_ocr),  # type: ignore[call-arg]
            indexing=IndexingConfig(
                frame_extraction=FrameExtractionConfig(max_frames=500),
            ),
        )
        patch_everywhere(self.monkeypatch, "settings", self.settings)
        patch_everywhere(self.monkeypatch, "resolve_file_path", lambda _drive, path: path)
        self.monkeypatch.setattr("app.policy_client.is_feature_enabled", policy)
        patch_everywhere(self.monkeypatch, "is_feature_enabled", policy)
        patch_everywhere(self.monkeypatch, "load_image_model", lambda *a, **k: None)
        patch_everywhere(self.monkeypatch, "embed_image", self._embed)
        patch_everywhere(self.monkeypatch, "embed_passages", self._embed_passages)
        patch_everywhere(self.monkeypatch, "_get_engine", lambda: self.engine)
        self.box_shim = WholeImageBoxShim()
        for name in ("detect_boxes", "recognise_boxes"):
            patch_everywhere(
                self.monkeypatch, name, getattr(self.box_shim, name),
            )

    def _embed(self, *_args, **_kwargs):
        self.embedded += 1
        dim = self.index.dim("vec_clip")
        return np.full(dim, 1.0 / np.sqrt(dim), dtype=np.float32)

    def _embed_passages(self, texts):
        dim = self.index.dim("vec_text")
        return [np.full(dim, 1.0 / np.sqrt(dim), dtype=np.float32) for _ in texts]

    @property
    def fetch_dir(self) -> Path:
        return Path(self.settings.intelligence_data_dir) / "loft_media"

    def serve(self, source: Path, *, duration: float | None = None,
              filesize: int | None = None, name: str = "rendition") -> dict:
        """Describe one usable 720p rendition whose stream is ``source``."""
        path = f"/v/{name}"
        self.youtube.server.files[path] = source.read_bytes()
        fmt = video_format(name, 720, url=f"{self.youtube.base}{path}?sig={SIGNED_TOKEN}")
        if filesize is not None:
            fmt["filesize"] = filesize
        self.youtube.set_ytdlp(describe=describe([fmt], duration=duration))
        return fmt

    # -- rows ------------------------------------------------------------

    def add_loft(self, file_id: str, body: str | dict | None = None, *,
                 duration: float | None = None, **flags) -> Path:
        path = self.drive_dir / f"{file_id}.loft"
        if body is None:
            body = {"provider": "youtube", "url": YOUTUBE_URL}
        path.write_text(body if isinstance(body, str) else json.dumps(body))
        self._row(file_id, path, LOFT_MIME, duration, flags)
        return path

    def add_native(self, file_id: str, source: Path, **flags) -> Path:
        path = self.drive_dir / f"{file_id}{source.suffix}"
        shutil.copyfile(source, path)
        self._row(file_id, path, "video/mp4", None, flags)
        return path

    def _row(self, file_id, path, mime, duration, flags) -> None:
        values = dict(frame_scan_indexed=False, frame_ocr_read=True, frame_ocr_indexed=True)
        values.update(flags)
        with database.get_search_db() as s:
            s.add(IndexedFile(
                file_id=file_id, drive=DRIVE, filename=path.name, file_path=str(path),
                file_type="video", mime_type=mime, file_size=1, active=True,
                duration=duration, **values,
            ))

    def set(self, file_id: str, **columns) -> None:
        for column, value in columns.items():
            self.index.sql(
                f"UPDATE indexed_files SET {column} = :v WHERE file_id = :f",
                v=value, f=file_id,
            )

    # -- reading back ----------------------------------------------------

    def value(self, file_id: str, column: str):
        with self.index.engine.connect() as conn:
            return conn.execute(
                text(f"SELECT {column} FROM indexed_files WHERE file_id = :f"),
                {"f": file_id},
            ).scalar()

    def flags(self, file_id: str) -> tuple:
        return tuple(bool(self.value(file_id, f)) for f in FLAGS)

    def source(self, file_id: str):
        return self.value(file_id, "loft_source")

    def failures(self, file_id: str):
        return self.value(file_id, "loft_fetch_failures")

    def change_seconds(self, file_id: str) -> list[float]:
        with self.index.engine.connect() as conn:
            return [r[0] for r in conn.execute(text(
                "SELECT timestamp_start FROM embeddings WHERE file_id = :f "
                "AND embedding_type = 'clip_change' ORDER BY timestamp_start"
            ), {"f": file_id})]

    def scan_rows(self, file_id: str) -> int:
        with self.index.engine.connect() as conn:
            return conn.execute(text(
                "SELECT COUNT(*) FROM video_frame_scans WHERE file_id = :f"
            ), {"f": file_id}).scalar()

    def frame_texts(self, file_id: str) -> list[tuple]:
        with self.index.engine.connect() as conn:
            return [tuple(r) for r in conn.execute(text(
                "SELECT chunk_index, start_time, end_time, text FROM video_frame_texts "
                "WHERE file_id = :f ORDER BY chunk_index"
            ), {"f": file_id})]

    def fetch_dir_files(self) -> list[str]:
        if not self.fetch_dir.exists():
            return []
        return sorted(p.name for p in self.fetch_dir.iterdir())

    def lookups(self) -> int:
        return len(self.youtube.lookups())

    def downloads(self) -> list[dict]:
        return self.youtube.fetched("/v/")


def disk_usage_with(free: int):
    usage = shutil.disk_usage("/")

    def _usage(_path):
        return type(usage)(usage.total, usage.total - free, free)

    return _usage


class FreeSpace:
    """The free space the disk-usage seam reports, changeable while a download runs."""

    def __init__(self, free: int) -> None:
        self.free = free
        self._usage = shutil.disk_usage("/")

    def __call__(self, _path):
        usage = self._usage
        return type(usage)(usage.total, usage.total - self.free, self.free)
