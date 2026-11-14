"""Stored rows the representative-frame selection reads, written straight into
a search DB built by ``init_search_db`` (sqlite-vec loaded).

Vectors are basis vectors or exact mixes of two, so every cosine the selection
computes is known in advance and ties stay ties.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from sqlalchemy import text

import app.database as database
from app.models import Embedding, IndexedFile

CHANGE_TABLE = "vec_clip_change"


def basis(dim: int, axis: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def mix(dim: int, cos: float, a: int = 0, b: int = 1) -> np.ndarray:
    """A unit vector at cosine ``cos`` from ``basis(dim, a)``."""
    v = np.zeros(dim, dtype=np.float32)
    v[a] = cos
    v[b] = float(np.sqrt(1.0 - cos * cos))
    return v


def distinct_chars(first: int, count: int) -> str:
    """``count`` different CJK ideographs, so every bigram in it is distinct."""
    return "".join(chr(0x4E00 + i) for i in range(first, first + count))


def hash_with_bits(bits: int) -> bytes:
    """A 256-bit difference hash with its first ``bits`` bits set, packed MSB first."""
    flags = np.zeros(256, dtype=np.uint8)
    flags[:bits] = 1
    return np.packbits(flags).tobytes()


class Video:
    """One file's stored inputs, buffered and written together."""

    def __init__(self, index, file_id: str, *, duration: float | None = 600.0,
                 drive: str = "family", mime: str = "video/mp4"):
        self.index = index
        self.file_id = file_id
        self.duration = duration
        self.drive = drive
        self.mime = mime
        self.clip_dim = index.dim("vec_clip")
        self.text_dim = index.dim("vec_text")
        self._rows: list[tuple[str, str, str, float, np.ndarray | None]] = []
        self._ocr: list[tuple[float, float, str]] = []
        self._chunks: list[tuple[float, float, str]] = []
        self._scan: tuple[int, int, bytes, bytes] | None = None
        self._n = 0

    def clip(self, t: float, vector: np.ndarray | None) -> "Video":
        self._n += 1
        self._rows.append(
            (f"clip_{self.file_id}_{self._n}", "clip", "vec_clip", t, vector)
        )
        return self

    def change(self, t: float, vector: np.ndarray | None) -> "Video":
        self._rows.append(
            (f"chg_{self.file_id}_{int(t)}", "clip_change", CHANGE_TABLE, t, vector)
        )
        return self

    def whisper(self, t: float, vector: np.ndarray) -> "Video":
        self._n += 1
        self._rows.append(
            (f"wh_{self.file_id}_{self._n}", "whisper", "vec_text", t, vector)
        )
        return self

    def ocr(self, start: float, end: float, body: str) -> "Video":
        self._ocr.append((start, end, body))
        return self

    def transcript_chunk(self, start: float, end: float, body: str) -> "Video":
        self._chunks.append((start, end, body))
        return self

    def scan(self, seconds: int, *, hashes: dict[int, int] | None = None,
             sizes: dict[int, int] | None = None, default_size: int = 1000,
             version: int = 1, hash_bytes: bytes | None = None,
             size_bytes: bytes | None = None) -> "Video":
        """``hashes`` maps a second to how many leading bits its hash has set
        (every other second is all zero); ``sizes`` overrides a second's size."""
        hashes = hashes or {}
        sizes = sizes or {}
        if hash_bytes is None:
            hash_bytes = b"".join(hash_with_bits(hashes.get(s, 0)) for s in range(seconds))
        if size_bytes is None:
            size_bytes = np.array(
                [sizes.get(s, default_size) for s in range(seconds)], dtype="<u2",
            ).tobytes()
        self._scan = (version, seconds, hash_bytes, size_bytes)
        return self

    def commit(self, *, reverse: bool = False, with_file: bool = True) -> None:
        rows = list(reversed(self._rows)) if reverse else list(self._rows)
        with database.get_search_db() as s:
            if with_file:
                s.add(IndexedFile(
                    file_id=self.file_id, drive=self.drive,
                    filename=f"{self.file_id}.mp4",
                    file_path=f"/drives/{self.drive}/{self.file_id}.mp4",
                    file_type="video", mime_type=self.mime, file_size=1,
                    duration=self.duration, active=True,
                ))
            s.add_all([
                Embedding(
                    id=emb_id, file_id=self.file_id, embedding_type=etype,
                    vector_table=table, timestamp_start=t, content_preview="x",
                )
                for emb_id, etype, table, t, _ in rows
            ])
        with self.index.engine.begin() as conn:
            for emb_id, etype, table, _, vector in rows:
                if vector is None:
                    continue
                conn.execute(
                    text(
                        f"INSERT INTO {table}(embedding_id, embedding_type, vector) "
                        "VALUES (:i, :t, :v)"
                    ),
                    {"i": emb_id, "t": etype,
                     "v": np.asarray(vector, dtype=np.float32).tobytes()},
                )
            ocr = list(reversed(self._ocr)) if reverse else self._ocr
            for n, (start, end, body) in enumerate(ocr):
                conn.execute(
                    text(
                        "INSERT INTO video_frame_texts(file_id, chunk_index, start_time, "
                        "end_time, text) VALUES (:f, :c, :s, :e, :t)"
                    ),
                    {"f": self.file_id, "c": n, "s": start, "e": end, "t": body},
                )
            for n, (start, end, body) in enumerate(self._chunks):
                conn.execute(
                    text(
                        "INSERT INTO transcript_chunks(file_id, chunk_index, text, language, "
                        "timestamp_start, timestamp_end, created_at) "
                        "VALUES (:f, :c, :t, 'en', :s, :e, :now)"
                    ),
                    {"f": self.file_id, "c": n, "t": body, "s": start, "e": end,
                     "now": datetime.now(UTC)},
                )
            if self._scan is not None:
                version, seconds, hash_bytes, size_bytes = self._scan
                conn.execute(
                    text(
                        "INSERT INTO video_frame_scans(file_id, version, seconds, hashes, "
                        "sizes, created_at) VALUES (:f, :v, :n, :h, :s, :now)"
                    ),
                    {"f": self.file_id, "v": version, "n": seconds, "h": hash_bytes,
                     "s": size_bytes, "now": datetime.now(UTC)},
                )
