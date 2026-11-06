"""Seeding, recording doubles and wire helpers for the file-question tests.

The search DB is the one ``init_search_db`` builds (``tests.vec_partition_db``),
so ``fts_text_content``, ``transcript_chunks`` and the partitioned ``vec_text``
table are the production ones.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import numpy as np

import app.database as database
from app.config import FeaturesConfig, LLMConfig, RagConfig
from app.llm import TextGeneration
from app.models import Embedding, IndexedFile, TranscriptChunk

DRIVE = "family"
OTHER = "elsewhere"

# Ids are 12 characters, the width of a core File id.
MEDIA = "vid000000001"
ASKABLE = "vid000000009"
PDF = "pdf000000001"
EPUB = "epub00000001"
NOTE = "note00000001"


def chunk(tag: str, size: int) -> str:
    """A chunk body of exactly ``size`` characters, identifiable in a prompt."""
    return tag.ljust(size, ".")


def toward(dim: int, cos: float) -> np.ndarray:
    """A unit vector whose cosine similarity to the first axis is ``cos``."""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = cos
    v[1] = float(np.sqrt(max(0.0, 1.0 - cos * cos)))
    return v


def file_type_for(mime: str) -> str:
    head = mime.split("/")[0]
    return head if head in ("video", "audio", "image") else "document"


class SearchDB:
    """Seeds files, chunks, section titles and stored vectors by hand."""

    def __init__(self, index):
        self.index = index
        self.dim = index.dim("vec_text")
        self._transcript_index: dict[str, int] = {}

    # -- files ---------------------------------------------------------------

    def file(
        self, file_id: str, *, drive: str = DRIVE, mime: str = "video/mp4",
        active: bool = True, file_path: str | None = None,
        file_size: int = 1, duration: float | None = None,
    ) -> None:
        name = f"{file_id}.bin"
        with database.get_search_db() as s:
            s.add(IndexedFile(
                file_id=file_id, drive=drive, filename=name,
                file_path=file_path or f"/drives/{drive}/{name}",
                file_type=file_type_for(mime), mime_type=mime,
                file_size=file_size, duration=duration, active=active,
            ))

    # -- chunks --------------------------------------------------------------

    def transcript(
        self, file_id: str, text: str, *, start: float, end: float | None = None,
        cos: float | None = 1.0,
    ) -> None:
        end = start + 5.0 if end is None else end
        index = self._transcript_index.get(file_id, 0)
        self._transcript_index[file_id] = index + 1
        with database.get_search_db() as s:
            s.add(TranscriptChunk(
                file_id=file_id, chunk_index=index, text=text,
                timestamp_start=start, timestamp_end=end,
            ))
        self.index.sql(
            "INSERT INTO fts_transcripts(file_id, chunk_index, text) "
            "VALUES (:f, :c, :t)",
            f=file_id, c=str(index), t=text,
        )
        if cos is not None:
            self._embed(
                f"wh_{file_id}_{start:07.2f}", file_id, "whisper", cos,
                timestamp_start=start, timestamp_end=end,
            )

    def document(
        self, file_id: str, text: str, *, chunk_index: int, page: str = "",
        cos: float | None = 1.0,
    ) -> None:
        self.index.sql(
            "INSERT INTO fts_text_content(file_id, chunk_index, page, text) "
            "VALUES (:f, :c, :p, :t)",
            f=file_id, c=str(chunk_index), p=page, t=text,
        )
        if cos is not None:
            self._embed(
                f"txt_{file_id}_{chunk_index:04d}", file_id, "text_content", cos,
                chunk_index=chunk_index,
            )

    def section(self, file_id: str, page: int, title: str) -> None:
        self.index.sql(
            "INSERT INTO document_sections(file_id, page, title) VALUES (:f, :p, :t)",
            f=file_id, p=page, t=title,
        )

    def _embed(
        self, embedding_id: str, file_id: str, embedding_type: str, cos: float,
        **columns,
    ) -> None:
        with database.get_search_db() as s:
            s.add(Embedding(
                id=embedding_id, file_id=file_id, embedding_type=embedding_type,
                vector_table="vec_text", content_preview="x", **columns,
            ))
        self.index.put_vector(
            "vec_text", embedding_id, toward(self.dim, cos), partition=embedding_type,
        )


class Seeder:
    """``SearchDB`` bound to one file at a time, so a test reads as a file's story."""

    def __init__(self, index):
        self.db = SearchDB(index)
        self._file = ""

    def file(self, file_id: str, **kwargs) -> "Seeder":
        self.db.file(file_id, **kwargs)
        self._file = file_id
        return self

    def transcript(self, text: str, **kwargs) -> "Seeder":
        self.db.transcript(self._file, text, **kwargs)
        return self

    def document(self, text: str, **kwargs) -> "Seeder":
        self.db.document(self._file, text, **kwargs)
        return self

    def section(self, page: int, title: str) -> "Seeder":
        self.db.section(self._file, page, title)
        return self

    def on(self, file_id: str) -> "Seeder":
        self._file = file_id
        return self


class RecordingLLM:
    """The routed LLM, recording every prompt and every stream it is given."""

    def __init__(
        self,
        deltas: tuple[str, ...] = (),
        *,
        stall: bool = False,
        fail_after: int | None = None,
    ):
        self.enabled = True
        self.deltas = deltas
        self.stall = stall
        self.fail_after = fail_after
        self.prompts: list[tuple[str, str]] = []
        self.closed = False

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens_override: int | None = None,
        temperature: float | None = None,
    ):
        self.prompts.append((system_prompt, user_prompt))
        try:
            for i, delta in enumerate(self.deltas):
                if self.fail_after is not None and i == self.fail_after:
                    raise RuntimeError("provider fell over")
                yield delta
            if self.stall:
                await asyncio.Event().wait()
        finally:
            self.closed = True

    async def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> TextGeneration:
        self.prompts.append((system_prompt, user_prompt))
        return TextGeneration("".join(self.deltas), None)

    @property
    def user_prompt(self) -> str:
        return self.prompts[-1][1]

    @property
    def system_prompt(self) -> str:
        return self.prompts[-1][0]


def frames(body: str) -> list[tuple[str, dict]]:
    """The ``(event, data)`` frames of an SSE body, in order."""
    out: list[tuple[str, dict]] = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        kind = ""
        data = ""
        for line in frame.splitlines():
            if line.startswith("event:"):
                kind = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        out.append((kind, json.loads(data) if data else {}))
    return out


def mentioned(prompt: str, texts: list[str]) -> list[str]:
    """The texts the prompt holds, in the order it holds them."""
    return [text for _, text in sorted(
        (prompt.index(text), text) for text in texts if text in prompt
    )]


def numbered_in_order(prompt: str, texts: list[str]) -> bool:
    """True when each text follows its own ``[n]``, numbering from 1 in order."""
    for number, text in enumerate(texts, start=1):
        marker = prompt.find(f"[{number}]")
        if marker == -1 or marker > prompt.find(text):
            return False
    return True


def settings_for_question(
    make_settings, *, rag: bool = True, output_language: str = "auto",
    budget: int | None = None,
):
    """Addon settings for a file question.

    ``budget`` sets ``rag.file_question_max_chars``; ``None`` leaves the
    section at its defaults, which is the 16,000-character fallback.
    """
    rag_section = RagConfig() if budget is None else replace(
        RagConfig(), file_question_max_chars=budget,
    )
    return make_settings(
        features=FeaturesConfig(rag=rag),
        llm=LLMConfig(
            provider="ollama", base_url="http://localhost:11434",
            model="llama3", output_language=output_language,
        ),
        rag=rag_section,
    )


async def drive_to_disconnect(
    app, *, path: str, body: bytes, headers: list[tuple[bytes, bytes]],
    first_body_frame: int = 1,
) -> None:
    """Call ``app`` over ASGI and hang up after ``first_body_frame`` body messages.

    Starlette reads ``http.disconnect`` on the receive channel and cancels the
    response task, which is what a closed browser tab does.
    """
    received: list[dict] = []
    queue: asyncio.Queue[dict] = asyncio.Queue()
    await queue.put({"type": "http.request", "body": body, "more_body": False})
    bodies = 0

    async def receive() -> dict:
        return await queue.get()

    async def send(message: dict) -> None:
        nonlocal bodies
        received.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            bodies += 1
            if bodies == first_body_frame:
                queue.put_nowait({"type": "http.disconnect"})

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "server": ("intelligence", 8100),
        "client": ("test", 1),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    assert received, "the app sent no response at all"
