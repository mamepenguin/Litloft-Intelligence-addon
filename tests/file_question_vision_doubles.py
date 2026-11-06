"""Fixtures and wire doubles for the file-question picture and position tests.

The LLM is the addon's real client class with only its HTTP layer replaced by
an ``httpx.MockTransport``, so a test reads the request the provider would have
received without contacting a model. The request is the observable thing: the
new streaming call may have any name, but it still has to leave through the
client's own transport.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI

from app.config import FeaturesConfig, LLMConfig, RagConfig
from app.frame_cache import frame_cache_path
from app.llm import (
    _PROBE_IMAGE_B64,
    LLMClient,
    OllamaLLMClient,
)
from app.llm_routing import Defer, LLMProfile, Resolved, Skip
from app.routers import rag as rag_router
from tests.file_question_doubles import (  # noqa: F401 - re-exported fixtures
    ASKABLE,
    DRIVE,
    EPUB,
    MEDIA,
    NOTE,
    OTHER,
    PDF,
    RecordingLLM,
    Seeder,
    chunk,
    drive_to_disconnect,
    frames,
    mentioned,
    numbered_in_order,
    settings_for_question,
    toward,
)
from tests.search_endpoint_db import patch_everywhere
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture

TEXT_MODEL = "text-model"
VISION_MODEL = "vision-model"
LLM_BASE = "http://llm.test/v1"
OLLAMA_BASE = "http://ollama.test"

#: An image body the picture loader can hand through unchanged when Pillow is
#: the suite's MagicMock: the marker makes an accidental log or store visible.
PICTURE_BYTES = b"PICTURE-BYTES-MARKER"


# ---------------------------------------------------------------------------
# Settings and routing
# ---------------------------------------------------------------------------


def vision_settings(
    make_settings,
    *,
    rag: bool = True,
    output_language: str = "auto",
    budget: int | None = None,
    vision_model: str = VISION_MODEL,
    text_model: str = TEXT_MODEL,
    profiles: dict | None = None,
    **extra,
):
    rag_section = (
        RagConfig()
        if budget is None
        else replace(RagConfig(), file_question_max_chars=budget)
    )
    kwargs = dict(
        features=FeaturesConfig(rag=rag),
        llm=LLMConfig(
            provider="openai_compatible",
            base_url=LLM_BASE,
            model=text_model,
            vision_model=vision_model,
            output_language=output_language,
        ),
        rag=rag_section,
    )
    if profiles is not None:
        kwargs["llm_section"] = profiles
    kwargs.update(extra)
    return make_settings(**kwargs)


def profiles(
    *,
    vision_model: str = VISION_MODEL,
    text_model: str = TEXT_MODEL,
    base_url: str = LLM_BASE,
    offhost: bool = False,
) -> dict:
    """A one-profile ``llm`` section, the shape ``build_routing`` reads."""
    return {
        "profiles": {
            "default": {
                "provider": "openai_compatible",
                "base_url": base_url,
                "model": text_model,
                "vision_model": vision_model,
                "offhost": offhost,
            }
        },
        "routing": {"default": "default"},
    }


class Routing:
    """What ``llm_routing.resolve`` answers, and every call it was asked about."""

    def __init__(self):
        self.asked: list[tuple[str, str, str | None]] = []
        self.result = None

    def serve(self, resolved: Resolved) -> Resolved:
        self.result = resolved
        return resolved

    def skip(self) -> None:
        self.result = Skip("disabled")

    def defer(self) -> None:
        self.result = Defer("policy unreadable")


@pytest.fixture()
def db(search_index):  # noqa: F811
    return Seeder(search_index)


@pytest.fixture()
def addon(monkeypatch, make_settings, tmp_path):
    """The addon's settings, applied everywhere; files may live under ``tmp_path``."""

    def _apply(**overrides):
        from app.main import app  # noqa: F401 - import every router before patching

        overrides.setdefault("allowed_base_dirs", (str(tmp_path) + os.sep,))
        settings = vision_settings(make_settings, **overrides)
        patch_everywhere(monkeypatch, "settings", settings)
        return settings

    _apply()
    return _apply


@pytest.fixture()
def llm(monkeypatch):
    from app import llm_routing

    routing = Routing()

    async def _resolve(drive, feature, requested=None):
        routing.asked.append((drive, feature, requested))
        return routing.result

    monkeypatch.setattr(llm_routing, "resolve", _resolve)
    return routing


@pytest.fixture()
async def call(search_index):  # noqa: F811
    from app.main import app

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://intelligence"
    ) as client:

        async def _call(
            method: str, path: str, *, json=None, content=None, drive=DRIVE,
        ):
            headers = {"Content-Type": "application/json"}
            if drive is not None:
                headers["X-Lit-Drive"] = drive
            return await client.request(
                method, path, json=json, content=content, headers=headers,
            )

        yield _call


@pytest.fixture()
async def ask_slots(monkeypatch):
    monkeypatch.setattr(rag_router, "_ask_semaphore", None)
    monkeypatch.setattr(rag_router, "_ask_semaphore_loop", None)
    return rag_router._get_ask_semaphore()


@pytest.fixture()
def statements(search_index):  # noqa: F811
    """Every statement the search engine runs, and the writes among them."""
    from sqlalchemy import event

    import app.database as database

    class Log:
        def __init__(self):
            self.statements: list[str] = []

        def _before(self, _conn, _cursor, statement, _params, _context, _many):
            self.statements.append(statement)

        def reset(self) -> None:
            self.statements.clear()

        def writes(self) -> list[str]:
            verbs = ("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "DROP", "ALTER")
            return [s for s in self.statements if s.lstrip().upper().startswith(verbs)]

    log = Log()
    engine = database._search_engine
    event.listen(engine, "before_cursor_execute", log._before)
    try:
        yield log
    finally:
        event.remove(engine, "before_cursor_execute", log._before)


# ---------------------------------------------------------------------------
# The provider wire
# ---------------------------------------------------------------------------


def openai_config(**overrides) -> LLMConfig:
    defaults = dict(
        provider="openai_compatible",
        base_url=LLM_BASE,
        model=TEXT_MODEL,
        vision_model=VISION_MODEL,
        max_tokens=256,
        temperature=0.0,
        vision_max_tokens=128,
        vision_temperature=0.0,
        retry_attempts=0,
        retry_base_delay=0.0,
        retry_max_delay=0.0,
        request_timeout_seconds=5.0,
        request_connect_timeout_seconds=5.0,
    )
    return LLMConfig(**{**defaults, **overrides})


def ollama_config(**overrides) -> LLMConfig:
    defaults = dict(
        provider="ollama",
        base_url=OLLAMA_BASE,
        model=TEXT_MODEL,
        vision_model=VISION_MODEL,
        max_tokens=256,
        temperature=0.0,
        vision_max_tokens=128,
        vision_temperature=0.0,
        retry_attempts=0,
        retry_base_delay=0.0,
        retry_max_delay=0.0,
        request_timeout_seconds=5.0,
        request_connect_timeout_seconds=5.0,
    )
    return LLMConfig(**{**defaults, **overrides})


class Wire:
    """The requests a client would send, and the script answering them.

    ``script(body)`` returns ``(status, text)`` or raises an exception. The
    default answers a stream with a usable answer.
    """

    def __init__(self, provider: str, script=None):
        self.provider = provider
        self.requests: list[dict] = []
        self.paths: list[str] = []
        self.script = script or (lambda body: (200, self.default_answer(body)))
        # Values of the shared Ask semaphore seen from inside the stream, one
        # per text-only request. ``None`` means the script did not record it.
        self.text_request_slots: list[int] = []

    def default_answer(self, body: dict) -> str:
        return self.vision_answer if self.has_image(body) else self.text_answer

    # -- reading the request ------------------------------------------------

    def messages(self, body: dict) -> list[dict]:
        return body.get("messages", [])

    def has_image(self, body: dict) -> bool:
        if self.provider == "ollama":
            return any(message.get("images") for message in self.messages(body))
        for message in self.messages(body):
            content = message.get("content")
            if isinstance(content, list) and any(
                part.get("type") == "image_url" for part in content
            ):
                return True
        return False

    def is_probe(self, body: dict) -> bool:
        if self.provider == "ollama":
            user = [m for m in self.messages(body) if m.get("role") == "user"]
            return bool(user) and user[0].get("images") == [_PROBE_IMAGE_B64]
        return len(self.messages(body)) == 1

    def image_payloads(self, body: dict) -> list[str]:
        """The image as the provider receives it: a data URL or bare base64."""
        out: list[str] = []
        if self.provider == "ollama":
            for message in self.messages(body):
                out.extend(message.get("images") or [])
            return out
        for message in self.messages(body):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "image_url":
                    out.append(part["image_url"]["url"])
        return out

    def text_of(self, body: dict) -> str:
        parts: list[str] = []
        for message in self.messages(body):
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(
                    part.get("text", "")
                    for part in content
                    if part.get("type") == "text"
                )
        return "\n".join(parts)

    def system_of(self, body: dict) -> str:
        return "\n".join(
            self._content_text(message.get("content"))
            for message in self.messages(body)
            if message.get("role") == "system"
        )

    @staticmethod
    def _content_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "")
                for part in content
                if part.get("type") == "text"
            )
        return ""

    @property
    def image_bodies(self) -> list[dict]:
        return [body for body in self.requests if self.has_image(body)]

    @property
    def text_bodies(self) -> list[dict]:
        return [
            body
            for body in self.requests
            if not self.has_image(body) and not self.is_probe(body)
        ]

    @property
    def images(self) -> list[str]:
        return [
            payload for body in self.requests for payload in self.image_payloads(body)
        ]

    @property
    def prompts(self) -> list[str]:
        return [self.text_of(body) for body in self.requests]

    @property
    def system_prompts(self) -> list[str]:
        return [self.system_of(body) for body in self.requests]

    vision_answer = "VISION ANSWER"
    text_answer = "TEXT ONLY ANSWER"



def _openai_handler(wire: Wire):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        wire.requests.append(body)
        wire.paths.append(request.url.path)
        status, text = _script_result(wire, body)
        if body.get("stream"):
            chunk = {
                "id": "1", "object": "chat.completion.chunk", "created": 0,
                "model": body.get("model"),
                "choices": [{
                    "index": 0, "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }],
            }
            done = {
                "id": "1", "object": "chat.completion.chunk", "created": 0,
                "model": body.get("model"),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            content = (
                f"data: {json.dumps(chunk)}\n\n"
                f"data: {json.dumps(done)}\n\n"
                "data: [DONE]\n\n"
            )
            return httpx.Response(
                status,
                headers={"content-type": "text/event-stream"},
                content=content.encode(),
            )
        return httpx.Response(
            status,
            json={
                "id": "1", "object": "chat.completion", "created": 0,
                "model": body.get("model"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
            },
        )

    return handler


def _ollama_handler(wire: Wire):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        wire.requests.append(body)
        wire.paths.append(request.url.path)
        status, text = _script_result(wire, body)
        if body.get("stream"):
            lines = [
                {"message": {"role": "assistant", "content": text}, "done": False},
                {"message": {"role": "assistant", "content": ""}, "done": True},
            ]
            return httpx.Response(
                status,
                content=("\n".join(json.dumps(line) for line in lines) + "\n").encode(),
            )
        return httpx.Response(
            status,
            json={
                "message": {"role": "assistant", "content": text},
                "done": True,
                "done_reason": "stop",
            },
        )

    return handler


def _script_result(wire: Wire, body: dict):
    result = wire.script(body)
    if isinstance(result, BaseException):
        raise result
    return result


def openai_client(config: LLMConfig, wire: Wire) -> LLMClient:
    client = LLMClient(config)
    client._client = AsyncOpenAI(  # noqa: SLF001 - the test replaces the transport
        base_url=config.base_url,
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_openai_handler(wire))),
    )
    return client


def ollama_client(config: LLMConfig, wire: Wire) -> OllamaLLMClient:
    client = OllamaLLMClient(config)
    client._http = httpx.AsyncClient(  # noqa: SLF001
        transport=httpx.MockTransport(_ollama_handler(wire))
    )
    return client


def resolved(
    client, config: LLMConfig, *, name: str = "test",
) -> Resolved:
    profile = LLMProfile(
        name=name, config=config, offhost=False, agentic=False, api_key_env="",
    )
    return Resolved(profile=profile, client=client)


def vision_client(
    llm: Routing, *, provider: str = "openai_compatible", name: str = "test",
    script=None, config: LLMConfig | None = None, wire: Wire | None = None,
):
    """Install a real client behind ``resolve`` and return it with its wire."""
    config = config or (
        openai_config() if provider == "openai_compatible" else ollama_config()
    )
    wire = wire or Wire(provider, script)
    client = (
        openai_client(config, wire)
        if provider == "openai_compatible"
        else ollama_client(config, wire)
    )
    llm.serve(resolved(client, config, name=name))
    return client, wire


# ---------------------------------------------------------------------------
# Scripts the provider plays
# ---------------------------------------------------------------------------


def refusing(wire: Wire):
    """Reject every image request, and the capability probe, with a 400."""

    def script(body):
        if wire.is_probe(body) or wire.has_image(body):
            return 400, ""
        return 200, wire.text_answer

    return script


def failing(wire: Wire):
    def script(body):
        if wire.has_image(body):
            return 500, ""
        return 200, wire.text_answer

    return script


def timing_out(wire: Wire):
    def script(body):
        if wire.has_image(body):
            return httpx.ReadTimeout("timed out"), None
        return 200, wire.text_answer

    return script


async def record_vision_unsupported(client) -> None:
    """Make the description path record an unsupported verdict for this model."""
    from app.llm import (
        FAILURE_VISION_UNSUPPORTED,
        _vision_capability_cache,
        reset_vision_capability_cache,
    )

    reset_vision_capability_cache()
    await client.generate_vision(PICTURE_BYTES, "image/png", "describe")
    config = client._config
    key = (config.base_url or "", config.vision_model or "")
    assert _vision_capability_cache._verdicts.get(key) == FAILURE_VISION_UNSUPPORTED


# ---------------------------------------------------------------------------
# Seeding files on disk and in the index
# ---------------------------------------------------------------------------


def seed_picture(
    db: Seeder, file_id: str, *, text: str | None = None, path: Path,
    drive: str = DRIVE, mime: str = "image/png", active: bool = True,
    file_size: int | None = None, data: bytes = PICTURE_BYTES,
) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    db.file(
        file_id, drive=drive, mime=mime, active=active, file_path=str(path),
        file_size=file_size if file_size is not None else len(data),
    )
    if text is not None:
        db.document(text, chunk_index=0)
    return data


def seed_video(
    db: Seeder, file_id: str, *, text: str, path: Path, duration: float | None,
    drive: str = DRIVE, start: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"VIDEO-BYTES")
    db.file(
        file_id, drive=drive, mime="video/mp4", file_path=str(path),
        file_size=11, duration=duration,
    )
    db.transcript(text, start=start)


def seed_audio(
    db: Seeder, file_id: str, *, text: str, path: Path,
    drive: str = DRIVE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"AUDIO-BYTES")
    db.file(
        file_id, drive=drive, mime="audio/mpeg", file_path=str(path),
        file_size=11,
    )
    db.transcript(text, start=0.0)


def seed_document(
    db: Seeder, file_id: str, *, text: str, path: Path,
    mime: str = "application/pdf", drive: str = DRIVE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"DOCUMENT-BYTES")
    db.file(
        file_id, drive=drive, mime=mime, file_path=str(path), file_size=14,
    )
    db.document(text, chunk_index=0)


def write_large_frame(file_id: str, position: float, data: bytes) -> Path:
    path = frame_cache_path(file_id, position, "large")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def ffmpeg_writes(data: bytes, calls: list[subprocess.CompletedProcess] | None = None):
    """Stand in for ``subprocess.run`` and let the real cache write its file."""
    def _run(command, **kwargs):
        calls and calls.append(command)
        output = Path(command[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        return subprocess.CompletedProcess(command, 0)

    return _run


def ffmpeg_fails():
    def _run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1)

    return _run


def ffmpeg_times_out():
    def _run(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=15)

    return _run


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def ask_body(
    *, question: str = "what is this", position: float | None = None,
    profile: str | None = None,
) -> dict:
    body: dict = {"question": question}
    if position is not None:
        body["position_seconds"] = position
    if profile is not None:
        body["profile"] = profile
    return body


async def ask(
    call, file_id: str, *, question: str = "what is this",
    position: float | None = None, profile: str | None = None,
):
    return await call(
        "POST", f"/files/{file_id}/ask",
        json=ask_body(question=question, position=position, profile=profile),
    )


def events(response) -> list[tuple[str, dict]]:
    assert response.status_code == 200, response.text
    return frames(response.text)


def by_kind(response) -> dict[str, dict]:
    return dict(events(response))


async def disconnect_during_preparation(
    app, *, path: str, body: bytes, headers: list[tuple[bytes, bytes]],
    entered, release, timeout: float = 5.0,
):
    """Call ``app`` and hang up once the preparation seam has been reached.

    ``entered`` is set by the preparation double; ``release`` is what the
    double waits on, set here so a cancelled thread does not outlive the test.
    The receive channel is what Starlette watches, so the disconnect cancels
    the response task exactly as a closed tab does.
    """
    import asyncio

    queue: asyncio.Queue[dict] = asyncio.Queue()
    await queue.put({"type": "http.request", "body": body, "more_body": False})
    sent: list[dict] = []

    async def receive() -> dict:
        return await queue.get()

    async def send(message: dict) -> None:
        sent.append(message)

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
    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(asyncio.to_thread(entered.wait, timeout), timeout)
        await queue.put({"type": "http.disconnect"})
        try:
            await asyncio.wait_for(task, timeout)
        except asyncio.CancelledError:
            # Starlette cancels the response task on the disconnect; that is
            # the outcome the test is provoking.
            pass
    finally:
        release.set()
        if not task.done():
            task.cancel()
