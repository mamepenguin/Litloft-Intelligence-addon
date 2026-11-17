"""Shared rig for the refine-alignment acceptance tests (SPEC-ADDON-127, SPEC-ADDON-128).

A fake ``whisperx`` stands in for the model: it raises ``ValueError`` for a language it has
no alignment model for (as the real one does) and aligns by spreading the text evenly over
the segment it is given. The refine job itself runs for real against a temp SQLite DB.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

for _mod in ("torch", "sentence_transformers", "faster_whisper", "sqlite_vec"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.database import Base  # noqa: E402
from app.models import IndexedFile, KeptOriginal, TranscriptChunk, TranscriptWord  # noqa: E402
from app.workers import aligner, refine  # noqa: E402
from tests.llm_helpers import resolved_with  # noqa: E402

FILE = "f00000000001"
PUNCT = "。、，．.,!?！？"

_FTS_DDL = (
    "CREATE VIRTUAL TABLE fts_files USING fts5(file_id, filename, title, description, "
    "tags_text, tokenize='trigram')",
    "CREATE VIRTUAL TABLE fts_files_word USING fts5(file_id, filename, title, description, "
    "tags_text, tokenize=\"unicode61 remove_diacritics 2\")",
    "CREATE VIRTUAL TABLE fts_transcripts USING fts5(file_id, chunk_index, text, "
    "tokenize='trigram')",
    "CREATE VIRTUAL TABLE fts_transcripts_word USING fts5(file_id, chunk_index, text, "
    "tokenize=\"unicode61 remove_diacritics 2\")",
    "CREATE VIRTUAL TABLE fts_text_content USING fts5(file_id, chunk_index, page, text, "
    "tokenize='trigram')",
    "CREATE VIRTUAL TABLE fts_text_content_word USING fts5(file_id, chunk_index, page, text, "
    "tokenize=\"unicode61 remove_diacritics 2\")",
    "CREATE VIRTUAL TABLE fts_retrieval_keywords USING fts5(file_id, keywords, "
    "tokenize='trigram')",
)


class FakeWhisperX:
    """Configurable stand-in for the ``whisperx`` module."""

    def __init__(
        self,
        *,
        models=("ja", "en", "ko", "zh", "pt"),
        audio_ok=True,
        fail_texts=(),
        fail_mode="empty",
        load_error=None,
    ):
        self.models = set(models)
        self.audio_ok = audio_ok
        self.fail_texts = set(fail_texts)
        self.fail_mode = fail_mode
        self.load_error = load_error
        self.loaded: list[str] = []
        self.align_calls = 0

    def load_audio(self, path):
        if not self.audio_ok:
            raise RuntimeError("cannot decode audio")
        return np.zeros(16000 * 60, dtype=np.float32)

    def load_align_model(self, language_code, device, model_dir=None):
        self.loaded.append(language_code)
        if self.load_error is not None:
            raise self.load_error
        if language_code not in self.models:
            raise ValueError(f"No default align-model for language: {language_code}")
        return object(), {"language": language_code}

    def align(self, transcript, model, align_model_metadata, audio, device,
              return_char_alignments=False, **_kw):
        self.align_calls += 1
        seg = transcript[0]
        body = seg["text"]
        if body in self.fail_texts:
            if self.fail_mode == "raises":
                raise RuntimeError("alignment blew up")
            return {"segments": []}
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 1.0))
        glyphs = [c for c in body if not c.isspace()]
        step = (end - start) / max(len(glyphs), 1)
        chars = []
        for n, ch in enumerate(glyphs):
            if ch in PUNCT:
                chars.append({"char": ch})
            else:
                chars.append({"char": ch, "start": start + n * step,
                              "end": start + (n + 1) * step})
        tokens = body.split()
        wstep = (end - start) / max(len(tokens), 1)
        words = [{"word": t, "start": start + n * wstep, "end": start + (n + 1) * wstep}
                 for n, t in enumerate(tokens)]
        return {"segments": [{"text": body, "start": start, "end": end,
                              "chars": chars, "words": words}]}


def install_whisperx(monkeypatch, **kwargs) -> FakeWhisperX:
    import app.config as app_config

    fake = FakeWhisperX(**kwargs)
    module = types.ModuleType("whisperx")
    module.load_audio = fake.load_audio
    module.load_align_model = fake.load_align_model
    module.align = fake.align
    monkeypatch.setitem(sys.modules, "whisperx", module)
    monkeypatch.setattr(app_config, "validate_file_path", lambda *a, **k: True)
    aligner.release_all()
    return fake


def make_db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'search.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for ddl in _FTS_DDL:
            conn.execute(text(ddl))
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def _glyph_words(body):
    return [c for c in body if not c.isspace()]


def seed_file(Session, chunks, *, language, words=True, file_id=FILE, refined=()):
    """chunks: (id, index, text, start, end). Words are one row per character for CJK
    languages (``ja``/``jpn``/``zh``) and one per whitespace-separated token otherwise.
    ``refined`` lists chunk ids that already carry refined marks."""
    cjk = language[:2].lower() in {"ja", "jp", "zh"}
    with Session() as s:
        s.add(IndexedFile(
            file_id=file_id, drive="d", filename="talk.mp4", file_path="/nowhere/talk.mp4",
            file_type="video", mime_type="video/mp4", file_size=1, active=True,
        ))
        for cid, idx, body, start, end in chunks:
            s.add(TranscriptChunk(
                id=cid, file_id=file_id, chunk_index=idx, text=body, language=language,
                timestamp_start=start, timestamp_end=end, created_at=datetime.now(UTC),
                text_refined_at=datetime(2026, 1, 1) if cid in refined else None,
                refined_model="old-model" if cid in refined else None,
            ))
            if words:
                toks = _glyph_words(body) if cjk else body.split()
                step = (end - start) / max(len(toks), 1)
                for n, tok in enumerate(toks):
                    s.add(TranscriptWord(
                        file_id=file_id, text=tok, language=language,
                        timestamp_start=start + n * step, timestamp_end=start + (n + 1) * step,
                    ))
        s.flush()
        for _cid, idx, body, _s, _e in chunks:
            for table in ("fts_transcripts", "fts_transcripts_word"):
                s.execute(
                    text(f"INSERT INTO {table}(file_id, chunk_index, text) VALUES (:f, :i, :t)"),
                    {"f": file_id, "i": str(idx), "t": body},
                )
        s.commit()


def snapshot(Session, file_id=FILE):
    """Everything the spec says an unapplied run must leave alone."""
    with Session() as s:
        chunks = [
            (r.id, r.chunk_index, r.text, r.timestamp_start, r.timestamp_end,
             r.text_refined_at, r.refined_model)
            for r in s.query(TranscriptChunk).filter_by(file_id=file_id)
            .order_by(TranscriptChunk.timestamp_start, TranscriptChunk.id)
        ]
        words = [
            (r.text, r.timestamp_start, r.timestamp_end)
            for r in s.query(TranscriptWord).filter_by(file_id=file_id)
            .order_by(TranscriptWord.timestamp_start, TranscriptWord.id)
        ]
        kept = [
            (r.kind, r.trigger, r.chunks_json, r.words_json)
            for r in s.query(KeptOriginal).filter_by(file_id=file_id)
        ]
        generation = s.get(IndexedFile, file_id).transcript_generation
    return {"chunks": chunks, "words": words, "kept": kept, "generation": generation}


def chunk_rows(Session, file_id=FILE):
    with Session() as s:
        return [
            {"id": r.id, "text": r.text, "start": r.timestamp_start, "end": r.timestamp_end,
             "refined_at": r.text_refined_at, "model": r.refined_model}
            for r in s.query(TranscriptChunk).filter_by(file_id=file_id)
            .order_by(TranscriptChunk.timestamp_start, TranscriptChunk.id)
        ]


def word_rows_in(Session, start, end, file_id=FILE):
    with Session() as s:
        return [
            (r.text, r.timestamp_start, r.timestamp_end)
            for r in s.query(TranscriptWord)
            .filter(TranscriptWord.file_id == file_id,
                    TranscriptWord.timestamp_start >= start,
                    TranscriptWord.timestamp_start < end)
            .order_by(TranscriptWord.timestamp_start, TranscriptWord.id)
        ]


def squash(body):
    return "".join(body.split())


def joined(Session, file_id=FILE):
    return squash("".join(r["text"] for r in chunk_rows(Session, file_id)))


def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def completed_payloads(events):
    found = []

    def _walk(value):
        if isinstance(value, dict):
            if "refined_count" in value:
                found.append(value)
            for v in value.values():
                _walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                _walk(v)

    for call in events.call_args_list:
        _walk(list(call.args) + list(call.kwargs.values()))
    return found


class Run:
    def __init__(self, events, rechunk_calls):
        self.events = events
        self.rechunk_calls = rechunk_calls

    @property
    def completed(self):
        payloads = completed_payloads(self.events)
        assert payloads, self.events.call_args_list
        return payloads[-1]


async def run_refine(Session, monkeypatch, refined, chunk_ids=None, *, model="new-model",
                     file_id=FILE):
    """Run ``_run_refine_job`` for real; ``refined`` maps chunk id -> refined text."""
    events = AsyncMock()
    rechunk_calls: list = []
    real_rechunk = refine.rechunk_from_words

    def _spy(*a, **k):
        rechunk_calls.append(1)
        return real_rechunk(*a, **k)

    @contextmanager
    def _db():
        session = Session()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(refine, "get_search_db", _db)
    monkeypatch.setattr(refine, "_emit_ws_event", events)
    monkeypatch.setattr(refine, "post_addon_event", AsyncMock())
    monkeypatch.setattr(refine, "recompute_chunk_embeddings", AsyncMock())
    monkeypatch.setattr(refine, "rechunk_from_words", _spy)
    monkeypatch.setattr(refine.aligner, "job_scope", nullcontext)
    client = MagicMock()
    client.enabled = True
    client.generate_json = AsyncMock(
        return_value=[{"id": cid, "text_refined": body} for cid, body in refined.items()]
    )
    ids = list(refined) if chunk_ids is None else chunk_ids
    await refine._run_refine_job(file_id, "job1", ids, resolved_with(client, model=model))
    return Run(events, rechunk_calls)
