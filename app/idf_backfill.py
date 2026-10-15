"""Keeps ``idf_doc_terms`` equal to each file's TF-IDF term set.

The corpus IDF (``tfidf._build_corpus_idf``) is an aggregate over these
rows, so a file is tokenized when its transcript or filename changes
instead of on every IDF rebuild. Writers mark the file with
:func:`mark_idf_terms_stale` in their own transaction; a single loop per
process re-tokenizes marked files in a child process.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field

from sqlalchemy import event
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from app import tfidf
from app.database import get_search_db, get_search_db_read

logger = logging.getLogger(__name__)

_IDF_BACKFILL_CHILD_TIMEOUT_SECONDS = 300.0
#: How long the loop waits after a request before starting a pass, so a
#: burst of writes (refine windows, a whisper backlog) becomes one pass.
_IDF_BACKFILL_COALESCE_SECONDS = 5.0
_IDF_BACKFILL_RETRY_SECONDS = 60.0
_POLL_SECONDS = 0.05

_SESSION_KEY = "idf_backfill_requested"

_requested = threading.Event()
_state_lock = threading.Lock()
#: file_id -> idf_terms_gen at which it failed on its own in this process.
_held: dict[str, int] = {}
_current_child: tfidf.TokenizerChild | None = None


def request_idf_backfill() -> None:
    """Ask the loop for a pass. Safe from any thread, with or without a loop."""
    _requested.set()


def idf_backfill_requested() -> bool:
    return _requested.is_set()


def reset_idf_backfill_state() -> None:
    """Forget pending requests and held files, as a fresh process would."""
    _requested.clear()
    with _state_lock:
        _held.clear()


def mark_idf_terms_stale(session: Session, file_id: str) -> None:
    """Flag ``file_id``'s term set stale in ``session``'s transaction.

    The backfill request goes out when that transaction commits, so a
    pass never starts before the change is visible, and a rollback sends
    nothing.
    """
    session.execute(
        sql_text(
            "UPDATE indexed_files SET idf_terms_indexed = 0, "
            "idf_terms_gen = idf_terms_gen + 1 WHERE file_id = :fid"
        ),
        {"fid": file_id},
    )
    session.info[_SESSION_KEY] = True


@event.listens_for(Session, "after_commit")
def _request_after_commit(session: Session) -> None:
    if session.info.pop(_SESSION_KEY, False):
        request_idf_backfill()


@event.listens_for(Session, "after_rollback")
def _forget_after_rollback(session: Session) -> None:
    session.info.pop(_SESSION_KEY, None)


# --- One pass ---------------------------------------------------------------


@dataclass
class PassResult:
    written: int = 0
    skipped: int = 0
    failed: int = 0
    held: int = 0
    changed: bool = False
    write_failed: bool = False
    drives: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _Doc:
    file_id: str
    gen: int
    drive: str
    filename: str
    transcript: str


def _queue_paused() -> bool:
    from app import dependencies

    try:
        return dependencies.get_index_manager().is_paused()
    except RuntimeError:
        return False


def _wait_while_paused(stop: threading.Event) -> bool:
    """Block while the index queue is paused. False when asked to stop."""
    while _queue_paused():
        if stop.wait(_POLL_SECONDS * 4):
            return False
    return not stop.is_set()


def _collect() -> list[tuple[str, int]]:
    with get_search_db_read() as session:
        rows = session.execute(sql_text(
            "SELECT file_id, idf_terms_gen FROM indexed_files "
            "WHERE active = 1 AND idf_terms_indexed = 0 ORDER BY file_id"
        )).all()
    return [(r[0], int(r[1])) for r in rows]


def _read_docs(file_ids: list[str]) -> tuple[list[_Doc], int]:
    """Read each file's generation before its inputs.

    A change committed between the two reads bumps the generation past
    the value read here, so the write that would store the old inputs
    is refused. Files gone or inactive by now are counted, not read.
    """
    docs: list[_Doc] = []
    gone = 0
    with get_search_db_read() as session:
        for fid in file_ids:
            row = session.execute(
                sql_text(
                    "SELECT idf_terms_gen, active, drive, filename "
                    "FROM indexed_files WHERE file_id = :fid"
                ),
                {"fid": fid},
            ).first()
            if row is None or not row[1]:
                gone += 1
                continue
            transcript = tfidf._get_transcript_text(session, fid)
            docs.append(_Doc(fid, int(row[0]), row[2], row[3], transcript))
    return docs, gone


def _write(doc: _Doc, terms: list[str]) -> str:
    """Store ``terms`` for ``doc``. Returns "written" or "skipped"."""
    with get_search_db() as session:
        row = session.execute(
            sql_text(
                "SELECT idf_terms_gen, active FROM indexed_files "
                "WHERE file_id = :fid"
            ),
            {"fid": doc.file_id},
        ).first()
        if row is None or not row[1] or int(row[0]) != doc.gen:
            return "skipped"
        session.execute(
            sql_text("DELETE FROM idf_doc_terms WHERE file_id = :fid"),
            {"fid": doc.file_id},
        )
        if terms:
            session.execute(
                sql_text(
                    "INSERT INTO idf_doc_terms (file_id, term) VALUES (:fid, :term)"
                ),
                [{"fid": doc.file_id, "term": t} for t in terms],
            )
        session.execute(
            sql_text(
                "UPDATE indexed_files SET idf_terms_indexed = 1 WHERE file_id = :fid"
            ),
            {"fid": doc.file_id},
        )
    return "written"


def _set_child(child: tfidf.TokenizerChild | None) -> None:
    global _current_child
    with _state_lock:
        _current_child = child


def _terminate_current_child() -> None:
    with _state_lock:
        child = _current_child
    if child is not None:
        child.stop_process()


class _Tokenizer:
    """The pass's child: started on first use, replaced after a failure."""

    def __init__(self) -> None:
        self._child: tfidf.TokenizerChild | None = None

    def term_sets(self, docs: list[_Doc]) -> list[list[str]]:
        if self._child is None:
            self._child = tfidf.TokenizerChild()
            _set_child(self._child)
        try:
            return self._child.term_sets(
                [(d.filename, d.transcript) for d in docs],
                _IDF_BACKFILL_CHILD_TIMEOUT_SECONDS,
            )
        except tfidf.TokenizerChildFailed:
            self.close()
            raise

    def close(self) -> None:
        if self._child is not None:
            self._child.terminate()
            self._child = None
            _set_child(None)


def _write_all(docs: list[_Doc], term_sets: list[list[str]], result: PassResult,
               stop: threading.Event) -> None:
    for doc, terms in zip(docs, term_sets, strict=True):
        if stop.is_set():
            return
        try:
            outcome = _write(doc, terms)
        except Exception:
            logger.warning(
                "IDF term backfill: writing %s failed", doc.file_id, exc_info=True,
            )
            result.failed += 1
            result.write_failed = True
            continue
        if outcome == "written":
            result.written += 1
            result.drives.add(doc.drive)
        else:
            result.skipped += 1
            result.changed = True


def _tokenize_one_at_a_time(docs: list[_Doc], tokenizer: _Tokenizer,
                            result: PassResult, stop: threading.Event) -> None:
    for doc in docs:
        if not _wait_while_paused(stop):
            return
        try:
            [terms] = tokenizer.term_sets([doc])
        except tfidf.TokenizerChildFailed as exc:
            if stop.is_set():
                return
            logger.warning(
                "IDF term backfill: cannot tokenize %s (%s); held until it changes",
                doc.file_id, exc,
            )
            with _state_lock:
                _held[doc.file_id] = doc.gen
            result.failed += 1
            continue
        _write_all([doc], [terms], result, stop)


def _run_batch(file_ids: list[str], tokenizer: _Tokenizer, result: PassResult,
               stop: threading.Event) -> None:
    docs, gone = _read_docs(file_ids)
    result.skipped += gone
    if not docs:
        return
    try:
        term_sets = tokenizer.term_sets(docs)
    except tfidf.TokenizerChildFailed as exc:
        if stop.is_set():
            return
        logger.info(
            "IDF term backfill: batch %s..%s failed (%s); retrying one file at a time",
            docs[0].file_id, docs[-1].file_id, exc,
        )
        _tokenize_one_at_a_time(docs, tokenizer, result, stop)
    else:
        _write_all(docs, term_sets, result, stop)
    if result.drives:
        tfidf.drop_corpus_idf(result.drives)


def _remaining(held_at_start: set[str]) -> int:
    with get_search_db_read() as session:
        ids = session.execute(sql_text(
            "SELECT file_id FROM indexed_files "
            "WHERE active = 1 AND idf_terms_indexed = 0"
        )).scalars().all()
    return sum(1 for fid in ids if fid not in held_at_start)


def _take_unheld(collected: list[tuple[str, int]]) -> tuple[list[str], set[str]]:
    """Drop held files from ``collected``; release those whose generation moved."""
    todo: list[str] = []
    held: set[str] = set()
    with _state_lock:
        for fid, gen in collected:
            held_gen = _held.get(fid)
            if held_gen == gen:
                held.add(fid)
                continue
            if held_gen is not None:
                del _held[fid]
            todo.append(fid)
    return todo, held


def run_idf_backfill_pass(stop: threading.Event | None = None) -> PassResult:
    """Write the term set of every active file flagged stale.

    Runs on the calling thread; tokenization happens in a child process.
    ``stop`` makes the pass return at the next file boundary.
    """
    stop = stop or threading.Event()
    result = PassResult()
    todo, held = _take_unheld(_collect())
    result.held = len(held)
    if not todo:
        logger.debug("IDF term backfill: nothing to write")
        return result

    tokenizer = _Tokenizer()
    try:
        batch_size = tfidf._CORPUS_BATCH_FILES
        for start in range(0, len(todo), batch_size):
            if not _wait_while_paused(stop):
                break
            _run_batch(todo[start:start + batch_size], tokenizer, result, stop)
    finally:
        tokenizer.close()
        try:
            remaining = _remaining(held)
        except Exception:  # noqa: BLE001 — the line still reports the rest
            remaining = -1
        logger.info(
            "IDF term backfill: written=%d skipped=%d failed=%d held=%d remaining=%d",
            result.written, result.skipped, result.failed, result.held, remaining,
        )
    return result


# --- The loop ---------------------------------------------------------------


async def _wait_for_request(retry_at: float | None) -> None:
    while not _requested.is_set():
        if retry_at is not None and time.monotonic() >= retry_at:
            return
        await asyncio.sleep(_POLL_SECONDS)
    await asyncio.sleep(_IDF_BACKFILL_COALESCE_SECONDS)


async def idf_backfill_loop() -> None:
    """Run a pass at startup, then one per (coalesced) request, forever.

    A pass that raises is logged and retried later; the loop itself ends
    only when cancelled, and then terminates the pass's child.
    """
    stop = threading.Event()
    retry_at: float | None = None
    first = True
    pass_task: asyncio.Future | None = None
    try:
        while True:
            if not first:
                await _wait_for_request(retry_at)
            first = False
            retry_at = None
            _requested.clear()
            pass_task = asyncio.ensure_future(
                asyncio.to_thread(run_idf_backfill_pass, stop)
            )
            try:
                result = await asyncio.shield(pass_task)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("IDF term backfill pass failed; retrying later")
                retry_at = time.monotonic() + _IDF_BACKFILL_RETRY_SECONDS
                continue
            finally:
                if pass_task.done():
                    pass_task = None
            if result.changed:
                request_idf_backfill()
            if result.write_failed:
                retry_at = time.monotonic() + _IDF_BACKFILL_RETRY_SECONDS
    except asyncio.CancelledError:
        stop.set()
        _terminate_current_child()
        if pass_task is not None:
            await asyncio.wait({pass_task}, timeout=5)
        raise
