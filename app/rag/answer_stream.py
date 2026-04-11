"""Incremental extractor for streamed JSON RAG answers.

The RAG system prompt instructs the LLM to return a single JSON
object of the form ``{"answer": "…", "citations": [{…}]}``. If we
forward the raw token stream to the UI verbatim, users watch JSON
syntax (``{``, ``"answer":``, quotes, escape sequences) scroll past
and the "answer" only becomes legible once the full buffer has been
post-parsed — which defeats the point of streaming.

``AnswerStreamExtractor`` parses the stream character-by-character,
emitting **only** the decoded characters of the ``answer`` string
value. Everything else (the opening brace, the key, the colon, the
quotes, the subsequent ``citations`` array) is silently swallowed.

Mode detection
--------------

The extractor scans up to ``_MODE_LOOKAHEAD`` characters of buffered
output looking for the first ``{``. When one is found:

* If the prefix before it is only whitespace and/or a Markdown code
  fence opener (e.g. ``\\`\\`\\``` or ``\\`\\`\\`json``), the prefix
  is discarded and the extractor enters JSON mode. This is the
  common case for local LLMs that wrap JSON output in a code fence
  even when the system prompt tells them not to.
* If the prefix contains arbitrary prose ("Here is the answer: {…}"),
  the extractor commits to prose mode and flushes the buffer
  verbatim — we cannot silently delete user-visible text just
  because a brace followed it.

If no ``{`` appears within ``_MODE_LOOKAHEAD`` chars, the extractor
also commits to prose mode. This means the UI still sees *some*
text even when the LLM ignores the JSON instruction entirely; the
post-stream citations parse just returns an empty list.

State machine
-------------

* ``start``  – buffering until a mode can be decided.
* ``prose``  – forwarding chunks verbatim.
* ``search`` – inside JSON, scanning for the ``"answer"`` key.
* ``value``  – inside the answer string, decoding escapes and
               streaming decoded characters.
* ``done``   – the answer string closed; subsequent chunks belong to
               the citations array and are silently dropped.

The class is **not** thread-safe. Use one instance per stream.
"""

from __future__ import annotations

import re


# JSON single-character escape table. ``\\u`` is handled separately
# because it consumes four hex digits rather than a single char.
_JSON_ESCAPES: dict[str, str] = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


# Whitespace characters recognised as JSON insignificant whitespace.
# We keep this as a module-level tuple so membership checks compile
# to a single JUMP instead of building a set each call.
_WS = (" ", "\t", "\n", "\r")


# Upper bound on the buffered "searching for answer key" tail. A
# non-conforming model that emits a huge JSON preamble must not be
# able to pin memory; we trim to the last 16 chars (enough to hold
# a partial ``"answer"`` key) once we cross the cap.
_SEARCH_BUFFER_CAP = 512


# Maximum number of chars buffered while deciding JSON vs prose mode.
# If no ``{`` appears within this window the extractor commits to
# prose mode. 256 is enough to cover the longest realistic preamble
# (code fence + language tag + a sentence of explanation) while
# still bounding memory on pathological models.
_MODE_LOOKAHEAD = 256


# A "safe to skip" JSON preamble: whitespace, optionally followed by
# a Markdown code fence opener and an optional language tag such as
# ``json``. Anything else between the stream start and the first
# ``{`` is treated as user-visible prose that must not be deleted.
#
# Examples that match (will be discarded when followed by ``{``):
#   ""                           (empty)
#   "   "                        (leading whitespace only)
#   "```"                        (bare fence)
#   "```json\n"                  (fence with language tag)
#   "  ```json  \n"              (fence with surrounding whitespace)
#
# Examples that do NOT match (trigger prose fallback):
#   "Here is the answer: "       (prose preamble)
#   "Sure! "                     (chatty preamble)
_SAFE_PREFIX_RE = re.compile(
    r"""
    ^\s*                           # leading whitespace
    (?:```\s*[A-Za-z0-9_]*\s*)?    # optional markdown code fence + lang
    \s*$                           # trailing whitespace only
    """,
    re.VERBOSE,
)


class AnswerStreamExtractor:
    """Stateful extractor that yields only the RAG answer field value."""

    __slots__ = ("_state", "_buffer")

    def __init__(self) -> None:
        self._state = "start"
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        """Process one upstream chunk and return decoded answer text.

        The returned string may be empty (e.g. while scanning JSON
        syntax), a single character (during escape processing), or a
        multi-character slice. Callers should concatenate the returns
        as-is — the extractor already handles all escape decoding.
        """
        if not chunk or self._state == "done":
            return ""
        if self._state == "prose":
            return chunk

        self._buffer = self._buffer + chunk

        if self._state == "start":
            flushed = self._decide_mode()
            if flushed is not None:
                return flushed
            if self._state == "start":
                return ""

        out: list[str] = []

        if self._state == "search":
            start = self._find_answer_key(self._buffer)
            if start is None:
                if len(self._buffer) > _SEARCH_BUFFER_CAP:
                    self._buffer = self._buffer[-16:]
                return ""
            self._buffer = self._buffer[start:]
            self._state = "value"

        if self._state == "value":
            consumed = self._drain_value(self._buffer, out)
            self._buffer = self._buffer[consumed:]

        return "".join(out)

    def finalize(self) -> str:
        """Flush any buffered content after the upstream stream ends.

        The caller invokes this once the LLM has stopped producing
        deltas. It handles three tail cases:

        * ``start`` – we never saw a non-whitespace char; treat the
          buffered whitespace as prose and flush it.
        * ``value`` – the answer string was truncated mid-string (the
          model hit its token budget). Emit whatever decoded chars
          remain, dropping any incomplete trailing escape.
        * everything else – nothing to emit.
        """
        if self._state == "start":
            self._state = "done"
            out = self._buffer
            self._buffer = ""
            return out
        if self._state == "prose":
            self._state = "done"
            return ""
        if self._state == "value":
            out: list[str] = []
            self._drain_value(self._buffer, out)
            self._buffer = ""
            self._state = "done"
            return "".join(out)
        # search / done — nothing useful left.
        self._buffer = ""
        self._state = "done"
        return ""

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _decide_mode(self) -> str | None:
        """Choose JSON vs prose mode based on the lookahead buffer.

        The decision is:

        * ``{`` found AND prefix is whitespace/code-fence only →
          JSON mode, discard prefix + ``{``.
        * ``{`` found AND prefix contains prose → prose mode, flush
          the entire buffer (cannot delete user-visible text).
        * ``{`` not found AND buffer < lookahead → wait for more.
        * ``{`` not found AND buffer >= lookahead → prose mode.

        Returns the buffered text to emit (prose fallback), or None
        when a JSON mode was selected or when more data is still
        needed to decide.
        """
        idx = self._buffer.find("{")
        if idx == -1:
            if len(self._buffer) < _MODE_LOOKAHEAD:
                return None  # keep buffering
            # Definitely not JSON; flush as prose.
            self._state = "prose"
            flushed = self._buffer
            self._buffer = ""
            return flushed

        prefix = self._buffer[:idx]
        if _SAFE_PREFIX_RE.match(prefix):
            # Safe to discard the prefix (it was whitespace or a code
            # fence opener) and enter JSON mode.
            self._buffer = self._buffer[idx + 1:]
            self._state = "search"
            return None

        # The prefix is real prose we cannot silently delete. Fall
        # back to forwarding the whole stream verbatim.
        self._state = "prose"
        flushed = self._buffer
        self._buffer = ""
        return flushed

    @staticmethod
    def _find_answer_key(buf: str) -> int | None:
        """Return the index just past ``"answer": "`` in ``buf``.

        Returns None when the full ``"answer" : "`` pattern is not
        yet present, so the caller can wait for more data. The search
        only succeeds on an exact lowercase key — the system prompt
        pins the key, so being permissive here would just mask bugs.
        """
        key_idx = buf.find('"answer"')
        if key_idx == -1:
            return None
        pos = key_idx + len('"answer"')
        n = len(buf)
        while pos < n and buf[pos] in _WS:
            pos += 1
        if pos >= n or buf[pos] != ":":
            return None
        pos += 1
        while pos < n and buf[pos] in _WS:
            pos += 1
        if pos >= n or buf[pos] != '"':
            return None
        return pos + 1

    def _drain_value(self, buf: str, out: list[str]) -> int:
        """Decode as many answer-value characters from ``buf`` as possible.

        Appends decoded characters to ``out`` and returns the number
        of chars consumed. Stops at the first unescaped closing
        quote (transitioning to ``done``) or when a partial escape
        sequence is reached (leaving the tail in ``buf`` for the next
        ``feed`` call to complete).
        """
        i = 0
        n = len(buf)
        while i < n:
            ch = buf[i]
            if ch == "\\":
                if i + 1 >= n:
                    # Partial escape at end of buffer: wait for more.
                    break
                esc = buf[i + 1]
                if esc == "u":
                    if i + 6 > n:
                        # Partial \uXXXX: wait for more hex digits.
                        break
                    try:
                        out.append(chr(int(buf[i + 2 : i + 6], 16)))
                    except ValueError:
                        # Malformed escape: emit the raw char and move
                        # on so the stream never deadlocks.
                        out.append(esc)
                    i += 6
                else:
                    out.append(_JSON_ESCAPES.get(esc, esc))
                    i += 2
            elif ch == '"':
                self._state = "done"
                i += 1
                break
            else:
                out.append(ch)
                i += 1
        return i
