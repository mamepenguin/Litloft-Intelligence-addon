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

import json
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


# ---------------------------------------------------------------------------
# AnswerSanitizer
# ---------------------------------------------------------------------------


class AnswerSanitizer:
    """Strip inline citation markers like ``[file_id: ...]`` from the stream.

    Weak local LLMs sometimes ignore the system prompt's instruction
    not to inline source references in the answer body and emit
    fragments such as ``[file_id: abc, location: 0:45]`` next to the
    prose. Because we already render a dedicated citations panel from
    the JSON ``citations`` array, those fragments are pure visual
    noise and we strip them here.

    The sanitizer is fed the decoded answer-text chunks produced by
    ``AnswerStreamExtractor``. It holds back characters starting at
    ``[`` until it can decide whether the bracket opens a target
    marker. Brackets whose head does NOT begin with one of the
    suppression keywords are flushed back unchanged so legitimate
    uses such as Markdown links ``[label](url)`` are unaffected.

    Decision rules (head = chars between ``[`` and the next non-name
    char, case-folded with leading whitespace stripped):

    * head matches ``file_id`` or ``location`` → suppress entire
      bracket including the closing ``]``
    * head can no longer become any keyword prefix → flush as-is
    * otherwise → keep holding until ``]`` or until ``_MAX_HOLD``

    The class is **not** thread-safe. Use one instance per stream.
    """

    _KEYWORDS = ("file_id", "location")
    _MAX_HOLD = 512

    __slots__ = ("_held",)

    def __init__(self) -> None:
        self._held = ""

    def feed(self, text: str) -> str:
        if not text:
            return ""
        out: list[str] = []
        for ch in text:
            if self._held:
                self._held = self._held + ch
                if ch == "]":
                    if self._matches_keyword():
                        # suppress the entire bracket
                        pass
                    else:
                        out.append(self._held)
                    self._held = ""
                elif len(self._held) > self._MAX_HOLD:
                    out.append(self._held)
                    self._held = ""
                elif not self._could_match_keyword():
                    out.append(self._held)
                    self._held = ""
            elif ch == "[":
                self._held = "["
            else:
                out.append(ch)
        return "".join(out)

    def finalize(self) -> str:
        """Flush any held content when the upstream stream ends.

        If the tail looks like an unfinished suppression marker (e.g.
        ``[file_id: abc`` with no closing bracket because the model hit
        its token budget), drop it — emitting it half-finished would be
        worse than emitting nothing.
        """
        flushed = self._held
        self._held = ""
        if flushed.startswith("[") and any(
            flushed[1:].lstrip().lower().startswith(kw)
            for kw in self._KEYWORDS
        ):
            return ""
        return flushed

    def _head(self) -> str:
        return self._held[1:].lstrip().lower()

    def _matches_keyword(self) -> bool:
        head = self._head()
        return any(head.startswith(kw) for kw in self._KEYWORDS)

    def _could_match_keyword(self) -> bool:
        head = self._head()
        if not head:
            return True
        return any(
            kw.startswith(head) or head.startswith(kw)
            for kw in self._KEYWORDS
        )


# ---------------------------------------------------------------------------
# CitationStreamExtractor
# ---------------------------------------------------------------------------


# Upper bound on the buffered "searching for citations array" tail. A
# non-conforming model that emits a huge JSON preamble must not be
# able to pin memory; we trim to the last 32 chars (enough to hold
# a partial ``"citations"`` key) once we cross the cap.
_CITATION_SEARCH_CAP = 1024

# Hard upper bound on a single buffered citation object. Stops a
# broken model from growing the buffer without bound if it never
# closes the object; we bail out by skipping to the next ``,`` or
# ``]`` rather than crashing.
_CITATION_OBJECT_CAP = 16384


class CitationStreamExtractor:
    """Stateful extractor that yields each citation object as a dict.

    Designed to be fed the SAME raw LLM chunks as
    ``AnswerStreamExtractor`` — the two classes do not need to
    coordinate because this one runs its own small state machine:

    1. ``waiting_for_answer_close``: swallow everything until the
       answer string closes (handling ``\\"`` escape and the JSON
       object's opening brace on the way). We track just enough
       string-state to know when the unescaped closing quote of the
       answer value arrives.
    2. ``searching_key``: scan for ``"citations"`` + ``:`` + ``[``.
       We also accept a preamble with whitespace / code-fence wrappers
       so the extractor behaves the same way as
       ``AnswerStreamExtractor``.
    3. ``in_array``: skip whitespace and commas until an opening
       ``{`` is found; transition to ``in_object``.
    4. ``in_object``: buffer the object body, tracking nested brace
       depth and string state so ``{``/``}`` inside a quoted value do
       not confuse boundary detection. When depth returns to 0 we
       call ``json.loads`` on the accumulated text and — on success —
       return the resulting dict to the caller via ``feed``'s return
       list.
    5. ``done``: the closing ``]`` was seen (or the extractor
       committed to "no citations to extract" due to prose / null /
       missing key); further chunks are no-ops.

    The class never raises on malformed input. Garbage objects are
    skipped silently (logged at debug in the service layer via the
    hallucination filter). Partial final objects at stream end are
    dropped by ``finalize``.
    """

    __slots__ = (
        "_state",
        "_buffer",
        "_escape_next",
        "_in_string",
        "_depth",
        "_obj_buffer",
    )

    def __init__(self) -> None:
        self._state = "waiting_for_answer_close"
        # Shared search buffer (states 1+2). Rolled over when it
        # grows past _CITATION_SEARCH_CAP so memory stays bounded.
        self._buffer = ""
        # String-aware flags for state 1 (skipping the answer value)
        # and state 4 (object brace tracking that must not be fooled
        # by ``{``/``}`` inside strings).
        self._escape_next = False
        self._in_string = False
        # Brace depth for the object currently being accumulated.
        self._depth = 0
        # Accumulator for the in-progress citation object text.
        self._obj_buffer = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, chunk: str) -> list[dict]:
        """Process one upstream chunk and return completed citations.

        The returned list may be empty (most of the time), contain one
        dict (when the chunk closes a single object), or contain
        several (e.g. when a single chunk holds the tail of the JSON
        payload with multiple citations).

        Implementation note: all states read from and write to
        ``self._buffer``. The incoming chunk is appended once up
        front, then each state drains as much as it can before
        transitioning. This means a single large chunk holding the
        whole JSON is processed in one call, while chunks that split
        anywhere (mid-key, mid-object, mid-escape) naturally resume
        on the next feed — the buffer is the only cross-call state.
        """
        if not chunk or self._state == "done":
            return []
        self._buffer = self._buffer + chunk
        results: list[dict] = []

        while self._state != "done":
            before_state = self._state
            before_len = len(self._buffer)
            if self._state == "waiting_for_answer_close":
                self._drain_answer_value()
            elif self._state == "searching_key":
                self._try_enter_array()
            elif self._state == "in_array":
                self._drain_array_until_object()
            elif self._state == "in_object":
                self._drain_object(results)
            else:
                break
            # Break when the handler made no progress AND did not
            # transition. Otherwise loop: a transition may unlock
            # more buffer consumption in the next state.
            if (
                self._state == before_state
                and len(self._buffer) == before_len
            ):
                break
        return results

    def finalize(self) -> list[dict]:
        """Flush state; return any completed citations still buffered.

        A partially-written final object is dropped silently — we
        prefer "show nothing" over "show a malformed card".
        """
        # No extra work needed today; all completions are emitted as
        # soon as their closing ``}`` arrives. The method exists for
        # symmetry with AnswerStreamExtractor and so a future
        # tolerant-parse mode can graft on here without changing the
        # caller.
        self._state = "done"
        self._buffer = ""
        self._obj_buffer = ""
        return []

    # ------------------------------------------------------------------
    # State 1: waiting_for_answer_close
    # ------------------------------------------------------------------

    def _drain_answer_value(self) -> None:
        """Advance past the answer string's closing quote if possible.

        Consumes from ``self._buffer``. On success, transitions to
        ``searching_key`` and leaves any post-quote tail in the
        buffer. On stream exhaustion without finding the close, we
        either wait (keeping the buffer trimmed) or commit to
        ``done`` when the output is clearly prose.
        """
        if not self._in_string:
            idx = AnswerStreamExtractor._find_answer_key(self._buffer)
            if idx is None:
                if len(self._buffer) > _CITATION_SEARCH_CAP:
                    # Keep a small tail so a late-arriving key is
                    # still discoverable when a chunk boundary splits
                    # the literal.
                    self._buffer = self._buffer[-64:]
                # If we've clearly crossed the mode-decision
                # threshold without finding ``{``, give up entirely
                # — this is prose, no citations ever.
                if (
                    "{" not in self._buffer
                    and len(self._buffer) >= _MODE_LOOKAHEAD
                ):
                    self._state = "done"
                    self._buffer = ""
                return
            # idx points one past the opening ``"`` of the answer
            # value. Discard everything up to it and commit to
            # "inside the string".
            self._buffer = self._buffer[idx:]
            self._in_string = True
            self._escape_next = False

        # Scan self._buffer for the unescaped closing quote.
        buf = self._buffer
        i = 0
        n = len(buf)
        while i < n:
            ch = buf[i]
            if self._escape_next:
                self._escape_next = False
                i += 1
                continue
            if ch == "\\":
                self._escape_next = True
                i += 1
                continue
            if ch == '"':
                self._in_string = False
                self._state = "searching_key"
                self._buffer = buf[i + 1 :]
                return
            i += 1
        # Buffer exhausted inside the value. Drop everything — it's
        # answer body we don't need to retain. ``_escape_next`` is
        # preserved on the instance so the next chunk's first char
        # is interpreted as the escape payload.
        self._buffer = ""

    # ------------------------------------------------------------------
    # State 2: searching_key
    # ------------------------------------------------------------------

    def _try_enter_array(self) -> None:
        """Look for ``"citations": [`` in ``self._buffer``.

        Commits the extractor to one of three outcomes:

        * ``[`` found → enter ``in_array`` and drop the consumed
          prefix from the buffer.
        * ``"citations"`` found followed by a non-array value
          (``null``, ``false``, ``123``) → enter ``done`` (no
          citations available).
        * Not yet found → stay in ``searching_key`` and wait for
          more data. The buffer is trimmed if it grows past the cap.
        """
        key_idx = self._buffer.find('"citations"')
        if key_idx == -1:
            if len(self._buffer) > _CITATION_SEARCH_CAP:
                # Keep the tail so a late-arriving key is still
                # discoverable when the boundary splits the literal.
                self._buffer = self._buffer[-32:]
            return

        pos = key_idx + len('"citations"')
        n = len(self._buffer)
        while pos < n and self._buffer[pos] in _WS:
            pos += 1
        if pos >= n or self._buffer[pos] != ":":
            return  # wait for ``:``
        pos += 1
        while pos < n and self._buffer[pos] in _WS:
            pos += 1
        if pos >= n:
            return  # wait for value
        ch = self._buffer[pos]
        if ch == "[":
            # Happy path: entering the array.
            self._buffer = self._buffer[pos + 1 :]
            self._state = "in_array"
            return
        # Anything else after the colon (null, false, 0, etc.) means
        # there are no citations to extract. We commit to done and
        # stop buffering.
        self._state = "done"
        self._buffer = ""

    # ------------------------------------------------------------------
    # State 3: in_array — scan for next ``{`` or ``]``
    # ------------------------------------------------------------------

    def _drain_array_until_object(self) -> None:
        """Skip whitespace and commas in ``self._buffer`` until ``{`` or ``]``.

        On ``{`` we switch to ``in_object`` with a depth of 1 and
        seed ``_obj_buffer`` with the opening brace. On ``]`` the
        array is closed and we move to ``done``.
        """
        buf = self._buffer
        i = 0
        n = len(buf)
        while i < n:
            ch = buf[i]
            if ch == "{":
                self._obj_buffer = "{"
                self._depth = 1
                self._in_string = False
                self._escape_next = False
                self._state = "in_object"
                self._buffer = buf[i + 1 :]
                return
            if ch == "]":
                self._state = "done"
                self._buffer = ""
                return
            # Whitespace, commas, anything else — skip defensively.
            i += 1
        # All whitespace/commas — wait for more.
        self._buffer = ""

    # ------------------------------------------------------------------
    # State 4: in_object — accumulate until matching ``}``
    # ------------------------------------------------------------------

    def _drain_object(self, results: list[dict]) -> None:
        """Accumulate ``self._buffer`` bytes into the object under parse.

        Depth tracking ignores braces that appear inside string
        literals. On depth returning to 0 we try ``json.loads``; a
        success appends the dict to ``results`` and we transition
        back to ``in_array`` with the post-close tail returned to
        ``self._buffer``. A parse failure silently discards the
        object.
        """
        buf = self._buffer
        i = 0
        n = len(buf)
        while i < n:
            ch = buf[i]
            self._obj_buffer = self._obj_buffer + ch
            i += 1

            # Safety valve: fires on every char, including those inside
            # string values. A model that opens a 1-megabyte quote and
            # never closes it must not grow the buffer indefinitely.
            if len(self._obj_buffer) > _CITATION_OBJECT_CAP:
                self._obj_buffer = ""
                self._depth = 0
                self._in_string = False
                self._escape_next = False
                self._state = "in_array"
                self._buffer = buf[i:]
                return

            if self._escape_next:
                self._escape_next = False
                continue
            if self._in_string:
                if ch == "\\":
                    self._escape_next = True
                elif ch == '"':
                    self._in_string = False
                continue
            # Outside strings.
            if ch == '"':
                self._in_string = True
                continue
            if ch == "{":
                self._depth += 1
                continue
            if ch == "}":
                self._depth -= 1
                if self._depth == 0:
                    self._flush_object(results)
                    self._state = "in_array"
                    self._buffer = buf[i:]
                    return
                continue
        # Buffer exhausted inside the object — wait for more.
        self._buffer = ""

    def _flush_object(self, results: list[dict]) -> None:
        """Parse the accumulated buffer and push to ``results`` on success."""
        buf = self._obj_buffer
        self._obj_buffer = ""
        try:
            obj = json.loads(buf)
        except (json.JSONDecodeError, ValueError):
            return
        if isinstance(obj, dict):
            results.append(obj)
