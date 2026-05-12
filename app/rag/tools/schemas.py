"""OpenAI tools-API JSON schemas for the 4 Phase 1.B tools.

These schemas are emitted verbatim to the LLM in the ``tools=[...]``
parameter. The agentic loop dispatches by ``function.name``, so the
``TOOL_NAMES`` tuple here is the single source of truth — tests
should import from this module rather than re-declaring the names.

Schema design notes:

* ``search_files`` takes ``drive`` even though the request already
  carries it via ``X-Lit-Drive`` because the tool wrapper signs the
  hybrid search call. Forcing the LLM to name the drive every call
  makes a leakage bug (search outside the drive) impossible at the
  contract layer.
* ``get_file_chunks.mode`` enum is open-coded to ``summary`` / ``full``
  with a default of ``summary``. The system prompt teaches the LLM to
  start with ``summary`` and only ask for ``full`` when it knows the
  range it needs.
* ``range`` is two integers (``[start, end]`` inclusive) rather than
  the more conventional half-open ``[start, end)``. Aligns with
  ``chunk_index`` which is 0-based but always referenced as
  "chunk 0 through chunk 12".
"""

from __future__ import annotations

from typing import Final


_TOOL_SEARCH_FILES = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": (
            "Hybrid search across files in a drive. Returns up to top_k "
            "files (NOT chunks), one row per file, sorted by best chunk "
            "score. Use this first to discover candidates; then drill "
            "into a specific file with get_file_detail or get_file_chunks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query to search for.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max number of files to return (1-30).",
                    "minimum": 1,
                    "maximum": 30,
                    "default": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


_TOOL_GET_FILE_DETAIL = {
    "type": "function",
    "function": {
        "name": "get_file_detail",
        "description": (
            "Fetch metadata for a single file by file_id: title, mime, "
            "tags, folder_path, drive, and counts (chunk_count, "
            "relations summary by kind). Does NOT return chunk text "
            "or summaries — use get_file_chunks for the body."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "ID of the file to inspect.",
                }
            },
            "required": ["file_id"],
            "additionalProperties": False,
        },
    },
}


_TOOL_GET_FILE_CHUNKS = {
    "type": "function",
    "function": {
        "name": "get_file_chunks",
        "description": (
            "Read chunks of a file. type='transcript' for audio/video "
            "transcripts (intelligence DB), type='text' for markdown / "
            "plain text bodies (core API). Default mode='summary' "
            "returns chunk_id + location + 200-char preview for every "
            "chunk so you can scout cheaply. mode='full' returns text "
            "for a specific range (required, max 50 chunks)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "type": {
                    "type": "string",
                    "enum": ["transcript", "text"],
                },
                "mode": {
                    "type": "string",
                    "enum": ["summary", "full"],
                    "default": "summary",
                },
                "range": {
                    "type": "array",
                    "description": (
                        "[start, end] inclusive chunk indices. Required "
                        "when mode='full'."
                    ),
                    "items": {"type": "integer", "minimum": 0},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["file_id", "type"],
            "additionalProperties": False,
        },
    },
}


_TOOL_GET_RELATED_FILES = {
    "type": "function",
    "function": {
        "name": "get_related_files",
        "description": (
            "List files related to a given file via the file_relations "
            "table. Optional kind filter (e.g. 'cite', 'see_also'). "
            "Returns both directions (incoming + outgoing) flagged by "
            "'direction'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "kind": {
                    "type": "string",
                    "description": (
                        "Optional kind filter; omit to list all kinds."
                    ),
                },
            },
            "required": ["file_id"],
            "additionalProperties": False,
        },
    },
}


TOOL_SCHEMAS: Final = (
    _TOOL_SEARCH_FILES,
    _TOOL_GET_FILE_DETAIL,
    _TOOL_GET_FILE_CHUNKS,
    _TOOL_GET_RELATED_FILES,
)

TOOL_NAMES: Final = tuple(s["function"]["name"] for s in TOOL_SCHEMAS)


__all__ = ["TOOL_SCHEMAS", "TOOL_NAMES"]
