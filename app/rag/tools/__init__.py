"""Agentic Ask tool wrappers (Phase 1.B).

Thin Python adapters around existing retrieval / internal-API surface so
the agentic loop (Phase 1.C) can drive them as OpenAI tools-API calls.

Public surface (re-exported here for convenience):

* ``ToolContext`` — per-answer state carrier (drive, viewer, allowed
  citation file_ids, cumulative token usage).
* ``TOOL_SCHEMAS`` — JSON schemas to advertise to the LLM.
* ``search_files``, ``get_file_detail``, ``get_file_chunks``,
  ``get_related_files`` — async tool implementations.

Tier 3 (auto_tags suggested, detailed_summary) is intentionally absent
from every return shape: see hako ``SKiYgE6GtttlQW7fEwAY6``.
"""

from app.rag.tools.context import ToolContext, ToolResultEnvelope
from app.rag.tools.schemas import TOOL_NAMES, TOOL_SCHEMAS

# Tool functions are NOT re-exported here on purpose: each function
# shares its name with its module (``get_file_detail`` etc.), so a
# package-level ``from .get_file_detail import get_file_detail``
# would shadow the module attribute. That shadowing breaks
# ``monkeypatch.setattr("app.rag.tools.get_file_detail.X", ...)`` in
# tests, which resolves the string path through the package namespace.
# Callers import the functions from their modules directly:
#   ``from app.rag.tools.search_files import search_files``

__all__ = [
    "ToolContext",
    "ToolResultEnvelope",
    "TOOL_NAMES",
    "TOOL_SCHEMAS",
]
