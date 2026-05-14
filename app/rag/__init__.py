"""RAG (question answering) package.

Public entry point is ``answer_question``. See ``app/rag/service.py``
for the orchestration pipeline and the sibling modules for the
individual stages (retriever, context, prompt, parser).

Note: the ``answer_question`` / ``AnswerResponse`` re-exports are
loaded lazily via ``__getattr__`` rather than imported at package
init. The eager form created an import cycle with
``app.dependencies`` once ``app.workers.retrieval_keywords`` (which
imports ``app.rag.keyword_filter``) joined the worker registry. All
actual call sites use submodule paths (``from app.rag.service
import ...``) so the lazy attribute lookup is only a safety net for
the documented public surface.
"""

from typing import TYPE_CHECKING

__all__ = ["AnswerResponse", "answer_question"]

if TYPE_CHECKING:
    from app.rag.service import AnswerResponse, answer_question  # noqa: F401


def __getattr__(name: str):  # PEP 562 module-level __getattr__
    if name in __all__:
        from app.rag import service as _service
        return getattr(_service, name)
    raise AttributeError(f"module 'app.rag' has no attribute {name!r}")
