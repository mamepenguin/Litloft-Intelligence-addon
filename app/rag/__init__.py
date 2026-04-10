"""RAG (question answering) package.

Public entry point is ``answer_question``. See ``app/rag/service.py``
for the orchestration pipeline and the sibling modules for the
individual stages (retriever, context, prompt, parser).
"""

from app.rag.service import AnswerResponse, answer_question

__all__ = ["AnswerResponse", "answer_question"]
