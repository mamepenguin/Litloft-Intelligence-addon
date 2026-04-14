"""Dev-time eval harness for the Ask (RAG) pipeline.

This package is a developer tool, not part of the production request path.
Importing from ``app.*`` (e.g. ``app.rag``, ``app.search``, ``app.llm``)
is expected; the reverse direction is forbidden. In particular, the
production startup path (``app.main``, ``app.routers.*``) MUST NOT import
``app.evals`` — keeping the dependency one-way lets the eval harness pull
in heavier test fixtures and CLI deps without bloating the runtime.

Entry point: ``python -m app.evals`` (see ``__main__.py``).

Spec: docs/superpowers/specs/2026-04-14-intelligence-ask-eval-harness.md
"""
