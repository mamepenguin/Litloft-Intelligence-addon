"""Runtime config helpers for the eval harness.

Phase A exposes just enough to let Phase C wire up the snapshot swap:
the runner needs to know which ``search.db`` to point intelligence at
when executing cases. The standard mechanism is the
``INTELLIGENCE_SEARCH_DB_PATH`` env var, set by the caller (compose
override or shell) before ``python -m app.evals`` is invoked.

Keeping this module small and side-effect-free (no imports from
``app.config``) avoids triggering production settings initialization
when the harness is merely inspected (e.g. ``--help``).
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_SEARCH_DB_PATH = "INTELLIGENCE_SEARCH_DB_PATH"


def get_search_db_override() -> Path | None:
    """Return the snapshot search.db path if the env var is set.

    Phase C will consume this to override ``settings.search_db_path``
    during a runner invocation. Returns ``None`` when unset so callers
    can fall through to the standard settings-resolved path.
    """
    raw = os.environ.get(ENV_SEARCH_DB_PATH, "").strip()
    if not raw:
        return None
    return Path(raw)
