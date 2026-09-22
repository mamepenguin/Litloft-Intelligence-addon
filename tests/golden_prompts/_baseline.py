"""Regenerate the summaries system-prompt goldens.

Run inside the intelligence Docker test image:

    python -m tests.golden_prompts._baseline

Only the two prompts that include ``summaries/_common_rules.jinja2`` are
captured whole. Every other builder is checked by naming the decision it
makes, in ``test_prompt_loader.py``.

A golden file is the prompt bytes verbatim, with no trailing newline of
its own.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Make the addon's app package importable when run from /app.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Stub heavy ML deps the same way tests/conftest.py does so module
# imports below succeed without GPUs, torch wheels, etc.
_ml_stubs = (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
)
for _mod in _ml_stubs:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
if "numpy" not in sys.modules:
    try:
        import numpy  # noqa: F401
    except ImportError:
        _numpy_stub = MagicMock()
        _numpy_stub.bool_ = bool
        sys.modules["numpy"] = _numpy_stub

GOLDEN_DIR = Path(__file__).resolve().parent


def _write(name: str, content: str) -> None:
    out = GOLDEN_DIR / name
    out.write_bytes(content.encode("utf-8"))
    print(f"wrote {name} ({len(content.encode('utf-8'))} bytes)")


class _OutputLangContext:
    """Mutate the shared settings.llm.output_language across all callers.

    summaries / auto_tags / etc. capture ``settings`` at import time
    (``from app.config import settings``), so a module-level
    ``patch.object`` would only reach ``app.config``. The simpler safe
    move is to flip the attribute on the singleton itself for the
    duration of the with-block.
    """

    def __init__(self, lang: str) -> None:
        self._lang = lang
        self._previous: str | None = None

    def __enter__(self):
        from app.config import settings
        self._previous = settings.llm.output_language
        object.__setattr__(settings.llm, "output_language", self._lang)
        return self

    def __exit__(self, exc_type, exc, tb):
        from app.config import settings
        object.__setattr__(settings.llm, "output_language", self._previous)
        return False


def _patch_settings(output_language: str):
    return _OutputLangContext(output_language)


def capture_summaries() -> None:
    from app.workers import summaries

    for lang in ("ja", "en", "auto"):
        with _patch_settings(lang):
            out = summaries._build_system_prompt()
        _write(f"summaries_short_long_system_{lang}.txt", out)

    for lang in ("ja", "en", "auto"):
        with _patch_settings(lang):
            out = summaries._build_detailed_system_prompt()
        _write(f"summaries_detailed_system_{lang}.txt", out)


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    capture_summaries()
    print("done")


if __name__ == "__main__":
    main()
