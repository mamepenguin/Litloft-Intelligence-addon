"""Capture baseline prompt outputs for byte-identical regression testing.

Run inside the intelligence Docker test image. Writes golden text files
to tests/golden_prompts/ that the test_prompt_loader.py snapshot tests
read back as expected output.

Usage (inside Docker):
    python -m tests.golden_prompts._baseline

Designed to run twice safely:
- Once before the Jinja migration (capture current f-string output).
- Once after, to confirm regenerated golden files match (sanity check).

Each golden file ends without a trailing newline beyond what the prompt
itself produces — i.e. the file content IS the prompt bytes verbatim.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    # short_long_system: ja, en, auto
    for lang in ("ja", "en", "auto"):
        with _patch_settings(lang):
            out = summaries._build_system_prompt()
        _write(f"summaries_short_long_system_{lang}.txt", out)

    # detailed_system: ja, en, auto
    for lang in ("ja", "en", "auto"):
        with _patch_settings(lang):
            out = summaries._build_detailed_system_prompt()
        _write(f"summaries_detailed_system_{lang}.txt", out)

    # short_long_user variants
    base_indexed = {
        "filename": "test.mp4",
        "title": "",
        "description": "",
    }
    out = summaries._build_user_prompt(
        base_indexed, "video", "ここに本文。", was_truncated=False
    )
    _write("summaries_short_long_user_minimal.txt", out)

    full_indexed = {
        "filename": "test.mp4",
        "title": "別のタイトル",
        "description": "テスト動画の説明文。",
    }
    out = summaries._build_user_prompt(
        full_indexed, "video", "本文ここ。", was_truncated=True
    )
    _write("summaries_short_long_user_full.txt", out)

    # title equal to filename → suppressed
    same_indexed = {
        "filename": "same.mp4",
        "title": "same.mp4",
        "description": "",
    }
    out = summaries._build_user_prompt(
        same_indexed, "video", "ctx", was_truncated=False
    )
    _write("summaries_short_long_user_title_eq_filename.txt", out)

    # detailed_user
    out = summaries._build_detailed_user_prompt(
        base_indexed, "video", "本文。", was_truncated=False
    )
    _write("summaries_detailed_user_minimal.txt", out)

    out = summaries._build_detailed_user_prompt(
        full_indexed, "video", "本文。", was_truncated=True
    )
    _write("summaries_detailed_user_full.txt", out)


def capture_auto_tags() -> None:
    from app.workers import auto_tags

    for lang in ("ja", "en", "auto"):
        with _patch_settings(lang):
            out = auto_tags._build_system_prompt()
        _write(f"auto_tags_system_{lang}.txt", out)

    minimal = {
        "filename": "f.mp4",
        "title": "",
        "description": "",
        "tags_text": "",
    }
    out = auto_tags._build_user_prompt(minimal, "video", "", [])
    _write("auto_tags_user_minimal.txt", out)

    rich = {
        "filename": "f.mp4",
        "title": "別タイトル",
        "description": "説明文。",
        "tags_text": "tag1, tag2",
    }
    out = auto_tags._build_user_prompt(
        rich, "video", "本文ここ。", ["existing1", "existing2"]
    )
    _write("auto_tags_user_rich.txt", out)

    cands = auto_tags.TagCandidates(
        clip=["clip1", "clip2"],
        tfidf=["tfidf1"],
        knn=["knn1"],
    )
    out = auto_tags._build_user_prompt(
        rich, "video", "本文。", ["existing"], cands
    )
    _write("auto_tags_user_with_candidates.txt", out)


def capture_refine() -> None:
    from app.workers import refine

    out = refine._build_system_prompt()
    _write("refine_system.txt", out)

    class _Chunk:
        def __init__(self, cid: int, text: str) -> None:
            self.id = cid
            self.text = text

    out = refine._build_user_prompt(
        [_Chunk(1, "hello"), _Chunk(2, "world")]
    )
    _write("refine_user.txt", out)


def capture_vision() -> None:
    from app import llm

    for lang in ("ja", "en", "auto", ""):
        out = llm._build_vision_system_prompt(lang)
        label = lang if lang else "empty"
        _write(f"vision_system_{label}.txt", out)


def capture_rag() -> None:
    from app.rag import prompt as rag_prompt

    for lang in ("ja", "en", "auto"):
        out = rag_prompt.build_system_prompt(lang)
        _write(f"rag_answer_system_{lang}.txt", out)

    # build_user_prompt
    from app.rag.context import ContextSnippet, FileContext

    out = rag_prompt.build_user_prompt("クエリだ", [])
    _write("rag_answer_user_empty.txt", out)

    ctx = FileContext(
        file_id="abc123",
        filename="video.mp4",
        drive="movies",
        file_type="video",
        title="title",
        description="description",
        snippets=(
            ContextSnippet(source="transcript", location="0:45", text="snippet text"),
            ContextSnippet(source="metadata", location=None, text="meta snippet"),
        ),
        total_chars=42,
    )
    out = rag_prompt.build_user_prompt("クエリ", [ctx])
    _write("rag_answer_user_one_ctx.txt", out)

    # query_decomposer
    from app.rag import query_decomposer
    _write("rag_query_decomposer_system.txt", query_decomposer._SYSTEM_PROMPT)

    # query_transform
    from app.rag import query_transform
    _write("rag_query_transform_system.txt", query_transform._SYSTEM_PROMPT)

    # clue_generator (with clue_count=4) — rendered via prompt_loader.
    from app.prompt_loader import render
    out = render("rag/clue_generator_system.jinja2", clue_count=4)
    _write("rag_clue_generator_system_4.txt", out)

    # category_expander (with max_terms=5) — rendered via prompt_loader.
    out = render("rag/category_expander_system.jinja2", max_terms=5)
    _write("rag_category_expander_system_5.txt", out)


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    capture_summaries()
    capture_auto_tags()
    capture_refine()
    capture_vision()
    capture_rag()
    print("done")


if __name__ == "__main__":
    main()
