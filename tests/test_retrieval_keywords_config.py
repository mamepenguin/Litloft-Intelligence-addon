"""Tests for the retrieval_keywords FeaturesConfig field.

The field follows the same three-value enum (false / manual / on_index)
as auto_tags / summaries / vision_describe, and defaults to "false"
because it sends file content to the LLM API.
"""

import sys
from unittest.mock import MagicMock

# Heavy ML deps are stubbed so importing the worker (which transitively
# pulls app.search via the rarity filter / DB module) does not need real
# torch / sentence-transformers / sqlite-vec.
for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "numpy",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.config import FeaturesConfig  # noqa: E402


class TestRetrievalKeywordsDefault:
    def test_default_is_false(self):
        # Opt-in by default — file content goes to the LLM API, so the
        # operator must explicitly enable per drive (or globally).
        assert FeaturesConfig().retrieval_keywords == "false"


class TestRetrievalKeywordsValues:
    def test_accepts_manual(self):
        cfg = FeaturesConfig(retrieval_keywords="manual")
        assert cfg.retrieval_keywords == "manual"

    def test_accepts_on_index(self):
        cfg = FeaturesConfig(retrieval_keywords="on_index")
        assert cfg.retrieval_keywords == "on_index"

    def test_accepts_false(self):
        cfg = FeaturesConfig(retrieval_keywords="false")
        assert cfg.retrieval_keywords == "false"
