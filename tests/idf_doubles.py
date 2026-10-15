"""Tokenizers that stand in for Janome in the corpus-IDF child process.

They live in their own module because the child is spawned: it imports
the factory by name, so it cannot be a closure inside a test.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Token:
    surface: str
    part_of_speech: str = "名詞,一般,*,*"


class WordTokenizer:
    """Every whitespace-separated word is a noun."""

    def tokenize(self, text: str):
        return [_Token(word) for word in text.split()]


class PidTokenizer(WordTokenizer):
    """Also emits ``pid<n>`` for the process that did the tokenizing."""

    def tokenize(self, text: str):
        return [*super().tokenize(text), _Token(f"pid{os.getpid()}")]


class MarkerTokenizer(WordTokenizer):
    """Misbehaves on text that carries a marker word.

    ``CRASHWORD`` kills the process, ``BOOMWORD`` raises, ``SLOWWORD``
    sleeps for two minutes, and ``GATEWORD`` creates ``entered`` in
    ``$IDF_TEST_GATE_DIR`` and waits there until ``release`` exists.
    """

    def tokenize(self, text: str):
        if "CRASHWORD" in text:
            os._exit(3)
        if "BOOMWORD" in text:
            raise RuntimeError("tokenizer failed on BOOMWORD")
        if "SLOWWORD" in text:
            time.sleep(120)
        if "GATEWORD" in text:
            gate = Path(os.environ["IDF_TEST_GATE_DIR"])
            (gate / "entered").touch()
            deadline = time.monotonic() + 60
            while not (gate / "release").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
        return super().tokenize(text)


class ExplodingTokenizer:
    """Any use of it is a failure."""

    def __init__(self):
        raise AssertionError("the tokenizer was constructed")
