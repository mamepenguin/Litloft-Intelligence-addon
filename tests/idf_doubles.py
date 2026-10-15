"""Tokenizers that stand in for Janome in the corpus-IDF child process.

They live in their own module because the child is spawned: it imports
the factory by name, so it cannot be a closure inside a test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


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
