"""DF-based rarity filter for LLM-generated retrieval clues.

The Stage 2 clue generator emits multi-word search queries by prompting a
small LLM with the user question plus the shortlist's summaries. The LLM
occasionally leaks corpus-common tokens (particles, generic nouns like
"内容" / "情報", or even drive-wide stop words it inferred from the
summary block) into the clue. These poison the downstream FTS AND-query
because every chunk matches them, and the genuine rare term carries
nothing extra.

SIRA (arxiv:2605.06647) drops query-expansion tokens whose corpus
document frequency exceeds a threshold. This module implements the same
idea against Litloft's existing word-tokenized FTS tables:

* ``fts_transcripts_word`` — whisper transcript chunks
* ``fts_text_content_word`` — PDF / office / text document chunks

We expose the per-table ``doc`` counts via the standard ``fts5vocab``
auxiliary virtual tables (created idempotently in
``database._create_vec_tables``). A token's DF is the sum of its
``doc`` counts across both vocab tables; the corpus size is the sum of
the two FTS tables' total row counts. A token is rejected when
``df / corpus_size > threshold_ratio`` (default 0.5).

Failure modes are fail-open: if the DB isn't initialised, the vocab
table doesn't exist yet, or the corpus is empty, the clue is returned
unchanged. The goal is recall — never silently drop legitimate clues
because the auxiliary infrastructure is missing.
"""

from __future__ import annotations

import logging
import unicodedata
from functools import lru_cache

from sqlalchemy import text as sql_text

from app.database import get_search_db_read

logger = logging.getLogger(__name__)

# Tokens shorter than this skip the DF lookup entirely. Single-character
# tokens are almost always particles or noise the LLM should not have
# emitted; the trigram FTS handles short queries on its own.
_MIN_TOKEN_LEN = 2

# Default rarity threshold: a token appearing in more than half of all
# indexed chunks is treated as a stop word. Conservative on purpose —
# SIRA itself uses tighter cuts but Litloft drives are small (hundreds
# to low-thousands of chunks) so an aggressive cut would over-prune.
DEFAULT_THRESHOLD_RATIO = 0.5


#: Above this code point a combining mark is not a Latin diacritic.
#: ``remove_diacritics=2`` strips marks from Latin script only, so
#: anything beyond the Latin blocks is left exactly as the tokenizer
#: stored it.
_LATIN_MAX = 0x0370


def _normalize_token(token: str) -> str:
    """Approximate the ``unicode61 remove_diacritics 2`` tokenizer.

    The vocab tables store terms after the host FTS tokenizer has
    normalised them. ``unicode61`` lowercases, applies a set of
    compatibility mappings (``µ``→``μ``, ``ϕ``→``φ``), and strips
    diacritics **from Latin script**. Verified against SQLite: of ten
    probe words this agrees on nine.

    The tenth is a ligature. ``ﬁ`` is left alone by the tokenizer and
    decomposed to ``fi`` here, because NFKD carries ligature splitting
    along with the mappings we do want. Matching that exactly would mean
    enumerating the tokenizer's own table; the lookup fails open, so the
    cost is a term kept that might have been dropped.

    The decomposition has to be per character, and guarded. Applying it
    to a whole token strips Japanese voiced sound marks as if they were
    diacritics — ``ポケモン`` became ``ホケモン`` and ``データ`` became
    ``テータ``, neither of which is in the vocab, so every voiced
    katakana term reported a document frequency of 0 and read as
    maximally rare. SQLite keeps those marks; so do we.

    Text that reaches the index *already* decomposed is a different
    matter and not one normalisation can repair: ``unicode61`` treats a
    standalone U+309A as a separator, so the vocab ends up holding
    ``ホ`` and ``ケモン`` as two terms and no spelling of the query
    matches. That lookup returns 0 and the caller fails open, which is
    this module's stance throughout.
    """
    if not token:
        return ""

    # Decompose first, so compatibility mappings still happen: SQLite
    # stores µ as μ and ϕ as φ, and skipping the mapping to protect kana
    # traded one mismatch for another.
    out: list[str] = []
    base = ""
    for ch in unicodedata.normalize("NFKD", token):
        if unicodedata.combining(ch):
            # Drop the mark only where it is a Latin diacritic. On a kana
            # base it is a voiced sound mark and part of the word —
            # dropping it turned ポケモン into ホケモン, which is in no
            # vocab table, so every voiced katakana term reported a
            # frequency of 0 and read as maximally rare.
            if base and ord(base) < _LATIN_MAX:
                continue
            out.append(ch)
        else:
            base = ch
            out.append(ch)

    # Recompose, because the vocab holds ポ as one character.
    return unicodedata.normalize("NFC", "".join(out)).lower().strip()


@lru_cache(maxsize=1)
def _corpus_size() -> int:
    """Total indexed chunks across both word-tokenized FTS tables.

    Cached for the process lifetime once a non-zero value is observed.
    A zero result is *not* effectively cached — the caller in
    ``filter_clue_by_rarity`` invalidates the cache on a 0 return so
    the next query after the worker indexes anything will re-read the
    real count. This avoids a permanent no-op when Ask handlers boot
    before the indexer.

    Returns 0 on any failure (uninitialised DB, missing tables) so the
    caller's fail-open + self-heal path triggers.
    """
    try:
        with get_search_db_read() as session:
            total = 0
            for table in ("fts_transcripts_word", "fts_text_content_word"):
                try:
                    row = session.execute(
                        sql_text(f"SELECT count(*) FROM {table}")
                    ).fetchone()
                except Exception:
                    # Table may not exist on a fresh DB before the first
                    # index run — treat as 0 contribution.
                    continue
                if row and row[0]:
                    total += int(row[0])
            return total
    except Exception as exc:
        logger.debug("Rarity corpus size lookup failed: %s", exc)
        return 0


@lru_cache(maxsize=4096)
def _token_df(normalized: str) -> int:
    """Document frequency of ``normalized`` across both vocab tables.

    Per-call DB hit, then LRU-cached for the process. The cache is
    keyed by the normalised form so case / diacritic variants share a
    slot. A token absent from both vocab tables returns 0 — the
    rarity check then treats it as rare and keeps it.

    Returns 0 on any DB failure (fail-open path).
    """
    if not normalized:
        return 0
    try:
        with get_search_db_read() as session:
            total = 0
            for vocab in (
                "fts_transcripts_word_vocab",
                "fts_text_content_word_vocab",
            ):
                try:
                    row = session.execute(
                        sql_text(
                            f"SELECT doc FROM {vocab} WHERE term = :t"
                        ),
                        {"t": normalized},
                    ).fetchone()
                except Exception:
                    # Vocab table may not exist on legacy DBs that
                    # haven't been migrated yet — treat as DF 0.
                    continue
                if row and row[0]:
                    total += int(row[0])
            return total
    except Exception as exc:
        logger.debug("Rarity DF lookup for %r failed: %s", normalized, exc)
        return 0


def reset_cache() -> None:
    """Drop the corpus-size and per-token LRU caches.

    Tests use this to assert behaviour across fixture rebuilds. Workers
    do not call this; cache staleness within a process lifetime is
    acceptable for a soft filter.
    """
    _corpus_size.cache_clear()
    _token_df.cache_clear()


def filter_clue_by_rarity(
    clue: str,
    *,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
) -> str:
    """Drop corpus-common tokens from ``clue``; return the rejoined string.

    Splits ``clue`` on whitespace, normalises each token to match the
    word-FTS tokenizer's storage form, and looks up the per-token DF
    across the transcript + text-content vocab tables. Tokens whose
    ``df / corpus_size`` exceeds ``threshold_ratio`` are dropped; the
    rest are rejoined in original order with single spaces.

    ``threshold_ratio`` is bounded to ``(0.0, 1.0]``. ``0.5`` means
    "drop tokens appearing in more than half of all indexed chunks" —
    aggressive enough to catch particles and weak generic nouns but
    forgiving enough to keep domain vocabulary in small drives.

    Fail-open behaviour:

    * Empty or whitespace-only input → empty string (caller drops the
      clue, matching the existing fallback path).
    * Corpus size unknown (uninitialised DB / missing tables) →
      returned unchanged.
    * Token shorter than ``_MIN_TOKEN_LEN`` after normalisation →
      kept without lookup.
    * DF lookup raises → token kept (already covered by ``_token_df``'s
      try/except).

    The post-filter empty case is the same as the input-empty case so
    ``clue_generator.generate_clues`` can branch on truthiness without
    a special path.
    """
    if not clue or not clue.strip():
        return ""

    corpus = _corpus_size()
    if corpus <= 0:
        # Fail-open: never silently drop clues when the rarity
        # infrastructure isn't ready yet. Also clear the lru_cache slot
        # so the next call after the indexer warms up re-reads the
        # actual count — without this, a 0 from a pre-index startup
        # would lock the filter to no-op until process restart.
        _corpus_size.cache_clear()
        return clue

    ratio_cap = max(0.0, min(1.0, threshold_ratio))
    if ratio_cap >= 1.0:
        # Threshold disables filtering entirely. Honour it as a
        # documented escape hatch (e.g. tests / debug runs).
        return clue

    df_cap = corpus * ratio_cap

    kept: list[str] = []
    for token in clue.split():
        if not token:
            continue
        normalized = _normalize_token(token)
        if len(normalized) < _MIN_TOKEN_LEN:
            kept.append(token)
            continue
        df = _token_df(normalized)
        if df > df_cap:
            logger.debug(
                "Rarity filter dropped %r (df=%d, cap=%.1f, corpus=%d)",
                token, df, df_cap, corpus,
            )
            continue
        kept.append(token)

    return " ".join(kept)
