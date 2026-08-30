"""What a pair of passages has literally in common.

Feeds the chips on a Related-passages row: the words that appear, word
for word, in **both** passages. Nothing is generated and nothing is
summarised — the row points at places, it does not write prose (hako
``DPcjrRgspKAXqHjHOkJ8L``).

This deliberately does **not** reuse ``citations._SEG_TOKEN_RE``. That
regex generates query terms for an FTS5 BM25 ``OR`` query, where IDF
discounts ``the`` / ``different`` / ``something`` on the retriever's
behalf. **BM25 was supplying the term weighting the tokeniser lacks**,
and a display surface has no such downstream weighting to lean on.
Sharing the two would also mean every change here lands in Ask's
citation matching, where a regression shows up as slightly worse
citations rather than as a failure.

Spec ``2026-08-30-related-passages-recognition-ui.md`` §7.2-7.3.
"""

from __future__ import annotations

import re

#: Words a reader could act on. Kanji runs, katakana runs, latin words,
#: and a number carrying a kanji or percent unit.
#:
#: **Hiragana matches nothing on purpose.** Japanese grammar lives in
#: kana, so particles and fillers cannot become terms and no stoplist is
#: needed — which is also exactly why §8's gate exists: no other script
#: separates its function words this way.
#:
#: The unit class holds no katakana. Letting it take one character split
#: the word that followed a number (``175センチ`` → ``175セ`` + ``ンチ``,
#: and ``ンチ`` was observed reaching the UI). A katakana unit is picked
#: up whole by the katakana-run alternative instead; the number becomes
#: its own term and is dropped by the ceiling, bare numbers being among
#: the most corpus-common tokens there are.
#:
#: **A multi-character kanji unit is still cut**: ``10周年`` yields
#: ``10周`` (and ``年`` alone is too short to survive), which was seen on
#: a real row. It is left alone deliberately — every fix available
#: without a lexicon of units trades it for something worse. Making the
#: unit greedy swallows the following word (``120度回転`` → one token);
#: refusing a unit that is followed by more kanji turns the same phrase
#: into ``120`` + ``度回転``; dropping digit-leading terms entirely would
#: have cost ``21日`` and ``8月``, which were the two most telling chips
#: on the row that surfaced this. A truncated term is a display blemish
#: on a word both passages share; the alternatives are wrong tokens.
_TERM_RE = re.compile(
    r"\d+(?:\.\d+)?[一-龥%]"  # 3日, 30% — a unit too short to survive alone
    r"|[一-龥々]+"  # kanji run (nouns, technical terms)
    r"|[ァ-ヴー]+"  # katakana run (loanwords, names)
    r"|[A-Za-z][A-Za-z0-9]+"  # latin words / acronyms
    r"|\d+(?:\.\d+)?"  # bare numbers
)

#: Single characters over-match, and the intersection is not enough to
#: rescue them: measured on real pairs, 86% of intersections are already
#: non-empty at two characters.
_MIN_TERM_LEN = 2

#: Hiragana only. See :func:`has_kana`.
_KANA_RE = re.compile(r"[ぁ-ゖ]")

#: A term that is only a number says nothing about what a pair is about,
#: and bare numbers are the most corpus-common tokens there are — ``20``
#: was the single most frequent term found in any intersection.
_BARE_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")

#: HTML entity names, which reach the text when extraction leaks the
#: markup around a word. ``amp`` was the fourth most frequent term found
#: in any intersection; it is not a word anybody wrote.
_MARKUP_ARTEFACTS = frozenset(
    {"amp", "quot", "nbsp", "lt", "gt", "apos", "rsquo", "ldquo", "rdquo"}
)

#: Rows returned to the UI. 22% of pairs produce five or more terms.
DEFAULT_CAP = 4


def salient_terms(text: str) -> list[str]:
    """Words from ``text`` that could name what it is about."""
    return [t for t in _TERM_RE.findall(text) if len(t) >= _MIN_TERM_LEN]


def has_kana(text: str) -> bool:
    """Whether the tokeniser's premise holds for ``text``.

    Not a language detector — a direct test of the condition under which
    :data:`_TERM_RE` also acts as a stopword filter. Measured across
    eight languages, where that separation is absent the intersection
    fills with function words instead of content: an unrelated pair of
    English passages yields ``different · something · because · there``,
    and Russian yields a single preposition once inflection has removed
    every content word. A confident wrong chip is worse than none.

    **Hiragana only.** Katakana was admitted at first on the reasoning
    that it could not open the gate for any of the scripts above — which
    is false. English prose quotes katakana freely, in names and
    loanwords, so two unrelated English passages mentioning ``ポケモン``
    and ``マリオ`` passed and were handed back
    ``different · something · because · there``: the exact failure the
    gate exists to prevent, through the gate. Hiragana is what carries
    the grammar, so hiragana is the precondition, stated as itself.

    A Japanese passage written without particles — a run of katakana
    terms — is therefore refused. That is the conservative side: it
    loses chips on a row that would have had good ones, rather than
    inventing chips on a row that should have none.
    """
    return bool(_KANA_RE.search(text))


def overlap_terms(
    mine: str,
    theirs: str,
    cap: int = DEFAULT_CAP,
) -> list[str]:
    """Words present in both passages, longest first.

    Ties are broken by where the term appears in ``mine``. Length is a
    crude stand-in for how much a word narrows things down, and it holds
    up in Japanese, which :func:`has_kana` makes the only place this
    runs. It does **not** hold in English, where ``something`` and
    ``because`` are long — which is the gate's job, not this one's.

    **No corpus statistics.** A document-frequency ceiling was tried and
    measured against the production library: on 42 real pairs it changed
    the visible four terms in 5% of rows, and in those it traded ``本当``
    for ``失礼`` and ``更新`` for ``4万`` while leaving ``今日`` standing.
    The two terms it was supposed to catch — ``20`` and ``amp`` — are
    removed above by shape, not by frequency. For that it cost a
    constant fitted to one corpus size (silently inert on a small
    library, over-eager on a large one) and made a drive-scoped response
    depend on counts drawn from drives the reader cannot open. Neither
    was worth 5%.
    """
    if not has_kana(mine) or not has_kana(theirs):
        return []

    other = {t.lower() for t in salient_terms(theirs)}
    seen: set[str] = set()
    shared: list[str] = []
    for term in salient_terms(mine):
        key = term.lower()
        if key in seen or key not in other:
            continue
        if _BARE_NUMBER.match(term) or key in _MARKUP_ARTEFACTS:
            continue
        seen.add(key)
        shared.append(term)

    # Stable, so equal-length terms keep the order they appear in.
    shared.sort(key=lambda t: -len(t))
    return shared[:cap]
