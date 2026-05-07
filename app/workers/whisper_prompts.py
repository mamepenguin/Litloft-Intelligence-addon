"""Language-keyed default ``initial_prompt`` values for Whisper.

``initial_prompt`` is a short sentence (≤224 tokens) fed to the Whisper
decoder to bias its output style — primarily punctuation insertion in
languages that often omit it (Japanese, Chinese). Because the value is
inherently language-specific, we ship sensible defaults here and only
ask users to set ``whisper.initial_prompt`` in ``search-config.yml``
when they need a domain-specific override (e.g. specialised
vocabulary). The override, when provided, fully replaces the default —
we do not concatenate.

## Design constraint: form only, low content overlap

Whisper conditions on ``initial_prompt`` strongly enough that prompt
words bleed into transcripts as hallucinated repetitions when their
vocabulary overlaps the actual audio. So a "natural" prompt like
"Hello, everyone — today's weather is nice." is *worse* than no prompt
at all for a typical lecture/podcast/conversation library, because
"hello", "today", "nice" are extremely likely to appear in the audio
and Whisper will over-weight them.

The defaults below therefore aim for:

1. **Natural conversational register** — Whisper imitates style, so
   the prompt should match the casual/spoken register that most user
   recordings will fall under.
2. **Specific, low-overlap content** — a personal anecdote about a
   bakery croissant. Picked because (a) it's mundane enough to read
   as natural speech, and (b) the specific nouns (bakery, croissant)
   rarely appear in typical home-video / lecture / meeting audio, so
   even if Whisper conditions on them, the audio never matches and
   they don't bleed through.
3. **Rich punctuation** (``、。！？`` for CJK, ``,.;:!?—`` for Latin
   scripts) — the actual signal we want Whisper to imitate.
4. **Short** — every token spent on the prompt steals from the
   224-token decoder context.

Do not include filenames, names, or curated vocabulary lists; those
belong in a dedicated glossary layer.

## Chinese script note

Whisper's ``detect_language`` returns ``"zh"`` only — it does not
distinguish Simplified (zh-Hans) from Traditional (zh-Hant). The
default below is in Simplified Chinese. Users primarily writing
Traditional Chinese should set ``whisper.initial_prompt`` in
``search-config.yml`` to override.
"""

DEFAULT_INITIAL_PROMPTS: dict[str, str] = {
    "ja": "昨日、近所のパン屋でクロワッサンを買ったよ。思ったより美味しかったな、あれ。",
    "zh": "昨天去街角面包店买了个可颂，比想象中好吃多了。",
    "ko": "어제 동네 빵집에서 크루아상 하나 샀어. 생각보다 맛있더라, 진짜.",
    "en": "Yesterday, I picked up a croissant at the corner bakery. Better than expected, honestly.",
    "es": "Ayer compré un cruasán en la panadería de la esquina. La verdad, mejor de lo que esperaba.",
    "fr": "Hier, j'ai pris un croissant à la boulangerie du coin — finalement, meilleur que prévu.",
    "de": "Gestern habe ich beim Bäcker an der Ecke ein Croissant geholt. Ehrlich, besser als gedacht.",
    "pt": "Ontem peguei um croissant na padaria da esquina. Sinceramente, melhor do que eu esperava.",
    "it": "Ieri ho preso un cornetto al bar all'angolo. Sinceramente, meglio del previsto.",
    "ru": "Вчера купил круассан в булочной на углу. Честно, оказался вкуснее, чем ожидал.",
}


def resolve_initial_prompt(
    detected_language: str | None,
    override: str,
) -> str | None:
    """Pick the ``initial_prompt`` to feed Whisper for a given file.

    Resolution order:

    1. If the user set a non-blank override in ``search-config.yml``,
       use it verbatim. The override is treated as authoritative and
       fully replaces any language default — we never concatenate, to
       avoid the user inheriting hidden boilerplate.
    2. Otherwise, look up the detected language in
       ``DEFAULT_INITIAL_PROMPTS``.
    3. If the language is unknown / unsupported / detection failed,
       return ``None`` so Whisper runs without an initial_prompt
       (preferable to feeding it text in the wrong language, which
       wastes the 224-token context window).
    """
    if override and override.strip():
        return override

    if not detected_language:
        return None

    return DEFAULT_INITIAL_PROMPTS.get(detected_language)
