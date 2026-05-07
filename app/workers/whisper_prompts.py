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

1. **High punctuation density** — Whisper imitates the comma/period
   rhythm it sees, so a one-sentence civic notice with several commas
   produces transcripts with noticeably more punctuation than two
   short sentences with one comma each. Each language is calibrated
   to its own native punctuation norms (e.g. Japanese tolerates 4
   commas in one sentence; English does not — comma counts vary
   across languages on purpose).
2. **Civic-notice / public-information register** — vocabulary like
   "sidewalk paving", "detour signs", "construction works" almost
   never appears in family videos / lectures / meetings / podcasts,
   so even though Whisper conditions on these tokens, the audio
   never matches and they don't bleed through.
3. **Conversational-but-informative tone** — close enough to natural
   speech that Whisper imitates the right register for typical user
   recordings, distinct enough in content to avoid hallucinating
   construction terminology into them.
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
    "ja": "市街地の中心部では、歩道の舗装工事が行われており、通行する際には、う回路が案内されています。",
    "zh": "市中心区域，正在进行人行道铺装施工，通行时，请按照指示牌指引绕行。",
    "ko": "시내 중심부에서, 보도 포장 공사가 진행 중입니다. 통행 시에는, 안내 표지에 따라 우회로를 이용해 주시기 바랍니다.",
    "en": "In the downtown area, sidewalk paving is currently underway; please follow the posted detour signs while passing through.",
    "es": "En el centro de la ciudad, se están realizando obras de pavimentación de la acera; al transitar, conviene seguir las señales de desvío indicadas.",
    "fr": "Dans le centre-ville, des travaux de pavage du trottoir sont en cours ; lors du passage, veuillez suivre les indications de la déviation signalée.",
    "de": "In der Innenstadt werden derzeit Pflasterarbeiten am Gehweg durchgeführt; beim Passieren sollten Sie, wie ausgeschildert, der Umleitung folgen.",
    "pt": "No centro da cidade, estão sendo realizadas obras de pavimentação da calçada; ao passar, siga as placas de desvio indicadas.",
    "it": "Nel centro città sono in corso lavori di pavimentazione del marciapiede; durante il passaggio, seguite le indicazioni del percorso alternativo.",
    "ru": "В центре города ведутся работы по укладке тротуара; при проходе, пожалуйста, следуйте указателям объездного маршрута.",
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
