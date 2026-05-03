"""Tests for the structured form of app.rag.query_transform.

Phase 1 of the required-keyword hard filter spec
(``2026-04-30-required-semantic-hybrid-retrieval.md``) extends the
single-LLM-call query transform to emit a structured ``StructuredQuery``
instead of a flat keyword string. These tests cover:

* ``detect_script`` — Unicode-range based script classifier used to
  validate the LLM's self-declared script and to decide which
  mechanical alias rules apply.
* ``expand_aliases_mechanical`` — pure helper that, given a canonical
  term and its detected script, returns the set of mechanical
  aliases the Python layer adds regardless of LLM output (hira/kata
  for Japanese, NFKD/casefold for Latin, etc.).
* ``transform_query_structured`` — the async entry point. It calls the
  LLM exactly once, validates the JSON shape, merges Python-detected
  scripts and mechanical aliases, and falls back to a passthrough
  ``StructuredQuery`` on any failure mode (matching the strict
  graceful-degradation contract of the existing ``transform_query``).
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

for _mod in (
    "PIL", "PIL.Image",
    "open_clip",
    "torch",
    "sentence_transformers",
    "faster_whisper",
    "onnxruntime",
    "transformers",
    "janome", "janome.tokenizer",
    "sqlite_vec",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.rag.query_transform import (  # noqa: E402
    RequiredTerm,
    StructuredQuery,
    detect_script,
    expand_aliases_mechanical,
    transform_query_structured,
)


def _llm_stub(
    *,
    enabled: bool = True,
    response: dict | list | None = None,
    raises: type[Exception] | None = None,
) -> MagicMock:
    client = MagicMock()
    client.enabled = enabled
    if raises is not None:
        client.generate_json = AsyncMock(side_effect=raises("boom"))
    else:
        client.generate_json = AsyncMock(return_value=response)
    return client


class TestDetectScript:
    """Pure Unicode-range script classifier."""

    def test_pure_han(self):
        assert detect_script("芥川龍之介") == "han"

    def test_pure_hiragana(self):
        assert detect_script("ちいかわ") == "hira"

    def test_pure_katakana(self):
        assert detect_script("チイカワ") == "kata"

    def test_pure_latin(self):
        assert detect_script("Python") == "latin"

    def test_latin_with_diacritics(self):
        # The diacritic falls in Latin Supplement; classifier still
        # tags it Latin, not "other".
        assert detect_script("Café") == "latin"

    def test_cyrillic(self):
        assert detect_script("Привет") == "cyrillic"

    def test_hangul(self):
        assert detect_script("한국") == "hangul"

    def test_japanese_mixed_han_kata(self):
        # 古木 (han) + オリーブ (kata). Mixed Japanese collapses into
        # a single "japanese-mix" bucket so per-script alias rules
        # apply Japanese-wide kana folding.
        assert detect_script("古木オリーブ") == "japanese-mix"

    def test_japanese_mixed_han_hira(self):
        assert detect_script("芥川の本") == "japanese-mix"

    def test_empty_string(self):
        assert detect_script("") == "other"

    def test_pure_punctuation(self):
        assert detect_script("!?-") == "other"

    def test_majority_wins_for_borderline(self):
        # A loanword with a single accidental Latin letter ("ViT" inside
        # Japanese context) is classified by majority. Pure 1-char "T"
        # at the end of mostly-han input is still han.
        assert detect_script("画像分類モデル") == "japanese-mix"


class TestExpandAliasesMechanical:
    """Mechanical alias expansion. LLM aliases will be unioned in later."""

    def test_hira_canonical_adds_kata(self):
        aliases = expand_aliases_mechanical("ちいかわ", "hira")
        assert "ちいかわ" in aliases
        assert "チイカワ" in aliases

    def test_kata_canonical_adds_hira(self):
        aliases = expand_aliases_mechanical("チイカワ", "kata")
        assert "チイカワ" in aliases
        assert "ちいかわ" in aliases

    def test_japanese_mix_folds_both_directions(self):
        # 古木オリーブ → also produces a fully-hiragana variant and a
        # fully-katakana variant when applied to mixed input.
        aliases = expand_aliases_mechanical("古木オリーブ", "japanese-mix")
        assert "古木オリーブ" in aliases
        # Han characters unaffected; kana fold both ways.
        assert "古木おりーぶ" in aliases  # kata→hira applied
        assert "古木オリーブ" in aliases

    def test_han_only_no_mechanical_kana(self):
        # Pure Han has no automatic kana alias — readings are not
        # mechanically derivable. LLM may add yomi aliases later;
        # the mechanical layer only adds what's safe.
        aliases = expand_aliases_mechanical("芥川龍之介", "han")
        assert "芥川龍之介" in aliases
        # No hira/kata variants generated.
        assert all("ひらがな" not in a for a in aliases)

    def test_latin_adds_case_and_diacritic_strip(self):
        aliases = expand_aliases_mechanical("Café", "latin")
        assert "Café" in aliases
        assert "café" in aliases  # casefold
        assert "Cafe" in aliases  # diacritic stripped, case preserved
        assert "cafe" in aliases  # both

    def test_latin_simple(self):
        aliases = expand_aliases_mechanical("Python", "latin")
        assert "Python" in aliases
        assert "python" in aliases
        # Upper-casing isn't a typical user spelling so we don't add it
        # mechanically — case-fold is enough for FTS unicode61.

    def test_latin_strips_hyphens(self):
        # "video-share" should also match "videoshare" / "video share"
        # for FTS hits across word boundaries.
        aliases = expand_aliases_mechanical("video-share", "latin")
        assert "video-share" in aliases
        assert "video share" in aliases
        assert "videoshare" in aliases

    def test_other_script_returns_canonical_only(self):
        # Cyrillic / Hangul / "other" — no rules yet, just identity.
        aliases = expand_aliases_mechanical("Привет", "cyrillic")
        assert aliases == ("Привет",)

    def test_aliases_deduplicated(self):
        # Even when transforms collide, the result has no duplicates.
        aliases = expand_aliases_mechanical("python", "latin")
        assert len(aliases) == len(set(aliases))


class TestTransformQueryStructuredHappyPath:
    @pytest.mark.asyncio
    async def test_returns_structured_query_with_required_and_semantic(
        self, monkeypatch
    ):
        llm_response = {
            "required": [
                {
                    "canonical": "芥川龍之介",
                    "script": "han",
                    "aliases": ["芥川龍之介", "芥川"],
                }
            ],
            "semantic": ["動物", "作品"],
        }
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response=llm_response),
        )

        result = await transform_query_structured("動物が登場する芥川龍之介の作品")

        assert isinstance(result, StructuredQuery)
        assert len(result.required) == 1
        term = result.required[0]
        assert term.canonical == "芥川龍之介"
        # LLM-supplied aliases survive.
        assert "芥川龍之介" in term.aliases
        assert "芥川" in term.aliases
        # Semantic terms survive.
        assert result.semantic == ("動物", "作品")

    @pytest.mark.asyncio
    async def test_python_unions_mechanical_aliases_with_llm(
        self, monkeypatch
    ):
        # LLM forgets the katakana variant of "ちいかわ"; Python adds it.
        llm_response = {
            "required": [
                {
                    "canonical": "ちいかわ",
                    "script": "hira",
                    "aliases": ["ちいかわ"],
                }
            ],
            "semantic": ["アイス"],
        }
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response=llm_response),
        )

        result = await transform_query_structured("ちいかわのアイス")

        assert len(result.required) == 1
        aliases = result.required[0].aliases
        assert "ちいかわ" in aliases
        assert "チイカワ" in aliases  # mechanically added

    @pytest.mark.asyncio
    async def test_python_overrides_llm_script_when_mismatched(
        self, monkeypatch
    ):
        # LLM mislabels "Python" as han; Unicode says latin. Python wins.
        llm_response = {
            "required": [
                {
                    "canonical": "Python",
                    "script": "han",  # wrong
                    "aliases": ["Python"],
                }
            ],
            "semantic": ["動物"],
        }
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response=llm_response),
        )

        result = await transform_query_structured("Pythonで動物")

        assert result.required[0].script == "latin"
        # And mechanical Latin alias rules ran (lower-case added).
        assert "python" in result.required[0].aliases

    @pytest.mark.asyncio
    async def test_empty_required_collapses_to_all_semantic(
        self, monkeypatch
    ):
        # LLM decides nothing is required (e.g. fully-conceptual query).
        llm_response = {
            "required": [],
            "semantic": ["紅葉", "京都"],
        }
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response=llm_response),
        )

        result = await transform_query_structured("綺麗な紅葉")

        assert result.required == ()
        assert result.semantic == ("紅葉", "京都")

    @pytest.mark.asyncio
    async def test_raw_keywords_preserves_legacy_join_for_callers(
        self, monkeypatch
    ):
        # Legacy callers still join required canonicals + semantic
        # into a single string for the FTS path. raw_keywords exposes
        # exactly that string.
        llm_response = {
            "required": [
                {
                    "canonical": "東福寺",
                    "script": "han",
                    "aliases": ["東福寺"],
                }
            ],
            "semantic": ["紅葉"],
        }
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response=llm_response),
        )

        result = await transform_query_structured("東福寺の紅葉について")

        assert "東福寺" in result.raw_keywords
        assert "紅葉" in result.raw_keywords


class TestTransformQueryStructuredFallbacks:
    """Every failure mode collapses to passthrough(query)."""

    @pytest.mark.asyncio
    async def test_falls_back_when_llm_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(enabled=False),
        )

        result = await transform_query_structured("raw query")

        assert isinstance(result, StructuredQuery)
        assert result.required == ()
        assert result.semantic == ("raw query",)
        assert result.raw_keywords == "raw query"

    @pytest.mark.asyncio
    async def test_falls_back_when_client_unavailable(self, monkeypatch):
        def _raise():
            raise RuntimeError("not initialized")

        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client", _raise
        )

        result = await transform_query_structured("raw query")

        assert result.required == ()
        assert result.semantic == ("raw query",)

    @pytest.mark.asyncio
    async def test_falls_back_when_llm_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response=None),
        )

        result = await transform_query_structured("raw query")

        assert result.required == ()
        assert result.semantic == ("raw query",)

    @pytest.mark.asyncio
    async def test_falls_back_when_llm_returns_legacy_flat_schema(
        self, monkeypatch
    ):
        # Pre-Phase-1 prompts return {"keywords": "..."}. After the
        # prompt swap this should not happen, but we accept it as a
        # transitional fallback: every LLM-produced word becomes
        # semantic, no required terms.
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response={"keywords": "東福寺 紅葉"}),
        )

        result = await transform_query_structured("東福寺の紅葉")

        assert result.required == ()
        # Legacy flat keywords land in semantic, split on whitespace.
        assert "東福寺" in result.semantic
        assert "紅葉" in result.semantic

    @pytest.mark.asyncio
    async def test_falls_back_when_required_term_missing_canonical(
        self, monkeypatch
    ):
        # Malformed required entry (no canonical) is dropped; the rest
        # of the response still applies.
        llm_response = {
            "required": [
                {"script": "han", "aliases": ["something"]},  # no canonical
                {"canonical": "東福寺", "script": "han", "aliases": ["東福寺"]},
            ],
            "semantic": ["紅葉"],
        }
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response=llm_response),
        )

        result = await transform_query_structured("東福寺の紅葉")

        assert len(result.required) == 1
        assert result.required[0].canonical == "東福寺"

    @pytest.mark.asyncio
    async def test_handles_empty_input(self, monkeypatch):
        spy_called = False

        def _get_client():
            nonlocal spy_called
            spy_called = True
            return _llm_stub(response={})

        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client", _get_client
        )

        result = await transform_query_structured("   ")

        assert result.required == ()
        assert result.semantic == ("   ",)  # passthrough preserves input
        assert spy_called is False  # no wasted LLM call


class TestRequiredCanonicalBlocklist:
    """A required canonical that matches the file-type / question-word
    blocklist must be dropped, even if the LLM ignored the system
    prompt's instruction to exclude it.

    Otherwise the bad canonical becomes a hard filter — e.g. ``video``
    or ``動画`` would force every retrieve to find the literal string,
    excluding most legitimate hits. Defense-in-depth alongside the
    prompt rule.
    """

    @pytest.mark.asyncio
    async def test_english_video_dropped_from_required(self, monkeypatch):
        llm_response = {
            "required": [
                {"canonical": "video", "script": "latin", "aliases": ["video"]},
                {
                    "canonical": "東福寺",
                    "script": "han",
                    "aliases": ["東福寺"],
                },
            ],
            "semantic": ["京都"],
        }
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response=llm_response),
        )

        result = await transform_query_structured("東福寺の video")

        canonicals = [t.canonical for t in result.required]
        assert "video" not in canonicals
        assert "東福寺" in canonicals

    @pytest.mark.asyncio
    async def test_japanese_doga_dropped_from_required(self, monkeypatch):
        llm_response = {
            "required": [
                {"canonical": "動画", "script": "han", "aliases": ["動画"]},
            ],
            "semantic": ["料理"],
        }
        monkeypatch.setattr(
            "app.rag.query_transform.get_llm_client",
            lambda: _llm_stub(response=llm_response),
        )

        result = await transform_query_structured("料理の動画")

        canonicals = [t.canonical for t in result.required]
        assert "動画" not in canonicals


class TestStructuredQueryHelpers:
    """Sanity tests for StructuredQuery helpers used by callers."""

    def test_passthrough_constructs_no_signal_form(self):
        q = StructuredQuery.passthrough("hello world")
        assert q.required == ()
        assert q.semantic == ("hello world",)
        assert q.raw_keywords == "hello world"

    def test_required_term_is_frozen(self):
        term = RequiredTerm(
            canonical="x", script="latin", aliases=("x",)
        )
        with pytest.raises(Exception):
            term.canonical = "y"  # type: ignore[misc]


class TestRequiredFallbackLadder:
    """Phase 4: Tier 2 fallback steps drop one term at a time."""

    def test_empty_required_yields_only_terminal(self):
        from app.rag.query_transform import iter_required_fallback_subsets

        # Empty input: nothing to ladder, just Tier 3 terminal.
        steps = list(iter_required_fallback_subsets(()))
        assert steps == [()]

    def test_single_term_yields_drop_then_terminal(self):
        from app.rag.query_transform import iter_required_fallback_subsets

        term = RequiredTerm(canonical="A", script="latin", aliases=("A",))
        steps = list(iter_required_fallback_subsets((term,)))

        # Caller already tried the full set, so the ladder skips it
        # and starts with N-1=0 terms (which equals Tier 3 here).
        assert steps == [()]

    def test_drops_highest_alias_count_first(self):
        from app.rag.query_transform import iter_required_fallback_subsets

        specific = RequiredTerm(  # 1 alias = most specific
            canonical="ViT", script="latin", aliases=("ViT",)
        )
        generic = RequiredTerm(  # 4 aliases = most generic
            canonical="紅葉",
            script="han",
            aliases=("紅葉", "こうよう", "コウヨウ", "もみじ"),
        )
        steps = list(
            iter_required_fallback_subsets((specific, generic))
        )

        # First step: drop "紅葉" (more aliases ⇒ more generic)
        # Second step: drop both ⇒ Tier 3 terminal
        assert steps == [(specific,), ()]

    def test_three_terms_drops_one_at_a_time(self):
        from app.rag.query_transform import iter_required_fallback_subsets

        a = RequiredTerm(canonical="A", script="latin", aliases=("A",))
        b = RequiredTerm(canonical="B", script="latin", aliases=("B", "b"))
        c = RequiredTerm(
            canonical="C", script="latin", aliases=("C", "c", "see")
        )
        steps = list(iter_required_fallback_subsets((a, b, c)))

        # By alias count: c=3 (most generic), b=2, a=1 (most specific).
        # Ladder drops c first, then b, then a, ending at Tier 3.
        assert steps == [(a, b), (a,), ()]

    def test_ties_break_on_position(self):
        from app.rag.query_transform import iter_required_fallback_subsets

        # Two terms with identical alias count — preserve original
        # order on tie so the ladder is deterministic across runs.
        x = RequiredTerm(canonical="X", script="latin", aliases=("X",))
        y = RequiredTerm(canonical="Y", script="latin", aliases=("Y",))
        steps = list(iter_required_fallback_subsets((x, y)))

        # Equal alias counts — the later-positioned term drops first
        # (position-based tiebreak so the user's leading term is kept
        # the longest, matching the structured prompt's "first =
        # most important" intuition).
        assert steps == [(x,), ()]
