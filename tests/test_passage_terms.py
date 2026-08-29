"""What a pair of passages has literally in common.

The fixtures are the eight-language probe and the production
measurement recorded in spec
``2026-08-30-related-passages-recognition-ui.md`` §6. They are the
reason this module exists in its present shape, so they are asserted
here rather than described in a comment.
"""

import pytest

from app.passage_terms import has_kana, overlap_terms, salient_terms


# --- Tokenising -----------------------------------------------------


class TestSalientTerms:
    @pytest.mark.parametrize(
        "text,expected",
        [
            # Spec §11 defect 1: the unit class took one character, so a
            # katakana word after a number lost its first letter and the
            # remainder (`ンチ`) reached the UI as a candidate chip.
            ("身長が175センチで", "センチ"),
            ("体重66キロで", "キロ"),
            ("30メートル先", "メートル"),
        ],
    )
    def test_a_number_does_not_eat_the_word_after_it(self, text, expected):
        assert expected in salient_terms(text)

    @pytest.mark.parametrize("text,expected", [("3日で", "3日"), ("30%の", "30%")])
    def test_a_kanji_or_percent_unit_still_rides_with_its_number(
        self, text, expected
    ):
        # Without this the unit is a single character and the length
        # filter drops it, which is what the alternative was added for.
        assert expected in salient_terms(text)

    def test_hiragana_never_becomes_a_term(self):
        # The whole reason the tokeniser works in Japanese: particles and
        # fillers cannot match, so no stoplist is needed.
        assert salient_terms("それはとてもよいことですね") == []

    def test_single_characters_are_dropped(self):
        assert "軸" not in salient_terms("軸として回転させる")

    def test_kanji_katakana_and_latin_runs_are_terms(self):
        terms = salient_terms("対角線をXYZ軸でスリット回転")
        assert "対角線" in terms
        assert "スリット" in terms
        assert "XYZ" in terms


# --- The kana gate --------------------------------------------------


class TestHasKana:
    def test_japanese_opens_the_gate(self):
        assert has_kana("対角線を軸として回転させると")

    def test_katakana_alone_does_not_open_it(self):
        # English prose quotes katakana freely. Admitting it let two
        # unrelated English passages through the gate that exists to
        # keep them out.
        assert not has_kana("I played ポケモン and マリオ all weekend")

    @pytest.mark.parametrize(
        "text",
        [
            "the diagonal axis is what we rotate around",
            "这就是说对角线是我们旋转的轴所以立方体看起来一样",
            "이것은 대각선이 우리가 회전하는 축이라는 뜻이며",
            "это означает что диагональ является осью вокруг которой",
            "das bedeutet die Raumdiagonale ist die Achse um die wir drehen",
        ],
    )
    def test_every_other_script_keeps_it_closed(self, text):
        assert not has_kana(text)


# --- The intersection -----------------------------------------------


def df_none(_term: str) -> int:
    return 0


# Document frequencies measured against the production library
# (spec §6.2). The junk sits in the extreme tail; the content words sit
# well below it, and the ceiling's only job is to land in that gap.
MEASURED_DF = {
    "20": 7796,
    "更新": 1415,
    "本当": 1288,
    "amp": 697,
    "1本": 154,
    "身長": 113,
    "山根": 77,
}


def df_measured(term: str) -> int:
    return MEASURED_DF.get(term, 0)


class TestOverlapTerms:
    def test_only_words_present_on_both_sides(self):
        terms = overlap_terms(
            "対角線を軸として回転させると立方体になります",
            "同じ性質を持つ回転が立方体にもあり対角線について",
            df=df_none,
        )

        assert set(terms) == {"対角線", "回転", "立方体"}

    def test_ordered_by_length_then_first_seen(self):
        terms = overlap_terms(
            "回転と対角線と立方体の話",
            "立方体の対角線をめぐる回転",
            df=df_none,
        )

        # 対角線 and 立方体 are both three characters; the order they
        # appear in this file breaks the tie.
        assert terms == ["対角線", "立方体", "回転"]

    def test_capped(self):
        mine = "対角線と立方体と回転と軸性と要素と性質の話"
        terms = overlap_terms(mine, mine, df=df_none)

        assert len(terms) == 4

    def test_corpus_common_words_are_dropped(self):
        mine = "20回の更新は本当に amp と1本の身長と山根の話"
        terms = overlap_terms(mine, mine, df=df_measured, cap=10)

        # The tail is junk; everything under the ceiling is a word a
        # reader could act on.
        assert "20" not in terms
        assert "更新" not in terms
        assert "本当" not in terms
        assert "amp" not in terms
        assert {"1本", "身長", "山根"} <= set(terms)

    def test_no_document_frequency_source_means_no_filtering(self):
        # rarity_filter is fail-open: an uninitialised DB reports 0 for
        # everything. Chips are still better than a blank row.
        terms = overlap_terms("更新と20の話", "20の更新について", df=None)

        assert set(terms) == {"更新", "20"}

    def test_case_is_folded(self):
        # Spec §11 defect 2: `You` and `you` counted as different terms.
        terms = overlap_terms(
            "これはケースの話です You know",
            "ですね you know について",
            df=df_none,
        )

        assert [t.lower() for t in terms].count("you") == 1

    def test_nothing_in_common_yields_nothing(self):
        assert (
            overlap_terms("対角線の話です", "全然ちがう話題ですね", df=df_none) == []
        )

    @pytest.mark.parametrize(
        "mine,theirs",
        [
            # The case that made "a wrong chip is worse than no chip" a
            # constraint: two unrelated English passages share four
            # function words and would assert a connection that is not
            # there.
            (
                "the weather this morning was different from what they had "
                "expected because there was something about the pressure",
                "there is something different about the way that people talk "
                "about this because it was never actually something",
            ),
            # Inflection leaves only prepositions to match on.
            (
                "это означает что диагональ является осью вокруг которой",
                "вращение куба вокруг диагонали соответствует перестановке",
            ),
        ],
    )
    def test_a_pair_the_tokeniser_cannot_read_yields_nothing(self, mine, theirs):
        assert overlap_terms(mine, theirs, df=df_none) == []

    def test_english_passages_that_mention_katakana_get_nothing(self):
        mine = (
            "the weather this morning was different from what they had "
            "expected because there was something about ポケモン"
        )
        theirs = (
            "there is something different about the way people talk about "
            "this because it was never actually about マリオ"
        )

        # Four shared function words, and a confident chip row asserting
        # a connection that is not there, is worse than no chips at all.
        assert overlap_terms(mine, theirs, df=df_none) == []

    def test_one_japanese_side_is_not_enough(self):
        # Both halves get tokenised, so both must be readable.
        assert (
            overlap_terms(
                "対角線を軸として回転させる", "the diagonal axis we rotate", df=df_none
            )
            == []
        )
