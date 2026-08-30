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


class TestOverlapTerms:
    def test_only_words_present_on_both_sides(self):
        terms = overlap_terms(
            "対角線を軸として回転させると立方体になります",
            "同じ性質を持つ回転が立方体にもあり対角線について",
        )

        assert set(terms) == {"対角線", "回転", "立方体"}

    def test_ordered_by_length_then_first_seen(self):
        terms = overlap_terms(
            "回転と対角線と立方体の話",
            "立方体の対角線をめぐる回転",
        )

        # 対角線 and 立方体 are both three characters; the order they
        # appear in this file breaks the tie.
        assert terms == ["対角線", "立方体", "回転"]

    def test_capped(self):
        mine = "対角線と立方体と回転と軸性と要素と性質の話"
        terms = overlap_terms(mine, mine)

        assert len(terms) == 4

    def test_a_bare_number_is_not_a_term(self):
        # `20` was the single most frequent term found in any real
        # intersection, and says nothing about what a pair is about.
        terms = overlap_terms("20回と1本の身長の話", "身長は20で1本だけ", cap=10)

        assert "20" not in terms
        assert {"1本", "身長"} <= set(terms)

    def test_markup_that_leaked_into_the_text_is_not_a_term(self):
        # `amp` reaches the text when extraction leaks the markup around
        # a word. Nobody wrote it.
        terms = overlap_terms("amp と身長の話です", "身長の話に amp が混ざる", cap=10)

        assert "amp" not in terms
        assert "身長" in terms

    def test_an_ordinary_word_is_kept_rather_than_guessed_at(self):
        # A document-frequency ceiling used to remove these. Measured on
        # 42 real pairs it changed the visible four terms in 5% of rows,
        # in exchange for a constant fitted to one corpus size and counts
        # drawn from drives the reader cannot open.
        terms = overlap_terms("更新は本当に大事です", "本当に更新しました", cap=10)

        assert set(terms) == {"更新", "本当"}

    def test_case_is_folded(self):
        # Spec §11 defect 2: `You` and `you` counted as different terms.
        terms = overlap_terms(
            "これはケースの話です You know",
            "ですね you know について",
        )

        assert [t.lower() for t in terms].count("you") == 1

    def test_nothing_in_common_yields_nothing(self):
        assert (
            overlap_terms("対角線の話です", "全然ちがう話題ですね") == []
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
        assert overlap_terms(mine, theirs) == []

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
        assert overlap_terms(mine, theirs) == []

    def test_one_japanese_side_is_not_enough(self):
        # Both halves get tokenised, so both must be readable.
        assert (
            overlap_terms(
                "対角線を軸として回転させる", "the diagonal axis we rotate"
            )
            == []
        )
