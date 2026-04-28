"""
tests/test_auto_scope.py

Unit tests for detect_scope_from_query() in src/utils.py.

Run with: python -m pytest tests/test_auto_scope.py -v
"""

import pytest
from src.utils import detect_scope_from_query

# Simulated total chapter count for tests that need max_chapter.
TOTAL = 26


# ---------------------------------------------------------------------------
# 1. Explicit chapter numbers / words / ranges
# ---------------------------------------------------------------------------

class TestExplicitChapter:
    def test_single_digit(self):
        r = detect_scope_from_query("chapter 18")
        assert r["chapters"] == [18]

    def test_single_word(self):
        r = detect_scope_from_query("chapter eighteen")
        assert r["chapters"] == [18]

    def test_abbrev_ch(self):
        r = detect_scope_from_query("see ch 14")
        assert r["chapters"] == [14]

    def test_abbrev_ch_dot_word(self):
        r = detect_scope_from_query("ch. fourteen is about indexing")
        assert r["chapters"] == [14]

    def test_range_dash(self):
        r = detect_scope_from_query("chapters 14-18")
        assert set(r["chapters"]) == {14, 15, 16, 17, 18}

    def test_range_to(self):
        r = detect_scope_from_query("chapter 1 to 3")
        assert set(r["chapters"]) >= {1, 2, 3}

    def test_range_word_to_word(self):
        r = detect_scope_from_query("chapters fourteen to eighteen")
        assert set(r["chapters"]) == {14, 15, 16, 17, 18}

    def test_comma_list_digits(self):
        r = detect_scope_from_query("chapters 14, 15, and 18 are on the exam")
        assert 14 in r["chapters"]
        assert 15 in r["chapters"]
        assert 18 in r["chapters"]

    def test_comma_list_words(self):
        r = detect_scope_from_query("chapters two, five, and eighteen")
        assert 2 in r["chapters"]
        assert 5 in r["chapters"]
        assert 18 in r["chapters"]

    def test_comma_list_mixed(self):
        r = detect_scope_from_query("chapters two, 5, and eighteen are relevant")
        assert 2 in r["chapters"]
        assert 5 in r["chapters"]
        assert 18 in r["chapters"]

    def test_section_extracts_chapter(self):
        r = detect_scope_from_query("section 18.3 explains deadlocks")
        assert 18 in r["chapters"]

    def test_twenty_one(self):
        r = detect_scope_from_query("chapter twenty-one")
        assert r["chapters"] == [21]

    def test_no_false_positive_2pl(self):
        r = detect_scope_from_query("explain 2PL protocol and T1 deadlock")
        assert r["chapters"] is None

    def test_no_topic_word_trigger(self):
        r = detect_scope_from_query("how does ARIES recovery work?")
        assert r["chapters"] is None


# ---------------------------------------------------------------------------
# 2. Count-based: "first/last N chapters"
# ---------------------------------------------------------------------------

class TestCountBased:
    def test_first_5_chapters(self):
        r = detect_scope_from_query("quiz me on the first 5 chapters")
        assert r["chapters"] == list(range(1, 6))

    def test_first_five_chapters_word(self):
        r = detect_scope_from_query("I only studied the first five chapters")
        assert r["chapters"] == list(range(1, 6))

    def test_first_one_chapter(self):
        r = detect_scope_from_query("just the first chapter")
        # "first chapter" (no N) — should NOT trigger; needs "first N chapters"
        assert r["chapters"] is None

    def test_last_3_chapters(self):
        r = detect_scope_from_query("last 3 chapters", max_chapter=TOTAL)
        assert r["chapters"] == list(range(TOTAL - 2, TOTAL + 1))

    def test_last_three_chapters_word(self):
        r = detect_scope_from_query("the last three chapters of the book", max_chapter=TOTAL)
        assert r["chapters"] == list(range(TOTAL - 2, TOTAL + 1))

    def test_last_n_skipped_without_max(self):
        # Without max_chapter, "last N chapters" is unresolvable → no scope
        r = detect_scope_from_query("last 3 chapters")
        assert r["chapters"] is None


# ---------------------------------------------------------------------------
# 3. Boundary-based
# ---------------------------------------------------------------------------

class TestBoundary:
    def test_up_to_chapter(self):
        r = detect_scope_from_query("up to chapter 10")
        assert r["chapters"] == list(range(1, 11))

    def test_through_chapter(self):
        r = detect_scope_from_query("through chapter ten")
        assert r["chapters"] == list(range(1, 11))

    def test_until_chapter(self):
        r = detect_scope_from_query("until chapter 8")
        assert r["chapters"] == list(range(1, 9))

    def test_before_chapter(self):
        r = detect_scope_from_query("everything before chapter 18")
        assert r["chapters"] == list(range(1, 18))

    def test_before_chapter_word(self):
        r = detect_scope_from_query("before chapter eighteen")
        assert r["chapters"] == list(range(1, 18))

    def test_from_chapter_onwards(self):
        r = detect_scope_from_query("from chapter 14 onwards", max_chapter=TOTAL)
        assert r["chapters"] == list(range(14, TOTAL + 1))

    def test_chapter_and_beyond(self):
        r = detect_scope_from_query("chapter 14 and beyond", max_chapter=TOTAL)
        assert r["chapters"] == list(range(14, TOTAL + 1))

    def test_starting_from_chapter(self):
        r = detect_scope_from_query("starting from chapter 14", max_chapter=TOTAL)
        assert r["chapters"] == list(range(14, TOTAL + 1))

    def test_after_chapter(self):
        r = detect_scope_from_query("after chapter 13", max_chapter=TOTAL)
        assert r["chapters"] == list(range(14, TOTAL + 1))

    def test_after_chapter_word(self):
        r = detect_scope_from_query("after chapter thirteen", max_chapter=TOTAL)
        assert r["chapters"] == list(range(14, TOTAL + 1))

    def test_open_ended_skipped_without_max(self):
        # "from chapter N onwards" needs max_chapter to know the upper bound
        r = detect_scope_from_query("from chapter 14 onwards")
        assert r["chapters"] is None


# ---------------------------------------------------------------------------
# 4. Relative fractions (dynamic, need max_chapter)
# ---------------------------------------------------------------------------

class TestFractions:
    def test_first_half(self):
        r = detect_scope_from_query("first half of the book", max_chapter=TOTAL)
        assert r["chapters"] == list(range(1, 14))      # ceil(26/2)=13

    def test_second_half(self):
        r = detect_scope_from_query("second half", max_chapter=TOTAL)
        assert r["chapters"] == list(range(14, TOTAL + 1))

    def test_last_half(self):
        r = detect_scope_from_query("last half of the textbook", max_chapter=TOTAL)
        assert r["chapters"] == list(range(14, TOTAL + 1))

    def test_first_quarter(self):
        r = detect_scope_from_query("the first quarter", max_chapter=TOTAL)
        assert r["chapters"] == list(range(1, 8))       # ceil(26/4)=7

    def test_last_quarter(self):
        r = detect_scope_from_query("last quarter of the course", max_chapter=TOTAL)
        assert 26 in r["chapters"]
        assert r["chapters"][0] > 13  # well into second half

    def test_first_third(self):
        r = detect_scope_from_query("first third of the book", max_chapter=TOTAL)
        assert r["chapters"] == list(range(1, 10))      # ceil(26/3)=9

    def test_middle_third(self):
        r = detect_scope_from_query("the middle third", max_chapter=TOTAL)
        assert 10 in r["chapters"] or 9 in r["chapters"]   # middle band
        assert 26 not in r["chapters"]

    def test_last_third(self):
        r = detect_scope_from_query("last third of the material", max_chapter=TOTAL)
        assert 26 in r["chapters"]

    def test_final_third(self):
        r = detect_scope_from_query("final third", max_chapter=TOTAL)
        assert 26 in r["chapters"]

    def test_fractions_skipped_without_max(self):
        r = detect_scope_from_query("first half of the book")
        assert r["chapters"] is None

    def test_different_total(self):
        # Works correctly for a different-sized book (e.g. 20 chapters)
        r = detect_scope_from_query("first half of the book", max_chapter=20)
        assert r["chapters"] == list(range(1, 11))      # ceil(20/2)=10


# ---------------------------------------------------------------------------
# 5. Page detection
# ---------------------------------------------------------------------------

class TestPageDetection:
    def test_single_page(self):
        r = detect_scope_from_query("what is on page 312?")
        assert r["pages"] == (312, 312)

    def test_page_range_dash(self):
        r = detect_scope_from_query("pages 1295-1310")
        assert r["pages"] == (1295, 1310)

    def test_page_range_to(self):
        r = detect_scope_from_query("pages 100 to 120")
        assert r["pages"] == (100, 120)

    def test_pp_abbreviation(self):
        r = detect_scope_from_query("see pp. 312-320")
        assert r["pages"] == (312, 320)

    def test_no_pages(self):
        r = detect_scope_from_query("chapter 19")
        assert r["pages"] is None


# ---------------------------------------------------------------------------
# 6. Combined and edge cases
# ---------------------------------------------------------------------------

class TestCombinedAndEdgeCases:
    def test_chapter_and_page(self):
        r = detect_scope_from_query("chapter 18 pages 1295-1310")
        assert 18 in r["chapters"]
        assert r["pages"] == (1295, 1310)

    def test_empty_query(self):
        r = detect_scope_from_query("")
        assert r["chapters"] is None
        assert r["pages"] is None

    def test_no_scope_plain_question(self):
        r = detect_scope_from_query("how does a B+ tree work?")
        assert r["chapters"] is None
        assert r["pages"] is None
