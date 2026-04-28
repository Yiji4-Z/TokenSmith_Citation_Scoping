"""
utils.py

Shared utility helpers for TokenSmith.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple


def parse_chapter_arg(values: list[str]) -> list[int]:
    """Parse --scope_chapter tokens into a flat list of chapter numbers.

    Accepts any mix of:
      - individual integers:  "18"       -> [18]
      - ranges:               "1-5"      -> [1, 2, 3, 4, 5]
      - combined:             "1-3" "7"  -> [1, 2, 3, 7]

    Raises ValueError on unrecognised tokens.
    """
    chapters: list[int] = []
    for token in values:
        if "-" in token:
            parts = token.split("-", 1)
            try:
                lo, hi = int(parts[0]), int(parts[1])
                chapters.extend(range(lo, hi + 1))
            except ValueError:
                raise ValueError(
                    f"Invalid chapter range '{token}'. Use format like '1-5'."
                )
        else:
            try:
                chapters.append(int(token))
            except ValueError:
                raise ValueError(
                    f"Invalid chapter number '{token}'. Must be an integer."
                )
    return chapters


# ---------------------------------------------------------------------------
# Auto-scope detection
# ---------------------------------------------------------------------------

# Map English number words to integers.
# Covers 1–50, which handles virtually all textbook chapter counts.
_WORD_TO_NUM: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
    "twenty-one": 21, "twenty-two": 22, "twenty-three": 23,
    "twenty-four": 24, "twenty-five": 25, "twenty-six": 26,
    "twenty-seven": 27, "twenty-eight": 28, "twenty-nine": 29,
    "thirty": 30,
    "thirty-one": 31, "thirty-two": 32, "thirty-three": 33,
    "thirty-four": 34, "thirty-five": 35, "thirty-six": 36,
    "thirty-seven": 37, "thirty-eight": 38, "thirty-nine": 39,
    "forty": 40,
    "forty-one": 41, "forty-two": 42, "forty-three": 43,
    "forty-four": 44, "forty-five": 45, "forty-six": 46,
    "forty-seven": 47, "forty-eight": 48, "forty-nine": 49,
    "fifty": 50,
}

# Longest-first so "twenty-one" matches before "twenty" or "one".
_WORD_PATTERN = "|".join(sorted(_WORD_TO_NUM, key=len, reverse=True))
_NUM_NC  = rf"(?:\d+|{_WORD_PATTERN})"   # non-capturing, for compound patterns
_NUM_CAP = rf"(\d+|{_WORD_PATTERN})"     # capturing, for extraction
_LIST_SEP = r"(?:\s*,\s*(?:and\s+)?|\s+and\s+)"  # ", " / ", and " / " and "


def _parse_num(token: str) -> Optional[int]:
    """Convert a digit string or English word to int, or return None."""
    t = token.strip().lower()
    return int(t) if t.isdigit() else _WORD_TO_NUM.get(t)


def _fraction_range(ordinal: str, fraction: str, total: int) -> Optional[Tuple[int, int]]:
    """
    Compute chapter range for a relative book-position phrase given total chapters.

    ordinal  : "first" | "second" | "third" | "fourth" | "middle" | "last" | "final"
    fraction : "half"  | "quarter" | "third"
    total    : total number of chapters in the book

    Returns (lo, hi) inclusive, or None if the combination is unrecognised.
    """
    o, f = ordinal.lower(), fraction.lower()

    if f == "half":
        mid = math.ceil(total / 2)
        mapping = {
            "first":  (1, mid),
            "second": (mid + 1, total),
            "last":   (mid + 1, total),
        }
    elif f == "quarter":
        q = math.ceil(total / 4)
        mapping = {
            "first":  (1, q),
            "second": (q + 1, 2 * q),
            "third":  (2 * q + 1, 3 * q),
            "fourth": (3 * q + 1, total),
            "last":   (3 * q + 1, total),
        }
    elif f == "third":
        t3 = math.ceil(total / 3)
        mapping = {
            "first":  (1, t3),
            "second": (t3 + 1, 2 * t3),
            "middle": (t3 + 1, 2 * t3),
            "last":   (2 * t3 + 1, total),
            "final":  (2 * t3 + 1, total),
        }
    else:
        return None

    return mapping.get(o)


def detect_scope_from_query(
    query: str,
    max_chapter: Optional[int] = None,
) -> Dict[str, object]:
    """Parse a natural-language query for chapter or page scope hints.

    Only explicit structural references trigger auto-scope; topic words and
    technical acronyms do not, to avoid false positives.

    Parameters
    ----------
    query       : raw user query string
    max_chapter : total number of chapters in the indexed book; required to
                  resolve open-ended ranges ("last 3 chapters", "from chapter
                  14 onwards") and relative fractions ("first half").  When
                  None these patterns are silently skipped.

    Recognised patterns
    -------------------
    Explicit chapter/section:
        "chapter 18"         "ch 18"           "ch. 18"
        "chapter eighteen"   "ch. fourteen"
        "chapters 14-18"     "chapters 14 to 18"
        "chapters 14, 15, and 18"
        "section 18.3"  (leading chapter number extracted)

    Count-based:
        "first 5 chapters"   "first five chapters"
        "last 3 chapters"    (needs max_chapter)

    Boundary-based (require the word "chapter"):
        "up to chapter 10"   "through chapter 10"  "until chapter 10"
        "before chapter 10"  → [1, 9]
        "from chapter 14"    "chapter 14 onwards"  "starting from chapter 14"
        "after chapter 14"   "chapter 14 and beyond"  (needs max_chapter)

    Relative fractions (needs max_chapter):
        "first half"    "second half"    "last half"
        "first quarter" "last quarter"   "third quarter"
        "first third"   "middle third"   "last third"

    Pages:
        "page 312"       "pages 312-320"
        "pages 312 to 320"   "pp. 312-320"   "pp 312"

    Returns
    -------
    dict with keys:
        "chapters" : list[int] | None
        "pages"    : tuple[int, int] | None
    """
    q = query.lower()
    chapter_nums: List[int] = []

    # ------------------------------------------------------------------
    # 1. Explicit chapter numbers / words / ranges
    # ------------------------------------------------------------------

    # Regex that matches words signalling a boundary expression just before
    # "chapter N" — used to skip those matches in the general pattern so that
    # "before chapter 10" and "after chapter 10" are handled exclusively by the
    # boundary patterns in section 3, not double-counted here.
    _boundary_pfx = re.compile(
        r"(?:before|after|from|through|until|thru|up\s+to"
        r"|starting(?:\s+from)?|onwards?)\s*$",
        re.IGNORECASE,
    )

    ch_range_re = re.compile(
        rf"\b(?:chapters?|ch\.?)\s+{_NUM_CAP}"
        rf"(?:\s*[-–]\s*{_NUM_CAP}|\s+to\s+{_NUM_CAP})?",
        re.IGNORECASE,
    )
    for m in ch_range_re.finditer(q):
        # Skip if this "chapter N" is the argument of a boundary expression
        if _boundary_pfx.search(q[: m.start()]):
            continue
        lo = _parse_num(m.group(1))
        hi_raw = m.group(2) or m.group(3)
        hi = _parse_num(hi_raw) if hi_raw else lo
        if lo is not None and hi is not None:
            chapter_nums.extend(range(lo, hi + 1))

    # Comma/and list: "chapters 14, 15, and 18" / "chapters two, five, and eighteen"
    ch_list_re = re.compile(
        rf"\bchapters?\s+((?:{_NUM_NC}{_LIST_SEP})*{_NUM_NC})",
        re.IGNORECASE,
    )
    for m in ch_list_re.finditer(q):
        if _boundary_pfx.search(q[: m.start()]):
            continue
        for tok in re.findall(rf"\b(?:\d+|{_WORD_PATTERN})\b", m.group(1), re.IGNORECASE):
            n = _parse_num(tok)
            if n is not None and n not in chapter_nums:
                chapter_nums.append(n)

    # "section X.Y" → chapter X (digits only, not word form)
    for m in re.finditer(r"\bsection\s+(\d+)\.\d+", q):
        n = int(m.group(1))
        if n not in chapter_nums:
            chapter_nums.append(n)

    # ------------------------------------------------------------------
    # 2. Count-based: "first N chapters" / "last N chapters"
    # ------------------------------------------------------------------
    first_n_re = re.compile(rf"\bfirst\s+{_NUM_CAP}\s+chapters?\b", re.IGNORECASE)
    for m in first_n_re.finditer(q):
        n = _parse_num(m.group(1))
        if n is not None:
            for c in range(1, n + 1):
                if c not in chapter_nums:
                    chapter_nums.append(c)

    if max_chapter is not None:
        last_n_re = re.compile(rf"\blast\s+{_NUM_CAP}\s+chapters?\b", re.IGNORECASE)
        for m in last_n_re.finditer(q):
            n = _parse_num(m.group(1))
            if n is not None:
                lo = max(1, max_chapter - n + 1)
                for c in range(lo, max_chapter + 1):
                    if c not in chapter_nums:
                        chapter_nums.append(c)

    # ------------------------------------------------------------------
    # 3. Boundary-based ("up to / through / until / before / from / after")
    #    All require the word "chapter" to avoid false matches on page numbers.
    # ------------------------------------------------------------------

    # "up to chapter N" / "through chapter N" / "until chapter N" → [1, N]
    up_to_re = re.compile(
        rf"\b(?:up\s+to|through|until|thru)\s+chapter\s+{_NUM_CAP}\b",
        re.IGNORECASE,
    )
    for m in up_to_re.finditer(q):
        n = _parse_num(m.group(1))
        if n is not None:
            for c in range(1, n + 1):
                if c not in chapter_nums:
                    chapter_nums.append(c)

    # "before chapter N" → [1, N-1]
    before_re = re.compile(
        rf"\bbefore\s+chapter\s+{_NUM_CAP}\b",
        re.IGNORECASE,
    )
    for m in before_re.finditer(q):
        n = _parse_num(m.group(1))
        if n is not None and n > 1:
            for c in range(1, n):
                if c not in chapter_nums:
                    chapter_nums.append(c)

    # "from chapter N" / "starting (from) chapter N" / "chapter N onwards/forward"
    # / "chapter N and beyond/above" / "after chapter N"  → [N, max]
    if max_chapter is not None:
        from_re = re.compile(
            rf"\b(?:from|starting\s+(?:from\s+)?)\s*chapter\s+{_NUM_CAP}\b"
            rf"|\bchapter\s+{_NUM_CAP}\s+(?:onwards?|forward|and\s+(?:beyond|above|on))\b",
            re.IGNORECASE,
        )
        for m in from_re.finditer(q):
            raw = m.group(1) or m.group(2)
            n = _parse_num(raw) if raw else None
            if n is not None:
                for c in range(n, max_chapter + 1):
                    if c not in chapter_nums:
                        chapter_nums.append(c)

        after_re = re.compile(
            rf"\bafter\s+chapter\s+{_NUM_CAP}\b",
            re.IGNORECASE,
        )
        for m in after_re.finditer(q):
            n = _parse_num(m.group(1))
            if n is not None:
                for c in range(n + 1, max_chapter + 1):
                    if c not in chapter_nums:
                        chapter_nums.append(c)

    # ------------------------------------------------------------------
    # 4. Relative fractions ("first half", "last quarter", etc.)
    #    Computed dynamically from max_chapter; skipped when unknown.
    # ------------------------------------------------------------------
    if max_chapter is not None:
        fraction_re = re.compile(
            r"\b(first|second|third|fourth|middle|last|final)\s+"
            r"(half|quarter|third)\b"
            r"(?:\s+of(?:\s+the)?\s+(?:book|course|text(?:book)?|material|content))?"
            r"(?:\s+of(?:\s+the)?\s+(?:book|course|text(?:book)?|material|content))?",
            re.IGNORECASE,
        )
        for m in fraction_re.finditer(q):
            span = _fraction_range(m.group(1), m.group(2), max_chapter)
            if span:
                lo, hi = span
                for c in range(lo, hi + 1):
                    if c not in chapter_nums:
                        chapter_nums.append(c)

    detected_chapters: Optional[List[int]] = sorted(set(chapter_nums)) if chapter_nums else None

    # ------------------------------------------------------------------
    # 5. Page patterns (digits only)
    # ------------------------------------------------------------------
    page_m = re.search(
        r"\b(?:pages?|pp\.?)\s+(\d+)(?:\s*[-–]\s*(\d+)|\s+to\s+(\d+))?",
        q,
    )
    detected_pages: Optional[Tuple[int, int]] = None
    if page_m:
        lo = int(page_m.group(1))
        hi_raw = page_m.group(2) or page_m.group(3)
        hi = int(hi_raw) if hi_raw else lo
        detected_pages = (lo, hi)

    return {"chapters": detected_chapters, "pages": detected_pages}
