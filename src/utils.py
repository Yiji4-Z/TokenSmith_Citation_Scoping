"""
utils.py

Shared utility helpers for TokenSmith.
"""

from __future__ import annotations


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
