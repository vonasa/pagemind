"""Extract an explicit chapter reference from a user's question, for exact scoping.

Used by the orchestrators to decide whether a question is *about* a specific chapter
("what character appears in chapter 3") and, if so, to pin retrieval to that chapter's
display ``number``. Deliberately **strict** — only matches an explicit "chapter N" /
"ch. N" / "chap. N" or a spelled-out ordinal ("the third chapter") — so a bare count
like "3 quotes about war" is never mistaken for a chapter scope.
"""
from __future__ import annotations

import re

_CHAPTER_RE = re.compile(r"\b(?:chapter|chap\.?|ch\.?)\s*(\d+)\b", re.IGNORECASE)

_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
# "the third chapter" / "chapter the third" — an ordinal word adjacent to "chapter".
_ORDINAL_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+chapter\b"
    r"|\bchapter\s+(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    re.IGNORECASE,
)


def parse_chapter_reference(question: str) -> int | None:
    """Return the explicit chapter number referenced by *question*, or None.

    Matches "chapter 3", "ch. 3", "chap 3", "the third chapter", "chapter the third".
    Does NOT fall back to a bare digit — that avoids scoping counts/years.
    """
    m = _CHAPTER_RE.search(question)
    if m:
        return int(m.group(1))
    m = _ORDINAL_RE.search(question)
    if m:
        word = m.group(1) or m.group(2)
        return _ORDINAL_WORDS.get(word.lower())
    return None
