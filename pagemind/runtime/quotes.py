"""Quote selection helpers shared by the verbatim recipe and the streaming path.

The reader fan-out can surface the same line from several overlapping sections or
return more quotes than a reader wants to show. ``select_quotes`` deduplicates and
caps them without disturbing the reader results the synthesizer prose is built
from, and ``format_quote_answer`` produces a short framing line (never the quote
text itself — quotes live in cards).
"""
from __future__ import annotations

import dataclasses
import re

from pagemind.runtime.reader import _normalize
from pagemind.runtime.types import ReadResult

# Quote-count planning. A request like "7 quotes describing warmth" should surface
# 7 quote cards, not the default 3. ``parse_requested_quote_count`` extracts the
# number, ``plan_quotes`` turns it into (want, sections-to-read) — shared by both
# the web streaming path and the CLI recipe so they can't drift.
_DEFAULT_QUOTES = 3
_MAX_QUOTES = 12
_MIN_SECTIONS = 3
_MAX_SECTIONS = 10
# Read a few more sections than quotes wanted: per-section yield is < 1:1 after the
# reader drops unlocatable quotes and select_quotes dedupes across sections.
_SECTION_MARGIN = 3

# A count is only honoured when it's attached to a quote-ish noun, so chapter
# numbers and years ("quotes from chapter 3", "in 1984") don't masquerade as counts.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_QUOTE_NOUNS = "quotes?|passages?|lines?|excerpts?|examples?|moments?|instances?|verses?"
# digit or number-word, then up to two intervening adjectives, then a quote noun.
_COUNT_RE = re.compile(
    rf"\b(\d+|{'|'.join(_NUMBER_WORDS)})\b\s+(?:\w+\s+){{0,2}}(?:{_QUOTE_NOUNS})\b",
    re.IGNORECASE,
)
# Strip a leading "chapter/ch./chap. N" so its number can't be read as a count
# ("chapter 7 quotes about warmth" → not 7). A count that precedes the chapter
# reference ("7 quotes from chapter 3") is left intact.
_CHAPTER_PREFIX_RE = re.compile(r"\b(?:chapter|chap\.?|ch\.?)\s*\d+", re.IGNORECASE)


def parse_requested_quote_count(question: str) -> int | None:
    """Extract an explicitly requested quote count, or None. Pure — no clamping."""
    cleaned = _CHAPTER_PREFIX_RE.sub(" ", question)
    m = _COUNT_RE.search(cleaned)
    if not m:
        return None
    token = m.group(1).lower()
    return _NUMBER_WORDS.get(token, None) if token.isalpha() else int(token)


def plan_quotes(requested: int | None) -> tuple[int, int]:
    """Return (want, n_sections). Positive count → widen; else today's defaults."""
    if requested and requested > 0:
        want = min(requested, _MAX_QUOTES)
        return want, min(_MAX_SECTIONS, want + _SECTION_MARGIN)
    return _DEFAULT_QUOTES, _MIN_SECTIONS


def select_quotes(results: list[ReadResult], want: int = 3) -> list[ReadResult]:
    """Return copies of *results* pruned to a deduped, globally-capped quote set.

    Quotes are considered across all results in order; duplicates (by normalised
    text — so straight/curly and whitespace variants collapse) are dropped, and at
    most *want* quotes survive in total. Each returned ``ReadResult`` is a fresh
    copy whose ``verbatim_quotes``/``char_offsets`` are rebuilt together (never
    filtered independently) to the kept subset for that result; some may end empty.
    Inputs are not mutated.
    """
    kept_norms: set[str] = set()
    remaining = want
    out: list[ReadResult] = []

    for r in results:
        pairs: list[tuple[str, tuple[int, int]]] = []
        for quote, offset in zip(r.verbatim_quotes, r.char_offsets):
            if remaining <= 0:
                break
            key = _normalize(quote)
            if key in kept_norms:
                continue
            kept_norms.add(key)
            pairs.append((quote, offset))
            remaining -= 1
        quotes = [q for q, _o in pairs]
        offsets = [o for _q, o in pairs]
        out.append(dataclasses.replace(r, verbatim_quotes=quotes, char_offsets=offsets))

    return out


def format_quote_answer(n: int, *, weak: bool = False) -> str:
    """Short framing line for a quote answer. Contains no quote text."""
    if n == 0:
        return "I couldn't find any passages that clearly speak to that."
    passages = "passage" if n == 1 else "passages"
    if weak:
        return f"The closest {passages} I found only touch on this loosely:"
    return f"Here {'is' if n == 1 else 'are'} {n} {passages} that speak to that:"
