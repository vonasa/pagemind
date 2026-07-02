"""Quote selection helpers shared by the verbatim recipe and the streaming path.

The reader fan-out can surface the same line from several overlapping sections or
return more quotes than a reader wants to show. ``select_quotes`` deduplicates and
caps them without disturbing the reader results the synthesizer prose is built
from, and ``format_quote_answer`` produces a short framing line (never the quote
text itself — quotes live in cards).
"""
from __future__ import annotations

import dataclasses

from pagemind.runtime.reader import _normalize
from pagemind.runtime.types import ReadResult


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
