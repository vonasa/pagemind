"""verbatim_quote recipe: search locate → fan-out readers → return exact quotes."""
from __future__ import annotations

import uuid

import psycopg

from pagemind.models.chat import ChatClient
from pagemind.retrieval import hybrid_search
from pagemind.runtime.quotes import format_quote_answer, select_quotes
from pagemind.runtime.reader import fan_out
from pagemind.runtime.synthesizer import _build_output
from pagemind.runtime.types import QueryResult

# Read several top sections, not just the single best — thematic quote requests
# ("quotes about passion") draw from passages scattered across the book.
_TOP_SECTIONS = 5
_WANT_QUOTES = 3


async def run(
    conn: psycopg.Connection,
    chat: ChatClient,
    book_id: uuid.UUID,
    question: str,
    *,
    up_to_chapter: int | None = None,
) -> QueryResult:
    hits = await hybrid_search(conn, book_id, question, top_k=_TOP_SECTIONS, up_to_chapter=up_to_chapter)
    section_ids = [sid for sid, _ in hits[:_TOP_SECTIONS]]
    if not section_ids:
        return _build_output(book_id, question, format_quote_answer(0), [])

    focused = f"Find and copy the exact passages relevant to: {question}"
    results = await fan_out(conn, chat, section_ids, focused)

    # Dedupe + cap the verbatim quotes the readers surfaced; the QueryResult's
    # quote cards carry the text, so the answer body is only a short framing line.
    selected = select_quotes(results, want=_WANT_QUOTES)
    n = sum(len(r.verbatim_quotes) for r in selected)
    text = format_quote_answer(n, weak=n < 2)
    return _build_output(book_id, question, text, selected)
