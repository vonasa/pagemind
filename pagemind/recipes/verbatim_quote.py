"""verbatim_quote recipe: search locate → fan-out readers → return exact quotes."""
from __future__ import annotations

import uuid

import psycopg

from pagemind.models.chat import ChatClient
from pagemind.retrieval import hybrid_search
from pagemind.runtime.quotes import (
    format_quote_answer,
    parse_requested_quote_count,
    plan_quotes,
    select_quotes,
)
from pagemind.runtime.reader import fan_out
from pagemind.runtime.synthesizer import _build_output
from pagemind.runtime.types import QueryResult


async def run(
    conn: psycopg.Connection,
    chat: ChatClient,
    book_id: uuid.UUID,
    question: str,
    *,
    up_to_chapter: int | None = None,
) -> QueryResult:
    # Read several top sections, not just the single best — thematic quote requests
    # ("quotes about passion") draw from passages scattered across the book. An
    # explicit "N quotes…" widens both the sections read and the cards kept; the
    # floor of 5 keeps this recipe's long-standing default when no count is given.
    want, planned = plan_quotes(parse_requested_quote_count(question))
    n_sections = max(5, planned)
    hits = await hybrid_search(conn, book_id, question, top_k=n_sections, up_to_chapter=up_to_chapter)
    section_ids = [sid for sid, _ in hits[:n_sections]]
    if not section_ids:
        return _build_output(book_id, question, format_quote_answer(0), [])

    focused = f"Find and copy the exact passages relevant to: {question}"
    results = await fan_out(conn, chat, section_ids, focused)

    # Dedupe + cap the verbatim quotes the readers surfaced; the QueryResult's
    # quote cards carry the text, so the answer body is only a short framing line.
    selected = select_quotes(results, want=want)
    n = sum(len(r.verbatim_quotes) for r in selected)
    text = format_quote_answer(n, weak=n < 2)
    return _build_output(book_id, question, text, selected)
