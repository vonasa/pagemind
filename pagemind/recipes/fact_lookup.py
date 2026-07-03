"""fact_lookup recipe: index/locate → fan-out readers → synth."""
from __future__ import annotations

import uuid

import psycopg

from pagemind.models.chat import ChatClient
from pagemind.retrieval import hybrid_search
from pagemind.runtime.reader import fan_out
from pagemind.runtime.synthesizer import synthesize
from pagemind.runtime.types import QueryResult

_TOP_SECTIONS = 3


async def run(
    conn: psycopg.Connection,
    chat: ChatClient,
    book_id: uuid.UUID,
    question: str,
    *,
    up_to_chapter: int | None = None,
    chapter: int | None = None,
) -> QueryResult:
    hits = await hybrid_search(
        conn, book_id, question, top_k=5, up_to_chapter=up_to_chapter, chapter=chapter
    )
    section_ids = [sid for sid, _ in hits[:_TOP_SECTIONS]]
    if not section_ids:
        return QueryResult(
            text="No relevant passages found for that question.", weak=True
        )

    results = await fan_out(conn, chat, section_ids, question)
    return await synthesize(chat, book_id, question, results)
