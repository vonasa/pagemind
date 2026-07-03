"""Recipe registry: dispatch recipe name → run function."""
from __future__ import annotations

import uuid
from typing import Callable, Awaitable

import psycopg

from pagemind.models.chat import ChatClient
from pagemind.recipes import (
    chapter_summary,
    contextual_why,
    fact_lookup,
    generic_fallback,
    locate_entity,
    structured_view,
    verbatim_quote,
)
from pagemind.runtime.types import QueryResult

RecipeFn = Callable[
    [psycopg.Connection, ChatClient, uuid.UUID, str],
    Awaitable[QueryResult],
]

_REGISTRY: dict[str, RecipeFn] = {
    "fact_lookup": fact_lookup.run,
    "verbatim_quote": verbatim_quote.run,
    "locate_entity": locate_entity.run,
    "chapter_summary": chapter_summary.run,
    "contextual_why": contextual_why.run,
    "structured_view": structured_view.run,
    "generic_fallback": generic_fallback.run,
}


async def dispatch(
    recipe: str,
    conn: psycopg.Connection,
    chat: ChatClient,
    book_id: uuid.UUID,
    question: str,
    *,
    up_to_chapter: int | None = None,
    chapter: int | None = None,
) -> QueryResult:
    """Run named recipe; fall back to generic_fallback on unknown name.

    *up_to_chapter* (spoiler ceiling) and *chapter* (exact chapter to scope to) are
    forwarded uniformly to every recipe; recipes that don't scope simply ignore them.
    """
    fn = _REGISTRY.get(recipe, generic_fallback.run)
    return await fn(conn, chat, book_id, question, up_to_chapter=up_to_chapter, chapter=chapter)
