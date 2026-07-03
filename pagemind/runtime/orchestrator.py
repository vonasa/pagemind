"""Orchestrator: route → recipe → QueryResult, with round-cap guard (ADR 0004, 0005)."""
from __future__ import annotations

import uuid

import psycopg

from pagemind.config import settings
from pagemind.models.chat import ChatClient
from pagemind.recipes import dispatch
from pagemind.runtime.fallback import summary_fallback
from pagemind.runtime.router import route
from pagemind.runtime.scope import parse_chapter_reference
from pagemind.runtime.types import QueryResult


async def ask(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    question: str,
    *,
    up_to_chapter: int | None = None,
    max_rounds: int | None = None,
) -> QueryResult:
    """Answer *question* about *book_id* using the recipe engine.

    *max_rounds* defaults to settings.orchestrator_max_rounds; each round is one
    router+recipe invocation. Exceeding the cap returns a safe error result.
    """
    cap = max_rounds if max_rounds is not None else settings.orchestrator_max_rounds
    chat = ChatClient.from_config("query")

    # An explicit "chapter N" pins retrieval to that chapter's display number.
    chapter = parse_chapter_reference(question)

    for round_n in range(cap):
        recipe = await route(chat, question)
        result = await dispatch(
            recipe, conn, chat, book_id, question,
            up_to_chapter=up_to_chapter, chapter=chapter,
        )
        # A recipe that couldn't answer (structured lookup with nothing, or a
        # search that found no passages) falls back to the chapter summaries —
        # scoped to *chapter* when set, so a zero-hit answer never widens to the
        # whole book.
        if result.weak:
            return await summary_fallback(
                conn, chat, book_id, question,
                up_to_chapter=up_to_chapter, chapter=chapter, original=result,
            )
        # Recipes are deterministic DAGs — a single round always yields the answer.
        # The loop exists to bound any future retry/fallback extension.
        return result

    return QueryResult(
        text=f"Orchestrator exceeded the {cap}-round limit without producing an answer."
    )
