"""Summaries-only fallback: answer from chapter summaries when a recipe dead-ends.

Fires when a recipe returns ``QueryResult.weak`` — a structured lookup that found
nothing (e.g. an entity with zero occurrences) or a search that found no passages.
Some questions ("where does the main character move at the start") are answered
plainly in a chapter summary but not in any single body section, and a section
search can even surface an unrelated passage. So this fallback feeds the ordered
in-scope chapter summaries to the synthesizer and lets it compose the answer.

Deliberately does NOT run ``hybrid_search``/``fan_out`` — no misleading citations,
no double work. The answer is uncited prose (the source is a summary, not a
quotable passage). Imports only leaf modules to stay clear of the pre-existing
recipes↔runtime import cycle.
"""
from __future__ import annotations

import uuid
from typing import AsyncGenerator

import psycopg

from pagemind.models.chat import ChatClient
from pagemind.retrieval.structured import get_chapter_summaries
from pagemind.runtime.synthesizer import answer_prompts, build_output
from pagemind.runtime.types import QueryResult

# Combined char budget for the outline block: full summaries in ordinal order
# until the budget is hit, then micro-summaries for any remaining in-scope
# chapters — so early chapters (which answer "beginning" questions) keep full
# detail while the whole in-scope range stays represented.
_OUTLINE_BUDGET = 8000
_MAX_TOKENS = 512


def _build_outline(chapters: list[dict]) -> str:
    """Render chapters into an evidence block within the combined char budget."""
    parts: list[str] = []
    used = 0
    for ch in chapters:
        number = ch["number"]
        title = ch.get("title") or f"Chapter {number}"
        full = (ch.get("summary") or "").strip()
        micro = (ch.get("micro_summary") or "").strip()
        body = full if (full and used + len(full) <= _OUTLINE_BUDGET) else micro
        if not body:
            continue
        parts.append(f"[Chapter {number} — {title}]\n{body}")
        used += len(body)
    return "\n\n".join(parts)


def _messages(question: str, grounded: bool, outline: str) -> list[dict]:
    sys_prompt, answer_tmpl = answer_prompts(grounded)
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": answer_tmpl.format(question=question, evidence=outline)},
    ]


def _no_summaries_result(
    original: QueryResult | None, chapter: int | None = None
) -> QueryResult:
    """When there are no summaries to consult, produce the right null text.

    Under an exact-chapter scope, be explicit that nothing was found *in that chapter*
    rather than silently widening to the whole book (see the zero-hit policy).
    """
    if chapter is not None:
        return QueryResult(
            text=f"No relevant passages found in chapter {chapter}.", weak=True
        )
    if original is not None:
        return original
    return QueryResult(text="No relevant passages found for that question.", weak=True)


async def summary_fallback(
    conn: psycopg.Connection,
    chat: ChatClient,
    book_id: uuid.UUID,
    question: str,
    *,
    up_to_chapter: int | None = None,
    chapter: int | None = None,
    grounded: bool = True,
    original: QueryResult | None = None,
) -> QueryResult:
    """Answer *question* from the in-scope chapter-summary outline (non-streaming).

    Under an exact *chapter* scope the outline is restricted to that chapter, so a
    zero-hit recipe answer never silently widens to the whole book. Returns *original*
    (or a chapter-specific null) when there are no summaries to consult.
    """
    chapters = get_chapter_summaries(conn, book_id, up_to_chapter=up_to_chapter, chapter=chapter)
    outline = _build_outline(chapters)
    if not outline:
        return _no_summaries_result(original, chapter)

    text = await chat.complete(_messages(question, grounded, outline), max_tokens=_MAX_TOKENS)
    return build_output(book_id, question, text.strip(), [])


async def stream_summary_fallback(
    conn: psycopg.Connection,
    chat: ChatClient,
    book_id: uuid.UUID,
    question: str,
    *,
    up_to_chapter: int | None = None,
    chapter: int | None = None,
    grounded: bool = True,
    original: QueryResult | None = None,
) -> AsyncGenerator[tuple[str, object], None]:
    """Streaming form: yields ("token", chunk) then a terminal ("done", QueryResult).

    The caller frames these into whatever transport it uses (SSE for the web API).
    """
    chapters = get_chapter_summaries(conn, book_id, up_to_chapter=up_to_chapter, chapter=chapter)
    outline = _build_outline(chapters)
    if not outline:
        yield ("done", _no_summaries_result(original, chapter))
        return

    full_text = ""
    async for chunk in chat.stream_complete(_messages(question, grounded, outline), max_tokens=_MAX_TOKENS):
        full_text += chunk
        yield ("token", chunk)
    yield ("done", build_output(book_id, question, full_text.strip(), []))
