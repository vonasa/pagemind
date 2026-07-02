"""Streaming orchestrator: yields SSE event strings for the /ask endpoint."""
from __future__ import annotations

import json
import uuid
from typing import AsyncGenerator

import httpx
import psycopg

from pagemind.models.chat import ChatClient
from pagemind.retrieval import hybrid_search
from pagemind.runtime.fallback import stream_summary_fallback
from pagemind.runtime.history import condense_question
from pagemind.runtime.quotes import select_quotes
from pagemind.runtime.reader import fan_out
from pagemind.runtime.router import route
from pagemind.runtime.synthesizer import answer_prompts as _answer_prompts, format_evidence as _format_evidence, build_output as _build_output
from pagemind.runtime.types import QueryResult

_NON_SEARCH_RECIPES = frozenset(("chapter_summary", "locate_entity", "structured_view"))
_TOP_SECTIONS = 3
_WANT_QUOTES = 3


def _result_to_dict(result: QueryResult) -> dict:
    return {
        "text": result.text,
        "quotes": [
            {
                "text": q.text,
                "citation": {
                    "book_id": str(q.citation.book_id),
                    "chapter": q.citation.chapter,
                    "section_id": str(q.citation.section_id),
                    "char_offset": q.citation.char_offset,
                },
            }
            for q in result.quotes
        ],
        "citations": [
            {
                "book_id": str(c.book_id),
                "chapter": c.chapter,
                "section_id": str(c.section_id),
                "char_offset": c.char_offset,
            }
            for c in result.citations
        ],
    }


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _emit_fallback(
    conn: psycopg.Connection,
    chat: ChatClient,
    book_id: uuid.UUID,
    question: str,
    *,
    up_to_chapter: int | None,
    grounded: bool,
    original: QueryResult | None,
) -> AsyncGenerator[str, None]:
    """Stream the summaries-only fallback as SSE token/done events."""
    yield _sse({"type": "step", "text": "Consulting chapter summaries…"})
    async for kind, payload in stream_summary_fallback(
        conn, chat, book_id, question,
        up_to_chapter=up_to_chapter, grounded=grounded, original=original,
    ):
        if kind == "token":
            yield _sse({"type": "token", "text": payload})
        else:  # "done"
            yield _sse({"type": "done", "result": _result_to_dict(payload)})


async def ask_stream(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    question: str,
    up_to_chapter: int | None = None,
    grounded: bool = True,
    history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE data lines for streaming the ask flow.

    When *history* (prior conversation turns) is present, the follow-up question
    is first condensed into a self-contained ``search_question`` that drives
    routing, retrieval, and synthesis — so pronouns resolve without injecting
    prior turns into the grounded synthesizer.

    Events:
      {"type": "step",  "text": "…"}     — progress update
      {"type": "token", "text": "…"}     — answer token chunk
      {"type": "done",  "result": {…}}   — final QueryResult dict
      {"type": "error", "text": "…"}     — error (terminal)
    """
    chat = ChatClient.from_config("query")

    try:
        if history:
            yield _sse({"type": "step", "text": "Rephrasing…"})
            search_question = await condense_question(chat, history, question)
        else:
            search_question = question

        yield _sse({"type": "step", "text": "Routing question…"})
        recipe = await route(chat, search_question)

        if recipe in _NON_SEARCH_RECIPES:
            # structured_view reads several scenes before answering (no token stream on
            # this path), so flag the longer wait; the others are quick lookups.
            step = "Reading the key scenes…" if recipe == "structured_view" else "Looking up…"
            yield _sse({"type": "step", "text": step})
            from pagemind.recipes import dispatch
            result = await dispatch(recipe, conn, chat, book_id, search_question, up_to_chapter=up_to_chapter)
            if result.weak:
                # Structured lookup found nothing usable — answer from summaries.
                async for ev in _emit_fallback(
                    conn, chat, book_id, search_question,
                    up_to_chapter=up_to_chapter, grounded=grounded, original=result,
                ):
                    yield ev
                return
            yield _sse({"type": "token", "text": result.text})
            yield _sse({"type": "done", "result": _result_to_dict(result)})
            return

        # Search-based path (fact_lookup, verbatim_quote, contextual_why, generic_fallback)
        yield _sse({"type": "step", "text": "Searching passages…"})
        hits = await hybrid_search(conn, book_id, search_question, top_k=5, up_to_chapter=up_to_chapter)
        section_ids = [sid for sid, _ in hits[:_TOP_SECTIONS]]

        if not section_ids:
            # No passages matched — answer from the chapter summaries instead.
            original = QueryResult(text="No relevant passages found for that question.", weak=True)
            async for ev in _emit_fallback(
                conn, chat, book_id, search_question,
                up_to_chapter=up_to_chapter, grounded=grounded, original=original,
            ):
                yield ev
            return

        n = len(section_ids)
        plural = "s" if n != 1 else ""
        yield _sse({"type": "step", "text": f"Reading {n} passage{plural}…"})
        results = await fan_out(conn, chat, section_ids, search_question, grounded=grounded)

        yield _sse({"type": "step", "text": "Writing answer…"})

        if not results:
            # Readers surfaced nothing usable — answer from the chapter summaries.
            original = QueryResult(text="No relevant passages found.", weak=True)
            async for ev in _emit_fallback(
                conn, chat, book_id, search_question,
                up_to_chapter=up_to_chapter, grounded=grounded, original=original,
            ):
                yield ev
            return

        # Prose is written from the full reader evidence; the quote *cards* are
        # deduped and capped separately so a thematic quote request shows a few
        # clean cards rather than every overlapping repeat. select_quotes copies —
        # it does not shrink the evidence the synthesizer reads.
        evidence = _format_evidence(results)
        sys_prompt, answer_tmpl = _answer_prompts(grounded)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": answer_tmpl.format(question=search_question, evidence=evidence)},
        ]

        full_text = ""
        async for chunk in chat.stream_complete(messages, max_tokens=512):
            full_text += chunk
            yield _sse({"type": "token", "text": chunk})

        final = _build_output(
            book_id, search_question, full_text.strip(), select_quotes(results, want=_WANT_QUOTES)
        )
        yield _sse({"type": "done", "result": _result_to_dict(final)})
    except (httpx.HTTPError, ConnectionError) as exc:
        # A backend (embedding server or LLM) is unreachable/erroring — surface a
        # terminal error event instead of dropping the SSE stream mid-flight.
        yield _sse({
            "type": "error",
            "text": "A model backend is unavailable. Check the embedding server "
                    "and LLM are running, then try again.",
        })
        return
