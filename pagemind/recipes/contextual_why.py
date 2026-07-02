"""contextual_why recipe: two-stage multi-hop via forward-scan + fan-out (ADR 0005)."""
from __future__ import annotations

import uuid

import psycopg

from pagemind.models.chat import ChatClient
from pagemind.retrieval import hybrid_search
from pagemind.runtime.reader import fan_out, read
from pagemind.runtime.synthesizer import synthesize
from pagemind.runtime.types import QueryResult

_FAN_OUT_CAP = 3


def _chapter_ordinal(conn: psycopg.Connection, section_id: uuid.UUID) -> int | None:
    row = conn.execute(
        """
        SELECT c.ordinal
        FROM sections s
        JOIN chapters c ON c.chapter_id = s.chapter_id
        WHERE s.section_id = %s
        """,
        (section_id,),
    ).fetchone()
    return row[0] if row else None


def _filter_to_forward(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    section_ids: list[uuid.UUID],
    from_ordinal: int,
) -> list[uuid.UUID]:
    """Keep only section_ids whose chapter ordinal > from_ordinal."""
    if not section_ids:
        return []
    rows = conn.execute(
        """
        SELECT s.section_id
        FROM sections s
        JOIN chapters c ON c.chapter_id = s.chapter_id
        WHERE s.section_id = ANY(%s)
          AND s.book_id = %s
          AND c.ordinal > %s
        """,
        (list(section_ids), book_id, from_ordinal),
    ).fetchall()
    # Preserve original order
    forward_set = {r[0] for r in rows}
    return [sid for sid in section_ids if sid in forward_set]


async def run(
    conn: psycopg.Connection,
    chat: ChatClient,
    book_id: uuid.UUID,
    question: str,
    *,
    up_to_chapter: int | None = None,
) -> QueryResult:
    # ── Stage 1: anchor ────────────────────────────────────────────────────────
    hits = await hybrid_search(conn, book_id, question, top_k=3, up_to_chapter=up_to_chapter)
    if not hits:
        return QueryResult(
            text="No relevant passages found for that question.", weak=True
        )

    anchor_id = hits[0][0]
    anchor_chapter = _chapter_ordinal(conn, anchor_id)
    if anchor_chapter is None:
        return QueryResult(text="Could not locate anchor section.", weak=True)

    read_1 = await read(conn, chat, anchor_id, question)

    # ── Stage 2: forward scan ─────────────────────────────────────────────────
    forward_query = read_1.answer  # use first-hop answer as new search signal
    fwd_hits = await hybrid_search(conn, book_id, forward_query, top_k=8)
    fwd_ids = [sid for sid, _ in fwd_hits]
    fwd_ids = _filter_to_forward(conn, book_id, fwd_ids, anchor_chapter)[:_FAN_OUT_CAP]

    forward_results = await fan_out(conn, chat, fwd_ids, question, max_concurrent=_FAN_OUT_CAP)

    all_results = [read_1] + [r for r in forward_results if r.answer]
    return await synthesize(chat, book_id, question, all_results)
