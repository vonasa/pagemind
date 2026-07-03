"""pgvector HNSW semantic search over chunk embeddings."""
from __future__ import annotations

import uuid

import psycopg

from pagemind.retrieval.scope import chapter_scope_clause, chapter_scope_params


def _vec_to_pg(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.8g}" for v in vec) + "]"


def semantic_search(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    query_vec: list[float],
    *,
    top_k: int = 20,
    up_to_chapter: int | None = None,
    chapter: int | None = None,
) -> list[uuid.UUID]:
    """Return up to *top_k* section_ids ranked by cosine similarity.

    Retrieves chunks via HNSW, then maps each chunk to its parent section,
    keeping the minimum distance per section (best-matching chunk wins).
    Fetches *top_k * 4* chunks internally to ensure enough unique sections
    survive the deduplication step. Scope (both on the display ``number``):
    *up_to_chapter* is a spoiler ceiling, *chapter* pins one exact chapter.
    """
    scope_sql = chapter_scope_clause("c", up_to_chapter=up_to_chapter, chapter=chapter)
    named_params: dict = {
        "vec": _vec_to_pg(query_vec),
        "book_id": book_id,
        "inner_k": top_k * 4,
        "top_k": top_k,
        **chapter_scope_params(up_to_chapter=up_to_chapter, chapter=chapter),
    }

    sql = f"""
        WITH ranked_chunks AS (
            SELECT
                c.section_id,
                c.embedding <=> %(vec)s::halfvec(2048) AS dist
            FROM chunks c
            WHERE c.book_id = %(book_id)s
              AND c.is_body
              AND c.embedding IS NOT NULL
              {scope_sql}
            ORDER BY dist
            LIMIT %(inner_k)s
        ),
        best_per_section AS (
            SELECT section_id, MIN(dist) AS dist
            FROM ranked_chunks
            GROUP BY section_id
        )
        SELECT section_id
        FROM best_per_section
        ORDER BY dist
        LIMIT %(top_k)s
    """

    rows = conn.execute(sql, named_params).fetchall()
    return [r[0] for r in rows]
