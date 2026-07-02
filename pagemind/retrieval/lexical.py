"""Postgres FTS lexical search over sections_fts."""
from __future__ import annotations

import re
import uuid

import psycopg

# ── Soft-nudge detection ──────────────────────────────────────────────────────

_QUOTED_RE = re.compile(r'"[^"]+"')
_DATE_RE = re.compile(
    r'\b(\d{4}|january|february|march|april|may|june|july|august'
    r'|september|october|november|december|'
    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
    re.IGNORECASE,
)


def detect_nudges(query: str) -> dict[str, bool]:
    """Return soft-weight nudge signals for a query string.

    boost_lexical: query has quoted phrases or mid-sentence capitalized tokens
                   (likely proper names) → favour lexical ranking in RRF.
    boost_date:    query contains year, month name, or date pattern
                   → also pull sections from the dates index.
    """
    has_quoted = bool(_QUOTED_RE.search(query))
    words = query.split()
    has_capitalized = any(w[0].isupper() for w in words[1:] if w)
    has_date = bool(_DATE_RE.search(query))
    return {
        "boost_lexical": has_quoted or has_capitalized,
        "boost_date": has_date,
    }


# ── Lexical search ────────────────────────────────────────────────────────────

def lexical_search(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    query: str,
    *,
    top_k: int = 20,
    up_to_chapter: int | None = None,
) -> list[uuid.UUID]:
    """Return up to *top_k* section_ids ranked by ts_rank against *query*.

    Accepts an optional *up_to_chapter* ordinal: only sections whose chapter
    has ordinal <= that value are returned (plumbing for spoiler-safe reads).
    """
    scope_sql = ""
    params: list = [query, book_id]

    if up_to_chapter is not None:
        scope_sql = """
            AND sf.chapter_id IN (
                SELECT chapter_id FROM chapters
                WHERE book_id = %(book_id)s AND ordinal <= %(upto)s
            )
        """

    sql = f"""
        WITH tsq AS (
            SELECT websearch_to_tsquery('english', %(query)s) AS q
        )
        SELECT sf.section_id
        FROM sections_fts sf, tsq
        WHERE sf.book_id = %(book_id)s
          AND sf.fts_vector @@ tsq.q
          {scope_sql}
        ORDER BY ts_rank(sf.fts_vector, tsq.q) DESC
        LIMIT %(top_k)s
    """

    named_params: dict = {"query": query, "book_id": book_id, "top_k": top_k}
    if up_to_chapter is not None:
        named_params["upto"] = up_to_chapter

    rows = conn.execute(sql, named_params).fetchall()
    return [r[0] for r in rows]


def date_section_ids(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    query: str,
    *,
    top_k: int = 10,
    up_to_chapter: int | None = None,
) -> list[uuid.UUID]:
    """Return section_ids from the dates table whose raw_text matches *query* tokens.

    Used as a soft boost when the query looks date-like.
    """
    # Extract all tokens from the query that look like years or month names
    tokens = _DATE_RE.findall(query)
    if not tokens:
        return []

    scope_sql = ""
    named_params: dict = {"book_id": book_id, "top_k": top_k}
    if up_to_chapter is not None:
        scope_sql = """
            AND d.chapter_id IN (
                SELECT chapter_id FROM chapters
                WHERE book_id = %(book_id)s AND ordinal <= %(upto)s
            )
        """
        named_params["upto"] = up_to_chapter

    # Build ILIKE conditions for each token
    conditions = " OR ".join(
        f"d.raw_text ILIKE %(tok{i}s)s" for i, _ in enumerate(tokens)
    )
    for i, tok in enumerate(tokens):
        named_params[f"tok{i}s"] = f"%{tok}%"

    sql = f"""
        SELECT DISTINCT d.section_id
        FROM dates d
        WHERE d.book_id = %(book_id)s
          AND d.section_id IS NOT NULL
          AND ({conditions})
          {scope_sql}
        LIMIT %(top_k)s
    """

    rows = conn.execute(sql, named_params).fetchall()
    return [r[0] for r in rows]
