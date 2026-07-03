"""Postgres FTS lexical search over sections_fts."""
from __future__ import annotations

import re
import uuid

import psycopg

from pagemind.retrieval.scope import chapter_scope_clause, chapter_scope_params

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
    chapter: int | None = None,
) -> list[uuid.UUID]:
    """Return up to *top_k* section_ids ranked by ts_rank against *query*.

    Scope (both on the display ``number``): *up_to_chapter* is a spoiler ceiling,
    *chapter* pins one exact chapter. Non-body sections are excluded — their chapter
    has no ``number`` and the table of contents should never be cited — via a join to
    ``sections`` (``sections_fts`` carries no ``is_body`` column).
    """
    scope_sql = chapter_scope_clause("sf", up_to_chapter=up_to_chapter, chapter=chapter)

    sql = f"""
        WITH tsq AS (
            SELECT websearch_to_tsquery('english', %(query)s) AS q
        )
        SELECT sf.section_id
        FROM sections_fts sf
        JOIN sections s ON s.section_id = sf.section_id
        CROSS JOIN tsq
        WHERE sf.book_id = %(book_id)s
          AND s.is_body
          AND sf.fts_vector @@ tsq.q
          {scope_sql}
        ORDER BY ts_rank(sf.fts_vector, tsq.q) DESC
        LIMIT %(top_k)s
    """

    named_params: dict = {
        "query": query,
        "book_id": book_id,
        "top_k": top_k,
        **chapter_scope_params(up_to_chapter=up_to_chapter, chapter=chapter),
    }

    rows = conn.execute(sql, named_params).fetchall()
    return [r[0] for r in rows]


def date_section_ids(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    query: str,
    *,
    top_k: int = 10,
    up_to_chapter: int | None = None,
    chapter: int | None = None,
) -> list[uuid.UUID]:
    """Return section_ids from the dates table whose raw_text matches *query* tokens.

    Used as a soft boost when the query looks date-like. Honours *up_to_chapter*
    (ceiling) and exact *chapter*, both on the display ``number``.
    """
    # Extract all tokens from the query that look like years or month names
    tokens = _DATE_RE.findall(query)
    if not tokens:
        return []

    scope_sql = chapter_scope_clause("d", up_to_chapter=up_to_chapter, chapter=chapter)
    named_params: dict = {
        "book_id": book_id,
        "top_k": top_k,
        **chapter_scope_params(up_to_chapter=up_to_chapter, chapter=chapter),
    }

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
