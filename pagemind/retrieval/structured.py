"""Structured lookups: chapter, entities, dates, events."""
from __future__ import annotations

import uuid
from typing import Any

import psycopg

from pagemind.retrieval.scope import chapter_scope_clause, chapter_scope_params


# ── Chapter ───────────────────────────────────────────────────────────────────

def get_chapter(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    chapter_n: int,
) -> dict[str, Any] | None:
    """Return metadata for the body chapter whose display *number* is *chapter_n*.

    Keyed on the human-facing 1-based ``number`` (not the internal ``ordinal``), so a
    user typing "chapter 3" resolves to the chapter shown as "Chapter 3". Non-body
    chapters have NULL ``number`` and are never matched. Returns None if not found.
    """
    row = conn.execute(
        """
        SELECT chapter_id, ordinal, number, title, micro_summary, summary, is_body
        FROM chapters
        WHERE book_id = %s AND number = %s AND is_body
        """,
        (book_id, chapter_n),
    ).fetchone()
    if row is None:
        return None
    return {
        "chapter_id": row[0],
        "ordinal": row[1],
        "number": row[2],
        "title": row[3],
        "micro_summary": row[4],
        "summary": row[5],
        "is_body": row[6],
    }


def get_chapter_summaries(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    *,
    up_to_chapter: int | None = None,
    chapter: int | None = None,
) -> list[dict[str, Any]]:
    """Return in-scope body-chapter summaries ordered by ordinal.

    No keyword ranking: an abstract question rarely overlaps interpretive summary
    prose lexically, so the caller feeds the whole ordered outline to the LLM and
    lets it do the matching. ``is_body`` excludes non-body chapters (e.g. a table
    of contents, whose summary is NULL). Honours *up_to_chapter* (spoiler ceiling)
    and *chapter* (exact chapter). Scope predicates are on the display ``number``.

    This queries the ``chapters`` row directly (not via an occurrence table), so it
    uses direct ``number`` predicates rather than the shared IN-subquery helper.
    """
    scope_sql = ""
    named_params: dict = {"book_id": book_id}
    if up_to_chapter is not None:
        scope_sql += " AND number <= %(upto)s"
        named_params["upto"] = up_to_chapter
    if chapter is not None:
        scope_sql += " AND number = %(chapter)s"
        named_params["chapter"] = chapter

    rows = conn.execute(
        f"""
        SELECT ordinal, number, title, micro_summary, summary
        FROM chapters
        WHERE book_id = %(book_id)s
          AND is_body
          {scope_sql}
        ORDER BY ordinal
        """,
        named_params,
    ).fetchall()

    return [
        {"ordinal": r[0], "number": r[1], "title": r[2], "micro_summary": r[3], "summary": r[4]}
        for r in rows
    ]


# ── Entities ──────────────────────────────────────────────────────────────────

def lookup_entities(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    name: str,
    *,
    up_to_chapter: int | None = None,
    chapter: int | None = None,
) -> list[dict[str, Any]]:
    """Return entities whose canonical name or any alias matches *name* (case-insensitive).

    Each result includes the entity metadata and a list of occurrence section_ids
    within scope (*up_to_chapter* ceiling and/or exact *chapter*, both on ``number``).
    """
    rows = conn.execute(
        """
        SELECT entity_id, name, entity_type, aliases
        FROM entities
        WHERE book_id = %s
          AND (
              name ILIKE %s
              OR EXISTS (
                  SELECT 1 FROM unnest(aliases) a WHERE a ILIKE %s
              )
          )
        """,
        (book_id, f"%{name}%", f"%{name}%"),
    ).fetchall()

    scope_sql = chapter_scope_clause("o", up_to_chapter=up_to_chapter, chapter=chapter)
    scope_params = chapter_scope_params(up_to_chapter=up_to_chapter, chapter=chapter)

    results = []
    for entity_id, ename, etype, aliases in rows:
        occ_params: dict = {"book_id": book_id, "entity_id": entity_id, **scope_params}

        occ_rows = conn.execute(
            f"""
            SELECT o.section_id, o.context
            FROM occurrences o
            WHERE o.book_id = %(book_id)s
              AND o.entity_id = %(entity_id)s
              AND o.section_id IS NOT NULL
              {scope_sql}
            ORDER BY o.char_offset_start
            """,
            occ_params,
        ).fetchall()

        results.append(
            {
                "entity_id": entity_id,
                "name": ename,
                "entity_type": etype,
                "aliases": aliases or [],
                "occurrences": [
                    {"section_id": r[0], "context": r[1]} for r in occ_rows
                ],
            }
        )
    return results


# ── Dates ─────────────────────────────────────────────────────────────────────

def lookup_dates(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    query: str,
    *,
    up_to_chapter: int | None = None,
    chapter: int | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return date records whose raw_text contains *query* (case-insensitive).

    Also matches on ISO normalized_date when *query* looks like a year (4 digits).
    Honours *up_to_chapter* (ceiling) and exact *chapter*, both on ``number``.
    """
    scope_sql = chapter_scope_clause("d", up_to_chapter=up_to_chapter, chapter=chapter)
    named_params: dict = {
        "book_id": book_id,
        "pat": f"%{query}%",
        "limit": limit,
        **chapter_scope_params(up_to_chapter=up_to_chapter, chapter=chapter),
    }

    year_clause = ""
    if query.strip().isdigit() and len(query.strip()) == 4:
        year_clause = "OR EXTRACT(YEAR FROM d.normalized_date)::text = %(year)s"
        named_params["year"] = query.strip()

    rows = conn.execute(
        f"""
        SELECT d.date_id, d.section_id, d.raw_text, d.normalized_date, d.context
        FROM dates d
        WHERE d.book_id = %(book_id)s
          AND (d.raw_text ILIKE %(pat)s {year_clause})
          {scope_sql}
        ORDER BY d.normalized_date NULLS LAST, d.char_offset_start
        LIMIT %(limit)s
        """,
        named_params,
    ).fetchall()

    return [
        {
            "date_id": r[0],
            "section_id": r[1],
            "raw_text": r[2],
            "normalized_date": r[3],
            "context": r[4],
        }
        for r in rows
    ]


# ── Events ────────────────────────────────────────────────────────────────────

def lookup_events(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    query: str,
    *,
    up_to_chapter: int | None = None,
    chapter: int | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return events whose description contains *query* (case-insensitive).

    Honours *up_to_chapter* (ceiling) and exact *chapter*, both on ``number``.
    """
    scope_sql = chapter_scope_clause("ev", up_to_chapter=up_to_chapter, chapter=chapter)
    named_params: dict = {
        "book_id": book_id,
        "pat": f"%{query}%",
        "limit": limit,
        **chapter_scope_params(up_to_chapter=up_to_chapter, chapter=chapter),
    }

    rows = conn.execute(
        f"""
        SELECT ev.event_id, ev.section_id, ev.description, ev.event_date, ev.entity_ids
        FROM events ev
        WHERE ev.book_id = %(book_id)s
          AND ev.description ILIKE %(pat)s
          {scope_sql}
        ORDER BY ev.char_offset_start
        LIMIT %(limit)s
        """,
        named_params,
    ).fetchall()

    return [
        {
            "event_id": r[0],
            "section_id": r[1],
            "description": r[2],
            "event_date": r[3],
            "entity_ids": r[4] or [],
        }
        for r in rows
    ]
