"""Database helpers for ingestion (sync psycopg3)."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Generator

import psycopg

from pagemind.config import settings
from pagemind.ingest.epub import ParsedBook
from pagemind.segment import SegmentResult


@contextmanager
def get_conn() -> Generator[psycopg.Connection, None, None]:
    with psycopg.connect(settings.database_url) as conn:
        yield conn


# ── Book-level operations ─────────────────────────────────────────────────────

def find_book_by_hash(conn: psycopg.Connection, source_hash: str) -> tuple[uuid.UUID, str] | None:
    """Return (book_id, status) if a book with this hash exists, else None."""
    row = conn.execute(
        "SELECT book_id, status FROM book_meta WHERE source_hash = %s",
        (source_hash,),
    ).fetchone()
    return (row[0], row[1]) if row else None


def create_book(conn: psycopg.Connection, parsed: ParsedBook) -> uuid.UUID:
    """Insert a new book_meta row and return its book_id."""
    row = conn.execute(
        """
        INSERT INTO book_meta (title, author, cover_mime, cover_data, source_hash, status)
        VALUES (%s, %s, %s, %s, %s, 'ingesting')
        RETURNING book_id
        """,
        (parsed.title, parsed.author, parsed.cover_mime, parsed.cover_data, parsed.source_hash),
    ).fetchone()
    return row[0]


def update_status(conn: psycopg.Connection, book_id: uuid.UUID, status: str) -> None:
    conn.execute(
        "UPDATE book_meta SET status = %s, updated_at = now() WHERE book_id = %s",
        (status, book_id),
    )


def clear_book_content(conn: psycopg.Connection, book_id: uuid.UUID) -> None:
    """Delete a book's derived content, keeping the book_meta row for retry.

    Deleting chapters cascades to sections, chunks, occurrences, events, dates and
    sections_fts. Entities and precompute_checkpoints reference book_meta directly
    (not chapters), so they are not covered by that cascade and must be cleared
    explicitly — otherwise a retried ingestion leaves orphaned rows from the previous
    partial run (and stale checkpoints would make resume skip real work).
    """
    conn.execute("DELETE FROM chapters WHERE book_id = %s", (book_id,))
    conn.execute("DELETE FROM entities WHERE book_id = %s", (book_id,))
    conn.execute("DELETE FROM precompute_checkpoints WHERE book_id = %s", (book_id,))


# ── Bulk insert ───────────────────────────────────────────────────────────────

def store_segments(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    result: SegmentResult,
) -> None:
    """Insert chapters, sections, and chunks within the current transaction."""
    # 1. Chapters → collect chapter_id per ordinal.
    #    `number` is the 1-based position among BODY chapters only (the human-facing
    #    chapter number shown in the UI and referenced in chat); NULL for non-body.
    chapter_id_map: dict[int, uuid.UUID] = {}
    body_number = 0
    for ch in result.chapters:
        if ch.is_body:
            body_number += 1
            number = body_number
        else:
            number = None
        row = conn.execute(
            """
            INSERT INTO chapters (book_id, ordinal, number, title, is_body)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING chapter_id
            """,
            (book_id, ch.ordinal, number, ch.title, ch.is_body),
        ).fetchone()
        chapter_id_map[ch.ordinal] = row[0]

    # 2. Sections → collect section_id per (chapter_ordinal, section_ordinal)
    section_id_map: dict[tuple[int, int], uuid.UUID] = {}
    for s in result.sections:
        chapter_id = chapter_id_map[s.chapter_ordinal]
        row = conn.execute(
            """
            INSERT INTO sections
              (book_id, chapter_id, ordinal, content, char_offset_start, char_offset_end, is_body)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING section_id
            """,
            (book_id, chapter_id, s.ordinal, s.content,
             s.char_offset_start, s.char_offset_end, s.is_body),
        ).fetchone()
        section_id_map[(s.chapter_ordinal, s.ordinal)] = row[0]

    # 3. Chunks — batch with executemany (no IDs needed until Phase 2)
    chunk_rows = [
        (
            book_id,
            chapter_id_map[c.chapter_ordinal],
            section_id_map[(c.chapter_ordinal, c.section_ordinal)],
            c.ordinal,
            c.content,
            c.char_offset_start,
            c.char_offset_end,
            c.is_body,
        )
        for c in result.chunks
    ]
    # Connection has no .executemany shortcut in psycopg3 — use a cursor.
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks
              (book_id, chapter_id, section_id, ordinal, content,
               char_offset_start, char_offset_end, is_body)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            chunk_rows,
        )
