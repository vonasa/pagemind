"""Precompute pipeline: summaries → book_summary → entities → embeddings → FTS.

Each stage records completed units in the `precompute_checkpoints` ledger and sets a
'*' completion sentinel, so an interrupted run auto-resumes: `run_precompute` skips any
stage whose sentinel is present, and each stage internally skips already-done units.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable

import psycopg

from pagemind.precompute.checkpoint import mark_stage_done, stage_done
from pagemind.precompute.embeddings import embed_chunks
from pagemind.precompute.entities import extract_entities
from pagemind.precompute.summaries import generate_book_summary, generate_summaries

# A progress callback receives a single human-readable message string.
Progress = Callable[[str], None]


async def run_precompute(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    *,
    on_stage: Progress | None = None,
    on_tick: Progress | None = None,
) -> None:
    """Run all precompute stages for *book_id*, resuming from the ledger.

    *on_stage* receives stage headlines; *on_tick* receives per-unit progress lines.
    Raises on the first failure; the caller sets book_meta.status = 'failed'.
    """

    def _stage(msg: str) -> None:
        if on_stage:
            on_stage(msg)

    # ── Summaries ────────────────────────────────────────────────────────────
    if stage_done(conn, book_id, "summaries"):
        _stage("✓ Summarising chapters (already done)")
    else:
        _stage("Summarising chapters …")
        await generate_summaries(conn, book_id, progress=on_tick)

    # ── Book-level summary ───────────────────────────────────────────────────
    if stage_done(conn, book_id, "book_summary"):
        _stage("✓ Summarising the book (already done)")
    else:
        _stage("Summarising the book …")
        await generate_book_summary(conn, book_id)
        mark_stage_done(conn, book_id, "book_summary")
        conn.commit()

    # ── Entities ─────────────────────────────────────────────────────────────
    if stage_done(conn, book_id, "entities"):
        _stage("✓ Extracting entities, dates, events (already done)")
    else:
        _stage("Extracting entities, dates, events …")
        await extract_entities(conn, book_id, progress=on_tick)

    # ── Embeddings ───────────────────────────────────────────────────────────
    if stage_done(conn, book_id, "embeddings"):
        _stage("✓ Embedding chunks (already done)")
    else:
        _stage("Embedding chunks …")
        await embed_chunks(conn, book_id, progress=on_tick)

    # ── Full-text search ─────────────────────────────────────────────────────
    if stage_done(conn, book_id, "fts"):
        _stage("✓ Populating full-text search index (already done)")
    else:
        _stage("Populating full-text search index …")
        _populate_fts(conn, book_id)
        mark_stage_done(conn, book_id, "fts")
        conn.commit()


def _populate_fts(conn: psycopg.Connection, book_id: uuid.UUID) -> None:
    conn.execute(
        """
        INSERT INTO sections_fts (section_id, book_id, chapter_id, fts_vector)
        SELECT s.section_id,
               s.book_id,
               s.chapter_id,
               to_tsvector('english', s.content)
        FROM sections s
        WHERE s.book_id = %s
        ON CONFLICT (section_id) DO UPDATE
          SET fts_vector = EXCLUDED.fts_vector
        """,
        (book_id,),
    )
