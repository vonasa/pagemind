"""Checkpoint ledger for per-unit auto-resume of the precompute pipeline.

One `precompute_checkpoints` row records a completed unit of work; a row with
`unit_key = '*'` marks an entire stage as complete (a fast-forward marker).

These helpers are **commit-free** — the caller owns the transaction. `get_conn()`
connections are non-autocommit, and `conn.commit()` inside a `with conn.transaction()`
block raises `ProgrammingError`. Per-unit stages call `mark_unit` then `conn.commit()`
in their loop; the entities back-half runs inside `with conn.transaction()` and lets the
block commit on exit.
"""
from __future__ import annotations

import uuid

import psycopg
from psycopg.types.json import Jsonb

# Sentinel unit_key marking a stage fully complete.
STAGE_DONE = "*"


def stage_done(conn: psycopg.Connection, book_id: uuid.UUID, stage: str) -> bool:
    """True iff the stage's '*' completion sentinel is present."""
    row = conn.execute(
        """
        SELECT 1 FROM precompute_checkpoints
        WHERE book_id = %s AND stage = %s AND unit_key = %s
        """,
        (book_id, stage, STAGE_DONE),
    ).fetchone()
    return row is not None


def mark_stage_done(conn: psycopg.Connection, book_id: uuid.UUID, stage: str) -> None:
    """Upsert the '*' sentinel marking *stage* complete (does not commit)."""
    mark_unit(conn, book_id, stage, STAGE_DONE)


def done_units(conn: psycopg.Connection, book_id: uuid.UUID, stage: str) -> set[str]:
    """Return completed unit_keys for *stage*, excluding the '*' sentinel."""
    rows = conn.execute(
        """
        SELECT unit_key FROM precompute_checkpoints
        WHERE book_id = %s AND stage = %s AND unit_key <> %s
        """,
        (book_id, stage, STAGE_DONE),
    ).fetchall()
    return {r[0] for r in rows}


def load_payloads(
    conn: psycopg.Connection, book_id: uuid.UUID, stage: str
) -> dict[str, dict]:
    """Return {unit_key: payload} for *stage*, excluding the '*' sentinel.

    Used by the entities stage to reload per-section raw extraction on resume.
    psycopg3 returns JSONB columns as already-parsed Python objects.
    """
    rows = conn.execute(
        """
        SELECT unit_key, payload FROM precompute_checkpoints
        WHERE book_id = %s AND stage = %s AND unit_key <> %s AND payload IS NOT NULL
        """,
        (book_id, stage, STAGE_DONE),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def mark_unit(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    stage: str,
    unit_key: str,
    payload: dict | None = None,
) -> None:
    """Upsert a single completed-unit row (does not commit).

    `payload` is wrapped in `Jsonb(...)` — psycopg3 has no default dict→jsonb adapter.
    """
    conn.execute(
        """
        INSERT INTO precompute_checkpoints (book_id, stage, unit_key, payload, updated_at)
        VALUES (%s, %s, %s, %s, now())
        ON CONFLICT (book_id, stage, unit_key) DO UPDATE
          SET payload = EXCLUDED.payload, updated_at = now()
        """,
        (book_id, stage, unit_key, Jsonb(payload) if payload is not None else None),
    )


def clear(conn: psycopg.Connection, book_id: uuid.UUID) -> None:
    """Delete all ledger rows for *book_id* (does not commit).

    Required on reindex: the book_meta FK cascade does not fire because the
    book_meta row is kept on retry.
    """
    conn.execute("DELETE FROM precompute_checkpoints WHERE book_id = %s", (book_id,))
