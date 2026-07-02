"""Embed body chunks via Infinity and record model metadata on book_meta."""
from __future__ import annotations

import uuid
from collections.abc import Callable

import psycopg

from pagemind.models.embeddings import EMBEDDING_DIM, _DEFAULT_MODEL, EmbeddingsClient
from pagemind.precompute.checkpoint import mark_stage_done, mark_unit

_BATCH_SIZE = 64

_STAGE = "embeddings"


def _vec_to_pg(vec: list[float]) -> str:
    """Serialise a float list to the PostgreSQL vector literal '[x,y,...]'."""
    return "[" + ",".join(f"{v:.8g}" for v in vec) + "]"


async def embed_chunks(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    *,
    progress: Callable[[str], None] | None = None,
) -> None:
    client = EmbeddingsClient.from_config()

    # Total body chunks for the counter; how many already embedded (resume baseline).
    total = conn.execute(
        "SELECT count(*) FROM chunks WHERE book_id = %s AND is_body",
        (book_id,),
    ).fetchone()[0]
    already = conn.execute(
        "SELECT count(*) FROM chunks WHERE book_id = %s AND is_body AND embedding IS NOT NULL",
        (book_id,),
    ).fetchone()[0]

    # Only unembedded chunks — the nullable `embedding` column is the resume marker,
    # robust even if _BATCH_SIZE changes between runs.
    rows = conn.execute(
        """
        SELECT chunk_id, content
        FROM chunks
        WHERE book_id = %s AND is_body AND embedding IS NULL
        ORDER BY ordinal
        """,
        (book_id,),
    ).fetchall()

    done = already
    for i in range(0, len(rows), _BATCH_SIZE):
        batch = rows[i : i + _BATCH_SIZE]
        chunk_ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]

        vectors = await client.embed(texts)

        for chunk_id, vec in zip(chunk_ids, vectors):
            conn.execute(
                "UPDATE chunks SET embedding = %s::halfvec(2048) WHERE chunk_id = %s",
                (_vec_to_pg(vec), chunk_id),
            )

        done += len(batch)
        mark_unit(conn, book_id, _STAGE, str(i))
        conn.commit()
        if progress:
            progress(f"Embedding {done}/{total} chunks …")

    # Record embedding model metadata. Idempotent across resume re-entries: set the
    # version to a stable value rather than blindly incrementing.
    conn.execute(
        """
        UPDATE book_meta
        SET embed_model   = %s,
            embed_dim     = %s,
            embed_version = greatest(embed_version, 1),
            updated_at    = now()
        WHERE book_id = %s
        """,
        (_DEFAULT_MODEL, EMBEDDING_DIM, book_id),
    )
    mark_stage_done(conn, book_id, _STAGE)
    conn.commit()
