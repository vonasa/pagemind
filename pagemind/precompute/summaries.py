"""Generate two-tier chapter summaries (micro ~20 tok, full ~300-500 tok)."""
from __future__ import annotations

import uuid
from collections.abc import Callable

import psycopg

from pagemind.models.chat import ChatClient
from pagemind.precompute.checkpoint import done_units, mark_stage_done, mark_unit

_SYS = (
    "You are a concise literary analyst. "
    "Reply only with the requested text — no preamble, no labels, no markdown."
)

_MICRO_PROMPT = (
    "Summarize the following chapter in 15–20 words, capturing the main action:\n\n{content}"
)

_FULL_PROMPT = (
    "Write a 300–500 word summary of the following chapter. "
    "Cover: key events, characters present, and narrative significance.\n\n{content}"
)

# Roughly 3 K tokens — enough orientation without overrunning small-context local models.
_CONTENT_CAP = 12_000

_STAGE = "summaries"

# ── Book-level summary ────────────────────────────────────────────────────────

_BOOK_PROMPT = (
    "The following are per-chapter summaries of a book, in order. Write a single "
    "200–400 word overview of the whole book: what it is about, its main characters "
    "or subjects, and its overall arc. Do not enumerate chapters.\n\n{content}"
)

# Char budget for the concatenated chapter summaries fed into the book reduce. Generous
# enough for typical books; very long books are truncated rather than hierarchically
# reduced (deferred until one actually overflows). ~50 K chars ≈ 12–15 K tokens.
_BOOK_INPUT_CAP = 50_000

_BOOK_STAGE = "book_summary"


async def generate_book_summary(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    *,
    axis: str = "index",
) -> str:
    """Reduce the per-chapter full summaries into one whole-book overview.

    Reads body chapters' `summary` (in order), sends a single reduce call, and stores
    the result in `book_meta.summary`. Idempotent; returns the generated text.
    """
    client = ChatClient.from_config(axis=axis)

    rows = conn.execute(
        """
        SELECT c.title, c.summary
        FROM chapters c
        WHERE c.book_id = %s AND c.is_body AND c.summary IS NOT NULL
        ORDER BY c.ordinal
        """,
        (book_id,),
    ).fetchall()

    parts = [
        f"## {(title or 'Untitled')}\n{summary.strip()}"
        for title, summary in rows
        if summary and summary.strip()
    ]
    content = "\n\n".join(parts)[:_BOOK_INPUT_CAP]

    overview = ""
    if content:
        overview = (
            await client.complete(
                [
                    {"role": "system", "content": _SYS},
                    {"role": "user", "content": _BOOK_PROMPT.format(content=content)},
                ],
                max_tokens=600,
            )
        ).strip()

    conn.execute(
        "UPDATE book_meta SET summary = %s WHERE book_id = %s",
        (overview, book_id),
    )
    conn.commit()
    return overview


async def generate_summaries(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    *,
    progress: Callable[[str], None] | None = None,
) -> None:
    client = ChatClient.from_config(axis="index")

    # Total body chapters (for the progress counter) and the already-done set.
    total = conn.execute(
        """
        SELECT count(DISTINCT c.chapter_id)
        FROM chapters c
        JOIN sections s ON s.chapter_id = c.chapter_id
        WHERE c.book_id = %s AND c.is_body AND s.is_body
        """,
        (book_id,),
    ).fetchone()[0]
    done = done_units(conn, book_id, _STAGE)

    chapters = conn.execute(
        """
        SELECT c.chapter_id,
               c.title,
               string_agg(s.content, E'\\n\\n' ORDER BY s.ordinal) AS body
        FROM chapters c
        JOIN sections s ON s.chapter_id = c.chapter_id
        WHERE c.book_id = %s AND c.is_body AND s.is_body
        GROUP BY c.chapter_id, c.ordinal, c.title
        ORDER BY c.ordinal
        """,
        (book_id,),
    ).fetchall()

    base = total - len([c for c in chapters if str(c[0]) not in done])
    i = base
    for chapter_id, title, body in chapters:
        if str(chapter_id) in done:
            continue
        if not body:
            # Nothing to summarise, but mark it done so resume doesn't retry it.
            mark_unit(conn, book_id, _STAGE, str(chapter_id))
            conn.commit()
            continue

        i += 1
        snippet = body[:_CONTENT_CAP]
        label = title or "(untitled)"
        if progress:
            progress(f"Summarising chapter {i}/{total}: {label} …")

        micro = await client.complete(
            [
                {"role": "system", "content": _SYS},
                {"role": "user", "content": _MICRO_PROMPT.format(content=snippet)},
            ],
            max_tokens=60,
        )
        full = await client.complete(
            [
                {"role": "system", "content": _SYS},
                {"role": "user", "content": _FULL_PROMPT.format(content=snippet)},
            ],
            max_tokens=700,
        )

        conn.execute(
            "UPDATE chapters SET micro_summary = %s, summary = %s WHERE chapter_id = %s",
            (micro.strip(), full.strip(), chapter_id),
        )
        mark_unit(conn, book_id, _STAGE, str(chapter_id))
        conn.commit()

    mark_stage_done(conn, book_id, _STAGE)
    conn.commit()
