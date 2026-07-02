"""Small-to-big: expand a chunk hit to its parent section text (capped)."""
from __future__ import annotations

import uuid

import psycopg

# ~2 K tokens at 4 chars/token — keeps a single expansion within the reader budget.
_DEFAULT_CHAR_CAP = 8_000


def expand_to_section(
    conn: psycopg.Connection,
    section_id: uuid.UUID,
    *,
    char_cap: int = _DEFAULT_CHAR_CAP,
) -> str | None:
    """Return the content of *section_id*, hard-capped at *char_cap* characters.

    Returns None if the section does not exist.
    Never returns a full chapter — the cap is a deliberate ceiling.
    """
    row = conn.execute(
        "SELECT content FROM sections WHERE section_id = %s",
        (section_id,),
    ).fetchone()
    if row is None:
        return None
    text: str = row[0]
    return text[:char_cap] if len(text) > char_cap else text
