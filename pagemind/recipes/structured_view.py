"""structured_view recipe: entity co-occurrence map (rich schema deferred, ADR 0014)."""
from __future__ import annotations

import uuid
from collections import defaultdict

import psycopg

from pagemind.models.chat import ChatClient
from pagemind.runtime.types import QueryResult


def _get_all_entities(conn: psycopg.Connection, book_id: uuid.UUID) -> list[dict]:
    rows = conn.execute(
        """
        SELECT e.entity_id, e.name, e.entity_type,
               array_agg(o.section_id) FILTER (WHERE o.section_id IS NOT NULL) AS section_ids
        FROM entities e
        LEFT JOIN occurrences o ON o.entity_id = e.entity_id AND o.book_id = e.book_id
        WHERE e.book_id = %s
        GROUP BY e.entity_id, e.name, e.entity_type
        ORDER BY e.name
        """,
        (book_id,),
    ).fetchall()
    return [
        {"entity_id": r[0], "name": r[1], "entity_type": r[2], "section_ids": r[3] or []}
        for r in rows
    ]


def _build_cooccurrence(entities: list[dict]) -> list[tuple[str, str, int]]:
    """Return (entity_a, entity_b, shared_section_count) pairs, sorted by count desc."""
    section_to_entities: dict[uuid.UUID, list[str]] = defaultdict(list)
    for ent in entities:
        for sid in ent["section_ids"]:
            section_to_entities[sid].append(ent["name"])

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for names in section_to_entities.values():
        names_sorted = sorted(set(names))
        for i in range(len(names_sorted)):
            for j in range(i + 1, len(names_sorted)):
                pair_counts[(names_sorted[i], names_sorted[j])] += 1

    return sorted(
        ((a, b, c) for (a, b), c in pair_counts.items()),
        key=lambda t: -t[2],
    )


def _format_view(entities: list[dict], pairs: list[tuple[str, str, int]]) -> str:
    lines: list[str] = [f"Entities in book: {len(entities)}"]
    by_type: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        by_type[e["entity_type"]].append(e["name"])
    for etype, names in sorted(by_type.items()):
        lines.append(f"\n{etype}s: {', '.join(names[:20])}")

    if pairs:
        lines.append("\nTop co-occurrences (share scenes):")
        for a, b, count in pairs[:10]:
            lines.append(f"  {a} ↔ {b}: {count} shared section(s)")
    else:
        lines.append("\n(No co-occurrences found.)")
    return "\n".join(lines)


async def run(
    conn: psycopg.Connection,
    chat: ChatClient,
    book_id: uuid.UUID,
    question: str,
    *,
    up_to_chapter: int | None = None,
) -> QueryResult:
    entities = _get_all_entities(conn, book_id)
    if not entities:
        return QueryResult(
            text="No entities have been indexed for this book yet.", weak=True
        )

    pairs = _build_cooccurrence(entities)
    text = _format_view(entities, pairs)
    return QueryResult(text=text)
