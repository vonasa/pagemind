"""structured_view recipe: edge-ranked reader + events enrichment (ADR 0007 Level 3).

The deterministic co-occurrence graph ranks character relationships and, together with
the events index, selects a *bounded* set of sections to read. The reader fan-out
returns verbatim-quoted context for those sections; a two-part synthesis describes how
the principal characters connect and where events happen. The read budget is fixed
regardless of book size — this is an overview of the strongest connections, not
exhaustive coverage (whole-book relationship synthesis is the deferred global case in
[[0007-relationships-not-graphrag]]).
"""
from __future__ import annotations

import uuid
from collections import defaultdict

import psycopg

from pagemind.models.chat import ChatClient
from pagemind.retrieval.structured import lookup_events
from pagemind.runtime.reader import fan_out
from pagemind.runtime.synthesizer import build_output, format_evidence
from pagemind.runtime.types import QueryResult

_MAX_EDGES = 12   # strongest character/place pairs considered
_MAX_READS = 8    # sections actually read (bounds cost, size-invariant)
_MAX_EVENTS = 20  # precomputed scene summaries pulled for breadth

_READ_Q = (
    "Describe every relationship between the people in this passage: for each pair, "
    "state who they are to each other (family, spouse, parent and child, friend, "
    "stranger, employer, etc.) and how they interact. Also note any place or setting "
    "and what happens there."
)

_SYS = (
    "You are a literary assistant describing how the principal characters and places "
    "in a book relate. Use only the evidence provided; never invent relationships or "
    "events it does not support. Treat places as settings, never as characters."
)

_PROMPT = """\
QUESTION: {question}

EVIDENCE:
{evidence}

Using only the evidence, write a concise answer in two short labelled parts:
1. How the characters are connected — for the principal characters, who each is to the
   others and how they interact.
2. Where it happens — the significance of the notable places or settings.
Ground every claim in the evidence. Present these as the principal connections, not an
exhaustive list.\
"""


def _get_all_entities(conn: psycopg.Connection, book_id: uuid.UUID) -> list[dict]:
    rows = conn.execute(
        """
        SELECT e.entity_id, e.name, e.entity_type, e.aliases,
               array_agg(o.section_id) FILTER (WHERE o.section_id IS NOT NULL) AS section_ids
        FROM entities e
        LEFT JOIN occurrences o ON o.entity_id = e.entity_id AND o.book_id = e.book_id
        WHERE e.book_id = %s
        GROUP BY e.entity_id, e.name, e.entity_type, e.aliases
        ORDER BY e.name
        """,
        (book_id,),
    ).fetchall()
    return [
        {
            "entity_id": r[0],
            "name": r[1],
            "entity_type": r[2],
            "aliases": r[3] or [],
            "section_ids": r[4] or [],
        }
        for r in rows
    ]


# Edge: (name_a, name_b, shared_section_count, shared_section_ids)
Edge = tuple[str, str, int, frozenset[uuid.UUID]]


def _build_cooccurrence(
    entities: list[dict],
) -> tuple[list[Edge], dict[uuid.UUID, set[str]]]:
    """Deterministic co-occurrence graph over sections. Pure (no DB).

    Returns (edges, section_to_entities). Each edge carries the *deduped* set of
    sections its pair shares (an entity seen multiple times in one section counts
    once). Edges are sorted by shared-section count desc with a (name_a, name_b)
    tiebreak so ordering is stable for tests and output.
    """
    section_to_entities: dict[uuid.UUID, set[str]] = defaultdict(set)
    for ent in entities:
        for sid in set(ent["section_ids"]):
            section_to_entities[sid].add(ent["name"])

    pair_sections: dict[tuple[str, str], set[uuid.UUID]] = defaultdict(set)
    for sid, names in section_to_entities.items():
        names_sorted = sorted(names)
        for i in range(len(names_sorted)):
            for j in range(i + 1, len(names_sorted)):
                pair_sections[(names_sorted[i], names_sorted[j])].add(sid)

    edges: list[Edge] = sorted(
        ((a, b, len(sids), frozenset(sids)) for (a, b), sids in pair_sections.items()),
        key=lambda e: (-e[2], e[0], e[1]),
    )
    return edges, section_to_entities


def _select_read_sections(
    top_edges: list[Edge],
    section_to_entities: dict[uuid.UUID, set[str]],
    event_section_ids: list[uuid.UUID],
    max_reads: int,
) -> list[uuid.UUID]:
    """Bounded, size-invariant read set. Pure (no DB).

    Candidates are the sections backing the top edges plus the sections of the
    filtered events. Prioritise sections that are both edge-central and event-backed,
    then edge-central (by how many top edges they support, then entity density), then
    event-only. Capped at *max_reads*.
    """
    edge_support: dict[uuid.UUID, int] = defaultdict(int)
    for _a, _b, _c, sids in top_edges:
        for sid in sids:
            edge_support[sid] += 1
    event_set = set(event_section_ids)
    candidates = set(edge_support) | event_set

    def rank_key(sid: uuid.UUID) -> tuple:
        both = sid in edge_support and sid in event_set
        return (
            0 if both else 1,
            -edge_support.get(sid, 0),
            -len(section_to_entities.get(sid, ())),
            str(sid),
        )

    return sorted(candidates, key=rank_key)[:max_reads]


def _format_evidence(
    top_edges: list[Edge],
    type_map: dict[str, str],
    results: list,
    events: list[dict],
) -> str:
    """Assemble the synthesis evidence: typed connection strength + reader excerpts +
    labelled index scene summaries. Avoids the token 'PASSAGE' so the reader-vs-synth
    call branch stays unambiguous in tests."""

    def _is_pp(a: str, b: str) -> bool:
        return type_map.get(a) == "PERSON" and type_map.get(b) == "PERSON"

    pp = [(a, b, c) for a, b, c, _s in top_edges if _is_pp(a, b)]
    ppl = [(a, b, c) for a, b, c, _s in top_edges if not _is_pp(a, b)]

    strength: list[str] = ["CONNECTION STRENGTH (shared scenes):", "  Character-character:"]
    strength.extend(f"    {a} + {b}: {c}" for a, b, c in pp)
    if not pp:
        strength.append("    (none)")
    if ppl:
        strength.append("  Character-place:")
        strength.extend(f"    {a} + {b}: {c}" for a, b, c in ppl)

    parts = ["\n".join(strength), "SCENE READINGS:\n" + format_evidence(results)]
    if events:
        summaries = "\n".join(f"  - {ev['description']}" for ev in events)
        parts.append("SCENE SUMMARIES (from the index):\n" + summaries)
    return "\n\n".join(parts)


async def run(
    conn: psycopg.Connection,
    chat: ChatClient,
    book_id: uuid.UUID,
    question: str,
    *,
    up_to_chapter: int | None = None,
    chapter: int | None = None,  # accepted for dispatch uniformity; a relationship map
    # is inherently whole-book, so this recipe is not scoped to a single chapter.
) -> QueryResult:
    entities = _get_all_entities(conn, book_id)
    if not entities:
        return QueryResult(
            text="No entities have been indexed for this book yet.", weak=True
        )

    edges, section_to_entities = _build_cooccurrence(entities)
    if not edges:
        # Entities exist but never share a scene: nothing to describe. Defer to the
        # summary fallback rather than emitting a bare roster.
        return QueryResult(
            text="No connections between characters or places were found in the indexed text.",
            weak=True,
        )
    top_edges = edges[:_MAX_EDGES]
    type_map = {e["name"]: e["entity_type"] for e in entities}
    aliases_by_name = {e["name"]: e["aliases"] for e in entities}

    # Terms for filtering events to the principal characters — canonical name AND all
    # aliases (event descriptions use aliases, e.g. "Juliet" for canonical "Julietta").
    char_terms: set[str] = set()
    for a, b, _c, _s in top_edges:
        for name in (a, b):
            if type_map.get(name) == "PERSON":
                char_terms.add(name.lower())
                char_terms.update(al.lower() for al in aliases_by_name.get(name, []))

    events = lookup_events(conn, book_id, "", up_to_chapter=up_to_chapter, limit=_MAX_EVENTS)
    if char_terms:
        events = [
            ev for ev in events
            if any(t in ev["description"].lower() for t in char_terms)
        ]
    event_section_ids = [ev["section_id"] for ev in events if ev["section_id"] is not None]

    read_sections = _select_read_sections(
        top_edges, section_to_entities, event_section_ids, _MAX_READS
    )
    results = await fan_out(conn, chat, read_sections, _READ_Q) if read_sections else []
    usable = [r for r in results if r.answer and r.answer != "[section not found]"]
    if not usable:
        return QueryResult(
            text="Could not read the scenes behind these connections.", weak=True
        )

    evidence = _format_evidence(top_edges, type_map, usable, events)
    synthesis = await chat.complete(
        [
            {"role": "system", "content": _SYS},
            {"role": "user", "content": _PROMPT.format(question=question, evidence=evidence)},
        ],
        max_tokens=700,
    )
    return build_output(book_id, question, synthesis.strip(), usable)
