"""locate_entity recipe: pure index query, often no LLM (ADR 0005)."""
from __future__ import annotations

import re
import uuid

import psycopg

from pagemind.models.chat import ChatClient
from pagemind.retrieval.structured import lookup_entities
from pagemind.runtime.types import Citation, QueryResult

_STRIP_PREFIXES = re.compile(
    r"^(where\s+(does|is|do|did|can|are)\s+|find\s+(all\s+)?(occurrences\s+of\s+)?|"
    r"locate\s+|show\s+(me\s+)?|list\s+)",
    re.IGNORECASE,
)
_STRIP_SUFFIXES = re.compile(
    r"\s+(appear(s)?(\s+in)?|occur(s)?(\s+in)?|in\s+the\s+book|locations?|mentions?|"
    r"scenes?|chapters?|sections?|times?).*$",
    re.IGNORECASE,
)
_CAPITALIZED_RE = re.compile(r"\b[A-Z][a-zA-Z'-]+\b")


def _extract_entity_name(question: str) -> str:
    """Heuristically extract the entity name from a locate_entity query."""
    s = _STRIP_PREFIXES.sub("", question).strip()
    s = _STRIP_SUFFIXES.sub("", s).strip()
    # If result is long (> 4 words), try extracting capitalized tokens
    words = s.split()
    if len(words) > 4:
        caps = _CAPITALIZED_RE.findall(s)
        if caps:
            return " ".join(caps)
    return s or question


def _format_occurrences(name: str, entities: list[dict]) -> str:
    if not entities:
        return f"No entity named '{name}' was found in the index."

    lines: list[str] = []
    for ent in entities:
        occs = ent["occurrences"]
        lines.append(
            f"{ent['name']} ({ent['entity_type']}): {len(occs)} occurrence(s)"
        )
        for o in occs[:5]:
            ctx = (o.get("context") or "").strip()
            if ctx:
                lines.append(f"  • {ctx[:120]}")
    return "\n".join(lines)


async def run(
    conn: psycopg.Connection,
    chat: ChatClient,
    book_id: uuid.UUID,
    question: str,
    *,
    up_to_chapter: int | None = None,
) -> QueryResult:
    entity_name = _extract_entity_name(question)
    entities = lookup_entities(conn, book_id, entity_name, up_to_chapter=up_to_chapter)
    text = _format_occurrences(entity_name, entities)

    citations: list[Citation] = []
    seen: set[uuid.UUID] = set()
    for ent in entities:
        for occ in ent["occurrences"]:
            sid = occ["section_id"]
            if sid and sid not in seen:
                # chapter is unknown here without an extra query; use 0 as placeholder
                citations.append(Citation(book_id=book_id, chapter=0, section_id=sid))
                seen.add(sid)

    # No citations means the structured index had nothing usable for this query
    # (no matching entity, or a matched entity with zero in-scope occurrences).
    # Flag it so the orchestrator/streaming layer can fall back to summaries.
    return QueryResult(text=text, citations=citations, weak=not citations)
