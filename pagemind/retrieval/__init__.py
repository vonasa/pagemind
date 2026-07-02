"""Retrieval substrate: hybrid search + structured lookups."""
from __future__ import annotations

import uuid

import psycopg

from pagemind.models.embeddings import EmbeddingsClient
from pagemind.retrieval.expand import expand_to_section
from pagemind.retrieval.lexical import date_section_ids, detect_nudges, lexical_search
from pagemind.retrieval.rrf import rrf_fuse
from pagemind.retrieval.semantic import semantic_search
from pagemind.retrieval.structured import (
    get_chapter,
    lookup_dates,
    lookup_entities,
    lookup_events,
)

__all__ = [
    "hybrid_search",
    "expand_to_section",
    "get_chapter",
    "lookup_entities",
    "lookup_dates",
    "lookup_events",
    "semantic_search",
    "lexical_search",
    "rrf_fuse",
    "detect_nudges",
]

# RRF defaults
_K = 60
_DEFAULT_LEX_WEIGHT = 1.0
_BOOSTED_LEX_WEIGHT = 1.5
_SEM_WEIGHT = 1.0
_DATE_BOOST_WEIGHT = 0.5


async def hybrid_search(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    query: str,
    *,
    top_k: int = 10,
    up_to_chapter: int | None = None,
) -> list[tuple[uuid.UUID, float]]:
    """Hybrid retrieval: lexical + semantic fused with RRF, with soft nudges.

    Returns up to *top_k* (section_id, rrf_score) pairs in descending score order.

    Soft-weight nudges (not gates):
      - Capitalized/quoted tokens → boost lexical weight in RRF.
      - Date-like tokens          → add a third ranked list from the dates index.

    The *up_to_chapter* ordinal gates all three sub-queries identically; callers
    can leave it None (default) to search the whole book.
    """
    nudges = detect_nudges(query)

    # Lexical and semantic run independently; no dependency between them.
    lex_ids = lexical_search(
        conn, book_id, query, top_k=top_k * 2, up_to_chapter=up_to_chapter
    )

    client = EmbeddingsClient.from_config()
    query_vec = await client.embed_one(query)
    sem_ids = semantic_search(
        conn, book_id, query_vec, top_k=top_k * 2, up_to_chapter=up_to_chapter
    )

    lex_weight = _BOOSTED_LEX_WEIGHT if nudges["boost_lexical"] else _DEFAULT_LEX_WEIGHT
    ranked_lists: list[list[uuid.UUID]] = [lex_ids, sem_ids]
    weights: list[float] = [lex_weight, _SEM_WEIGHT]

    if nudges["boost_date"]:
        date_ids = date_section_ids(
            conn, book_id, query, top_k=top_k, up_to_chapter=up_to_chapter
        )
        if date_ids:
            ranked_lists.append(date_ids)
            weights.append(_DATE_BOOST_WEIGHT)

    fused = rrf_fuse(ranked_lists, k=_K, weights=weights)
    return fused[:top_k]
