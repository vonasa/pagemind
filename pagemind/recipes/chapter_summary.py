"""chapter_summary recipe: fetch precomputed full summary — no reader (ADR 0005)."""
from __future__ import annotations

import re
import uuid

import psycopg

from pagemind.models.chat import ChatClient
from pagemind.retrieval.structured import get_chapter
from pagemind.runtime.types import QueryResult

_CHAPTER_RE = re.compile(r"\b(?:chapter|ch\.?|chap\.?)\s*(\d+)\b", re.IGNORECASE)
_ORDINAL_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    re.IGNORECASE,
)
_ORDINAL_MAP = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}


def _parse_chapter_number(question: str) -> int | None:
    m = _CHAPTER_RE.search(question)
    if m:
        return int(m.group(1))
    m = _ORDINAL_RE.search(question)
    if m:
        return _ORDINAL_MAP.get(m.group(1).lower())
    # Bare digit?
    digits = re.findall(r"\b(\d+)\b", question)
    if digits:
        return int(digits[0])
    return None


async def run(
    conn: psycopg.Connection,
    chat: ChatClient,
    book_id: uuid.UUID,
    question: str,
    *,
    up_to_chapter: int | None = None,
) -> QueryResult:
    chapter_n = _parse_chapter_number(question)
    if chapter_n is None:
        return QueryResult(
            text="Could not determine which chapter to summarize. "
                 "Please specify a chapter number (e.g., 'chapter 3').",
            weak=True,
        )

    chapter = get_chapter(conn, book_id, chapter_n)
    if chapter is None:
        return QueryResult(
            text=f"Chapter {chapter_n} was not found in this book.", weak=True
        )

    summary = chapter.get("summary") or chapter.get("micro_summary") or "(no summary available)"
    title = chapter.get("title") or f"Chapter {chapter_n}"
    text = f"**{title}** (Chapter {chapter_n})\n\n{summary}"
    return QueryResult(text=text)
