"""Single output seam for all recipe returns (ADR 0014)."""
from __future__ import annotations

from dataclasses import dataclass, field
import uuid


@dataclass
class Citation:
    book_id: uuid.UUID
    chapter: int
    section_id: uuid.UUID
    char_offset: int | None = None


@dataclass
class Quote:
    text: str
    citation: Citation


@dataclass
class ReadResult:
    section_id: uuid.UUID
    chapter: int
    answer: str
    verbatim_quotes: list[str] = field(default_factory=list)
    char_offsets: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class QueryResult:
    """Minimal envelope returned by every recipe."""
    text: str
    quotes: list[Quote] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    # A recipe sets weak=True when it could not actually answer the question
    # (a structured lookup with nothing, or a search that found no passages).
    # The orchestrator/streaming layer uses it to trigger the summary fallback.
    weak: bool = False
