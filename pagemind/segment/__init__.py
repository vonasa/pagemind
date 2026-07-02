"""Segment a ParsedBook into chapters, sections, and chunks."""

from __future__ import annotations

from dataclasses import dataclass, field

from pagemind.ingest.epub import ParsedBook, RawChapter
from pagemind.segment.split import ChunkRecord, SectionRecord, split_into_chunks, split_into_sections
from pagemind.segment.validate import validate as _validate


@dataclass
class SegmentResult:
    chapters: list[RawChapter]
    sections: list[SectionRecord]
    chunks: list[ChunkRecord]


def segment_book(parsed: ParsedBook) -> SegmentResult:
    """Produce sections and chunks for every chapter in parsed."""
    sections: list[SectionRecord] = []
    chunks: list[ChunkRecord] = []

    for chapter in parsed.chapters:
        chapter_sections = split_into_sections(
            chapter.text,
            chapter.ordinal,
            chapter.is_body,
        )
        for section in chapter_sections:
            chunks.extend(split_into_chunks(section))
        sections.extend(chapter_sections)

    return SegmentResult(
        chapters=parsed.chapters,
        sections=sections,
        chunks=chunks,
    )


def validate(result: SegmentResult, raw_length: int) -> tuple[bool, list[str]]:
    return _validate(result.chapters, result.sections, result.chunks, raw_length)


__all__ = [
    "SegmentResult",
    "segment_book",
    "validate",
    "SectionRecord",
    "ChunkRecord",
]
