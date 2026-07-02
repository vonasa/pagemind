"""Split chapters into sections and chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pagemind.segment.tokens import (
    SECTION_CAP,
    cap_chars,
    chunk_chars,
    count_tokens,
    overlap_chars,
)

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


@dataclass
class SectionRecord:
    chapter_ordinal: int
    ordinal: int
    content: str
    char_offset_start: int   # byte offset within chapter text
    char_offset_end: int
    is_body: bool


@dataclass
class ChunkRecord:
    chapter_ordinal: int
    section_ordinal: int
    ordinal: int
    content: str
    char_offset_start: int   # byte offset within chapter text (same base as section)
    char_offset_end: int
    is_body: bool


# ── Section splitting ─────────────────────────────────────────────────────────

def _slice(text: str, start: int, end: int) -> tuple[str, int, int]:
    """Return (stripped_content, adjusted_start, adjusted_end) with accurate offsets."""
    raw = text[start:end]
    lstripped = raw.lstrip()
    rstripped = lstripped.rstrip()
    adj_start = start + (len(raw) - len(lstripped))
    adj_end = adj_start + len(rstripped)
    return rstripped, adj_start, adj_end


def _find_split_point(text: str, pos: int, end: int) -> int:
    """Find the best character position to split text at [pos, end)."""
    # Prefer paragraph break
    pb = text.rfind('\n\n', pos, end)
    if pb > pos + end // 4:
        return pb

    # Fall back to sentence boundary
    for punct in ('. ', '! ', '? '):
        sb = text.rfind(punct, pos, end)
        if sb > pos + end // 4:
            return sb + 2  # include the trailing space

    return end


def _split_large_para(
    text: str,
    base_offset: int,
    chapter_ordinal: int,
    is_body: bool,
    start_ordinal: int,
) -> list[SectionRecord]:
    """Sentence-fallback for a single paragraph that exceeds SECTION_CAP."""
    sentences = _SENTENCE_END.split(text)
    sections: list[SectionRecord] = []
    parts: list[str] = []
    token_acc = 0
    abs_pos = base_offset

    for sent in sentences:
        stokens = count_tokens(sent)
        if token_acc + stokens > SECTION_CAP and parts:
            content = ' '.join(parts)
            content = content.strip()
            if content:
                sections.append(SectionRecord(
                    chapter_ordinal=chapter_ordinal,
                    ordinal=start_ordinal + len(sections),
                    content=content,
                    char_offset_start=abs_pos,
                    char_offset_end=abs_pos + len(content),
                    is_body=is_body,
                ))
            abs_pos += len(content) + 1
            parts = []
            token_acc = 0

        parts.append(sent)
        token_acc += stokens

    if parts:
        content = ' '.join(parts).strip()
        if content:
            sections.append(SectionRecord(
                chapter_ordinal=chapter_ordinal,
                ordinal=start_ordinal + len(sections),
                content=content,
                char_offset_start=abs_pos,
                char_offset_end=abs_pos + len(content),
                is_body=is_body,
            ))

    return sections


def split_into_sections(
    chapter_text: str,
    chapter_ordinal: int,
    is_body: bool,
) -> list[SectionRecord]:
    """Split chapter_text into sections, each ≤ SECTION_CAP tokens.

    Split order: paragraph boundary → sentence boundary.
    Offsets are byte positions within chapter_text, satisfying:
        chapter_text[s.char_offset_start:s.char_offset_end] == s.content
    """
    if count_tokens(chapter_text) <= SECTION_CAP:
        content, s, e = _slice(chapter_text, 0, len(chapter_text))
        if not content:
            return []
        return [SectionRecord(
            chapter_ordinal=chapter_ordinal,
            ordinal=0,
            content=content,
            char_offset_start=s,
            char_offset_end=e,
            is_body=is_body,
        )]

    cap = cap_chars()
    sections: list[SectionRecord] = []
    pos = 0

    while pos < len(chapter_text):
        end = min(pos + cap, len(chapter_text))

        if end < len(chapter_text):
            end = _find_split_point(chapter_text, pos, end)

        chunk_text = chapter_text[pos:end]

        # Check if this single chunk still exceeds cap (e.g. one huge paragraph)
        if count_tokens(chunk_text) > SECTION_CAP:
            # Sentence-level fallback
            sub = _split_large_para(
                chunk_text, pos, chapter_ordinal, is_body, len(sections)
            )
            sections.extend(sub)
        else:
            content, s, e = _slice(chapter_text, pos, end)
            if content:
                sections.append(SectionRecord(
                    chapter_ordinal=chapter_ordinal,
                    ordinal=len(sections),
                    content=content,
                    char_offset_start=s,
                    char_offset_end=e,
                    is_body=is_body,
                ))

        pos = end

    return sections


# ── Chunk splitting ───────────────────────────────────────────────────────────

def split_into_chunks(section: SectionRecord) -> list[ChunkRecord]:
    """Split a section into overlapping chunks of ~CHUNK_TARGET tokens.

    Each chunk's char_offset_start/end is an absolute offset in the chapter
    text (same coordinate space as the section's own offsets).
    """
    text = section.content
    c_chars = chunk_chars()
    o_chars = overlap_chars()
    chunks: list[ChunkRecord] = []
    pos = 0

    while pos < len(text):
        end = min(pos + c_chars, len(text))

        if end < len(text):
            # Prefer paragraph break
            pb = text.rfind('\n\n', pos, end)
            if pb > pos + c_chars // 2:
                end = pb
            else:
                # Fall back to sentence boundary
                for punct in ('. ', '! ', '? '):
                    sb = text.rfind(punct, pos, end)
                    if sb > pos + c_chars // 2:
                        end = sb + 2
                        break

        chunk_text = text[pos:end].strip()
        if chunk_text:
            abs_start = section.char_offset_start + pos
            abs_end = section.char_offset_start + pos + len(text[pos:end].rstrip())
            chunks.append(ChunkRecord(
                chapter_ordinal=section.chapter_ordinal,
                section_ordinal=section.ordinal,
                ordinal=len(chunks),
                content=chunk_text,
                char_offset_start=abs_start,
                char_offset_end=abs_end,
                is_body=section.is_body,
            ))

        if end >= len(text):
            break
        # Advance with overlap: step back o_chars from where we ended
        pos = max(pos + 1, end - o_chars)

    return chunks
