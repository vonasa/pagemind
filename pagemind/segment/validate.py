"""Validation gate for ingested books ([0009])."""

from __future__ import annotations

from pagemind.segment.split import ChunkRecord, SectionRecord
from pagemind.segment.tokens import SECTION_CAP, count_tokens
from pagemind.ingest.epub import RawChapter

# Over-segmentation guard (check 3b): flag a book only when it has at least this
# many body sections AND more than the allowed fraction of them are tiny.
OVERSEG_MIN_SECTIONS = 10
OVERSEG_TINY_CHARS = 100
OVERSEG_MAX_TINY_FRAC = 0.40


def validate(
    chapters: list[RawChapter],
    sections: list[SectionRecord],
    chunks: list[ChunkRecord],
    raw_length: int,
) -> tuple[bool, list[str]]:
    """Run the validation gate.  Returns (passed, issues)."""
    issues: list[str] = []

    # 1. Plausible chapter count
    body_chapters = [c for c in chapters if c.is_body]
    synthetic = any(c.is_synthetic for c in chapters)
    if not synthetic and len(body_chapters) < 1:
        issues.append(f"Too few body chapters: {len(body_chapters)}")
    if len(chapters) > 10_000:
        issues.append(f"Implausibly many chapters: {len(chapters)}")

    # 2. No section exceeds CAP
    oversized = [s for s in sections if count_tokens(s.content) > SECTION_CAP]
    if oversized:
        worst = max(oversized, key=lambda s: len(s.content))
        issues.append(
            f"{len(oversized)} section(s) exceed {SECTION_CAP}-token cap; "
            f"worst is {count_tokens(worst.content)} tokens in "
            f"chapter {worst.chapter_ordinal}, section {worst.ordinal}"
        )

    # 3. Non-pathological size distribution among body sections
    body_sections = [s for s in sections if s.is_body]
    if body_sections:
        sizes = [len(s.content) for s in body_sections]
        min_size, max_size = min(sizes), max(sizes)
        if min_size > 0 and max_size / min_size > 500:
            issues.append(
                f"Pathological section size distribution: "
                f"min={min_size} chars, max={max_size} chars (ratio {max_size // min_size}×)"
            )

    # 3b. Over-segmentation: many body sections, a large fraction of them tiny.
    # Catches catastrophic chapter shattering (e.g. a heading regex that matches
    # blank lines producing hundreds of one-line "chapters") that the max/min ratio
    # check above can slip past — for the original broken book that ratio was only
    # 338× yet 47% of sections were under 100 chars. The min-count guard keeps
    # legitimately short books (a handful of small sections) from being flagged.
    if len(body_sections) >= OVERSEG_MIN_SECTIONS:
        tiny = sum(1 for s in body_sections if len(s.content) < OVERSEG_TINY_CHARS)
        frac = tiny / len(body_sections)
        if frac > OVERSEG_MAX_TINY_FRAC:
            issues.append(
                f"Over-segmented: {tiny}/{len(body_sections)} body sections under "
                f"{OVERSEG_TINY_CHARS} chars = {frac:.0%} (expected ≤{OVERSEG_MAX_TINY_FRAC:.0%})"
            )

    # 4. Reconstructed length ≈ source length (within 25%)
    reconstructed = sum(len(s.content) for s in sections)
    if raw_length > 0:
        ratio = reconstructed / raw_length
        if not (0.75 <= ratio <= 1.25):
            issues.append(
                f"Reconstructed text length {reconstructed} deviates from "
                f"source {raw_length} by {abs(1 - ratio):.0%} (expected ≤25%)"
            )

    # 5. At least some chunks were produced for body content
    body_chunks = [c for c in chunks if c.is_body]
    if not body_chunks:
        issues.append("No body chunks produced")

    return len(issues) == 0, issues
