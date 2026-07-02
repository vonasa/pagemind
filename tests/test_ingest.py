"""Unit tests for segmentation logic (no DB, no real EPUB required)."""

import io
import pytest

from pagemind.segment.tokens import count_tokens, SECTION_CAP, CHUNK_TARGET, CHUNK_OVERLAP
from pagemind.segment.split import split_into_sections, split_into_chunks, SectionRecord
from pagemind.segment.validate import validate
from pagemind.ingest.epub import RawChapter


# ── Token heuristic ───────────────────────────────────────────────────────────

def test_count_tokens_basic() -> None:
    assert count_tokens("a" * 4) == 1
    assert count_tokens("a" * 8) == 2
    assert count_tokens("") == 1  # min 1


def test_count_tokens_heuristic_order_of_magnitude() -> None:
    # 1000-word paragraph ≈ 5000 chars → should be ~1250 tokens
    text = ("word " * 1000)
    tokens = count_tokens(text)
    assert 1000 < tokens < 2000


# ── Section splitting ─────────────────────────────────────────────────────────

def _make_text(tokens: int) -> str:
    """Produce a text of approximately `tokens` tokens."""
    chars = tokens * 4
    para = "A" * 200
    parts = []
    while sum(len(p) + 2 for p in parts) < chars:
        parts.append(para)
    return "\n\n".join(parts)


def test_single_section_when_under_cap() -> None:
    text = _make_text(500)
    sections = split_into_sections(text, chapter_ordinal=0, is_body=True)
    assert len(sections) == 1
    assert sections[0].char_offset_start == 0
    assert sections[0].content == text.strip()


def test_splits_when_over_cap() -> None:
    text = _make_text(SECTION_CAP * 3)
    sections = split_into_sections(text, chapter_ordinal=0, is_body=True)
    assert len(sections) >= 2
    for s in sections:
        assert count_tokens(s.content) <= SECTION_CAP


def test_no_section_exceeds_cap() -> None:
    # Edge: text is exactly 2× cap
    text = _make_text(SECTION_CAP * 2)
    sections = split_into_sections(text, chapter_ordinal=0, is_body=True)
    for s in sections:
        assert count_tokens(s.content) <= SECTION_CAP, (
            f"section {s.ordinal} has {count_tokens(s.content)} tokens"
        )


def test_offsets_roundtrip() -> None:
    """chapter_text[start:end] == section.content for every section."""
    text = _make_text(SECTION_CAP * 2 + 100)
    sections = split_into_sections(text, chapter_ordinal=0, is_body=True)
    for s in sections:
        extracted = text[s.char_offset_start:s.char_offset_end]
        assert extracted == s.content, (
            f"section {s.ordinal}: offset slice doesn't match content"
        )


def test_is_body_propagated() -> None:
    text = _make_text(100)
    sections = split_into_sections(text, chapter_ordinal=0, is_body=False)
    assert all(not s.is_body for s in sections)


# ── Chunk splitting ───────────────────────────────────────────────────────────

def _make_section(content: str, ordinal: int = 0, chapter_ordinal: int = 0) -> SectionRecord:
    return SectionRecord(
        chapter_ordinal=chapter_ordinal,
        ordinal=ordinal,
        content=content,
        char_offset_start=0,
        char_offset_end=len(content),
        is_body=True,
    )


def test_chunks_overlap() -> None:
    # Make a section large enough to produce multiple chunks
    section = _make_section(_make_text(CHUNK_TARGET * 5))
    chunks = split_into_chunks(section)
    assert len(chunks) >= 2

    # Adjacent chunks should overlap: chunk[i].end > chunk[i+1].start
    for i in range(len(chunks) - 1):
        assert chunks[i].char_offset_end > chunks[i + 1].char_offset_start, (
            f"chunks {i} and {i+1} don't overlap"
        )


def test_chunk_offsets_roundtrip() -> None:
    """chapter_text[chunk.start:chunk.end] == chunk.content."""
    # The section content IS the chapter text in this test
    section = _make_section(_make_text(CHUNK_TARGET * 4))
    chapter_text = section.content

    chunks = split_into_chunks(section)
    for c in chunks:
        extracted = chapter_text[c.char_offset_start:c.char_offset_end]
        assert extracted.strip() == c.content, (
            f"chunk {c.ordinal}: offset slice doesn't match content"
        )


def test_single_chunk_for_small_section() -> None:
    section = _make_section("Short section. Only one chunk expected here.")
    chunks = split_into_chunks(section)
    assert len(chunks) == 1


# ── Validation gate ───────────────────────────────────────────────────────────

def _make_result(n_chapters: int = 5, sections_per: int = 2, token_size: int = 500):
    chapters = [RawChapter(ordinal=i, title=f'Ch {i}', text=_make_text(token_size), is_body=True) for i in range(n_chapters)]
    sections = []
    chunks = []
    for ch in chapters:
        secs = split_into_sections(ch.text, ch.ordinal, ch.is_body)
        for s in secs:
            chunks.extend(split_into_chunks(s))
        sections.extend(secs)
    from pagemind.segment import SegmentResult
    return SegmentResult(chapters=chapters, sections=sections, chunks=chunks)


def test_validation_passes_normal_book() -> None:
    result = _make_result()
    raw_length = sum(len(ch.text) for ch in result.chapters)
    passed, issues = validate(result.chapters, result.sections, result.chunks, raw_length)
    assert passed, f"Expected pass but got: {issues}"


def test_validation_fails_on_oversized_section() -> None:
    # Fabricate a section that exceeds the cap
    big_section = SectionRecord(
        chapter_ordinal=0, ordinal=0,
        content="x" * (SECTION_CAP * 4 * 2),  # 2× cap in tokens
        char_offset_start=0, char_offset_end=SECTION_CAP * 4 * 2,
        is_body=True,
    )
    chapter = RawChapter(ordinal=0, title='Ch 0', text=big_section.content, is_body=True)
    chunks = split_into_chunks(big_section)
    passed, issues = validate([chapter], [big_section], chunks, len(big_section.content))
    assert not passed
    assert any('cap' in i.lower() for i in issues)


def test_validation_fails_no_body_chunks() -> None:
    chapter = RawChapter(ordinal=0, title='Ch 0', text='Short text.', is_body=False)
    section = SectionRecord(
        chapter_ordinal=0, ordinal=0,
        content='Short text.',
        char_offset_start=0, char_offset_end=11,
        is_body=False,
    )
    from pagemind.segment.split import ChunkRecord
    chunk = ChunkRecord(
        chapter_ordinal=0, section_ordinal=0, ordinal=0,
        content='Short text.',
        char_offset_start=0, char_offset_end=11,
        is_body=False,
    )
    passed, issues = validate([chapter], [section], [chunk], 11)
    assert not passed
    assert any('chunk' in i.lower() for i in issues)


# ── EPUB parsing (requires ebooklib; uses in-memory book) ────────────────────

def _make_epub_book() -> bytes:
    """Create a minimal multi-chapter EPUB in memory."""
    import ebooklib
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title('Test Novel')
    book.add_author('Test Author')

    items = []
    for i in range(3):
        item = epub.EpubHtml(
            title=f'Chapter {i + 1}',
            file_name=f'chapter{i + 1}.xhtml',
            lang='en',
        )
        item.content = (
            f'<html><body><h1>Chapter {i + 1}</h1>'
            + '<p>' + ('Lorem ipsum dolor sit amet. ' * 50) + '</p>'
            + '</body></html>'
        ).encode()
        book.add_item(item)
        items.append(item)

    book.spine = [(item.id, True) for item in items]
    book.toc = [epub.Link(item.file_name, item.title, item.id) for item in items]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    buf = io.BytesIO()
    epub.write_epub(buf, book, {})
    return buf.getvalue()


def test_parse_epub_extracts_chapters(tmp_path) -> None:
    from pagemind.ingest import parse_epub

    epub_bytes = _make_epub_book()
    epub_path = tmp_path / 'test.epub'
    epub_path.write_bytes(epub_bytes)

    parsed = parse_epub(epub_path)
    assert parsed.title == 'Test Novel'
    assert parsed.author == 'Test Author'
    assert len(parsed.chapters) >= 2


def test_parse_epub_headingless_uses_fallback(tmp_path) -> None:
    """A single-file EPUB with no TOC should still segment via fallback ladder."""
    import ebooklib
    from ebooklib import epub
    from pagemind.ingest import parse_epub

    book = epub.EpubBook()
    book.set_title('Headingless')
    book.add_author('Author')

    # One big HTML file with no heading tags
    item = epub.EpubHtml(title='Content', file_name='content.xhtml', lang='en')
    item.content = (
        b'<html><body>'
        + (b'<p>' + b'Word ' * 200 + b'</p>') * 20
        + b'</body></html>'
    )
    book.add_item(item)
    book.spine = [(item.id, True)]
    book.toc = []
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    buf = io.BytesIO()
    epub.write_epub(buf, book, {})

    epub_path = tmp_path / 'headingless.epub'
    epub_path.write_bytes(buf.getvalue())

    parsed = parse_epub(epub_path)
    assert len(parsed.chapters) >= 1
    # Entire pipeline should still work
    from pagemind.segment import segment_book, validate as seg_validate
    result = segment_book(parsed)
    assert len(result.sections) >= 1
    assert len(result.chunks) >= 1


# ── Single-file EPUBs (Gutenberg layout): anchor split + blank-line robustness ──

def _write_epub(tmp_path, name, *, html: bytes, toc, spine_titles=None):
    """Build a single-content-file EPUB and return its parsed result."""
    import io
    from ebooklib import epub
    from pagemind.ingest import parse_epub

    book = epub.EpubBook()
    book.set_title(name)
    book.add_author('Author')
    item = epub.EpubHtml(title='Content', file_name='content.xhtml', lang='en')
    item.content = html
    book.add_item(item)
    book.spine = [(item.id, True)]
    book.toc = toc
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    buf = io.BytesIO()
    epub.write_epub(buf, book, {})
    path = tmp_path / f'{name}.epub'
    path.write_bytes(buf.getvalue())
    return parse_epub(path)


def test_parse_epub_single_file_with_toc_anchors(tmp_path) -> None:
    """Whole book in one HTML file with TOC #anchors → split at the anchors,
    not collapsed to a single chapter (regression for the 221-fragment bug)."""
    from ebooklib import epub

    prose = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. ' * 6
    html = (
        '<html><body>'
        '<h1 id="title">A Title Page</h1><p>by Someone</p>'
        f'<h2 id="c1">I</h2><p>{prose}</p>'
        f'<h2 id="c2">II</h2><p>{prose}</p>'
        f'<h2 id="c3">III</h2><p>{prose}</p>'
        '</body></html>'
    ).encode()
    toc = [
        epub.Link('content.xhtml#title', 'Title Page', 'title'),
        epub.Link('content.xhtml#c1', 'I', 'c1'),
        epub.Link('content.xhtml#c2', 'II', 'c2'),
        epub.Link('content.xhtml#c3', 'III', 'c3'),
    ]
    parsed = _write_epub(tmp_path, 'Anchored', html=html, toc=toc)

    body = [c for c in parsed.chapters if c.is_body]
    # The three real sections survive; the tiny title-page anchor is dropped.
    assert [c.title for c in body] == ['I', 'II', 'III']
    assert all('Lorem ipsum' in c.text for c in body)
    # Not exploded, not collapsed to one.
    assert 3 <= len(parsed.chapters) <= 5


def test_parse_epub_blank_line_heavy_no_explosion(tmp_path) -> None:
    """A blank-line-heavy single file with standalone Roman numerals must not
    shatter into one chapter per blank line (regression for the empty-match regex)."""
    from ebooklib import epub

    prose = 'The river ran wide and slow beneath the bridge that morning. ' * 5
    # Many <br> produce long runs of blank lines in the flattened text.
    blanks = '<br/>' * 30
    html = (
        '<html><body>'
        f'{blanks}<p>I</p>{blanks}<p>{prose}</p>'
        f'{blanks}<p>II</p>{blanks}<p>{prose}</p>'
        f'{blanks}<p>III</p>{blanks}<p>{prose}</p>'
        '</body></html>'
    ).encode()
    # No TOC → forces the heading-regex fallback (rung 3).
    parsed = _write_epub(tmp_path, 'Blanky', html=html, toc=[])

    assert len(parsed.chapters) <= 6, f"exploded into {len(parsed.chapters)} chapters"
    assert all(c.text.strip() for c in parsed.chapters)
    assert any('river ran wide' in c.text for c in parsed.chapters)


# ── Over-segmentation validation backstop (check 3b) ──────────────────────────

def _section(content: str, ordinal: int) -> SectionRecord:
    return SectionRecord(
        chapter_ordinal=ordinal, ordinal=ordinal, content=content,
        char_offset_start=0, char_offset_end=len(content), is_body=True,
    )


def test_validation_flags_over_segmentation() -> None:
    from pagemind.segment.split import ChunkRecord
    # 20 body sections, 12 of them tiny (<100 chars) → 60% > 40% → flagged.
    sections, chunks, chapters = [], [], []
    for i in range(20):
        content = 'x' * (10 if i < 12 else 400)
        s = _section(content, i)
        sections.append(s)
        chunks.append(ChunkRecord(
            chapter_ordinal=i, section_ordinal=i, ordinal=0, content=content,
            char_offset_start=0, char_offset_end=len(content), is_body=True,
        ))
        chapters.append(RawChapter(ordinal=i, title=f'{i}', text=content, is_body=True))
    raw_length = sum(len(s.content) for s in sections)
    passed, issues = validate(chapters, sections, chunks, raw_length)
    assert not passed
    assert any('over-segmented' in i.lower() for i in issues)


def test_validation_ignores_few_tiny_sections() -> None:
    """The count guard spares legitimately short books (a handful of small sections)."""
    from pagemind.segment.split import ChunkRecord
    sections, chunks, chapters = [], [], []
    for i in range(5):  # below the min-count guard
        content = 'x' * 20
        sections.append(_section(content, i))
        chunks.append(ChunkRecord(
            chapter_ordinal=i, section_ordinal=i, ordinal=0, content=content,
            char_offset_start=0, char_offset_end=len(content), is_body=True,
        ))
        chapters.append(RawChapter(ordinal=i, title=f'{i}', text=content, is_body=True))
    passed, issues = validate(chapters, sections, chunks, sum(len(s.content) for s in sections))
    assert not any('over-segmented' in i.lower() for i in issues)
