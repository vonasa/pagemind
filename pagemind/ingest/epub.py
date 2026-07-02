"""Parse an EPUB into a ParsedBook with chapter-level text and metadata."""

import hashlib
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# ── Gutenberg boilerplate ──────────────────────────────────────────────────────

_GUTENBERG_START = re.compile(
    r'\*{3}\s*START OF (?:THIS |THE )?PROJECT GUTENBERG[^\*]*\*{3}',
    re.IGNORECASE,
)
_GUTENBERG_END = re.compile(
    r'\*{3}\s*END OF (?:THIS |THE )?PROJECT GUTENBERG.*',
    re.IGNORECASE | re.DOTALL,
)

# ── Non-body chapter detection ────────────────────────────────────────────────

_NON_BODY_TITLE = re.compile(
    r'^\s*('
    r'title\s*page?|copyright|table\s*of\s*contents?|contents?'
    r'|dedication|preface|foreword|introduction\s+by'
    r'|acknowledgements?|appendix|bibliography|index'
    r'|glossary|notes?|about\s*the\s*author'
    r'|colophon|epigraph|halftitle|cover'
    r')\s*$',
    re.IGNORECASE,
)

# Spine filenames that indicate navigation/structural items, not prose
_NON_CONTENT_FILE = re.compile(
    r'(cover|toc|nav|ncx|colophon|titlepage|frontmatter|copyright)',
    re.IGNORECASE,
)

# Minimum body length for a detected chapter. Below this a "chapter" is almost
# always a bare heading or table-of-contents artifact (e.g. a title-page anchor or
# a contents listing), not real prose — so it is dropped during heading/anchor
# splitting. Shared by anchor-aware rung 1 and the heading-regex rung 3.
_MIN_CHAPTER_CHARS = 100

# ── Heading patterns for fallback detection (rung 3) ─────────────────────────

_CHAPTER_HEADINGS = [
    # "CHAPTER I", "Chapter One", "PART IV", "BOOK THREE", "VOLUME II"
    re.compile(
        r'^[ \t]*(CHAPTER|PART|BOOK|VOLUME)[ \t]+([IVXLCDM]+|\d+|[A-Z][a-z]+)\b.*$',
        re.MULTILINE | re.IGNORECASE,
    ),
    # Standalone Roman numerals on their own line (1–99).
    # The lookahead `(?=[IVXL])` requires at least one numeral, so this can never
    # match a blank/whitespace line (the old pattern's `V?I{0,3}` branch matched the
    # empty string, shattering blank-line-heavy HTML into one chapter per blank line).
    # Restricted to I/V/X/L (no M/D/C) so it doesn't collide with real all-caps words
    # like MIX, DIV, MM, DC, CM that a full Roman-numeral pattern would accept.
    re.compile(
        r'^[ \t]*(?=[IVXL])(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})[ \t]*$',
        re.MULTILINE,
    ),
    # Scene break: * * *
    re.compile(r'^[ \t]*\*[ \t]*\*[ \t]*\*[ \t]*$', re.MULTILINE),
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RawChapter:
    ordinal: int
    title: str | None
    text: str            # full plain-text content of this chapter
    is_body: bool = True
    is_synthetic: bool = False  # True if produced by the last-resort split


@dataclass
class ParsedBook:
    title: str
    author: str | None
    cover_mime: str | None
    cover_data: bytes | None
    source_hash: str
    chapters: list[RawChapter] = field(default_factory=list)
    raw_length: int = 0   # total chars across all chapters


# ── Internal helpers ──────────────────────────────────────────────────────────

def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def _get_meta(book: epub.EpubBook, dc_field: str) -> str | None:
    values = book.get_metadata('DC', dc_field)
    if not values:
        return None
    val = values[0]
    return val[0] if isinstance(val, tuple) else str(val)


def _extract_cover(book: epub.EpubBook) -> tuple[str | None, bytes | None]:
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_COVER:
            return item.media_type, item.get_content()
    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        if 'cover' in item.file_name.lower():
            return item.media_type, item.get_content()
    return None, None


def _html_to_text(html_bytes: bytes) -> str:
    """Strip HTML tags, preserving paragraph structure."""
    try:
        content = html_bytes.decode('utf-8', errors='replace')
    except Exception:
        content = str(html_bytes)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(content, 'lxml')
    for tag in soup.find_all(['p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        tag.insert_after('\n')
    return soup.get_text(separator='\n')


def _strip_gutenberg(text: str) -> str:
    m = _GUTENBERG_END.search(text)
    if m:
        text = text[:m.start()]
    m = _GUTENBERG_START.search(text)
    if m:
        text = text[m.end():]
    return text.strip()


def _is_non_body(title: str | None) -> bool:
    return bool(title and _NON_BODY_TITLE.match(title.strip()))


def _get_spine_items(book: epub.EpubBook) -> list[tuple[str, str]]:
    """Return (file_name, plain_text) for each HTML spine item, in reading order."""
    items = []
    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        text = _html_to_text(item.get_content())
        items.append((item.file_name, text))
    return items


def _get_spine_html(book: epub.EpubBook) -> dict[str, bytes]:
    """Map both full path and basename → raw HTML bytes for each spine document.

    Used by anchor-aware TOC splitting (rung 1), which needs the original markup to
    locate `id=` anchors that `_html_to_text` discards.
    """
    out: dict[str, bytes] = {}
    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        html = item.get_content()
        out[item.file_name] = html
        out[item.file_name.rsplit('/', 1)[-1]] = html
    return out


# Sentinel inserted before each anchored element so we can recover chapter
# boundaries after flattening HTML to text. Chosen to never occur in real prose.
# NUL/control chars are avoided: they are invalid in (X)HTML and get rewritten to
# U+FFFD when the modified soup is re-serialized, so the token must be plain text.
_ANCHOR_SENTINEL = '\n@@PM_CHAPTER@@\n'
_ANCHOR_SPLIT_TOKEN = '@@PM_CHAPTER@@'


def _split_html_by_anchors(
    html: bytes,
    anchors: list[tuple[str, str | None]],
) -> list[RawChapter] | None:
    """Split a single HTML document into chapters at TOC anchor positions.

    `anchors` is the ordered list of (fragment_id, title) the TOC points at within
    this file. Each anchored element gets a sentinel inserted before it; after
    flattening to text, the spans between sentinels become chapters. Bare-heading /
    title-page fragments shorter than ``_MIN_CHAPTER_CHARS`` are dropped (body) or
    kept as non-body (e.g. a CONTENTS listing). Returns ``None`` if fewer than two
    anchors resolve (caller falls back to whole-file behaviour).
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html.decode('utf-8', errors='replace'), 'lxml')

    placed: list[str | None] = []
    for frag, title in anchors:
        el = soup.find(id=frag)
        if el is None:
            continue
        el.insert_before(_ANCHOR_SENTINEL)
        placed.append(title)

    if len(placed) < 2:
        return None

    text = _html_to_text(str(soup).encode('utf-8'))
    parts = text.split(_ANCHOR_SPLIT_TOKEN)
    # parts[0] is the text before the first anchor (front matter); parts[1:] align
    # one-to-one with `placed`.
    chapters: list[RawChapter] = []
    for title, segment in zip(placed, parts[1:]):
        body_text = segment.strip()
        if not body_text:
            continue
        is_body = not _is_non_body(title)
        if is_body and len(body_text) < _MIN_CHAPTER_CHARS:
            continue  # bare heading / title-page anchor, not prose
        chapters.append(RawChapter(
            ordinal=len(chapters),
            title=title,
            text=body_text,
            is_body=is_body,
        ))

    return chapters if len(chapters) >= 2 else None


# ── Chapter detection — fallback ladder ──────────────────────────────────────

def _rung1_toc(
    book: epub.EpubBook,
    spine: list[tuple[str, str]],
    spine_html: dict[str, bytes],
) -> list[RawChapter] | None:
    """Rung 1: use the EPUB NCX/nav TOC.

    Two layouts are handled:
    - One spine file per chapter (each TOC entry → a distinct file): one chapter per
      file, deduped by text identity.
    - Whole book in a single HTML file with TOC `#anchors` (common for Gutenberg):
      any file referenced by ≥2 anchored TOC entries is split *at those anchors*
      (`_split_html_by_anchors`) rather than collapsed to a single chapter. Without
      this the dedup-by-identity below would yield just one chapter and rung 1 would
      bail, dropping the book down to the brittle heading-regex fallback.
    """
    if not book.toc:
        return None

    # Build filename → text mapping from spine (basename and full path as keys)
    file_index: dict[str, str] = {}
    for fname, text in spine:
        file_index[fname.rsplit('/', 1)[-1]] = text
        file_index[fname] = text

    def _flatten(toc_items):
        for item in toc_items:
            if isinstance(item, tuple):
                section, children = item
                yield section
                yield from _flatten(children)
            else:
                yield item

    # Parse TOC into ordered (basename, full_path, fragment, title) entries.
    entries: list[tuple[str, str, str | None, str | None]] = []
    for entry in _flatten(book.toc):
        href = getattr(entry, 'href', '') or ''
        title = getattr(entry, 'title', None)
        path = href.split('#', 1)[0]
        fragment = href.split('#', 1)[1] if '#' in href else None
        entries.append((path.rsplit('/', 1)[-1], path, fragment, title))

    def _html_key(basename: str, path: str) -> str | None:
        if basename in spine_html:
            return basename
        if path in spine_html:
            return path
        return None

    # Count anchored TOC entries per file; files with ≥2 are split by anchor.
    anchored_counts: dict[str, int] = defaultdict(int)
    for basename, path, fragment, _title in entries:
        key = _html_key(basename, path)
        if key and fragment:
            anchored_counts[key] += 1

    chapters: list[RawChapter] = []
    seen_texts: set[int] = set()
    consumed_files: set[str] = set()

    for basename, path, fragment, title in entries:
        hkey = _html_key(basename, path)

        # Anchor-split path: a single file holding several chapters via #fragments.
        if hkey is not None and anchored_counts.get(hkey, 0) >= 2:
            if hkey in consumed_files:
                continue  # already split this file as a whole
            consumed_files.add(hkey)
            file_anchors = [
                (frag, t)
                for (b2, p2, frag, t) in entries
                if _html_key(b2, p2) == hkey and frag
            ]
            split = _split_html_by_anchors(spine_html[hkey], file_anchors)
            if split:
                chapters.extend(split)
            continue

        # Whole-file path: one chapter per distinct spine file (existing behaviour).
        text = file_index.get(basename) or file_index.get(path)
        if not text or id(text) in seen_texts:
            continue
        seen_texts.add(id(text))
        chapters.append(RawChapter(
            ordinal=len(chapters),
            title=title,
            text=text,
            is_body=not _is_non_body(title),
        ))

    # Reassign contiguous ordinals (anchor splits insert several at once).
    for i, ch in enumerate(chapters):
        ch.ordinal = i

    return chapters if len(chapters) >= 2 else None


def _rung2_spine(spine: list[tuple[str, str]]) -> list[RawChapter] | None:
    """Rung 2: one spine item = one chapter (filter obvious non-content items)."""
    chapters: list[RawChapter] = []
    for fname, text in spine:
        text = text.strip()
        if not text or len(text) < 100:
            continue
        basename = fname.rsplit('/', 1)[-1]
        is_nav = bool(_NON_CONTENT_FILE.search(basename))
        chapters.append(RawChapter(
            ordinal=len(chapters),
            title=None,
            text=text,
            is_body=not is_nav,
        ))
    return chapters if len(chapters) >= 2 else None


def _rung3_heading_regex(spine: list[tuple[str, str]]) -> list[RawChapter] | None:
    """Rung 3: detect chapter breaks using heading regexes."""
    full_text = '\n'.join(text for _, text in spine)

    for pattern in _CHAPTER_HEADINGS:
        matches = list(pattern.finditer(full_text))
        if len(matches) < 2:
            continue

        chapters: list[RawChapter] = []
        for i, m in enumerate(matches):
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            text = full_text[body_start:body_end].strip()
            # Skip bare-heading / contents-listing artifacts (a heading immediately
            # followed by the next heading), not just empty spans.
            if len(text) < _MIN_CHAPTER_CHARS:
                continue
            title = m.group(0).strip()
            chapters.append(RawChapter(
                ordinal=len(chapters),
                title=title,
                text=text,
                is_body=True,
            ))

        if len(chapters) >= 2:
            return chapters

    return None


def _rung4_spine_all(spine: list[tuple[str, str]]) -> list[RawChapter] | None:
    """Rung 4: treat each spine item (including short ones) as a chapter."""
    chapters: list[RawChapter] = []
    for fname, text in spine:
        text = text.strip()
        if not text:
            continue
        chapters.append(RawChapter(
            ordinal=len(chapters),
            title=None,
            text=text,
            is_body=True,
        ))
    return chapters if chapters else None


def _rung5_synthetic(spine: list[tuple[str, str]]) -> list[RawChapter]:
    """Rung 5 (last resort): fixed-size synthetic chapters."""
    from pagemind.segment.tokens import SECTION_CAP

    full_text = '\n'.join(text for _, text in spine).strip()
    # Aim for ~10× section cap per synthetic chapter
    target_chars = SECTION_CAP * 10 * 4

    chapters: list[RawChapter] = []
    pos = 0
    while pos < len(full_text):
        end = min(pos + target_chars, len(full_text))
        if end < len(full_text):
            pb = full_text.rfind('\n\n', pos, end)
            if pb > pos:
                end = pb
        text = full_text[pos:end].strip()
        if text:
            n = len(chapters)
            chapters.append(RawChapter(
                ordinal=n,
                title=f'Section {n + 1}',
                text=text,
                is_body=True,
                is_synthetic=True,
            ))
        pos = end

    if not chapters:
        chapters = [RawChapter(ordinal=0, title='Section 1', text=full_text, is_body=True, is_synthetic=True)]
    return chapters


# ── Public interface ──────────────────────────────────────────────────────────

def parse_epub(path: Path) -> ParsedBook:
    """Parse an EPUB file into a ParsedBook.

    Applies the chapter-detection fallback ladder from [0009]:
    TOC → spine items → heading regex → all spine → synthetic split.
    """
    source_hash = _hash_file(path)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        book = epub.read_epub(str(path), {'check_name': False})

    title = _get_meta(book, 'title') or path.stem
    author = _get_meta(book, 'creator')
    cover_mime, cover_data = _extract_cover(book)

    raw_spine = _get_spine_items(book)
    # Strip Gutenberg boilerplate from each spine item
    spine = [(fname, _strip_gutenberg(text)) for fname, text in raw_spine]
    spine_html = _get_spine_html(book)

    chapters = (
        _rung1_toc(book, spine, spine_html)
        or _rung2_spine(spine)
        or _rung3_heading_regex(spine)
        or _rung4_spine_all(spine)
        or _rung5_synthetic(spine)
    )

    raw_length = sum(len(ch.text) for ch in chapters)

    return ParsedBook(
        title=title,
        author=author,
        cover_mime=cover_mime,
        cover_data=cover_data,
        source_hash=source_hash,
        chapters=chapters,
        raw_length=raw_length,
    )
