"""Reader sub-call (ADR 0004): stateless, single-shot, context-quarantined."""
from __future__ import annotations

import asyncio
import json
import re
import unicodedata
import uuid

import psycopg

from pagemind.models.chat import ChatClient
from pagemind.retrieval.expand import expand_to_section
from pagemind.runtime.types import ReadResult

_FAN_OUT_DEFAULT = 4  # concurrent readers; RAM-constrained (ADR 0004)

_SYS = (
    "You are a precise reading assistant. "
    "Answer only from the passage given — never add outside knowledge. "
    "Return ONLY a JSON object, no other text."
)

_PROMPT = """\
PASSAGE:
{text}

QUESTION: {question}

Return a JSON object with this exact shape:
{{"answer": "1-3 sentence answer based only on the passage", "verbatim_quotes": ["exact substring from passage", ...]}}

Rules:
- verbatim_quotes must be exact substrings copied character-for-character from PASSAGE.
- Include 1-3 quotes that directly support the answer.
- answer must be 1-3 sentences.\
"""

# Ungrounded ("open-book") variants: the answer may draw on the model's own
# knowledge of the book in addition to the passage, but verbatim_quotes must
# still be exact substrings of PASSAGE so quote cards stay honest.
_SYS_OPEN = (
    "You are a precise reading assistant. "
    "Answer using the passage as your primary evidence; you may also draw on "
    "your own knowledge of this book to add context, but do not contradict the "
    "passage. "
    "Return ONLY a JSON object, no other text."
)

_PROMPT_OPEN = """\
PASSAGE:
{text}

QUESTION: {question}

Return a JSON object with this exact shape:
{{"answer": "1-3 sentence answer", "verbatim_quotes": ["exact substring from passage", ...]}}

Rules:
- verbatim_quotes must be exact substrings copied character-for-character from PASSAGE.
- Include 1-3 quotes that directly support the answer.
- answer must be 1-3 sentences; it may supplement the passage with your own
  knowledge of the book, but must not contradict the passage.\
"""


def _reader_prompts(grounded: bool) -> tuple[str, str]:
    """Return (system, user-template) for the reader in the requested mode."""
    return (_SYS, _PROMPT) if grounded else (_SYS_OPEN, _PROMPT_OPEN)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_reader_response(raw: str) -> tuple[str, list[str]]:
    """Extract (answer, verbatim_quotes) from LLM output; graceful on bad JSON."""
    text = raw.strip()
    # Try the whole string first, then the first {...} block
    for candidate in (text, *(_JSON_BLOCK_RE.findall(text))):
        try:
            data = json.loads(candidate)
            answer = str(data.get("answer", "")).strip() or text
            quotes = [str(q) for q in data.get("verbatim_quotes", []) if str(q).strip()]
            return answer, quotes
        except (json.JSONDecodeError, AttributeError):
            continue
    return text, []


# ── Quote realignment ─────────────────────────────────────────────────────────
# The reader model reliably reproduces a quote's words but not its exact bytes:
# EPUB prose carries hard line-breaks mid-sentence (``a flame flew\nover her body``,
# from _html_to_text) which the model collapses to a space, and it normalises
# typographic quotes/dashes. A raw ``quote in source`` check then drops every
# otherwise-correct quote. We match on a normalised view but recover the ORIGINAL
# source slice + offsets so quote cards and the passage popover stay byte-honest.

# Length-preserving character folds applied before NFKC (which may change length).
_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",  # single quotes
    "“": '"', "”": '"', "„": '"', "‟": '"',  # double quotes
    "‐": "-", "‑": "-", "‒": "-", "–": "-",  # hyphens/dashes
    "—": "-", "―": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",  # nbsp / thin spaces
}


def _norm_pass(
    source: str, build_map: bool
) -> tuple[str, list[int]] | str:
    """Normalise *source*: fold punctuation, collapse whitespace, NFKC, case-fold.

    Case-folding is included because the reader model routinely capitalises the
    first letter of a fragment it extracts from mid-sentence; we still return the
    original-case source slice, so quote cards stay verbatim.

    When *build_map* is True, also return a parallel list mapping each normalised
    character index back to the source index that produced it (so a match in the
    normalised view can be projected onto original byte offsets). NFKC/case-fold can
    expand a single source char into several normalised chars (``½`` → ``1⁄2``,
    ``ﬃ`` → ``ffi``); each shares the same source index.
    """
    out: list[str] = []
    idx: list[int] = []
    prev_space = False
    for i, ch in enumerate(source):
        mapped = _PUNCT_MAP.get(ch, ch)
        if mapped.isspace():
            if not prev_space:
                out.append(" ")
                if build_map:
                    idx.append(i)
            prev_space = True
            continue
        prev_space = False
        for c in unicodedata.normalize("NFKC", mapped).casefold():
            out.append(c)
            if build_map:
                idx.append(i)
    s = "".join(out)
    return (s, idx) if build_map else s


def _normalize(s: str) -> str:
    """String→string normalisation (whitespace-collapsed, folded, NFKC, case-folded)."""
    return _norm_pass(s, build_map=False)  # type: ignore[return-value]


def _locate_quote(source: str, quote: str) -> tuple[str, int, int] | None:
    """Locate *quote* in *source*, tolerant of whitespace/typography differences.

    Returns ``(original_slice, start, end)`` where ``original_slice ==
    source[start:end]`` is the verbatim source text and
    ``_normalize(quote) in _normalize(original_slice)``. Returns None when the
    quote cannot be located even after normalisation (a genuine hallucination /
    cross-passage stitch we cannot cite honestly), so the caller drops it rather
    than surface an uncitable "verbatim" quote.
    """
    pos = source.find(quote)
    if pos != -1:
        return quote, pos, pos + len(quote)

    norm, idx_map = _norm_pass(source, build_map=True)  # type: ignore[misc]
    qn = _normalize(quote).strip()
    if not qn:
        return None
    k = norm.find(qn)
    if k == -1:
        return None
    start = idx_map[k]
    end = idx_map[k + len(qn) - 1] + 1  # snap outward to the full source char
    return source[start:end], start, end


def _section_chapter(conn: psycopg.Connection, section_id: uuid.UUID) -> int:
    """Return the display chapter *number* for a section, defaulting to 0 if not found.

    Retrieval only surfaces body sections (see the ``is_body`` filters in
    lexical/semantic search), and every body chapter has a non-NULL ``number``, so a
    cited section always resolves to a real chapter number. The ``0`` fallback is only
    the genuine "section not found" sentinel.
    """
    row = conn.execute(
        """
        SELECT c.number
        FROM sections s
        JOIN chapters c ON c.chapter_id = s.chapter_id
        WHERE s.section_id = %s
        """,
        (section_id,),
    ).fetchone()
    return row[0] if row and row[0] is not None else 0


async def read(
    conn: psycopg.Connection,
    chat: ChatClient,
    section_id: uuid.UUID,
    question: str,
    *,
    grounded: bool = True,
) -> ReadResult:
    """Single-shot reader: fetch section text, ask LLM, return distilled result.

    When *grounded* is False the reader may supplement the passage with the
    model's own knowledge of the book (verbatim quotes stay passage substrings).
    """
    text = expand_to_section(conn, section_id)
    if not text:
        return ReadResult(section_id=section_id, chapter=0, answer="[section not found]")

    chapter = _section_chapter(conn, section_id)
    sys_prompt, user_tmpl = _reader_prompts(grounded)
    raw = await chat.complete(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_tmpl.format(text=text, question=question)},
        ],
        max_tokens=512,
    )
    answer, quotes = _parse_reader_response(raw)
    # Realign each quote to a verbatim source slice (tolerant of whitespace /
    # typography); drop only quotes we cannot locate at all. Keeps the two lists
    # index-aligned for the synthesizer's per-quote zip.
    located = [loc for q in quotes if (loc := _locate_quote(text, q)) is not None]
    valid_quotes = [t for (t, _s, _e) in located]
    offsets = [(s, e) for (_t, s, e) in located]
    return ReadResult(
        section_id=section_id,
        chapter=chapter,
        answer=answer,
        verbatim_quotes=valid_quotes,
        char_offsets=offsets,
    )


async def fan_out(
    conn: psycopg.Connection,
    chat: ChatClient,
    section_ids: list[uuid.UUID],
    question: str,
    *,
    max_concurrent: int = _FAN_OUT_DEFAULT,
    grounded: bool = True,
) -> list[ReadResult]:
    """Run reader against N sections in parallel, capped at max_concurrent (ADR 0004)."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _guarded(sid: uuid.UUID) -> ReadResult:
        async with sem:
            return await read(conn, chat, sid, question, grounded=grounded)

    return list(await asyncio.gather(*(_guarded(sid) for sid in section_ids)))
