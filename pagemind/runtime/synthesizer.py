"""Synthesizer: compose final answer from reader results + citations."""
from __future__ import annotations

import uuid

from pagemind.models.chat import ChatClient
from pagemind.runtime.types import Citation, Quote, QueryResult, ReadResult

_SYS = (
    "You are a literary assistant. "
    "Compose a clear, accurate answer using only the provided evidence. "
    "Do not add information that is not present in the evidence."
)

_PROMPT = """\
ORIGINAL QUESTION: {question}

EVIDENCE:
{evidence}

Write a concise answer (2-5 sentences) based solely on the evidence.\
"""

# Ungrounded ("open-book") variants: the answer may supplement the evidence with
# the model's own knowledge of the book, but must prefer and never contradict it.
_SYS_OPEN = (
    "You are a literary assistant. "
    "Compose a clear, accurate answer. Prefer the provided evidence and stay "
    "consistent with it; where the evidence is incomplete you may supplement with "
    "your own knowledge of this book. Never contradict the evidence."
)

_PROMPT_OPEN = """\
ORIGINAL QUESTION: {question}

EVIDENCE:
{evidence}

Write a concise answer (2-5 sentences). Base it primarily on the evidence; you may
supplement with your broader knowledge of the book where the evidence is incomplete,
without contradicting it.\
"""


def answer_prompts(grounded: bool) -> tuple[str, str]:
    """Return (system, answer-template) for the synthesizer in the requested mode."""
    return (_SYS, _PROMPT) if grounded else (_SYS_OPEN, _PROMPT_OPEN)


def _format_evidence(results: list[ReadResult]) -> str:
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        lines = [f"[Source {i} — chapter {r.chapter}]", r.answer]
        if r.verbatim_quotes:
            lines.append("Quotes: " + " | ".join(f'"{q}"' for q in r.verbatim_quotes))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _build_output(
    book_id: uuid.UUID,
    question: str,
    synthesis_text: str,
    results: list[ReadResult],
) -> QueryResult:
    quotes: list[Quote] = []
    citations: list[Citation] = []
    seen_sections: set[uuid.UUID] = set()

    for r in results:
        if r.section_id not in seen_sections:
            citations.append(Citation(
                book_id=book_id,
                chapter=r.chapter,
                section_id=r.section_id,
            ))
            seen_sections.add(r.section_id)
        for q, (start, _) in zip(r.verbatim_quotes, r.char_offsets):
            cit = Citation(
                book_id=book_id,
                chapter=r.chapter,
                section_id=r.section_id,
                char_offset=start if start >= 0 else None,
            )
            quotes.append(Quote(text=q, citation=cit))

    return QueryResult(text=synthesis_text, quotes=quotes, citations=citations)


# Public aliases used by streaming.py
SYS_PROMPT = _SYS
ANSWER_PROMPT = _PROMPT
format_evidence = _format_evidence
build_output = _build_output


async def synthesize(
    chat: ChatClient,
    book_id: uuid.UUID,
    question: str,
    results: list[ReadResult],
) -> QueryResult:
    """Compose final answer from reader results; return through single output seam."""
    if not results:
        return QueryResult(text="No relevant passages found.", citations=[], weak=True)

    evidence = _format_evidence(results)
    synthesis_text = await chat.complete(
        [
            {"role": "system", "content": _SYS},
            {"role": "user", "content": _PROMPT.format(question=question, evidence=evidence)},
        ],
        max_tokens=512,
    )
    return _build_output(book_id, question, synthesis_text.strip(), results)
