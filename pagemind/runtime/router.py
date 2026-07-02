"""Query router: classify → recipe enum (ADR 0005)."""
from __future__ import annotations

from pagemind.models.chat import ChatClient

RECIPES = (
    "fact_lookup",
    "verbatim_quote",
    "locate_entity",
    "chapter_summary",
    "contextual_why",
    "structured_view",
    "generic_fallback",
)

_SYS = (
    "You are a query classifier for a book reading assistant. "
    "Classify the user question into exactly one category. "
    "Reply with only the category name — nothing else."
)

_PROMPT = """\
Categories:
- fact_lookup: specific facts, who/what/when/where questions
- verbatim_quote: find or reproduce a specific passage or exact wording
- locate_entity: where does a person, place, or thing appear in the book
- chapter_summary: summarize a chapter or what has happened so far
- contextual_why: why/how questions that require context from later in the book
- structured_view: relationship maps, character networks, connection charts
- generic_fallback: anything that does not fit the above

QUESTION: {question}

Reply with exactly one category name.\
"""


def _parse_recipe(raw: str) -> str:
    """Return the first matching recipe keyword, falling back to generic_fallback."""
    lowered = raw.strip().lower()
    for recipe in RECIPES:
        if recipe in lowered:
            return recipe
    return "generic_fallback"


async def route(chat: ChatClient, question: str) -> str:
    """Return a recipe name for *question*. Always returns a valid recipe."""
    raw = await chat.complete(
        [
            {"role": "system", "content": _SYS},
            {"role": "user", "content": _PROMPT.format(question=question)},
        ],
        max_tokens=32,
    )
    return _parse_recipe(raw)
