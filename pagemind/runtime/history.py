"""Conversation history helpers: condense a follow-up into a standalone question.

The chat is stateless per request; the client sends prior turns with each ask.
We use that history in exactly one place — a *condense* step that rewrites an
elliptical follow-up ("and then what?") into a self-contained question. That
rewritten question then drives routing, retrieval, and synthesis, so pronouns
resolve without ever injecting prior turns into the (grounded) synthesizer.
"""
from __future__ import annotations

from pagemind.models.chat import ChatClient

# A history turn is a plain dict: {"role": "user"|"assistant", "content": str}.
Turn = dict

_MAX_TURNS = 6
_MAX_CHARS = 500

_SYS = (
    "You rewrite a follow-up question from a book-chat conversation into a single "
    "self-contained question. Resolve pronouns and ellipsis using the conversation "
    "so it can be understood on its own. Preserve every specific detail verbatim — "
    "chapter numbers, character and place names, and any quoted phrases. Do not "
    "answer the question. Reply with only the rewritten question, nothing else."
)

_PROMPT = """\
CONVERSATION SO FAR:
{history}

FOLLOW-UP QUESTION: {question}

Rewritten standalone question:\
"""


def cap_history(
    history: list[Turn],
    max_turns: int = _MAX_TURNS,
    max_chars: int = _MAX_CHARS,
) -> list[Turn]:
    """Keep the last *max_turns* turns, truncating each content to *max_chars*.

    Bounds prompt size regardless of what the client sends.
    """
    trimmed: list[Turn] = []
    for turn in history[-max_turns:]:
        content = str(turn.get("content", ""))
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "…"
        trimmed.append({"role": turn.get("role", "user"), "content": content})
    return trimmed


def format_history(history: list[Turn]) -> str:
    """Render turns as 'User: …' / 'PageMind: …' lines for the condense prompt."""
    lines: list[str] = []
    for turn in history:
        label = "User" if turn.get("role") == "user" else "PageMind"
        lines.append(f"{label}: {turn.get('content', '')}")
    return "\n".join(lines)


async def condense_question(
    chat: ChatClient,
    history: list[Turn] | None,
    question: str,
) -> str:
    """Rewrite *question* into a standalone question given *history*.

    Returns *question* unchanged (no LLM call) when there is no history. On any
    failure — an empty rewrite or a raised exception — falls back to the raw
    question so a condense hiccup never aborts the stream.
    """
    if not history:
        return question

    capped = cap_history(history)
    if not capped:
        return question

    try:
        raw = await chat.complete(
            [
                {"role": "system", "content": _SYS},
                {
                    "role": "user",
                    "content": _PROMPT.format(
                        history=format_history(capped), question=question
                    ),
                },
            ],
            max_tokens=128,
        )
    except Exception:
        return question

    standalone = raw.strip()
    return standalone or question
