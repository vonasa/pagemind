"""Chapter-scope SQL fragment shared by lexical/semantic/structured retrieval.

Both bounds are on the human-facing ``chapters.number`` (1-based among body chapters,
NULL for non-body):

- ``up_to_chapter`` — a spoiler ceiling (``number <= N``).
- ``chapter``       — pins one exact chapter (``number = N``).

Non-body chapters have NULL ``number`` and are naturally excluded by either predicate.
The clause references ``%(book_id)s`` (already present in every caller's param dict) plus
``%(upto)s`` / ``%(chapter)s``, which :func:`chapter_scope_params` supplies.
"""
from __future__ import annotations


def chapter_scope_clause(
    alias: str,
    *,
    up_to_chapter: int | None = None,
    chapter: int | None = None,
) -> str:
    """Return an ``AND {alias}.chapter_id IN (…)`` fragment, or ``''`` if unscoped."""
    preds = []
    if up_to_chapter is not None:
        preds.append("number <= %(upto)s")
    if chapter is not None:
        preds.append("number = %(chapter)s")
    if not preds:
        return ""
    return (
        f" AND {alias}.chapter_id IN ("
        "SELECT chapter_id FROM chapters "
        "WHERE book_id = %(book_id)s AND " + " AND ".join(preds) + ")"
    )


def chapter_scope_params(
    *,
    up_to_chapter: int | None = None,
    chapter: int | None = None,
) -> dict:
    """Return the named params referenced by :func:`chapter_scope_clause`."""
    params: dict = {}
    if up_to_chapter is not None:
        params["upto"] = up_to_chapter
    if chapter is not None:
        params["chapter"] = chapter
    return params
