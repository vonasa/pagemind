"""Reciprocal Rank Fusion over arbitrary ranked lists."""
from __future__ import annotations

import uuid


def rrf_fuse(
    ranked_lists: list[list[uuid.UUID]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[uuid.UUID, float]]:
    """Fuse ranked lists with RRF.

    score(doc) = Σ_j  w_j / (k + rank_j(doc))

    rank is 1-indexed; documents absent from a list contribute 0 from that list.
    Returns (doc_id, score) pairs sorted by descending score.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[uuid.UUID, float] = {}
    for ranked, w in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
