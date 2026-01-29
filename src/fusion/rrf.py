"""Reciprocal Rank Fusion (RRF) for hybrid retrieval list merging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    k: int = 60,
    max_items: int = 300,
) -> List[Tuple[int, float]]:
    """Fuse multiple ranked lists of ids using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists:
        A sequence of ranked id lists. Each list is ordered best->worst.
    k:
        RRF constant. Typical values: 10..100. Higher => less rank-sensitive.
    max_items:
        Maximum number of fused items to return.

    Returns
    -------
    List[(id, score)]
        Fused ranking as (id, rrf_score), sorted desc by score.
    """
    scores: Dict[int, float] = {}
    for lst in ranked_lists:
        for rank, item_id in enumerate(lst, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:max_items]
