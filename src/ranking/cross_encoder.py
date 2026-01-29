"""Cross-encoder reranking.

Phase 0 note:
- Only interfaces here.
- Implementation comes in Phase 4 (Ranking development).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RerankResult:
    movie_ids: List[int]
    scores: List[float]


class CrossEncoderReranker:
    def __init__(self) -> None:
        # TODO(phase4): load model
        pass

    def rerank(self, query_text: str, candidate_texts: List[str]) -> List[float]:
        """Return a score for each (query, candidate) pair."""
        raise NotImplementedError
