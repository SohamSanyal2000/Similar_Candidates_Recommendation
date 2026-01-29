"""Lexical retrieval using BM25.

Phase 0 note:
- Only interfaces here.
- Implementation comes in Phase 3 (Hybrid retrieval).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BM25RetrievalResult:
    movie_ids: List[int]
    scores: List[float]


class BM25Retriever:
    def __init__(self) -> None:
        # TODO(phase3): store corpus and BM25 index
        pass

    def retrieve(self, query_text: str, top_k: int) -> BM25RetrievalResult:
        raise NotImplementedError
