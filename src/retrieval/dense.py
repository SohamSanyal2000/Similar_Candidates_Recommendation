"""Dense semantic retrieval (bi-encoder + FAISS).

Phase 0 note:
- We only define interfaces here.
- Implementation comes in Phase 3 (Retrieval development).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class DenseRetrievalResult:
    movie_ids: List[int]
    scores: List[float]


class DenseRetriever:
    """Bi-encoder + FAISS retriever.

    Responsibilities:
    1) Build embeddings for all movie profiles (offline).
    2) Build and persist FAISS index (offline).
    3) Load index and serve nearest-neighbor retrieval (online).
    """

    def __init__(self) -> None:
        # TODO(phase3): store model, index, and id mapping
        pass

    def retrieve(self, query_text: str, top_k: int) -> DenseRetrievalResult:
        """Return top_k similar movies for a text query."""
        raise NotImplementedError
