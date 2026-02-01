"""Cross-encoder reranking.

This module provides a small wrapper around Sentence-Transformers' `CrossEncoder`
so that the service layer (FastAPI / TwoStageMovieRecommender) can depend on a
stable, testable interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, List, Sequence


@dataclass(frozen=True)
class RerankResult:
    movie_ids: List[int]
    scores: List[float]


class CrossEncoderReranker:
    """Score (query, candidate) pairs with a cross-encoder.

    Notes
    -----
    - By default this loads `sentence_transformers.CrossEncoder`.
    - For tests, you can inject a `model` implementing either:
        * `.predict(pairs, batch_size=..., show_progress_bar=False)` (ST style), or
        * `.rerank(query_text, candidate_texts)` (simple custom interface).
    """

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model: Any | None = None,
    ) -> None:
        self.model_name = str(model_name)
        self._lock = Lock()

        if model is None:
            self.model = self._load_model(self.model_name)
        else:
            self.model = model

    def _load_model(self, model_name: str) -> Any:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise ImportError("`sentence-transformers` is required for cross-encoder reranking.") from exc

        t0 = time.perf_counter()
        m = CrossEncoder(str(model_name))
        try:
            m.model.eval()  # type: ignore[attr-defined]
        except Exception:
            # Not all backends expose .model
            pass
        # Keep this module logger-free; service layer can log overall startup times.
        _ = time.perf_counter() - t0
        return m

    def score_pairs(self, pairs: Sequence[tuple[str, str]], *, batch_size: int = 32) -> List[float]:
        """Return a score for each (query, candidate) pair."""
        if not pairs:
            return []

        with self._lock:
            if hasattr(self.model, "predict"):
                scores = self.model.predict(list(pairs), batch_size=int(batch_size), show_progress_bar=False)
                return [float(x) for x in list(scores)]

            if hasattr(self.model, "rerank"):
                q = str(pairs[0][0])
                cands = [str(p[1]) for p in pairs]
                scores = self.model.rerank(q, cands)
                return [float(x) for x in list(scores)]

        raise TypeError("cross-encoder model must provide .predict(pairs) or .rerank(query_text, candidate_texts)")

    def rerank(self, query_text: str, candidate_texts: List[str], *, batch_size: int = 32) -> List[float]:
        """Convenience wrapper: score a single query against many candidates."""
        pairs = [(str(query_text), str(t)) for t in candidate_texts]
        return self.score_pairs(pairs, batch_size=int(batch_size))

    def rerank_movie_ids(
        self,
        *,
        query_text: str,
        movie_ids: List[int],
        candidate_texts: List[str],
        batch_size: int = 32,
    ) -> RerankResult:
        """Rerank `movie_ids` given aligned `candidate_texts`.

        Returns movie_ids sorted by score desc.
        """
        if len(movie_ids) != len(candidate_texts):
            raise ValueError("movie_ids and candidate_texts must have the same length")

        scores = self.rerank(str(query_text), [str(t) for t in candidate_texts], batch_size=int(batch_size))
        if len(scores) != len(movie_ids):
            raise RuntimeError("cross-encoder returned mismatched number of scores")

        order = sorted(range(len(movie_ids)), key=lambda i: float(scores[i]), reverse=True)
        sorted_ids = [int(movie_ids[i]) for i in order]
        sorted_scores = [float(scores[i]) for i in order]
        return RerankResult(movie_ids=sorted_ids, scores=sorted_scores)
