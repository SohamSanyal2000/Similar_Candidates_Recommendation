"""Lexical retrieval artifacts (BM25 index over movie profiles)."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass(frozen=True)
class BM25RetrievalResult:
    movie_ids: List[int]
    scores: List[float]

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def simple_tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase and regex word tokens."""
    if text is None:
        return []
    return _WORD_RE.findall(str(text).lower())


@dataclass(frozen=True)
class BM25BuildConfig:
    tokenizer: str = "simple"
    store_tokenized_corpus: bool = False


@dataclass
class BM25Index:
    movie_ids: list[int]
    bm25: object
    tokenizer: str = "simple"
    tokenized_corpus: list[list[str]] | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "BM25Index":
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, BM25Index):
            raise TypeError(f"Expected BM25Index, got {type(obj)}")
        return obj


def build_bm25_index(movie_ids: list[int], texts: list[str], cfg: BM25BuildConfig) -> BM25Index:
    """Build a BM25Okapi index aligned to input ordering."""
    if len(movie_ids) != len(texts):
        raise ValueError(f"movie_ids/texts length mismatch: {len(movie_ids)} vs {len(texts)}")
    if cfg.tokenizer != "simple":
        raise ValueError(f"Unsupported tokenizer: {cfg.tokenizer}")

    try:
        from rank_bm25 import BM25Okapi  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise ImportError("`rank-bm25` is required to build BM25 artifacts.") from exc

    tokenized_corpus = [simple_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return BM25Index(
        movie_ids=[int(x) for x in movie_ids],
        bm25=bm25,
        tokenizer=cfg.tokenizer,
        tokenized_corpus=(tokenized_corpus if cfg.store_tokenized_corpus else None),
    )


def build_and_save_bm25_index(
    movie_ids: list[int],
    texts: list[str],
    cfg: BM25BuildConfig,
    artifacts_dir: Path,
) -> str:
    """Build and persist BM25 artifacts to `artifacts_dir`."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    index = build_bm25_index(movie_ids, texts, cfg)

    out_path = artifacts_dir / "bm25_index.pkl"
    index.save(out_path)
    return str(out_path)


class BM25Retriever:
    """Online BM25 retriever wrapper around an offline-built BM25Index."""

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    @classmethod
    def from_artifact(cls, path: Path) -> "BM25Retriever":
        """Load a BM25Index saved by the offline build."""
        return cls(BM25Index.load(Path(path)))

    def search(self, query_text: str, *, top_k: int) -> BM25RetrievalResult:
        """Retrieve the top-k documents for a query string."""
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        query_text = "" if query_text is None else str(query_text)
        if query_text.strip() == "":
            raise ValueError("query_text must be non-empty")

        if self.index.tokenizer != "simple":
            raise ValueError(f"Unsupported tokenizer: {self.index.tokenizer!r}")

        query_tokens = simple_tokenize(query_text)
        scores = np.asarray(self.index.bm25.get_scores(query_tokens), dtype=np.float32)
        if scores.ndim != 1 or len(scores) != len(self.index.movie_ids):
            raise RuntimeError("BM25 get_scores returned unexpected shape")

        k = int(min(int(top_k), len(scores)))
        if k == 0:
            return BM25RetrievalResult(movie_ids=[], scores=[])

        # Efficient top-k: argpartition then sort those k indices.
        top_idx = np.argpartition(-scores, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx], kind="mergesort")]

        movie_ids = [int(self.index.movie_ids[int(i)]) for i in top_idx.tolist()]
        out_scores = [float(scores[int(i)]) for i in top_idx.tolist()]
        return BM25RetrievalResult(movie_ids=movie_ids, scores=out_scores)
