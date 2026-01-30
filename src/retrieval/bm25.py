"""Lexical retrieval artifacts (BM25 index over movie profiles)."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


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
    """Online BM25 retriever (not implemented in this prompt)."""

    def __init__(self) -> None:
        raise NotImplementedError("Online BM25Retriever is not part of the offline build task.")
