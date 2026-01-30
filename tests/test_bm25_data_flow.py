from __future__ import annotations

import math
from collections import Counter

import pytest

from src.retrieval.bm25 import BM25BuildConfig, build_bm25_index, simple_tokenize


def test_bm25_data_flow_with_prints() -> None:
    """
    Run with `pytest -s -k bm25_data_flow_with_prints` to see the trace output.
    """
    pytest.importorskip("rank_bm25")

    movie_ids = [1, 2, 3]
    texts = [
        "Title: Toy Story. Year: 1995. Genres: Adventure, Animation, Children, Comedy, Fantasy. Tags: pixar, fun",
        "Title: Heat. Year: 1995. Genres: Action, Crime, Thriller. Tags: heist, detective",
        "Title: Grumpier Old Men. Year: 1995. Genres: Comedy, Romance. Tags: moldy, old",
    ]

    print("\n" + "=" * 110)
    print("BM25 TRACE: input corpus (movieId -> text)")
    print("=" * 110)
    for mid, text in zip(movie_ids, texts):
        print(f"movieId={mid} text={text}")

    cfg = BM25BuildConfig(tokenizer="simple", store_tokenized_corpus=True)
    index = build_bm25_index(movie_ids, texts, cfg)
    assert index.tokenized_corpus is not None

    print("\n" + "=" * 110)
    print("BM25 TRACE: tokenized corpus (simple_tokenize)")
    print("=" * 110)
    for mid, tokens in zip(index.movie_ids, index.tokenized_corpus):
        print(f"movieId={mid} tokens={tokens}")

    query = "pixar toy"
    query_tokens = simple_tokenize(query)

    print("\n" + "=" * 110)
    print("BM25 TRACE: query tokenization")
    print("=" * 110)
    print(f"query={query!r}")
    print(f"query_tokens={query_tokens}")

    doc_sets = [set(doc) for doc in index.tokenized_corpus]
    doc_freq = {t: sum(1 for s in doc_sets if t in s) for t in query_tokens}

    print("\n" + "=" * 110)
    print("BM25 TRACE: document frequency / IDF for query tokens")
    print("=" * 110)
    for t in query_tokens:
        df = doc_freq[t]
        n = len(doc_sets)
        # Standard BM25 IDF (note: implementations may vary slightly).
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        print(f"token={t!r} doc_freq={df} corpus_size={n} approx_idf={idf:.6f}")

    scores = index.bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)

    print("\n" + "=" * 110)
    print("BM25 TRACE: scores (sorted)")
    print("=" * 110)
    for rank, (i, score) in enumerate(ranked, start=1):
        tokens = index.tokenized_corpus[i]
        overlap = sorted(set(tokens) & set(query_tokens))
        tf = Counter(tokens)
        tf_query = {t: int(tf.get(t, 0)) for t in query_tokens}
        doc_len = len(tokens)
        print(
            f"{rank:02d}. movieId={index.movie_ids[i]} score={float(score):.6f} "
            f"doc_len={doc_len} tf={tf_query} overlap={overlap} text={texts[i]}"
        )

    # Deterministic check: only movieId=1 contains both 'pixar' and 'toy'.
    best_i, _ = ranked[0]
    assert index.movie_ids[int(best_i)] == 1
