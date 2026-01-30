from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.dense import DenseBuildConfig, build_and_save_dense_index, embed_texts, load_bi_encoder


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def bi_encoder() -> object:
    pytest.importorskip("sentence_transformers")
    try:
        return load_bi_encoder(MODEL_NAME, device="cpu")
    except Exception as exc:
        pytest.skip(f"Could not load SentenceTransformer model {MODEL_NAME!r}: {exc}")


def test_dense_embeddings_one_per_text_and_normalized(bi_encoder: object) -> None:
    texts = [
        "Toy Story (1995). Genres: Animation|Children's|Comedy.",
        "Jumanji (1995). Genres: Adventure|Children's|Fantasy.",
        "Heat (1995). Genres: Action|Crime|Thriller.",
    ]

    embeddings = embed_texts(bi_encoder, texts, batch_size=2, normalize_embeddings=True)

    assert embeddings.shape[0] == len(texts)
    assert embeddings.ndim == 2
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3, rtol=1e-3)


def test_dense_embeddings_consistent_across_batch_sizes(bi_encoder: object) -> None:
    texts = [
        "A space opera with rebels and an evil empire.",
        "A quiet drama about friendship and loss.",
        "A detective investigates a series of mysterious crimes.",
    ]

    emb_bs1 = embed_texts(bi_encoder, texts, batch_size=1, normalize_embeddings=True)
    emb_bs3 = embed_texts(bi_encoder, texts, batch_size=3, normalize_embeddings=True)

    np.testing.assert_allclose(emb_bs1, emb_bs3, rtol=1e-4, atol=1e-5)


def test_dense_index_build_saves_one_embedding_per_movie(
    bi_encoder: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    pytest.importorskip("faiss")
    import json
    from pathlib import Path

    import faiss  # type: ignore[import-not-found]

    movie_ids = [1, 2, 3]
    texts = [
        "Toy Story (1995). Genres: Animation|Children's|Comedy.",
        "Jumanji (1995). Genres: Adventure|Children's|Fantasy.",
        "Heat (1995). Genres: Action|Crime|Thriller.",
    ]

    # Avoid re-downloading/re-loading the model: reuse the fixture.
    import src.retrieval.dense as dense_mod

    monkeypatch.setattr(dense_mod, "load_bi_encoder", lambda *_args, **_kwargs: bi_encoder)

    cfg = DenseBuildConfig(model_name=MODEL_NAME, batch_size=2, save_embeddings=True, device="cpu")
    out = build_and_save_dense_index(movie_ids, texts, cfg, tmp_path / "dense_artifacts")

    emb_path = out.get("dense_embeddings")
    assert emb_path is not None

    embeddings = np.load(emb_path)
    assert embeddings.shape[0] == len(movie_ids)
    assert embeddings.shape[1] > 0
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3, rtol=1e-3)

    index = faiss.read_index(out["dense_faiss_index"])
    assert int(index.ntotal) == len(movie_ids)

    ids = json.loads(Path(out["dense_movie_ids"]).read_text())
    assert ids == movie_ids
