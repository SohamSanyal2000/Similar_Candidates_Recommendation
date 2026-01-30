"""Dense semantic retrieval artifacts (bi-encoder embeddings + FAISS index)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import faiss  # type: ignore[import-not-found]
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]


@dataclass(frozen=True)
class DenseRetrievalResult:
    movie_ids: List[int]
    scores: List[float]


@dataclass(frozen=True)
class DenseBuildConfig:
    model_name: str
    index_type: str = "IndexFlatIP"
    normalize_embeddings: bool = True
    batch_size: int = 64
    device: Optional[str] = None
    save_embeddings: bool = True


def load_bi_encoder(model_name: str, device: Optional[str] = None) -> "SentenceTransformer":
    """Load a SentenceTransformer bi-encoder."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise ImportError("`sentence-transformers` is required for dense retrieval artifacts.") from exc

    model = SentenceTransformer(model_name, device=device)
    try:
        model.eval()  # type: ignore[attr-defined]
    except Exception:
        pass
    return model


def embed_texts(
    model: "SentenceTransformer",
    texts: list[str],
    *,
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    """Embed texts into a float32 matrix aligned with input ordering."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if normalize_embeddings:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        embeddings = embeddings / norms

    return embeddings.astype(np.float32, copy=False)


def build_faiss_index(embeddings: np.ndarray, index_type: str) -> "faiss.Index":
    """Build a FAISS index and add embeddings (row-aligned)."""
    try:
        import faiss  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise ImportError("`faiss-cpu` is required to build dense retrieval artifacts.") from exc

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape={embeddings.shape}")

    dim = int(embeddings.shape[1])
    if index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "IndexFlatL2":
        index = faiss.IndexFlatL2(dim)
    else:
        try:
            index = faiss.index_factory(dim, index_type)
        except Exception as exc:
            raise ValueError(f"Unsupported FAISS index type: {index_type}") from exc

    index.add(embeddings)
    return index


def save_dense_artifacts(
    out_dir: Path,
    *,
    index: "faiss.Index",
    movie_ids: list[int],
    embeddings: Optional[np.ndarray],
    meta: dict[str, Any],
) -> dict[str, str]:
    """Persist FAISS index + row-aligned movie id mapping (+ optional embeddings)."""
    try:
        import faiss  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise ImportError("`faiss-cpu` is required to save dense retrieval artifacts.") from exc

    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "dense_faiss.index"
    ids_path = out_dir / "dense_movie_ids.json"
    meta_path = out_dir / "dense_index_meta.json"
    emb_path = out_dir / "dense_embeddings.npy"

    faiss.write_index(index, str(index_path))
    ids_path.write_text(json_dumps(movie_ids) + "\n")

    meta_out = dict(meta)
    meta_out.setdefault("built_at_utc", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    meta_out["ntotal"] = int(index.ntotal)
    meta_out["artifact_type"] = "faiss_dense_index"
    meta_path.write_text(json_dumps(meta_out, indent=2, sort_keys=True) + "\n")

    if embeddings is not None:
        np.save(emb_path, embeddings)

    return {
        "dense_faiss_index": str(index_path),
        "dense_movie_ids": str(ids_path),
        "dense_index_meta": str(meta_path),
        **({"dense_embeddings": str(emb_path)} if embeddings is not None else {}),
    }


def build_and_save_dense_index(
    movie_ids: list[int],
    texts: list[str],
    cfg: DenseBuildConfig,
    artifacts_dir: Path,
) -> dict[str, str]:
    """Build embeddings + FAISS index and persist artifacts.

    Note: `movie_ids` and `texts` must be aligned and already sorted by movieId.
    """
    if len(movie_ids) != len(texts):
        raise ValueError(f"movie_ids/texts length mismatch: {len(movie_ids)} vs {len(texts)}")
    if movie_ids != sorted(movie_ids):
        raise ValueError("movie_ids must be sorted by movieId for deterministic row alignment")

    model = load_bi_encoder(cfg.model_name, device=cfg.device)
    embeddings = embed_texts(
        model,
        texts,
        batch_size=cfg.batch_size,
        normalize_embeddings=cfg.normalize_embeddings,
    )

    index = build_faiss_index(embeddings, cfg.index_type)
    if int(index.ntotal) != len(movie_ids):
        raise RuntimeError(f"FAISS index size mismatch: ntotal={index.ntotal} expected={len(movie_ids)}")

    meta = {
        "model_name": cfg.model_name,
        "index_type": cfg.index_type,
        "normalize_embeddings": cfg.normalize_embeddings,
        "dim": int(embeddings.shape[1]),
    }
    return save_dense_artifacts(
        artifacts_dir,
        index=index,
        movie_ids=movie_ids,
        embeddings=(embeddings if cfg.save_embeddings else None),
        meta=meta,
    )


def json_dumps(obj: Any, **kwargs: Any) -> str:
    import json

    return json.dumps(obj, **kwargs)


class DenseRetriever:
    """Online retriever (not implemented in this prompt)."""

    def __init__(self) -> None:
        raise NotImplementedError("Online DenseRetriever is not part of the offline build task.")
