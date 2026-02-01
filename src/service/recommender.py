"""Two-stage movie similarity recommender (retrieval + reranking)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Optional

import pandas as pd
import yaml

from ..fusion.rrf import reciprocal_rank_fusion
from ..paths import get_repo_root
from ..ranking.cross_encoder import CrossEncoderReranker
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.dense import DenseRetriever, embed_texts, load_bi_encoder
from ..store.catalog import MovieCatalog

logger = logging.getLogger(__name__)

Mode = Literal["dense", "hybrid"]


def _resolve_path(repo_root: Path, path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    obj = yaml.safe_load(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected YAML mapping at {path}, got {type(obj)}")
    return obj


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(obj)}")
    return obj


def _parse_genres(genres: str) -> set[str]:
    if genres is None:
        return set()
    text = str(genres).strip()
    if not text:
        return set()
    if "|" in text:
        parts = [p.strip() for p in text.split("|")]
    else:
        parts = [p.strip() for p in text.split(",")]
    return {p for p in parts if p}


def _parse_tags(tags_text: str) -> set[str]:
    if tags_text is None:
        return set()
    text = str(tags_text).strip().lower()
    if not text:
        return set()
    if "|" in text:
        parts = [p.strip() for p in text.split("|")]
    else:
        parts = [p.strip() for p in text.split(",")]
    return {p for p in parts if p}


@dataclass
class Candidate:
    movieId: int
    dense_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    retrieval_sources: list[str] | None = None


class TwoStageMovieRecommender:
    """Two-stage recommender: hybrid retrieval (dense + BM25) + cross-encoder reranking.

    The class is designed to be instantiated once at process startup and reused across requests.
    """

    def __init__(
        self,
        *,
        offline_manifest_path: Path,
        config_path: Path | None = None,
        bi_encoder: Any | None = None,
        cross_encoder: Any | None = None,
    ) -> None:
        self.repo_root = get_repo_root()
        self.offline_manifest_path = Path(offline_manifest_path).resolve()

        self.config_path = (Path(config_path).resolve() if config_path else (self.repo_root / "config.yaml"))
        self.config = _load_yaml(self.config_path)

        self.manifest = _load_json(self.offline_manifest_path)
        self.manifest_config = self.manifest.get("config") if isinstance(self.manifest.get("config"), dict) else {}

        outputs = self.manifest.get("outputs", {})
        if not isinstance(outputs, dict):
            raise ValueError("offline_manifest.json missing 'outputs' object")

        processed = outputs.get("processed", {})
        dense = outputs.get("dense", {})
        bm25 = outputs.get("bm25", {})
        if not isinstance(processed, dict) or not isinstance(dense, dict) or not isinstance(bm25, dict):
            raise ValueError("offline_manifest.json outputs must include processed/dense/bm25 objects")

        movie_catalog_parquet = processed.get("movie_catalog_parquet")
        if not isinstance(movie_catalog_parquet, str):
            raise ValueError("offline_manifest.json missing outputs.processed.movie_catalog_parquet")

        dense_faiss_index = dense.get("dense_faiss_index")
        dense_movie_ids = dense.get("dense_movie_ids")
        if not isinstance(dense_faiss_index, str) or not isinstance(dense_movie_ids, str):
            raise ValueError("offline_manifest.json missing outputs.dense.{dense_faiss_index,dense_movie_ids}")

        bm25_index_path = bm25.get("bm25_index")
        if not isinstance(bm25_index_path, str):
            raise ValueError("offline_manifest.json missing outputs.bm25.bm25_index")

        self.movie_catalog_path = _resolve_path(self.repo_root, movie_catalog_parquet)
        self.dense_faiss_index_path = _resolve_path(self.repo_root, dense_faiss_index)
        self.dense_movie_ids_path = _resolve_path(self.repo_root, dense_movie_ids)
        self.bm25_index_path = _resolve_path(self.repo_root, bm25_index_path)

        # ----- Config knobs (with safe defaults) -----
        ranking_cfg = self.config.get("ranking", {}) if isinstance(self.config.get("ranking"), dict) else {}
        online_cfg = self.config.get("online", {}) if isinstance(self.config.get("online"), dict) else {}

        self.cross_encoder_model_name = str(ranking_cfg.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        self.rerank_top_k_default = int(ranking_cfg.get("rerank_top_k", 200))

        default_mode_raw = str(online_cfg.get("default_mode", "hybrid"))
        if default_mode_raw not in ("dense", "hybrid"):
            raise ValueError(f"config.yaml online.default_mode must be 'dense' or 'hybrid', got {default_mode_raw!r}")
        self.default_mode: Mode = default_mode_raw  # type: ignore[assignment]
        self.dense_top_k_default = int(online_cfg.get("dense_top_k", 200))
        self.bm25_top_k_default = int(online_cfg.get("bm25_top_k", 200))
        self.fused_top_k_default = int(online_cfg.get("fused_top_k", 300))
        self.output_top_k_default = int(online_cfg.get("output_top_k", 10))
        self.rrf_k_default = int(online_cfg.get("rrf_k", 60))

        title_res_cfg = online_cfg.get("title_resolution", {}) if isinstance(online_cfg.get("title_resolution"), dict) else {}
        self.title_resolution_enabled_by_default = bool(title_res_cfg.get("enabled_by_default", True))
        self.title_resolution_min_similarity = float(title_res_cfg.get("min_similarity", 0.85))

        # Retrieval config (prefer manifest snapshot if present).
        retrieval_cfg = self.manifest_config.get("retrieval") if isinstance(self.manifest_config.get("retrieval"), dict) else {}
        retrieval_cfg = retrieval_cfg if isinstance(retrieval_cfg, dict) else {}

        dense_cfg = retrieval_cfg.get("dense") if isinstance(retrieval_cfg.get("dense"), dict) else None
        if dense_cfg is None:
            dense_cfg = self.config.get("retrieval", {}).get("dense", {}) if isinstance(self.config.get("retrieval"), dict) else {}

        self.bi_encoder_model_name = str(dense_cfg.get("bi_encoder_model", "sentence-transformers/all-MiniLM-L6-v2"))
        self.normalize_embeddings = bool(dense_cfg.get("normalize_embeddings", True))

        # ----- Load catalog + retrieval artifacts -----
        t0 = time.perf_counter()
        self.catalog = MovieCatalog.from_parquet(self.movie_catalog_path)
        logger.info("Loaded movie catalog rows=%d from %s", len(self.catalog.df), self.movie_catalog_path)
        t_catalog = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.dense_retriever = DenseRetriever.from_artifacts(
            faiss_index_path=self.dense_faiss_index_path,
            dense_movie_ids_path=self.dense_movie_ids_path,
            normalize_embeddings=self.normalize_embeddings,
        )
        t_dense = time.perf_counter() - t0
        logger.info(
            "Loaded dense artifacts: index_ntotal=%d dim=%d",
            int(self.dense_retriever.index.ntotal),
            int(self.dense_retriever.dim),
        )

        t0 = time.perf_counter()
        self.bm25_retriever = BM25Retriever.from_artifact(self.bm25_index_path)
        t_bm25 = time.perf_counter() - t0
        logger.info("Loaded BM25 artifacts: docs=%d", len(self.bm25_retriever.index.movie_ids))

        # ----- Models (can be injected for tests) -----
        self._embed_lock = Lock()

        if bi_encoder is None:
            t0 = time.perf_counter()
            self.bi_encoder = load_bi_encoder(self.bi_encoder_model_name, device=None)
            t_bienc = time.perf_counter() - t0
            logger.info("Loaded bi-encoder model=%s (%.2fs)", self.bi_encoder_model_name, t_bienc)
        else:
            self.bi_encoder = bi_encoder

        if cross_encoder is None:
            self.reranker = CrossEncoderReranker(model_name=self.cross_encoder_model_name)
        else:
            # Allow injecting a model-like object in tests.
            self.reranker = CrossEncoderReranker(model_name=self.cross_encoder_model_name, model=cross_encoder)

        # Per-instance LRU caches.
        self._embed_cached = self._make_embedding_cache(maxsize=2048)
        self._resolve_title_cached = self._make_title_resolution_cache(maxsize=4096)

        logger.info(
            "Startup load times: catalog=%.2fs dense=%.2fs bm25=%.2fs",
            t_catalog,
            t_dense,
            t_bm25,
        )

    def _make_embedding_cache(self, *, maxsize: int) -> Any:
        """Create an LRU cache function for query embeddings (keyed by query_text)."""

        @lru_cache(maxsize=int(maxsize))
        def _embed(query_text: str) -> Any:
            with self._embed_lock:
                emb = embed_texts(
                    self.bi_encoder,
                    [str(query_text)],
                    batch_size=1,
                    normalize_embeddings=self.normalize_embeddings,
                )
            return emb

        return _embed

    def _make_title_resolution_cache(self, *, maxsize: int) -> Any:
        """Create an LRU cache for title resolution results."""

        min_sim = float(self.title_resolution_min_similarity)

        @lru_cache(maxsize=int(maxsize))
        def _resolve(query: str) -> tuple[int | None, float]:
            return self.catalog.resolve_title_to_movie_id(query, min_similarity=min_sim)

        return _resolve

    def has_movie_id(self, movie_id: int) -> bool:
        """Return True if `movie_id` exists in the catalog."""
        return self.catalog.has_movie(int(movie_id))

    def resolve_query(self, query: str, *, resolve_title: bool = True) -> dict[str, Any]:
        """Resolve a user query into the retrieval/reranking query text.

        Rules:
        - If query is digits and exists as a movieId, treat as movieId query.
        - Else, if resolve_title=True, attempt exact then fuzzy match against `title_clean`.
          If resolved, use the resolved movie's `movie_profile` as the query_text.
        - Else fall back to free text.
        """
        display_query = "" if query is None else str(query)
        q = display_query.strip()
        if q == "":
            raise ValueError("query must be non-empty")

        if q.isdigit():
            mid = int(q)
            if self.catalog.has_movie(mid):
                return {
                    "query_type": "movieId",
                    "resolved_movieId": mid,
                    "query_text": self.catalog.get_movie_profile(mid),
                    "display_query": display_query,
                }

        if resolve_title:
            resolved_mid, sim = self._resolve_title_cached(q)
            if resolved_mid is not None:
                return {
                    "query_type": "movieId",
                    "resolved_movieId": int(resolved_mid),
                    "query_text": self.catalog.get_movie_profile(int(resolved_mid)),
                    "display_query": display_query,
                    "title_resolution": {"similarity": float(sim)},
                }

        return {
            "query_type": "text",
            "resolved_movieId": None,
            "query_text": q,
            "display_query": display_query,
        }

    def retrieve_dense(self, query_text: str, *, top_k: int) -> list[dict[str, Any]]:
        """Dense retrieval using a cached bi-encoder embedding + FAISS."""
        emb = self._embed_cached(str(query_text))
        result = self.dense_retriever.search_embedding(emb, top_k=int(top_k))
        out: list[dict[str, Any]] = []
        for mid, score in zip(result.movie_ids, result.scores):
            out.append({"movieId": int(mid), "dense_score": float(score)})
        return out

    def retrieve_bm25(self, query_text: str, *, top_k: int) -> list[dict[str, Any]]:
        """BM25 lexical retrieval."""
        result = self.bm25_retriever.search(str(query_text), top_k=int(top_k))
        out: list[dict[str, Any]] = []
        for mid, score in zip(result.movie_ids, result.scores):
            out.append({"movieId": int(mid), "bm25_score": float(score)})
        return out

    def generate_candidates(
        self,
        *,
        query_text: str,
        query_type: str,
        resolved_movieId: int | None,
        mode: Mode,
        dense_top_k: int,
        bm25_top_k: int,
        fused_top_k: int,
        rrf_k: int,
        query_embedding: Any | None = None,
        timings: dict[str, float] | None = None,
    ) -> list[Candidate]:
        """Generate retrieval candidates for reranking."""
        if mode not in ("dense", "hybrid"):
            raise ValueError(f"Unsupported mode: {mode!r} (expected: dense, hybrid)")

        candidates: dict[int, Candidate] = {}

        if query_embedding is None and mode in ("dense", "hybrid"):
            query_embedding = self._embed_cached(str(query_text))

        dense_ids: list[int] = []
        dense_scores: dict[int, float] = {}
        if mode in ("dense", "hybrid"):
            t0 = time.perf_counter()
            dense_result = self.dense_retriever.search_embedding(query_embedding, top_k=int(dense_top_k))
            if timings is not None:
                timings["dense_search_s"] = time.perf_counter() - t0

            dense_ids = [int(mid) for mid in dense_result.movie_ids]
            dense_scores = {int(mid): float(score) for mid, score in zip(dense_result.movie_ids, dense_result.scores)}

        if mode == "dense":
            for mid in dense_ids:
                if not self.catalog.has_movie(int(mid)):
                    continue
                candidates[mid] = Candidate(
                    movieId=mid,
                    dense_score=float(dense_scores[int(mid)]),
                    retrieval_sources=["dense"],
                )

            out = list(candidates.values())
            out.sort(key=lambda c: float(c.dense_score or float("-inf")), reverse=True)
        else:  # mode == "hybrid"
            t0 = time.perf_counter()
            bm25_result = self.bm25_retriever.search(str(query_text), top_k=int(bm25_top_k))
            if timings is not None:
                timings["bm25_s"] = time.perf_counter() - t0

            bm25_ids = [int(mid) for mid in bm25_result.movie_ids]
            bm25_scores = {int(mid): float(score) for mid, score in zip(bm25_result.movie_ids, bm25_result.scores)}

            t0 = time.perf_counter()
            fused = reciprocal_rank_fusion([dense_ids, bm25_ids], k=int(rrf_k), max_items=int(fused_top_k))
            if timings is not None:
                timings["fuse_s"] = time.perf_counter() - t0
            for mid, fused_score in fused:
                if not self.catalog.has_movie(int(mid)):
                    continue
                sources: list[str] = []
                cand = Candidate(movieId=int(mid), fused_score=float(fused_score))
                if int(mid) in dense_scores:
                    cand.dense_score = float(dense_scores[int(mid)])
                    sources.append("dense")
                if int(mid) in bm25_scores:
                    cand.bm25_score = float(bm25_scores[int(mid)])
                    sources.append("bm25")
                cand.retrieval_sources = sources
                candidates[int(mid)] = cand

            out = list(candidates.values())
            out.sort(key=lambda c: float(c.fused_score or float("-inf")), reverse=True)

        # If the query resolves to a specific movie, do not recommend itself.
        if query_type == "movieId" and resolved_movieId is not None:
            out = [c for c in out if int(c.movieId) != int(resolved_movieId)]

        return out

    def rerank(
        self,
        *,
        query_text: str,
        candidates: list[Candidate],
        rerank_top_k: int,
        batch_size: int = 32,
    ) -> list[Candidate]:
        """Cross-encoder reranking over the top candidates."""
        if not candidates:
            return []

        # Score only the top rerank_top_k by retrieval score.
        def _base_score(c: Candidate) -> float:
            if c.fused_score is not None:
                return float(c.fused_score)
            if c.dense_score is not None:
                return float(c.dense_score)
            if c.bm25_score is not None:
                return float(c.bm25_score)
            return float("-inf")

        candidates_sorted = sorted(candidates, key=_base_score, reverse=True)
        k = int(min(int(rerank_top_k), len(candidates_sorted)))
        to_score = candidates_sorted[:k]

        candidate_texts = [self.catalog.get_movie_profile(int(c.movieId)) for c in to_score]
        scores = self.reranker.rerank(str(query_text), candidate_texts, batch_size=int(batch_size))
        for cand, score in zip(to_score, list(scores)):
            cand.rerank_score = float(score)

        # Final sort: rerank_score desc, then weighted_rating/rating_count if present.
        def _tie_breakers(movie_id: int) -> tuple[float, float]:
            row = self.catalog.get_row(int(movie_id))
            wr = row.get("weighted_rating", None)
            rc = row.get("rating_count", None)
            wr_v = 0.0
            if wr is not None and not pd.isna(wr):
                try:
                    wr_v = float(wr)
                except Exception:
                    wr_v = 0.0

            rc_v = 0.0
            if rc is not None and not pd.isna(rc):
                try:
                    rc_v = float(rc)
                except Exception:
                    rc_v = 0.0
            return wr_v, rc_v

        to_score.sort(
            key=lambda c: (
                float(c.rerank_score if c.rerank_score is not None else float("-inf")),
                *_tie_breakers(int(c.movieId)),
            ),
            reverse=True,
        )
        return to_score

    def recommend(
        self,
        *,
        query: str,
        k: int | None = None,
        mode: Mode | None = None,
        resolve_title: bool | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:
        """Run two-stage recommendation.

        Parameters
        ----------
        query:
            Movie title / keywords, or a movieId-like string.
        k:
            Number of results to return.
        mode:
            Override retrieval mode ("dense" or "hybrid").
        resolve_title:
            If True, attempt to resolve a title to a movieId and use that movie's
            profile as the query. If None, falls back to config default.
        debug:
            If True, include `debug_info` in the response.
        """

        resolve_title_final: bool = (
            bool(resolve_title) if resolve_title is not None else bool(self.title_resolution_enabled_by_default)
        )
        mode_final: Mode = (mode if mode is not None else self.default_mode)

        k_out = int(k if k is not None else self.output_top_k_default)
        k_out = max(1, min(50, k_out))

        t_start = time.perf_counter()
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        resolved = self.resolve_query(query, resolve_title=bool(resolve_title_final))
        timings["resolve_s"] = time.perf_counter() - t0
        query_text = str(resolved["query_text"])
        query_type = str(resolved["query_type"])
        resolved_mid = resolved.get("resolved_movieId")
        resolved_mid_int = int(resolved_mid) if resolved_mid is not None else None

        dense_top_k = int(self.dense_top_k_default)
        bm25_top_k = int(self.bm25_top_k_default)
        fused_top_k = int(self.fused_top_k_default)
        rrf_k = int(self.rrf_k_default)
        rerank_top_k = int(max(self.rerank_top_k_default, k_out))

        t0 = time.perf_counter()
        query_embedding = self._embed_cached(query_text)
        timings["embed_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        candidates = self.generate_candidates(
            query_text=query_text,
            query_type=query_type,
            resolved_movieId=resolved_mid_int,
            mode=mode_final,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            fused_top_k=fused_top_k,
            rrf_k=rrf_k,
            query_embedding=query_embedding,
            timings=timings,
        )
        timings["candidate_gen_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        reranked = self.rerank(query_text=query_text, candidates=candidates, rerank_top_k=rerank_top_k)
        timings["rerank_s"] = time.perf_counter() - t0

        # Explainability counts for movieId-based queries.
        query_genres: set[str] = set()
        query_tags: set[str] = set()
        if query_type == "movieId" and resolved_mid_int is not None:
            row = self.catalog.get_row(int(resolved_mid_int))
            query_genres = _parse_genres(str(row.get("genres", "")))
            if "tags_top_text" in self.catalog.df.columns:
                query_tags = _parse_tags(str(row.get("tags_top_text", "")))

        results: list[dict[str, Any]] = []
        for cand in reranked[:k_out]:
            fields = self.catalog.get_display_fields(int(cand.movieId))
            item: dict[str, Any] = {**fields}

            if query_type == "movieId" and resolved_mid_int is not None:
                row = self.catalog.get_row(int(cand.movieId))
                cand_genres = _parse_genres(str(row.get("genres", "")))
                item["genre_overlap_count"] = int(len(query_genres & cand_genres)) if query_genres else 0

                if query_tags and "tags_top_text" in self.catalog.df.columns:
                    cand_tags = _parse_tags(str(row.get("tags_top_text", "")))
                    item["tag_overlap_count"] = int(len(query_tags & cand_tags))
                else:
                    item["tag_overlap_count"] = None

            results.append(item)

        total_s = time.perf_counter() - t_start
        timings["total_s"] = total_s

        logger.info(
            "recommend query_type=%s resolved_movieId=%s mode=%s k=%d | dense_top_k=%d bm25_top_k=%d fused_top_k=%d rerank_top_k=%d | timings_s=%s",
            query_type,
            resolved_mid_int,
            mode_final,
            k_out,
            dense_top_k,
            bm25_top_k,
            fused_top_k,
            rerank_top_k,
            {k: round(v, 4) for k, v in timings.items()},
        )

        out: dict[str, Any] = {
            "query": str(resolved.get("display_query", query)),
            "query_type": query_type,
            "resolved_movieId": resolved_mid_int,
            "mode": str(mode_final),
            "k": k_out,
            "results": results,
        }

        if debug:
            out["debug_info"] = {
                "timings_s": {k: float(v) for k, v in timings.items()},
                "candidate_count": int(len(candidates)),
                "reranked_count": int(min(int(rerank_top_k), len(candidates))),
                "dense_top_k": int(dense_top_k),
                "bm25_top_k": int(bm25_top_k),
                "fused_top_k": int(fused_top_k),
                "rrf_k": int(rrf_k),
            }

        return out

    def search_titles(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Search titles in the catalog (UI helper endpoint)."""
        return self.catalog.search_titles(query, limit=int(limit))
