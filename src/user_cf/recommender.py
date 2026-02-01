from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data import load_raw_data
from ..paths import ProjectPaths, get_repo_root
from ..utils import setup_logging
from .model import MatrixFactorization
from .train import UserCFTrainConfig, train_user_cf


logger = logging.getLogger(__name__)


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of a and rows of b."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0.0] = 1.0
    b_norm[b_norm == 0.0] = 1.0
    a_n = a / a_norm
    b_n = b / b_norm
    return a_n @ b_n.T


@dataclass(frozen=True)
class SimilarUser:
    userId: int
    similarity: float
    common_rated: int


@dataclass(frozen=True)
class RecommendedMovie:
    movieId: int
    score: float
    title: str | None = None
    genres: str | None = None


class UserUserCFRecommender:
    """User-user CF recommender based on learned user embeddings.

    Loads artifacts from `artifacts/user_cf/` by default (trains if missing).
    """

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        artifacts_dir: Path | None = None,
        device: str | None = None,
        allow_train_if_missing: bool = True,
    ) -> None:
        setup_logging("INFO")
        repo_root = get_repo_root()
        self.repo_root = repo_root
        self.config_path = (Path(config_path).resolve() if config_path else (repo_root / "config.yaml"))

        self.artifacts_dir = (
            Path(artifacts_dir).resolve()
            if artifacts_dir is not None
            else (repo_root / "artifacts" / "user_cf").resolve()
        )
        self.device = device
        self.allow_train_if_missing = bool(allow_train_if_missing)

        self._load_or_build()

    def _artifacts_exist(self) -> bool:
        return (
            (self.artifacts_dir / "user_cf_model.pt").exists()
            and (self.artifacts_dir / "user_classes.npy").exists()
            and (self.artifacts_dir / "movie_classes.npy").exists()
        )

    def _load_or_build(self) -> None:
        # Load raw data (ratings + movies) for recommendation-time joins.
        paths = ProjectPaths.from_repo_root(self.repo_root)
        data = load_raw_data(paths.raw_dir)
        self.ratings = data.ratings[["userId", "movieId", "rating"]].copy()
        self.movies = data.movies[["movieId", "title", "genres"]].copy()

        if not self._artifacts_exist():
            if not self.allow_train_if_missing:
                raise FileNotFoundError(
                    f"UserCF artifacts missing under {self.artifacts_dir}. Run the build pipeline first."
                )

            logger.info("UserCF artifacts not found in %s — training now", self.artifacts_dir)

            # Best-effort defaults if config.yaml lacks user_cf section.
            cfg = UserCFTrainConfig()
            train_user_cf(self.ratings, out_dir=self.artifacts_dir, cfg=cfg, device=self.device)

        # ----- Load artifacts -----
        ckpt = torch.load(self.artifacts_dir / "user_cf_model.pt", map_location="cpu")
        self.mean_rating = float(ckpt.get("mean_rating", 0.0))
        self.n_users = int(ckpt["n_users"])
        self.n_movies = int(ckpt["n_movies"])
        self.embed_dim = int(ckpt["embed_dim"])

        self.user_classes = np.load(self.artifacts_dir / "user_classes.npy")
        self.movie_classes = np.load(self.artifacts_dir / "movie_classes.npy")

        # maps raw ids -> encoded indices
        self.user_to_idx = {int(u): int(i) for i, u in enumerate(self.user_classes.tolist())}
        self.movie_to_idx = {int(m): int(i) for i, m in enumerate(self.movie_classes.tolist())}

        # Build model and load weights
        self.model = MatrixFactorization(n_users=self.n_users, n_movies=self.n_movies, embed_dim=self.embed_dim)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        # Cache user embedding matrix for similarity queries.
        with torch.no_grad():
            u = self.model.user_embed.weight.detach().cpu().numpy().astype(np.float32)
        self.user_embeds = u

        # Precompute rated sets for fast filtering.
        self._user_rated_movies: dict[int, set[int]] = {}
        for uid, grp in self.ratings.groupby("userId"):
            self._user_rated_movies[int(uid)] = set(grp["movieId"].astype(int).tolist())

        # For "common_rated" stats.
        self._user_movie_set_by_idx: list[set[int]] = [set() for _ in range(self.n_users)]
        for uid, grp in self.ratings.groupby("userId"):
            uid_int = int(uid)
            idx = self.user_to_idx.get(uid_int)
            if idx is None:
                continue
            self._user_movie_set_by_idx[idx] = set(grp["movieId"].astype(int).tolist())

        logger.info(
            "UserCF loaded: users=%d movies=%d ratings=%d artifacts_dir=%s",
            self.n_users,
            self.n_movies,
            len(self.ratings),
            self.artifacts_dir,
        )

        meta_path = self.artifacts_dir / "user_cf_meta.json"
        if meta_path.exists():
            try:
                self.meta = json.loads(meta_path.read_text())
            except Exception:
                self.meta = None
        else:
            self.meta = None

    def has_user(self, userId: int) -> bool:
        return int(userId) in self.user_to_idx

    def similar_users(self, userId: int, *, top_n: int = 10, min_common_rated: int = 2) -> list[SimilarUser]:
        uid = int(userId)
        if uid not in self.user_to_idx:
            raise KeyError(f"Unknown userId: {uid}")

        uidx = self.user_to_idx[uid]
        target = self.user_embeds[uidx : uidx + 1]
        sims = _cosine_sim_matrix(target, self.user_embeds).reshape(-1)

        # Exclude self
        sims[uidx] = -1.0

        # Rank by similarity
        order = np.argsort(-sims)

        out: list[SimilarUser] = []
        target_movies = self._user_movie_set_by_idx[uidx]
        for j in order:
            if len(out) >= int(top_n):
                break
            sim = float(sims[int(j)])
            other_userId = int(self.user_classes[int(j)])
            common = len(target_movies & self._user_movie_set_by_idx[int(j)])
            if common < int(min_common_rated):
                continue
            out.append(SimilarUser(userId=other_userId, similarity=sim, common_rated=int(common)))
        return out

    def recommend_movies(
        self,
        userId: int,
        *,
        k: int = 10,
        top_n_sim_users: int = 25,
        min_common_rated: int = 2,
        min_rating_liked: float = 4.0,
    ) -> list[RecommendedMovie]:
        """Recommend movies for a user based on movies liked by similar users.

        Scoring:
        - Find similar users (cosine on user embedding)
        - Candidate movies: those similar users rated >= min_rating_liked
          and the target user has NOT rated
        - Final score: similarity-weighted average of those neighbor ratings
        """
        uid = int(userId)
        if uid not in self.user_to_idx:
            raise KeyError(f"Unknown userId: {uid}")

        sims = self.similar_users(uid, top_n=int(top_n_sim_users), min_common_rated=int(min_common_rated))
        if not sims:
            return []

        seen = self._user_rated_movies.get(uid, set())
        neighbor_ids = [s.userId for s in sims]
        sim_w = {int(s.userId): float(s.similarity) for s in sims}

        neigh_r = self.ratings[self.ratings["userId"].isin(neighbor_ids)].copy()
        neigh_r = neigh_r[neigh_r["rating"] >= float(min_rating_liked)]

        # Exclude already-seen movies
        if seen:
            neigh_r = neigh_r[~neigh_r["movieId"].isin(list(seen))]

        if neigh_r.empty:
            return []

        # similarity-weighted rating
        neigh_r["w"] = neigh_r["userId"].map(sim_w).astype(float)
        neigh_r["weighted"] = neigh_r["w"] * neigh_r["rating"].astype(float)

        agg = neigh_r.groupby("movieId", as_index=False).agg(
            score=("weighted", "sum"),
            w_sum=("w", "sum"),
            support=("userId", "nunique"),
        )
        agg = agg[agg["w_sum"] > 0.0]
        agg["score"] = agg["score"] / agg["w_sum"]

        agg = agg.sort_values(["score", "support"], ascending=[False, False]).head(int(k))

        # Join titles/genres for display
        enriched = agg.merge(self.movies, on="movieId", how="left")

        out: list[RecommendedMovie] = []
        for _, row in enriched.iterrows():
            out.append(
                RecommendedMovie(
                    movieId=int(row["movieId"]),
                    score=float(row["score"]),
                    title=(None if pd.isna(row.get("title")) else str(row.get("title"))),
                    genres=(None if pd.isna(row.get("genres")) else str(row.get("genres"))),
                )
            )
        return out
