from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from .model import MatrixFactorization


logger = logging.getLogger(__name__)


class RatingsDataset(Dataset):
    def __init__(self, user_idx: np.ndarray, movie_idx: np.ndarray, rating: np.ndarray) -> None:
        self.user_idx = user_idx.astype(np.int64, copy=False)
        self.movie_idx = movie_idx.astype(np.int64, copy=False)
        self.rating = rating.astype(np.float32, copy=False)

    def __len__(self) -> int:  # pragma: no cover
        return int(len(self.user_idx))

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "users": torch.tensor(self.user_idx[i], dtype=torch.long),
            "movies": torch.tensor(self.movie_idx[i], dtype=torch.long),
            "ratings": torch.tensor(self.rating[i], dtype=torch.float32),
        }


@dataclass(frozen=True)
class UserCFTrainConfig:
    embed_dim: int = 32
    epochs: int = 5
    batch_size: int = 1024
    lr: float = 1e-2
    weight_decay: float = 0.0
    test_size: float = 0.1
    random_state: int = 42
    min_rating: float = 0.5
    max_rating: float = 5.0


@dataclass(frozen=True)
class UserCFArtifacts:
    model_path: Path
    user_classes_path: Path
    movie_classes_path: Path
    meta_path: Path


def _device_from_str(device: str | None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(str(device))


def train_user_cf(
    ratings: pd.DataFrame,
    *,
    out_dir: Path,
    cfg: UserCFTrainConfig,
    device: str | None = None,
) -> UserCFArtifacts:
    """Train MF model from MovieLens ratings and persist artifacts.

    Expected ratings columns: userId, movieId, rating
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {"userId", "movieId", "rating"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings missing required columns: {sorted(missing)}")

    df = ratings[["userId", "movieId", "rating"]].copy()
    df = df.dropna(subset=["userId", "movieId", "rating"]).reset_index(drop=True)
    df["rating"] = df["rating"].astype(float)

    # Label encoders to map raw ids => contiguous indices [0..n)
    le_user = LabelEncoder()
    le_movie = LabelEncoder()
    df["user_idx"] = le_user.fit_transform(df["userId"].astype(np.int64).values)
    df["movie_idx"] = le_movie.fit_transform(df["movieId"].astype(np.int64).values)

    n_users = int(df["user_idx"].nunique())
    n_movies = int(df["movie_idx"].nunique())
    mean_rating = float(df["rating"].mean())
    logger.info("UserCF: users=%d movies=%d ratings=%d mean_rating=%.4f", n_users, n_movies, len(df), mean_rating)

    # Stratified split is nice for MovieLens (many samples per rating bucket), but it
    # breaks for tiny datasets/tests where some buckets may have only 1 sample.
    stratify = df["rating"].astype(str).values
    try:
        counts = pd.Series(stratify).value_counts()
        if int(counts.min()) < 2:
            stratify = None
    except Exception:
        stratify = None

    df_train, df_test = model_selection.train_test_split(
        df,
        test_size=float(cfg.test_size),
        random_state=int(cfg.random_state),
        stratify=stratify,
    )

    train_ds = RatingsDataset(
        df_train["user_idx"].to_numpy(),
        df_train["movie_idx"].to_numpy(),
        df_train["rating"].to_numpy(),
    )
    test_ds = RatingsDataset(
        df_test["user_idx"].to_numpy(),
        df_test["movie_idx"].to_numpy(),
        df_test["rating"].to_numpy(),
    )

    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)

    torch_device = _device_from_str(device)
    model = MatrixFactorization(n_users=n_users, n_movies=n_movies, embed_dim=int(cfg.embed_dim)).to(torch_device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    loss_fn = torch.nn.MSELoss()

    def _clamp(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=float(cfg.min_rating), max=float(cfg.max_rating))

    logger.info("UserCF training on device=%s epochs=%d batch_size=%d", torch_device, int(cfg.epochs), int(cfg.batch_size))
    model.train()
    for epoch in range(int(cfg.epochs)):
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            users = batch["users"].to(torch_device)
            movies = batch["movies"].to(torch_device)
            ratings_t = batch["ratings"].to(torch_device).view(-1, 1)

            # Add back mean_rating so global_bias learns residuals.
            preds = model(users, movies) + mean_rating
            preds = _clamp(preds)

            loss = loss_fn(preds, ratings_t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = int(users.shape[0])
            total_loss += float(loss.item()) * bs
            n += bs

        train_mse = total_loss / max(1, n)
        train_rmse = float(np.sqrt(train_mse))

        # quick eval
        model.eval()
        test_total = 0.0
        test_n = 0
        with torch.no_grad():
            for batch in test_loader:
                users = batch["users"].to(torch_device)
                movies = batch["movies"].to(torch_device)
                ratings_t = batch["ratings"].to(torch_device).view(-1, 1)
                preds = _clamp(model(users, movies) + mean_rating)
                loss = loss_fn(preds, ratings_t)
                bs = int(users.shape[0])
                test_total += float(loss.item()) * bs
                test_n += bs

        test_mse = test_total / max(1, test_n)
        test_rmse = float(np.sqrt(test_mse))
        logger.info("UserCF epoch=%d train_rmse=%.4f test_rmse=%.4f", epoch + 1, train_rmse, test_rmse)
        model.train()

    # ----- Save artifacts -----
    model_path = out_dir / "user_cf_model.pt"
    user_classes_path = out_dir / "user_classes.npy"
    movie_classes_path = out_dir / "movie_classes.npy"
    meta_path = out_dir / "user_cf_meta.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_users": n_users,
            "n_movies": n_movies,
            "embed_dim": int(cfg.embed_dim),
            "mean_rating": mean_rating,
        },
        model_path,
    )
    np.save(user_classes_path, le_user.classes_.astype(np.int64), allow_pickle=False)
    np.save(movie_classes_path, le_movie.classes_.astype(np.int64), allow_pickle=False)

    meta = {
        "n_users": n_users,
        "n_movies": n_movies,
        "embed_dim": int(cfg.embed_dim),
        "mean_rating": mean_rating,
        "train_config": {
            "epochs": int(cfg.epochs),
            "batch_size": int(cfg.batch_size),
            "lr": float(cfg.lr),
            "weight_decay": float(cfg.weight_decay),
            "test_size": float(cfg.test_size),
            "random_state": int(cfg.random_state),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    return UserCFArtifacts(
        model_path=model_path,
        user_classes_path=user_classes_path,
        movie_classes_path=movie_classes_path,
        meta_path=meta_path,
    )
