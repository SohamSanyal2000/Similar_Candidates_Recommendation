from __future__ import annotations

import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """Simple MF model: dot(user_emb, movie_emb) + biases.

    This is a more standard variant of the notebook's embedding approach and is
    well-suited for extracting user embeddings for similarity search.
    """

    def __init__(self, n_users: int, n_movies: int, *, embed_dim: int = 32) -> None:
        super().__init__()
        self.user_embed = nn.Embedding(int(n_users), int(embed_dim))
        self.movie_embed = nn.Embedding(int(n_movies), int(embed_dim))

        self.user_bias = nn.Embedding(int(n_users), 1)
        self.movie_bias = nn.Embedding(int(n_movies), 1)

        # A learnable global bias (initialized to 0; we also add a fixed mean rating externally).
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Init similar to common MF baselines
        nn.init.normal_(self.user_embed.weight, std=0.02)
        nn.init.normal_(self.movie_embed.weight, std=0.02)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_embed(user_idx)
        m = self.movie_embed(movie_idx)
        dot = (u * m).sum(dim=1, keepdim=True)
        out = dot + self.user_bias(user_idx) + self.movie_bias(movie_idx) + self.global_bias
        return out
