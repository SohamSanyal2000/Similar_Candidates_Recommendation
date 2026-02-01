"""User-user collaborative filtering (similar rating patterns) using MovieLens ratings.

This module implements the assignment:
"Finds users who have similar rating patterns [use ratings.csv]".

Core idea:
- Train a simple matrix-factorization model in PyTorch from (userId, movieId, rating)
- Use the learned user embeddings to find similar users (cosine similarity)
- Recommend movies liked by similar users that the target user has not seen
"""
