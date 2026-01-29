from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class RawMovieLensData:
    movies: pd.DataFrame
    ratings: pd.DataFrame
    tags: pd.DataFrame
    links: pd.DataFrame


REQUIRED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "movies": ("movieId", "title", "genres"),
    "ratings": ("userId", "movieId", "rating", "timestamp"),
    "tags": ("userId", "movieId", "tag", "timestamp"),
    "links": ("movieId", "imdbId", "tmdbId"),
}


def load_raw_data(raw_dir: Path) -> RawMovieLensData:
    """Load MovieLens CSV files from a directory.

    Notes
    -----
    We set dtypes explicitly for:
    - reproducibility (consistent downstream behavior)
    - correctness (imdbId contains leading zeros and must be read as string)
    - mild memory efficiency
    """
    movies = pd.read_csv(
        raw_dir / "movies.csv",
        dtype={"movieId": "int64", "title": "string", "genres": "string"},
    )
    ratings = pd.read_csv(
        raw_dir / "ratings.csv",
        dtype={"userId": "int64", "movieId": "int64", "rating": "float64", "timestamp": "int64"},
    )
    tags = pd.read_csv(
        raw_dir / "tags.csv",
        dtype={"userId": "int64", "movieId": "int64", "tag": "string", "timestamp": "int64"},
    )
    links = pd.read_csv(
        raw_dir / "links.csv",
        dtype={"movieId": "int64", "imdbId": "string", "tmdbId": "Int64"},
    )

    data = RawMovieLensData(movies=movies, ratings=ratings, tags=tags, links=links)
    validate_schema(data)
    return data


def validate_schema(data: RawMovieLensData) -> None:
    """Validate that all required columns exist and basic constraints hold."""
    for name, cols in REQUIRED_COLUMNS.items():
        df = getattr(data, name)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name}.csv missing columns: {missing}")

    # Basic constraints
    if data.movies["movieId"].duplicated().any():
        raise ValueError("movies.csv has duplicate movieId values")

    if data.links["movieId"].duplicated().any():
        raise ValueError("links.csv has duplicate movieId values")

    # Movie IDs should be consistent (ratings/tags/links should reference movies)
    movie_ids = set(data.movies["movieId"].astype(int).tolist())
    bad_ratings = set(data.ratings["movieId"].astype(int).tolist()) - movie_ids
    bad_tags = set(data.tags["movieId"].astype(int).tolist()) - movie_ids
    bad_links = set(data.links["movieId"].astype(int).tolist()) - movie_ids

    if bad_ratings:
        raise ValueError(f"ratings.csv has {len(bad_ratings)} movieIds not in movies.csv")
    if bad_tags:
        raise ValueError(f"tags.csv has {len(bad_tags)} movieIds not in movies.csv")
    if bad_links:
        raise ValueError(f"links.csv has {len(bad_links)} movieIds not in movies.csv")
