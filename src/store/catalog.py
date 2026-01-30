"""Movie catalog loading and title-resolution utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import pandas as pd


_TITLE_YEAR_SUFFIX_RE = re.compile(r"\(\d{4}\)\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_MULTISPACE_RE = re.compile(r"\s+")


def normalize_title(text: str) -> str:
    """Normalize title-ish text for matching.

    - Lowercase
    - Strip
    - Remove trailing "(YYYY)"
    - Remove punctuation (keep a-z, 0-9)
    - Collapse whitespace
    """
    text = "" if text is None else str(text)
    text = text.strip().lower()
    text = _TITLE_YEAR_SUFFIX_RE.sub("", text).strip()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def load_movie_catalog(parquet_path: Path) -> pd.DataFrame:
    """Load the canonical per-movie catalog table produced offline."""
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"movie_catalog.parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if "movieId" not in df.columns:
        raise ValueError("movie_catalog.parquet must contain column 'movieId'")

    df = df.copy()
    df["movieId"] = df["movieId"].astype("int64")
    for col in ["title", "title_clean", "genres", "movie_profile"]:
        if col not in df.columns:
            raise ValueError(f"movie_catalog.parquet missing required column: {col!r}")
        df[col] = df[col].astype("string")

    if "year" in df.columns:
        # Preserve nullability.
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    if "tmdbId" in df.columns:
        df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce").astype("Int64")

    if "imdbId" in df.columns:
        # imdbId may contain leading zeros; keep as string.
        df["imdbId"] = df["imdbId"].astype("string")

    return df


@dataclass(frozen=True)
class MovieCatalog:
    """In-memory view of the movie catalog with fast lookup maps."""

    df: pd.DataFrame
    movie_id_to_row: dict[int, int]
    norm_title_to_movie_ids: dict[str, list[int]]

    @classmethod
    def from_parquet(cls, parquet_path: Path) -> "MovieCatalog":
        """Load `movie_catalog.parquet` and build lookup indexes."""
        df = load_movie_catalog(parquet_path)
        df = df.reset_index(drop=True)

        movie_ids = df["movieId"].astype("int64").tolist()
        movie_id_to_row = {int(mid): int(i) for i, mid in enumerate(movie_ids)}

        norm_title_to_movie_ids: dict[str, list[int]] = {}
        title_clean = df["title_clean"].astype("string").fillna("").tolist()
        for mid, title in zip(movie_ids, title_clean):
            key = normalize_title(str(title))
            if not key:
                continue
            norm_title_to_movie_ids.setdefault(key, []).append(int(mid))

        return cls(df=df, movie_id_to_row=movie_id_to_row, norm_title_to_movie_ids=norm_title_to_movie_ids)

    def has_movie(self, movie_id: int) -> bool:
        """Return True if the catalog contains `movie_id`."""
        return int(movie_id) in self.movie_id_to_row

    def get_row(self, movie_id: int) -> pd.Series:
        """Return the pandas row for `movie_id`."""
        movie_id = int(movie_id)
        if movie_id not in self.movie_id_to_row:
            raise KeyError(f"movieId not found: {movie_id}")
        return self.df.iloc[int(self.movie_id_to_row[movie_id])]

    def get_movie_profile(self, movie_id: int) -> str:
        """Return `movie_profile` for `movie_id`."""
        row = self.get_row(movie_id)
        return str(row["movie_profile"])

    def get_display_fields(self, movie_id: int) -> dict[str, Any]:
        """Return user-facing display fields for a movie."""
        row = self.get_row(movie_id)

        year: Optional[int]
        year_val = row.get("year", None)
        if year_val is None or (isinstance(year_val, float) and pd.isna(year_val)):
            year = None
        else:
            try:
                year = int(year_val)
            except Exception:
                year = None

        imdb_val = row.get("imdbId", None)
        imdb_id = None if imdb_val is None or pd.isna(imdb_val) else str(imdb_val)

        tmdb_val = row.get("tmdbId", None)
        tmdb_id = None
        if tmdb_val is not None and not pd.isna(tmdb_val):
            try:
                tmdb_id = int(tmdb_val)
            except Exception:
                tmdb_id = None

        return {
            "movieId": int(row["movieId"]),
            "title": str(row.get("title", "")),
            "year": year,
            "genres": str(row.get("genres", "")),
            "imdbId": imdb_id,
            "tmdbId": tmdb_id,
        }

    def search_titles(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Return top title matches for UI testing."""
        query_norm = normalize_title(query)
        if not query_norm:
            return []

        candidates = list(self.norm_title_to_movie_ids.keys())
        scored: list[tuple[str, float]] = []
        for t in candidates:
            score = SequenceMatcher(None, query_norm, t).ratio()
            if score <= 0.0:
                continue
            scored.append((t, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        out: list[dict[str, Any]] = []
        for title_norm, score in scored[: max(1, int(limit))]:
            movie_ids = self.norm_title_to_movie_ids.get(title_norm, [])
            for mid in movie_ids:
                item = self.get_display_fields(int(mid))
                item["score"] = float(score)
                out.append(item)
                if len(out) >= int(limit):
                    return out
        return out

    def resolve_title_to_movie_id(
        self,
        query: str,
        *,
        min_similarity: float,
    ) -> tuple[int | None, float]:
        """Resolve a user query to a movieId via exact/fuzzy match against `title_clean`.

        Returns (movieId or None, similarity score).
        """
        query_norm = normalize_title(query)
        if not query_norm:
            return None, 0.0

        # Exact match first (handles duplicates).
        exact = self.norm_title_to_movie_ids.get(query_norm)
        if exact:
            return int(sorted(exact)[0]), 1.0

        # Fuzzy match across normalized titles.
        best_title: str | None = None
        best_score = 0.0
        for cand in self.norm_title_to_movie_ids.keys():
            s = SequenceMatcher(None, query_norm, cand).ratio()
            if s > best_score:
                best_title = cand
                best_score = float(s)

        if best_title is None or best_score < float(min_similarity):
            return None, float(best_score)

        movie_ids = self.norm_title_to_movie_ids.get(best_title, [])
        if not movie_ids:
            return None, float(best_score)

        return int(sorted(movie_ids)[0]), float(best_score)

