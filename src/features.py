from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .data import RawMovieLensData


_TITLE_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
_MULTISPACE_RE = re.compile(r"\s+")


def split_title_and_year(title: str) -> tuple[str, Optional[int]]:
    """Split a MovieLens `title` into (title_clean, year) when it ends with '(YYYY)'."""
    title = "" if title is None else str(title)
    title = title.strip()

    match = _TITLE_YEAR_RE.search(title)
    if not match:
        return title, None

    year = int(match.group(1))
    title_clean = title[: match.start()].rstrip()
    return title_clean, year


def parse_genres(genres: str) -> list[str]:
    """Parse pipe-separated genre tokens into a list."""
    if genres is None or (isinstance(genres, float) and pd.isna(genres)):
        return []
    tokens = [g.strip() for g in str(genres).split("|")]
    return [t for t in tokens if t]


def normalize_tag(tag: str) -> str:
    """Normalize a free-text tag: lowercase, strip, collapse spaces."""
    if tag is None or (isinstance(tag, float) and pd.isna(tag)):
        return ""
    tag = str(tag).lower().strip()
    tag = _MULTISPACE_RE.sub(" ", tag)
    return tag


def aggregate_tags(
    tags_df: pd.DataFrame,
    *,
    top_n_tags: int,
    stop_tags: set[str],
) -> pd.DataFrame:
    """Aggregate tags per movie into deterministic top-N lists and counts."""
    if tags_df.empty:
        return pd.DataFrame(
            columns=[
                "movieId",
                "tags_top",
                "tags_top_counts",
                "tags_top_text",
                "tag_events_count",
                "tag_unique_count",
            ]
        )

    stop_tags_norm = {normalize_tag(t) for t in stop_tags}

    tags = tags_df[["movieId", "tag"]].copy()
    tags["tag_norm"] = (
        tags["tag"]
        .astype("string")
        .fillna("")
        .str.lower()
        .str.strip()
        .str.replace(_MULTISPACE_RE, " ", regex=True)
    )
    tags = tags[(tags["tag_norm"] != "") & (~tags["tag_norm"].isin(stop_tags_norm))]

    if tags.empty:
        return pd.DataFrame(
            columns=[
                "movieId",
                "tags_top",
                "tags_top_counts",
                "tags_top_text",
                "tag_events_count",
                "tag_unique_count",
            ]
        )

    # Total events and unique tags per movie (post-filtering)
    tag_events = tags.groupby("movieId").size().rename("tag_events_count")
    tag_unique = tags.groupby("movieId")["tag_norm"].nunique().rename("tag_unique_count")

    # Count per (movieId, tag)
    tag_counts = (
        tags.groupby(["movieId", "tag_norm"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["movieId", "count", "tag_norm"], ascending=[True, False, True])
    )

    top = tag_counts.groupby("movieId", sort=False).head(top_n_tags)
    tags_top = top.groupby("movieId")["tag_norm"].apply(list).rename("tags_top")
    tags_top_counts = top.groupby("movieId")["count"].apply(list).rename("tags_top_counts")

    out = (
        pd.concat([tags_top, tags_top_counts, tag_events, tag_unique], axis=1)
        .reset_index()
        .rename(columns={"index": "movieId"})
    )
    out["tags_top_text"] = out["tags_top"].apply(lambda xs: ", ".join(xs) if isinstance(xs, list) else "")
    return out[  # stable column order
        [
            "movieId",
            "tags_top",
            "tags_top_counts",
            "tags_top_text",
            "tag_events_count",
            "tag_unique_count",
        ]
    ]


def aggregate_ratings(
    ratings_df: pd.DataFrame,
    *,
    bayes_m: int,
) -> tuple[pd.DataFrame, float]:
    """Aggregate ratings per movie and compute Bayesian-smoothed weighted ratings."""
    if ratings_df.empty:
        return pd.DataFrame(columns=["movieId", "rating_count", "rating_mean", "rating_std", "weighted_rating"]), 0.0

    global_mean = float(ratings_df["rating"].mean())

    grouped = ratings_df.groupby("movieId")["rating"]
    agg = grouped.agg(["count", "mean", "std"]).reset_index()
    agg = agg.rename(columns={"count": "rating_count", "mean": "rating_mean", "std": "rating_std"})

    v = agg["rating_count"].astype("float64")
    r = agg["rating_mean"].astype("float64")
    m = float(bayes_m)
    c = global_mean
    agg["weighted_rating"] = (v / (v + m)) * r + (m / (v + m)) * c

    return agg[["movieId", "rating_count", "rating_mean", "rating_std", "weighted_rating"]], global_mean


def build_movie_profile(
    title_clean: str,
    year: Optional[int],
    genres_list: list[str],
    tags_top: list[str],
    *,
    include_year: bool = True,
) -> str:
    """Build canonical movie profile text used for retrieval."""
    title_clean = "" if title_clean is None else str(title_clean)
    title_clean = title_clean.replace("\n", " ").strip()

    parts: list[str] = [f"Title: {title_clean}"]

    if include_year and year is not None:
        parts.append(f"Year: {int(year)}")

    genres_text = ", ".join([str(g).strip() for g in genres_list if str(g).strip()])
    if genres_text:
        parts.append(f"Genres: {genres_text}")

    if tags_top:
        tags_clean = [str(t).replace("|", " ").replace("\n", " ").strip() for t in tags_top]
        tags_clean = [t for t in tags_clean if t]
        if tags_clean:
            parts.append(f"Tags: {', '.join(tags_clean)}")

    return ". ".join(parts)


def build_movie_catalog(
    data: RawMovieLensData,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fuse structured + unstructured info into a single per-movie catalog table."""
    top_n_tags = int(config["text_profile"]["top_n_tags"])
    include_year = bool(config["text_profile"]["include_year"])
    stop_tags = set(config["text_profile"].get("stop_tags", []))
    bayes_m = int(config["ratings"]["bayes_m"])

    movies = data.movies.copy()
    title_year = movies["title"].astype(str).apply(split_title_and_year)
    movies["title_clean"] = title_year.apply(lambda x: x[0])
    movies["year"] = title_year.apply(lambda x: x[1])

    movies["genres_list"] = movies["genres"].apply(parse_genres)
    movies["genres_text"] = movies["genres_list"].apply(lambda xs: ", ".join(xs))

    ratings_agg, global_mean = aggregate_ratings(data.ratings, bayes_m=bayes_m)
    tags_agg = aggregate_tags(data.tags, top_n_tags=top_n_tags, stop_tags=stop_tags)

    # Merge (preserve movies.csv order; do not sort)
    catalog = movies.merge(data.links, on="movieId", how="left", sort=False)
    catalog = catalog.merge(ratings_agg, on="movieId", how="left", sort=False)
    catalog = catalog.merge(tags_agg, on="movieId", how="left", sort=False)

    # Fill missing aggregates for movies without ratings/tags
    catalog["rating_count"] = catalog["rating_count"].fillna(0).astype("int64")
    catalog["weighted_rating"] = catalog["weighted_rating"].fillna(global_mean).astype("float64")

    catalog["tag_events_count"] = catalog["tag_events_count"].fillna(0).astype("int64")
    catalog["tag_unique_count"] = catalog["tag_unique_count"].fillna(0).astype("int64")
    catalog["tags_top_text"] = catalog["tags_top_text"].fillna("").astype("string")

    def _ensure_list(x: Any) -> list[Any]:
        return x if isinstance(x, list) else []

    catalog["tags_top"] = catalog["tags_top"].apply(_ensure_list)
    catalog["tags_top_counts"] = catalog["tags_top_counts"].apply(_ensure_list)

    catalog["has_tags"] = catalog["tag_events_count"] > 0
    catalog["has_ratings"] = catalog["rating_count"] > 0

    catalog["movie_profile"] = catalog.apply(
        lambda row: build_movie_profile(
            title_clean=str(row["title_clean"]),
            year=(None if pd.isna(row["year"]) else int(row["year"])),
            genres_list=list(row["genres_list"]),
            tags_top=list(row["tags_top"]),
            include_year=include_year,
        ),
        axis=1,
    )

    built_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    meta = {
        "built_at_utc": built_at_utc,
        "top_n_tags": top_n_tags,
        "include_year": include_year,
        "stop_tags": sorted(stop_tags),
        "bayes_m": bayes_m,
        "global_mean_rating": global_mean,
        "rows": int(len(catalog)),
        "cols": int(len(catalog.columns)),
    }
    return catalog, meta


def save_movie_catalog(movie_catalog: pd.DataFrame, meta: dict[str, Any], processed_dir: Path) -> dict[str, str]:
    """Persist processed movie catalog artifacts."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = processed_dir / "movie_catalog.parquet"
    csv_path = processed_dir / "movie_catalog.csv"
    meta_path = processed_dir / "movie_catalog_build_meta.json"

    movie_catalog.to_parquet(parquet_path, index=False)

    csv_cols = [
        "movieId",
        "title",
        "title_clean",
        "year",
        "genres",
        "genres_text",
        "imdbId",
        "tmdbId",
        "rating_count",
        "rating_mean",
        "rating_std",
        "weighted_rating",
        "tag_events_count",
        "tag_unique_count",
        "tags_top_text",
        "movie_profile",
    ]
    existing_cols = [c for c in csv_cols if c in movie_catalog.columns]
    movie_catalog[existing_cols].to_csv(csv_path, index=False)

    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    return {
        "movie_catalog_parquet": str(parquet_path),
        "movie_catalog_csv": str(csv_path),
        "movie_catalog_meta": str(meta_path),
    }
