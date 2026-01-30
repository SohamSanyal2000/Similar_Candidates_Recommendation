from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import load_raw_data, validate_schema
from src.features import (
    build_movie_profile,
    normalize_tag,
    parse_genres,
    split_title_and_year,
)
from src.paths import ProjectPaths, get_repo_root


def _banner(title: str) -> None:
    line = "=" * 110
    print(f"\n{line}\n{title}\n{line}")


def _print_kv(rows: Iterable[tuple[str, Any]]) -> None:
    for k, v in rows:
        print(f"{k}: {v}")


def _print_df(
    name: str,
    df: pd.DataFrame,
    *,
    head: int,
    cols: list[str] | None = None,
    show_dtypes: bool = True,
) -> None:
    print(f"{name}: shape={df.shape}")
    if cols is None:
        cols = list(df.columns)
    cols = [c for c in cols if c in df.columns]
    if show_dtypes:
        dtypes = df[cols].dtypes.astype(str).to_dict()
        print(f"{name}: dtypes={dtypes}")
    print(df[cols].head(head).to_string(index=False))


def _chunk(xs: list[str], *, batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    return [xs[i : i + batch_size] for i in range(0, len(xs), batch_size)]


def _trace_aggregate_ratings(
    ratings_df: pd.DataFrame,
    *,
    bayes_m: int,
    sample_rows: int,
) -> tuple[pd.DataFrame, float]:
    if ratings_df.empty:
        _print_kv([("ratings_empty", True)])
        return (
            pd.DataFrame(columns=["movieId", "rating_count", "rating_mean", "rating_std", "weighted_rating"]),
            0.0,
        )

    _banner("2.4.1) ratings.csv subset used for aggregation (movieId, rating)")
    base = ratings_df[["movieId", "rating"]].copy()
    _print_df("ratings_base", base, head=sample_rows, cols=["movieId", "rating"], show_dtypes=False)

    _banner("2.4.2) Global mean rating")
    global_mean = float(ratings_df["rating"].mean())
    _print_kv([("global_mean_rating", global_mean)])

    _banner("2.4.3) Groupby(movieId)['rating'].agg(count, mean, std)")
    grouped = ratings_df.groupby("movieId")["rating"]
    agg_raw = grouped.agg(["count", "mean", "std"]).reset_index()
    _print_df("ratings_agg_raw", agg_raw, head=sample_rows, cols=["movieId", "count", "mean", "std"], show_dtypes=False)

    _banner("2.4.4) Rename columns to rating_count/rating_mean/rating_std")
    agg = agg_raw.rename(columns={"count": "rating_count", "mean": "rating_mean", "std": "rating_std"})
    _print_df(
        "ratings_agg",
        agg,
        head=sample_rows,
        cols=["movieId", "rating_count", "rating_mean", "rating_std"],
        show_dtypes=False,
    )

    _banner("2.4.5) Bayesian smoothing -> weighted_rating")
    v = agg["rating_count"].astype("float64")
    r = agg["rating_mean"].astype("float64")
    m = float(bayes_m)
    c = global_mean
    agg["weighted_rating"] = (v / (v + m)) * r + (m / (v + m)) * c
    _print_df(
        "ratings_agg",
        agg,
        head=sample_rows,
        cols=["movieId", "rating_count", "rating_mean", "weighted_rating"],
        show_dtypes=False,
    )

    return agg[["movieId", "rating_count", "rating_mean", "rating_std", "weighted_rating"]], global_mean


def _trace_aggregate_tags(
    tags_df: pd.DataFrame,
    *,
    top_n_tags: int,
    stop_tags: set[str],
    sample_rows: int,
) -> pd.DataFrame:
    if tags_df.empty:
        _print_kv([("tags_empty", True)])
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

    _banner("2.5.1) stop_tags normalization (normalize_tag)")
    stop_tags_norm = {normalize_tag(t) for t in stop_tags}
    _print_kv([("stop_tags_norm", sorted(stop_tags_norm))])

    _banner("2.5.2) tags.csv subset used for aggregation (movieId, tag)")
    tags = tags_df[["movieId", "tag"]].copy()
    _print_df("tags_base", tags, head=sample_rows, cols=["movieId", "tag"], show_dtypes=False)

    _banner("2.5.3) Normalize tags -> tag_norm (lower, strip, collapse spaces)")
    tags["tag_norm"] = (
        tags["tag"]
        .astype("string")
        .fillna("")
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    _print_df("tags_norm", tags, head=sample_rows, cols=["movieId", "tag", "tag_norm"], show_dtypes=False)

    _banner("2.5.4) Filter empty tags + stop_tags")
    tags = tags[(tags["tag_norm"] != "") & (~tags["tag_norm"].isin(stop_tags_norm))]
    _print_kv([("tags_after_filter_rows", len(tags))])
    _print_df("tags_filtered", tags, head=sample_rows, cols=["movieId", "tag_norm"], show_dtypes=False)

    if tags.empty:
        _print_kv([("tags_after_filter_empty", True)])
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

    _banner("2.5.5) tag_events_count per movie (post-filtering)")
    tag_events = tags.groupby("movieId").size().rename("tag_events_count")
    print(tag_events.head(sample_rows).to_string())

    _banner("2.5.6) tag_unique_count per movie (post-filtering)")
    tag_unique = tags.groupby("movieId")["tag_norm"].nunique().rename("tag_unique_count")
    print(tag_unique.head(sample_rows).to_string())

    _banner("2.5.7) Count per (movieId, tag_norm) + deterministic sort")
    tag_counts = (
        tags.groupby(["movieId", "tag_norm"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["movieId", "count", "tag_norm"], ascending=[True, False, True])
    )
    _print_df("tag_counts", tag_counts, head=sample_rows, cols=["movieId", "tag_norm", "count"], show_dtypes=False)

    _banner("2.5.8) Top-N tags per movie (groupby(movieId).head(top_n_tags))")
    top = tag_counts.groupby("movieId", sort=False).head(int(top_n_tags))
    _print_df("top_tags", top, head=sample_rows, cols=["movieId", "tag_norm", "count"], show_dtypes=False)

    _banner("2.5.9) tags_top + tags_top_counts lists")
    tags_top = top.groupby("movieId")["tag_norm"].apply(list).rename("tags_top")
    tags_top_counts = top.groupby("movieId")["count"].apply(list).rename("tags_top_counts")
    print(pd.concat([tags_top, tags_top_counts], axis=1).head(sample_rows).to_string())

    _banner("2.5.10) Final tags_agg table")
    out = (
        pd.concat([tags_top, tags_top_counts, tag_events, tag_unique], axis=1)
        .reset_index()
        .rename(columns={"index": "movieId"})
    )
    out["tags_top_text"] = out["tags_top"].apply(lambda xs: " | ".join(xs) if isinstance(xs, list) else "")
    out = out[
        [
            "movieId",
            "tags_top",
            "tags_top_counts",
            "tags_top_text",
            "tag_events_count",
            "tag_unique_count",
        ]
    ]
    _print_df(
        "tags_agg",
        out,
        head=sample_rows,
        cols=[
            "movieId",
            "tags_top",
            "tags_top_counts",
            "tags_top_text",
            "tag_events_count",
            "tag_unique_count",
        ],
        show_dtypes=False,
    )
    return out


def trace_offline_dense_input(
    *,
    config_path: Path,
    sample_rows: int,
    dense_batch_size: int,
    device: str | None,
) -> None:
    pd.set_option("display.max_colwidth", 200)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    _banner("0) Load config")
    repo_root = get_repo_root()
    # If invoked from a subdir (e.g. `tests/`), treat relative config paths as repo-root relative.
    if not config_path.is_absolute() and not config_path.exists():
        config_path = repo_root / config_path
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Expected config YAML to be a mapping, got {type(config)}")

    paths = ProjectPaths.from_repo_root(
        repo_root,
        raw_dir=Path(config["dataset"]["raw_dir"]),
        processed_dir=Path(config["dataset"]["processed_dir"]),
    )
    _print_kv(
        [
            ("repo_root", repo_root),
            ("config_path", config_path),
            ("raw_dir", paths.raw_dir),
            ("processed_dir", paths.processed_dir),
        ]
    )

    _banner("1) Load raw MovieLens data")
    data = load_raw_data(paths.raw_dir)
    validate_schema(data)
    _print_df("movies", data.movies, head=sample_rows, cols=["movieId", "title", "genres"])
    _print_df("ratings", data.ratings, head=sample_rows, cols=["userId", "movieId", "rating", "timestamp"])
    _print_df("tags", data.tags, head=sample_rows, cols=["userId", "movieId", "tag", "timestamp"])
    _print_df("links", data.links, head=sample_rows, cols=["movieId", "imdbId", "tmdbId"])

    _banner("2) Build movie catalog (step-by-step)")
    top_n_tags = int(config["text_profile"]["top_n_tags"])
    include_year = bool(config["text_profile"]["include_year"])
    stop_tags = set(config["text_profile"].get("stop_tags", []))
    bayes_m = int(config["ratings"]["bayes_m"])
    _print_kv(
        [
            ("top_n_tags", top_n_tags),
            ("include_year", include_year),
            ("stop_tags", sorted(stop_tags)),
            ("bayes_m", bayes_m),
        ]
    )

    _banner("2.1) movies.csv -> copy")
    movies = data.movies.copy()
    _print_df("movies", movies, head=sample_rows, cols=["movieId", "title", "genres"])

    _banner("2.2) Title parsing: split_title_and_year(title) -> (title_clean, year)")
    title_year = movies["title"].astype(str).apply(split_title_and_year)
    movies["title_clean"] = title_year.apply(lambda x: x[0])
    movies["year"] = title_year.apply(lambda x: x[1])
    _print_df(
        "movies",
        movies,
        head=sample_rows,
        cols=["movieId", "title", "title_clean", "year"],
        show_dtypes=False,
    )

    _banner("2.3) Genre parsing: parse_genres(genres) -> genres_list + genres_text")
    movies["genres_list"] = movies["genres"].apply(parse_genres)
    movies["genres_text"] = movies["genres_list"].apply(lambda xs: ", ".join(xs))
    _print_df(
        "movies",
        movies,
        head=sample_rows,
        cols=["movieId", "genres", "genres_list", "genres_text"],
        show_dtypes=False,
    )

    _banner("2.4) Ratings aggregation: aggregate_ratings(ratings, bayes_m)")
    ratings_agg, global_mean = _trace_aggregate_ratings(data.ratings, bayes_m=bayes_m, sample_rows=sample_rows)
    _print_kv([("ratings_agg_rows", len(ratings_agg))])

    _banner("2.5) Tags aggregation: aggregate_tags(tags, top_n_tags, stop_tags)")
    tags_agg = _trace_aggregate_tags(
        data.tags,
        top_n_tags=top_n_tags,
        stop_tags=stop_tags,
        sample_rows=sample_rows,
    )
    _print_kv([("tags_agg_rows", len(tags_agg))])

    _banner("2.6) Merge: movies + links (left join on movieId; preserve movies.csv order)")
    catalog = movies.merge(data.links, on="movieId", how="left", sort=False)
    _print_df(
        "catalog",
        catalog,
        head=sample_rows,
        cols=["movieId", "title_clean", "year", "imdbId", "tmdbId"],
        show_dtypes=False,
    )

    _banner("2.7) Merge: + ratings_agg (left join on movieId)")
    catalog = catalog.merge(ratings_agg, on="movieId", how="left", sort=False)
    _print_df(
        "catalog",
        catalog,
        head=sample_rows,
        cols=["movieId", "rating_count", "rating_mean", "rating_std", "weighted_rating"],
        show_dtypes=False,
    )

    _banner("2.8) Merge: + tags_agg (left join on movieId)")
    catalog = catalog.merge(tags_agg, on="movieId", how="left", sort=False)
    _print_df(
        "catalog",
        catalog,
        head=sample_rows,
        cols=["movieId", "tags_top_text", "tag_events_count", "tag_unique_count"],
        show_dtypes=False,
    )

    _banner("2.9) Fill missing aggregates + ensure list columns")
    _print_kv(
        [
            ("missing_rating_count_before", int(catalog["rating_count"].isna().sum())),
            ("missing_weighted_rating_before", int(catalog["weighted_rating"].isna().sum())),
            ("missing_tag_events_before", int(catalog["tag_events_count"].isna().sum())),
            ("missing_tags_top_text_before", int(catalog["tags_top_text"].isna().sum())),
        ]
    )

    catalog["rating_count"] = catalog["rating_count"].fillna(0).astype("int64")
    catalog["weighted_rating"] = catalog["weighted_rating"].fillna(global_mean).astype("float64")

    catalog["tag_events_count"] = catalog["tag_events_count"].fillna(0).astype("int64")
    catalog["tag_unique_count"] = catalog["tag_unique_count"].fillna(0).astype("int64")
    catalog["tags_top_text"] = catalog["tags_top_text"].fillna("").astype("string")

    def _ensure_list(x: Any) -> list[Any]:
        return x if isinstance(x, list) else []

    catalog["tags_top"] = catalog["tags_top"].apply(_ensure_list)
    catalog["tags_top_counts"] = catalog["tags_top_counts"].apply(_ensure_list)

    _print_kv(
        [
            ("missing_rating_count_after", int(catalog["rating_count"].isna().sum())),
            ("missing_weighted_rating_after", int(catalog["weighted_rating"].isna().sum())),
            ("missing_tag_events_after", int(catalog["tag_events_count"].isna().sum())),
            ("missing_tags_top_text_after", int(catalog["tags_top_text"].isna().sum())),
        ]
    )

    _banner("2.10) Derived flags: has_tags, has_ratings")
    catalog["has_tags"] = catalog["tag_events_count"] > 0
    catalog["has_ratings"] = catalog["rating_count"] > 0
    _print_kv(
        [
            ("has_tags_true", int(catalog["has_tags"].sum())),
            ("has_tags_false", int((~catalog["has_tags"]).sum())),
            ("has_ratings_true", int(catalog["has_ratings"].sum())),
            ("has_ratings_false", int((~catalog["has_ratings"]).sum())),
        ]
    )

    _banner("2.11) Text feature: build_movie_profile(...) -> movie_profile")
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
    sample_profiles = catalog[["movieId", "movie_profile"]].head(sample_rows)
    for movie_id, profile in sample_profiles.itertuples(index=False):
        print(f"\nmovieId={int(movie_id)}\n{profile}\n{'-' * 60}")

    _banner("3) Prepare input to SentenceTransformer.encode (offline dense embedding build)")
    catalog_sorted = catalog.sort_values("movieId", kind="mergesort")
    movie_ids = catalog_sorted["movieId"].astype("int64").tolist()
    texts = catalog_sorted["movie_profile"].astype(str).tolist()

    _print_kv(
        [
            ("catalog_rows", len(catalog)),
            ("sorted_rows", len(catalog_sorted)),
            ("movie_ids_len", len(movie_ids)),
            ("texts_len", len(texts)),
            ("movie_id_min", min(movie_ids) if movie_ids else None),
            ("movie_id_max", max(movie_ids) if movie_ids else None),
            ("any_empty_text", any((t is None) or (str(t).strip() == "") for t in texts)),
        ]
    )

    dense_cfg = {
        "model_name": str(config["retrieval"]["dense"]["bi_encoder_model"]),
        "index_type": str(config["retrieval"]["dense"]["faiss_index"]),
        "normalize_embeddings": bool(config["retrieval"]["dense"]["normalize_embeddings"]),
        "batch_size": int(dense_batch_size),
        "device": device,
    }
    _print_kv([("dense_build_config", dense_cfg)])

    _banner("3.1) Example (movieId, text) pairs passed to model.encode")
    for mid, text in list(zip(movie_ids, texts))[:sample_rows]:
        print(f"\nmovieId={int(mid)}\n{text}\n{'-' * 60}")

    _banner("3.2) How texts are chunked before passing to model.encode (batch_size)")
    batches = _chunk(texts, batch_size=int(dense_batch_size))
    _print_kv([("num_batches", len(batches)), ("batch_size", int(dense_batch_size))])
    for i, batch in enumerate(batches[: max(1, sample_rows)]):
        preview = batch[0].replace("\n", "\\n") if batch else ""
        print(f"batch[{i}]: size={len(batch)} first_text_preview={preview[:120]}")

    _banner("3.3) The exact call (this script stops BEFORE running it)")
    print(
        "SentenceTransformer(model_name, device=device).encode(\n"
        "  texts,\n"
        "  batch_size=batch_size,\n"
        "  convert_to_numpy=True,\n"
        "  show_progress_bar=False,\n"
        ")"
    )
    print("\nDone. (No embeddings were computed.)")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Trace the offline dense embedding build data flow, printing each transformation step, "
            "up to (but not including) the SentenceTransformer.encode() call."
        )
    )
    p.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config YAML.")
    p.add_argument("--sample-rows", type=int, default=5, help="How many sample rows to print per step.")
    p.add_argument("--dense-batch-size", type=int, default=64, help="Batch size that would be used for encode().")
    p.add_argument("--device", type=str, default=None, help="Device that would be used for SentenceTransformer.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    trace_offline_dense_input(
        config_path=args.config,
        sample_rows=int(args.sample_rows),
        dense_batch_size=int(args.dense_batch_size),
        device=args.device,
    )


if __name__ == "__main__":
    main()
