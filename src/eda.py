"""Phase 1 — Data validation checks.

This module focuses on *data quality checks* for the MovieLens small dataset.
Report generation has been removed; keep any exploratory summaries in notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .data import RawMovieLensData, validate_schema


# NOTE:
# The README shipped with MovieLens lists a genre set. In practice, the dataset
# includes "IMAX" and uses "Children" (without apostrophe) in ml-latest-small.
# We treat this as allowed. If new genre tokens appear, we *warn* instead of fail.
ALLOWED_GENRES: frozenset[str] = frozenset(
    {
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "IMAX",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
        "(no genres listed)",
    }
)


@dataclass(frozen=True)
class CheckResult:
    """Single validation check outcome."""

    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    details: str


def run_phase1_checks(
    data: RawMovieLensData,
    *,
    strict: bool = True,
    allowed_genres: Sequence[str] = tuple(ALLOWED_GENRES),
) -> Tuple[CheckResult, ...]:
    """Run Phase 1 validation checks.

    Parameters
    ----------
    data:
        Loaded raw data.
    strict:
        If True, raise ValueError on FAIL checks.
    allowed_genres:
        Genre vocabulary used for validation. Unknown genres -> WARN.

    Returns
    -------
    tuple[CheckResult, ...]
        All check results.
    """

    # Phase 0 schema checks (required columns, duplicate IDs, FK integrity)
    validate_schema(data)

    movies = data.movies
    ratings = data.ratings
    tags = data.tags
    links = data.links

    checks: List[CheckResult] = []

    def _fail_or_warn(name: str, ok: bool, fail_msg: str, warn: bool = False) -> None:
        if ok:
            checks.append(CheckResult(name=name, status="PASS", details="OK"))
            return
        status = "WARN" if warn else "FAIL"
        checks.append(CheckResult(name=name, status=status, details=fail_msg))
        if strict and status == "FAIL":
            raise ValueError(f"[FAIL] {name}: {fail_msg}")

    # 1) Null / empty checks for required columns
    _fail_or_warn(
        "movies.required_non_null",
        ok=(
            movies["movieId"].notna().all()
            and movies["title"].notna().all()
            and movies["genres"].notna().all()
        ),
        fail_msg="movies.csv has nulls in required columns (movieId/title/genres)",
    )

    _fail_or_warn(
        "ratings.required_non_null",
        ok=(
            ratings["userId"].notna().all()
            and ratings["movieId"].notna().all()
            and ratings["rating"].notna().all()
            and ratings["timestamp"].notna().all()
        ),
        fail_msg="ratings.csv has nulls in required columns",
    )

    _fail_or_warn(
        "tags.required_non_null",
        ok=(
            tags["userId"].notna().all()
            and tags["movieId"].notna().all()
            and tags["tag"].notna().all()
            and tags["timestamp"].notna().all()
        ),
        fail_msg="tags.csv has nulls in required columns",
    )

    # tmdbId can be missing; imdbId should not be null
    _fail_or_warn(
        "links.required_non_null",
        ok=(links["movieId"].notna().all() and links["imdbId"].notna().all()),
        fail_msg="links.csv has nulls in required columns (movieId/imdbId)",
    )

    # 2) Rating constraints
    valid_ratings = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}
    bad_rating_values = sorted(set(ratings["rating"].unique().tolist()) - valid_ratings)
    _fail_or_warn(
        "ratings.allowed_values",
        ok=(len(bad_rating_values) == 0),
        fail_msg=f"Unexpected rating values found: {bad_rating_values}",
    )

    _fail_or_warn(
        "ratings.range",
        ok=(ratings["rating"].between(0.5, 5.0).all()),
        fail_msg="ratings.csv has rating values outside [0.5, 5.0]",
    )

    # 3) Duplicate interaction checks
    _fail_or_warn(
        "ratings.user_movie_unique",
        ok=(ratings.duplicated(subset=["userId", "movieId"]).sum() == 0),
        fail_msg="ratings.csv contains duplicate (userId, movieId) rows",
    )

    # Tags can be repeated over time, but in this dataset they are unique.
    # We'll warn (not fail) if duplicates exist.
    _fail_or_warn(
        "tags.user_movie_tag_unique",
        ok=(tags.duplicated(subset=["userId", "movieId", "tag"]).sum() == 0),
        fail_msg="tags.csv contains duplicate (userId, movieId, tag) rows",
        warn=True,
    )

    # 4) Timestamp sanity
    _fail_or_warn(
        "ratings.timestamp_non_negative",
        ok=(ratings["timestamp"] >= 0).all(),
        fail_msg="ratings.csv contains negative timestamps",
    )
    _fail_or_warn(
        "tags.timestamp_non_negative",
        ok=(tags["timestamp"] >= 0).all(),
        fail_msg="tags.csv contains negative timestamps",
    )

    # 5) User coverage consistency
    tag_users = set(tags["userId"].unique().tolist())
    rating_users = set(ratings["userId"].unique().tolist())
    missing_users = sorted(list(tag_users - rating_users))
    _fail_or_warn(
        "tags_users_subset_of_rating_users",
        ok=(len(missing_users) == 0),
        fail_msg=f"Some tag users are missing in ratings.csv: {missing_users[:10]}",
    )

    # 6) Each user has >= 20 ratings (dataset property). We'll WARN if violated.
    ratings_per_user = ratings.groupby("userId").size()
    min_ratings_per_user = int(ratings_per_user.min())
    _fail_or_warn(
        "ratings.min_20_per_user",
        ok=(min_ratings_per_user >= 20),
        fail_msg=f"Found user(s) with < 20 ratings. min={min_ratings_per_user}",
        warn=True,
    )

    # 7) Genres vocabulary check
    allowed = set(allowed_genres)
    observed: set[str] = set()
    for g in movies["genres"].astype(str):
        observed.update(g.split("|"))
    unknown_genres = sorted(list(observed - allowed))
    _fail_or_warn(
        "movies.genres_vocabulary",
        ok=(len(unknown_genres) == 0),
        fail_msg=f"Unknown genre tokens found: {unknown_genres}",
        warn=True,
    )

    # 8) Title year parse success (not required; warn if some are missing)
    year_extracted = movies["title"].astype(str).str.extract(r"\((\d{4})\)\s*$")[0]
    n_missing_year = int(year_extracted.isna().sum())
    _fail_or_warn(
        "movies.title_year_parse",
        ok=(n_missing_year == 0),
        fail_msg=f"Could not parse year from {n_missing_year} titles (expected for some TV/edge cases)",
        warn=True,
    )

    return tuple(checks)
