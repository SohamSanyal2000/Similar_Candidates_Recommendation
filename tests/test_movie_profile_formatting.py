from __future__ import annotations

from src.features import build_movie_profile


def test_build_movie_profile_is_single_line_and_no_pipes() -> None:
    profile = build_movie_profile(
        title_clean="Grumpier Old Men",
        year=1995,
        genres_list=["Comedy", "Romance"],
        tags_top=["moldy", "old"],
        include_year=True,
    )

    assert "\n" not in profile
    assert "|" not in profile
    assert "Title: Grumpier Old Men" in profile
    assert "Year: 1995" in profile
    assert "Genres: Comedy, Romance" in profile
    assert "Tags: moldy, old" in profile

