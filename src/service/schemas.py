"""Pydantic schemas for the online recommendation API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """Minimal request payload for the `/recommend` endpoint."""

    query: str = Field(..., description="Free-text query (movie title/keywords) or a movieId string.")
    k: int = Field(10, ge=1, le=50, description="Number of recommendations to return (1..50).")


class RecommendationItem(BaseModel):
    """A single recommended movie item."""

    movieId: int
    title: str
    year: Optional[int] = None
    genres: str
    imdbId: Optional[str] = None
    tmdbId: Optional[int] = None

    genre_overlap_count: Optional[int] = None
    tag_overlap_count: Optional[int] = None


class RecommendResponse(BaseModel):
    """Response payload for recommendation endpoints."""

    query: str
    query_type: str
    resolved_movieId: Optional[int] = None
    mode: str
    k: int
    results: list[RecommendationItem]
    debug_info: Optional[dict] = None


class SimilarUsersRequest(BaseModel):
    """Request for similar-users endpoint using ratings-based CF."""

    userId: int = Field(..., ge=1, description="MovieLens userId from ratings.csv")
    top_n: int = Field(10, ge=1, le=100, description="Number of similar users to return")
    min_common_rated: int = Field(2, ge=0, le=1000, description="Minimum number of commonly-rated movies")


class SimilarUserItem(BaseModel):
    userId: int
    similarity: float
    common_rated: int


class SimilarUsersResponse(BaseModel):
    userId: int
    top_n: int
    results: list[SimilarUserItem]


class UserCFRecommendRequest(BaseModel):
    """Request for user-user CF recommendations."""

    userId: int = Field(..., ge=1, description="MovieLens userId from ratings.csv")
    k: int = Field(10, ge=1, le=50, description="Number of movie recommendations to return")
    top_n_sim_users: int = Field(25, ge=1, le=200, description="How many neighbors to consider")
    min_common_rated: int = Field(2, ge=0, le=1000, description="Min number of commonly-rated movies")
    min_rating_liked: float = Field(4.0, ge=0.5, le=5.0, description="Neighbor rating threshold for liked movies")


class UserCFRecommendationItem(BaseModel):
    movieId: int
    title: str | None = None
    genres: str | None = None
    score: float


class UserCFRecommendResponse(BaseModel):
    userId: int
    k: int
    results: list[UserCFRecommendationItem]
