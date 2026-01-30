"""Pydantic schemas for the online recommendation API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """Request payload for the `/recommend` endpoint."""

    query: str = Field(..., description="Free-text query (movie title/keywords) or a movieId string.")
    k: int = Field(10, ge=1, le=50, description="Number of recommendations to return (1..50).")
    mode: Literal["dense", "hybrid"] = Field("hybrid", description="Candidate generation mode.")
    resolve_title: bool = Field(True, description="If True, resolve query to a movieId by title matching.")
    debug: bool = Field(False, description="If True, include debug info in response.")


class RecommendationItem(BaseModel):
    """A single recommended movie item."""

    movieId: int
    title: str
    year: Optional[int] = None
    genres: str
    imdbId: Optional[str] = None
    tmdbId: Optional[int] = None
    rerank_score: float

    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    fused_score: Optional[float] = None

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

