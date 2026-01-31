"""FastAPI service entrypoint for the two-stage similarity recommender."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from src.pipelines.offline_build import run_offline_build

from ..paths import get_repo_root
from ..utils import setup_logging
from .recommender import TwoStageMovieRecommender
from .schemas import RecommendRequest, RecommendResponse

logger = logging.getLogger(__name__)


def _get_env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    p = Path(str(raw))
    return p if p.is_absolute() else (get_repo_root() / p).resolve()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))

    repo_root = get_repo_root()
    config_path = _get_env_path("CONFIG_PATH", repo_root / "config.yaml")
    manifest_path = _get_env_path("OFFLINE_MANIFEST_PATH", repo_root / "artifacts" / "offline_manifest.json")

    # Ensure offline artifacts exist by running the offline build at startup if missing.
    if not manifest_path.exists():
        logger.info("offline_manifest.json not found at %s — running offline build", manifest_path)
        run_offline_build(
            config_path=config_path,
            force=True,
            seed=42,
            device=None,
            dense_batch_size=64,
            save_embeddings=True,
        )
    else:
        logger.info("Found offline_manifest.json at %s — skipping build", manifest_path)

    logger.info("Starting service with config=%s manifest=%s", config_path, manifest_path)
    app.state.recommender = TwoStageMovieRecommender(offline_manifest_path=manifest_path, config_path=config_path)
    yield


app = FastAPI(title="MovieLens Two-Stage Similarity Service", lifespan=lifespan)


def _recommender(app_: FastAPI) -> TwoStageMovieRecommender:
    rec = getattr(app_.state, "recommender", None)
    if rec is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    return rec


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> dict:
    """Recommend similar movies from a query string or movieId-like string."""
    if req.query is None or str(req.query).strip() == "":
        raise HTTPException(status_code=400, detail="query must be non-empty")

    rec = _recommender(app)
    try:
        return rec.recommend(query=req.query, k=req.k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/movies/search")
def movies_search(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)) -> dict:
    """Search for titles in the catalog (useful for UI testing)."""
    rec = _recommender(app)
    return {"query": q, "results": rec.search_titles(q, limit=int(limit))}
