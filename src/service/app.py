"""FastAPI service entrypoint.

Phase 0 note:
- Only a skeleton. We'll implement endpoints in Phase 7.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="MovieLens Two-Stage Similarity Service")

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
