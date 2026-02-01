# MovieLens Two‑Stage Similar Movies Recommender (Retrieval + Reranking)

Build a **two-stage recommendation pipeline** that returns movies similar to:
1) a **free-text query** (e.g. a title or keywords), or
2) a specific **MovieLens `movieId`**.

The system fuses **structured metadata** (MovieLens `movies.csv`) and **unstructured user tags** (`tags.csv`) into a single `movie_profile` text used by retrieval and reranking.

---

## What this project does (high level)

**Offline (build artifacts)**
1. Load + validate MovieLens CSVs.
2. Create a `movie_profile` per movie (title + genres + tags + other engineered fields).
3. Build retrieval artifacts:
   - **Dense** semantic index: Sentence-Transformers bi-encoder embeddings + **FAISS**
   - **BM25** lexical index
4. Write `artifacts/offline_manifest.json` pointing to all generated outputs.

**Online (FastAPI service)**
1. Given a query, generate candidates using the default retrieval mode from `config.yaml`.
2. Fuse candidates (if hybrid) and **rerank using a cross-encoder**.
3. Return top-`k` recommendations.

> Important behavior: if your query resolves to a specific movie (by `movieId` or title match), that movie is **excluded** from the recommendations list.

---

## Prerequisites

- Python 3.10+ (recommended)
- macOS/Linux/Windows (tested locally on macOS)
- Enough RAM for building embeddings and loading models (first run downloads HF models)

---

## Step-by-step: run locally

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download the dataset (MovieLens `ml-latest-small`)

1. Download `ml-latest-small.zip` from GroupLens:
   - https://grouplens.org/datasets/movielens/
2. Unzip it.
3. Copy the CSVs into `data/raw/` so you have:

```
data/raw/
  movies.csv
  ratings.csv
  tags.csv
  links.csv
  README.txt
```

#### Dataset contract (important)
The raw file formats and column meanings are defined by the MovieLens README (`data/raw/README.txt`). Key assumptions enforced by this repo:
- `ratings.csv`: 5-star scale in half-star increments; `timestamp` is UNIX seconds
- `tags.csv`: free-text `tag` values with UNIX seconds timestamps
- `movies.csv`: `movieId,title,genres` where `genres` are pipe-separated
- `links.csv`: `imdbId` must be treated as a string (it may contain leading zeros); IDs link to IMDb/TMDB

> **License & attribution:** The dataset is provided by GroupLens Research. Please review `data/raw/README.txt` for usage license terms and citation requirements.

### 3) Start the API service
```bash
uvicorn src.service.app:app --reload --port 8000
```

On startup, the service checks for `artifacts/offline_manifest.json`:
- If it **exists**, startup is faster (it loads artifacts/models).
- If it **does not exist**, the service **automatically runs the offline build once** to generate all artifacts, then starts.

> The first startup can take a while because it may (1) build FAISS/BM25 artifacts and (2) download transformer models.

---

## Using the API

The core API is a single endpoint:

### `POST /recommend`

**Request JSON**
- `query` (string, required): free text (title/keywords) *or* a MovieLens `movieId` as a string
- `k` (int, optional, default=10): number of recommendations to return (1..50)

#### Example: free-text/title query
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query":"Toy Story","k":10}'
```

#### Example: `movieId` query
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query":"1","k":10}'
```

### `GET /movies/search` (helper)
Optional helper endpoint useful for quickly finding candidate titles/movieIds.
```bash
curl "http://localhost:8000/movies/search?q=toy%20story&limit=10"
```

---

# Similar rating patterns (User-User Collaborative Filtering)

This repo also includes a **ratings-based** recommender that:

- finds **users with similar rating patterns** using `data/raw/ratings.csv`
- recommends movies liked by similar users to a target user

### Build CF artifacts (optional)

Artifacts are stored in `artifacts/user_cf/`. The API will **train them automatically** at startup if missing, but you can prebuild:

```bash
python -m src.pipelines.user_cf_build --config config.yaml --device cpu
```

### API endpoints

#### `POST /user_cf/similar_users`
```bash
curl -X POST http://localhost:8000/user_cf/similar_users \
  -H "Content-Type: application/json" \
  -d '{"userId":1,"top_n":10,"min_common_rated":2}'
```

#### `POST /user_cf/recommend`
```bash
curl -X POST http://localhost:8000/user_cf/recommend \
  -H "Content-Type: application/json" \
  -d '{"userId":1,"k":10,"top_n_sim_users":25,"min_common_rated":2,"min_rating_liked":4.0}'
```

### CLI usage

```bash
python -m src.user_cf.cli --user-id 1 --top-similar 10 --k 10
```

---

## (Optional) Run the offline build manually

If you want to pre-build artifacts (recommended for faster API startup), run:
```bash
python -m src.pipelines.offline_build --config config.yaml --force
```

Expected outputs after a successful build:
- `data/processed/movie_catalog.parquet`
- `artifacts/dense/dense_faiss.index`
- `artifacts/bm25/bm25_index.pkl`
- `artifacts/offline_manifest.json`

---

## Running tests

```bash
./run_tests.sh
```

Or directly:
```bash
pytest -q
```

---

## Configuration

Main configuration lives in `config.yaml`.

Environment variables supported by the API service:
- `CONFIG_PATH` (default: `config.yaml`)
- `OFFLINE_MANIFEST_PATH` (default: `artifacts/offline_manifest.json`)
- `LOG_LEVEL` (default: `INFO`)

---

## Troubleshooting

### Startup fails with “Missing raw dataset files …”
Ensure the four CSVs exist under `data/raw/`:
`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`.

### Startup is slow the first time
This is expected: it may build artifacts and download transformer models.
After the first successful run, keep `artifacts/offline_manifest.json` to avoid rebuilding.

### Port already in use
Run on a different port:
```bash
uvicorn src.service.app:app --reload --port 8001
```

---

## Project structure

```
artifacts/                   # generated indexes + offline manifest
data/
  raw/                       # MovieLens CSVs (you provide)
  processed/                 # generated catalog/features
notebooks/                   # EDA + explanations
src/
  pipelines/                 # offline build pipeline
  retrieval/                 # dense (FAISS) + BM25 retrievers
  fusion/                    # RRF fusion
  ranking/                   # cross-encoder reranking
  service/                   # FastAPI app + recommender + schemas
tests/                       # unit / smoke tests
config.yaml
requirements.txt
```
