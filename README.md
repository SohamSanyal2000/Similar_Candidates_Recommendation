# MovieLens Two-Stage Similar Movies Recommender (Retrieval + Ranking)

This project builds a **two-stage recommendation pipeline** to identify movies similar to:
1) a **free-text user query**, or  
2) a specific **movieId**.

We implement two retrieval strategies:
- **Approach 2**: Dense semantic retrieval (Sentence-Transformers bi-encoder) → Cross-encoder reranker
- **Approach 3**: Hybrid retrieval (BM25 + Dense) → RRF fusion → Cross-encoder reranker

## Dataset

We use MovieLens `ml-latest-small` (movies, ratings, tags, links). The raw dataset files are placed under:

```
data/raw/
  movies.csv
  ratings.csv
  tags.csv
  links.csv
  README.txt
```

> **License & attribution:** The dataset is provided by GroupLens Research. Please review `data/raw/README.txt` for usage license terms and citation requirements.  

### Dataset contract (important)
The raw file formats and column meanings are defined by the MovieLens README (`data/raw/README.txt`). Key assumptions enforced by this repo:
- `ratings.csv`: 5-star scale in half-star increments; `timestamp` is UNIX seconds
- `tags.csv`: free-text `tag` values with UNIX seconds timestamps
- `movies.csv`: `movieId,title,genres` where `genres` are pipe-separated
- `links.csv`: `imdbId` must be treated as a string (it may contain leading zeros); IDs link to IMDb/TMDB

## Quickstart

### 1) Create environment & install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Build offline artifacts (catalog + indexes)
This is the single offline command that produces all local artifacts used later by the online service:

```bash
python -m src.pipelines.offline_build --config config.yaml --force
```

Outputs (must exist after a successful run):
- `data/processed/movie_catalog.parquet`
- `artifacts/dense/dense_faiss.index`
- `artifacts/bm25/bm25_index.pkl`
- `artifacts/offline_manifest.json`

### 3) Run the API (online service)
```bash
uvicorn src.service.app:app --reload --port 8000
```

Then:
- `GET /health` → `{"status":"ok"}`

Example (hybrid retrieval + cross-encoder rerank):
```bash
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" \
  -d '{"query":"Toy Story","k":10,"mode":"hybrid","resolve_title":true}'
```

Example (movieId query):
```bash
curl "http://localhost:8000/recommend/movie/1?k=10&mode=hybrid"
```

Optional (title search helper):
```bash
curl "http://localhost:8000/movies/search?q=toy%20story&limit=10"
```

## Project structure

```
mle_movielens_two_stage/
  artifacts/                 # saved indexes, embeddings, mappings (generated)
  data/
    raw/                     # raw MovieLens CSVs
    processed/               # processed features (generated)
  notebooks/                 # step-by-step notebooks per phase
  src/                       # reusable modules (retrieval, ranking, fusion, service)
  tests/                     # lightweight unit tests
  config.yaml                # hyperparameters and defaults
  requirements.txt
```

## Roadmap (phases)

- **Phase 0:** scaffold + reproducibility + schema validation
- **Phase 1:** EDA + deeper data validation
- **Phase 2:** feature engineering → build `movie_profile` text
- **Phase 3:** retrieval
  - dense bi-encoder embeddings + FAISS
  - BM25 lexical retrieval
  - hybrid fusion with RRF
- **Phase 4:** cross-encoder reranking
- **Phase 5:** offline evaluation (Recall@K, nDCG@K, etc.)
- **Phase 6:** packaging artifacts and basic model card
- **Phase 7:** serving (FastAPI) with `/recommend/text` and `/recommend/movie/{movieId}`
- **Phase 8:** monitoring + tests + regression checks

## Notes

- Tags are sparse in `ml-latest-small`. We still fuse them into the movie profile text because they are high-signal when present.
- All modeling choices and tradeoffs are documented in the notebooks.
