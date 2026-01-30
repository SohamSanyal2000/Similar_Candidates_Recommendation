from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..data import load_raw_data, validate_schema
from ..features import build_movie_catalog, save_movie_catalog
from ..paths import ProjectPaths, get_repo_root
from ..retrieval.bm25 import BM25BuildConfig, build_and_save_bm25_index
from ..retrieval.dense import DenseBuildConfig, build_and_save_dense_index
from ..utils import ReproducibilityConfig, set_global_seed, setup_logging


logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_fingerprint(raw_dir: Path) -> dict[str, Any]:
    """Compute a deterministic sha256 fingerprint over the raw CSV files."""
    raw_dir = raw_dir.resolve()
    csv_names = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]
    paths = [raw_dir / name for name in csv_names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw dataset files in {raw_dir}: {missing}")

    per_file = {p.name: _sha256_file(p) for p in paths}

    combined = hashlib.sha256()
    for name in csv_names:
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        with (raw_dir / name).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                combined.update(chunk)

    return {
        "fingerprint_sha256": combined.hexdigest(),
        "files_sha256": per_file,
    }


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


def _ensure_can_write(paths: ProjectPaths, *, force: bool) -> None:
    outputs = [
        paths.processed_dir / "movie_catalog.parquet",
        paths.processed_dir / "movie_catalog.csv",
        paths.processed_dir / "movie_catalog_build_meta.json",
        paths.dense_dir / "dense_faiss.index",
        paths.bm25_dir / "bm25_index.pkl",
        paths.artifacts_dir / "offline_manifest.json",
    ]
    existing = [p for p in outputs if p.exists()]
    if existing and not force:
        raise FileExistsError(
            "Offline artifacts already exist. Re-run with --force to overwrite.\n"
            + "\n".join([str(p) for p in existing])
        )

    if force:
        shutil.rmtree(paths.dense_dir, ignore_errors=True)
        shutil.rmtree(paths.bm25_dir, ignore_errors=True)
        (paths.artifacts_dir / "offline_manifest.json").unlink(missing_ok=True)


def run_offline_build(
    *,
    config_path: Path,
    force: bool,
    seed: int,
    device: str | None,
    dense_batch_size: int,
    save_embeddings: bool,
) -> dict[str, Any]:
    repo_root = get_repo_root()
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Expected config YAML to be a mapping, got: {type(config)}")

    paths = ProjectPaths.from_repo_root(
        repo_root,
        raw_dir=Path(config["dataset"]["raw_dir"]),
        processed_dir=Path(config["dataset"]["processed_dir"]),
    )

    _ensure_can_write(paths, force=force)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(ReproducibilityConfig(seed=seed, deterministic=True))

    logger.info("Loading raw data from %s", paths.raw_dir)
    data = load_raw_data(paths.raw_dir)
    validate_schema(data)

    logger.info("Building movie catalog")
    movie_catalog, catalog_meta = build_movie_catalog(data, config)
    processed_paths = save_movie_catalog(movie_catalog, catalog_meta, paths.processed_dir)

    # Prepare indexing input (deterministic: sort by movieId)
    catalog_sorted = movie_catalog.sort_values("movieId", kind="mergesort")
    movie_ids = catalog_sorted["movieId"].astype("int64").tolist()
    texts = catalog_sorted["movie_profile"].astype(str).tolist()

    if any((t is None) or (str(t).strip() == "") for t in texts):
        raise ValueError("Found empty movie_profile values; cannot build retrieval artifacts.")

    # Dense artifacts
    dense_cfg = DenseBuildConfig(
        model_name=str(config["retrieval"]["dense"]["bi_encoder_model"]),
        index_type=str(config["retrieval"]["dense"]["faiss_index"]),
        normalize_embeddings=bool(config["retrieval"]["dense"]["normalize_embeddings"]),
        batch_size=int(dense_batch_size),
        device=device,
        save_embeddings=bool(save_embeddings),
    )
    logger.info("Building dense FAISS artifacts in %s", paths.dense_dir)
    dense_paths = build_and_save_dense_index(movie_ids, texts, dense_cfg, paths.dense_dir)

    # BM25 artifacts
    bm25_cfg = BM25BuildConfig(tokenizer=str(config["retrieval"]["bm25"]["tokenizer"]))
    logger.info("Building BM25 artifacts in %s", paths.bm25_dir)
    bm25_index_path = build_and_save_bm25_index(movie_ids, texts, bm25_cfg, paths.bm25_dir)

    # Acceptance checks
    try:
        import faiss  # type: ignore[import-not-found]

        index = faiss.read_index(dense_paths["dense_faiss_index"])
        if int(index.ntotal) != len(movie_ids):
            raise RuntimeError(f"FAISS index ntotal={index.ntotal} expected={len(movie_ids)}")
    except Exception as exc:
        raise RuntimeError("FAISS acceptance check failed.") from exc

    if not Path(bm25_index_path).exists():
        raise RuntimeError("BM25 artifact does not exist after build.")

    # Manifest
    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    fingerprint = dataset_fingerprint(paths.raw_dir)

    config_snapshot = copy.deepcopy(config)
    config_snapshot.setdefault("build", {})
    config_snapshot["build"].update(
        {
            "seed": seed,
            "device": device,
            "dense_batch_size": dense_batch_size,
            "save_embeddings": save_embeddings,
        }
    )

    manifest = {
        "built_at_utc": now_utc,
        "dataset": {
            "raw_dir": _rel(repo_root, paths.raw_dir),
            **fingerprint,
        },
        "config": config_snapshot,
        "outputs": {
            "processed": {k: _rel(repo_root, Path(v)) for k, v in processed_paths.items()},
            "dense": {k: _rel(repo_root, Path(v)) for k, v in dense_paths.items()},
            "bm25": {"bm25_index": _rel(repo_root, Path(bm25_index_path))},
        },
    }

    manifest_path = paths.artifacts_dir / "offline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    logger.info("Offline build complete")
    logger.info("Catalog rows=%d", len(movie_catalog))
    logger.info("Sample profiles:\n%s", "\n---\n".join(texts[:3]))

    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build offline MovieLens artifacts (catalog + dense + BM25).")
    p.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config YAML.")
    p.add_argument("--force", action="store_true", help="Overwrite existing artifacts.")
    p.add_argument("--seed", type=int, default=42, help="Global random seed.")
    p.add_argument("--device", type=str, default=None, help="Embedding device (e.g. cpu, cuda, mps).")
    p.add_argument("--dense-batch-size", type=int, default=64, help="Batch size for bi-encoder embedding.")
    p.add_argument(
        "--no-save-embeddings",
        action="store_true",
        help="Do not persist dense_embeddings.npy (FAISS index + ids still saved).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    setup_logging("INFO")
    args = build_arg_parser().parse_args(argv)

    run_offline_build(
        config_path=args.config,
        force=bool(args.force),
        seed=int(args.seed),
        device=args.device,
        dense_batch_size=int(args.dense_batch_size),
        save_embeddings=(not bool(args.no_save_embeddings)),
    )


if __name__ == "__main__":
    main()

