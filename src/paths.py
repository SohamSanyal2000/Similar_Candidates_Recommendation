from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    dense_dir: Path
    bm25_dir: Path

    @classmethod
    def from_repo_root(
        cls,
        repo_root: Path,
        *,
        raw_dir: Path | str = "data/raw",
        processed_dir: Path | str = "data/processed",
        artifacts_dir: Path | str = "artifacts",
    ) -> "ProjectPaths":
        def _resolve(p: Path | str) -> Path:
            p_path = Path(p) if isinstance(p, str) else p
            if not p_path.is_absolute():
                p_path = repo_root / p_path
            return p_path.resolve()

        raw_dir_p = _resolve(raw_dir)
        processed_dir_p = _resolve(processed_dir)
        artifacts_dir_p = _resolve(artifacts_dir)
        return cls(
            raw_dir=raw_dir_p,
            processed_dir=processed_dir_p,
            artifacts_dir=artifacts_dir_p,
            dense_dir=artifacts_dir_p / "dense",
            bm25_dir=artifacts_dir_p / "bm25",
        )


def get_repo_root() -> Path:
    """Return repo root by searching upwards for `config.yaml` or `.git`."""
    start = Path.cwd().resolve()
    if start.is_file():
        start = start.parent

    for candidate in (start, *start.parents):
        if (candidate / "config.yaml").is_file() or (candidate / ".git").exists():
            return candidate

    # Fallback: search upwards from this file (useful if called from elsewhere).
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "config.yaml").is_file() or (candidate / ".git").exists():
            return candidate

    raise FileNotFoundError("Could not locate repo root (expected `config.yaml` or `.git`).")
