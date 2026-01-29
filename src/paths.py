from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Central place to resolve important project paths.

    Keeping paths in one place reduces "stringly-typed" bugs.
    """

    root: Path

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def notebooks(self) -> Path:
        return self.root / "notebooks"


def get_repo_root() -> Path:
    """Return repo root based on this file's location."""
    return Path(__file__).resolve().parents[1]
