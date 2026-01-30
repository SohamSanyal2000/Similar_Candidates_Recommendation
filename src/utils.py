from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int = 42
    deterministic: bool = True


def setup_logging(level: int | str = "INFO") -> None:
    """Configure stdlib logging with a consistent, project-wide format."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Avoid duplicate handlers if called multiple times (e.g., notebooks + CLI).
        root_logger.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_global_seed(cfg: ReproducibilityConfig) -> None:
    """Set seeds across common RNG sources for reproducibility.

    Note: full determinism in deep learning can reduce performance and isn't always
    possible across all ops/devices. For this project, we aim for "best-effort" determinism.
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # PyTorch is optional for early phases; only set if installed.
    try:
        import torch  # type: ignore

        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    # Some libraries read this env var for hash randomization determinism.
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
