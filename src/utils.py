from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int = 42
    deterministic: bool = True


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
