from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from ..data import load_raw_data
from ..paths import ProjectPaths, get_repo_root
from ..utils import setup_logging
from ..user_cf.train import UserCFTrainConfig, train_user_cf


logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build user-user CF artifacts (similar rating patterns).")
    p.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config YAML.")
    p.add_argument("--device", type=str, default=None, help="cpu/cuda/mps; default auto")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory for artifacts")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--embed-dim", type=int, default=None, help="Override embedding dimension")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return p


def main(argv: list[str] | None = None) -> None:
    setup_logging("INFO")
    args = build_arg_parser().parse_args(argv)
    repo_root = get_repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()

    cfg_yaml = yaml.safe_load(config_path.read_text())
    if not isinstance(cfg_yaml, dict):
        raise ValueError("config.yaml must be a mapping")

    dataset_cfg = cfg_yaml.get("dataset", {}) if isinstance(cfg_yaml.get("dataset"), dict) else {}
    raw_dir = Path(str(dataset_cfg.get("raw_dir", "data/raw")))
    paths = ProjectPaths.from_repo_root(repo_root, raw_dir=raw_dir)
    data = load_raw_data(paths.raw_dir)

    user_cf_cfg_raw = cfg_yaml.get("user_cf", {}) if isinstance(cfg_yaml.get("user_cf"), dict) else {}
    cfg = UserCFTrainConfig(
        embed_dim=int(args.embed_dim or user_cf_cfg_raw.get("embed_dim", 32)),
        epochs=int(args.epochs or user_cf_cfg_raw.get("epochs", 5)),
        batch_size=int(args.batch_size or user_cf_cfg_raw.get("batch_size", 1024)),
        lr=float(args.lr or user_cf_cfg_raw.get("lr", 1e-2)),
        weight_decay=float(user_cf_cfg_raw.get("weight_decay", 0.0)),
        test_size=float(user_cf_cfg_raw.get("test_size", 0.1)),
        random_state=int(user_cf_cfg_raw.get("random_state", 42)),
    )

    out_dir = Path(args.out_dir) if args.out_dir is not None else (repo_root / "artifacts" / "user_cf")
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()

    logger.info("Building UserCF artifacts to %s", out_dir)
    train_user_cf(data.ratings, out_dir=out_dir, cfg=cfg, device=args.device)


if __name__ == "__main__":
    main()
