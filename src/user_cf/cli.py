from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .recommender import UserUserCFRecommender


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="User-user collaborative filtering (similar rating patterns)")
    p.add_argument("--user-id", type=int, required=True, help="MovieLens userId (raw id from ratings.csv)")
    p.add_argument("--top-similar", type=int, default=10, help="How many similar users to show")
    p.add_argument("--k", type=int, default=10, help="How many movie recommendations to return")
    p.add_argument("--artifacts-dir", type=Path, default=None, help="Where to read/write user_cf artifacts")
    p.add_argument("--device", type=str, default=None, help="cpu/cuda/mps; default auto")
    p.add_argument("--no-train-if-missing", action="store_true", help="Fail if artifacts are missing")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    rec = UserUserCFRecommender(
        artifacts_dir=args.artifacts_dir,
        device=args.device,
        allow_train_if_missing=(not bool(args.no_train_if_missing)),
    )

    sims = rec.similar_users(int(args.user_id), top_n=int(args.top_similar))
    recs = rec.recommend_movies(int(args.user_id), k=int(args.k))

    print("\n=== Similar Users ===")
    if sims:
        df_s = pd.DataFrame([s.__dict__ for s in sims])
        print(df_s.to_string(index=False))
    else:
        print("No similar users found (try lowering min_common_rated).")

    print("\n=== Recommended Movies ===")
    if recs:
        df_r = pd.DataFrame([r.__dict__ for r in recs])
        print(df_r.to_string(index=False))
    else:
        print("No recommendations found.")


if __name__ == "__main__":
    main()
