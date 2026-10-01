"""Create data/iris.csv so the project can be reproduced from scratch.

The CSV itself is tracked by DVC (not Git), so anyone cloning the repo
without access to a DVC remote can regenerate it with this script.

Usage:
    python create_dataset.py              # v1: the original 150-row Iris dataset
    python create_dataset.py --version 2  # v2: v1 + 10 extra rows (to demo dataset versioning)
"""
import argparse
import os

import pandas as pd
from sklearn.datasets import load_iris

OUTPUT_PATH = "data/iris.csv"


def build_dataset(version: int) -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame  # feature columns + "target"

    if version == 2:
        # Reproducible extra rows: a small, slightly jittered sample of the original data
        extra = df.sample(n=10, random_state=0).copy()
        feature_cols = [c for c in df.columns if c != "target"]
        extra[feature_cols] = (extra[feature_cols] * 1.02).round(2)
        df = pd.concat([df, extra], ignore_index=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the Iris CSV used by train.py")
    parser.add_argument("--version", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    df = build_dataset(args.version)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH} (version {args.version})")


if __name__ == "__main__":
    main()
