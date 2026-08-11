#!/usr/bin/env python
"""Between-model spread per demographic category.

For each of the 9 CrowS-Pairs categories, report the spread of pct_stereotype
across the model panel. Where the spread is large, models disagree strongly
about how biased they are on that category, which is preliminary evidence
that "bias" depends on the model choice.

Writes results/divergence.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/harmonised.parquet"))
    ap.add_argument("--output", type=Path, default=Path("results/divergence.csv"))
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    df = df[df["benchmark_family"] == "crows_pairs"]
    df = df[df["category"] != "overall"]

    if df.empty:
        print("No per-category CrowS-Pairs data - cannot compute divergence.")
        return

    stats = (
        df.groupby("category")["score"]
        .agg(mean="mean", std="std", min="min", max="max", n_models="count")
        .assign(range=lambda x: x["max"] - x["min"])
        .sort_values("range", ascending=False)
        .round(4)
    )

    print("Between-model spread per category (sorted by range, largest first):")
    print(stats.to_string())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.output)
    print(f"\nWrote {args.output}")

    # Also print an aggregate summary
    print()
    print("Aggregate summary:")
    print(f"  Model panel size:               {df['model'].nunique()}")
    print(f"  Categories:                     {df['category'].nunique()}")
    print(f"  Overall mean pct_stereotype:    {df['score'].mean():.4f}")
    print(f"  Overall std pct_stereotype:     {df['score'].std():.4f}")
    print(
        f"  Widest between-model spread:    {stats['range'].max():.4f} on '{stats['range'].idxmax()}'"
    )
    print(
        f"  Narrowest between-model spread: {stats['range'].min():.4f} on '{stats['range'].idxmin()}'"
    )


if __name__ == "__main__":
    main()
