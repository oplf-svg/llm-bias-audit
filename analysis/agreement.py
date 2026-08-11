#!/usr/bin/env python
"""Cross-model rank agreement on the CrowS-Pairs per-category vector.

For each pair of models, we ask: do they rank the 9 demographic categories
in the same order of stereotype-preference? Spearman rho and Kendall tau
answer this. Higher values = models agree about which categories are worst.

This is a preliminary RQ1 signal computed within one benchmark family, until
BBQ and StereoSet are added.

Writes results/agreement.csv.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def bootstrap_ci(x: np.ndarray, y: np.ndarray, stat_fn, n=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(42)
    n_items = len(x)
    if n_items < 2:
        return (np.nan, np.nan)
    stats = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_items, size=n_items)
        stats[i] = stat_fn(x[idx], y[idx]).statistic
    lo, hi = np.quantile(stats[~np.isnan(stats)], [alpha / 2, 1 - alpha / 2])
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/harmonised.parquet"))
    ap.add_argument("--output-csv", type=Path, default=Path("results/agreement.csv"))
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    # Only per-category CrowS-Pairs rows (not the overall)
    df = df[df["benchmark_family"] == "crows_pairs"]
    df = df[df["category"] != "overall"]

    wide = df.pivot_table(index="model", columns="category", values="score", aggfunc="first")
    print("Model x Category matrix:")
    print(wide.round(3).to_string())
    print()

    if wide.shape[0] < 2:
        print(f"Only {wide.shape[0]} model(s) - cannot compute cross-model agreement.")
        return

    rows = []
    rng = np.random.default_rng(42)
    for a, b in combinations(wide.index, 2):
        x = wide.loc[a].to_numpy()
        y = wide.loc[b].to_numpy()
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        sp = spearmanr(x, y)
        kt = kendalltau(x, y)
        sp_lo, sp_hi = bootstrap_ci(x, y, spearmanr, rng=rng)
        kt_lo, kt_hi = bootstrap_ci(x, y, kendalltau, rng=rng)
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "n_categories": len(x),
                "spearman_rho": sp.statistic,
                "spearman_pvalue": sp.pvalue,
                "spearman_ci95_lo": sp_lo,
                "spearman_ci95_hi": sp_hi,
                "kendall_tau": kt.statistic,
                "kendall_pvalue": kt.pvalue,
                "kendall_ci95_lo": kt_lo,
                "kendall_ci95_hi": kt_hi,
            }
        )

    result = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print("Pairwise cross-model rank agreement (on the 9-category CrowS-Pairs vector):")
    print(
        result[
            [
                "model_a",
                "model_b",
                "spearman_rho",
                "spearman_ci95_lo",
                "spearman_ci95_hi",
                "kendall_tau",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
