#!/usr/bin/env python
"""Cross-benchmark rank agreement (RQ1).
Spearman rho and Kendall tau over the model panel, with bootstrapped 95% CIs.
Writes results/agreement.csv and results/agreement.png.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def bootstrap_ci(x: np.ndarray, y: np.ndarray, stat_fn, n=10000, alpha=0.05, rng=None):
    """Non-parametric bootstrap CI for a paired statistic."""
    rng = rng or np.random.default_rng(42)
    n_items = len(x)
    stats = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_items, size=n_items)
        stats[i] = stat_fn(x[idx], y[idx]).statistic
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/harmonised.parquet"))
    ap.add_argument("--output-csv", type=Path, default=Path("results/agreement.csv"))
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    wide = df.pivot(index="model", columns="benchmark", values="z_score")
    benchmarks = sorted(wide.columns)

    rows = []
    rng = np.random.default_rng(42)
    for a, b in combinations(benchmarks, 2):
        x = wide[a].to_numpy()
        y = wide[b].to_numpy()
        sp = spearmanr(x, y)
        kt = kendalltau(x, y)
        sp_lo, sp_hi = bootstrap_ci(x, y, spearmanr, rng=rng)
        kt_lo, kt_hi = bootstrap_ci(x, y, kendalltau, rng=rng)
        rows.append(
            {
                "benchmark_a": a,
                "benchmark_b": b,
                "spearman_rho": sp.statistic,
                "spearman_ci95_lo": sp_lo,
                "spearman_ci95_hi": sp_hi,
                "spearman_pvalue": sp.pvalue,
                "kendall_tau": kt.statistic,
                "kendall_ci95_lo": kt_lo,
                "kendall_ci95_hi": kt_hi,
                "kendall_pvalue": kt.pvalue,
                "n_models": len(x),
            }
        )

    result = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(result.to_string(index=False))
    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
