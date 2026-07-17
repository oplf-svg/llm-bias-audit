#!/usr/bin/env python
"""Per-category divergence between benchmarks (RQ2).
Paired Wilcoxon signed-rank + Friedman across all benchmarks, with rank-biserial effect sizes.
Applies Benjamini-Hochberg correction for multiple comparisons across categories.
Writes results/divergence.csv.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


def rank_biserial(x, y):
    """Effect size for a Wilcoxon signed-rank test."""
    diff = np.array(x) - np.array(y)
    pos = (diff > 0).sum()
    neg = (diff < 0).sum()
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/harmonised.parquet"))
    ap.add_argument("--output", type=Path, default=Path("results/divergence.csv"))
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    if "category" not in df.columns or df["category"].nunique() <= 1:
        raise SystemExit(
            "No per-category data in harmonised.parquet. Ensure --log_samples was passed "
            "to lm_eval and update analysis/harmonise.py to extract categories from the "
            "per-sample JSONL."
        )

    rows = []
    for cat, sub in df.groupby("category"):
        wide = sub.pivot(index="model", columns="benchmark", values="z_score").dropna()
        if wide.shape[1] < 2 or wide.shape[0] < 3:
            continue

        # Friedman across all benchmarks (>= 3 columns needed)
        if wide.shape[1] >= 3:
            f_stat, f_p = friedmanchisquare(*[wide[b].values for b in wide.columns])
        else:
            f_stat, f_p = np.nan, np.nan

        # Pairwise Wilcoxon
        for a, b in combinations(wide.columns, 2):
            try:
                w = wilcoxon(wide[a], wide[b], zero_method="pratt", correction=True)
                effect = rank_biserial(wide[a], wide[b])
                rows.append(
                    {
                        "category": cat,
                        "benchmark_a": a,
                        "benchmark_b": b,
                        "wilcoxon_statistic": w.statistic,
                        "wilcoxon_pvalue": w.pvalue,
                        "rank_biserial_r": effect,
                        "friedman_chi2": f_stat,
                        "friedman_pvalue": f_p,
                        "n_models": wide.shape[0],
                    }
                )
            except ValueError:
                # Wilcoxon requires >= 1 non-zero difference
                pass

    result = pd.DataFrame(rows)
    if len(result) > 0:
        _, result["wilcoxon_pvalue_bh"], _, _ = multipletests(
            result["wilcoxon_pvalue"], alpha=0.05, method="fdr_bh"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(result)} rows)")


if __name__ == "__main__":
    main()
