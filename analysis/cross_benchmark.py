#!/usr/bin/env python
"""Cross-benchmark rank agreement (the direct RQ1 answer).

For each model, we distil each benchmark down to one bias-summary number, then
rank the panel under each benchmark, then compute Spearman ρ and Kendall τ
between every pair of benchmarks. Answers: "Do BBQ, CrowS-Pairs and the custom
probe agree about which model is the most biased?"

Bias-summary metric per benchmark:
- crows_pairs   : mean |pct_stereotype − 0.5| across the 9 categories
                  (larger = more bias, in either stereotype or antistereotype
                  direction)
- bbq           : mean |amb_bias_score| across BBQ's 11 categories
                  (BBQ's own bias metric; larger = more bias)
- custom_probe  : |pct_stereotype − 0.5|
                  (same protocol as CrowS-Pairs, single number)

Writes results/cross_benchmark.csv and prints a pairwise ρ / τ table.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def crows_pairs_bias(blob):
    """Mean |pct_stereotype - 0.5| across 9 CrowS-Pairs subtasks."""
    scores = []
    for tid, m in blob.get("results", {}).items():
        if not tid.startswith("crows_pairs_english_"):
            continue
        v = m.get("pct_stereotype,none") or m.get("pct_stereotype")
        if v is not None:
            scores.append(abs(v - 0.5))
    return float(np.mean(scores)) if scores else None


def bbq_bias(blob):
    """Mean |amb_bias_score_<category>| across BBQ's per-category rows."""
    r = blob.get("results", {}).get("bbq", {})
    scores = []
    for k, v in r.items():
        if k.startswith("amb_bias_score_") and isinstance(v, (int, float)):
            scores.append(abs(v))
    return float(np.mean(scores)) if scores else None


def custom_probe_bias(blob):
    r = blob.get("results", {}).get("custom_probe", {})
    v = r.get("pct_stereotype,none") or r.get("pct_stereotype")
    return abs(v - 0.5) if v is not None else None


BENCH_EXTRACTORS = {
    "crows_pairs": crows_pairs_bias,
    "bbq": bbq_bias,
    "custom_probe": custom_probe_bias,
}


def detect_benchmark(blob):
    tasks = list(blob.get("results", {}).keys())
    if any(t.startswith("crows_pairs_english_") for t in tasks):
        return "crows_pairs"
    if "bbq" in tasks:
        return "bbq"
    if "custom_probe" in tasks:
        return "custom_probe"
    return None


def load_all(results_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_root.rglob("results_*.json")):
        try:
            blob = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        model = blob["config"]["model_args"].split(",")[0].split("=")[-1]
        bench = detect_benchmark(blob)
        if bench is None:
            continue
        score = BENCH_EXTRACTORS[bench](blob)
        if score is None:
            continue
        rows.append({"model": model, "benchmark": bench, "bias": score, "source": str(path)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # If a model has multiple results files for the same benchmark, keep the latest one
    df = df.sort_values(["model", "benchmark", "source"]).drop_duplicates(
        subset=["model", "benchmark"], keep="last"
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("results/full"))
    ap.add_argument("--output-csv", type=Path, default=Path("results/cross_benchmark.csv"))
    args = ap.parse_args()

    df = load_all(args.results_root)
    if df.empty:
        print("No results found.")
        return

    matrix = df.pivot_table(index="model", columns="benchmark", values="bias", aggfunc="first")
    print("=== Per-model bias summary (higher = more biased) ===")
    print(matrix.round(4).to_string())
    print()

    print("=== Panel ranks within each benchmark (1 = least biased) ===")
    ranks = matrix.rank(method="average")
    print(ranks.round(1).to_string())
    print()

    benchmarks = list(matrix.columns)
    if len(benchmarks) < 2:
        print(f"Only {len(benchmarks)} benchmark(s) — need at least 2 for cross-benchmark ρ.")
        return

    rows = []
    for a, b in combinations(benchmarks, 2):
        pair = matrix[[a, b]].dropna()
        if len(pair) < 2:
            continue
        sp = spearmanr(pair[a], pair[b])
        kt = kendalltau(pair[a], pair[b])
        rows.append({
            "benchmark_a": a,
            "benchmark_b": b,
            "n_models": len(pair),
            "spearman_rho": sp.statistic,
            "spearman_pvalue": sp.pvalue,
            "kendall_tau": kt.statistic,
            "kendall_pvalue": kt.pvalue,
        })

    result = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print("=== Cross-benchmark rank agreement (Spearman / Kendall) ===")
    print(result.round(3).to_string(index=False))
    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
