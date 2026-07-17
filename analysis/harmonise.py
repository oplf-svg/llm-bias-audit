#!/usr/bin/env python
"""Ingest raw JSONL from lm-eval-harness and produce a single wide table:
one row per (model, benchmark, category), columns = raw score, z-standardised score.
Writes results/harmonised.parquet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BENCHMARK_TO_METRIC = {
    "bbq": "bias_score",  # BBQ's own bias score
    "crows_pairs_english": "pct_stereotype",  # pair-preference rate
    "stereoset": "ss",  # stereotype score
    "custom_probe": "acc",  # pair-preference rate (same as CrowS-Pairs)
}


def load_all(results_root: Path) -> pd.DataFrame:
    """Walk results/**/*.json produced by lm_eval and collect (model, task, category, score)."""
    rows = []
    for path in results_root.rglob("results_*.json"):
        blob = json.loads(path.read_text())
        model_id = blob["config"]["model_args"].split(",")[0].split("=")[-1]
        for task_id, metrics in blob["results"].items():
            metric_key = BENCHMARK_TO_METRIC.get(task_id.split(",")[0])
            if metric_key is None:
                continue
            score = metrics.get(f"{metric_key},none") or metrics.get(metric_key)
            rows.append(
                {
                    "model": model_id,
                    "benchmark": task_id.split(",")[0],
                    "category": "overall",  # per-category rows come from log_samples files
                    "raw_score": score,
                }
            )
    return pd.DataFrame(rows)


def z_standardise(df: pd.DataFrame) -> pd.DataFrame:
    """z-standardise raw_score within each benchmark before cross-benchmark comparison."""
    df = df.copy()
    df["z_score"] = df.groupby("benchmark")["raw_score"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("results/full"))
    ap.add_argument("--output", type=Path, default=Path("results/harmonised.parquet"))
    args = ap.parse_args()

    df = load_all(args.results_root)
    df = z_standardise(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
