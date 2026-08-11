#!/usr/bin/env python
"""Ingest raw JSONL from lm-eval-harness and produce a single wide table:
one row per (model, task, metric, score). Handles the per-category
crows_pairs_english_* subtasks emitted by our full-panel run.

Writes results/harmonised.parquet and results/harmonised.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# For each task family, the primary metric of interest.
# Task-name prefix -> metric key
TASK_PREFIX_TO_METRIC = {
    "crows_pairs_english": "pct_stereotype,none",
    "bbq": "acc,none",
    "stereoset": "ss,none",
    "custom_probe": "acc,none",
}


def infer_metric(task_id: str) -> str | None:
    for prefix, metric in TASK_PREFIX_TO_METRIC.items():
        if task_id.startswith(prefix):
            return metric
    return None


def load_all(results_root: Path) -> pd.DataFrame:
    """Walk results/**/results_*.json and collect (model, task, category, score)."""
    rows = []
    for path in sorted(results_root.rglob("results_*.json")):
        try:
            blob = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"WARN: could not parse {path}")
            continue

        model_id = blob["config"]["model_args"].split(",")[0].split("=")[-1]

        for task_id, metrics in blob["results"].items():
            metric_key = infer_metric(task_id)
            if metric_key is None:
                continue
            # Value can also be at "metric,none" or "metric" depending on version
            score = metrics.get(metric_key) or metrics.get(metric_key.split(",")[0])
            stderr = metrics.get(metric_key.replace(",none", "_stderr,none"))

            # Derive a category label when the task is a CrowS-Pairs subtask
            category = task_id
            if task_id.startswith("crows_pairs_english_"):
                category = task_id.replace("crows_pairs_english_", "")
            elif task_id == "crows_pairs_english":
                category = "overall"

            rows.append(
                {
                    "model": model_id,
                    "task": task_id,
                    "benchmark_family": task_id.split("_")[0]
                    if not task_id.startswith("crows_pairs")
                    else "crows_pairs",
                    "category": category,
                    "metric": metric_key.split(",")[0],
                    "score": score,
                    "stderr": stderr,
                    "source_file": str(path),
                }
            )
    return pd.DataFrame(rows)


def z_standardise(df: pd.DataFrame) -> pd.DataFrame:
    """z-standardise within each benchmark_family so scores are comparable across families."""
    df = df.copy()
    df["z_score"] = df.groupby("benchmark_family")["score"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("results/full"))
    ap.add_argument("--output-parquet", type=Path, default=Path("results/harmonised.parquet"))
    ap.add_argument("--output-csv", type=Path, default=Path("results/harmonised.csv"))
    args = ap.parse_args()

    df = load_all(args.results_root)
    if df.empty:
        print("No results found. Did lm-eval write any results_*.json?")
        return

    df = z_standardise(df)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_parquet, index=False)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.output_parquet} and {args.output_csv}")
    print()
    print("=== Preview (model x category matrix) ===")
    matrix = df.pivot_table(index="model", columns="category", values="score", aggfunc="first")
    print(matrix.round(3).to_string())


if __name__ == "__main__":
    main()
