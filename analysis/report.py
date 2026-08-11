#!/usr/bin/env python
"""Generate summary tables and figures for the dissertation.

Reads results/harmonised.parquet, produces:
- results/figures/model_x_category_heatmap.png (models rows, categories cols)
- results/figures/per_category_ranking.png     (bar chart per category, bars per model)
- results/tables/model_x_category.csv
- results/tables/model_summary.csv             (mean pct_stereotype per model)
- results/tables/category_summary.csv          (mean pct_stereotype per category)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/harmonised.parquet"))
    ap.add_argument("--fig-dir", type=Path, default=Path("results/figures"))
    ap.add_argument("--table-dir", type=Path, default=Path("results/tables"))
    args = ap.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_parquet(args.input)
    df = df[df["benchmark_family"] == "crows_pairs"]
    df = df[df["category"] != "overall"]
    if df.empty:
        print("No per-category CrowS-Pairs data.")
        return

    # Shorten model names for readability
    df["model_short"] = df["model"].str.split("/").str[-1]

    matrix = df.pivot_table(
        index="model_short", columns="category", values="score", aggfunc="first"
    )

    # Table: model x category
    matrix.round(4).to_csv(args.table_dir / "model_x_category.csv")
    # Table: model summary
    model_summary = (
        df.groupby("model_short")["score"]
        .agg(mean="mean", std="std", min="min", max="max")
        .assign(range=lambda x: x["max"] - x["min"])
        .sort_values("mean", ascending=False)
        .round(4)
    )
    model_summary.to_csv(args.table_dir / "model_summary.csv")
    # Table: category summary
    cat_summary = (
        df.groupby("category")["score"]
        .agg(mean="mean", std="std", min="min", max="max")
        .assign(range=lambda x: x["max"] - x["min"])
        .sort_values("mean", ascending=False)
        .round(4)
    )
    cat_summary.to_csv(args.table_dir / "category_summary.csv")

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(matrix))))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0.5,
        vmin=0.3,
        vmax=0.9,
        ax=ax,
        cbar_kws={"label": "pct_stereotype"},
    )
    ax.set_title("Model x demographic category (CrowS-Pairs pct_stereotype)")
    ax.set_xlabel("category")
    ax.set_ylabel("model")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "model_x_category_heatmap.png", dpi=200)
    plt.close(fig)

    # Grouped bar chart: category on x, one bar per model
    plot_df = df.reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=plot_df, x="category", y="score", hue="model_short", ax=ax)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="unbiased ref (0.5)")
    ax.set_ylim(0.4, 0.85)
    ax.set_xlabel("demographic category")
    ax.set_ylabel("pct_stereotype")
    ax.set_title("Per-category bias by model (CrowS-Pairs)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "per_category_ranking.png", dpi=200)
    plt.close(fig)

    print(f"Wrote figures to {args.fig_dir}/ and tables to {args.table_dir}/")
    print()
    print("Model summary (mean pct_stereotype across 9 categories, sorted):")
    print(model_summary.to_string())
    print()
    print("Category summary (mean pct_stereotype across models, sorted):")
    print(cat_summary.to_string())


if __name__ == "__main__":
    main()
