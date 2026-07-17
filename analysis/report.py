#!/usr/bin/env python
"""Generate summary tables and plots for the dissertation.
Reads results/harmonised.parquet + results/agreement.csv + results/divergence.csv
and writes results/figures/*.png and results/tables/*.tex.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_agreement(agree_df: pd.DataFrame, out_dir: Path):
    """Bar chart of Spearman rho per (benchmark_a, benchmark_b) with CIs."""
    fig, ax = plt.subplots(figsize=(7, 4))
    xlabels = [f"{a[:6]} vs {b[:6]}" for a, b in zip(agree_df.benchmark_a, agree_df.benchmark_b)]
    positions = range(len(xlabels))
    ax.bar(
        positions,
        agree_df.spearman_rho,
        yerr=[
            agree_df.spearman_rho - agree_df.spearman_ci95_lo,
            agree_df.spearman_ci95_hi - agree_df.spearman_rho,
        ],
        color="#1F3864",
        alpha=0.8,
        capsize=4,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, rotation=15, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title("Cross-benchmark rank agreement (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out_dir / "agreement.png", dpi=200)
    plt.close(fig)


def plot_model_by_benchmark(harmonised: pd.DataFrame, out_dir: Path):
    """Heatmap of z-standardised bias scores by (model, benchmark)."""
    wide = harmonised.pivot(index="model", columns="benchmark", values="z_score")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        wide,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "z-standardised bias score"},
    )
    ax.set_title("Model x benchmark bias scores (z-standardised)")
    fig.tight_layout()
    fig.savefig(out_dir / "model_x_benchmark.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmonised", type=Path, default=Path("results/harmonised.parquet"))
    ap.add_argument("--agreement", type=Path, default=Path("results/agreement.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    harmonised = pd.read_parquet(args.harmonised)
    agree = pd.read_csv(args.agreement)

    plot_agreement(agree, args.out_dir)
    plot_model_by_benchmark(harmonised, args.out_dir)

    print(f"Figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
