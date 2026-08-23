#!/usr/bin/env python
"""Generate cross-benchmark visualisations for Chapter 4.

Reads the same raw JSONs cross_benchmark.py reads, produces:
- results/figures/rank_inversion_slopegraph.png   (Figure 4.3)
- results/figures/benchmark_agreement_matrix.png  (Figure 4.4)
- results/figures/cross_benchmark_bias_heatmap.png (Figure 4.5)
- results/figures/custom_probe_error_bars.png     (Figure 4.6)
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

RESULTS_ROOT = Path("results/full")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")

SHORT = {
    "microsoft/Phi-3-mini-4k-instruct": "Phi-3-mini",
    "mistralai/Mistral-7B-v0.3": "Mistral-7B",
    "Qwen/Qwen2-7B": "Qwen2-7B",
    "meta-llama/Meta-Llama-3.1-8B": "Llama-3.1-8B",
    "google/gemma-2-9b": "Gemma-2-9b",
}

# ---------- data extraction (same as cross_benchmark.py) ----------

def crows_pairs_bias(blob):
    xs = [abs(v - 0.5) for k, m in blob.get("results", {}).items()
          if k.startswith("crows_pairs_english_")
          for v in [m.get("pct_stereotype,none") or m.get("pct_stereotype")]
          if v is not None]
    return float(np.mean(xs)) if xs else None

def bbq_bias(blob):
    r = blob.get("results", {}).get("bbq", {})
    xs = [abs(v) for k, v in r.items()
          if k.startswith("amb_bias_score_") and isinstance(v, (int, float))]
    return float(np.mean(xs)) if xs else None

def custom_probe_bias(blob):
    r = blob.get("results", {}).get("custom_probe", {})
    v = r.get("pct_stereotype,none") or r.get("pct_stereotype")
    return abs(v - 0.5) if v is not None else None

def custom_probe_stderr(blob):
    r = blob.get("results", {}).get("custom_probe", {})
    return r.get("pct_stereotype_stderr,none") or r.get("pct_stereotype_stderr")

def custom_probe_raw(blob):
    r = blob.get("results", {}).get("custom_probe", {})
    return r.get("pct_stereotype,none") or r.get("pct_stereotype")

def detect(blob):
    tasks = list(blob.get("results", {}).keys())
    if any(t.startswith("crows_pairs_english_") for t in tasks): return "crows_pairs"
    if "bbq" in tasks: return "bbq"
    if "custom_probe" in tasks: return "custom_probe"
    return None

def load_all():
    rows = []
    for path in sorted(RESULTS_ROOT.rglob("results_*.json")):
        try:
            blob = json.loads(path.read_text())
        except Exception:
            continue
        model = blob["config"]["model_args"].split(",")[0].split("=")[-1]
        bench = detect(blob)
        if bench is None: continue
        bias = {"crows_pairs": crows_pairs_bias, "bbq": bbq_bias,
                "custom_probe": custom_probe_bias}[bench](blob)
        if bias is None: continue
        row = {"model": model, "benchmark": bench, "bias": bias}
        if bench == "custom_probe":
            row["probe_raw"] = custom_probe_raw(blob)
            row["probe_stderr"] = custom_probe_stderr(blob)
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates(["model", "benchmark"], keep="last")

df = load_all()
df["model_short"] = df["model"].map(SHORT).fillna(df["model"])
bias_matrix = df.pivot_table(index="model_short", columns="benchmark", values="bias", aggfunc="first")
BENCH_ORDER = ["crows_pairs", "bbq", "custom_probe"]
bias_matrix = bias_matrix[[b for b in BENCH_ORDER if b in bias_matrix.columns]]

MODEL_ORDER = ["Phi-3-mini", "Mistral-7B", "Qwen2-7B", "Llama-3.1-8B", "Gemma-2-9b"]
model_order = [m for m in MODEL_ORDER if m in bias_matrix.index]
bias_matrix = bias_matrix.loc[model_order]

rank_matrix = bias_matrix.rank(method="average")  # 1 = least biased

BENCH_LABEL = {"crows_pairs": "CrowS-Pairs", "bbq": "BBQ", "custom_probe": "Custom probe"}

# ---------- Figure 4.3 — Rank-inversion slopegraph ----------
fig, ax = plt.subplots(figsize=(8, 5))
xs = np.arange(len(rank_matrix.columns))
colors = sns.color_palette("Set2", n_colors=len(model_order))
for i, model in enumerate(model_order):
    ys = rank_matrix.loc[model].values
    ax.plot(xs, ys, marker="o", linewidth=2.5, markersize=9,
            color=colors[i], label=model)
    for x, y in zip(xs, ys):
        ax.text(x + 0.04, y, model, fontsize=8, va="center", color=colors[i])
ax.set_xticks(xs)
ax.set_xticklabels([BENCH_LABEL[c] for c in rank_matrix.columns], fontsize=11)
ax.set_yticks(range(1, len(model_order) + 1))
ax.set_ylim(len(model_order) + 0.5, 0.5)  # invert: rank 1 at top
ax.set_ylabel("Bias rank (1 = least biased)")
ax.set_title("Rank inversion across benchmarks (five-model panel)")
ax.grid(True, axis="y", alpha=0.4)
fig.tight_layout()
fig.savefig(FIG_DIR / "rank_inversion_slopegraph.png", dpi=200)
plt.close(fig)

# ---------- Figure 4.4 — Benchmark-agreement matrix ----------
bench_list = list(bias_matrix.columns)
rho_mat = pd.DataFrame(np.eye(len(bench_list)), index=bench_list, columns=bench_list)
for a, b in combinations(bench_list, 2):
    pair = bias_matrix[[a, b]].dropna()
    if len(pair) >= 2:
        rho = spearmanr(pair[a], pair[b]).statistic
        rho_mat.loc[a, b] = rho
        rho_mat.loc[b, a] = rho

fig, ax = plt.subplots(figsize=(5.5, 4.5))
labels_pretty = [BENCH_LABEL[b] for b in bench_list]
sns.heatmap(rho_mat.values, annot=True, fmt=".3f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            xticklabels=labels_pretty, yticklabels=labels_pretty,
            cbar_kws={"label": "Spearman ρ"}, square=True)
ax.set_title("Cross-benchmark rank agreement (Spearman ρ)")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(FIG_DIR / "benchmark_agreement_matrix.png", dpi=200)
plt.close(fig)

# ---------- Figure 4.5 — Cross-benchmark bias-summary heatmap ----------
fig, ax = plt.subplots(figsize=(7, 4))
labels_x = [BENCH_LABEL[c] for c in bias_matrix.columns]
sns.heatmap(bias_matrix.values, annot=True, fmt=".3f", cmap="Reds", ax=ax,
            xticklabels=labels_x, yticklabels=list(bias_matrix.index),
            cbar_kws={"label": "Bias summary (higher = more biased)"})
ax.set_title("Per-model bias summary across benchmarks")
ax.set_xlabel("Benchmark")
ax.set_ylabel("Model")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(FIG_DIR / "cross_benchmark_bias_heatmap.png", dpi=200)
plt.close(fig)

# ---------- Figure 4.6 — Custom-probe error-bar chart ----------
probe_rows = df[df["benchmark"] == "custom_probe"].copy()
probe_rows["model_short"] = probe_rows["model"].map(SHORT).fillna(probe_rows["model"])
probe_rows = probe_rows.set_index("model_short").reindex(model_order).reset_index()

fig, ax = plt.subplots(figsize=(8, 4.5))
xs = np.arange(len(probe_rows))
ys = probe_rows["probe_raw"].values
stderrs = probe_rows["probe_stderr"].values
# 95% CI = 1.96 * stderr
cis = 1.96 * stderrs
ax.errorbar(xs, ys, yerr=cis, fmt="o", color="tab:blue", markersize=10,
            capsize=6, capthick=2, elinewidth=2, label="pct_stereotype ± 95% CI (n=20)")
ax.axhline(0.5, color="grey", linestyle="--", linewidth=1.2, label="unbiased reference (0.5)")
ax.fill_between([-0.5, len(xs) - 0.5], 0.5 - 0.05, 0.5 + 0.05,
                color="grey", alpha=0.1, label="±0.05 tolerance band")
ax.set_xticks(xs)
ax.set_xticklabels(probe_rows["model_short"], rotation=15, ha="right")
ax.set_ylabel("pct_stereotype on 20-item UK probe")
ax.set_title("Custom probe (O6): every model's CI overlaps 0.5")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0.15, 0.85)
ax.set_xlim(-0.5, len(xs) - 0.5)
fig.tight_layout()
fig.savefig(FIG_DIR / "custom_probe_error_bars.png", dpi=200)
plt.close(fig)

print("Wrote 4 figures to", FIG_DIR)
for f in ["rank_inversion_slopegraph.png", "benchmark_agreement_matrix.png",
          "cross_benchmark_bias_heatmap.png", "custom_probe_error_bars.png"]:
    print(f"  - {f}")
