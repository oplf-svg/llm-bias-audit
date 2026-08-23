#!/usr/bin/env bash
# One-shot analysis pipeline for your Mac.
# Pulls latest results from GitHub and runs harmonise -> agreement -> divergence -> report.
# Assumes you have already run scripts/install-mac.sh at some point.

set -euo pipefail

echo "==> 1/8  Pull latest results from GitHub"
git pull

echo ""
echo "==> 2/8  Activate environment"
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "     No .venv found — run 'bash scripts/install-mac.sh' first."
  exit 1
fi

echo ""
echo "==> 3/8  Harmonise raw results"
python analysis/harmonise.py

echo ""
echo "==> 4/8  Cross-model rank agreement within CrowS-Pairs"
python analysis/agreement.py

echo ""
echo "==> 5/8  Cross-benchmark rank agreement (direct RQ1 answer)"
python analysis/cross_benchmark.py

echo ""
echo "==> 6/8  Between-model divergence per category (RQ2 signal)"
python analysis/divergence.py

echo ""
echo "==> 7/8  Generate figures and summary tables"
python analysis/report.py

echo ""
echo "==> 8/8  Cross-benchmark visualisations (Figures 4.3-4.6)"
python analysis/plot_cross_benchmark.py

echo ""
echo "==> Done. Outputs in:"
echo "     results/harmonised.parquet    (raw wide table)"
echo "     results/harmonised.csv        (same, spreadsheet-friendly)"
echo "     results/agreement.csv         (pairwise Spearman/Kendall, within CrowS-Pairs)"
echo "     results/cross_benchmark.csv   (pairwise Spearman/Kendall, across benchmarks — RQ1)"
echo "     results/divergence.csv        (per-category between-model spread)"
echo "     results/tables/               (model summary, category summary)"
echo "     results/figures/              (heatmap, per-category bar chart)"
echo ""
echo "==> Open the two figures with:"
echo "     open results/figures/model_x_category_heatmap.png"
echo "     open results/figures/per_category_ranking.png"
