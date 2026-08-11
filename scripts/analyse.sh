#!/usr/bin/env bash
# One-shot analysis pipeline for your Mac.
# Pulls latest results from GitHub and runs harmonise -> agreement -> divergence -> report.
# Assumes you have already run scripts/install-mac.sh at some point.

set -euo pipefail

echo "==> 1/6  Pull latest results from GitHub"
git pull

echo ""
echo "==> 2/6  Activate environment"
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "     No .venv found — run 'bash scripts/install-mac.sh' first."
  exit 1
fi

echo ""
echo "==> 3/6  Harmonise raw results"
python analysis/harmonise.py

echo ""
echo "==> 4/6  Cross-model rank agreement (preliminary RQ1 signal)"
python analysis/agreement.py

echo ""
echo "==> 5/6  Between-model divergence per category (preliminary RQ2 signal)"
python analysis/divergence.py

echo ""
echo "==> 6/6  Generate figures and summary tables"
python analysis/report.py

echo ""
echo "==> Done. Outputs in:"
echo "     results/harmonised.parquet    (raw wide table)"
echo "     results/harmonised.csv        (same, spreadsheet-friendly)"
echo "     results/agreement.csv         (pairwise Spearman/Kendall)"
echo "     results/divergence.csv        (per-category between-model spread)"
echo "     results/tables/               (model summary, category summary)"
echo "     results/figures/              (heatmap, per-category bar chart)"
echo ""
echo "==> Open the two figures with:"
echo "     open results/figures/model_x_category_heatmap.png"
echo "     open results/figures/per_category_ranking.png"
