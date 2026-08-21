#!/usr/bin/env bash
# Re-run CrowS-Pairs on Phi-3-mini and check bit-exact reproduction against the
# original dissertation-phase results already on GitHub.
#
# Writes results to results/repro_check/<pod-timestamp>/ so it doesn't clobber
# the original run.
#
# Usage:  bash scripts/reproducibility_check.sh

set -euo pipefail

# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

MODEL="microsoft/Phi-3-mini-4k-instruct"
TASKS="crows_pairs_english_age,crows_pairs_english_disability,crows_pairs_english_gender,crows_pairs_english_nationality,crows_pairs_english_physical_appearance,crows_pairs_english_race_color,crows_pairs_english_religion,crows_pairs_english_sexual_orientation,crows_pairs_english_socioeconomic"

OUT="./results/repro_check/$(date -u +%FT%TZ | tr : -)"
mkdir -p "${OUT}"

echo "==> Reproducibility check: ${MODEL} × CrowS-Pairs (9 categories)"
nvidia-smi --query-gpu=name --format=csv,noheader

lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL},dtype=float16" \
  --tasks "${TASKS}" \
  --seed 42 \
  --batch_size auto \
  --output_path "${OUT}" \
  --trust_remote_code \
  --verbosity INFO

echo ""
echo "==> Written to ${OUT}"
echo "==> Compare pct_stereotype values against:"
echo "    results/full/microsoft_Phi-3-mini-4k-instruct/microsoft__Phi-3-mini-4k-instruct/results_2026-08-11T*.json"
echo ""
echo "==> Suggested Python diff:"
cat <<'PY'
python - <<'EOF'
import json, glob
orig = sorted(glob.glob("results/full/microsoft_Phi-3-mini-4k-instruct/microsoft__Phi-3-mini-4k-instruct/results_2026-08-11T*.json"))[-1]
repro = sorted(glob.glob("results/repro_check/*/microsoft__Phi-3-mini-4k-instruct/results_*.json"))[-1]
o = json.load(open(orig))["results"]; r = json.load(open(repro))["results"]
print(f"{'category':<40} {'orig':>10} {'repro':>10} {'diff':>10}")
for k in sorted(o):
    ov = o[k].get("pct_stereotype,none"); rv = r.get(k, {}).get("pct_stereotype,none")
    if ov is None or rv is None: continue
    print(f"{k:<40} {ov:>10.4f} {rv:>10.4f} {ov-rv:>10.4f}")
EOF
PY
