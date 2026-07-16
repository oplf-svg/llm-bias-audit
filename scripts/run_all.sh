#!/usr/bin/env bash
# Full panel run with automatic resume.
# Safe to interrupt (Ctrl-C or spot-instance kill) and rerun - already-completed
# (model x task) pairs are skipped based on the presence of a completion sentinel.

set -euo pipefail

# shellcheck disable=SC1091
source .venv/bin/activate

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "!! Full panel is Linux/CUDA only. Use scripts/pilot.sh on macOS."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null; then
  echo "!! nvidia-smi not found. This script requires a CUDA GPU."
  exit 1
fi

MODELS=(
  "microsoft/Phi-3-mini-4k-instruct"
  "mistralai/Mistral-7B-v0.3"
  "Qwen/Qwen2-7B"
  "meta-llama/Meta-Llama-3.1-8B"
  # Add one 2025-2026 release here once selected
)

TASKS="crows_pairs_english,bbq,stereoset,custom_probe"
OUT_ROOT="./results/full"
mkdir -p "${OUT_ROOT}"

# CodeCarbon
CARBON_LOG="${OUT_ROOT}/emissions.csv"
python -c "from codecarbon import EmissionsTracker; t=EmissionsTracker(project_name='llm-bias-audit', output_dir='${OUT_ROOT}'); t.start(); print('carbon tracker started')" 2>/dev/null &
CC_PID=$!
trap "kill $CC_PID 2>/dev/null || true" EXIT

echo "==> Full panel: ${#MODELS[@]} models x $(echo ${TASKS} | tr ',' ' ' | wc -w) benchmarks"
echo "==> Output root: ${OUT_ROOT}"
echo ""

for MODEL in "${MODELS[@]}"; do
  SAFE="${MODEL//\//_}"
  MODEL_OUT="${OUT_ROOT}/${SAFE}"
  SENTINEL="${MODEL_OUT}/.completed"

  # ---- Skip if this model already ran to completion ----
  if [[ -f "${SENTINEL}" ]]; then
    echo "== SKIP  ${MODEL} (already complete: ${SENTINEL})"
    continue
  fi

  mkdir -p "${MODEL_OUT}"
  echo ""
  echo "================================================================"
  echo "== RUN   ${MODEL}"
  echo "================================================================"

  lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL},load_in_4bit=True,bnb_4bit_quant_type=nf4" \
    --tasks "${TASKS}" \
    --include_path ./custom_probe \
    --seed 42 \
    --batch_size auto \
    --output_path "${MODEL_OUT}" \
    --log_samples \
    --verbosity INFO

  # Mark this model complete
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${SENTINEL}"
  echo "== DONE  ${MODEL}"
done

echo ""
echo "==> All models complete."
echo ""
echo "==> Next: run the analysis pipeline"
echo "      make analyse"
echo "      make report"
