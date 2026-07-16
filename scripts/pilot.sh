#!/usr/bin/env bash
# Pilot run - single model (Phi-3-mini), single benchmark (CrowS-Pairs).
# On M4 MacBook: ~15 min. Sanity-checks the whole pipeline.

set -euo pipefail

# shellcheck disable=SC1091
source .venv/bin/activate

MODEL="microsoft/Phi-3-mini-4k-instruct"
TASK="crows_pairs_english"
OUT_DIR="./results/pilot"
mkdir -p "$OUT_DIR"

if [[ "$(uname -s)" == "Darwin" ]]; then
  # macOS: MPS backend, no bitsandbytes
  MODEL_ARGS="pretrained=${MODEL},dtype=float16,device=mps"
else
  # Linux/CUDA: bitsandbytes 4-bit NF4
  MODEL_ARGS="pretrained=${MODEL},load_in_4bit=True,bnb_4bit_quant_type=nf4"
fi

echo "==> Pilot: ${MODEL} on ${TASK}"
lm_eval \
  --model hf \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASK}" \
  --seed 42 \
  --batch_size 1 \
  --output_path "${OUT_DIR}" \
  --log_samples \
  --verbosity INFO

echo ""
echo "==> Pilot complete. Results in ${OUT_DIR}/"
ls -la "${OUT_DIR}"
