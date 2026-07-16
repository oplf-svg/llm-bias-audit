#!/usr/bin/env bash
# Pilot run - single model (Phi-3-mini), single benchmark (CrowS-Pairs).
# On M4 MacBook: ~15 min. On Colab T4: ~15 min.

set -euo pipefail

# Activate the venv if it exists (local runs); otherwise use system Python (Colab).
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

MODEL="microsoft/Phi-3-mini-4k-instruct"
TASK="crows_pairs_english"
OUT_DIR="./results/pilot"
mkdir -p "$OUT_DIR"

# Pick backend based on the host
if [[ "$(uname -s)" == "Darwin" ]]; then
  MODEL_ARGS="pretrained=${MODEL},dtype=float16,device=mps"
elif command -v nvidia-smi >/dev/null 2>&1; then
  MODEL_ARGS="pretrained=${MODEL},load_in_4bit=True,bnb_4bit_quant_type=nf4"
else
  MODEL_ARGS="pretrained=${MODEL},dtype=float16"
fi

echo "==> Pilot: ${MODEL} on ${TASK}"
echo "==> Model args: ${MODEL_ARGS}"
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
