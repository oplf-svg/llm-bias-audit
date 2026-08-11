#!/usr/bin/env bash
# Full panel run for H100-class GPUs (80 GB VRAM).
# Uses fp16 rather than 4-bit NF4 quantisation, so results are more
# accurate and remove the bitsandbytes non-determinism caveat.
# Safe to interrupt (Ctrl-C or spot-instance kill): the .completed sentinel
# per model means restarting the script skips already-finished models.

set -euo pipefail

# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "!! Linux/CUDA only."; exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "!! nvidia-smi not found."; exit 1
fi

echo "==> GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

MODELS=(
  "microsoft/Phi-3-mini-4k-instruct"
  "mistralai/Mistral-7B-v0.3"
  "Qwen/Qwen2-7B"
  "meta-llama/Meta-Llama-3.1-8B"
)

TASKS="crows_pairs_english_age,crows_pairs_english_disability,crows_pairs_english_gender,crows_pairs_english_nationality,crows_pairs_english_physical_appearance,crows_pairs_english_race_color,crows_pairs_english_religion,crows_pairs_english_sexual_orientation,crows_pairs_english_socioeconomic"

OUT_ROOT="./results/full"
mkdir -p "${OUT_ROOT}"

python -c "from codecarbon import EmissionsTracker; t=EmissionsTracker(project_name='llm-bias-audit-h100', output_dir='${OUT_ROOT}'); t.start(); print('carbon tracker started')" 2>/dev/null &
CC_PID=$!
trap "kill $CC_PID 2>/dev/null || true" EXIT

echo ""
echo "==> H100 full panel: ${#MODELS[@]} models x 9 CrowS-Pairs categories (fp16, no quantisation)"

for MODEL in "${MODELS[@]}"; do
  SAFE="${MODEL//\//_}"
  MODEL_OUT="${OUT_ROOT}/${SAFE}"
  SENTINEL="${MODEL_OUT}/.completed"

  if [[ -f "${SENTINEL}" ]]; then
    echo ""; echo "== SKIP  ${MODEL}"; continue
  fi

  mkdir -p "${MODEL_OUT}"
  echo ""
  echo "================================================================"
  echo "== RUN   ${MODEL}   @ $(date -u +%FT%TZ)"
  echo "================================================================"

  # fp16 — no bitsandbytes, no quantisation artefacts
  lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL},dtype=float16" \
    --tasks "${TASKS}" \
    --seed 42 \
    --batch_size auto \
    --output_path "${MODEL_OUT}" \
    --trust_remote_code \
    --log_samples \
    --verbosity INFO

  date -u +%FT%TZ > "${SENTINEL}"
  echo "== DONE  ${MODEL}   @ $(date -u +%FT%TZ)"
done

echo ""
echo "==> All models complete."
