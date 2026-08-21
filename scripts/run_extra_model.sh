#!/usr/bin/env bash
# Add a NEW model to the existing panel — runs it on both CrowS-Pairs and BBQ.
#
# Usage:  MODEL=google/gemma-2-9b bash scripts/run_extra_model.sh
#         MODEL=Qwen/Qwen2.5-7B     bash scripts/run_extra_model.sh
#         MODEL=microsoft/Phi-3.5-mini-instruct bash scripts/run_extra_model.sh
#
# Sentinel per (model, benchmark) → safe to interrupt and resume.
# Assumes autopush is already running (start it separately if not).

set -euo pipefail

: "${MODEL:?Set MODEL=<hf model id>, e.g. MODEL=google/gemma-2-9b}"

# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "!! Linux/CUDA only."; exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "!! nvidia-smi not found."; exit 1
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

BENCHMARKS=("crows_pairs_english" "bbq")
OUT_ROOT="./results/full"
SAFE="${MODEL//\//_}"
MODEL_OUT="${OUT_ROOT}/${SAFE}"
mkdir -p "${MODEL_OUT}"

# CrowS-Pairs is a group of 9 subtasks — expand explicitly
CROWS_TASKS="crows_pairs_english_age,crows_pairs_english_disability,crows_pairs_english_gender,crows_pairs_english_nationality,crows_pairs_english_physical_appearance,crows_pairs_english_race_color,crows_pairs_english_religion,crows_pairs_english_sexual_orientation,crows_pairs_english_socioeconomic"

for BENCH in "${BENCHMARKS[@]}"; do
  case "${BENCH}" in
    crows_pairs_english) BENCH_OUT="${MODEL_OUT}"; TASKS="${CROWS_TASKS}" ;;
    bbq)                 BENCH_OUT="${MODEL_OUT}/bbq"; TASKS="bbq" ;;
    *) echo "!! Unknown BENCH=${BENCH}"; exit 2 ;;
  esac

  SENTINEL="${BENCH_OUT}/.completed"
  if [[ -f "${SENTINEL}" ]]; then
    echo ""; echo "== SKIP  ${MODEL} × ${BENCH}"; continue
  fi

  mkdir -p "${BENCH_OUT}"
  echo ""
  echo "================================================================"
  echo "== RUN   ${MODEL} × ${BENCH}   @ $(date -u +%FT%TZ)"
  echo "================================================================"

  lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL},dtype=float16" \
    --tasks "${TASKS}" \
    --seed 42 \
    --batch_size auto \
    --output_path "${BENCH_OUT}" \
    --trust_remote_code \
    --log_samples \
    --verbosity INFO

  date -u +%FT%TZ > "${SENTINEL}"
  echo "== DONE  ${MODEL} × ${BENCH}   @ $(date -u +%FT%TZ)"
done

echo ""
echo "==> ${MODEL} extension complete on both CrowS-Pairs and BBQ."
echo "==> Next: on your Mac, git pull && bash scripts/analyse.sh"
