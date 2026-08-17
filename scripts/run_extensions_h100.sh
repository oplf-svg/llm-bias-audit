#!/usr/bin/env bash
# One-shot extension run: closes every GPU-side gap that RQ1, O6 and the wider
# panel need. Runs StereoSet, BBQ and (if present) the O6 custom probe against
# the same 4-model panel that produced the CrowS-Pairs numbers already on main.
#
# Sentinel per (model, benchmark) — safe to interrupt and resume.
# Autopush keeps GitHub fresh every 5 min (start scripts/autopush.sh separately).
#
# MODE=cheap : StereoSet only (~40 min, ~£2). Answers RQ1 with n=1 pairing.
# MODE=full  : StereoSet + BBQ (~7-10 h, ~£25-35). Answers RQ1 with n=3 pairings.
# MODE=all   : StereoSet + BBQ + custom_probe if custom_probe/ exists (default).

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

MODE="${MODE:-all}"
echo "==> MODE=${MODE}"

MODELS=(
  "microsoft/Phi-3-mini-4k-instruct"
  "mistralai/Mistral-7B-v0.3"
  "Qwen/Qwen2-7B"
  "meta-llama/Meta-Llama-3.1-8B"
)

# Benchmark selection per mode
BENCHMARKS=()
case "${MODE}" in
  cheap) BENCHMARKS=("stereoset") ;;
  full)  BENCHMARKS=("stereoset" "bbq") ;;
  all)   BENCHMARKS=("stereoset" "bbq")
         [[ -d "./custom_probe" ]] && BENCHMARKS+=("custom_probe") \
                                   || echo "   (skipping custom_probe — directory not present)" ;;
  *)     echo "!! Unknown MODE='${MODE}' (use cheap|full|all)"; exit 2 ;;
esac
echo "==> Benchmarks this run: ${BENCHMARKS[*]}"

OUT_ROOT="./results/full"
mkdir -p "${OUT_ROOT}"

# Carbon tracker (best-effort; ignore failure)
python -c "from codecarbon import EmissionsTracker; t=EmissionsTracker(project_name='llm-bias-audit-extensions', output_dir='${OUT_ROOT}'); t.start(); print('carbon tracker started')" 2>/dev/null &
CC_PID=$!
trap "kill $CC_PID 2>/dev/null || true" EXIT

# Optional: allow user to include the local custom_probe task
INCLUDE_PATH_ARG=()
if [[ -d "./custom_probe" ]]; then
  INCLUDE_PATH_ARG=(--include_path "./custom_probe")
fi

for MODEL in "${MODELS[@]}"; do
  SAFE="${MODEL//\//_}"
  MODEL_OUT="${OUT_ROOT}/${SAFE}"
  mkdir -p "${MODEL_OUT}"

  for BENCH in "${BENCHMARKS[@]}"; do
    BENCH_OUT="${MODEL_OUT}/${BENCH}"
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
      --tasks "${BENCH}" \
      --seed 42 \
      --batch_size auto \
      --output_path "${BENCH_OUT}" \
      --trust_remote_code \
      --log_samples \
      --verbosity INFO \
      "${INCLUDE_PATH_ARG[@]}"

    date -u +%FT%TZ > "${SENTINEL}"
    echo "== DONE  ${MODEL} × ${BENCH}   @ $(date -u +%FT%TZ)"
  done
done

echo ""
echo "==> Extension run complete. RQ1 cross-benchmark comparison is now populated."
echo "==> Next: on your Mac, git pull && bash scripts/analyse.sh"
