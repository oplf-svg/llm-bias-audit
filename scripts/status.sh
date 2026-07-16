#!/usr/bin/env bash
# One-shot status: what has been run, what has not.

set -euo pipefail

OUT_ROOT="./results/full"

echo "============================================================"
echo "  llm-bias-audit  --  pipeline status"
echo "============================================================"

if [[ ! -d "${OUT_ROOT}" ]]; then
  echo ""
  echo "  No runs yet. Start with:  make pilot   (smoke test)"
  echo "                   or       make full    (Linux/CUDA)"
  exit 0
fi

MODELS=(
  "microsoft_Phi-3-mini-4k-instruct"
  "mistralai_Mistral-7B-v0.3"
  "Qwen_Qwen2-7B"
  "meta-llama_Meta-Llama-3.1-8B"
)

echo ""
echo "Models:"
for M in "${MODELS[@]}"; do
  if [[ -f "${OUT_ROOT}/${M}/.completed" ]]; then
    printf "  [x] %s  (finished %s)\n" "${M}" "$(cat ${OUT_ROOT}/${M}/.completed)"
  elif [[ -d "${OUT_ROOT}/${M}" ]]; then
    printf "  [~] %s  (in progress)\n" "${M}"
  else
    printf "  [ ] %s  (pending)\n" "${M}"
  fi
done

echo ""
echo "Analysis artefacts:"
for F in harmonised.parquet agreement.csv divergence.csv figures/agreement.png; do
  if [[ -e "./results/${F}" ]]; then
    printf "  [x] results/%s\n" "${F}"
  else
    printf "  [ ] results/%s\n" "${F}"
  fi
done

echo ""
if [[ -f "${OUT_ROOT}/emissions.csv" ]]; then
  KG=$(tail -n +2 "${OUT_ROOT}/emissions.csv" | awk -F',' '{sum+=$5} END {print sum}')
  echo "Compute carbon (CodeCarbon):  ${KG} kg CO2eq"
fi
