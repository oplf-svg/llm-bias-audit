#!/usr/bin/env bash
# Delete the cached HuggingFace weights of a specific model.
# Use this between models on a small-disk pod to free space.
# Results in results/full/<model>/ are preserved.
#
# Usage:
#   bash scripts/cleanup-cache.sh <hf-repo-id>
# Example:
#   bash scripts/cleanup-cache.sh microsoft/Phi-3-mini-4k-instruct

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hf-repo-id>"
  echo "Example: $0 microsoft/Phi-3-mini-4k-instruct"
  exit 1
fi

MODEL="$1"
SAFE="${MODEL//\//_}"
HF_SAFE="${MODEL//\//--}"

echo "==> Deleting cached weights for ${MODEL}"
rm -rf "${HOME}/.cache/huggingface/hub/models--${HF_SAFE}"
rm -rf "/workspace/.cache/huggingface/hub/models--${HF_SAFE}"

# Also clear any partial (incomplete) downloads that may have failed
find "${HOME}/.cache/huggingface" -name "*.incomplete" -delete 2>/dev/null || true
find "/workspace/.cache/huggingface" -name "*.incomplete" -delete 2>/dev/null || true

echo "==> Done. Space now free:"
df -h /workspace 2>/dev/null | head -2

# Confirm the results folder is untouched
if [[ -d "results/full/${SAFE}" ]]; then
  echo ""
  echo "==> Results for ${MODEL} preserved at results/full/${SAFE}/"
  ls -la "results/full/${SAFE}" | head -10
fi
