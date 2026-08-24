#!/usr/bin/env bash
# One-shot RunPod bootstrap.
# Assumes GH_TOKEN and HF_TOKEN are set in the environment.
# Optional: set MODE=h100 to use the fp16 script instead of the 4-bit script.

set -euo pipefail

if [[ -z "${GH_TOKEN:-}" ]] || [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Please export GH_TOKEN and HF_TOKEN first."
  echo "  export GH_TOKEN=<your github PAT with Contents:Read+Write>"
  echo "  export HF_TOKEN=<your huggingface read token>"
  exit 1
fi

# Fork-friendly: override GH_USER / REPO_NAME on the command line to point at your own fork.
GH_USER="${GH_USER:-oplf-svg}"
REPO_NAME="${REPO_NAME:-llm-bias-audit}"

REPO_DIR="/workspace/${REPO_NAME}"
REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/${GH_USER}/${REPO_NAME}.git"

echo "==> Clone or update repo"
if [[ -d "${REPO_DIR}/.git" ]]; then
  cd "${REPO_DIR}"; git remote set-url origin "${REPO_URL}"; git pull --rebase origin main
else
  rm -rf "${REPO_DIR}"; git clone --depth=1 "${REPO_URL}" "${REPO_DIR}"; cd "${REPO_DIR}"
fi

echo ""
echo "==> Install pinned dependencies"
bash scripts/install-linux.sh
# shellcheck disable=SC1091
source .venv/bin/activate

echo ""
echo "==> Configure git identity for auto-push"
git config user.email "${GH_USER}@users.noreply.github.com"
git config user.name "${GH_USER}"

echo ""
echo "==> Authenticate HuggingFace"
huggingface-cli login --token "${HF_TOKEN}"

echo ""
echo "==> Start autopush in background (pushes results/ every 5 min)"
nohup bash scripts/autopush.sh > autopush.log 2>&1 &
echo "  autopush PID: $!"

echo ""
echo "==> Launch full panel"
if [[ "${MODE:-}" == "h100" ]]; then
  echo "  using H100 fp16 script"
  nohup bash scripts/run_all_h100.sh > run.log 2>&1 &
else
  echo "  using standard 4-bit script"
  nohup bash scripts/run_all.sh > run.log 2>&1 &
fi
RUN_PID=$!
echo "  run PID: ${RUN_PID}"

echo ""
echo "==> Follow progress with: tail -f run.log"
echo "==> Check autopush with:  tail -f autopush.log"
echo "==> Kill main run with:   kill ${RUN_PID}"
