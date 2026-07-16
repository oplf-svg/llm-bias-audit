#!/usr/bin/env bash
# Setup script for Linux/CUDA (RTX 30/40 series, A100, or cloud spot instances).
# Full production runs happen here.

set -euo pipefail

echo "==> Verifying Linux + NVIDIA CUDA"
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "!! Not Linux. Use scripts/install-mac.sh instead."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null; then
  echo "!! nvidia-smi not found. Install NVIDIA driver first."
  exit 1
fi

echo "==> GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "==> Installing Python 3.11 and uv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "==> Creating virtual environment"
uv venv --python 3.11
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing Linux/CUDA requirements"
uv pip install -r requirements.txt

echo "==> Verifying installation"
python scripts/verify-env.py

echo ""
echo "==> Done. Activate your environment with:"
echo "      source .venv/bin/activate"
echo ""
echo "==> Next: authenticate with HuggingFace:"
echo "      huggingface-cli login"
echo ""
echo "==> Then run the full panel:"
echo "      bash scripts/run_all.sh"
