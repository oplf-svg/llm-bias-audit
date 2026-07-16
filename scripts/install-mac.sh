#!/usr/bin/env bash
# Setup script for Apple Silicon MacBook (M1/M2/M3/M4).
# Development + pilot only - full production runs happen on Linux/CUDA.

set -euo pipefail

echo "==> Verifying macOS + Apple Silicon"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "!! Not macOS. Use scripts/install-linux.sh instead."
  exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
  echo "!! Not Apple Silicon (arch=$ARCH). MPS/MLX backends require arm64."
  exit 1
fi

echo "==> Checking Homebrew"
if ! command -v brew >/dev/null; then
  echo "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "==> Installing Python 3.11 and uv"
brew install python@3.11 uv git-lfs
git lfs install

echo "==> Creating virtual environment"
uv venv --python 3.11
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing macOS/MPS/MLX requirements"
uv pip install -r requirements-mac.txt

echo "==> Verifying installation"
python scripts/verify-env.py

echo ""
echo "==> Done. Activate your environment with:"
echo "      source .venv/bin/activate"
echo ""
echo "==> Next: authenticate with HuggingFace (needed for Llama-3.1 access):"
echo "      huggingface-cli login"
echo ""
echo "==> Then run the pilot:"
echo "      bash scripts/pilot.sh"
