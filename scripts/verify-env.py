#!/usr/bin/env python
"""Sanity-check that the installed environment matches the pinned versions
and that the compute backend is available (CUDA on Linux, MPS on macOS)."""

import platform
import sys

EXPECTED = {
    "torch": "2.4.0",
    "transformers": "4.44.2",
    "accelerate": "0.34.2",
    "datasets": "2.20.0",
    "huggingface_hub": "0.24.6",
    "pandas": "2.2.3",
    "numpy": "1.26.4",
    "scipy": "1.14.1",
}

print(f"Python: {sys.version.split()[0]} ({platform.machine()} on {platform.system()})")
print()

any_mismatch = False
for pkg, want in EXPECTED.items():
    try:
        mod = __import__(pkg)
        have = getattr(mod, "__version__", "?")
        ok = "OK" if have == want else "MISMATCH"
        if have != want:
            any_mismatch = True
        print(f"  {pkg:20s} want={want:10s} have={have:10s} [{ok}]")
    except ImportError as e:
        print(f"  {pkg:20s} MISSING: {e}")
        any_mismatch = True

# Compute backend check
print()
try:
    import torch

    if torch.cuda.is_available():
        print(
            f"  CUDA:   available ({torch.cuda.get_device_name(0)}, {torch.cuda.device_count()} device(s))"
        )
        print(f"          torch.version.cuda={torch.version.cuda}")
    elif torch.backends.mps.is_available():
        print("  MPS:    available (Apple Silicon)")
    else:
        print("  BACKEND: neither CUDA nor MPS available")
        any_mismatch = True
except Exception as e:
    print(f"  BACKEND check failed: {e}")
    any_mismatch = True

# bitsandbytes only on Linux/CUDA
if platform.system() == "Linux":
    try:
        import bitsandbytes as bnb

        print(f"  bitsandbytes: {bnb.__version__}")
    except ImportError:
        print("  bitsandbytes: MISSING (required on Linux for 4-bit NF4)")
        any_mismatch = True

# MLX only on macOS
if platform.system() == "Darwin":
    try:
        import mlx

        print(f"  mlx: {getattr(mlx, '__version__', '?')}")
    except ImportError:
        print("  mlx: MISSING (recommended on macOS)")

# lm-eval-harness
print()
try:
    import lm_eval

    print(f"  lm_eval: {lm_eval.__version__}")
except (ImportError, AttributeError) as e:
    print(f"  lm_eval: import failed ({e})")
    any_mismatch = True

print()
if any_mismatch:
    print("!! Environment mismatch detected. Reinstall with the appropriate requirements file.")
    sys.exit(1)
else:
    print("OK - environment matches pinned versions.")
