#!/usr/bin/env python
"""Sanity-check the installed environment.

Accepts:
- torch versions with a CUDA build tag (e.g. 2.4.0+cu121 matches 2.4.0)
- lm-eval package present (queried via importlib.metadata rather than __version__)
"""

import importlib.metadata as im
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


def base_version(v: str) -> str:
    # "2.4.0+cu121" -> "2.4.0"
    return v.split("+")[0]


print(f"Python: {sys.version.split()[0]} ({platform.machine()} on {platform.system()})")
print()

any_mismatch = False
for pkg, want in EXPECTED.items():
    try:
        mod = __import__(pkg)
        have = getattr(mod, "__version__", "?")
        ok = "OK" if base_version(have) == want else "MISMATCH"
        if base_version(have) != want:
            any_mismatch = True
        print(f"  {pkg:20s} want={want:10s} have={have:15s} [{ok}]")
    except ImportError as e:
        print(f"  {pkg:20s} MISSING: {e}")
        any_mismatch = True

# Backend check
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

if platform.system() == "Linux":
    try:
        bnb_ver = im.version("bitsandbytes")
        print(f"  bitsandbytes: {bnb_ver}")
    except im.PackageNotFoundError:
        # Only required if we plan to use 4-bit quantisation; H100 fp16 script doesn't need it
        print("  bitsandbytes: not installed (fp16 mode OK; required for 4-bit)")

if platform.system() == "Darwin":
    try:
        mlx_ver = im.version("mlx")
        print(f"  mlx: {mlx_ver}")
    except im.PackageNotFoundError:
        print("  mlx: MISSING (recommended on macOS)")

# lm-eval via package metadata (it doesn't expose __version__ at import time)
print()
try:
    lm_eval_ver = im.version("lm-eval")
    print(f"  lm-eval: {lm_eval_ver}")
except im.PackageNotFoundError:
    print("  lm-eval: not installed")
    any_mismatch = True

print()
if any_mismatch:
    print("!! Environment mismatch detected. Reinstall with the appropriate requirements file.")
    sys.exit(1)
print("OK - environment matches pinned versions.")
