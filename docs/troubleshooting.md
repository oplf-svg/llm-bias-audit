# Troubleshooting

## macOS pilot

**"No module named `bitsandbytes`" or "CUDA not available"**
Expected on macOS. The pilot script auto-detects Darwin and switches to the MPS backend. Full-panel runs are Linux/CUDA only.

**"MPS backend out of memory"**
Phi-3-mini (3.8B) needs ~6 GB unified RAM in float16. Close browser tabs and other memory-heavy apps. If it still fails, drop to a GGUF quantised model via `llama.cpp`.

**Model download hangs**
`huggingface-cli login` first, then rerun. Gated models (Llama) need a licence-acceptance click in the browser.

## Linux full-panel

**"CUDA out of memory"**
Reduce `--batch_size` from `auto` to `1`. If still out, the GPU has < 16 GB VRAM and cannot run 7-8B models in 4-bit; use Phi-3-mini only or move to a larger instance.

**"bitsandbytes: no CUDA-compatible library"**
Install order matters. Reinstall PyTorch first (with the matching CUDA), then bitsandbytes. On RunPod, pick the "PyTorch 2.4.0 + CUDA 12.1" template.

**Very slow evaluation**
Confirm the GPU is actually used: `nvidia-smi` should show 100% utilisation while `lm_eval` runs. If it's at 0%, the model is on CPU. Fix by passing `device=cuda` explicitly in `--model_args`.

## Analysis

**`results_*.json` not found**
`lm_eval` writes to `--output_path`. Check `results/full/<model>/` exists and contains at least one file.

**Zero rows in `divergence.csv`**
Per-category scores come from the per-sample JSONL (`--log_samples`), not from the aggregate `results_*.json`. Ensure `--log_samples` is set in `run_all.sh` and update `analysis/harmonise.py` to parse the per-sample files.
