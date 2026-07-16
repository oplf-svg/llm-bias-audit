# Reproducibility statement

This project aims for the strongest reproducibility that current LLM tooling permits, while being honest about where bit-exact reproducibility is not achievable.

## What is fixed

- **Dataset revisions.** BBQ, StereoSet and CrowS-Pairs are downloaded from HuggingFace at pinned dataset commit hashes (see `configs/benchmarks.yaml`). Not "latest".
- **Model revisions.** Every model in the panel is loaded by `repo_id` + explicit commit hash (see `configs/models.yaml`).
- **Harness revision.** EleutherAI `lm-evaluation-harness` is pinned to `v0.4.7`.
- **Random seed.** `42` everywhere (harness runs, bootstrap resampling, analysis).
- **Decoding.** Greedy decoding for all models. No sampling.
- **Quantisation.** 4-bit NF4 via `bitsandbytes` on Linux/CUDA (production). GGUF Q4_K_M via `llama.cpp` on macOS (dev/pilot only).
- **Hardware class.** All full-panel runs execute on the same GPU class (RTX 4090 24 GB, cloud spot). Same class = same floating-point behaviour for the comparison.
- **Python environment.** Pinned in `requirements.txt` / `requirements-mac.txt`. Docker image (`Dockerfile`) pins CUDA + Python + packages together.
- **Raw output.** Every per-item score is written to JSONL via `lm_eval --log_samples`. Anyone can re-analyse from primary output without re-running inference.

## What is NOT bit-deterministic

- **`bitsandbytes` 4-bit quantisation** is not bit-deterministic across CUDA versions. Same model + same seed on a different GPU can produce slightly different logits.
- **PyTorch floating-point ops** on different hardware families (T4 vs RTX 4090 vs A100) are not bit-identical.
- **Tokeniser drift.** HuggingFace occasionally updates tokenisers without changing the model revision; pin tokenisers separately if this matters.

## Multi-seed protocol

To establish an honest bar for interpreting cross-benchmark disagreement, one representative model (typically Phi-3-mini for cost reasons) is run three times with different seeds (`42`, `1337`, `2718`) across the whole benchmark suite. Between-seed variance is reported alongside the main results. If cross-benchmark disagreement is smaller than between-seed variance, the disagreement is not meaningful.

## Reproducing this study

1. `git clone` the repository at the tagged release used in the dissertation.
2. Build the Docker image: `docker build -t llm-bias-audit .`.
3. Provision an RTX 4090 24 GB (or equivalent) instance.
4. Run: `docker run --gpus all -v $(pwd):/work -w /work llm-bias-audit bash scripts/run_all.sh`.
5. Expected wall clock: 8-12 GPU-hours. Expected cost (RunPod spot): $5-10.

## Environmental footprint

Compute carbon cost is logged with CodeCarbon (`emissions.csv` in `results/full/`) and reported in Chapter 5 of the dissertation.
