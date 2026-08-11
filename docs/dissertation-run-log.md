# Dissertation-phase run log

Complete record of the run that produced the Chapter 4 numbers in the dissertation. Kept as evidence of what actually happened and what to fix next time.

## Overview

| Field | Value |
|---|---|
| **Purpose** | Full four-model panel on nine CrowS-Pairs English categories |
| **Date of pilot** | 16-17 July 2026 (6 runs on Google Colab T4) |
| **Date of full panel** | 11 August 2026 (single RunPod H100 session) |
| **Pilot cost** | £0 (Colab free tier) |
| **Full-panel cost** | ~£2 (H100 PCIe spot pricing, 1.5 h wall clock) |
| **Total (model × category) cells produced** | 36 (4 models × 9 categories) |

## Hardware

**Pilot (all 6 runs):**
- Provider: Google Colab (free tier)
- GPU: NVIDIA T4 16 GB
- Python: 3.12
- CUDA: 12.4
- Torch: 2.4.0

**Full-panel run:**
- Provider: RunPod Community Cloud
- Pod template: **PyTorch 2.8.0** (with CUDA 12.8 base)
- GPU: NVIDIA H100 PCIe 80 GB HBM3
- vCPU: 8
- RAM: 125 GB
- Storage: 60 GB volume (later resized to 94 GB — see Issues section)
- Rate: $3.29/hr
- Total pod runtime: ~50 min (~$2.75)

## Model panel

| Model | HF repo | Params | Licence | Gated? |
|---|---|---|---|---|
| Phi-3-mini-4k-instruct | `microsoft/Phi-3-mini-4k-instruct` | 3.8B | MIT | No |
| Mistral-7B-v0.3 | `mistralai/Mistral-7B-v0.3` | 7B | Apache-2.0 | No |
| Qwen2-7B | `Qwen/Qwen2-7B` | 7B | Apache-2.0 | No |
| Meta-Llama-3.1-8B | `meta-llama/Meta-Llama-3.1-8B` | 8B | Llama 3.1 Community | Yes |

## Benchmarks

Nine per-category CrowS-Pairs English subtasks, registered in `lm-evaluation-harness v0.4.7`:
`crows_pairs_english_age`, `_disability`, `_gender`, `_nationality`, `_physical_appearance`, `_race_color`, `_religion`, `_sexual_orientation`, `_socioeconomic`.

Each subtask evaluates the full 1,677-item CrowS-Pairs set (3,354 log-likelihood requests) filtered to that category's items. Every model was scored with `--seed 42`, greedy decoding, `--batch_size auto`, `--trust_remote_code`, `--log_samples`.

## Environment (locked)

Every dependency pinned in `requirements.txt` at commit `84cd492` (the tag `submission-2026-07-17`). Key versions:

- torch `2.4.0+cu121`
- transformers `4.44.2`
- accelerate `0.34.2`
- bitsandbytes `0.43.3` (installed but not used by the H100 fp16 run)
- datasets `2.20.0`
- huggingface_hub `0.24.6`
- hf_transfer `0.1.8`
- numpy `1.26.4`
- lm-eval `v0.4.7` (git commit pinned in `requirements.txt`)

## Timing

| Model | Started (UTC) | Ended (UTC) | Duration |
|---|---|---|---|
| Phi-3-mini-4k-instruct | 12:21:32 | 12:25:20 | 3 m 48 s |
| Mistral-7B-v0.3 | 12:25:20 | 12:33:29 | 8 m 09 s |
| Qwen2-7B | 12:33:29 | 12:43:23 | 9 m 54 s |
| Meta-Llama-3.1-8B (first attempt) | 12:43:23 | 12:43:53 | **failed** — disk full |
| *(pod resize from 60 GB → 94 GB)* | 13:XX | | |
| Meta-Llama-3.1-8B (retry) | 14:06:14 | 14:17:32 | 11 m 18 s |

**Total GPU-active time on H100:** ~33 minutes.

## Headline numbers (see also §4.2 of the dissertation)

Panel mean pct_stereotype: **0.7037** (std 0.0856 across 36 cells).

Per-model means (sorted): Meta-Llama-3.1-8B 0.7255, Mistral-7B-v0.3 0.7185, Qwen2-7B 0.7082, Phi-3-mini-4k-instruct 0.6625.

Cross-model rank agreement (Spearman ρ on the 9-category vector) ranges from **0.733** (Phi-3 vs Mistral) to **0.983** (Llama vs Mistral).

Between-model per-category spread ranges from **0.028** (physical_appearance) to **0.161** (sexual_orientation).

## Reproducibility validation

Pilot on Phi-3-mini executed 6 times on independent Colab sessions between 16 and 17 July 2026. All 6 runs produced bit-identical scores to 15 decimal places (`pct_stereotype = 0.629695885509839` on the aggregate CrowS-Pairs task). This confirms the pinned harness + dataset + model + seed produce repeatable results within a single hardware class.

Full-panel run on H100 (fp16, no quantisation) produced pct_stereotype values 3-4 percentage points lower than the pilot's 4-bit NF4 values for the same model on the same task. This is the expected magnitude of bitsandbytes 4-bit non-determinism, and is why the primary Chapter 4 results use fp16.

## Issues encountered and fixes applied

**Issue 1 — Pod bootstrap failed at verify step.** `install-linux.sh` returned exit 1 because `verify-env.py` did strict string equality on `torch.__version__` and treated `2.4.0+cu121` as a mismatch with `2.4.0`.  
*Fix:* strip the `+<cuda-tag>` suffix before comparing. Committed to `scripts/verify-env.py`.

**Issue 2 — HuggingFace CLI login failed.** `huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential=False` errored with "argument --add-to-git-credential: ignored explicit argument 'False'". Turns out `--add-to-git-credential` is a switch, not a value flag.  
*Fix:* drop the argument entirely. The token is saved to `/workspace/.cache/huggingface/token` and picked up by `transformers` automatically.

**Issue 3 — `hf_transfer` package missing.** The RunPod PyTorch template sets `HF_HUB_ENABLE_HF_TRANSFER=1` by default, but does not install the `hf_transfer` Python package. First evaluation attempt failed immediately at model download with `ValueError: Fast download using 'hf_transfer' is enabled ... but package is not available`.  
*Fix:* `pip install hf_transfer` (added to `requirements.txt` as a pin). Belt-and-braces: `unset HF_HUB_ENABLE_HF_TRANSFER` before running.

**Issue 4 — Disk full during Llama download.** 60 GB volume filled after Phi-3 + Mistral + Qwen2 caches consumed ~35 GB; Llama needed 16 more GB of blob space and only 1.7 GB was free.  
*Fix:* resized volume from 60 GB → 94 GB via the RunPod dashboard, then restarted the run. The `.completed` sentinels on `/workspace/` made the restart skip the three finished models and go straight to Llama.  
*Prevention:* provision 150 GB from the start, OR run `bash scripts/cleanup-cache.sh <finished-model>` after each `== DONE` (script added to the repo as evidence).

**Issue 5 — BBQ and StereoSet not registered under those names in lm-eval v0.4.7.** Instead the harness registers `bigbench_bbq_lite_json_multiple_choice` (a subset), and StereoSet is entirely absent.  
*Fix (interim):* run only the 9 CrowS-Pairs subtasks for the Chapter 4 results.  
*Fix (dissertation phase):* custom task YAMLs for full BBQ (11 categories) and StereoSet (intra + inter) added under `custom_tasks/bbq/` and `custom_tasks/stereoset/`. Templates, not yet validated.

**Issue 6 — Autopush process died alongside the failed run.** The `[1]- Exit 128` and `[2]+ Exit 1` bash notifications alongside the disk-full crash killed both the run script and the autopush loop.  
*Fix:* restart both after each crash. Autopush is idempotent and the run script is resumable.

## Artefacts

**On GitHub (tag `submission-2026-07-17`):**
- Source of every script exactly as it ran
- `results/pilot/environment.json` — pilot environment snapshot
- `results/pilot/REPRODUCIBILITY.md` — human-readable pilot log
- `results/pilot/microsoft__Phi-3-mini-4k-instruct/results_*.json` — raw pilot output
- `results/full/<model>/results_*.json` — raw full-panel output, 4 files, one per model
- `results/full/<model>/samples_*.jsonl` — per-item log-samples, 4 files
- `results/full/<model>/.completed` — 4 sentinels
- `results/full/emissions.csv` — CodeCarbon output

**On the researcher's local Mac (regenerated by `bash scripts/analyse.sh`):**
- `results/harmonised.parquet` and `.csv`
- `results/agreement.csv`, `results/divergence.csv`
- `results/tables/model_summary.csv`, `category_summary.csv`, `model_x_category.csv`
- `results/figures/model_x_category_heatmap.png`
- `results/figures/per_category_ranking.png`

## What comes next (dissertation phase)

1. Validate `custom_tasks/bbq/` against the `heegyu/bbq` schema and duplicate `task_bbq_age.yaml` for the 10 remaining BBQ categories.
2. Validate `custom_tasks/stereoset/` against `McGill-NLP/stereoset`.
3. Re-run `scripts/run_all_h100.sh` with the extended task list — this gives the true three-benchmark comparison for RQ1.
4. Multi-seed replication on one representative model (seeds 42, 1337, 2718) to establish between-seed variance for the reproducibility statement.
5. Expand `custom_tasks/probe/items.jsonl` from 5 seed items to 50-150 items.
6. Once the extended panel is done, rerun `bash scripts/analyse.sh` locally and update the dissertation Chapter 4 tables.
