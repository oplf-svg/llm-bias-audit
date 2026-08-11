# Code explained, in detail

This document is a script-by-script reference for every piece of code in the repository. Each entry explains **what the script does**, **how it works**, **why it is designed the way it is**, and **when to run it**.

Ordered by workflow role, not alphabetically:

1. [Environment setup](#environment-setup)
2. [Discovery](#discovery)
3. [Evaluation](#evaluation)
4. [Auto-push and cleanup](#auto-push-and-cleanup)
5. [Post-run wrap-up](#post-run-wrap-up)
6. [Analysis](#analysis)
7. [Custom task definitions](#custom-task-definitions)
8. [CI / repo hygiene](#ci--repo-hygiene)

---

## Environment setup

### `scripts/install-linux.sh`

**Purpose:** create a Python 3.11 virtualenv and install the pinned Linux/CUDA production stack.

**How it works:**
1. Checks the host is Linux (exits if macOS)
2. Checks `nvidia-smi` is present (exits if no NVIDIA driver)
3. Installs `uv` (fast Python installer) if not present
4. Creates `.venv/` with Python 3.11
5. `uv pip install -r requirements.txt` installs the exact pinned versions of torch 2.4.0, transformers 4.44.2, bitsandbytes 0.43.3, lm-eval @ v0.4.7, etc.
6. Runs `scripts/verify-env.py` to confirm every version matches the pin

**Why designed this way:** `uv` is 10× faster than `pip` for cold installs. Pinning every package means the same source tree produces the same numbers on any Linux+CUDA host.

**When to run:** once per fresh RunPod pod, or once per fresh Linux VM. Not needed between runs on the same pod.

---

### `scripts/install-mac.sh`

**Purpose:** same as `install-linux.sh` but for Apple Silicon macOS.

**How it works:**
- Uses Homebrew to install Python 3.11 + `uv`
- Creates `.venv/` and installs `requirements-mac.txt` (no bitsandbytes; adds MLX and llama-cpp-python for local dev)

**When to run:** once per fresh Mac, when you want to develop or run pilot experiments locally.

**Note on the current dissertation workflow:** the Mac is only used for the *analysis* step, not for evaluation. The analysis needs a much smaller subset of the stack (pandas/numpy/scipy/matplotlib/seaborn/statsmodels/pyarrow), which can be installed directly without going through `install-mac.sh`.

---

### `scripts/verify-env.py`

**Purpose:** sanity-check that every installed dependency matches its pin.

**How it works:**
- Prints Python version and platform
- Imports each pinned package (torch, transformers, accelerate, datasets, huggingface_hub, pandas, numpy, scipy) and compares its `__version__` to the expected pin
- Strips CUDA build tags (`2.4.0+cu121` matches `2.4.0`)
- Checks CUDA availability via `torch.cuda.is_available()` on Linux, MPS on macOS
- Reports `bitsandbytes` (Linux only) and `mlx` (macOS only) as optional
- Uses `importlib.metadata` for `lm-eval` because that package does not expose `__version__` at import time

**Why designed this way:** dependency drift is the single most common source of "reproducibility bug" in ML research. Explicit verification is cheaper than debugging.

**When to run:** automatically as the last step of `install-linux.sh` / `install-mac.sh`. Also manually after any `pip install`.

---

### `pyproject.toml`

**Purpose:** ruff configuration.

**How it works:** enables E (pycodestyle errors), F (pyflakes), W (pycodestyle warnings), I (isort). Ignores E501 (line-too-long, since we set line-length: 100). Python 3.11 target.

---

## Discovery

### `scripts/discover_tasks.sh`

**Purpose:** list every bias-benchmark task registered in the currently-installed `lm-evaluation-harness`.

**How it works:**
1. Activates `.venv`
2. Imports `lm_eval.tasks.TaskManager` and enumerates `all_tasks`
3. Filters for names containing `bbq`, `stereo` or `crows`
4. Prints the matches

**Why designed this way:** task names change between harness versions. The dissertation-phase harness (v0.4.7) does not register full BBQ or StereoSet by default; it does register `crows_pairs_english` and its nine `_<category>` subtasks. This script surfaces exactly which task names are usable, so you can put the right ones in `pilot.sh` / `run_all_h100.sh`.

**When to run:** once after installing lm-eval, or whenever you suspect the task registry has changed.

---

## Evaluation

### `scripts/pilot.sh`

**Purpose:** run a single benchmark on a single model to smoke-test the pipeline.

**How it works:**
1. Activates `.venv` if present (works both with and without a venv)
2. Auto-detects the backend based on host:
   - macOS → MPS (`dtype=float16,device=mps`)
   - Linux + NVIDIA → 4-bit NF4 via bitsandbytes
   - Anything else → plain float16
3. Runs `lm_eval` with `--seed 42`, `--batch_size 1`, `--trust_remote_code`, `--log_samples` on the nine CrowS-Pairs categories
4. Writes results to `results/pilot/`

**Why designed this way:** the pilot proves the whole chain (auth, download, quantisation, evaluation, output) works before you commit to a 90-minute full-panel run. `--batch_size 1` and `--seed 42` mean every run of the pilot produces the same JSON output; running it twice and diffing is a determinism test.

**When to run:** first thing on any fresh environment. Confirmed working when you see `Running loglikelihood requests` progress bars.

---

### `scripts/run_all.sh`

**Purpose:** full panel of four models × nine CrowS-Pairs categories, 4-bit NF4 (Linux+CUDA, RTX 4090 / A40 / cheaper GPUs).

**How it works:**
1. Activates `.venv`, verifies `nvidia-smi`
2. Starts CodeCarbon tracker for the panel-level compute-carbon reading (written to `results/full/emissions.csv`)
3. Loops over four models; for each model:
   - Skips if `results/full/<model>/.completed` already exists (resume-from-checkpoint)
   - Runs `lm_eval` with `load_in_4bit=True`, `bnb_4bit_quant_type=nf4`
   - Writes `.completed` sentinel on success

**Why designed this way:** the `.completed` sentinel makes the script safe to interrupt. If the pod dies mid-run (spot preemption, network hiccup, disk full), you rerun the same script and it skips already-finished models.

**When to run:** on a Linux+CUDA host with ≥16 GB VRAM (Phi-3 fits in 8 GB; the 7-8B models need ~10-12 GB in 4-bit).

---

### `scripts/run_all_h100.sh`

**Purpose:** same as `run_all.sh` but for H100-class GPUs with fp16 (no quantisation).

**How it works:**
- Identical structure to `run_all.sh` with two changes:
  - `dtype=float16` instead of `load_in_4bit=True`
  - Different CodeCarbon project name

**Why designed this way:** H100 has 80 GB VRAM, so the 7-8B models fit comfortably in fp16 (14-16 GB each). fp16 avoids the bitsandbytes non-determinism caveat that we footnote in the dissertation as a threat to validity. Better science, marginally slower per-token but the run finishes in the same wall clock because H100 is much faster.

**Why keep both `run_all.sh` and `run_all_h100.sh`:** documenting both paths lets the study be reproduced on cheaper hardware. `run_all.sh` on an A40 or RTX 4090 costs ~$1-2; `run_all_h100.sh` on an H100 costs ~$5-6. The values differ within the reproducibility caveat.

**When to run:** on H100 (or any GPU with ≥40 GB VRAM). Requires 150 GB disk for the four-model cache (see `cleanup-cache.sh` if less).

---

### `scripts/runpod-quickstart.sh`

**Purpose:** one-shot bootstrap for a fresh RunPod pod.

**How it works:**
1. Requires `GH_TOKEN` and `HF_TOKEN` in env; exits with instructions if missing
2. Clones (or `git pull`s) the repo into `/workspace/llm-bias-audit`
3. Runs `install-linux.sh` inside the venv
4. Configures git identity for autopush (`oplf-svg@users.noreply.github.com`)
5. Logs into HuggingFace with the token
6. Starts `autopush.sh` in background via `nohup`
7. Starts `run_all.sh` (default) or `run_all_h100.sh` (if `MODE=h100` set) in background via `nohup`
8. Prints instructions for following progress (`tail -f run.log`)

**Why designed this way:** everything after "deploy pod → paste 3 lines" is autonomous. You disconnect from the pod, the run continues, results push to GitHub every 5 minutes.

**When to run:** right after deploying a fresh RunPod pod.

---

## Auto-push and cleanup

### `scripts/autopush.sh`

**Purpose:** every N seconds, commit and push any new files in `results/` to GitHub.

**How it works:**
1. Infinite loop with `sleep 300` (5 minutes by default; override with `INTERVAL=X`)
2. Each iteration: `git pull --rebase --autostash origin main`, then `git add -f results/`, then commit + push if there are changes
3. Uses the git remote URL that `runpod-quickstart.sh` set (contains the PAT)

**Why designed this way:** ML experiments on cloud GPUs are expensive-to-restart. Autopush means that even if the pod dies mid-run, whatever finished before the crash is already on GitHub. No lost work.

**Why `git pull --rebase --autostash` first:** avoids merge conflicts if another process (e.g. the researcher manually) pushes to `main` in parallel.

**When to run:** started automatically by `runpod-quickstart.sh`. Runs in background until you kill it (`kill <PID>` or reboot).

---

### `scripts/cleanup-cache.sh <hf-repo-id>`

**Purpose:** delete the cached HuggingFace weights of one specific model to free disk space between models.

**How it works:**
1. Computes the safe path from the HF repo id (`microsoft/Phi-3-mini-4k-instruct` → `models--microsoft--Phi-3-mini-4k-instruct`)
2. Deletes from both `~/.cache/huggingface/hub/` and `/workspace/.cache/huggingface/hub/`
3. Deletes any `.incomplete` partial downloads
4. Reports disk free space and confirms the results folder for that model is untouched

**Why designed this way:** the dissertation-phase run started with a 60 GB volume and hit the disk-space limit mid-way through the panel (Phi + Mistral + Qwen caches consumed ~35 GB and Llama needed 16 more). This script is the mid-run intervention that frees space model-by-model without losing any results.

**When to run:** after each `== DONE <model>` line if you provisioned less than 150 GB volume. Provision 150 GB from the start to avoid needing this.

---

## Post-run wrap-up

### `scripts/finish-runpod.sh`

**Purpose:** sanity-check the run finished cleanly and push everything to GitHub.

**How it works:**
1. Counts `.completed` sentinels — bails if fewer than 4
2. Prints the full `== RUN`/`== DONE` timing summary from `run.log`
3. Force-pushes anything that autopush may have missed
4. Reports whether autopush is still alive

**When to run:** once, when the last `== DONE` line appears in `run.log`.

---

### `scripts/status.sh`

**Purpose:** at-a-glance progress report.

**How it works:** counts `.completed` sentinels, checks whether analysis artefacts exist, prints CodeCarbon reading if available.

**When to run:** any time you want a quick status without wading through `run.log`.

---

## Analysis

### `analysis/harmonise.py`

**Purpose:** ingest every `results_*.json` written by `lm_eval` and produce a single wide `pandas.DataFrame`, one row per (model, task, metric, score).

**How it works:**
1. Walks `results/full/**/*.json`, parses each with `json.loads`
2. For each `(model, task)`, looks up the primary metric based on the task-name prefix:
   - `crows_pairs_english*` → `pct_stereotype,none`
   - `bbq*` → `acc,none`
   - `stereoset*` → `ss,none`
   - `custom_probe*` → `acc,none`
3. Derives a `category` label (`crows_pairs_english_gender` → `gender`)
4. z-standardises the raw score within each `benchmark_family` so that cross-family aggregation is meaningful
5. Writes `results/harmonised.parquet` (for `agreement.py` / `divergence.py`) and `results/harmonised.csv` (for spreadsheet inspection)
6. Prints a preview `model × category` matrix so the user can eyeball the data immediately

**Why designed this way:** the analysis pipeline downstream needs a tidy long-form table. Deriving it from the raw JSONs (rather than from a bespoke intermediate format) means the same script works whether one, four, or twenty models have been evaluated.

**When to run:** first step of `analyse.sh`, or manually with `python analysis/harmonise.py` from the repo root.

---

### `analysis/agreement.py`

**Purpose:** pairwise cross-model rank agreement on the 9-category CrowS-Pairs vector.

**How it works:**
1. Loads `harmonised.parquet`; filters to `benchmark_family == crows_pairs` and drops any `overall` row
2. Pivots to `model × category` matrix
3. For each pair of models, computes Spearman ρ and Kendall τ over the 9-category vector
4. Bootstraps 95% confidence intervals for each statistic with 10,000 resamples (fixed seed 42)
5. Writes `results/agreement.csv` and prints the pairwise table

**Why Spearman and Kendall:** both are rank-based (invariant to the raw pct_stereotype scale) but Kendall τ is less sensitive to a single extreme value. Reporting both makes it hard for a reader to be fooled by one metric's edge case.

**When to run:** after `harmonise.py`. Second step of `analyse.sh`.

---

### `analysis/divergence.py`

**Purpose:** between-model spread per demographic category.

**How it works:**
1. Loads `harmonised.parquet`; filters to CrowS-Pairs categories
2. For each category, computes `mean`, `std`, `min`, `max`, `range = max - min` across the model panel
3. Sorts by `range` descending
4. Prints an aggregate summary: overall mean, overall std, widest between-model spread, narrowest

**Why designed this way:** RQ2 in the dissertation asks *which* categories differ most between models. `range` is the most direct answer; `std` supplements it as a distribution-aware alternative.

**When to run:** third step of `analyse.sh`.

---

### `analysis/report.py`

**Purpose:** generate the two figures and three summary CSVs the dissertation uses.

**How it works:**
1. Loads `harmonised.parquet`, filters to CrowS-Pairs categories
2. Shortens model names (`microsoft/Phi-3-mini-4k-instruct` → `Phi-3-mini-4k-instruct`) for figure readability
3. Writes three tables:
   - `results/tables/model_x_category.csv` — the full matrix
   - `results/tables/model_summary.csv` — one row per model (mean/std/min/max/range across categories)
   - `results/tables/category_summary.csv` — one row per category (mean/std/min/max/range across models)
4. Draws two figures:
   - `results/figures/model_x_category_heatmap.png` — 4 rows × 9 columns; RdBu_r colormap centred on 0.5; annotated cells
   - `results/figures/per_category_ranking.png` — grouped bar chart, 9 categories on the x-axis, one bar per model, dashed line at 0.5

**Why RdBu_r centred on 0.5:** anything above 0.5 is stereotype-preferring (red); anything below is anti-stereotype (blue). All 36 cells in the dissertation-phase run are red, which is itself an interpretable finding.

**When to run:** last step of `analyse.sh`.

---

### `scripts/analyse.sh`

**Purpose:** one-shot wrapper that pulls results and runs the four analysis scripts in order.

**How it works:**
1. `git pull` to sync results from the cloud run
2. `source .venv/bin/activate`
3. Runs `harmonise.py`, `agreement.py`, `divergence.py`, `report.py` in sequence
4. Prints a summary of output paths

**When to run:** once, on the researcher's Mac (or any host with the analysis-only stack installed), after the cloud run finishes.

---

## Custom task definitions

### `custom_tasks/probe/`

The Objective O6 custom probe. `task.yaml` registers `custom_probe` as a task; `items.jsonl` is the item file (starts with 5 seed items covering UK-specific SES, region and accent stereotypes, expandable to 50-150 for the full study). Uses the CrowS-Pairs likelihood-comparison protocol.

### `custom_tasks/bbq/`

BBQ full (11 categories, ~58,000 items) as a custom `lm-evaluation-harness` task. Template only — validate the `heegyu/bbq` dataset schema and duplicate `task_bbq_age.yaml` for each of the 10 remaining categories before running.

### `custom_tasks/stereoset/`

StereoSet (intra + inter, ~4,000 triples) as a custom task. Uses `utils.py` for the doc_to_choice helper functions that order the three candidate sentences (stereotype / anti-stereotype / unrelated). Template only — validate against `McGill-NLP/stereoset` before running.

Both `bbq/` and `stereoset/` are scheduled for the extended dissertation panel; they are not part of the current Chapter 4 results.

---

## CI / repo hygiene

### `.github/workflows/ci.yml`

Three jobs run on every push:
- **lint** — `ruff check` and `ruff format --check` on `analysis/` and `scripts/`
- **imports** — installs analysis-only deps and confirms `analysis.harmonise`, `.agreement`, `.divergence`, `.report` all import
- **syntax** — `bash -n` for every `.sh`, `yq` for every YAML, `jq` for every `.ipynb`

**Why designed this way:** cheap sanity checks that catch typos and formatting drift before they reach the researcher's Mac or the cloud pod.

---

## Files not covered above (intentionally minor)

- `scripts/push-to-github.sh` — one-shot bootstrap to push the repo to a fresh GitHub account. Only used once at repo creation.
- `scripts/determinism_check.sh` — runs the pilot twice and diffs the aggregate scores. Documents the reproducibility protocol in-repo.
- `notebooks/00-colab-pilot.ipynb` — the Colab-friendly pilot notebook (the one that produced the pilot data in Table 4.1 of the dissertation).
- `notebooks/01-*.ipynb`, `02-*.ipynb`, `03-*.ipynb` — placeholder analysis notebooks for the dissertation phase.
- `Dockerfile` — reproducible CUDA 12.1 + Python 3.11 image. Alternative to `runpod-quickstart.sh` for hosts that support Docker.
- `docker-compose.yml` — one-command wrapper for `docker build && docker run` with the three main commands (`pilot`, `full`, `analyse`).
- `Makefile` — friendly `make pilot`, `make full`, `make analyse`, `make status` targets.

---

## Cross-references

- The dissertation itself (`LD7236_PS_24034755Luis.docx`) points at the notebook via a permalink pinned to the git tag `submission-2026-07-17`.
- The dissertation-phase full-panel run is recorded in `docs/dissertation-run-log.md`.
- Reproducibility caveats (what is deterministic, what is not) live in `docs/reproducibility.md`.
- The methodology summary lives in `docs/methodology.md`.
- Troubleshooting for the most common failure modes lives in `docs/troubleshooting.md`.
