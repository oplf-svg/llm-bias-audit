# Reproducing the llm-bias-audit study

Every result in Chapter 4 of the dissertation was produced by the scripts in this repository. This document walks through the reproduction process end-to-end, explains what each script does, and points at deeper documentation where relevant.

For a script-by-script code reference see `docs/code-explained.md`. For the actual run's timings, hardware and issues see `docs/dissertation-run-log.md`. For the reproducibility caveats (what is deterministic, what is not) see `docs/reproducibility.md`.

---

## What the study runs

- **Models (4):** `microsoft/Phi-3-mini-4k-instruct`, `mistralai/Mistral-7B-v0.3`, `Qwen/Qwen2-7B`, `meta-llama/Meta-Llama-3.1-8B`
- **Benchmark:** CrowS-Pairs English, all 9 per-category subtasks (`crows_pairs_english_age`, `..._disability`, `..._gender`, `..._nationality`, `..._physical_appearance`, `..._race_color`, `..._religion`, `..._sexual_orientation`, `..._socioeconomic`)
- **Framework:** EleutherAI `lm-evaluation-harness v0.4.7`
- **Precision:** fp16 on H100 (primary results in Table 4.2), 4-bit NF4 via bitsandbytes on RTX 4090 / A40 (pilot results in Table 4.1)
- **Seed:** 42 (numpy, torch, few-shot sampler)
- **Decoding:** greedy
- **Output:** one JSON per (model, benchmark) pair, plus per-sample JSONL logs

Full benchmark and analysis parameters are pinned in `configs/models.yaml`, `configs/benchmarks.yaml`, and `configs/analysis.yaml`.

## Environment

Every dependency is pinned in `requirements.txt` (Linux/CUDA) or `requirements-mac.txt` (Apple Silicon). The full pinned list is authoritative — do not upgrade individual packages without also updating the pin.

Key pins:
- `torch==2.4.0`
- `transformers==4.44.2`
- `accelerate==0.34.2`
- `bitsandbytes==0.43.3` (Linux only)
- `datasets==2.20.0`
- `lm-eval @ v0.4.7`
- `hf_transfer==0.1.8`
- `numpy==1.26.4`

`scripts/verify-env.py` confirms every version matches the pin. See `docs/code-explained.md#scriptsverify-envpy` for the exact comparison logic (it accepts CUDA build tags such as `2.4.0+cu121`).

---

## Reproducing the full panel (cloud GPU)

### Fastest, easiest — RunPod H100 PCIe (the path the dissertation used)

1. Deploy a RunPod pod:
   - Template: **PyTorch 2.8.0** (comes with CUDA 12.8)
   - GPU: **H100 PCIe 80 GB** ($3.29/hr)
   - **Volume disk: 150 GB** (60 GB filled during the dissertation run and had to be resized — see `docs/dissertation-run-log.md#issues-encountered-and-fixes-applied`)
   - Container disk: 20 GB (default)
2. Accept the Llama-3.1 licence at <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B> (instant approval).
3. Create two tokens:
   - GitHub PAT with **Contents: Read and write** on `llm-bias-audit`: <https://github.com/settings/personal-access-tokens>
   - HuggingFace **Read** token: <https://huggingface.co/settings/tokens>
4. In the pod's Web Terminal, paste the bootstrap:

   ```bash
   export GH_TOKEN=<paste your GitHub PAT>
   export HF_TOKEN=<paste your HuggingFace token>
   MODE=h100 bash -c "$(curl -sSL https://x-access-token:$GH_TOKEN@raw.githubusercontent.com/oplf-svg/llm-bias-audit/main/scripts/runpod-quickstart.sh)"
   ```

   This runs `scripts/runpod-quickstart.sh` on the pod. That script:
   - Clones the repo into `/workspace/llm-bias-audit`
   - Installs the pinned dependencies via `scripts/install-linux.sh`
   - Configures git identity for autopush
   - Logs into HuggingFace
   - Starts `scripts/autopush.sh` in the background — pushes any new `results/` file to GitHub every 5 minutes
   - Starts `scripts/run_all_h100.sh` in the background — evaluates all four models sequentially

5. Wait ~1.5-2 hours. Check progress with:

   ```bash
   grep -E "^== (RUN|DONE|SKIP)" /workspace/llm-bias-audit/run.log
   ```

6. When the last `== DONE` line appears, wrap up:

   ```bash
   bash /workspace/llm-bias-audit/scripts/finish-runpod.sh
   ```

   Then stop the pod on the RunPod dashboard.

### Alternative — RTX 4090 (cheaper, slower, 4-bit NF4)

Same procedure, but drop `MODE=h100` from the bootstrap:

```bash
export GH_TOKEN=<...>; export HF_TOKEN=<...>
bash -c "$(curl -sSL https://x-access-token:$GH_TOKEN@raw.githubusercontent.com/oplf-svg/llm-bias-audit/main/scripts/runpod-quickstart.sh)"
```

This runs `scripts/run_all.sh` instead of `run_all_h100.sh`, using 4-bit NF4 quantisation via bitsandbytes. Values will differ slightly from the fp16 primary results by the amount documented in the reproducibility caveat.

### Alternative — Docker on any Linux+NVIDIA host

```bash
git clone https://github.com/oplf-svg/llm-bias-audit.git
cd llm-bias-audit
docker build -t llm-bias-audit .
docker run --gpus all -v $(pwd):/work -w /work \
  -e HF_TOKEN=<your HF token> \
  llm-bias-audit bash scripts/run_all_h100.sh
```

The `Dockerfile` pins CUDA 12.1 + Python 3.11 and installs `requirements.txt`. See `docker-compose.yml` for pilot/full/analyse services.

### Alternative — bare-metal Linux+CUDA

```bash
git clone https://github.com/oplf-svg/llm-bias-audit.git
cd llm-bias-audit
bash scripts/install-linux.sh
source .venv/bin/activate
huggingface-cli login --token <your HF token>
bash scripts/run_all_h100.sh   # or run_all.sh for 4-bit
```

## Reproducing the pilot (Apple Silicon or Colab)

Cheaper way to sanity-check the pipeline without a GPU rental.

### On your Mac

```bash
git clone https://github.com/oplf-svg/llm-bias-audit.git
cd llm-bias-audit
bash scripts/install-mac.sh
source .venv/bin/activate
huggingface-cli login --token <your HF token>
bash scripts/pilot.sh
```

Auto-detects MPS, uses float16 (no bitsandbytes on Mac). Takes ~15-30 min depending on the M-series chip.

### On Colab (free tier, T4 GPU)

Open <https://colab.research.google.com/github/oplf-svg/llm-bias-audit/blob/submission-2026-07-17/notebooks/00-colab-pilot.ipynb> and Runtime → Run all. Same code, but sets `HF_HUB_ENABLE_HF_TRANSFER=0` to avoid the fast-download package issue. Produces the same results the pilot in Table 4.1 of the dissertation reports.

## Data flow — where things come from and go

```
                (LLM evaluation, ONE-TIME)                       (analysis, RE-RUNNABLE)

  ┌─────────────────────┐   push   ┌────────────────────┐   pull   ┌──────────────────┐
  │  RunPod H100 pod    │─────────►│  GitHub main       │─────────►│  Colab dashboard │
  │  (all GPU work)     │  every   │  results/full/     │  once    │  (one notebook)  │
  └─────────────────────┘  5 min   └────────────────────┘          └──────────────────┘
                                              ▲                                │
                                              │  push derived CSVs + PNGs      │
                                              └────────────────────────────────┘
                                                        (optional)
```

- **Raw JSONs** (`results/full/<model>/results_*.json`, per-sample JSONLs, `.completed` sentinels) come from the RunPod pod running `scripts/run_all_h100.sh`; `scripts/autopush.sh` commits them every 5 min.
- **Derived tables and figures** (`results/harmonised.parquet`, `results/agreement.csv`, `results/divergence.csv`, `results/tables/*.csv`, `results/figures/*.png`) come from `analysis/harmonise.py` → `agreement.py` → `divergence.py` → `report.py`, run either from your terminal or from the Colab dashboard notebook.
- **Everything lives on GitHub `main`** at all times — the two directions of arrow are separate: the pod only ever pushes raw JSON, the notebook (or terminal) only ever pushes derived analysis.

## Notebook: Colab dashboard

There is **one** notebook. It does not run any LLMs; it is a viewer plus optional publisher.

| Notebook | What it does | Where it runs | GPU needed? |
|---|---|---|---|
| `notebooks/results-dashboard.ipynb` | Pull raw JSONs from GitHub → run the four analysis scripts → display heatmap + bar chart inline → show summary tables → **(optional Section 6)** push derived CSVs and PNGs back to GitHub so future readers see everything without re-running anything. | Colab free tier or any Jupyter | **No** |

Open it directly:
<https://colab.research.google.com/github/oplf-svg/llm-bias-audit/blob/main/notebooks/results-dashboard.ipynb>

**Recommended path** if you want a Colab-only experience with results from a real H100:

1. Open the notebook (link above)
2. Run all cells 1–5 (installs deps, clones repo at tag `submission-2026-07-17`, runs analysis, shows figures + tables)
3. (Optional) Run Section 6 with a GitHub PAT to publish the derived analysis back to `main`

**No re-running the LLMs is required.** The raw JSONs from the dissertation run are already committed at tag `submission-2026-07-17`.

## Reproducing the analysis (locally)

Once results are on GitHub, pull them and run the analysis on any machine. Doesn't need a GPU — just pandas / numpy / scipy / matplotlib / seaborn / statsmodels / pyarrow.

```bash
cd llm-bias-audit
git pull
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy scipy matplotlib seaborn statsmodels pyarrow
python analysis/harmonise.py    # ingest raw JSONs -> wide table
python analysis/agreement.py    # pairwise cross-model rank agreement
python analysis/divergence.py   # per-category between-model spread
python analysis/report.py       # heatmap + bar chart + summary CSVs
open results/figures/model_x_category_heatmap.png
open results/figures/per_category_ranking.png
```

Or, the same in one command via `bash scripts/analyse.sh` (once the venv is set up).

Outputs:
- `results/harmonised.parquet`, `.csv` — the wide `(model, task, category, score)` table
- `results/agreement.csv` — six pairwise Spearman ρ + Kendall τ + 95% bootstrap CIs
- `results/divergence.csv` — nine per-category rows sorted by between-model range
- `results/tables/model_x_category.csv`, `model_summary.csv`, `category_summary.csv`
- `results/figures/model_x_category_heatmap.png` — Figure 4.1 in the dissertation
- `results/figures/per_category_ranking.png` — Figure 4.2 in the dissertation

## What actually happened during the dissertation run

The specific execution that produced the Chapter 4 numbers is recorded in `docs/dissertation-run-log.md` with:
- exact dates, timings, hardware and cost
- issues encountered mid-run (disk full, HF CLI auth quirk, hf_transfer missing, etc.)
- fixes applied and where they were committed
- the git commit SHA at which each result was produced

That log is the ground-truth history. Read it before attempting to reproduce, so you know which issues to expect and which are already fixed in the current `main`.

## Handling the disk-space issue (if you use a smaller volume)

The dissertation-phase run started with a 60 GB volume, hit the limit mid-way through, resized to 94 GB, and continued. If you also use a small volume:

- **Cleanest:** provision ≥150 GB from the start
- **Alternative:** run `bash scripts/cleanup-cache.sh <finished-model>` after each `== DONE` line to delete that model's cached weights (results are preserved in `results/full/<model>/`):

   ```bash
   bash scripts/cleanup-cache.sh microsoft/Phi-3-mini-4k-instruct
   bash scripts/cleanup-cache.sh mistralai/Mistral-7B-v0.3
   bash scripts/cleanup-cache.sh Qwen/Qwen2-7B
   ```

## Reproducibility caveats

See `docs/reproducibility.md` for the full statement. The short version:

- **Deterministic across runs on the same GPU class:** results agree bit-exactly for CrowS-Pairs on Phi-3-mini across 6 independent Colab runs (validated).
- **Not deterministic across GPU classes:** fp16 numbers on H100 differ from 4-bit NF4 numbers on T4 by ~3-4 percentage points on the same model + benchmark. This is expected and is why the primary results use fp16 and report the 4-bit pilot values separately.
- **`bitsandbytes` 4-bit quantisation is not bit-deterministic** across CUDA versions or NVIDIA architectures. Reproducers who use 4-bit will see small per-item variance.

## Cross-references

- Full bibliography with clickable DOIs: `REFERENCES.md`
- Script-by-script code reference: `docs/code-explained.md`
- Dissertation run history: `docs/dissertation-run-log.md`
- Reproducibility statement: `docs/reproducibility.md`
- Replication risks (what is outside our control, and future-proofing): `docs/replication-risks.md`
- Methodology summary: `docs/methodology.md`
- Common failure modes and fixes: `docs/troubleshooting.md`
- The dissertation itself: `LD7236_PS_24034755Luis.docx` (not in the code repo)

Every URL referenced by the dissertation is pinned to the immutable git tag `submission-2026-07-17`, so future pushes to `main` do not change what an examiner sees when they click a link.
