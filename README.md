# llm-bias-audit

Comparative evaluation of three social-bias benchmarks (BBQ, StereoSet, CrowS-Pairs) plus a small hand-built custom probe, applied to a panel of open-source large language models.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-handle>/llm-bias-audit/blob/main/notebooks/00-colab-pilot.ipynb)

**Author:** L.F.O.P. (…4755)
**Supervisor:** P.S.
**Module:** LD7236 Professional Practice in Computing and Digital Technologies Project
**Institution:** Northumbria University
**Programme:** MSc Big Data and Data Science Technology with Advanced Practice

## Aim

To compare how three widely-used social-bias benchmarks (BBQ, StereoSet, CrowS-Pairs) characterise bias in a sample of open-source LLMs, and to measure how far they agree about model fairness.

## Quick start (one command)

```bash
git clone https://github.com/<your-handle>/llm-bias-audit.git
cd llm-bias-audit
make install                        # auto-detects OS
huggingface-cli login
make pilot                          # smoke test (~15 min)
make full                           # Linux/CUDA only (~8-12 hrs)
make analyse                        # stats
make report                         # figures + tables
```

Run `make` alone for the full menu of targets.

## Model panel

| Model | Params | HF repo | Notes |
|---|---|---|---|
| Phi-3-mini-4k-instruct | 3.8B | `microsoft/Phi-3-mini-4k-instruct` | Pilot target, smallest |
| Mistral-7B-v0.3 | 7B | `mistralai/Mistral-7B-v0.3` | Open licence |
| Qwen2-7B | 7B | `Qwen/Qwen2-7B` | Open licence |
| Llama-3.1-8B | 8B | `meta-llama/Meta-Llama-3.1-8B` | Gated, needs licence acceptance |
| One 2025-2026 release | TBD | TBD | To be selected during pilot |

## Benchmarks

- **BBQ** (Parrish et al., 2022) - QA bias in ambiguous vs. disambiguated contexts
- **StereoSet** (Nadeem et al., 2021) - stereotype vs. anti-stereotype vs. unrelated likelihood
- **CrowS-Pairs** (Nangia et al., 2020) - pair-preference over stereotypical sentences
- **Custom probe (O6)** - hand-built 50-150 items targeting a documented gap; run as sensitivity check

## Ease of use

- **`make install`** picks the right requirements file based on your OS.
- **`make pilot`** runs a 15-minute smoke test that exercises the whole pipeline.
- **`make full`** is **safe to interrupt** - a spot-instance kill or Ctrl-C only costs you the incomplete model; rerunning picks up where it left off (see `results/full/<model>/.completed` sentinels).
- **`make status`** shows which models have finished and current carbon usage.
- **`make determinism-check`** verifies reproducibility by running the pilot twice and diffing scores.
- **Colab pilot notebook** requires zero installation.

## Replicability

Every knob is pinned:

- Dataset revisions (see `configs/benchmarks.yaml`)
- Model commit hashes (see `configs/models.yaml`)
- `lm-evaluation-harness` at `v0.4.7`
- Random seed `42`, greedy decoding, 4-bit NF4 quantisation, single GPU class
- Python dependencies pinned in `requirements.txt` / `requirements-mac.txt`
- Full Docker image in `Dockerfile` for identical environment across hosts
- `make lock` regenerates `requirements.lock` with fully-resolved transitive versions

Honest limits are documented in `docs/reproducibility.md`.

## Repository layout

```
llm-bias-audit/
|-- Makefile                 <- one-command interface (run `make`)
|-- README.md
|-- LICENSE
|-- requirements.txt         (Linux/CUDA production)
|-- requirements-mac.txt     (macOS dev + pilot)
|-- Dockerfile               (reproducible CUDA image)
|-- docker-compose.yml       (one-command Docker runs)
|-- .gitignore
|-- configs/                 (models, benchmarks, analysis - all YAML)
|-- scripts/                 (install, pilot, run_all, status, determinism_check)
|-- custom_probe/            (Objective O6: hand-built probe + lm-eval task.yaml)
|-- analysis/                (harmonise, agreement, divergence, report)
|-- notebooks/               (pilot inspection, agreement matrix, category divergence)
|-- results/                 (raw JSONL, harmonised.parquet, figures)
|-- docs/                    (methodology, reproducibility, replicability, troubleshooting)
\-- .github/workflows/       (lint + import CI)
```

## Cost estimate

**M4 MacBook (pilot only):** free  
**Google Colab Pro (pilot + light full runs):** ~£10/month  
**RunPod / vast.ai spot RTX 4090 (full panel):** $5-15 total for the whole study  
**Modal / Lambda Labs on-demand:** $22-45 total

## Licence

Analysis code: MIT (see `LICENSE`).
Benchmark datasets retain their original licences (BBQ CC-BY-SA 4.0, StereoSet MIT, CrowS-Pairs academic use).
