# Replication risks: what is outside our control

This document catalogues everything that could break a future attempt to reproduce this study, and how to mitigate each risk. Written from the assumption that "future" means anywhere from 6 months to 5 years after the dissertation submission (tag `submission-2026-07-17`, dated August 2026).

Grouped by risk category. Highest-impact risks first.

---

## 1. External model weights (highest risk)

**What could break:**
- A model author republishes weights under the same `repo_id`. Our pinned commit hashes protect against this (see `configs/models.yaml`), but only if the specific commit is still fetchable — HuggingFace can and occasionally does purge old commits.
- A model is withdrawn entirely (`repo_id` returns 404). Meta, Mistral or Qwen could decide to unpublish. Precedent: Llama 1 was hard to obtain for years.
- Licence terms change. Llama-3.1 Community Licence has explicit revocation clauses; a future revision could restrict evaluation-only use.
- Gating expands. Currently Mistral, Qwen2 and Phi-3 are ungated; Llama-3.1 is gated. Any of the currently-open models could become gated.

**Mitigations already in the repo:**
- Every model pinned by `repo_id` + commit hash in `configs/models.yaml`
- `runpod-quickstart.sh` includes explicit HF token auth so gated access works
- `docs/troubleshooting.md` covers the "licence not accepted" error

**What a future replicator should do:**
- Before starting, check that each `repo_id` in `configs/models.yaml` still resolves at <https://huggingface.co/> — if not, look for a mirror
- Cache the model weights on a permanent, off-HuggingFace mirror (e.g. Zenodo, institutional storage) if you plan to reproduce more than 1 year from now
- Consider archiving the model weights alongside the results in an institutional repository (~30 GB for the four-model panel)

---

## 2. Benchmark datasets

**What could break:**
- `BigScienceBiasEval/crows_pairs_multilingual` (the CrowS-Pairs source used by the harness) could be moved, deleted, or restructured. The BigScience org has already had some organisational changes.
- The dataset uses `trust_remote_code=True` — future HuggingFace `datasets` versions may drop support for this pattern for security reasons.
- Individual items may be revised in response to Blodgett et al.'s (2021) "Stereotyping Norwegian Salmon" critique. Any revision changes the numbers.
- BBQ and StereoSet are not currently used (custom tasks in `custom_tasks/` are templates only), but the same issues apply when they get wired in.

**Mitigations already in the repo:**
- Dataset revision pinning documented as a design goal in `docs/reproducibility.md`
- `configs/benchmarks.yaml` records the dataset ID and version at time of study
- Custom task YAMLs point at specific dataset paths that can be substituted

**What a future replicator should do:**
- Mirror the CrowS-Pairs items JSON file (~1 MB) locally or in the project repo
- If the upstream schema changes, update `custom_tasks/*/task.yaml` to match the new field names
- Consider vendoring the dataset directly into `custom_tasks/crows_pairs/items.jsonl`

---

## 3. Evaluation framework (lm-evaluation-harness)

**What could break:**
- Task IDs get renamed. This happened between harness versions in the past — for example, the way BBQ tasks are registered has changed multiple times. `crows_pairs_english_<category>` may not exist under those names in a future release.
- The default scoring routine per task can change (Nadeem et al.'s original StereoSet metric has been tweaked in the harness at least twice).
- Model-loading arguments can change syntax (`load_in_4bit=True` was already deprecated in favour of `quantization_config` during this study).
- The harness itself could be replaced by a successor project.

**Mitigations already in the repo:**
- Harness pinned to `v0.4.7` (git tag) in `requirements.txt`
- `scripts/discover_tasks.sh` lets a replicator list the currently-registered tasks and surface any renames
- `docs/dissertation-run-log.md` records the exact deprecation warnings we saw

**What a future replicator should do:**
- Do not bump the `lm-eval` pin without first running `scripts/pilot.sh` and `scripts/determinism_check.sh` to confirm the numbers are unchanged
- If task IDs have changed, use `scripts/discover_tasks.sh` to find the new names and update `run_all.sh`
- Consider forking the harness at `v0.4.7` and pinning the fork instead of upstream

---

## 4. Compute infrastructure

**What could break:**
- RunPod stops offering H100 PCIe at $3.29/hr (very likely — GPU pricing shifts monthly)
- RunPod's Community Cloud tier disappears or is renamed
- Colab free tier removes T4 GPU access
- NVIDIA driver + CUDA version combinations diverge and the pinned torch 2.4.0+cu121 no longer runs on new drivers
- The PyTorch 2.8.0 template on RunPod is renamed or updated in a way that breaks our `install-linux.sh` assumptions

**Mitigations already in the repo:**
- `REPRODUCE.md` documents multiple provider paths (RunPod, vast.ai, Lambda, Docker anywhere)
- `Dockerfile` pins CUDA 12.1 + Python 3.11 so any Docker+GPU host works
- `docs/dissertation-run-log.md` records exact hardware (H100 PCIe, driver 580.126.09)

**What a future replicator should do:**
- Prefer the Docker path (`docker build && docker run --gpus all ...`) over cloud-specific templates
- If a specific provider is unavailable, any Linux host with NVIDIA driver + Docker + ≥1x GPU with ≥16 GB VRAM works
- Note the pod-image constraint: some cloud templates now use CUDA 13+ which forces torch 2.5+; if this happens, upgrade the pinned torch and re-run `scripts/determinism_check.sh`

---

## 5. HuggingFace platform

**What could break:**
- Token format changes (already happened once — classic tokens vs fine-grained PATs)
- `huggingface-cli login` CLI syntax changes (we hit `--add-to-git-credential` becoming a switch rather than a value during this study)
- Rate limits tighten — currently generous, but a policy change could break `hf_transfer` bulk downloads
- The gated-model access flow changes (currently a click-through; could become an application form)
- `hf_transfer` package could be deprecated or renamed

**Mitigations already in the repo:**
- `scripts/runpod-quickstart.sh` uses the `--token` argument directly, avoiding interactive login
- `HF_HUB_ENABLE_HF_TRANSFER=0` fallback documented for hosts without `hf_transfer`
- `docs/troubleshooting.md` covers the common auth errors

**What a future replicator should do:**
- Generate a fresh HF token before starting; don't reuse tokens older than a few months
- If `hf_transfer` fails to install, set `HF_HUB_ENABLE_HF_TRANSFER=0` and use the standard downloader (slower but reliable)
- Watch out for stricter licence-acceptance requirements on gated models

---

## 6. Python ecosystem

**What could break:**
- Individual packages get yanked from PyPI (rare but happens)
- New Python versions break older packages (we hit `pyarrow==17.0.0` not building on Python 3.13)
- Transitive dependencies become incompatible
- The pinned `numpy==1.26.4` will eventually be too old to install alongside newer packages
- Wheel availability shifts (a wheel for `bitsandbytes==0.43.3` on `cuda-13.x` may never exist)

**Mitigations already in the repo:**
- All top-level deps pinned in `requirements.txt` / `requirements-mac.txt`
- `Dockerfile` freezes the whole install pipeline
- Analysis-only deps (`pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `statsmodels`, `pyarrow`) also documented separately in `REPRODUCE.md` for GPU-free replicators

**What a future replicator should do:**
- Use the `Dockerfile` for the evaluation portion of the study — this insulates against most Python ecosystem drift
- If installing directly, use the same Python version we used (3.11 for Linux, 3.11+ for macOS)
- If a specific package fails to install, drop its pin and let pip resolve a compatible version — then re-run `scripts/determinism_check.sh` to confirm the numbers still match

---

## 7. Access and credentials

**What is outside our control (and always will be):**
- Your GitHub Personal Access Token expires (30/60/90 days depending on your setting)
- Your HuggingFace token doesn't expire by default but can be revoked
- Your acceptance of the Llama-3.1 Community Licence is per-account and per-token — a replicator needs their own signed licence
- Your GitHub account could be suspended or the repo made private (currently is)
- RunPod / Colab / your cloud provider account could be closed

**Mitigations already in the repo:**
- All tokens are passed via environment variables — none are committed
- `AUTHOR.local.md` is gitignored so personal contact info doesn't leak
- The repo works fine with tokens from any GitHub / HF account, not just the study author's

**What a future replicator should do:**
- Generate fresh tokens at study start
- Accept the Llama-3.1 licence at <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B> under your own HF account
- Use SSH keys instead of PATs for git operations where possible (deploy keys work well for CI-style setups)

---

## 8. Known determinism gaps (already documented, but flagged here for completeness)

**Documented in `docs/reproducibility.md`:**
- `bitsandbytes` 4-bit quantisation is not bit-deterministic across CUDA versions
- PyTorch floating-point ops differ across NVIDIA architectures (Ampere ↔ Hopper)
- HuggingFace tokeniser updates can silently change encoding — pin the tokeniser separately if this matters

**Empirically observed during this study:**
- fp16 on H100 gives ~3-4 percentage-point differences from 4-bit NF4 on T4 for the same model + benchmark. The primary Chapter 4 results use fp16 to avoid this class of variance.

---

## 9. Cost drift (not about correctness, about affordability)

- The £2 spent on the H100 run in August 2026 is not a stable estimate — cloud GPU pricing can double or halve within a year
- Free tiers (Colab, Kaggle) can disappear or add restrictions
- An independent researcher may need to budget £10-50 for a re-run, especially if provider pricing has increased

**Mitigation:** the study is small enough (~1.5 h on one H100) that even at 5× the current cost it remains under £10.

---

## 10. Regulatory / research-ethics drift

**What could change:**
- The EU AI Act (Regulation 2024/1689) enters application in phases through 2025-2027; specific Article 53 guidance is not yet finalised and could evolve after the dissertation is submitted
- The UK's AI regulation framework (HM Government, 2023) is a white paper, not statute; the actual regulatory instruments will land later
- Any interpretation in Chapter 5 of the dissertation should be understood as reflecting the regulatory state at time of writing

**Mitigation:** the dissertation explicitly cites the version of each regulatory document; interpretations will need updating if the underlying regulation changes.

---

## 11. What is genuinely permanent

- The raw `results_*.json` files pushed to GitHub before the pod was stopped are permanent (git history) — future replicators can always re-analyse the same primary output even if none of the above can be re-run
- The `docs/dissertation-run-log.md` is a permanent record of the exact hardware, timings and issues
- The immutable git tag `submission-2026-07-17` on GitHub freezes the entire code + docs + results state at submission time
- Every result cited in the dissertation is traceable to a specific `results_*.json` file at a specific git commit

---

## Priority actions for maximum future-proofing

If you want to make this study replicable 5+ years from now, in order of impact:

1. **Archive the model weights.** ~30 GB total for the four-model panel. Upload to Zenodo or your university's data repository. HuggingFace `repo_id` links will not survive indefinitely.
2. **Vendor the CrowS-Pairs items JSON directly into the repo.** ~1 MB. Removes the HuggingFace-dataset-hosting dependency entirely.
3. **Publish a Docker image on Docker Hub (or GHCR).** A frozen image that already has all pinned deps installed. `docker pull` beats `pip install` for reproducibility.
4. **Cross-post to a permanent archive.** Zenodo issues a DOI for a repository snapshot. That DOI outlasts GitHub.
5. **Multi-seed replication.** Run one model three times with different seeds and publish the between-seed variance — turns "should be reproducible" into "we measured how reproducible it is".

Every one of these is a single half-day of work. All are optional but each raises the study's shelf life.
