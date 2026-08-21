# Extensions staged for later pod sessions

Everything below is already committed to the repo and ready to kick off on any
fresh RunPod pod. Each extension is independent; run any/all in any order.

## 0. Reconnect and bootstrap

Same as the dissertation run:

```
export GH_TOKEN=<PAT>
export HF_TOKEN=<HF token>
bash -c "$(curl -sSL https://x-access-token:$GH_TOKEN@raw.githubusercontent.com/oplf-svg/llm-bias-audit/main/scripts/runpod-quickstart.sh)"
```

Then install the missing BBQ task YAMLs (not shipped by pip):

```
cd /tmp && git clone --depth=1 https://github.com/EleutherAI/lm-evaluation-harness lm-eval-main
cp -r lm-eval-main/lm_eval/tasks/bbq /workspace/llm-bias-audit/.venv/lib/python3.11/site-packages/lm_eval/tasks/
```

Then start autopush:

```
cd /workspace/llm-bias-audit
nohup bash scripts/autopush.sh > autopush.log 2>&1 &
```

## 1. Add a 5th model to the panel (Objective O2 upper bound)

Runs both CrowS-Pairs + BBQ for the specified model. ~1.5–2 h on H100 SXM,
~£5. Repeat for other models if desired.

```
tmux new -d -s work "cd /workspace/llm-bias-audit && source .venv/bin/activate && MODEL=google/gemma-2-9b HF_HUB_ENABLE_HF_TRANSFER=0 bash scripts/run_extra_model.sh > run.log 2>&1"
tmux ls
```

Swap `MODEL=` for any of:
- `google/gemma-2-9b` — different model family (recommended)
- `Qwen/Qwen2.5-7B` — direct comparison to the existing Qwen2
- `microsoft/Phi-3.5-mini-instruct` — successor to Phi-3
- `mistralai/Mistral-Nemo-Base-2407` — larger (12B)

## 2. Run the O6 custom UK-specific probe

Currently 20 items in `custom_probe/probe_items.jsonl` (target: 50-150).
Add more items to that JSONL, commit + push, then on the pod:

```
tmux new -d -s work "cd /workspace/llm-bias-audit && source .venv/bin/activate && HF_HUB_ENABLE_HF_TRANSFER=0 MODE=all bash scripts/run_extensions_h100.sh > run.log 2>&1"
tmux ls
```

The `MODE=all` branch of `run_extensions_h100.sh` picks up `custom_probe/`
automatically if the directory exists. Runs on all four (or five) panel
models. ~5 min total on H100 SXM, ~£0.25.

## 3. Reproducibility check (fp16 on new GPU vs original)

Re-runs Phi-3 × CrowS-Pairs on the current pod and prints a per-category diff
against the original dissertation-phase results. Should be within ~0.001
per-category. Fills the empirical claim in Ch 3 §Reproducibility Protocol.

```
tmux new -d -s work "cd /workspace/llm-bias-audit && source .venv/bin/activate && bash scripts/reproducibility_check.sh > run.log 2>&1"
```

~5 min, ~£0.30. Runs the suggested Python diff at the end — copy its output
into the dissertation as evidence.

## After any extension: pull + analyse locally

```
cd ~/Desktop/ProfessionalPracticeinComputingAssesstment/llm-bias-audit
git pull
bash scripts/analyse.sh
```

The existing `analysis/harmonise.py` handles the new benchmark family
automatically (BBQ metric is already in `TASK_PREFIX_TO_METRIC`). The
`custom_probe` and any new model rows appear in `results/harmonised.parquet`,
`agreement.csv` and `divergence.csv` without code changes.

## Housekeeping

- Free disk between models: `bash scripts/cleanup-cache.sh <finished-model>`
- Push results manually if autopush dies:
  ```
  git add -f $(find results/full/<model_slug> -name 'results_*.json' -o -name '.completed')
  git commit -m "<model> extension results"
  git push origin main
  ```
- `.gitignore` already excludes `samples_bbq_*.jsonl` (>100 MB rejected by
  GitHub); no action needed.
