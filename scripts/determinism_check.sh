#!/usr/bin/env bash
# Run the pilot twice, diff the results.
# If the pipeline is deterministic (seed + greedy + same hw), scores match exactly.
# Any mismatch is investigated and reported in docs/reproducibility.md.

set -euo pipefail

# shellcheck disable=SC1091
source .venv/bin/activate

OUT_A="./results/determinism_check/run_a"
OUT_B="./results/determinism_check/run_b"
rm -rf "${OUT_A}" "${OUT_B}"
mkdir -p "${OUT_A}" "${OUT_B}"

MODEL="microsoft/Phi-3-mini-4k-instruct"
TASK="crows_pairs_english"

if [[ "$(uname -s)" == "Darwin" ]]; then
  MODEL_ARGS="pretrained=${MODEL},dtype=float16,device=mps"
else
  MODEL_ARGS="pretrained=${MODEL},load_in_4bit=True,bnb_4bit_quant_type=nf4"
fi

for OUT in "${OUT_A}" "${OUT_B}"; do
  echo "==> Run into ${OUT}"
  lm_eval \
    --model hf \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASK}" \
    --seed 42 \
    --batch_size 1 \
    --output_path "${OUT}" \
    --log_samples \
    --verbosity WARNING
done

echo ""
echo "==> Comparing aggregate scores..."
python - << 'PY'
import json, glob, sys

def load(root):
    files = sorted(glob.glob(f"{root}/**/results_*.json", recursive=True))
    return json.loads(open(files[0]).read())["results"]

a = load("./results/determinism_check/run_a")
b = load("./results/determinism_check/run_b")

diffs = []
for task, metrics in a.items():
    for m, v in metrics.items():
        if isinstance(v, (int, float)):
            v_b = b[task].get(m)
            if v_b is None: continue
            if abs(v - v_b) > 1e-9:
                diffs.append((task, m, v, v_b, abs(v - v_b)))

if diffs:
    print("MISMATCH between runs:")
    for task, m, va, vb, d in diffs:
        print(f"  {task} / {m}:  {va:.6f}  vs  {vb:.6f}   (delta={d:.2e})")
    print()
    print("This is expected on some hardware/CUDA combinations. See docs/reproducibility.md.")
    sys.exit(1)
else:
    print("OK -- all aggregate scores match to within 1e-9.")
PY
