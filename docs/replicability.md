# One-command replication

## Zero setup (30 minutes, free)

Open the Colab notebook: **`notebooks/00-colab-pilot.ipynb`** (click the "Open in Colab" badge in the README). Runtime -> Run all. This runs the pilot on a free T4 GPU without you installing anything.

## Local, one command (30 minutes)

Requires: macOS Apple Silicon **or** Linux with an NVIDIA GPU, plus `git` and `make`.

```bash
git clone https://github.com/<your-handle>/llm-bias-audit.git
cd llm-bias-audit
make install                        # auto-detects OS, installs pinned deps
huggingface-cli login               # paste your HF token
make pilot                          # smoke test (~15 min)
```

## Cloud, one command (12 hours, $10)

```bash
# On any Linux/CUDA cloud instance (RunPod, vast.ai, Lambda, etc.)
git clone https://github.com/<your-handle>/llm-bias-audit.git
cd llm-bias-audit
make install
huggingface-cli login
make full                           # resumable full panel (~8-12 hrs)
make analyse                        # stats
make report                         # figures + tables
```

`make full` is **safe to interrupt** - a spot-instance kill or Ctrl-C only costs you the incomplete model; rerunning picks up where it left off.

## Docker (any host, exact bit-for-bit environment)

Requires: Docker + NVIDIA Container Toolkit on the host.

```bash
git clone https://github.com/<your-handle>/llm-bias-audit.git
cd llm-bias-audit
make docker-build                   # build the pinned image locally
make docker-full                    # run the full panel inside Docker
make analyse
make report
```

Or with docker-compose:

```bash
docker compose up pilot             # smoke test
docker compose up full              # full panel
docker compose up analyse           # stats
```

## Verifying reproducibility

Run the same benchmark twice and diff the scores:

```bash
make determinism-check
```

Passes if aggregate scores match to within `1e-9`. Any mismatch is documented in `docs/reproducibility.md` under "What is NOT bit-deterministic".

## Checking progress

```bash
make status
```

Shows which models have finished, which are pending, and the current CodeCarbon reading.
