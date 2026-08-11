# Makefile — one-command interface to every workflow.
# Run `make` (or `make help`) for the menu.

.PHONY: help install install-mac install-linux verify \
        pilot full full-h100 analyse report \
        finish cleanup status determinism-check \
        docker-build docker-pilot docker-full \
        lock clean push-repo colab-link

OS := $(shell uname -s)

help:                     ## List every make target
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ------------- SETUP -------------
install:                  ## Auto-detect OS and install everything
ifeq ($(OS),Darwin)
	@$(MAKE) install-mac
else
	@$(MAKE) install-linux
endif

install-mac:              ## Install on macOS (Apple Silicon)
	bash scripts/install-mac.sh

install-linux:            ## Install on Linux (CUDA)
	bash scripts/install-linux.sh

verify:                   ## Check installed versions match the pins
	.venv/bin/python scripts/verify-env.py

lock:                     ## Regenerate lock files (uv pip compile)
	uv pip compile requirements.txt --output-file requirements.lock
	uv pip compile requirements-mac.txt --output-file requirements-mac.lock

# ------------- PILOT -------------
pilot:                    ## Single-model pilot (Phi-3-mini x CrowS-Pairs, ~15 min)
	bash scripts/pilot.sh

# ------------- FULL PANEL -------------
full:                     ## Full panel (Linux/CUDA, 4-bit NF4, RTX 4090 / A40 / older GPUs)
	bash scripts/run_all.sh

full-h100:                ## Full panel (Linux/CUDA, fp16, H100+ GPUs)
	bash scripts/run_all_h100.sh

# ------------- WRAP-UP -------------
finish:                   ## Verify full-panel completion + final push (on pod)
	bash scripts/finish-runpod.sh

cleanup:                  ## Free cached weights for a specific model (usage: make cleanup MODEL=<repo-id>)
	bash scripts/cleanup-cache.sh $(MODEL)

status:                   ## Show what has been run and what has not
	bash scripts/status.sh

# ------------- ANALYSIS (on your Mac / analysis machine) -------------
analyse:                  ## Run analysis pipeline (harmonise -> agreement -> divergence -> report)
	.venv/bin/python analysis/harmonise.py
	.venv/bin/python analysis/agreement.py
	.venv/bin/python analysis/divergence.py
	.venv/bin/python analysis/report.py

report:                   ## Regenerate figures and tables only (assumes harmonised.parquet exists)
	.venv/bin/python analysis/report.py

# ------------- REPRODUCIBILITY -------------
determinism-check:        ## Run pilot twice, diff aggregate scores (proves reproducibility)
	bash scripts/determinism_check.sh

# ------------- DOCKER -------------
docker-build:             ## Build the pinned CUDA Docker image
	docker build -t llm-bias-audit:latest .

docker-pilot:             ## Run pilot inside the Docker image
	docker run --gpus all -v $(PWD):/work -w /work llm-bias-audit:latest bash scripts/pilot.sh

docker-full:              ## Run the full panel inside the Docker image
	docker run --gpus all -v $(PWD):/work -w /work llm-bias-audit:latest bash scripts/run_all_h100.sh

# ------------- HOUSEKEEPING -------------
clean:                    ## Remove venv, caches, and results (keeps configs)
	rm -rf .venv __pycache__ */__pycache__ .cache
	@echo "Cleaned. Configs and code preserved."

push-repo:                ## First-time push to a new private GitHub repo
	@echo "Set your handle then run:"
	@echo "  gh repo create llm-bias-audit --private --source=. --push"

colab-link:               ## Print the Colab link for the pilot notebook
	@echo "https://colab.research.google.com/github/oplf-svg/llm-bias-audit/blob/submission-2026-07-17/notebooks/00-colab-pilot.ipynb"
