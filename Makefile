# Makefile - one-command interface to every workflow.
# Run `make` (or `make help`) for the menu.

.PHONY: help install install-mac install-linux verify pilot full analyse report \
        clean docker-build docker-pilot docker-full colab-link push-repo lock \
        determinism-check status

# Auto-detect OS for the default `install` target
OS := $(shell uname -s)

help:                     ## List every make target
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

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

pilot:                    ## Run pilot (Phi-3-mini x CrowS-Pairs, ~15 min)
	bash scripts/pilot.sh

full:                     ## Run full panel (Linux/CUDA only, ~8-12 GPU-hrs)
	bash scripts/run_all.sh

analyse:                  ## Run stats pipeline (Spearman, Wilcoxon, effect sizes)
	.venv/bin/python analysis/harmonise.py
	.venv/bin/python analysis/agreement.py
	.venv/bin/python analysis/divergence.py

report:                   ## Generate summary tables + figures
	.venv/bin/python analysis/report.py

lock:                     ## Regenerate uv.lock file for exact reproducibility
	uv pip compile requirements.txt --output-file requirements.lock
	uv pip compile requirements-mac.txt --output-file requirements-mac.lock

docker-build:             ## Build the pinned CUDA Docker image locally
	docker build -t llm-bias-audit:latest .

docker-pilot:             ## Run pilot inside the Docker image
	docker run --gpus all -v $(PWD):/work -w /work llm-bias-audit:latest bash scripts/pilot.sh

docker-full:              ## Run the full panel inside the Docker image
	docker run --gpus all -v $(PWD):/work -w /work llm-bias-audit:latest bash scripts/run_all.sh

determinism-check:        ## Run pilot twice, diff the results (proves reproducibility)
	@bash scripts/determinism_check.sh

status:                   ## Show what has been run and what has not
	@bash scripts/status.sh

clean:                    ## Remove venv, caches, and results (keeps configs)
	rm -rf .venv __pycache__ */__pycache__ .cache
	@echo "Cleaned. Configs and code preserved."

push-repo:                ## First-time push to a new private GitHub repo
	@echo "Set your handle then run:"
	@echo "  gh repo create llm-bias-audit --private --source=. --push"

colab-link:               ## Print the Colab link for the pilot notebook
	@echo "https://colab.research.google.com/github/<your-handle>/llm-bias-audit/blob/main/notebooks/00-colab-pilot.ipynb"
