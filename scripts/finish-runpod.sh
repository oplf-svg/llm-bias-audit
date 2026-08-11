#!/usr/bin/env bash
# Sanity-check that the full-panel run finished and everything landed on GitHub.
# Run this on the pod once the last `== DONE` line appears in run.log.

set -euo pipefail
cd /workspace/llm-bias-audit

echo "==> 1/4  Model completion check"
n=$(ls results/full/*/.completed 2>/dev/null | wc -l)
echo "     ${n} of 4 models have a .completed sentinel"
if [[ "$n" -lt 4 ]]; then
  echo "     Run is not finished yet. Check: tail -f run.log"
  exit 1
fi

echo ""
echo "==> 2/4  Timing summary from run.log"
grep -E "^== (RUN|DONE)" run.log

echo ""
echo "==> 3/4  Final push of any pending results"
git add -f results/
if ! git diff --cached --quiet; then
  git commit -m "Final push: full panel complete $(date -u +%FT%TZ)"
  git push
else
  echo "     Nothing new to push (autopush already handled it)"
fi

echo ""
echo "==> 4/4  Autopush process status"
ps aux | grep -v grep | grep autopush.sh || echo "     autopush not running (OK now that we have done the final push)"

echo ""
echo "==> Done. All 4 models complete and pushed to GitHub."
echo "    Next: on your Mac,  bash scripts/analyse.sh"
echo "    Then: RunPod dashboard -> Stop this pod"
