#!/usr/bin/env bash
# Background: every 5 minutes, add-commit-push any new files in results/.
# Runs until killed. Use nohup so it survives SSH disconnect.
#
# Requires:
#   - git remote 'origin' is set to a URL with token embedded, e.g.
#     https://x-access-token:$GH_TOKEN@github.com/oplf-svg/llm-bias-audit.git
#   - git user.name and user.email are configured

set -euo pipefail

INTERVAL=${INTERVAL:-300}
echo "autopush started, interval=${INTERVAL}s"

while true; do
  sleep "${INTERVAL}"
  # Pull latest to reduce merge conflicts (in case another pod is also pushing)
  git pull --rebase --autostash origin main >/dev/null 2>&1 || true
  git add -f results/
  if ! git diff --cached --quiet; then
    msg="Auto-push results $(date -u +%FT%TZ)"
    git commit -m "$msg" && echo "  pushed: $msg" && git push origin main
  fi
done
