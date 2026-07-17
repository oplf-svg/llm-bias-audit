#!/usr/bin/env bash
# Discover which bias-benchmark tasks are actually registered in the installed
# lm-eval-harness. Use this to figure out the exact task names before adding
# them to pilot.sh or run_all.sh.

set -euo pipefail

# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

echo "==> lm-eval version"
python -c "import lm_eval; print(getattr(lm_eval, '__version__', 'unknown'))"

echo ""
echo "==> Tasks matching 'bbq', 'stereo' or 'crows':"
python - << 'PY'
from lm_eval import tasks
tm = tasks.TaskManager()
avail = sorted(tm.all_tasks)
hits = [t for t in avail if any(k in t.lower() for k in ('bbq', 'stereo', 'crows'))]
for t in hits:
    print(f"  {t}")
if not hits:
    print("  (no matches - benchmark may need to be added as a custom task)")
PY

echo ""
echo "==> All registered task groups:"
python -c "from lm_eval import tasks; tm = tasks.TaskManager(); [print(f'  {g}') for g in sorted(tm.task_group_table.get_group_names())]" 2>/dev/null || \
python -c "from lm_eval import tasks; tm = tasks.TaskManager(); print([g for g in dir(tm) if 'group' in g.lower()])"
