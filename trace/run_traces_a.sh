#!/usr/bin/env bash
# Workload A traces: one LongBench v2 item per domain with a 24K-64K-token context, seed 0,
# then one combined table in trace/results/SUMMARY_A.md.
# Usage (repo root, GPU VM):
#   bash trace/run_traces_a.sh                 # pick items, then collect each (~1.5 min per trace)
#   INDICES="12 40" bash trace/run_traces_a.sh # skip picking, collect these items
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/traces trace/results

python - <<'EOF' || { echo "Preflight failed: torch cannot see the GPU (see requirements.txt)."; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA is not available to torch")
EOF
others=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
if [ -n "$others" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "Other processes are using the GPU:"; echo "$others"
    echo "Re-run when the GPU is free, or set FORCE=1."; exit 1
fi

if [ -z "${INDICES:-}" ]; then
    python trace/pick_items.py | tee trace/results/workload_a_items.md || exit 1
    INDICES=$(cat trace/results/workload_a_indices.txt)
fi

for i in $INDICES; do
    echo "== LongBench v2 item $i =="
    if python trace/collect.py --index "$i" > "data/traces/run_lb$i.log" 2>&1; then
        grep -E "decode steps|^\| 128 \|" "data/traces/run_lb$i.log" | head -2
    else
        echo "item $i failed; last lines of data/traces/run_lb$i.log:"; tail -5 "data/traces/run_lb$i.log"
    fi
done

python trace/combine.py --prefix A_ | tee trace/results/SUMMARY_A.md
echo
echo "Done. Paste trace/results/SUMMARY_A.md (and workload_a_items.md) back."
