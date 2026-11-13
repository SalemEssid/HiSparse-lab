#!/usr/bin/env bash
# Phase 2 (G2): self-tests, then the first Workload A trace with Quest coverage.
# Usage (repo root, GPU VM):
#   bash trace/run_g2.sh               # LongBench v2 item 0, 32K prompt, up to 1000 decode steps
#   INDEX=7 bash trace/run_g2.sh       # another item
#   bash trace/run_g2.sh --steps 200   # extra flags go to collect.py
set -uo pipefail
cd "$(dirname "$0")/.."
INDEX="${INDEX:-0}"
mkdir -p data/traces trace/results

python - <<'EOF' || { echo "Preflight failed: torch cannot see the GPU (see requirements.txt)."; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA is not available to torch")
print(f"torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}")
EOF

others=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
if [ -n "$others" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "Other processes are using the GPU:"
    echo "$others"
    echo "A 32K trace needs ~16 GiB of GPU memory. Re-run when the GPU is free, or set FORCE=1."
    exit 1
fi

echo "== unit tests ==" && python trace/test_quest.py || exit 1
echo "== integration tests ==" && python trace/test_collect.py || exit 1
echo "== trace: LongBench v2 item $INDEX =="
python trace/collect.py --index "$INDEX" "$@" 2>&1 | tee "data/traces/run_lb${INDEX}.log"
echo
echo "Done. Paste the summary printed above (also in trace/results/*.md)."
