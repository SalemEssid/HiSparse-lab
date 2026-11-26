#!/usr/bin/env bash
# A1: page-level against token-level selection (repo root, GPU VM), about 15 minutes, unattended:
#   1. tests                                                       (CPU, ~1 min)
#   2. 3 prompts with token-level logging: structured data (fewest
#      misses), dialogue (most), single-document QA                 (GPU, ~8 min)
#   3. simulation of all three selection kinds                      (CPU, ~3 min)
# Usage:  bash run_a1.sh            INDICES="1 6 56" by default; FORCE=1 skips the idle-GPU check
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p data/traces trace/results sim/results/A1
INDICES="${INDICES:-1 6 56}"

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

echo "== 1. tests =="
python trace/test_quest.py && python trace/test_collect.py && python sim/test_sim.py || exit 1

echo "== 2. token-level traces =="
for i in $INDICES; do
    echo "-- item $i"
    python trace/collect.py --index "$i" --token-k 2048 > "data/traces/run_T2048_lb$i.log" 2>&1 \
        || { echo "item $i failed; see data/traces/run_T2048_lb$i.log"; tail -5 "data/traces/run_T2048_lb$i.log"; }
    grep -E "decode steps|Token-level" "data/traces/run_T2048_lb$i.log"
done

echo "== 3. simulation =="
python sim/run_a1.py --prefix T2048_ || exit 1
echo
echo "Done. Paste sim/results/A1/SUMMARY_A1.md back."
