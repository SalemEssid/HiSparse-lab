#!/usr/bin/env bash
# Phase 4 in one go (repo root, GPU VM), about 60 minutes, unattended:
#   1. tests                                                     (CPU, ~1 min)
#   2. G4 on the dense traces: held-out allocation vs uniform     (CPU, ~4 min)
#   3. the 7 Workload A prompts again with true sparse attention  (GPU, ~12 min)
#   4. G3 + G4 on the sparse-mode traces, and the cross test:
#      allocation learned on dense traces, used on sparse ones    (CPU, ~6 min)
#   5. quality check, including resident-page reading and the
#      Phase 4 allocation                                         (GPU, ~35 min)
# Usage:  bash run_phase4.sh            (FORCE=1 to skip the idle-GPU check)
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p data/traces trace/results sim/results/sparse

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
python sim/test_sim.py && python trace/test_quest.py && python trace/test_collect.py && python trace/test_sparse.py || exit 1

echo "== 2. G4 on dense traces =="
python sim/run_g4.py --profile A_ || exit 1

echo "== 3. sparse-mode traces =="
INDICES="${INDICES:-$(ls data/traces/A_*.npz | sed -E 's/.*_lb0*([0-9]+)_s.*/\1/' | sort -n | tr '\n' ' ')}"
for i in $INDICES; do
    echo "-- item $i"
    python trace/collect.py --index "$i" --sparse-k 128 > "data/traces/run_S128_lb$i.log" 2>&1 \
        || { echo "item $i failed; see data/traces/run_S128_lb$i.log"; tail -5 "data/traces/run_S128_lb$i.log"; }
done
python trace/combine.py --prefix S128_ | tee trace/results/SUMMARY_S128.md

echo "== 4. G3 / G4 on sparse-mode traces =="
python sim/run_g3.py --prefix S128_ --results sim/results/sparse || exit 1
python sim/run_g4.py --profile S128_ --results sim/results/sparse || exit 1
python sim/run_g4.py --profile A_ --eval S128_ --results sim/results/sparse || exit 1

echo "== 5. quality check =="
python trace/sparse_check.py --alloc sim/results/alloc_A_k128.json sim/results/alloc_A_k32.json || exit 1

cat sim/results/SUMMARY_G4_A.md sim/results/sparse/SUMMARY_G3.md sim/results/sparse/SUMMARY_G4_S128.md \
    sim/results/sparse/SUMMARY_G4_S128_from_A.md trace/results/SUMMARY_S128.md trace/results/SPARSE_CHECK.md \
    > PHASE4_SUMMARY.md 2>/dev/null
echo
echo "Done. Paste PHASE4_SUMMARY.md back."
