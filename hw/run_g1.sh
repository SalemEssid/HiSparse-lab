#!/usr/bin/env bash
# Run every G1 measurement and collect the summaries in hw/results/SUMMARY.md.
# Usage (from the repo root on the GPU VM):
#   bash hw/run_g1.sh            # Qwen3-4B only
#   RUN_8B=1 bash hw/run_g1.sh   # also try Qwen3-8B (16 GB download)
set -uo pipefail
cd "$(dirname "$0")/.."
R=hw/results
mkdir -p "$R"

# Preflight 1: torch must see the GPU, otherwise every step below fails.
python - <<'EOF' || { echo "Preflight failed: install a torch build that matches the driver (see requirements.txt)."; exit 1; }
import sys, torch
print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("CUDA is not available to torch")
print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

# Preflight 2: measurements need an idle GPU, and they would disturb other jobs.
others=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
if [ -n "$others" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "Other processes are using the GPU:"
    echo "$others"
    echo "Re-run when the GPU is idle, or set FORCE=1 to run anyway."
    exit 1
fi

echo "== env ==" && python hw/env_report.py --out "$R/env.json" 2>&1 | tee "$R/env.log"
echo "== bandwidth ==" && python hw/bandwidth.py --out "$R/bandwidth.json" 2>&1 | tee "$R/bandwidth.log"
echo "== decode Qwen3-4B ==" && python hw/decode_32k.py --model Qwen/Qwen3-4B \
    --contexts 8192 16384 32000 --out "$R/decode_qwen3-4b.json" 2>&1 | tee "$R/decode_qwen3-4b.log"

if [ "${RUN_8B:-0}" = "1" ]; then
    echo "== decode Qwen3-8B ==" && python hw/decode_32k.py --model Qwen/Qwen3-8B \
        --contexts 8192 16384 32000 --prefill-chunk 4096 \
        --out "$R/decode_qwen3-8b.json" 2>&1 | tee "$R/decode_qwen3-8b.log"
fi

cat "$R"/env.md "$R"/bandwidth.md "$R"/decode_*.md > "$R/SUMMARY.md" 2>/dev/null
echo
echo "Done. Paste hw/results/SUMMARY.md back (JSON files have the full detail)."
