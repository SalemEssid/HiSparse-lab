#!/usr/bin/env bash
# Run every G1 measurement and collect the summaries in hw/results/SUMMARY.md.
# Usage (from the repo root on the GPU VM):
#   bash hw/run_g1.sh            # Qwen3-4B only
#   RUN_8B=1 bash hw/run_g1.sh   # also try Qwen3-8B (16 GB download)
set -uo pipefail
cd "$(dirname "$0")/.."
R=hw/results
mkdir -p "$R"

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
