# HiSparse-lab

Experiments on the GPU-side cache of hierarchical KV-cache systems for top-k sparse
attention (reference system: HiSparse, arXiv 2608.07009).

## Layout

```text
hw/        machine and host<->device bandwidth measurements
trace/     Quest selector, decode hook, trace collection
sim/       cache policies, Belady, stack distance, allocation
analysis/  figures and tables
notes/     lab log
```

## Phase 1 measurements (GPU machine)

```bash
pip install -r requirements.txt   # install the matching CUDA build of torch first
bash hw/run_g1.sh                 # writes hw/results/, summary in hw/results/SUMMARY.md
```

## Phase 2 traces (GPU machine)

```bash
bash trace/run_g2.sh              # tests, then LongBench v2 item 0: data/traces/*.npz, trace/results/*.md
INDEX=7 bash trace/run_g2.sh      # another item
python trace/collect.py --smoke   # tiny random model, works on CPU
```

Run the scripts in `trace/` as files, not with `python -m`: the folder name shadows Python's built-in `trace` module.
