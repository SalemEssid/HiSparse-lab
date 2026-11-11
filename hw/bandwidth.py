"""Host <-> device bandwidth for the G1 GB/s table.

Two parts:

1. Contiguous copies, 4 KiB .. 256 MiB: H2D and D2H, pinned and pageable host memory.
   Timed with CUDA events (device time) and wall clock (includes launch + sync).

2. Scattered gathers that model miss fetches: P random units from a large pinned pool
   copied into GPU slots. Unit sizes default to the Qwen3-4B/8B per-layer sizes:
     token = 1 token  x 8 KV heads x 128 dims x (K+V) x bf16 =  4 KiB
     page  = 16 tokens                                        = 64 KiB
   Methods:
     loop  : one non_blocking copy per unit (what a naive PyTorch resolve does)
     stage : CPU index_select into a pinned staging buffer, then one H2D copy
   Wall-clock time includes everything, because a synchronous per-layer fetch
   stalls decode for exactly that long.

Assumption: one unit is contiguous in host memory (K and V of a page stored together).

Usage:
    python hw/bandwidth.py --out hw/results/bandwidth.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

KiB, MiB, GiB = 2**10, 2**20, 2**30


def fmt_bytes(n: int) -> str:
    if n >= MiB:
        return f"{n // MiB} MiB"
    return f"{n // KiB} KiB"


def timed(fn, iters: int, warmup: int = 3) -> tuple[float, float]:
    """Median device time (CUDA events) and median wall time, in seconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    dev_t, wall_t = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall_t.append(time.perf_counter() - t0)
        dev_t.append(start.elapsed_time(end) / 1e3)
    return statistics.median(dev_t), statistics.median(wall_t)


def contiguous(sizes: list[int]) -> list[dict]:
    rows = []
    for size in sizes:
        iters = max(10, min(200, int(4 * GiB / size) // 8))
        gpu = torch.empty(size, dtype=torch.uint8, device="cuda")
        pinned = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        pageable = torch.empty(size, dtype=torch.uint8)
        pinned.fill_(1)
        pageable.fill_(1)
        cases = {
            ("H2D", "pinned"): lambda: gpu.copy_(pinned, non_blocking=True),
            ("H2D", "pageable"): lambda: gpu.copy_(pageable),
            ("D2H", "pinned"): lambda: pinned.copy_(gpu, non_blocking=True),
            ("D2H", "pageable"): lambda: pageable.copy_(gpu),
        }
        for (direction, mem), fn in cases.items():
            dev_s, wall_s = timed(fn, iters)
            rows.append(
                {
                    "direction": direction,
                    "host_mem": mem,
                    "bytes": size,
                    "device_us": round(dev_s * 1e6, 1),
                    "wall_us": round(wall_s * 1e6, 1),
                    "device_GBps": round(size / dev_s / 1e9, 2),
                    "wall_GBps": round(size / wall_s / 1e9, 2),
                }
            )
        del gpu, pinned, pageable
    return rows


def scattered(unit_sizes: list[int], counts: list[int], pool_gb: float, iters: int, seed: int) -> list[dict]:
    rows = []
    gen = torch.Generator().manual_seed(seed)
    for unit in unit_sizes:
        n_units = int(pool_gb * GiB) // unit
        pool = torch.empty((n_units, unit), dtype=torch.uint8, pin_memory=True)
        pool.fill_(1)  # touch every page so first-touch cost is not measured
        max_p = max(counts)
        dst = torch.empty((max_p, unit), dtype=torch.uint8, device="cuda")
        staging = torch.empty((max_p, unit), dtype=torch.uint8, pin_memory=True)

        for p in counts:
            # Fresh random indices per iteration so the CPU cache does not help.
            idx_sets = [torch.randperm(n_units, generator=gen)[:p] for _ in range(iters + 3)]
            idx_lists = [ix.tolist() for ix in idx_sets]
            it = {"loop": 0, "stage": 0}

            def loop_fn():
                ids = idx_lists[it["loop"] % len(idx_lists)]
                it["loop"] += 1
                for j, u in enumerate(ids):
                    dst[j].copy_(pool[u], non_blocking=True)

            def stage_fn():
                ix = idx_sets[it["stage"] % len(idx_sets)]
                it["stage"] += 1
                torch.index_select(pool, 0, ix, out=staging[:p])
                dst[:p].copy_(staging[:p], non_blocking=True)

            for name, fn in (("loop", loop_fn), ("stage", stage_fn)):
                _, wall_s = timed(fn, iters)
                rows.append(
                    {
                        "method": name,
                        "unit_bytes": unit,
                        "units": p,
                        "total_bytes": unit * p,
                        "wall_us": round(wall_s * 1e6, 1),
                        "us_per_unit": round(wall_s * 1e6 / p, 2),
                        "wall_GBps": round(unit * p / wall_s / 1e9, 2),
                    }
                )
        del pool, dst, staging
    return rows


def to_markdown(cont: list[dict], scat: list[dict]) -> str:
    out = ["## Contiguous copies", "", "Device GB/s from CUDA events; wall GB/s includes launch and sync.", ""]
    out += ["| Size | H2D pinned | H2D pageable | D2H pinned | D2H pageable |", "| --- | --- | --- | --- | --- |"]
    by_size: dict[int, dict] = {}
    for r in cont:
        by_size.setdefault(r["bytes"], {})[(r["direction"], r["host_mem"])] = r
    for size, d in by_size.items():
        cells = []
        for key in (("H2D", "pinned"), ("H2D", "pageable"), ("D2H", "pinned"), ("D2H", "pageable")):
            r = d[key]
            cells.append(f"{r['device_GBps']} ({r['wall_GBps']} wall)")
        out.append(f"| {fmt_bytes(size)} | " + " | ".join(cells) + " |")

    out += ["", "## Scattered gathers (random units from a pinned pool, wall clock)", ""]
    out += ["| Unit | Units | Method | Total | Latency (us) | us/unit | GB/s |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in scat:
        out.append(
            f"| {fmt_bytes(r['unit_bytes'])} | {r['units']} | {r['method']} | {fmt_bytes(r['total_bytes'])} "
            f"| {r['wall_us']} | {r['us_per_unit']} | {r['wall_GBps']} |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hw/results/bandwidth.json")
    ap.add_argument("--pool-gb", type=float, default=4.0, help="pinned pool for scattered gathers (> CPU LLC)")
    ap.add_argument("--unit-bytes", type=int, nargs="+", default=[4 * KiB, 64 * KiB])
    ap.add_argument("--counts", type=int, nargs="+", default=[1, 4, 16, 64, 128, 256, 512])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available to torch"
    sizes = [4 * KiB, 64 * KiB, 256 * KiB, 1 * MiB, 4 * MiB, 16 * MiB, 64 * MiB, 256 * MiB]

    t0 = time.time()
    cont = contiguous(sizes)
    scat = scattered(args.unit_bytes, args.counts, args.pool_gb, args.iters, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "pool_gb": args.pool_gb,
        "iters": args.iters,
        "seed": args.seed,
        "runtime_s": round(time.time() - t0, 1),
    }
    out.write_text(json.dumps({"meta": meta, "contiguous": cont, "scattered": scat}, indent=2))
    md = f"# Bandwidth: {meta['gpu']}\n\n" + to_markdown(cont, scat)
    out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
