"""Phase 3 (G3): replay every trace through the GPU cache simulator.

For each trace data/traces/<prefix>*.npz and each k in --ks:
  - miss rate of every policy at B = 2k, plus B = k (top-k only) and LRU at B = 4k  (HiSparse's table)
  - check: one-pass LRU stack curves equal direct LRU simulation at B = k, 1.5k, 2k, 3k, every step and layer
  - open question Q1: how far HiSparse-LRU is from the plain-LRU curves at those sizes
  - per layer at B = 2k (LRU): miss rate, cold-miss share, miss-rate drop from 25% more cache
  - preview for Phase 4: oracle per-layer allocation on the trace's own curves (an upper bound)
Writes sim/results/<trace>_k<k>.json, data/curves/<trace>_k<k>.npz and sim/results/SUMMARY_G3.md.

Usage:  python sim/run_g3.py [--prefix A_] [--ks 128 32] [--workers 14]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alloc import total_misses, ucp_lookahead, uniform_extra  # noqa: E402
from cache import POLICIES, simulate  # noqa: E402
from stack import miss_curve, misses_per_step, stack_distances  # noqa: E402

PAGE = 16
COLUMNS = ("topk", "fifo", "random", "lru", "hisparse_lru", "belady", "lru_4k")
LABELS = {"topk": "Top-k only (B=k)", "fifo": "FIFO", "random": "Random", "lru": "LRU", "hisparse_lru": "HiSparse-LRU",
          "belady": "Bélády", "lru_4k": "LRU, B=4k"}
HISPARSE_PAPER = {"topk": 0.30, "fifo": 0.172, "random": 0.161, "lru": 0.134, "belady": 0.082, "lru_4k": 0.067}


def run_one(job: tuple[str, int, str, str]) -> dict:
    path, k, curve_dir, res_dir = job
    with np.load(path) as d:
        meta = json.loads(str(d["meta"]))
        ranked = d["ranked"]
        seq_len = d["seq_len"].astype(np.int64)
    T, L, kmax = ranked.shape
    if k > kmax:
        raise ValueError(f"k = {k} > logged k_max = {kmax}")
    newest = (seq_len + PAGE - 1) // PAGE - 1
    extra = L * k  # pages that can move between layers
    sizes = np.arange(k, k + extra + 1)
    checks = [k, k + k // 2, 2 * k, 3 * k]
    curves = np.zeros((L, extra + 1), np.int64)
    tot = {c: 0 for c in COLUMNS}
    n_sel = 0
    layer = {"miss_rate": [], "cold_share": [], "gain_25pct": []}
    check_ok = {B: 0 for B in checks}
    q1 = {B: {"lru": 0, "hisparse_lru": 0, "max_layer_rel_diff": 0.0} for B in checks}
    t0 = time.time()
    for l in range(L):
        sel = ranked[:, l, :k].astype(np.int64)
        n_l = int((sel >= 0).sum())
        n_sel += n_l
        dist = stack_distances(sel, newest)
        curves[l] = miss_curve(dist, sizes)
        sims = {}
        for pol in POLICIES:
            sims[pol] = simulate(sel, newest, 2 * k, pol, seed=1000 * l + 7)
            tot[pol] += int(sims[pol][0].sum())
        tot["topk"] += int(curves[l, 0])
        tot["lru_4k"] += int(curves[l, 3 * k])  # index i is B = k + i
        for B in checks:
            lru = sims["lru"][0] if B == 2 * k else simulate(sel, newest, B, "lru")[0]
            his = sims["hisparse_lru"][0] if B == 2 * k else simulate(sel, newest, B, "hisparse_lru")[0]
            check_ok[B] += int(np.array_equal(lru, misses_per_step(dist, B)))
            a, b = int(lru.sum()), int(his.sum())
            q1[B]["lru"] += a
            q1[B]["hisparse_lru"] += b
            q1[B]["max_layer_rel_diff"] = max(q1[B]["max_layer_rel_diff"], abs(b - a) / max(a, 1))
        m2k, c2k = int(sims["lru"][0].sum()), int(sims["lru"][1].sum())
        layer["miss_rate"].append(m2k / n_l)
        layer["cold_share"].append(c2k / max(m2k, 1))
        layer["gain_25pct"].append((int(curves[l, k]) - int(curves[l, k + k // 4])) / n_l)

    uni = uniform_extra(L, extra)
    ora = ucp_lookahead(curves, extra, unit=max(1, k // 16))
    u_tot, o_tot = total_misses(curves, uni), total_misses(curves, ora)
    name = Path(path).stem
    out = {
        "trace": name, "k": k, "steps": T, "layers": L, "prompt_tokens": meta.get("prompt_tokens"),
        "domain": (meta.get("prompt_info") or {}).get("domain"),
        "miss_rate": {c: tot[c] / n_sel for c in COLUMNS},
        "stack_check": {str(B): {"layers_equal": check_ok[B], "layers": L} for B in checks},
        "q1_hisparse_vs_lru": {str(B): {"lru_rate": q1[B]["lru"] / n_sel, "hisparse_rate": q1[B]["hisparse_lru"] / n_sel,
                                        "max_layer_rel_diff": q1[B]["max_layer_rel_diff"]} for B in checks},
        "per_layer_2k": layer,
        "oracle_preview": {"uniform_misses": u_tot, "oracle_misses": o_tot, "reduction": 1 - o_tot / max(u_tot, 1),
                           "B_per_layer": (k + ora).tolist()},
        "seconds": round(time.time() - t0, 1),
    }
    Path(curve_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(curve_dir) / f"{name}_k{k}.npz", curves=curves, sizes=sizes, k=k, steps=T,
                        selections=n_sel)
    Path(res_dir, f"{name}_k{k}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def order_ok(r: dict) -> bool:
    """HiSparse's ordering: top-k only > FIFO ~ random > LRU > Belady."""
    m = r["miss_rate"]
    return m["topk"] > max(m["fifo"], m["random"]) and min(m["fifo"], m["random"]) > m["lru"] > m["belady"]


def summary(results: list[dict], ks: list[int]) -> str:
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    lines = [f"# G3: cache simulation of {len({r['trace'] for r in results})} traces", ""]
    for k in ks:
        rs = [r for r in results if r["k"] == k]
        if not rs:
            continue
        lines += [f"## Miss rate by policy, k = {k} pages ({k * PAGE:,} tokens), B = 2k unless noted", "",
                  "| Trace | " + " | ".join(LABELS[c] for c in COLUMNS) + " | HiSparse order |",
                  "| --- |" + " --- |" * (len(COLUMNS) + 1)]
        for r in rs:
            lines.append(f"| {r['trace']} | " + " | ".join(pct(r["miss_rate"][c]) for c in COLUMNS)
                         + f" | {'yes' if order_ok(r) else 'NO'} |")
        mean = {c: float(np.mean([r["miss_rate"][c] for r in rs])) for c in COLUMNS}
        lines.append("| **Mean** | " + " | ".join(f"**{pct(mean[c])}**" for c in COLUMNS) + " | |")
        lines.append("| HiSparse paper (GLM-5.1, tokens, k=2048) | "
                     + " | ".join(pct(HISPARSE_PAPER[c]) if c in HISPARSE_PAPER else "–" for c in COLUMNS) + " | |")
        lines.append("")

    lines += ["## Check: one-pass LRU curves = direct LRU simulation (every step, every layer)", "",
              "| k | B | Layers matching exactly |", "| --- | --- | --- |"]
    all_ok = True
    for k in ks:
        rs = [r for r in results if r["k"] == k]
        for B in (k, k + k // 2, 2 * k, 3 * k):
            eq = sum(r["stack_check"][str(B)]["layers_equal"] for r in rs)
            n = sum(r["stack_check"][str(B)]["layers"] for r in rs)
            all_ok &= eq == n
            lines.append(f"| {k} | {B} | {eq} / {n} |")
    lines += ["", "## Open question Q1: HiSparse-LRU against plain LRU", "",
              "| k | B | LRU miss rate | HiSparse-LRU miss rate | Largest difference in one layer |",
              "| --- | --- | --- | --- | --- |"]
    for k in ks:
        rs = [r for r in results if r["k"] == k]
        for B in (k, k + k // 2, 2 * k, 3 * k):
            q = [r["q1_hisparse_vs_lru"][str(B)] for r in rs]
            lines.append(f"| {k} | {B} | {pct(np.mean([x['lru_rate'] for x in q]))} "
                         f"| {pct(np.mean([x['hisparse_rate'] for x in q]))} "
                         f"| {pct(max(x['max_layer_rel_diff'] for x in q))} |")

    for k in ks:
        rs = [r for r in results if r["k"] == k]
        if not rs:
            continue
        mr = np.mean([r["per_layer_2k"]["miss_rate"] for r in rs], axis=0)
        cs = np.mean([r["per_layer_2k"]["cold_share"] for r in rs], axis=0)
        gn = np.mean([r["per_layer_2k"]["gain_25pct"] for r in rs], axis=0)
        lo, hi = np.argsort(mr)[:3], np.argsort(mr)[::-1][:3]
        lines += ["", f"## Per layer, k = {k}, LRU at B = 2k (mean over traces)", "",
                  f"- Miss rate: min {pct(mr.min())}, median {pct(float(np.median(mr)))}, max {pct(mr.max())}. "
                  f"Highest: " + ", ".join(f"layer {int(l)} ({pct(mr[l])})" for l in hi)
                  + "; lowest: " + ", ".join(f"layer {int(l)} ({pct(mr[l])})" for l in lo) + ".",
                  f"- Cold-miss share (cannot be removed by any cache size): median {pct(float(np.median(cs)))}, "
                  f"range {pct(cs.min())} to {pct(cs.max())}.",
                  f"- Miss-rate drop from 25% more cache (B = 2k to 2.25k): median {pct(float(np.median(gn)))}, "
                  f"range {pct(gn.min())} to {pct(gn.max())} (layers {int(gn.argmin())} and {int(gn.argmax())}).",
                  "", "Layer-by-layer miss rate (%): " + ", ".join(f"{100 * v:.1f}" for v in mr)]

    lines += ["", "## Preview for Phase 4: best possible per-layer split, on each trace's own curves", "",
              "Upper bound only: Phase 4 tests allocations profiled on other prompts.", "",
              "| Trace | k | Uniform misses | Oracle misses | Reduction |", "| --- | --- | --- | --- | --- |"]
    for r in results:
        o = r["oracle_preview"]
        lines.append(f"| {r['trace']} | {r['k']} | {o['uniform_misses']:,} | {o['oracle_misses']:,} | {pct(o['reduction'])} |")

    orders = [order_ok(r) for r in results]
    lines += ["", "## G3 verdict", "",
              f"- Policy order matches HiSparse: {sum(orders)} / {len(orders)} (trace, k) runs.",
              f"- LRU curves equal direct simulation: {'yes, every layer at every checked B' if all_ok else 'NO: see table'}.",
              f"- **G3 {'passed' if all(orders) and all_ok else 'not passed'}.**"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/traces")
    ap.add_argument("--prefix", default="A_")
    ap.add_argument("--ks", type=int, nargs="+", default=[128, 32])
    ap.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 1))
    ap.add_argument("--curves", default="data/curves")
    ap.add_argument("--results", default="sim/results")
    args = ap.parse_args(argv)
    paths = sorted(str(p) for p in Path(args.traces).glob(f"{args.prefix}*.npz"))
    if not paths:
        sys.exit(f"no traces matching {args.traces}/{args.prefix}*.npz")
    Path(args.results).mkdir(parents=True, exist_ok=True)
    jobs = [(p, k, args.curves, args.results) for p in paths for k in args.ks]
    print(f"{len(paths)} traces x {len(args.ks)} k values = {len(jobs)} jobs on {args.workers} workers", flush=True)
    t0 = time.time()
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(run_one, jobs)
    else:
        results = [run_one(j) for j in jobs]
    md = summary(results, args.ks)
    Path(args.results, "SUMMARY_G3.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"[{time.time() - t0:.0f} s]")
    return md


if __name__ == "__main__":
    main()
