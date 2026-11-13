"""Phase 4 (G4): per-layer GPU cache allocation against a uniform split, on prompts the allocation never saw.

Inputs: curves from sim/run_g3.py (data/curves/<trace>_k<k>.npz, curves[l, i] = LRU misses of layer l with
k + i pages) and the traces themselves (data/traces/<trace>.npz) for per-step and HiSparse-LRU checks.

Budget: sum of B_l = L * 2k with every B_l >= k (project notes section 4). Schemes:
  uniform    B_l = 2k                                  (HiSparse default)
  pyramid    linear ramp, more pages to early layers   (PyramidKV-style fixed shape)
  inverted   linear ramp, more pages to late layers
  ucp_loo    UCP on the averaged miss-rate curves of the profile traces, leaving out every trace of the
             evaluated prompt: the real test of H2 / H3
  ucp_all    UCP on all profile traces (what a deployment would ship; in-sample when profile = eval)
  oracle     UCP on the evaluated trace's own curves (upper bound)

Metrics per evaluated trace: total misses (pages moved over PCIe), steady-state misses (without the one-time
cache fill at step 0), mean / p99 / max misses per decode step summed over layers, and total misses
re-simulated with HiSparse-LRU at the chosen sizes.

--profile and --eval may differ, e.g. allocations learned on dense traces (A_) tested on sparse-mode traces.

Usage:  python sim/run_g4.py                                   # profile = eval = A_
        python sim/run_g4.py --profile A_ --eval S128_         # cross test
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alloc import total_misses, ucp_lookahead, uniform_extra  # noqa: E402
from cache import simulate  # noqa: E402
from stack import misses_per_step, stack_distances  # noqa: E402

PAGE = 16
SCHEMES = ("uniform", "pyramid", "inverted", "ucp_loo", "ucp_all", "oracle")
LABELS = {"uniform": "Uniform (2k each)", "pyramid": "Pyramid (early layers more)", "inverted": "Inverted (late layers more)",
          "ucp_loo": "UCP, held-out prompts", "ucp_all": "UCP, all prompts", "oracle": "Oracle (own curves)"}


def item_of(name: str) -> str:
    m = re.search(r"lb\d+", name)
    return m.group(0) if m else name


def load_curves(curve_dir: str, prefix: str, k: int) -> dict:
    out = {}
    for p in sorted(Path(curve_dir).glob(f"{prefix}*_k{k}.npz")):
        name = p.name[: -len(f"_k{k}.npz")]
        with np.load(p) as d:
            out[name] = {"curves": d["curves"].astype(np.int64), "selections": int(d["selections"]),
                         "steps": int(d["steps"]), "item": item_of(name)}
    return out


def shape_extra(L: int, extra: int, weights: np.ndarray) -> np.ndarray:
    """Split `extra` pages in proportion to `weights` (largest remainder, exact total)."""
    raw = extra * weights / weights.sum()
    out = np.floor(raw).astype(np.int64)
    out[np.argsort(raw - out)[::-1][: extra - out.sum()]] += 1
    return out


def profile(curves_by_name: dict, names: list[str]) -> np.ndarray:
    """Average miss-rate curve over traces, so every prompt weighs the same regardless of its length."""
    return np.mean([curves_by_name[n]["curves"] / curves_by_name[n]["selections"] for n in names], axis=0)


def evaluate(job: tuple) -> dict:
    """All schemes on one evaluated trace."""
    name, k, trace_dir, ev, allocs = job
    curves = ev["curves"]
    L = curves.shape[0]
    with np.load(Path(trace_dir) / f"{name}.npz") as d:
        ranked, seq_len = d["ranked"], d["seq_len"].astype(np.int64)
    newest = (seq_len + PAGE - 1) // PAGE - 1
    T = ranked.shape[0]
    per_step = {s: np.zeros(T, np.int64) for s in allocs}
    his = {s: 0 for s in ("uniform", "ucp_loo", "oracle")}
    fill = 0
    for l in range(L):
        sel = ranked[:, l, :k].astype(np.int64)
        fill += int((sel[0] >= 0).sum()) - 1  # step 0: every selected page but the newest is a cold miss
        dist = stack_distances(sel, newest)
        for s, a in allocs.items():
            per_step[s] += misses_per_step(dist, k + int(a[l]))
        for s in his:
            his[s] += int(simulate(sel, newest, k + int(allocs[s][l]), "hisparse_lru")[0].sum())
    out = {"trace": name, "item": ev["item"], "steps": T, "fill": fill, "schemes": {}}
    for s, a in allocs.items():
        tot = total_misses(curves, a)
        ps = per_step[s][1:] if T > 1 else per_step[s]
        out["schemes"][s] = {"total": tot, "steady": tot - fill, "per_step_mean": float(ps.mean()),
                             "per_step_p99": float(np.percentile(ps, 99)), "per_step_max": int(ps.max()),
                             "B": (k + a).tolist()}
        assert tot == int(per_step[s].sum()), "curve totals and per-step stack misses disagree"
    out["hisparse_lru"] = his
    return out


def run(args, k: int) -> tuple[str, dict]:
    prof = load_curves(args.curves, args.profile, k)
    ev = load_curves(args.curves, args.eval, k)
    if not prof or not ev:
        sys.exit(f"no curves for k = {k} ({args.profile}* or {args.eval}*) in {args.curves}; run sim/run_g3.py first")
    L, width = next(iter(ev.values()))["curves"].shape
    extra = L * k
    unit = max(1, k // 16)
    depth = np.arange(L, dtype=np.float64)
    a_uni = uniform_extra(L, extra)
    a_pyr = shape_extra(L, extra, (L - 1 - depth) + 0.5)
    a_inv = shape_extra(L, extra, depth + 0.5)
    a_all = ucp_lookahead(profile(prof, list(prof)), extra, unit)
    jobs, loo_allocs = [], {}
    for name, e in ev.items():
        others = [n for n, p in prof.items() if p["item"] != e["item"]]
        a_loo = ucp_lookahead(profile(prof, others), extra, unit)
        loo_allocs[name] = a_loo
        allocs = {"uniform": a_uni, "pyramid": a_pyr, "inverted": a_inv, "ucp_loo": a_loo, "ucp_all": a_all,
                  "oracle": ucp_lookahead(e["curves"], extra, unit)}
        jobs.append((name, k, args.traces, e, allocs))
    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(evaluate, jobs)

    red = {s: [1 - r["schemes"][s]["total"] / r["schemes"]["uniform"]["total"] for r in results] for s in SCHEMES}
    red_ss = {s: [1 - r["schemes"][s]["steady"] / r["schemes"]["uniform"]["steady"] for r in results] for s in SCHEMES}
    his_red = {s: [1 - r["hisparse_lru"][s] / r["hisparse_lru"]["uniform"] for r in results] for s in ("ucp_loo", "oracle")}
    vecs = list(loo_allocs.values())
    sims = [float(np.corrcoef(a, b)[0, 1]) for a, b in itertools.combinations(vecs, 2)] if len(vecs) > 1 else [1.0]
    loo_mean = float(np.mean(red["ucp_loo"]))
    verdict = "Go" if loo_mean >= 0.10 else ("Extend" if loo_mean >= 0.03 else "Pivot")

    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    lines = [f"## k = {k} pages ({k * PAGE:,} tokens): allocations from `{args.profile}*`, evaluated on `{args.eval}*`", "",
             "Reduction in total misses (pages moved over PCIe) against the uniform split, same total GPU memory.", "",
             "| Trace | Uniform misses | " + " | ".join(LABELS[s] for s in SCHEMES[1:]) + " |",
             "| --- | --- |" + " --- |" * (len(SCHEMES) - 1)]
    for i, r in enumerate(results):
        lines.append(f"| {r['trace']} | {r['schemes']['uniform']['total']:,} | "
                     + " | ".join(pct(red[s][i]) for s in SCHEMES[1:]) + " |")
    lines.append("| **Mean** | | " + " | ".join(f"**{pct(np.mean(red[s]))}**" for s in SCHEMES[1:]) + " |")
    lines.append("| Mean, steady state (without the step-0 fill) | | "
                 + " | ".join(pct(np.mean(red_ss[s])) for s in SCHEMES[1:]) + " |")
    lines += ["", "Misses per decode step, all layers together (mean over traces of each trace's statistic):", "",
              "| Scheme | Mean per step | p99 per step | Max per step |", "| --- | --- | --- | --- |"]
    for s in SCHEMES:
        lines.append(f"| {LABELS[s]} | {np.mean([r['schemes'][s]['per_step_mean'] for r in results]):.1f} "
                     f"| {np.mean([r['schemes'][s]['per_step_p99'] for r in results]):.1f} "
                     f"| {np.mean([r['schemes'][s]['per_step_max'] for r in results]):.0f} |")
    lines += ["", f"Re-simulated with HiSparse-LRU at the same sizes: UCP held-out {pct(np.mean(his_red['ucp_loo']))} "
              f"fewer misses than uniform, oracle {pct(np.mean(his_red['oracle']))}.",
              f"Held-out UCP keeps {np.mean(red['ucp_loo']) / max(np.mean(red['oracle']), 1e-9):.0%} of the oracle's gain; "
              f"held-out allocations agree across folds (mean Pearson r of B per layer) at {np.mean(sims):.2f}.", "",
              f"UCP allocation from all profile traces, pages per layer (uniform = {2 * k}): "
              + ", ".join(str(int(b)) for b in (k + a_all)), "",
              f"**Pre-registered rule (section 11): held-out UCP cuts misses by {pct(loo_mean)} on average, so: {verdict}.** "
              f"Worst held-out trace: {pct(min(red['ucp_loo']))}; best: {pct(max(red['ucp_loo']))}.", ""]
    alloc_path = Path(args.results) / f"alloc_{args.profile.rstrip('_')}_k{k}.json"
    alloc_path.write_text(json.dumps({"k": k, "profile": list(prof), "B": (k + a_all).tolist()}, indent=2), encoding="utf-8")
    summary = {"k": k, "profile": args.profile, "eval": args.eval, "reduction": red, "reduction_steady": red_ss,
               "hisparse_reduction": his_red, "fold_similarity": sims, "verdict": verdict,
               "ucp_all_B": (k + a_all).tolist(), "results": results}
    return "\n".join(lines), summary


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="A_", help="trace prefix the allocations are learned from")
    ap.add_argument("--eval", default=None, help="trace prefix the allocations are tested on (default: same)")
    ap.add_argument("--ks", type=int, nargs="+", default=[128, 32])
    ap.add_argument("--curves", default="data/curves")
    ap.add_argument("--traces", default="data/traces")
    ap.add_argument("--results", default="sim/results")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    args.eval = args.eval or args.profile
    Path(args.results).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    parts, summaries = [], []
    for k in args.ks:
        md, s = run(args, k)
        parts.append(md)
        summaries.append(s)
    tag = args.eval.rstrip("_") if args.eval == args.profile else f"{args.eval.rstrip('_')}_from_{args.profile.rstrip('_')}"
    md = f"# G4: per-layer allocation vs uniform ({tag})\n\n" + "\n".join(parts)
    Path(args.results, f"SUMMARY_G4_{tag}.md").write_text(md, encoding="utf-8")
    Path(args.results, f"G4_{tag}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(md)
    print(f"[{time.time() - t0:.0f} s]")
    return md


if __name__ == "__main__":
    main()
