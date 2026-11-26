"""A1: does token-level selection explain why our traces miss about 10x less than HiSparse's?

Input: T<k>_*.npz traces from `trace/collect.py --token-k 2048`. Each holds three selections recorded on the same
generated text, all with the same budget of 2,048 tokens per layer and step:
  page       Quest, 128 pages of 16 tokens (as in Phases 2-4)
  tok_prob   2,048 single tokens with the highest attention weight averaged over heads (an ideal indexer)
  tok_logit  2,048 single tokens with the highest scaled logit averaged over heads (closer to DSA-style scores)
Each cache holds B = 2k units (8,192 KiB per layer either way) and is replayed like in Phase 3.

Per kind: policy miss rates (HiSparse's table), LRU curves and per-layer profile, cold-miss share, bytes and copies
over PCIe per decode step, and the per-layer split (oracle and held-out UCP over the other prompts).
The one-pass LRU curves are re-checked against direct simulation at B = 2k for every layer.

Usage:  python sim/run_a1.py [--traces data/traces] [--prefix T2048_]
Writes sim/results/A1/SUMMARY_A1.md and A1.json, curves to data/curves/A1/.
"""
from __future__ import annotations

import argparse
import itertools
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
BYTES_PER_TOKEN_LAYER = 4096  # Qwen3-4B: 8 KV heads x 128 dims x (K+V) x 2 bytes
KINDS = {  # name: (npz key, k units, tokens per unit)
    "page": ("ranked", 128, PAGE),
    "tok_prob": ("tok_prob", 2048, 1),
    "tok_logit": ("tok_logit", 2048, 1),
}
LABELS = {"page": "Quest pages (128 × 16 tokens)", "tok_prob": "Tokens by attention weight (2,048)",
          "tok_logit": "Tokens by logit (2,048)"}
COLUMNS = ("topk", "fifo", "random", "lru", "hisparse_lru", "belady", "lru_4k")
HISPARSE_PAPER = {"topk": 0.30, "fifo": 0.172, "random": 0.161, "lru": 0.134, "belady": 0.082, "lru_4k": 0.067}


def load_selection(path: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Selections [T, L, k] (int64, -1 = none) and the newest unit per step, for one kind."""
    key, k, unit = KINDS[kind]
    with np.load(path) as d:
        arr = d[key][:, :, :k].astype(np.int64)
        seq_len = d["seq_len"].astype(np.int64)
    if kind != "page":
        arr[arr == 65535] = -1
    newest = (seq_len + unit - 1) // unit - 1
    return arr, newest


def analyze(job: tuple[str, str, str]) -> dict:
    path, kind, curve_dir = job
    sel_all, newest = load_selection(path, kind)
    _, k, unit = KINDS[kind]
    T, L, _ = sel_all.shape
    extra = L * k
    sizes = np.arange(k, k + extra + 1)
    curves = np.zeros((L, extra + 1), np.int64)
    tot = {c: 0 for c in COLUMNS}
    per_step = np.zeros(T, np.int64)
    layer_rate, layer_cold, checks = [], [], 0
    n_sel = 0
    for l in range(L):
        sel = sel_all[:, l]
        n_l = int((sel >= 0).sum())
        n_sel += n_l
        dist = stack_distances(sel, newest)
        curves[l] = miss_curve(dist, sizes)
        tot["topk"] += int(curves[l, 0])
        tot["lru_4k"] += int(curves[l, 3 * k])  # index i is B = k + i
        for pol in POLICIES:
            m, c = simulate(sel, newest, 2 * k, pol, seed=1000 * l + 7)
            tot[pol] += int(m.sum())
            if pol == "lru":
                stack_m = misses_per_step(dist, 2 * k)
                checks += int(np.array_equal(m, stack_m))
                per_step += stack_m
                layer_rate.append(int(m.sum()) / n_l)
                layer_cold.append(int(c.sum()) / max(int(m.sum()), 1))
    name = Path(path).stem
    Path(curve_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(curve_dir) / f"{name}_{kind}.npz", curves=curves, selections=n_sel, k=k, steps=T)
    ps = per_step[1:] if T > 1 else per_step  # without the one-time fill at step 0
    bytes_per_unit = unit * BYTES_PER_TOKEN_LAYER
    return {"trace": name, "kind": kind, "k": k, "unit_tokens": unit, "steps": T, "layers": L,
            "miss_rate": {c: tot[c] / n_sel for c in COLUMNS},
            "per_layer_lru_2k": layer_rate, "per_layer_cold_share": layer_cold,
            "stack_check_layers": checks,
            "copies_per_step_mean": float(ps.mean()), "copies_per_step_p99": float(np.percentile(ps, 99)),
            "mib_per_step_mean": float(ps.mean()) * bytes_per_unit / 2**20,
            "mib_per_step_p99": float(np.percentile(ps, 99)) * bytes_per_unit / 2**20,
            "curve_file": str(Path(curve_dir) / f"{name}_{kind}.npz")}


def splits(results: list[dict]) -> dict:
    """Per kind and trace: oracle and held-out UCP reductions against the uniform split."""
    out = {}
    for kind in KINDS:
        rs = [r for r in results if r["kind"] == kind]
        if not rs:
            continue
        cur = {}
        for r in rs:
            with np.load(r["curve_file"]) as d:
                cur[r["trace"]] = (d["curves"], int(d["selections"]))
        k = rs[0]["k"]
        L = rs[0]["layers"]
        extra, unit = L * k, max(1, k // 16)
        rows = []
        for r in rs:
            mine, _ = cur[r["trace"]]
            others = [n for n in cur if n != r["trace"]]
            uni = total_misses(mine, uniform_extra(L, extra))
            ora = total_misses(mine, ucp_lookahead(mine, extra, unit))
            row = {"trace": r["trace"], "uniform": uni, "oracle_reduction": 1 - ora / max(uni, 1)}
            if others:
                prof = np.mean([cur[n][0] / cur[n][1] for n in others], axis=0)
                a = ucp_lookahead(prof, extra, unit)
                row["heldout_reduction"] = 1 - total_misses(mine, a) / max(uni, 1)
                row["heldout_B"] = (k + a).tolist()
            rows.append(row)
        out[kind] = rows
    return out


def coverage_of(paths: list[str]) -> dict:
    vals = {"page": [], "tok_prob": [], "tok_logit": []}
    for p in paths:
        with np.load(p) as d:
            meta = json.loads(str(d["meta"]))
            if "cov_quest_logit" in d:
                j = meta["ks"].index(128) if 128 in meta["ks"] else -1
                vals["page"].append(float(d["cov_quest_logit"][:, j].mean()))
            for kind in ("tok_prob", "tok_logit"):
                if f"cov_{kind}" in d:
                    vals[kind].append(float(d[f"cov_{kind}"].mean()))
    return {k: float(np.mean(v)) for k, v in vals.items() if v}


def summary(results: list[dict], sp: dict, cov: dict) -> str:
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    kinds = [k for k in KINDS if any(r["kind"] == k for r in results)]
    traces = sorted({r["trace"] for r in results})
    mean = {kind: {c: float(np.mean([r["miss_rate"][c] for r in results if r["kind"] == kind])) for c in COLUMNS}
            for kind in kinds}
    lines = [f"# A1: page-level against token-level selection ({len(traces)} prompts)", "",
             "Same generated text and the same budget for every kind: 2,048 tokens selected per layer and step, "
             "a cache of 2 × that (8 MiB of KV per layer).", "",
             "## Miss rate by policy, B = 2k unless noted (mean of prompts)", "",
             "| Selection | Top-k only (B=k) | FIFO | Random | LRU | HiSparse-LRU | Bélády | LRU, B=4k |",
             "| --- |" + " --- |" * len(COLUMNS)]
    for kind in kinds:
        lines.append(f"| {LABELS[kind]} | " + " | ".join(pct(mean[kind][c]) for c in COLUMNS) + " |")
    lines.append("| HiSparse paper (GLM-5.1 DSA, tokens) | "
                 + " | ".join(pct(HISPARSE_PAPER[c]) if c in HISPARSE_PAPER else "–" for c in COLUMNS) + " |")
    lines += ["", "## Per prompt, LRU at B = 2k", "", "| Trace | " + " | ".join(LABELS[k] for k in kinds) + " |",
              "| --- |" + " --- |" * len(kinds)]
    for t in traces:
        row = {r["kind"]: r for r in results if r["trace"] == t}
        lines.append(f"| {t} | " + " | ".join(pct(row[k]["miss_rate"]["lru"]) if k in row else "–" for k in kinds) + " |")
    lines += ["", f"## Traffic per decode step, all {results[0]['layers']} layers, LRU at B = 2k (after the first step)", "",
              "| Selection | Copies per step (mean / p99) | MiB per step (mean / p99) | Coverage of the true attention |",
              "| --- | --- | --- | --- |"]
    for kind in kinds:
        rs = [r for r in results if r["kind"] == kind]
        lines.append(f"| {LABELS[kind]} | {np.mean([r['copies_per_step_mean'] for r in rs]):,.0f} / "
                     f"{np.mean([r['copies_per_step_p99'] for r in rs]):,.0f} | "
                     f"{np.mean([r['mib_per_step_mean'] for r in rs]):.2f} / {np.mean([r['mib_per_step_p99'] for r in rs]):.2f} | "
                     f"{pct(cov[kind]) if kind in cov else '–'} |")
    lines += ["", "## Per-layer profile, LRU at B = 2k", ""]
    prof = {kind: np.mean([r["per_layer_lru_2k"] for r in results if r["kind"] == kind], axis=0) for kind in kinds}
    cold = {kind: np.mean([r["per_layer_cold_share"] for r in results if r["kind"] == kind], axis=0) for kind in kinds}
    for kind in kinds:
        worst = np.argsort(prof[kind])[::-1][:5]
        lines.append(f"- {LABELS[kind]}: layers from {pct(prof[kind].min())} to {pct(prof[kind].max())}; worst "
                     + ", ".join(f"{int(l)} ({pct(prof[kind][l])})" for l in worst)
                     + f"; median cold share {pct(float(np.median(cold[kind])))}.")
    for a, b in itertools.combinations(kinds, 2):
        lines.append(f"- Per-layer profiles {a} vs {b}: Pearson r = {np.corrcoef(prof[a], prof[b])[0, 1]:.2f}.")
    lines += ["", "## Per-layer split (same total memory as 2k per layer)", "",
              "| Selection | Best possible (own curves) | Held-out (other prompts) |", "| --- | --- | --- |"]
    for kind in kinds:
        rows = sp.get(kind, [])
        ora = np.mean([r["oracle_reduction"] for r in rows])
        held = [r["heldout_reduction"] for r in rows if "heldout_reduction" in r]
        lines.append(f"| {LABELS[kind]} | {pct(ora)} fewer misses | "
                     + (f"{pct(np.mean(held))} fewer (worst {pct(min(held))})" if held else "–") + " |")
    ok = all(r["stack_check_layers"] == r["layers"] for r in results)
    lines += ["", f"Check: one-pass LRU curves equal direct simulation at B = 2k in every layer: {'yes' if ok else 'NO'}.",
              "", "## Headline", ""]
    if "page" in mean and "tok_prob" in mean:
        ratio = mean["tok_prob"]["lru"] / max(mean["page"]["lru"], 1e-12)
        mib = {kind: np.mean([r["mib_per_step_mean"] for r in results if r["kind"] == kind]) for kind in kinds}
        lines.append(f"Token-level selection by attention weight misses {ratio:.1f}× as often as Quest pages at the same "
                     f"budget (LRU at 2k: {pct(mean['tok_prob']['lru'])} vs {pct(mean['page']['lru'])}), and moves "
                     f"{mib['tok_prob'] / max(mib['page'], 1e-12):.1f}× the bytes per step.")
    return "\n".join(lines) + "\n"


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/traces")
    ap.add_argument("--prefix", default="T2048_")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--workers", type=int, default=min(9, os.cpu_count() or 1))
    ap.add_argument("--curves", default="data/curves/A1")
    ap.add_argument("--results", default="sim/results/A1")
    args = ap.parse_args(argv)
    paths = sorted(str(p) for p in Path(args.traces).glob(f"{args.prefix}*.npz"))
    if not paths:
        sys.exit(f"no traces matching {args.traces}/{args.prefix}*.npz")
    jobs = [(p, kind, args.curves) for p in paths for kind in args.kinds]
    print(f"{len(paths)} traces x {len(args.kinds)} kinds = {len(jobs)} jobs on {args.workers} workers", flush=True)
    t0 = time.time()
    if args.workers > 1:
        with Pool(min(args.workers, len(jobs))) as pool:
            results = pool.map(analyze, jobs)
    else:
        results = [analyze(j) for j in jobs]
    sp = splits(results)
    md = summary(results, sp, coverage_of(paths))
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUMMARY_A1.md").write_text(md, encoding="utf-8")
    (out / "A1.json").write_text(json.dumps({"results": results, "splits": sp}, indent=2), encoding="utf-8")
    print(md)
    print(f"[{time.time() - t0:.0f} s]")
    return md


if __name__ == "__main__":
    main()
