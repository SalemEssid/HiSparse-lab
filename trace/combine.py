"""Combine per-trace summaries (trace/results/<prefix>*.json) into one table.

Adds an early H2 signal: how similar the per-layer locality profiles of different prompts are
(Pearson correlation of per-layer step-to-step overlap, for every pair of traces).

Usage:  python trace/combine.py [--prefix A_] [--dir trace/results]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def load(dir_: str, prefix: str) -> list[dict]:
    out = []
    for p in sorted(Path(dir_).glob(f"{prefix}*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if "summary" in d and "meta" in d:
            out.append(d)
    return out


def table(traces: list[dict]) -> str:
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    k = traces[0]["summary"]["k_main"] if traces else 128
    lines = [f"| Trace | Domain | Prompt | Steps | Quest k={k} | Exact k={k} | Quest / exact | Weakest layer "
             "| Overlap mean | Overlap min (layer) | Overlap max (layer) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for d in traces:
        m, s = d["meta"], d["summary"]
        cov = {r["k_pages"]: r for r in s["coverage_by_k"]}.get(s["k_main"], {})
        pl = np.array(s["per_layer_quest_at_k_main"] or [np.nan])
        ov = np.array(s["overlap"]["per_layer"]) if s["overlap"] else np.array([np.nan])
        dom = (m.get("prompt_info") or {}).get("domain", "–")
        lines.append(
            f"| {m['name']} | {dom} | {m['prompt_tokens']:,} | {m['steps']} | {pct(cov.get('quest_logit', np.nan))} "
            f"| {pct(cov.get('token_oracle', np.nan))} | {cov.get('quest_over_exact', np.nan):.3f} "
            f"| {int(np.nanargmin(pl))} ({pct(np.nanmin(pl))}) | {pct(s['overlap']['mean']) if s['overlap'] else '–'} "
            f"| {pct(ov.min())} ({int(ov.argmin())}) | {pct(ov.max())} ({int(ov.argmax())}) |")
    return "\n".join(lines)


def profile_similarity(traces: list[dict]) -> str:
    prof = [np.array(d["summary"]["overlap"]["per_layer"]) for d in traces if d["summary"]["overlap"]]
    if len(prof) < 2:
        return "Fewer than two traces: no cross-prompt comparison yet."
    rs = [float(np.corrcoef(a, b)[0, 1]) for a, b in itertools.combinations(prof, 2)]
    mean = np.mean(prof, axis=0)
    order = np.argsort(mean)
    return (f"Per-layer overlap profiles across {len(prof)} prompts: pairwise Pearson r mean {np.mean(rs):.2f} "
            f"(min {min(rs):.2f}, max {max(rs):.2f}). Lowest-locality layers on average: "
            + ", ".join(f"{int(l)} ({100 * mean[l]:.1f}%)" for l in order[:3])
            + "; highest: " + ", ".join(f"{int(l)} ({100 * mean[l]:.1f}%)" for l in order[::-1][:3]) + ".")


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="trace/results")
    ap.add_argument("--prefix", default="A_")
    args = ap.parse_args(argv)
    traces = load(args.dir, args.prefix)
    md = (f"# Traces `{args.prefix}*`: {len(traces)} found\n\n" + table(traces) + "\n\n## Early H2 signal\n\n"
          + profile_similarity(traces) + "\n")
    print(md)
    return md


if __name__ == "__main__":
    main()
