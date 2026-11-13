"""Pick Workload A items from LongBench v2: per domain, the first items whose context is 24K-64K tokens.

Such contexts fit the 32,000-token prompt with little or no middle truncation.
Writes the chosen indices to trace/results/workload_a_indices.txt and prints a table.

Usage:  python trace/pick_items.py [--per-domain 1] [--exclude 0]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pick(items: list[dict], count_tokens, per_domain: int, lo: int, hi: int, exclude: set[int]) -> dict:
    """First `per_domain` items per domain, in index order, with lo <= context tokens <= hi."""
    picks: dict[str, list[tuple[int, int]]] = {}
    for i, it in enumerate(items):
        dom = it.get("domain", "?")
        chosen = picks.setdefault(dom, [])
        if i in exclude or len(chosen) >= per_domain:
            continue
        chars = len(it["context"])
        if chars < lo or chars > 12 * hi:  # a token is at least one character; skip the hopeless cases cheaply
            continue
        n = count_tokens(it["context"])
        if lo <= n <= hi:
            chosen.append((i, n))
    return picks


def main(argv=None) -> list[int]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="tokenizer used to count context tokens")
    ap.add_argument("--per-domain", type=int, default=1)
    ap.add_argument("--min-tokens", type=int, default=24000)
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--exclude", type=int, nargs="*", default=[0], help="indices already traced")
    ap.add_argument("--out", default="trace/results/workload_a_indices.txt")
    args = ap.parse_args(argv)

    from collect import load_longbench
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    items = load_longbench()

    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=False).input_ids)

    picks = pick(items, count, args.per_domain, args.min_tokens, args.max_tokens, set(args.exclude))
    rows, indices = [], []
    for dom in sorted(picks):
        for i, n in picks[dom]:
            it = items[i]
            rows.append(f"| {i} | {dom} | {it.get('sub_domain')} | {it.get('difficulty')} | {it.get('length')} | {n:,} |")
            indices.append(i)
        if not picks[dom]:
            rows.append(f"| – | {dom} | no item with {args.min_tokens:,}–{args.max_tokens:,} context tokens | | | |")
    print(f"# Workload A items ({args.min_tokens:,}–{args.max_tokens:,} context tokens, {args.per_domain} per domain)\n")
    print("| Index | Domain | Sub-domain | Difficulty | Length class | Context tokens |")
    print("| --- | --- | --- | --- | --- | --- |")
    print("\n".join(rows))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(" ".join(map(str, indices)) + "\n", encoding="utf-8")
    print(f"\nIndices written to {args.out}: {' '.join(map(str, indices))}")
    return indices


if __name__ == "__main__":
    main()
